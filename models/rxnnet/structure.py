"""
structure: stoichmat, flux mat, pool mul mat, link, Nr, ddynvarids, etc.

Caching N as computing it from scratch takes about ~0.15 seconds, and
the cost is propagated to all computations:
    - P takes _one_ computation of N (verification)
    - idynvarids/ddynvarids take _one_ computation of N (computing P)
    - Nr takes _two_ computations of N (N + idynvarids) 
    - L0 takes _three_ computations of N (P + idynvarids + ddynvarids)
    - L takes _four_ computations of N (P + 2 idynvarids + ddynvarids)
    
FIXME: ***
1. caching other attributes (eg, L0 might be expensive)
2. p-denpendent matrix and check p
"""

import fractions
import re
import subprocess

import numpy as np
import pandas as pd

from util import butil

from util.matrix import Matrix


def reorder_xids(net):
    """This functions returns a new network with dynamic variables reordered 
    so that all dependent dynamic variables come last. It should be 
    called before calling any of the following functions in this module. 
    """
    return net.reorder_species(net.ixids + net.dxids)


def get_stoich_mat(net=None, rxnid2stoich=None, only_dynvar=True, 
                   integerize=False, to_mat=True):
    """Return the stoichiometry matrix (N) of the given network or 
    dict rxnid2stoich. Rows correspond to species, and columns correspond to 
    reactions.
    
    Input:
        rxnid2stoich: eg, {'R1':{'A1:1}, 'R2':{'A1':1}}; 
                      net & rxnid2stoich: one and only one should be given
        only_dynvar: if True, use *dynamic* species as rows (keep out 
                        constant/buffered species);
                     if False, use species as row
        integerize: if True, make all stoichcoefs integers
    """
    if net:
        try:
            ## assume network structure has not changed and 
            # precalculated N is up-to-date
            return net.stoich_mat
        except (AttributeError, ValueError):
            if only_dynvar:
                rowvarids = net.xids
            else:
                rowvarids = net.spids
            N = Matrix(np.zeros((len(rowvarids), len(net.rxnids))),
                       rowvarids, net.rxnids)
            for spid in rowvarids:
                for rxnid in net.rxnids:
                    try:
                        stoichcoef = net.rxns[rxnid].stoichiometry[spid]
                        # sometimes stoichcoefs are strings
                        if isinstance(stoichcoef, str):
                            stoichcoef = net.evaluate_expr(stoichcoef)
                        N.loc[spid, rxnid] = stoichcoef
                    except KeyError:
                        pass  # mat[i,j] remains zero

    if rxnid2stoich:
        rxnids = rxnid2stoich.keys()
        spids = []
        for stoich in rxnid2stoich.values():
            for spid, stoichcoef in stoich.items():
                if int(stoichcoef) != 0 and spid not in spids:
                    spids.append(spid)
        N = Matrix(np.zeros((len(spids), len(rxnids))), spids, rxnids)
        for spid in spids:
            for rxnid in rxnids:
                try:
                    N.loc[spid, rxnid] = rxnid2stoich[rxnid][spid]
                except KeyError:
                    pass  # mat[i,j] remains zero
    
    # make all stoichcoefs integers by first expressing them in fractions
    if integerize: 
        for i in range(N.ncol):
            col = N.iloc[:,i]
            nonzeros = [num for num in butil.flatten(col) if num]
            denoms = [fractions.Fraction(nonzero).limit_denominator().denominator 
                      for nonzero in nonzeros]
            denom = np.prod(list(set(denoms)))
            N.iloc[:,i] = col * denom
    
    if net is not None:
        net.stoich_mat = N
        
    if not to_mat:
        N = N.values
    return N


def get_pool_mul_mat(net, to_mat=True):
    """
    Return a matrix whose row vectors are multiplicities of dynamic variables
    in conservation pools. 
    Mathematically, the matrix has rows spanning the left null space of the
    stoichiometry matrix of the network.
    
    The function is computationally costly, because it calls *sage* to perform 
    matrix computations over the integer ring. 
    (Note that the matrix is converted to floats before being returned.)
    """
    try:
        P = net.pool_mul_mat
        N = net.N
        if (P.nrow == 0 and N.rank == N.nrow) or\
            ((P*N).is_zero() and P.nrow + N.rank == N.nrow):
            return P
        else:
            raise ValueError("net has P but its N has changed.")
    except (AttributeError, ValueError):
        ## The following codes compute the INTEGER basis of left null space
        #  of stoichiometry matrix.

        ## Convert the matrix into a string recognizable by sage.
        N = net.N
        if N.rank == N.nrow:
            P = Matrix(None, columns=net.xids)
        else:
            matstr = re.sub('\s|[a-z]|\(|\)', '', np.matrix(N).__repr__())
    
            ## Write a (sage) python script "tmp_sage.py".
            # for more info of the sage commands: 
            # http://www.sagemath.org/doc/faq/faq-usage.html#how-do-i
            # -import-sage-into-a-python-script
            # http://www.sagemath.org/doc/tutorial/tour_linalg.html
            f = open('.tmp_sage.py', 'w')
            f.write('from sage.all import *\n\n')
            if np.float in N.dtypes.values:
                f.write('A = matrix(RR, %s)\n\n' % matstr)  # real field
            else:
                f.write('A = matrix(ZZ, %s)\n\n' % matstr)  # integer field
            f.write('print A.kernel()')  # this returns the left nullspace vectors
            f.close()
    
            ## Call sage and run .tmp_sage.py.
            out = subprocess.Popen(['sage', '-python', '.tmp_sage.py'],
                                   stdout=subprocess.PIPE)
            
            ## Process the output from sage.
            vecstrs = out.communicate()[0].split('\n')[2:-1]
            vecs = [eval(re.sub('(?<=\d)\s+(?=\d|-)', ',', vec)) for vec in vecstrs]
            #poolids = ['Pool%d'%idx for idx in range(1, len(vecs)+1)]
            poolids = net.xids[-len(vecs):][::-1]
            P = Matrix(vecs, poolids, N.rowvarids)
            # Clean things up: so far P can be, eg, 
            #        X1  X2  X3  X4
            # Pool1   0   0   1   1  # say, adenonine backbone
            # Pool2   2   1   3   2  # say, phospho group
            # We want it be the following, via Gaussian row reduction:
            #        X1  X2  X3  X4
            # Pool1   2   1   1   0
            # Pool2  -2  -1   0   1
            # so that X3 and X4 as the dependent dynvars can be easily 
            # selected and expressed as the linear combinations of 
            # independent dynvars
            P = P.ix[:, ::-1].rref().ix[::-1, ::-1]
        net.pool_mul_mat = P
        
        if not to_mat:
            p = P.values
    
        return P


def get_ss_flux_mat(net, integerize=True, to_mat=True):
    """
    Input:
        net & stoichmat: one and only one of them should be given
        integerize: does not work very well so far: eg, for N = 
        
              RBCO  PGAK  GAPDH  REGN  RK  PGAT  GAPT
        RuBP    -1     0      0     0   1     0     0
        PGA      2    -1      0     0   0    -1     0
        BPGA     0     1     -1     0   0     0     0
        GAP      0     0      1    -5   0     0    -1
        Ru5P     0     0      0     3  -1     0     0, 
        
        or: N = array([[-1.,  0.,  0.,  0.,  1.,  0.,  0.],
                       [ 2., -1.,  0.,  0.,  0., -1.,  0.],
                       [ 0.,  1., -1.,  0.,  0.,  0.,  0.],
                       [ 0.,  0.,  1., -5.,  0.,  0., -1.],
                       [ 0.,  0.,  0.,  3., -1.,  0.,  0.]])
        K should be:
        RBCO    3   0
        PGAK    0   1
        GAPDH   0   1
        REGN    1   0
        RK      3   0
        PGAT    6  -1
        GAPT   -5   1
        
        But this code, using "integerize" gives: 
        RBCO    3   0
        PGAK    0   1
        GAPDH   0   1
        REGN    0   0
        RK      3   0
        PGAT    6  -1
        GAPT   -5   1

        For this reason, "integerize" should not be used for now unless debugged 
        and more testings have been done, and integer N is required for now.
                      
    """
    try:
        K = net.ss_flux_mat
        N = net.N
        if (K.ncol == 0 and N.rank == N.ncol) or\
            ((N*K).is_zero() and K.ncol + N.rank == N.ncol):  
            return K
        else:
            raise ValueError("")
    except (AttributeError, ValueError):
        ## The following codes compute the INTEGER basis of right null space
        ## of stoichiometry matrix.

        ## convert the matrix into a string recognizable by sage
        N = net.N
        N_int = Matrix.astype(N, np.int)  
        
        if np.allclose(N, N_int):
            N = N_int
        else:
            raise ValueError("N is not in integers.")
        
        if N.rank == N.ncol:
            K = Matrix(None, net.rxnids)
        else:
            matstr = re.sub('\s|[a-z]|\(|\)', '', np.matrix(N).__repr__())
    
            ## write a (sage) python script ".tmp_sage.py"
            # for more info of the sage commands: 
            # http://www.sagemath.org/doc/faq/faq-usage.html#how-do-i
            # -import-sage-into-a-python-script
            # http://www.sagemath.org/doc/tutorial/tour_linalg.html
            f = open('.tmp_sage.py', 'w')
            f.write('from sage.all import *\n\n')
            if np.float in N.dtypes.values:
                f.write('A = matrix(RR, %s)\n\n' % matstr)  # real field
            else:
                f.write('A = matrix(ZZ, %s)\n\n' % matstr)  # integer field
            f.write('print kernel(A.transpose())')  # return right nullspace vectors
            f.close()
            
            ## call sage and run .tmp_sage.py
            out = subprocess.Popen(['sage', '-python', '.tmp_sage.py'],
                                   stdout=subprocess.PIPE)
            
            ## process the output from sage
            vecstrs = out.communicate()[0].split('\n')[2:-1]
            #vecs = [eval(re.sub('(?<=\d)\s*(?=\d|-)', ',', vec)) 
            #        for vec in vecstrs]
            vecs = [filter(None, vec.strip('[]').split(' ')) for vec in vecstrs]
            try:
                vecs = [[int(elem) for elem in vec if elem] for vec in vecs]
            except ValueError:
                vecs = [[float(elem) for elem in vec if elem] for vec in vecs]
            fdids = ['J%d'%idx for idx in range(1, len(vecs)+1)]
            K = Matrix(np.transpose(vecs), net.rxnids, fdids)
            
        if integerize:  # buggy, see the docstring FIXME **
            for i in range(K.ncol):
                col = K.iloc[:,i]
                nonzeros = [num for num in butil.flatten(col) if num]
                denoms = [fractions.Fraction(nonzero).limit_denominator(1000).denominator 
                          for nonzero in nonzeros]
                denom = np.prod(list(set(denoms)))
                K.iloc[:,i] = np.asarray(col * denom, dtype='int')
        
        assert (N*K).is_zero(), "The calculated K is not really in"\
        "the nullspace of N!"
                        
        net.ss_flux_mat = K
        
        if not to_mat:
            K = K.values

        return K


def get_dxids(net):
    """A typical look of P:
            X1  X2  X3  X4
    Pool1   -2  -1   0   1
    Pool2    2   1   1   0
    
    It should return ['X3', 'X4'] 
    """
    P = net.P
    # dependent dynamic variables are picked at the end of each pool so that
    # networks that have been reordered or not will give the same variables
    if P is None:
        dxids = []    
    else:
        # pick the last xid in each pool
        dxids = [P.iloc[i][P.iloc[i]!=0].index[-1] for i in range(P.nrow)]
        
        # the following scenario should be impossible as P has been in rref. 
        if len(dxids) != len(set(dxids)):
            # eg, ATP can be part of both the adenylate and phosphate pools;
            # in this case, it is easier to manually pick ddynvarids
            raise StandardError("The same dynamic variable is picked out as\
                                 a dependent dynamic variable from two pools.")
    return dxids
get_ddynvarids = get_dxids  # deprecated


def get_ixids(net):
    """
    """
    return [_ for _ in net.xids if _ not in net.dxids]
get_idynvarids = get_ixids  # deprecated


def get_reduced_stoich_mat(net, to_mat=True):
    """
    """
    Nr = net.N.loc[net.ixids]
    if not to_mat:
        Nr = Nr.values
    return Nr


def get_reduced_link_mat(net, to_mat=True):
    """
    L0: L = [I ]
            [L0]
    """
    if len(net.ixids) == len(net.xids):
        L0 = Matrix(columns=net.ixids)
    else:
        L0 = -net.P.loc[:, net.ixids].ch_rowvarids(net.dxids)
    if not to_mat:
        L0 = L0.values
    return L0


def get_link_mat(net):
    """
    L0: L = [I ]
            [L0]
            
    N = L * Nr
    """
    I = Matrix.eye(net.ixids)
    L0 = get_reduced_link_mat(net)
    L = Matrix(pd.concat((I, L0)))
    return L


def get_modules(net):
    N0 = get_stoich_mat(net, only_dynvar=Fasle)
    N = N0.loc[net.xids]
    J = get_ss_flux_mat(net)
    sets = filter(lambda s: len(s)>1, butil.powerset(range(net.rxnids)))
    m2rxn = OD()
    for m in sets:
        Jm = np.zeros(J.shape)
        Jm[s] = J[s]
        dxdtm = np.dot(N, Jm)
        
        
    


"""
def get_link_mat0(net):

    idynvarids = net.idynvarids
    I = Matrix.eye(idynvarids)
    if len(idynvarids) == len(net.dynvarids):
        L = I
    else:
        L0 = net.L0  # np.flipud(get_reduced_link_mat(net))
        L = I.vstack(L0)
    return L
"""




#get_stoichiometry_matrix = get_stoich_mat
#get_pool_multiplicity_matrix = get_pool_mul_mat
#get_steady_state_flux_matrix = get_ss_flux_mat
#get_dependent_dynamic_variable_ids = get_ddynvarids