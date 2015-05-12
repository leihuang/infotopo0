"""
This is a library of class and functions for computations related to
steady states and Metabolic Control Analysis (MCA).
MCA is essentially first-order sensitivity analysis of a metabolic model at
steady-state. For this reason, a network instance passed to any function in
this library is checked for steady state.
"""

from __future__ import division
import re
import subprocess
import fractions
from collections import OrderedDict as OD
import time

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from SloppyCell.ReactionNetworks import *
from SloppyCell import ExprManip as Ex

from util import butil
reload(butil)
from util.sloppycell import netutil
reload(netutil)

import mca


ABBREVIATIONS =    [('parameter', 'param'),
                    ('concentration', 'concn'),
                    ('matrix', 'mat'),
                    ('stoichiometry', 'stoich'),
                    ('variable', 'var'),
                    ('control', 'ctrl'),
                    ('elasticity', 'elas'),
                    ('dynamic', 'dyn'),
                    # ('independent', 'indep'), nesting
                    # ('dependent', 'dep'),
                    ('response', 'resp'),
                    ('multiplicity', 'mul'),
                    ('epsilon', 'eps'),
                    ('steadystate', 'ss'),
                    ('jacobian', 'jac'),
                    ('coefficient', 'coef')]

def long2short(string):
    for long, short in ABBREVIATIONS:
        string = string.replace(long, short)
    return string

def short2long(string):
    for long, short in ABBREVIATIONS:
        string = string.replace(short, long)
    return string


class MCAMatrix(np.matrix):
    """
    stoichiometry matrix,
    elasticity matrix of concentrations,
    elasticity matrix of parameters,
    concentration control matrix 
    flux control matrix
    concentration response matrix
    flux response matrix

    reduced stoichiometry matrix
    link matrix L
    L0    
    """
    def __new__(cls, mat, rowvarids=None, colvarids=None, dtype='float'):
        obj = np.matrix(mat, dtype=dtype).view(cls)
        obj.rowvarids = rowvarids
        obj.colvarids = colvarids
        return obj

 
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.rowvarids = getattr(obj, 'rowvarids', None)
        self.colvarids = getattr(obj, 'colvarids', None)

    
    def get_row(self, rowvarid):
        i = self.rowvarids.index(rowvarid)
        return self[i]
        
        
    def get_element(self, rowvarid, colvarid):
        i = self.rowvarids.index(rowvarid)
        j = self.colvarids.index(colvarid)
        return self[i, j]
    
    
    def to_df(self):
        """
        Convert to pandas.DataFrame for the ease of visualization.
        """
        return pd.DataFrame(self, index=self.rowvarids, columns=self.colvarids)
    
    
    def to_mat(self):
        return mca.MCAMatrix(self.to_df())
    
    
    def __repr__(self):
        pd = self.to_df()
        return pd.__repr__()
    
    
    def vstack(self, other):
        """
        Stack vertically.
        """
        mat = MCAMatrix(np.vstack((self, other)), 
                        rowvarids=self.rowvarids+other.rowvarids,
                        colvarids=self.colvarids)
        return mat
    
    
    def rank(self, tol=None):
        return np.linalg.matrix_rank(self, tol=tol)
    
    
    def __mul__(self, other):
        """
        A customization of np.matrix.__mul__ so that if two MCAMatrix instances
        are passed in, their meta-info of rowvarids and colvarids are kept.
        """
        if isinstance(other, MCAMatrix):
            rowvarids = self.rowvarids
            colvarids = other.colvarids
            return MCAMatrix(np.matrix.__mul__(self, other), 
                             rowvarids=rowvarids, colvarids=colvarids)
        else:
            return np.matrix.__mul__(self, other)
        

    @property
    def nrow(self):
        return self.shape[0]
        
        
    @property
    def ncol(self):
        return self.shape[1]
    
    
    @property
    def T(self):
        return MCAMatrix(np.transpose(self), 
                         rowvarids=self.colvarids, colvarids=self.rowvarids) 
        
    
    @property
    def I(self):
        return MCAMatrix(np.linalg.inv(self), 
                         rowvarids=self.rowvarids, colvarids=self.colvarids) 
            
    # define matrix concatenation and slicing operations,
    # especially rowvarids and colvarids attributes


def get_reordered_net(net):
    """
    This functions returns a new network instance with dynamic variables 
    reordered so that dependent dynamic variables come last. It should be
    called before calling any of the following functions in this module. 
    """
    N = get_stoich_mat(net)
    if N.rank() == N.shape[0]: 
        # rank of N equals the number of rows
        # net has no conserved pools or dependent dynamic variables
        return net.copy()
    else:
        # rank of N smaller than the number of rows
        # net has conserved pools and dependent dynamic variables
        ddynvarids = get_dep_dyn_var_ids(net)  # d for dependent
        net2 = Network(net.id, name=net.name)
        ## Make new variables: move the dependent dynamic variables to 
        # the end of keyedlist net.variables, and call 
        # method _makeCrossReferences later so that
        # attributes of the network reflect the new order.
        variables2 = net.variables.copy()
        for ddynvarid in ddynvarids:
            idx = variables2.keys().index(ddynvarid)
            var = variables2.pop(idx)
            variables2.set(ddynvarid, var)
        ## attach attributes
        net2.variables = variables2
        net2.reactions = net.reactions.copy()
        net2.assignmentRules = net.assignmentRules.copy()
        net2.algebraicRules = net.algebraicRules.copy()
        net2.rateRules = net.rateRules.copy()
        net2.events = net.events.copy()
        ## final processings
        # method _makeCrossReferences will take care of at least
        # the following attributes:
        # assignedVars, constantVars, optimizableVars, dynamicVars, 
        # algebraicVars
        net2._makeCrossReferences()
        net2.reordered = True
        net2.ddynvarids = ddynvarids
        return net2


def get_matrix_trial(net, attr, paramvals=None,
                     check_steadystate=True, tol=1e-6):
    """
    A slightly more sophisticated function than *getattr*
    
    check_steadystate?
    """
    # if check_steadystate and not is_net_steadystate(net, tol=tol):
    #    raise ValueError("The network has not reached steady state yet.")
    if hasattr(net, attr):
        # parameter values are either not given or
        # the same as those of the network
        if paramvals is None or butil.true_if_close(paramvals, 
                [p.value for p in net.optimizableVars], cutoff=1e-9):
            return getattr(net, attr)
        else:
            raise ValueError("The network has attribute %s, but the provided\
                  parameter values are different from the network's." % attr)
    else:
        raise AttributeError("The network has no attribute %s." % attr)


def get_stoich_mat(net=None, rxnid2stoich=None, dynamic_only=True, 
                   integerize=True):
    """
    Return the stoichiometry matrix (N) of the given network or 
    dict rxnid2stoich. Rows correspond to species, and columns correspond to 
    reactions. 
    
    Input:
        rxnid2stoich: eg, {'R1':{'A1:1}, 'R2':{'A1':1}}; 
                      net & rxnid2stoich: one and only one should be given
        dynamic_only: if True, use *dynamic* species as rows (keep out 
                        constant/buffered species);
                      if False, use species as row
        integerize: if True, make all stoichcoefs integers
    """
    if net:
        attr = 'stoich_mat'
        try:
            return get_matrix_trial(net, attr)
        
        except AttributeError:
            if dynamic_only:
                spids = [spid for spid, sp in net.species.items() 
                         if not sp.is_constant] 
            else:
                spids = net.species.keys()
            rxnids = net.reactions.keys()
            mat = MCAMatrix(np.zeros((len(spids), len(rxnids))),
                            rowvarids=spids, colvarids=rxnids)
            
            for i in range(len(spids)):
                for j in range(len(rxnids)):
                    spid, rxnid = spids[i], rxnids[j]
                    try:
                        stoichcoef = net.reactions.get(rxnid).\
                                     stoichiometry[spid]
                        # sometimes stoichcoefs are strings
                        if isinstance(stoichcoef, str):
                            stoichcoef = net.evaluate_expr(stoichcoef)
                        mat[i,j] = stoichcoef
                    except KeyError:
                        pass  # mat[i,j] remains zero
            
            # make all stoichcoefs integers by first expressing them in fractions
            if integerize: 
                for i in range(mat.ncol):
                    col = mat[:,i]
                    nonzeros = [num for num in col.flatten() if num]
                    denoms = [fractions.Fraction(nonzero).denominator 
                              for nonzero in nonzeros]
                    denom = np.prod(list(set(denoms)))
                    mat[:,i] = col * denom
                        
            setattr(net, attr, mat)
            setattr(net, short2long(attr), mat)
        
            # update the meta-attribute 'attrs_structural'
            if not hasattr(net, 'attrs_structural'):
                net.attrs_structural = set([])
            net.attrs_structural.update([attr, short2long(attr)])
    
    if rxnid2stoich:
        rxnids = rxnid2stoich.keys()
        spids = [stoich.keys() for stoich in rxnid2stoich.values()]
        spids = list(set(butil.flatten(spids)))
        mat = MCAMatrix(np.zeros((len(spids), len(rxnids))),
                        rowvarids=spids, colvarids=rxnids)
        for i in range(len(spids)):
            for j in range(len(rxnids)):
                spid, rxnid = spids[i], rxnids[j]
                try:
                    stoichcoef = rxnid2stoich[rxnid][spid]
                    mat[i,j] = stoichcoef
                except KeyError:
                    pass  # mat[i,j] remains zero
                                
    return mat
get_stoichiometry_matrix = get_stoich_mat


def get_pool_mul_mat(net):
    """
    Return a matrix whose row vectors are multiplicities of dynamic variables
    in conservation pools. 
    Mathematically, the matrix has rows spanning the left null space of the
    stoichiometry matrix of the network.
    
    The function is computationally costly, because it calls *sage* to perform 
    matrix computations over the integer ring. 
    (Note that the matrix is converted to floats before being returned.)
    """
    attr = 'pool_mul_mat'
    
    try:
        return get_matrix_trial(net, attr)
    except AttributeError:
        ## The following codes compute the INTEGER basis of left null space
        #  of stoichiometry matrix.

        ## Convert the matrix into a string recognizable by sage.
        stoichmat = get_stoich_mat(net)
        matstr = re.sub('\s|[a-z]|\(|\)', '', np.matrix.__repr__(stoichmat))

        ## Write a (sage) python script "tmp_sage.py".
        # for more info of the sage commands: 
        # http://www.sagemath.org/doc/faq/faq-usage.html#how-do-i
        # -import-sage-into-a-python-script
        # http://www.sagemath.org/doc/tutorial/tour_linalg.html
        f = open('.tmp_sage.py', 'w')
        f.write('from sage.all import *\n\n')
        f.write('A = matrix(ZZ, %s)\n\n' % matstr)  # integers as the field
        f.write('print A.kernel()')  # this returns the left nullspace vectors
        f.close()

        ## Call sage and run .tmp_sage.py.
        out = subprocess.Popen(['sage', '-python', '.tmp_sage.py'],
                               stdout=subprocess.PIPE)
        
        ## Process the output from sage.
        vecstrs = out.communicate()[0].split('\n')[2:-1]
        vecs = [eval(re.sub('(?<=\d)\s*(?=\d|-)', ',', vec)) 
                for vec in vecstrs]
        
        mat = MCAMatrix(vecs, colvarids=net.dynamicVars.keys())
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        ## Update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_pool_multiplicity_matrix = get_pool_mul_mat


def get_ss_flux_mat(net=None, stoichmat=None):
    """
    Input:
        net & stoichmat: one and only one of them should be given
    """
    attr = 'ss_flux_mat'
    try:
        return get_matrix_trial(net, attr)
    except AttributeError:
        ## The following codes compute the INTEGER basis of right null space
        ## of stoichiometry matrix.

        ## convert the matrix into a string recognizable by sage
        if net:
            stoichmat = get_stoich_mat(net)
        matstr = re.sub('\s|[a-z]|\(|\)', '', stoichmat.__repr__())

        ## write a (sage) python script ".tmp_sage.py"
        # for more info of the sage commands: 
        # http://www.sagemath.org/doc/faq/faq-usage.html#how-do-i
        # -import-sage-into-a-python-script
        # http://www.sagemath.org/doc/tutorial/tour_linalg.html
        
        f = open('.tmp_sage.py', 'w')
        f.write('from sage.all import *\n\n')
        f.write('A = matrix(ZZ, %s)\n\n' % matstr)  # integers as the field
        f.write('print kernel(A.transpose())')  # return right nullspace vectors
        f.close()
        
        ## call sage and run .tmp_sage.py
        out = subprocess.Popen(['sage', '-python', '.tmp_sage.py'],
                               stdout=subprocess.PIPE)
        
        ## process the output from sage
        vecstrs = out.communicate()[0].split('\n')[2:-1]
        #vecs = [eval(re.sub('(?<=\d)\s*(?=\d|-)', ',', vec)) 
        #        for vec in vecstrs]
        vecs = [vec.strip('[]').split(' ') for vec in vecstrs]
        vecs = [[int(elem) for elem in vec if elem] for vec in vecs]
        mat = MCAMatrix(np.transpose(vecs), colvarids=stoichmat.colvarids)
        
        if net:
            setattr(net, attr, mat)
            setattr(net, short2long(attr), mat)
            # update the meta-attribute 'attrs_structural'
            if not hasattr(net, 'attrs_structural'):
                net.attrs_structural = set([])
            net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_steadystate_flux_matrix = get_ss_flux_mat



def get_independent_fluxes(net):
    """
    """
    ssfluxmat = mcautil.get_ss_flux_mat(net)
    
    # add independent fluxes
    # first get the estimated values of independent fluxes of control
    b = get_steadystate_fluxes(net)
    Js_ind = np.linalg.lstsq(ssfluxmat, b)[0]
    ids_J_ind = []  # independent flux ids
    for i in range(ssfluxmat.shape[1]):
        id_J_ind = 'J' + str(i+1)
        ids_J_ind.append(id_J_ind)
        net.add_parameter(id_J_ind, Js_ind[i], is_optimizable=True)
        
    net.add_parameter('RT', RT, is_optimizable=False)
    
    # add dependent fluxes, reactions (can be reversible)
    for rxnid, info in rxnmap.items():
        stoich, J_dep, dG = info
        # muls: multiplicities
        idx = ssfluxmat.colvarids.index(rxnid)
        muls = ['(%s)' % str(int(mul)) for mul in
                list(np.array(ssfluxmat[idx, :]).flatten())]
        
        id_J_dep = 'J' + rxnid
        net.add_parameter(id_J_dep, J_dep,
                          is_constant=False, is_optimizable=False)
        assignmentrule = '+'.join(['*'.join([muls[i], ids_J_ind[i]]) 
                                   for i in range(len(muls))])
        # simplify the assignment rule symbolically
        # e.g., '1*J1 + 0*J2' becomes 'J1'
        assignmentrule = str(sympy.simplify(assignmentrule))
        net.add_assignment_rule(id_J_dep, assignmentrule)
    

def get_pools(net):
    """
    Return the conservation pools of the network. 
    pools: ordereddicts mapping from dynamic variable ids to
           their multiplicities in the pool.
    """
    attr = 'pools'
    
    try:
        return getattr(net, attr)
    except AttributeError:
        poolmat = get_pool_mul_mat(net)
        vecs = poolmat.tolist()
        pools = [OD(zip(net.dynamicVars.keys(), vec)) for vec in vecs]
        setattr(net, attr, pools)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.add(attr)
        
        return pools


def get_dep_dyn_var_ids(net):
    """
    """
    attr = 'ddynvarids'
    
    try:
        return getattr(net, attr)
    except AttributeError:
        mat = get_pool_mul_mat(net)
        # dependent dynamic variables are picked at the end of each pool so that
        # networks that have been reordered or not will give the same variables
        if mat.shape == (1,0):
            ddynvarids = []    
        else:
            ddynvarids = [mat.colvarids[np.where(np.array(mat)[i] == 1)[0][-1]]
                          for i in range(mat.shape[0])]
            if len(ddynvarids) != len(set(ddynvarids)):
                raise StandardError("The same dynamic variable is picked out as\
                                    a dependent dynamic variable from two pools.")
        setattr(net, attr, ddynvarids)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.add(attr)
    
        return ddynvarids
get_dependent_dynamic_variable_ids = get_dep_dyn_var_ids


def get_indep_dyn_var_ids(net):
    """
    """
    attr = 'idynvarids'
    try:
        return getattr(net, attr)
    except AttributeError:
        ddynvarids = get_dep_dyn_var_ids(net)
        idynvarids = [varid for varid in net.dynamicVars.keys() 
                      if varid not in ddynvarids]
        setattr(net, attr, idynvarids)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.add(attr)
        
        return idynvarids
get_independent_dynamic_variables_ids = get_indep_dyn_var_ids


def get_reduced_link_mat(net):
    """
    L0: L = [I ]
            [L0]
    """
    if not hasattr(net, 'reordered'):
        # raise StandardError("Network has not been reordered yet.")
        net = get_reordered_net(net)
    attr = 'reduced_link_mat'
    try:
        return get_matrix_trial(net, attr)
    except AttributeError:
        poolmat = get_pool_mul_mat(net)
        idynvarids = get_indep_dyn_var_ids(net)
        ddynvarids = get_dep_dyn_var_ids(net)
        mat = MCAMatrix(poolmat[:, :len(idynvarids)], colvarids=idynvarids)
        for i in range(len(ddynvarids)):
            # in place modifications of mat: 
            # each row is replaced by (-1 * row / multiplicity(ddynvarid))
            # len(idynvarids) + i: index of the ddynvarid
            mat[i] /= -poolmat[i, len(idynvarids) + i]
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_reduced_link_matrix = get_reduced_link_mat


def get_link_mat(net):
    """
    L: N = L * Nr
    """
    attr = 'link_mat'
    try:
        return get_matrix_trial(net, attr)
    except AttributeError:
        idynvarids = get_indep_dyn_var_ids(net)
        dynvarids = net.dynamicVars.keys()
        I = np.matrix(np.identity(len(idynvarids)))
        if len(dynvarids) == len(idynvarids):
            mat = I
        else:
            L0 = get_reduced_link_mat(net)
            mat = np.concatenate((I, L0))
        mat = MCAMatrix(mat, rowvarids=dynvarids, colvarids=idynvarids)
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_link_matrix = get_link_mat


def get_reduced_stoich_mat(net):
    """
    Nr
    """
    attr = 'reduced_stoich_mat'
    try:
        return get_matrix_trial(net, attr) 
    except AttributeError:
        stoichmat = get_stoich_mat(net)
        idynvarids = get_indep_dyn_var_ids(net)
        mat = MCAMatrix(stoichmat[:len(idynvarids)],
                        rowvarids=idynvarids, colvarids=stoichmat.colvarids)
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_reduced_stoichiometry_matrix = get_reduced_stoich_mat


def get_concn_elas_mat(net, paramvals=None, normed=True):
    """
    Calculate concentration elasticity matrix (eps_s = dv/ds).
    
    Input:
        normed: if True, return normalized matrix used in MCA 
                (eps_s = (dv/v)/(ds/s)).   
    """
    attr = 'concn_elas_mat'
    try:
        return get_matrix_trial(net, attr, paramvals)
    
    except (ValueError, AttributeError):
        update_net(net, paramvals=paramvals, time=np.inf)
        
        rxnids = net.reactions.keys()
        dynvarids = net.dynamicVars.keys()
        mat = MCAMatrix(np.zeros((len(rxnids), len(dynvarids))),
                        rowvarids=rxnids, colvarids=dynvarids)
        
        for i in range(len(rxnids)):
            kineticlaw = net.reactions[i].kineticLaw
            for j in range(len(dynvarids)):
                dynvarid = dynvarids[j]
                mat[i,j] = net.evaluate_expr(Ex.diff_expr(kineticlaw, dynvarid))
        
        if normed:
            D_J = get_ss_fluxes(net, pvals=paramvals, mat=True)
            D_s = get_dynvarssvals(net, paramvals=paramvals, mat=True)
            mat = D_J.I * mat * D_s
            
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_concentration_elasticity_matrix = get_concn_elas_mat


def get_param_elas_mat(net, paramvals=None):
    """
    eps_p
    """
    attr = 'param_elas_mat'
    try:
        return get_matrix_trial(net, attr, paramvals)
    except (ValueError, AttributeError):
        update_net(net, paramvals=paramvals, time=np.inf)
        rxnids = net.reactions.keys()
        paramids = net.optimizableVars.keys()
        mat = MCAMatrix(np.zeros((len(rxnids), len(paramids))),
                        rowvarids=rxnids, colvarids=paramids)
        for i in range(len(rxnids)):
            kineticlaw = net.reactions[i].kineticLaw
            for j in range(len(paramids)):
                paramid = paramids[j]
                mat[i, j] = net.evaluate_expr(Ex.diff_expr(kineticlaw, paramid))
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_parameter_elasticity_matrix = get_param_elas_mat


def get_jac_mat(net, paramvals=None):
    """
    Return the jacobian matrix (M) of the network, which, IN THE MCA CONTEXT, 
    is the jacobian of the *reduced* vector field dx_i/dt = Nr * v(x,p).
    (reduced so that M is invertible)
    """
    attr = 'jac_mat'
    try: 
        return get_matrix_trial(net, 'jac_mat', paramvals)
    except (ValueError, AttributeError):
        update_net(net, paramvals=paramvals, time=np.inf)
        L = get_link_mat(net)
        eps_s = get_concn_elas_mat(net)
        Nr = get_reduced_stoich_mat(net)
        mat = Nr * eps_s * L
        mat = MCAMatrix(mat, rowvarids=net.idynvarids, colvarids=net.idynvarids)
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_jacobian_matrix = get_jac_mat


def get_concn_ctrl_mat(net, paramvals=None, normed=False):
    """
    """
    attr = 'concn_ctrl_mat'
    try:
        return get_matrix_trial(net, attr, paramvals)
    except (ValueError, AttributeError):
        update_net(net, paramvals=paramvals, time=np.inf)
        L = get_link_mat(net)
        M = get_jac_mat(net)
        Nr = get_reduced_stoich_mat(net)
        mat = -L * M.I * Nr
        mat = MCAMatrix(mat, rowvarids=net.dynamicVars.keys(),
                        colvarids=net.reactions.keys())
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_concentration_control_matrix = get_concn_ctrl_mat
    

def get_flux_ctrl_mat(net, paramvals=None, normed=True):
    """
    """
    attr = 'flux_ctrl_mat'
    try:
        return get_matrix_trial(net, attr, paramvals)
    except (ValueError, AttributeError):
        update_net(net, paramvals=paramvals, time=np.inf)
        
        C_s = get_concn_ctrl_mat(net)
        eps_s = get_concn_elas_mat(net)
        I = np.matrix(np.identity(len(net.reactions)))
        mat = eps_s * C_s + I
        if normed:
            D_J = get_ss_fluxes(net, pvals=paramvals, mat=True)
            mat = D_J.I * mat * D_J
        mat = MCAMatrix(mat, rowvarids=net.reactions.keys(),
                        colvarids=net.reactions.keys())
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_flux_control_matrix = get_flux_ctrl_mat


def get_concn_resp_mat(net, paramvals=None, normed_param=False,
                           normed_concn=False):
    """
    """
    attr = 'concn_response_mat'
    if normed_param:
        attr = attr + '_normed_param'
    if normed_concn:
        if normed_param:
            attr = attr + '_concn'
        else:
            attr = attr + '_normed_concn'
    try:
        return get_matrix_trial(net, attr, paramvals=paramvals)
    except (ValueError, AttributeError):
        update_net(net, paramvals=paramvals, time=np.inf)
        C_s = get_concn_ctrl_mat(net)
        eps_p = get_param_elas_mat(net)
        mat = C_s * eps_p
        if normed_param:
            paramvals = [var.value for var in net.optimizableVars]
            # row-wise multiplication
            mat = np.multiply(mat, paramvals)
        if normed_concn:
            dynvarssvals = [var.value for var in net.dynamicVars]
            # row-wise division
            mat = np.divide(mat.transpose(), dynvarssvals).transpose()
        mat = MCAMatrix(mat, rowvarids=net.dynamicVars.keys(),
                        colvarids=net.optimizableVars.keys())
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_concentration_response_matrix = get_concn_resp_mat


def get_flux_resp_mat(net, paramvals=None, normed_param=False,
                          normed_flux=False):
    """
    """
    attr = 'flux_response_mat'
    if normed_param:
        attr = attr + '_normed_param'
    if normed_flux:
        if normed_param:
            attr = attr + '_flux'
        else:
            attr = attr + '_normed_flux'
    try:
        return get_matrix_trial(net, attr, paramvals=paramvals)
    except (ValueError, AttributeError):
        if not hasattr(net, 'fluxVars'):
            net = netutil.add_fluxes(net)
        update_net(net, paramvals=paramvals, time=np.inf)
        C_J = get_flux_ctrl_mat(net)
        eps_p = get_param_elas_mat(net)
        mat = C_J * eps_p
        if normed_param:
            # row-wise multiplication
            mat = np.multiply(mat, paramvals)
        if normed_flux:
            ssfluxes = get_steadystate_fluxes(net)
            # row-wise division
            mat = np.divide(mat.transpose(), ssfluxes).transpose()
        mat = MCAMatrix(mat, rowvarids=net.fluxVars.keys(),
                        colvarids=net.optimizableVars.keys())
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat) 
        
        # update the meta-attribute 'attrs_kinetic'
        if not hasattr(net, 'attrs_kinetic'):
            net.attrs_kinetic = set([])
        net.attrs_kinetic.update([attr, short2long(attr)])
        
        return mat
get_flux_response_matrix = get_flux_resp_mat

"""
        mat_paramvals = np.matrix(np.diag(paramvals))
        mat_flux = np.matrix(np.diag(get_steady_state_fluxes(net)))
        suffix = ''
        if normed_param and not normed_flux:
            mat = mat * mat_paramvals
            suffix = '_normed_param'
        if not normed_param and normed_flux:
            mat = np.linalg.pinv(mat_flux) * mat
            suffix = '_normed_flux'
        if normed_param and normed_flux:
            mat = np.linalg.pinv(mat_flux) * mat * mat_paramvals
            suffix = '_normed_param_flux'
        mat = MCAMatrix(mat, rowvarids=net.fluxVars.keys(),
                        colvarids=net.optimizableVars.keys())
        setattr(net, 'flux_response_matrix'+suffix, mat)
        setattr(net, 'R_J'+suffix, mat)
        return mat
"""
    

def get_dres_dp_function(net, paramvals=None):
    """
    """
    update_net(net, paramvals=paramvals)
    net.compile()
    optvarids = [var.id for var in net.optimizableVars]
    
    # a collection of parameter sensitivity functions
    def dres_dp_function(time, dynvarvals, yprime, consts, net=net):
        dres_dp_funcs = [eval('net.dres_d' + varid) for varid in optvarids]
        dres_dp = [func(time, dynvarvals, yprime, consts) 
                   for func in dres_dp_funcs]
        dres_dp = np.transpose(np.array(dres_dp))
        return dres_dp

    return dres_dp_function


def get_concentration_response_matrix2(net, paramvals=None, recalculate=False,
                                       method='root', T=1000, tol_fsolve=1e-6,
                                       tol_ss=1e-4):
    """
    An old function for computing concentration response matrix, and
    its result agrees with that of the new MCA one very well. 
    The notations here are different and obselete.

    method: the method to get steady states, 'rootfinding' or 'simulation'
    x: m-dim, concentration vector
    p: p-dim, parameter vector
    
    R: R(x,p) = N*v(x,p), where N is stoichiometry matrix and v is rate vector
       R for RHS, also residual as SloppyCell calls it
    C: C(x,p) = Ax - a, where A is conservation matrix and a is pool sizes
       C for conservation
    f: f(x,p) = [R(x,p); C(x,p)]
       f for fsolvefunc
    
    mul: multiplicity
    """
    if paramvals is not None:
        net.update_optimizable_vars(paramvals)
    net.compile()
    if is_net_steadystate(net):
        dynvarssvals = [var.value for var in net.dynamicVars]
    else:
        dynvarssvals = get_dynvarssvals(net, paramvals=paramvals, T=T,
                                        tol_fsolve=tol_fsolve, tol_ss=tol_ss,
                                        method=method)
    pools = get_pools(net)
    ddynvarids = get_dependent_dynamic_variables(net)
    ddynvaridxs = [net.dynamicVars.keys().index(ddynvarid)
                   for ddynvarid in ddynvarids]

    # dRdx, m by m
    dRdx = net.dres_dc_function(0, dynvarssvals, np.zeros(len(dynvarssvals)),
                                net.constantVarValues)
    # replace rows of dres_dc corresponding to dependent variables with 
    # algebraic constraints
    # dRdx_reduced, r by m
    dRdx_reduced = np.delete(dRdx, ddynvaridxs, 0)
    # dCdx, (m-r) by m: elements are just multiplicities of vars in a pool
    dCdx = np.array([[pool[varid] for varid in net.dynamicVars.keys()]
                     for pool in pools])
    # dfdx, m by m
    dfdx = np.concatenate((dRdx_reduced, dCdx), 0)
    # dRdp, m by p
    if hasattr(net, 'dres_dp_function'):
        dres_dp_function = net.dres_dp_function
    else:
        dres_dp_function = get_dres_dp_function(net)
    dRdp = dres_dp_function(0, dynvarssvals, np.zeros(len(dynvarssvals)),
                            net.constantVarValues)
    # dRdp_reduced, r by p
    dRdp_reduced = np.delete(dRdp, ddynvaridxs, 0)
    # dCdp, (m-r) by p: a zero matrix
    dCdp = np.zeros((len(pools), len(net.optimizableVars)))
    # dfdp, m by p
    dfdp = np.concatenate((dRdp_reduced, dCdp), 0)
    # jac, m by p
    mat = -np.matrix(dfdx).I * np.matrix(dfdp)
    mat = MCAMatrix(mat, rowvarids=net.dynamicVars.keys(),
                    colvarids=net.optimizableVars.keys())
    net.concentration_response_matrix = net.R_s = mat
    return mat


def get_assigned_variable_sensitivity_matrix(net, assignvarid, paramids=None,
                                             paramvals=None, normalize=False):
    """
    Compute the total derivative of an assigned variable w.r.t. all 
    optimizable parameters.
    The total derivative is defined as:
    Drule/Dpi = drule/dpi + sum_over_sj(drule/dsj * dsj/dpi) for parameter pi,
    where D means total derivative and d means partial derivative.
    
    It requires that the assigned variable is a function of only parameters and 
    dynamic variables, but *not* of other assigned variables.
    """
    update_net(net, paramvals=paramvals, time=np.inf)
    rule = net.assignmentRules.get(assignvarid)
    if paramids is None:
        paramids = net.optimizableVars.keys()
    dynvarids = net.dynamicVars.keys()
    drule_dp_fix_s = [net.evaluate_expr(Ex.diff_expr(rule, paramid)) 
                      for paramid in paramids]
    drule_ds_fix_p = [net.evaluate_expr(Ex.diff_expr(rule, dynvarid))
                      for dynvarid in dynvarids]
    ds_dp = get_concentration_response_matrix(net, paramvals=paramvals)
    mat = np.array(drule_dp_fix_s) + \
          np.matrix(drule_ds_fix_p).reshape(1, len(dynvarids)) * ds_dp
    if normalize:
        paramvals = [net.variables.get(paramid).value for paramid
                     in paramids]
        assignvarval = net.assignedVars.get(assignvarid).value
        mat = mat / assignvarval * np.matrix(np.diag(paramvals))
    mat = MCAMatrix(mat, rowvarids=[assignvarid], colvarids=paramids)
    return mat


def get_sensitivity_matrix_finite_difference(net, varids, paramid,
                                             paramvals=None, eps=0.01, 
                                             normed=False):
    """
    Return sensitivity matrix using two-sided finite difference.

    Compared to sensitivities computed through analytical differentiations,
    the errors are around 10% for most variables, but around 100% for some, 
    no matter whether using one-sided or two-sided finite difference.
    That different variables have different errors is probably due to that 
    different variables have different characteristic scales.
      
    Arguments:
      varids: a list of variable ids for which the sensitivities w.r.t.
              a single parameter are computed
      paramid: the id of the single parameter
      paramvals: parameter values, corresponding to the location in 
                 the parameter space to calculate sensitivities at
      eps: relative epsilon, step size h = parameter value * eps
    """
    update_net(net, paramvals=paramvals)
    paramval = net.parameters.get(paramid).value
    varvals = np.array([net.variables.get(varid).value 
                        for varid in varids])
    h = paramval * eps
    # l: left; r: right
    net_l = net.copy()
    update_net(net_l, deltaparamvals={paramid: -eps}, relative=True, 
               time=np.inf)
    varvals_l = np.array([net_l.variables.get(varid).value 
                          for varid in varids])
    
    net_r = net.copy()
    update_net(net_r, deltaparamvals={paramid: eps}, relative=True, 
               time=np.inf)
    varvals_r = np.array([net_r.variables.get(varid).value 
                          for varid in varids])
    
    sensmat = (varvals_r - varvals_l) / (2 * h)
    if normed:
        sensmat = sensmat * paramval / varvals 
    sensmat = np.reshape(sensmat, (len(varids), 1))
    return MCAMatrix(sensmat, rowvarids=varids, colvarids=[paramid])


def get_2nd_order_sensitivities_finite_difference(net, varids, paramid1,
                                                  paramid2, paramvals=None,
                                                  dynvarvals=None, eps=0.01):
    """
    """
    if paramvals is None:
        paramvals = [p.value for p in net.optimizableVars]
    netutil.update_net(net, paramvals=paramvals, dynvarvals=dynvarvals)
    # ll: lower left; lr: lower right; ul: upper left; ur: upper right
    h = net.parameters.get(paramid1).value * eps
    k = net.parameters.get(paramid2).value * eps
    update_net(net, time=np.inf, paramvals=paramvals,
               deltaparamvals={paramid1:-h, paramid2:-k}, relative=False)
    varvals_ll = np.array([net.variables.get(varid).value for varid in varids]) 
    update_net(net, time=np.inf, paramvals=paramvals,
               deltaparamvals={paramid1: h, paramid2:-k}, relative=False)
    varvals_lr = np.array([net.variables.get(varid).value for varid in varids])
    update_net(net, time=np.inf, paramvals=paramvals,
               deltaparamvals={paramid1:-h, paramid2: k}, relative=False)
    varvals_ul = np.array([net.variables.get(varid).value for varid in varids])
    update_net(net, time=np.inf, paramvals=paramvals,
               deltaparamvals={paramid1: h, paramid2: k}, relative=False)
    varvals_ur = np.array([net.variables.get(varid).value for varid in varids]) 
    
    sens = (varvals_ur + varvals_ll - varvals_lr - varvals_ul) / (4 * h * k)
    return dict(zip(varids, sens))


def get_2nd_order_sensitivity_matrix_finite_difference(net, varid,
                                    paramvals=None, dynvarvals=None, eps=0.01):
    """
    """
    n = len(net.optimizableVars)
    paramids = net.optimizableVars.keys()
    mat = np.matrix(np.zeros((n, n)))
    # make an upper triangular matrix
    for i in range(n):
        for j in range(i + 1):
            sens = get_2nd_order_sensitivities_finite_difference(net,
                varids=[varid], paramid1=paramids[i], paramid2=paramids[j],
                paramvals=paramvals, dynvarvals=dynvarvals, eps=eps)
            mat[i, j] = sens.values()[0]  # sens is a dict
    # make a symmetric matrix
    mat = mat + mat.getT() - np.diag(np.diag(mat))
    return MCAMatrix(mat, rowvarids=paramids, colvarids=paramids)


def get_pool_sizes(net=None, dynvarvals=None, poolmat=None):
    """
    Return a numpy 1D array with length the same as the number of conservation
    pools.
    """
    if dynvarvals is None:
        dynvarvals = [var.value for var in net.dynamicVars]
    if poolmat is None:
        poolmat = get_pool_mul_mat(net)
    dynvarvals = np.matrix(dynvarvals).reshape(len(dynvarvals), 1)
    poolsizes = np.array(poolmat * dynvarvals).flatten()
    return poolsizes


def get_rhs_full(net):
    """
    """
    pass
        

def get_fsolve_function(net, paramvals=None):
    """
    Return a function to be passed to scipy.optimization.fsolve for 
    root finding.
    """
    update_net(net, paramvals=paramvals)
    if hasattr(net, 'fsolvefunc') and\
        net.fsolvefunc.paramvals == [p.value for p in net.optimizableVars]:
        return net.fsolvefunc
    # either the network does not have fsolvefunc stored, or the parameter
    # values of the network do not agree with the provided ones;
    # in either case fsolvefunc needs to be built from scratch
    net.compile()
    poolmat = get_pool_mul_mat(net)
    if np.any(poolmat):  # if true then there are conservation pools
        poolsizes_init = get_pool_sizes(net)
        idynvarids = get_indep_dyn_var_ids(net)
        idynvaridxs = [net.dynamicVars.keys().index(idynvarid) 
                       for idynvarid in idynvarids]
    
    def fsolvefunc(dynvarvals):
        """
        This is a function to be passed to scipy.optimization.fsolve, 
        (hence the name), which takes values of all dynamic variable (x) 
        as input and outputs the time-derivatives of independent 
        dynamic variables (dxi/dt) and the differences between
        the current pool sizes (as determined by the argument dynvarvals)
        and the initial pool sizes.
        """
        # from SloppyCell doc: 
        # res_function(time,dynamicVars,yprime,constants)
        derivs = net.res_function(0, dynvarvals, np.zeros(len(dynvarvals)),
                                  net.constantVarValues)
        if np.any(poolmat):
            derivs_indep = derivs[idynvaridxs]  # dxi/dt
            poolsizes_curr = get_pool_sizes(dynvarvals=dynvarvals, poolmat=poolmat)
            poolsizes_diff = poolsizes_curr - poolsizes_init
            return np.concatenate((derivs_indep, poolsizes_diff))
        else:
            return derivs

    fsolvefunc.paramvals = [p.value for p in net.optimizableVars]
    net.fsolvefunc = fsolvefunc
    
    # update the meta-attribute 'attrs_kinetic'
    if not hasattr(net, 'attrs_kinetic'):
        net.attrs_kinetic = set([])
    net.attrs_kinetic.add('fsolvefunc')
    
    return fsolvefunc


def is_net_steadystate(net, dynvarvals=None, tol=1e-6):
    """
    Return True or False.
    """
    if dynvarvals is None:
        dynvarvals = [var.value for var in net.dynamicVars]
    fsolvefunc = get_fsolve_function(net)
    if np.sum(np.abs(fsolvefunc(dynvarvals))) < tol:
        return True
    else:
        return False
    

def get_dynvarssvals_integration(net, paramvals=None, tol=1e-4, T=1e3,
                                 T_lim=1e6):
    """
    Return the steady state values of dynamic variables found by 
    the (adaptive) integration method, which may or may not represent
    the true steady state.
    Dynvarvals of the network get updated.
    """
    dynvarvals_init = [var.value for var in net.dynamicVars]
    netutil.update_net(net, paramvals=paramvals)
    try:
        traj = Dynamics.integrate(net, [0, T], fill_traj=False)
        dynvarssvals = traj.values[-1, :len(net.dynamicVars)]
        # stop when either the network has reached steady state or
        # the integration time has exceeded the given limit T_lim
        while not is_net_steadystate(net, tol=tol) and T < T_lim:
            T_new = T * 10
            traj = Dynamics.integrate(net, [T, T_new], fill_traj=False)
            dynvarssvals = traj.values[-1, :len(net.dynamicVars)]
            T = T_new
        
        if T == T_lim:
            print "integration limit"
    except:
        dynvarssvals = dynvarvals_init
    return dynvarssvals


def get_dynvarssvals_rootfinding(net, paramvals=None, tol_fsolve=1e-6,
                                 tol_ss=1e-4):
    """
    Return the steady state values of dynamic variables found by 
    the root-finding method, which may or may not represent the true
    steady state. 
    Dynvarvals of the network do NOT get updated.
    """
    netutil.update_net(net, paramvals=paramvals)
    fsolvefunc = get_fsolve_function(net, paramvals=paramvals)
    dynvarvals_net = [var.value for var in net.dynamicVars]
    dynvarssvals = sp.optimize.fsolve(fsolvefunc, dynvarvals_net,
                                      xtol=tol_fsolve)
    return dynvarssvals
    

def get_dynvarssvals(net, paramvals=None, method='integration', T=1e3,
                     T_lim=1e9, tol_fsolve=1e-6, tol_ss=1e-4, 
                     update_dynvarvals=False, mat=False):
    """
    Input:
      method: 'rootfinding' or 'integration'
      T: integration time (when method is 'integration')
      T_max: 
      tol_fsolve: tolerance passed to scipy.optimization.fsolve (when 
                  method is 'rootfinding')
      tol_ss: tolerance passed to steady-state checker
    """
    dynvarvals_init = [var.value for var in net.dynamicVars]
    netutil.update_net(net, paramvals=paramvals)
    
    # the following codes are commented out because they are potentially buggy:
    # imagine a network of a reversible reaction with mass action kinetics of
    # rate constants 1e-16 and 3e-16 respectively: dynvarssvals = [0.75, 0.25];
    # but if net has dynvarvals = [1, 0], it will pass the following 
    # steady-state test: fsolvefunc([1,0]) < tol. 
    # if is_net_steadystate(net, tol=tol_ss):
    #     return [var.value for var in net.dynamicVars]
    
    if method == 'integration':
        dynvarssvals =\
            get_dynvarssvals_integration(net, paramvals=paramvals,
                                         T=T, T_lim=T_lim, tol=tol_ss)
        if not is_net_steadystate(net, dynvarssvals, tol_ss):
            dynvarssvals =\
                get_dynvarssvals_rootfinding(net, paramvals=paramvals,
                                             tol_fsolve=tol_fsolve,
                                             tol_ss=tol_ss)
    if method == 'rootfinding':
        dynvarssvals =\
            get_dynvarssvals_rootfinding(net, paramvals=paramvals,
                                         tol_fsolve=tol_fsolve, tol_ss=tol_ss)
        if not is_net_steadystate(net, dynvarssvals, tol_ss):
            dynvarssvals =\
                get_dynvarssvals_integration(net, paramvals=paramvals, T=T,
                                             T_lim=T_lim, tol=tol_ss)
    
    # change back the dynamic variable values to prevent them from
    # drifting too far to make steady state calculations unstable
    
    # update the dynvarvals of the network to help subsequent 
    # callings of function get_dynvarssvals_rootfinding as dynvarvals
    # will be fed in as the initial values for scipy.optimize.fsolve    
    if not update_dynvarvals:
        netutil.update_net(net, dynvarvals=dynvarvals_init)
    
    if is_net_steadystate(net, dynvarssvals, tol=tol_ss):
        if mat:
            dynvarssvals = MCAMatrix(np.diag(dynvarssvals), 
                                     rowvarids=net.dynamicVars.keys(),
                                     colvarids=net.dynamicVars.keys())
        return dynvarssvals
    else:
        print "Warning: net has not reached steady state for parameters: ", \
              [p.value for p in net.optimizableVars]
        return [np.nan] * len(dynvarssvals)
get_steadystate_dynvarvals = get_dynvarssvals


def get_steadystate_fluxes(net, pvals=None, mat=False):
    """
    Input: 
        mat: if True, return diagonal matrix
    
    """
    if not hasattr(net, 'fluxVars'):
        net = netutil.add_fluxes(net)
    update_net(net, paramvals=pvals, time=np.inf)
    Js = np.array([var.value for var in net.fluxVars])
    if mat:
        # note net.reactions.keys() instead of net.fluxVars.keys() are used; 
        # eg, 'R1' vs 'J_R1'
        Js = MCAMatrix(np.diag(Js), rowvarids=net.reactions.keys(), 
                       colvarids=net.reactions.keys())
    return Js
get_ss_fluxes = get_steadystate_fluxes


def get_ssvals(net, varids=None, paramvals=None):
    """
    Get steady state values for the provided varids. 
    """
    if varids is None:
        # SloppyCell by default uses these varids for traj
        varids = net.dynamicVars.keys() + net.assignedVars.keys()
    update_net(net, paramvals=paramvals, time=np.inf)
    ssvals = np.array([net.variables.get(varid).value for varid in varids])
    return ssvals
        

def get_dynvarvals(net, paramvals=None, time=np.inf):
    """
    """
    netutil.update_net(net, paramvals=paramvals)
    if time == np.inf:
        return get_dynvarssvals(net, paramvals=paramvals)
    else:
        traj = Dynamics.integrate(net, [0, time], params=paramvals,
                                  fill_traj=False)
        # take values of only dynamic variables (others are assigned variables)
        return traj.values[-1, :len(net.dynamicVars)]
    
    
def update_net(net, paramvals=None, deltaparamvals=None, relative=False,
               time=None):
    """
    The function does the following three things:
        1. Update parameters if provided paramvals or deltaparamvals
        2. Remove parameter-dependent attributes if 1
        3. Update dynamic and assigned variables if provided time
    """
    ## Update parameters
    optvarids = net.optimizableVars.keys()
    # determine paramvals (unify the type to list)
    if paramvals is None:
        paramvals = [p.value for p in net.optimizableVars]
    else:
        try:  # a mapping type
            paramvals = butil.get_values(paramvals, optvarids)
        except AttributeError:  # a sequence type
            paramvals = list(paramvals)
    # determine deltaparamvals (unify the type to list)
    if deltaparamvals is None:
        deltaparamvals = [0] * len(net.optimizableVars)
    else:
        try:  # a mapping type
            deltaparamvals = butil.get_values(deltaparamvals, optvarids, 0)
        except AttributeError:  # a sequence type
            deltaparamvals = list(deltaparamvals)
    # change the type to np.array
    paramvals, deltaparamvals = np.array(paramvals), np.array(deltaparamvals)
    # determine the new paramvals
    if relative:
        paramvals = paramvals * (1 + deltaparamvals)
    else:
        paramvals = paramvals + deltaparamvals
    if list(paramvals) != [p.value for p in net.optimizableVars]:
        net.update_optimizable_vars(paramvals)
        ## Remove parameter-dependent attributes
        remove_attrs(net, rm_kinetic=True, rm_struct=False)  
    
    ## Update dynamic and assigned variables
    if time:
        dynvarvals = get_dynvarvals(net, time=time)
        if np.nan in dynvarvals:
            raise ValueError("dynvarvals of net %s has np.nan"%net.id)
        net.updateVariablesFromDynamicVars(dynvarvals, time=time)


"""
def get_updated_net(net, netid=None, paramvals=None, deltaparamvals=None,
                    relative=False, time=None):
    Get an network updated in the following attributes:
        parameters, dynamic variables and assigned variables
    Also wipe out old attributes that might depend on parameters.
    net2 = net.copy(netid)
    update_net(net2, netid=netid, paramvals=paramvals, 
               deltaparamvals=deltaparamvals, relative=relative, time=time)
    return net2
"""

"""
    def GetResult(net, varids):

    # A replacement of SloppyCell.ReactionNetworks.Network_mod.Network.GetResult 

    result = {}
    for varid in varids:
        traj = net.trajectory.getVariableTrajectory(varid)
        result[varid] = dict(zip(net.trajectory.timepoints, traj))
    return result
"""


def jacobian_log_params_sens(mod, paramvals, log=False):
    """
    A replacemnet of SloppyCell.Model_mod.Model.jacobian_log_params_sens
    to work for *steady states*. Like the SloppyCell version, it returns 
    a keyedlist. 
    
    Input:
        paramvals: parameter values (log=False) 
                   or log-parameter values (log=True, SloppyCell's convention) 
    """
    if log:
        paramvals = np.exp(paramvals)
    jac = jacobian_sens(mod, paramvals)
    jac_logparams = jac.copy()
    jac_logparams.update(np.array(jac)*paramvals)
    return jac_logparams


def jacobian_sens(mod, paramvals):
    """
    A replacement of SloppyCell.Model_mod.Model.jacobian_sens to work for
    *steady states*. Like the SloppyCell version, it returns a keyedlist.
    """
    ## require that all nets have the same optimizable parameters
    optvarids_nets = [net.optimizableVars.keys() 
                      for net in mod.calcColl.values()]
    if not butil.all_same(optvarids_nets):
        raise StandardError("nets do not have the same optimizable variables.")
    
    # require that all times are np.inf
    times = np.array([res.xVal for res in mod.residuals.values() 
                      if hasattr(res, 'xVal')])
    if np.any(times != np.inf):
        raise StandardError("not all times in the residuals are np.inf.")
    
    ## compute jac_norm, the jacobian normalized by sigmas
    # initialize jac_norm
    jac_norm = KeyedList(zip(mod.residuals.keys(), 
                             [None]*len(mod.residuals)))
    # update jac_norm
    for (reskey, res) in mod.residuals.items():
        if jac_norm.get(reskey) is None:  # the value has not been filled yet
            net = mod.calcColl.get(res.calcKey)
            varid = res.yKey
            # if the variable is a steady-state concentration
            if varid in net.dynamicVars.keys():
                # by default jac_concn is not normalized
                jac_concn = get_concn_response_mat(net, paramvals)
                # fill all possible entries of jac_norm
                for rowvarid in jac_concn.rowvarids:
                    for (reskey, res) in mod.residuals.items():
                        if hasattr(res, 'calcKey') and res.calcKey==net.id and\
                                hasattr(res, 'yKey') and res.yKey==rowvarid:
                            rowidx = jac_concn.rowvarids.index(rowvarid)
                            # normalized by sigma
                            sens_norm = jac_concn[rowidx] / res.ySigma
                            sens_norm = np.array(sens_norm).flatten().tolist()
                            jac_norm.set(reskey, sens_norm)
            # if the variable is a steady-state flux
            if hasattr(net, 'fluxVars') and varid in net.fluxVars.keys():
                # by default jac_flux is not normalized
                jac_flux = get_flux_response_mat(net, paramvals)
                # fill all possible entries of jac_norm
                for rowvarid in jac_flux.rowvarids:
                    for (reskey, res) in mod.residuals.items():
                        if hasattr(res, 'calcKey') and res.calcKey==net.id and\
                                hasattr(res, 'yKey') and res.yKey==rowvarid:
                            rowidx = jac_flux.rowvarids.index(rowvarid)
                            # normalized by sigma
                            sens_norm = jac_flux[rowidx] / res.ySigma
                            sens_norm = np.array(sens_norm).flatten().tolist()
                            jac_norm.set(reskey, sens_norm)
                                          
    return jac_norm

    """
    nets = mod.GetCalculationCollection()  # a list of nets 
    residuals = mod.GetResiduals()
    jac_scaled = KeyedList()
    for idx, net in enumerate(nets):
        expt = mod.exptColl[idx]
        jac_net = libmca.get_concentration_response_matrix(net, paramvals)
        for (resid, res) in residuals.items():
            # two kinds of residuals: one from data and one from priors;
            # only data residuals have ids in tuple
            if isinstance(resid, tuple) and res.calcKey == net.id:
                rowidx = net.dynamicVars.keys().index(res.yKey)
                # scaled by sigma
                sens_scaled = jac_net[rowidx] / res.ySigma
                sens_scaled = np.array(sens_scaled).flatten().tolist()
                jac_scaled.set(resid, sens_scaled)
    return jac_scaled
    """    


def remove_attrs(net, rm_kinetic=True, rm_struct=False):
    """
    Remove the following attributes:
    * Parameter-independent attributes (structural):
        stoich_mat
        pool_mul_mat
        link_mat
        reduced_link_mat
        reduced_stoich_mat
    * Parameter-dependent attributes (structural + kinetic):
        concn_elas_mat
        param_elas_mat
        jac
        concn_ctrl_mat
        flux_ctrl_mat
        concn_response_mat
        flux_response_mat
    """
    # collect the attributes to be removed
    attrs_rm = set([])
    if rm_kinetic:
        attrs_rm = set.union(attrs_rm, net.attrs_kinetic)
        net.attrs_kinetic = set([])
    if rm_struct:
        attrs_rm = set.union(attrs_rm, net.attrs_structural)
        net.attrs_struct = set([])
    # remove the attributes
    for attr in attrs_rm:
        delattr(net, attr)


