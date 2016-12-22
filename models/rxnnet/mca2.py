"""
This is a library of class and functions for computations related to
steady states and Metabolic Control Analysis (MCA).
MCA is essentially first-order sensitivity analysis of a metabolic models at
steady-state. For this reason, a network instance passed to any function in
this library is checked for steady state.

            -1                 -1
Cs = -(N Es)   N = -L (Nr Es L)   Nr

"""

from __future__ import division
from collections import OrderedDict as OD

import numpy as np
import scipy as sp
from SloppyCell import ExprManip as exprmanip

from util import butil
from util.butil import Series
from util.matrix import Matrix


TMIN = 1e2
TMAX = 1e8
TOL_SS = 1e-9
K = 100  # can be confused with net.K?
METHOD = 'integration'


def is_ss(net, tol=None):
    """Determine whether the net has reached steady state.
    
    Caution: the code is problematic. It will falsely conclude that a net 
    has reached steady state if the nets has a very long characteristic time scale, 
    which can happen when exploring some corners of parameter space, eg,
    k=1e-9. ## FIXME ****  
    
    Output:
        True or False
    """
    if tol is None:
        tol = TOL_SS
        
    if net.get_dxdt().abs().max() < tol:
        return True
    else:
        return False


# to be timed for realistic nets; FIXME ***
def get_s_integration(net, p=None, Tmin=None, Tmax=None, k=None, tol_ss=None,
                      to_ser=True):
    """
    Input:
        T0:
        Tmax:
        k:
        tol_ss:
    """
    # Delayed argument binding to propagate changes in the settings of 
    # global variables
    if Tmin is None:
        Tmin = TMIN
    if Tmax is None:
        Tmax = TMAX
    if k is None:
        k = K
    if tol_ss is None:
        tol_ss = TOL_SS
        
    if p is not None:
        net.update(p=p)
        
    # Comment the following out because:
    # imagine this situation: I update net.p, and want to get net.s; 
    # now net.x is the s for the previous p, and is hence erroneously returned. 
    
    #if net.is_ss(tol=tol_ss):  
    #    return net.x
    #else:
    t = Tmin
    while t <= Tmax:
        net.update(t=t)  # could spit out daeintException
        if net.is_ss(tol=tol_ss):
            return net.x
        else:
            t *= k
    raise Exception("Unable to reach steady state for p: %s"%\
                    str(net.p.tolist()))
    

"""
def get_s_integration(net, p=None, Tmax=TMAX, tol_ss=TOL_SS):
    if p is not None:
        net.update(p=p)
    net.update(t=Tmax)
    if net.is_ss(tol=tol_ss):
        return net.x
    else:
        raise Exception("unable to reach steady state for p: %s"%
                        str(net.p.tolist()))
"""


def get_s_rootfinding(net, p=None, x0=None, to_ser=True, **kwargs_fsolve):

    """Return the steady state values of dynamic variables found by 
    the root-finding method, which may or may not represent the true
    steady state. 
    
    Dynvarvals of the network do NOT get updated.
    
    Input:
        p:
        x0: initial guess in rootfinding; by default the current x of net
        to_ser:
        kwargs_fsolve: 
        
    Documentation of scipy.optimize.fsolve:
    """
    if net.is_ss():
        return net.x
        
    if p is not None:
        net.update(p=p)
    P = net.P
    if P.nrow > 0:
        poolsizes = P.apply(lambda row: np.dot(row, net.x0), axis=1)
    
    def _f(x):
        """This is a function to be passed to scipy.optimization.fsolve, 
        which takes values of all dynamic variable (x) 
        as input and outputs the time-derivatives of independent 
        dynamic variables (dxi/dt) and the differences between
        the current pool sizes (as determined by the argument dynvarvals)
        and the correct pool sizes.
        """
        dxdt = net.get_dxdt(x=x)
        
        if P.nrow > 0:
            dxidt = dxdt[net.ixids]
            poolsizes_diff = P.apply(lambda row: np.dot(row, x), axis=1) -\
                poolsizes
            return dxidt.append(poolsizes_diff)
        else:
            return dxdt

    if x0 is None:
        x0 = net.x
    
    s = sp.optimize.fsolve(_f, x0, **kwargs_fsolve)
    if to_ser:
        s = Series(s, net.xids)
    return s

get_s_rootfinding.__doc__ += sp.optimize.fsolve.__doc__
    

def get_s(net, p=None, method=None, Tmin=None, Tmax=None, k=None, 
          tol_ss=None, x0=None, to_ser=False, **kwargs_fsolve):
    """Get steady-state concentrations of dynamic variables.
    
    Input: 
        p:
        method: str, 'integration', 'rootfinding' or 'mixed'; default is 'mixed';
            'rootfinding' should be the preferred way of getting steady-state
            if accuracy is desired (two levels of error sources in 'integration',
            integration itself and calling steady-state). Need to time the methods
            to have a better sense of efficiency comparison. FIXME *** 
        T0: 
        Tmax:
        k: 
        tol_ss:
        x0: initial guess for rootfinding
        to_ser: 
        kwargs_fsolve: kwargs for scipy.optimize.fsolve; its docstring is appended
            below for convenience.
    
    Output: 
        s: a numpy.ndarray (default) or Series  
        
    Documentation of scipy.optimize.fsolve: 
    
    """
    if method is None:
        method = METHOD
        
    if method == 'integration':
        return get_s_integration(net, p=p, 
                                 Tmin=Tmin, Tmax=Tmax, k=k, tol_ss=tol_ss,
                                 to_ser=to_ser)
    if method == 'rootfinding':
        kwargs_fsolve = butil.get_submapping(kwargs_fsolve, f_key=lambda key:\
                                key in ['xtol', 'maxfev', 'epsfcn', 'factor'])
        return get_s_rootfinding(net, p=p, x0=x0, to_ser=to_ser, **kwargs_fsolve)
    if method == 'mixed':
        # first get down to the basin of attraction for fsolve
        if Tmin is None:
            Tmin = TMIN
        return get_s_rootfinding(net, p=p, x0=net.get_x(p=p, t=Tmin), 
                                 to_ser=to_ser, **kwargs_fsolve)
get_s.__doc__ += sp.optimize.fsolve.__doc__
    

def set_ss(net, *args, **kwargs):
    """Signature is the same as get_s, whose docstring is provided below 
    for convenience.
    
    Input:
    
    Output: 
        None

    Return the steady state values of dynamic variables found by 
    the (adaptive) integration method, which may or may not represent
    the true steady state.
    Dynvarvals of the network get updated.
    """
    s = get_s(net, *args, **kwargs)
    net.updateVariablesFromDynamicVars(s, time=np.inf)
    net.t = np.inf
set_ss.__doc__ += get_s.__doc__


"""
# How it was done before...

def get_concn_elas_mat0(net, p=None, normed=False):
    net.update(p=p, t=np.inf)
    rxnids, xids = net.rxnids, net.xids
    Es = Matrix(np.zeros((len(rxnids), len(xids))), rxnids, xids)
    for rxnid in rxnids:
        ratelaw = net.rxns[rxnid].kineticLaw
        for xid in xids:
                Es[rxnid, xid] = net.evaluate_expr(expr.diff_expr(ratelaw, xid))
    if normed:
        return Es.normalize(net.J, net.s)
    else:
        return Es
"""


def get_Ex_str(net):
    """
    """
    Ex = []
    for rxnid in net.rxnids:
        ratelaw = exprmanip.sub_for_vars(net.rxns[rxnid].kineticLaw, 
                                         net.asgrules.to_od())
        Ex_rxn = []
        for xid in net.xids:
            Ex_rxn.append(exprmanip.simplify_expr(exprmanip.diff_expr(ratelaw, xid)))
        Ex.append(Ex_rxn)
    Ex_str = str(Ex).replace("'", "")   
    Ex_code = compile(Ex_str, '', 'eval')  # compile to code object
    net.Ex_str, net.Ex_code = Ex_str, Ex_code
    return Ex_str, Ex_code


def get_Ep_str(net):
    """
    """
    Ep = []
    for rxnid in net.rxnids:
        ratelaw = exprmanip.sub_for_vars(net.rxns[rxnid].kineticLaw, 
                                         net.asgrules.to_od())
        Ep_rxn = []
        for pid in net.pids:
            Ep_rxn.append(exprmanip.simplify_expr(exprmanip.diff_expr(ratelaw, pid)))
        Ep.append(Ep_rxn)
    Ep_str = str(Ep).replace("'", "")   
    Ep_code = compile(Ep_str, '', 'eval')  # compile to code object
    net.Ep_str, net.Ep_code = Ep_str, Ep_code
    return Ep_str, Ep_code

"""
def get_E_strs(net):

    rxnids, xids, pids = net.rxnids, net.xids, net.pids
    Ex, Ep = [], []
    for rxnid in rxnids:
        ratelaw = net.rxns[rxnid].kineticLaw
        Ex_rxn = []
        Ep_rxn = []
        for xid in xids:
            Ex_rxn.append(expr.diff_expr(ratelaw, xid))  # diff also simplifies
        for pid in pids:
            Ep_rxn.append(expr.diff_expr(ratelaw, pid))  # diff also simplifies
        Ex.append(Ex_rxn)
        Ep.append(Ep_rxn)
    Ex_str = str(Ex).replace("'", "")   
    Ep_str = str(Ep).replace("'", "")
    Ex_code = compile(Ex_str, '', 'eval')
    Ep_code = compile(Ep_str, '', 'eval')
    net.Ex_str = Ex_str
    net.Ep_str = Ep_str
    net.Ex_code = Ex_code
    net.Ep_code = Ep_code
    return Ex_code, Ep_code

def set_E_funcs(net):

    Ex_str, Ep_str = get_E_strs(net)
    exec('def get_Ex(net, **vals):\n\treturn '+Ex_str, )
    exec('def get_Ep(net, **vals):\n\treturn ' + Ep_str)
    setattr(net, 'get_Ex', )
    setattr(net, 'get_Ep', )
"""


def get_concn_elas_mat(net, p=None, normed=False):
    """
    FIXME ***: compile or generate dynamic Python functions
    """
    net.update(p=p, t=np.inf)
    ns = net.namespace.copy()  # without copy, the namespace is contaminated
    ns.update(net.varvals.to_dict())
    if not hasattr(net, 'Ex_code'):
        Ex_code = get_Ex_str(net)[1]
    else:
        Ex_code = net.Ex_code
    Es = Matrix(eval(Ex_code, ns), net.vids, net.xids)
    if normed:
        return Es.normalize(net.v, net.s)
    else:
        return Es


def get_param_elas_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)      
    ns = net.namespace.copy()
    ns.update(net.varvals.to_dict())
    if not hasattr(net, 'Ep_str'):
        Ep_code = get_Ep_str(net)[1]
    else:
        Ep_code = net.Ep_code
    Ep = Matrix(eval(Ep_code, ns), net.vids, net.pids)
    if normed:
        return Ep.normalize(net.v, net.p)
    else:
        return Ep


def get_jac_mat(net, p=None, x=None):
    """
    Return the jacobian matrix (M) of the network, which, _in the MCA context_,
    is the jacobian of the independent vector field dxi/dt = Nr * v(xi,xd,p)
    (so that M is invertible).
    """
    if x is None:
        net.update(p=p, t=np.inf)
        L, Es, Nr = net.L, net.Es, net.Nr.ch_colvarids(net.vids)
        M = Nr * Es * L
    else:
        if p is not None:
            net.update(p=p)
        Exstr = get_Ex_str(net)[0]
        x = Series(x, net.xids)
        Ex = Matrix(eval(Exstr, dict(net.varvals.items()+x.items())),
                    net.rxnids, net.xids)
        M = net.N * Ex
    return M


def get_concn_ctrl_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    L, M, Nr = net.L, net.M, net.Nr.ch_colvarids(net.vids)
    Cs = -L * M.inv() * Nr
    if normed:
        return Cs.normalize(net.s, net.v)
    else:
        return Cs
        

def get_flux_ctrl_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    I, Es, Cs = Matrix.eye(net.Jids, net.vids), net.Es, net.Cs
    CJ = I + (Es * Cs).ch_rowvarids(net.Jids)
    if normed:
        return CJ.normalize(net.J, net.v)
    else:
        return CJ
    

def get_concn_resp_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    Ep, Cs = net.Ep, net.Cs
    Rs = Cs * Ep
    if normed:
        return Rs.normalize(net.s, net.p)
    else:
        return Rs


def get_flux_resp_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    Ep, CJ = net.Ep, net.CJ
    RJ = CJ * Ep
    if normed:
        return RJ.normalize(net.J, net.p)
    else:
        return RJ


def get_concn_resp_mat_fd(net, p=None, normed=False):
    """
    """
    if p is None:
        p = net.p
    for pid_, p_ in p.items():
        pass 
    net.update(p=p, t=np.inf)
    Ep, Cs = net.Ep, net.Cs
    Rs = Cs * Ep
    if normed:
        return Rs.normalize(net.s, net.p)
    else:
        return Rs


def jws2mat(filepath, name=None):
    """Parse the output file of JWS online.
    
    Input:
        filepath:
        name: str; 'Cs', 'nCs', 'CJ', 'nCJ'
    """
    fh = open(filepath)
    lines = fh.readlines()
    fh.close()
    if name is None:
        name = filepath.split('/')[-1].split('_')[0]
    
    if name == 'nEs':
        add_prefix_row = lambda varid: 'log_v_' + varid
        add_prefix_col = lambda varid: 'log_' + varid
    elif name == 'nCs':
        add_prefix_row = lambda varid: 'log_' + varid
        add_prefix_col = lambda varid: 'log_v_' + varid
    else:
        add_prefix_row = lambda varid: varid
        add_prefix_col = lambda varid: varid

    colvarids = [add_prefix_col(s.strip()) for s in lines[0].split(',')[1:]]
    rowvarids = []
    mat = []
    for line in lines[1:]:
        rowvarids.append(add_prefix_row(line.split(',')[0]))
        mat.append([float(s) for s in line.split(',')[1:]])
    mat = Matrix(mat, rowvarids, colvarids)
    return mat


def copasi2mats(filepath, name=None):
    """Parse the output file of Copasi. 
    
    Input:
        filepath: 
        name: 
    """
    _trim0 = lambda s: s.replace('(','').replace(')','')
    _trim = lambda s: _trim0(s) if isinstance(s,str) else [_trim0(_) for _ in s]
    fh = open(filepath)
    lines = fh.readlines()
    fh.close()
    parts = ''.join(lines).split('\n\n')[2:]
    name2mat = OD()
    try:
        for part in parts:
            lines_part = part.split('\n')
            lines_mat = lines_part[4:]
            colvarids = _trim(lines_mat[0].split('\t')[1:])
            mat = []
            rowvarids = []
            for line in lines_mat[1:]:
                rowvarids.append(_trim(line.split('\t')[0]))
                mat.append([float(s) for s in line.split('\t')[1:]])
            name2mat[lines_part[1]] = Matrix(mat, rowvarids, colvarids)
    except:
        pass
    if name:
        return name2mat[name]
    else:
        return name2mat

"""
def cmp_pysces(rxnid_v='RBCO', rxnid_J='TPI'):
    
    import pandas as pd
    import pysces  # the import would force changing the directory (can we change it?)


    sbmlname = 'model_poolman2000_arnold1.xml'
    sbmldir = '/Users/lei/Dropbox/Research/CalvinCycle/models'
    
    pysces.interface.convertSBML2PSC(sbmlname, sbmldir=sbmldir)
    
    mod = pysces.model(sbmlname)
    
    mod.doLoad()
    mod.doMca()
    
    s = pd.Series(dict([(k.rstrip('_ss'),v) for (k,v) in mod.__dict__.items() 
                        if k.endswith('_ss')]))
    J = pd.Series(dict([(k,v) for (k,v) in mod.__dict__.items() 
                        if k.startswith('J_')]))
    
    CJ_Jid_vid = mod.__dict__['ccJ%s_%s'%(rxnid_J, rxnid_v)]
    
    return CJ_Jid_vid
"""