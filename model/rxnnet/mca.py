"""
This is a library of class and functions for computations related to
steady states and Metabolic Control Analysis (MCA).
MCA is essentially first-order sensitivity analysis of a metabolic model at
steady-state. For this reason, a network instance passed to any function in
this library is checked for steady state.
"""

from __future__ import division
from collections import OrderedDict as OD

import numpy as np
#import scipy as sp
from SloppyCell import ExprManip as expr

#from util2 import butil
#from util2.sloppycell.mca import mcautil
#reload(butil)
#reload(mcautil)

#import matrix
#reload(matrix)
from matrix import Matrix
    
    
"""
def update(net, p=None):
    used at the beginning of all mca calculations... 

    #if p is not None:
    #    net.set_p(p)
    #net.set_ss()
    net.update(p=p, t=np.inf)
"""
    

def is_ss(net, tol=1e-6):
    """
    Output:
        True or False
    """
    vels = net.get_velocities()
    if vels.abs().max() < tol:
        return True
    else:
        return False


def set_ss(net, tol=1e-6, method='integration', T0=1e3, Tmax=1e6):
    """
    Input:
    
    Output: 
        No output

    Return the steady state values of dynamic variables found by 
    the (adaptive) integration method, which may or may not represent
    the true steady state.
    Dynvarvals of the network get updated.
    """
    if method == 'integration':
        t = T0
        while not net.is_ss() and t < Tmax:
            net.update(t=t)
            t *= 10
            
        if net.is_ss():
            pass
        else:
            raise Exception("not being able to reach steady state")
    if method == 'rootfinding':
        pass


def get_concn_elas_mat_old(net, p=None, normed=False):
    """
    How it was done before...
    """
    net.update(p=p, t=np.inf)
    rxnids, dynvarids = net.rxnids, net.dynvarids
    Es = Matrix(np.zeros((len(rxnids), len(dynvarids))),
                rowvarids=rxnids, colvarids=dynvarids)
    for rxnid in rxnids:
        ratelaw = net.rxns[rxnid].kineticLaw
        for dynvarid in dynvarids:
                Es[rxnid, dynvarid] = net.evaluate_expr(expr.diff_expr(ratelaw, dynvarid))
    if normed:
        J, s = net.J, net.s
        Es = Es.normalize(J, s)
    return Es


def get_E_strs(net):
    """
    FIXME: **
    Check structures?
    """
    rxnids, dynvarids, pids = net.rxnids, net.dynvarids, net.pids
    Es, Ep = [], []
    for rxnid in rxnids:
        ratelaw = net.rxns[rxnid].kineticLaw
        Es_rxnid = []
        Ep_rxnid = []
        for dynvarid in dynvarids:
            Es_rxnid.append(expr.diff_expr(ratelaw, dynvarid))
        for pid in pids:
            Ep_rxnid.append(expr.diff_expr(ratelaw, pid))
        Es.append(Es_rxnid)
        Ep.append(Ep_rxnid)
    Es_str = str(Es).replace("'","")     
    Ep_str = str(Ep).replace("'","")
    net.Es_str = Es_str
    net.Ep_str = Ep_str
    #Es_str = compile(Es_str, "logfile.txt", "eval")
    #Ep_str = compile(Ep_str, "logfile.txt", "eval")
    return Es_str, Ep_str


def get_concn_elas_mat(net, p=None, normed=False):
    """
    FIXME ***: compile or generate dynamic Python functions
    """
    net.update(p=p, t=np.inf)
    ns = net.namespace.copy()  # without copy, the namespace is contaminated
    ns.update(dict(net.varvals))
    if not hasattr(net, 'Es_str'):
        Es_str = get_E_strs(net)[0]
    else:
        Es_str = net.Es_str
    Es = Matrix(eval(Es_str, ns), rowvarids=net.rateids, colvarids=net.dynvarids)
    if normed:
        Es = Es.normalize(net.J, net.s)
    return Es


def get_param_elas_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)      
    ns = net.namespace.copy()
    ns.update(dict(net.varvals))
    if not hasattr(net, 'Ep_str'):
        Ep_str = get_E_strs(net)[1]
    else:
        Ep_str = net.Ep_str
    Ep = Matrix(eval(Ep_str, ns), rowvarids=net.rateids, colvarids=net.pids)
    if normed:
        Ep = Ep.normalize(net.J, net.p)
    return Ep


def get_jac_mat(net, p=None):
    """
    Return the jacobian matrix (M) of the network, which, _in the MCA context_,
    is the jacobian of the independent vector field dxi/dt = Nr * v(xi,xd,p)
    (so that M is invertible).
    """
    net.update(p=p, t=np.inf)
    L, Es, Nr = net.L, net.Es, net.Nr.ch_colvarids(net.rateids)
    M = Nr * Es * L
    return M


def get_concn_ctrl_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    L, M, Nr = net.L, net.M, net.Nr.ch_colvarids(net.rateids)
    Cs = -L * M.I * Nr
    if normed:
        Cs = Cs.normalize(net.s, net.v)
    return Cs
        

def get_flux_ctrl_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    I = Matrix.eye(net.fluxids, net.rateids)
    Es, Cs = net.Es, net.Cs
    CJ = I + (Es * Cs).ch_rowvarids(net.fluxids)
    if normed:
        CJ = CJ.normalize(net.J, net.v)
    return CJ
    

def get_concn_resp_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    Ep, Cs = net.Ep, net.Cs
    Rs = Cs * Ep
    if normed:
        Rs = Rs.normalize(net.s, net.p)
    return Rs


def get_flux_resp_mat(net, p=None, normed=False):
    """
    """
    net.update(p=p, t=np.inf)
    Ep, CJ = net.Ep, net.CJ
    RJ = CJ * Ep
    if normed:
        RJ = RJ.normalize(net.J, net.p)
    return RJ

    
def jws2mat(filepath):
    """
    Parse the output file of JWS online.
    
    Input:
        filepath:
    """
    fh = open(filepath)
    lines = fh.readlines()
    fh.close()
    colvarids = ['v_'+s.strip() for s in lines[0].split(',')[1:]]
    rowvarids = []
    mat = []
    for line in lines[1:]:
        rowvarids.append('J_'+line.split(',')[0])
        mat.append([float(s) for s in line.split(',')[1:]])
    mat = Matrix(mat, rowvarids=rowvarids, colvarids=colvarids)
    return mat


def copasi2mats(filepath, name=None):
    """
    Parse the output file of Copasi. 
    
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
            name2mat[lines_part[1]] = Matrix(mat, rowvarids=rowvarids, colvarids=colvarids)
    except:
        pass
    if name:
        return name2mat[name]
    else:
        return name2mat


def cmp_pysces(rxnid_v='RBCO', rxnid_J='TPI'):
    """
    FIXME **
    """
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