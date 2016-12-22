"""
"""

from __future__ import division

import numpy as np
import sympy

from SloppyCell import daskr
from SloppyCell import ExprManip as exprmanip
from SloppyCell.ReactionNetworks import Dynamics

from util import butil
from util.matrix import Matrix

from infotopo import predict
from infotopo.models.rxnnet import mca
reload(predict)
reload(mca)



def get_s(net, p, tol=1e-12, Tmin=1e3, Tmax=1e9, k=1000, to_ser=False):
    #import ipdb
    #ipdb.set_trace()
    
    net.update_optimizable_vars(p)
    nsp = len(net.dynamicVars)
    tmin, tmax = 0, Tmin
    x0 = net.x0.copy()
    constants = net.constantVarValues
    while tmax <= Tmax:
        #traj = Dynamics.integrate(net, [0, t], fill_traj=False)
        yp0 = Dynamics.find_ics(net.x0, net.x0, tmin, net._dynamic_var_algebraic, 
                                [1e-6]*nsp, [1e-6]*nsp, constants, net)[1]
        out = daskr.daeint(res=net.res_function, t=[tmin, tmax], y0=x0, 
                           yp0=yp0, atol=[1e-6]*nsp, rtol=[1e-6]*nsp, 
                           intermediate_output=False, rpar=constants,
                           max_steps=100000.0, max_timepoints=100000.0, 
                           jac=net.ddaskr_jac)
        xt = out[0][-1]
        dxdt = net.res_function(tmax, xt, [0]*nsp, constants)
        if np.max(np.abs(dxdt)) < tol:
            net.updateVariablesFromDynamicVars(xt, tmax)
            if to_ser:
                return butil.Series(xt, net.xids)
            else:
                return xt
        else:
            tmin, tmax = tmax, tmax*k
            x0 = xt
    raise Exception("Cannot reach steady state for p=%s" % p)


def get_J(net, p, tol=1e-12, Tmin=1e4, Tmax=1e8, k=100, to_ser=False):
    #net.add_ratevars()
    net.update_optimizable_vars(p)
    nsp = len(net.dynamicVars)
    tmin, tmax = 0, Tmin
    x0 = net.x0.copy()
    constants = net.constantVarValues
    while tmax <= Tmax:
        #traj = Dynamics.integrate(net, [0, t], fill_traj=False)
        yp0 = Dynamics.find_ics(net.x0, net.x0, tmin, net._dynamic_var_algebraic, 
                                [1e-6]*nsp, [1e-6]*nsp, constants, net)[1]
        out = daskr.daeint(res=net.res_function, t=[tmin, tmax], y0=x0, 
                           yp0=yp0, atol=[1e-6]*nsp, rtol=[1e-6]*nsp, 
                           intermediate_output=False, rpar=constants,
                           max_steps=100000.0, max_timepoints=100000.0, 
                           jac=net.ddaskr_jac)
        xt = out[0][-1]
        dxdt = net.res_function(tmax, xt, [0]*nsp, constants)
        if np.max(np.abs(dxdt)) < tol:
            net.updateVariablesFromDynamicVars(xt, tmax)
            if to_ser:
                return net.J
            else:
                return net.J.values
        else:
            tmin, tmax = tmax, tmax*k
            x0 = xt
    raise Exception("Cannot reach steady state for p=%s" % p)


def get_Rs(net, p, Nr, L, to_mat=False, **kwargs_ss):
    """
    """
    if not hasattr(net, 'Ep_code'):
        net.get_Ep_str()
        net.get_Ex_str()
    
    get_s(net, p, **kwargs_ss)  # also sets net.x = s (crucial) 
    
    ns = net.namespace.copy()
    ns.update(net.varvals.to_dict())
    Ep = eval(net.Ep_code, ns)
    Es = eval(net.Ex_code, ns)
    jac = np.dot(np.dot(Nr, Es), L)
    Cs = -np.dot(np.dot(L, np.linalg.inv(jac)), Nr)
    Rs = np.dot(Cs, Ep)
    if to_mat:
        Rs = Matrix(Rs, net.xids, net.pids)
    return Rs


def get_RJ(net, p, Nr, L, to_mat=False, **kwargs_ss):
    if not hasattr(net, 'Ep_code'):
        net.get_Ep_str()
        net.get_Ex_str()
    
    get_s(net, p, **kwargs_ss)  # also sets net.x = s (crucial) 
    
    ns = net.namespace.copy()
    ns.update(net.varvals.to_dict())
    Ep = eval(net.Ep_code, ns)
    Es = eval(net.Ex_code, ns)
    jac = np.dot(np.dot(Nr, Es), L)
    Cs = -np.dot(np.dot(L, np.linalg.inv(jac)), Nr)
    CJ = np.eye(len(net.rxns)) + np.dot(Es, Cs)
    RJ = np.dot(CJ, Ep)
    if to_mat:
        RJ = Matrix(RJ, net.Jids, net.pids)
    return RJ
    

def get_predict(net, expts, **kwargs_ss):
    """
    """
    varids = list(set(butil.flatten(expts['varids'])))
    if set(varids) <= set(net.xids):
        vartype = 's'
    elif set(varids) <= set(net.Jids):
        vartype = 'J'
    else:
        vartype = 'sJ'
        
    net0 = net.copy()
    if not net0.compiled:
        net0.compile()
    nets = [net0.perturb(cond) for cond in expts.conds]
    [net.get_Ep_str() for net in nets]
    [net.get_Ex_str() for net in nets]
    
    L, Nr = net.L.values, net.Nr.values
    
    def f(p):
        y = []
        for net in nets:
            if vartype == 's':
                y_cond = get_s(net, p, to_ser=1, **kwargs_ss)[varids].tolist() 
            if vartype == 'J':
                y_cond = get_J(net, p, to_ser=1, **kwargs_ss)[varids].tolist() 
            if vartype == 'sJ':
                #import ipdb
                #ipdb.set_trace()
                sJ = get_s(net, p, to_ser=1, **kwargs_ss).append(
                    get_J(net, p, to_ser=1, **kwargs_ss))
                y_cond = sJ[varids].tolist()
            y.extend(y_cond)
        return np.array(y)
    
    def Df(p):
        jac = []
        for net in nets:
            if vartype == 's':
                jac_cond = get_Rs(net, p, Nr, L, to_mat=1, **kwargs_ss).loc[varids].dropna()
            if vartype == 'J':
                jac_cond = get_RJ(net, p, Nr, L, to_mat=1, **kwargs_ss).loc[varids].dropna()
            if vartype == 'sJ':
                R = get_Rs(net, p, Nr, L, to_mat=1, **kwargs_ss).append(
                    get_RJ(net, p, Nr, L, to_mat=1, **kwargs_ss))
                jac_cond = R.loc[varids].dropna()
            jac.extend(jac_cond.values.tolist())
        return np.array(jac)

    pred = predict.Predict(f=f, Df=Df, p0=net.p0, pids=net0.pids, 
                           yids=expts.yids, expts=expts, nets=nets)
    
    return pred


def solve_path2(s1, s2):
    """
    Input:
        s1 and s2: strs of ratelaws; variable is 'X'
        solve_for: 'X' or 'J'
        #idx_root: 0 or 1, since there can be two roots (if v1 - v2 == 0 
        #    is a quadratic equation)
    
    Output:
        A tuple of  (X string, J string)
    """
    if 'inf' in s1 and 'inf' in s2:
        raise ValueError
    elif 'inf' in s1:
        Xstr = str(sympy.solve(s1.replace('inf','1'), 'X', simplify=True)[0])
        Jstr = exprmanip.simplify_expr(exprmanip.sub_for_var(s2, 'X', Xstr))
    elif 'inf' in s2:
        Xstr = str(sympy.solve(s2.replace('inf','1'), 'X', simplify=True)[0])
        Jstr = exprmanip.simplify_expr(exprmanip.sub_for_var(s1, 'X', Xstr))
    else:
        eqn = '(%s) - (%s)' % (s1, s2)
        roots = sympy.solve(eqn, 'X', simplify=True)
        if len(roots) == 2:
            # choose the positive root (can be the 1st or the 2nd of the two)
            bools = ['+ sqrt' in str(sympy.expand(root)) for root in roots]
            idx = bools.index(True)
            Xstr = str(roots[idx])
        else:
            Xstr = str(roots[0])
        Jstr = exprmanip.simplify_expr(exprmanip.sub_for_var(s1, 'X', Xstr))
    """
    try:
        xsol = str(roots[1])
        varids = list(exprmanip.extract_vars(xsol))
        tests = [] 
        for i in range(10):
            subs = dict(zip(varids, np.random.lognormal(size=len(varids))))
            subs['sqrt'] = np.sqrt
            tests.append(eval(xsol, subs) > 0)
        if not all(tests):
            raise ValueError
    except (IndexError, ValueError):
        xsol = str(roots[0])
    """
    
    return Xstr, Jstr
    

