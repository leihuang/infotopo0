"""
"""

from __future__ import division

import numpy as np
import scipy as sp
import sympy 

from SloppyCell import daskr
from SloppyCell import ExprManip as exprmanip
from SloppyCell.ReactionNetworks import Dynamics

from util import butil
from util.matrix import Matrix

from infotopo import predict
reload(predict)


TMIN = 1e3  # integration
TMAX = 1e9  # integration
TOL_SS = 1e-10  # integration
K = 1000  # integration
NTRIAL = 3  # rootfinding
METHOD = 'integration'


def get_s_integration(net, p=None, Tmin=None, Tmax=None, k=None, 
                      tol=None, to_ser=False):
    """
    """
    if p is not None:
        net.update_optimizable_vars(p)
        
    if Tmin is None:
        Tmin = TMIN
    if Tmax is None:
        Tmax = TMAX
    if k is None:
        k = K
    if tol is None:
        tol = TOL_SS
        
    nsp = len(net.dynamicVars)
    tmin, tmax = 0, Tmin
    x0 = net.x0.copy()
    constants = net.constantVarValues
    
    while tmax <= Tmax:
        # yp0 helps integration stability
        yp0 = Dynamics.find_ics(net.x0, net.x0, tmin, net._dynamic_var_algebraic, 
                                [1e-6]*nsp, [1e-6]*nsp, constants, net)[1]
        # using daskr to save computational overhead
        out = daskr.daeint(res=net.res_function, t=[tmin, tmax], y0=x0, 
                           yp0=yp0, atol=[1e-6]*nsp, rtol=[1e-6]*nsp, 
                           intermediate_output=False, rpar=constants,
                           max_steps=100000.0, max_timepoints=100000.0, 
                           jac=net.ddaskr_jac)
        xt = out[0][-1]
        dxdt = net.res_function(tmax, xt, [0]*nsp, constants)
        if np.max(np.abs(dxdt)) < tol:
            net.updateVariablesFromDynamicVars(xt, tmax)
            net.t = tmax
            if to_ser:
                return butil.Series(xt, net.xids)
            else:
                return xt
        else:
            tmin, tmax = tmax, tmax*k
            x0 = xt
    raise Exception("Cannot reach steady state for p=%s" % p)



def get_s_rootfinding(net, p=None, x0=None, tol=None, ntrial=3, seeds=None, 
                      test_stability=True, 
                      full_output=False, to_ser=False, **kwargs_fsolve):
    """Return the steady state values of dynamic variables found by 
    the root-finding method, which may or may not represent the true
    steady state. 
    
    It may be time-consuming the first time it is called, as attributes
    like P are calculated and cached.
    
    Input:
        p:
        x0: initial guess in rootfinding; by default the current x of net
        to_ser:
        kwargs_fsolve: 
        
    Documentation of scipy.optimize.fsolve:
    """
    if p is not None:
        net.update_optimizable_vars(p)
        
    if tol is None:
        tol = TOL_SS
    if ntrial is None:
        ntrial  = NTRIAL
    
    x = np.array([var.value for var in net.dynamicVars])
    if np.max(np.abs(net.get_dxdt(x=x))) < tol:  # steady-state
        if full_output:
            return x, {}, 1, ""
        else:
            return x    
    
    if not hasattr(net, 'pool_mul_mat'):
        print "net has no P: calculating P."
        P = net.P
    P = net.P.values
    npool = P.shape[0]
    if npool > 0:
        poolsizes = np.dot(P, [var.initialValue for var in net.dynamicVars])
    
    # Indices of independent dynamic variables
    ixidxs = [net.xids.index(xid) for xid in net.ixids]  
    
    def _f(x):
        """This is a function to be passed to scipy.optimization.fsolve, 
        which takes values of all dynamic variable (x) 
        as input and outputs the time-derivatives of independent 
        dynamic variables (dxi/dt) and the differences between
        the current pool sizes (as determined by the argument dynvarvals)
        and the correct pool sizes.
        """
        dxdt = net.get_dxdt(x=x)
        if npool > 0:
            dxidt = dxdt[ixidxs]
            diffs = np.dot(P, x) - poolsizes
            return np.concatenate((dxidt, diffs))
        else:
            return dxdt
        
    def _Df(x):
        """
        """
        dfidx = net.dres_dc_function(0, x, [0]*len(x), net.constantVarValues)[ixidxs]
        if npool > 0:
            return np.concatenate((dfidx, P))
        else:
            return dfidx
            
    if x0 is None:
        x0 = net.x0
    if tol is None:
        tol = 1.49012e-08  # scipy default
        
    out = sp.optimize.fsolve(_f, x0, fprime=_Df, xtol=tol, full_output=1, 
                             **kwargs_fsolve)
    count = 1
    while out[2] != 1 and count <= ntrial:
        count += 1
        if seeds is None:
            seed = count
        else:
            seed = seeds.pop(0)
        out = sp.optimize.fsolve(_f, butil.Series(x0).randomize(seed=seed), 
                                 fprime=_Df, xtol=tol, full_output=1, 
                                 **kwargs_fsolve)
    
    s = out[0]
    
    net.update(x=s, t_x=np.inf)
    
    if test_stability:
        jac = net.get_jac_mat()
        if any(np.linalg.eigvals(jac) > 0):
            print "Warning: the solution is an unstable steady state."

    if to_ser:
        s = Series(s, net.xids)
        out[0] = s
    
    if full_output:
        return out
    else:
        return s

get_s_rootfinding.__doc__ += sp.optimize.fsolve.__doc__




def get_s(net, p=None, method=None, Tmin=None, Tmax=None, k=None, 
          x0=None, tol=None, to_ser=False, **kwargs_rootfinding):
    """
    Input:
        method:
        Tmin, Tmax, k:
        x0: 
        
    """
    if p is not None:
        net.update_optimizable_vars(p)
        
    if method is None:
        method = METHOD
    if Tmin is None:
        Tmin = TMIN
    
    if method == 'integration':
        return get_s_integration(net, Tmin=Tmin, Tmax=Tmax, k=k, 
                                 tol=tol, to_ser=to_ser)
    elif method == 'rootfinding':
        return get_s_rootfinding(net, x0=x0, tol=tol, to_ser=to_ser,
                                 **kwargs_rootfinding)
    elif method == 'mixed':
        nsp = len(net.dynamicVars)
        constants = net.constantVarValues
        yp0 = Dynamics.find_ics(net.x0, net.x0, 0, net._dynamic_var_algebraic, 
                                [1e-6]*nsp, [1e-6]*nsp, constants, net)[1]
        out = daskr.daeint(res=net.res_function, t=[0, Tmin], y0=net.x0.copy(), 
                           yp0=yp0, atol=[1e-6]*nsp, rtol=[1e-6]*nsp, 
                           intermediate_output=False, rpar=constants,
                           max_steps=100000.0, max_timepoints=100000.0, 
                           jac=net.ddaskr_jac)
        xt = out[0][-1]
        return get_s_rootfinding(net, x0=xt, tol=tol, to_ser=to_ser,
                                 **kwargs_rootfinding)
    else:
        raise ValueError("Unrecognized value for method: %s"%method)


def set_ss(net, **kwargs):
    get_s(net, **kwargs)


def get_J(net, p=None, to_ser=False, **kwargs):
    """
    if p is not None:
        net.update_optimizable_vars(p)
        
    nsp = len(net.dynamicVars)
    tmin, tmax = 0, Tmin
    x0 = net.x0.copy()
    constants = net.constantVarValues
    
    while tmax <= Tmax:
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
    """
    set_ss(net, p=p, **kwargs)
    return net.get_v(to_ser=to_ser)
    
    

def get_Rs(net, Nr, L, p=None, to_mat=False, **kwargs_ss):
    """
    """
    if p is not None:
        net.update_optimizable_vars(p)
         
    if not hasattr(net, 'Ep_code'):
        net.get_Ep_str()
        net.get_Ex_str()
    
    set_ss(net, **kwargs_ss)  # also sets net.x = s (crucial) 
    
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


def get_RJ(net, Nr, L, p=None, to_mat=False, **kwargs_ss):
    if p is not None:
        net.update_optimizable_vars(p)
    
    if not hasattr(net, 'Ep_code'):
        net.get_Ep_str()
        net.get_Ex_str()
    
    set_ss(net, **kwargs_ss)  # also sets net.x = s (crucial) 
    
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
        idxs = [idx for idx, xid in enumerate(net.xids) if xid in varids]
    elif set(varids) <= set(net.Jids):
        vartype = 'J'
        idxs = [idx for idx, Jid in enumerate(net.Jids) if Jid in varids]
    else:
        vartype = 'sJ'
        xJids = net.xids + net.Jids
        idxs = [idx for idx, xJid in enumerate(xJids) if xJid in varids]
        
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
            net.update_optimizable_vars(p)
            s = get_s(net, **kwargs_ss)
            if vartype == 's':
                y_cond = s[idxs].tolist() 
            if vartype == 'J':
                y_cond = [net.evaluate_expr(net.reactions[idx].kineticLaw) 
                          for idx in idxs]
            if vartype == 'sJ':
                sJ = np.concatenate((get_s(net, **kwargs_ss),
                                     get_J(net, **kwargs_ss)))
                y_cond = sJ[idxs].tolist()
            y.extend(y_cond)
        return np.array(y)
    
    def Df(p):
        jac = []
        for net in nets:
            net.update_optimizable_vars(p)
            if vartype == 's':
                #jac_cond = get_Rs(net, p, Nr, L, to_mat=1, **kwargs_ss).loc[varids].dropna()  # why dropna?
                jac_cond = get_Rs(net, Nr, L, **kwargs_ss)[idxs]
            if vartype == 'J':
                jac_cond = get_RJ(net, Nr, L, **kwargs_ss)[idxs]
            if vartype == 'sJ':  # to be refined
                R = np.vstack((get_Rs(net, Nr, L, **kwargs_ss),
                               get_RJ(net, Nr, L, **kwargs_ss)))
                jac_cond = R[idxs]
            jac.extend(jac_cond.tolist())
        return np.array(jac)

    pred = predict.Predict(f=f, Df=Df, p0=net.p0, pids=net.pids, 
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
    

