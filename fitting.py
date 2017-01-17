"""
"""


from __future__ import division
from collections import OrderedDict as OD
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize

from SloppyCell import lmopt

from util.matrix import Matrix

from util import butil
Series, DF = butil.Series, butil.DF


class Fit(object):
    """
    """
    def __init__(self, pids, **kwargs):
        if 'cost' in kwargs:
            self.cost_ = kwargs.pop('cost')
            
        if 'p' in kwargs:
            self.p_ = Series(kwargs.pop('p'), pids) 
        
        if 'costs' in kwargs:
            costs = Series(kwargs.pop('costs'))
            costs.index.name = 'step'
        else:
            costs = None
        if 'ps' in kwargs:
            ps = DF(kwargs.pop('ps'), columns=pids)
            ps.index.name = 'step'
        else:
            ps = None 
        if 'lambs' in kwargs:
            lambs = Series(kwargs.pop('lambs'))
            lambs.index.name = 'step'
        else:
            lambs = None 
        
        kwargs['costs'] = costs
        kwargs['ps'] = ps
        kwargs['lambs'] = lambs
        
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    
    @property
    def cost(self):
        if hasattr(self, 'cost_'):
            return self.cost_
        elif hasattr(self, 'costs'):
            return self.costs.iloc[-1]
        else:
            raise AttributeError("fit does not have cost information.")
    
    
    @property
    def p(self):
        if hasattr(self, 'p_'):
            return self.p_
        elif hasattr(self, 'ps'):
            return self.ps.iloc[-1]
        else:
            raise AttributeError("fit does not have p information.")
        
    
    def add_step(self):
        pass
   


def fit_lm_scipy(res, p0=None, in_logp=True, **kwargs):
    """
    """

    if p0 is None:
        p0 = res.p0
    else:
        p0 = Series(p0, res.pids)
        
    if in_logp:
        res = res.get_in_logp()
        p0 = p0.log()
    else:
        res = res
    
    kwargs = butil.get_submapping(kwargs, keys=['full_output', 'col_deriv', 
        'ftol', 'xtol', 'gtol', 'maxfev', 'epsfcn', 'factor', 'diag'])
    kwargs_ls = kwargs.copy()
    kwargs_ls['full_output'] = True

    p, cov, infodict, mesg, ier = leastsq(res, p0, Dfun=res.Dr, 
                                          **kwargs_ls)
    
    if in_logp:
        p = np.exp(p)
        # cov = ... FIXME ***

    r = Series(infodict['fvec'], res.rids)
    cost = _r2cost(r) 
    covmat = Matrix(cov, res.pids, res.pids)
    nfcall = infodict['nfev']
    
    fit = Fit(cost=cost, p=p, pids=res.pids, covmat=covmat, 
              nfcall=nfcall, r=r, message=mesg, ier=ier) 
    
    return fit
fit_lm_scipy.__doc__ += leastsq.__doc__
                 
    
    
def fit_lm_sloppycell(res, p0=None, in_logp=True, **kwargs):
    """Get the best fit using Levenberg-Marquardt algorithm.
    
    Input:
        p0: initial guess
        in_logp: optimizing in log parameters
        *args and **kwargs: additional parameters to be passed to 
            SloppyCell.lm_opt.fmin_lm, whose docstring is appended below: 
    
    Output:
        out: a Series
    """
    if p0 is None:
        p0 = res.p0
    else:
        p0 = Series(p0, res.pids)
    
    if in_logp:
        res = res.get_in_logp()
        p0 = p0.log()
    else:
        res = res
    
    keys = ['args', 'avegtol', 'epsilon', 'maxiter', 'full_output', 'disp', 
            'retall', 'lambdainit', 'jinit', 'trustradius']
    kwargs_lm = butil.get_submapping(kwargs, f_key=lambda k: k in keys)
    kwargs_lm['full_output'] = True
    kwargs_lm['retall'] = True
    p, cost, nfcall, nDfcall, convergence, lamb, Df, ps =\
        lmopt.fmin_lm(f=res.r, x0=p0, fprime=res.Dr, **kwargs_lm)
    
    if in_logp:
        p = np.exp(p)
        ps = np.exp(ps)

    fit = Fit(p=p, ps=ps, cost=cost, pids=res.pids,
              nfcall=nfcall, nDfcall=nDfcall, convergence=convergence,
              lamb=lamb, Df=Df)
            
    return fit
fit_lm_sloppycell.__doc__ += lmopt.fmin_lm.__doc__    


def fit_lm_custom(res, p0=None, in_logp=True,
                  maxnstep=1000, disp=False, #ret_full=False, ret_steps=False, 
                  lamb0=1e-3, tol=1e-6, k_up=10, k_down=10, ndone=5, **kwargs):
    """
    
    Input:
        k_up and k_down: parameters used in tuning lamb at each step;
            in the traditional scheme, typically 
                k_up = k_down = 10;
            in the delayed gratification scheme, typically 
                k_up = 2, k_down = 10 (see, [1])
    
    grad C = Jt * r
    J = U * S * Vt
     ______     ______  
    |      |   |      |  ______   ______
    |      |   |      | |      | |      |
    |   J  | = |   U  | |   S  | |  Vt  |
    |      |   |      | |______| |______|
    |______|   |______|
    
    V.T * V = V * V.T = I
    U.T * U = I =/= U * U.t
    J.T * J = (V * S * U.T) * (U * S * V.T) = V * S^2 * V.T
    
     ______     ____________   ______
    |      |   |            | |      |  ______
    |      |   |            | |      | |      |
    |      | = |            | |      | |      |
    |      |   |            | |      | |______|
    |______|   |____________| |______|
    
    
    
    Gradient-descent step: 
        delta p = - grad C = - J.T * r  
        
    Gauss-Newton step:
        delta p = - (J.T * J).inv * grad C = - (J.T * J).inv * J.T * r

    Levenberg step:
        delta p = - (J.T * J + lamb * I).inv * grad C
                = - (V * (S^2 + lamb * I) * V.T).I * J.T * r
                = - (V * (S^2 + lamb * I).inv * V.T) * V * S * U.T * r
                = - V * (S^2 + lamb * I).inv * S * U.T * r
    
    References:
    [1] Transtrum
    [2] Numerical Recipes
    """
    if p0 is None:
        p0 = res.p0
    else:
        p0 = Series(p0, res.pids)

    if in_logp:
        res = res.get_in_logp()
        p0 = p0.log()
    else:
        res = res
    
    if maxnstep is None :
        maxnstep = len(res.pids) * 100

    nstep = 0
    nfcall = 0
    nDfcall = 0  

    p = p0
    lamb = lamb0
    done = 0
    accept = True
    convergence = False
    
    r = res(p0)
    cost = _r2cost(r)        
    nfcall += 1

    ps = [p0]
    deltaps = []
    costs = [cost]
    lambs = [lamb]
    
    while not convergence and nstep < maxnstep:
        
        if accept:
            ## FIXME ***
            jac = res.Dr(p, to_mat=True)
            U, S, Vt = jac.svd(to_mat=True)
            nDfcall += 1            
        
        deltap = - Vt.T * (S**2 + lamb * Matrix.eye(res.pids)).I * S * U.T * r
        deltap = deltap[0]  # convert 1-d DF to series
        p2 = p + deltap
        nstep += 1
        
        if disp:
            #print nstep
            print deltap.exp()[:10]
            print lamb
            #print p2
            #from util import butil
            #butil.set_global(p=p, deltap=deltap, p2=p2, nstep=nstep)
            
        r2 = res(p2)
        cost2 = _r2cost(r2)
        nfcall += 1
        
        if np.abs(cost - cost2) < max(tol, cost * tol):
            done += 1
            
        if cost2 < cost:
            accept = True
            lamb /= k_down
            p = p2
            r = r2
            cost = cost2    
        else:
            accept = False
            lamb *= k_up
        
        ps.append(p)
        deltaps.append(deltap)
        costs.append(cost)
        lambs.append(lamb)
            
        if done == ndone:
            convergence = True
            # lamb = 0
            
    if in_logp:
        ps = np.exp(ps)
        pids = map(lambda pid: pid.lstrip('log_'), res.pids)
    else:
        pids = res.pids
            
    ## need to calculate cov  FIXME ***
    
    fit = Fit(costs=costs, ps=ps, pids=pids, lambs=lambs, 
              nfcall=nfcall, nDfcall=nDfcall, convergence=convergence,
              nstep=nstep)
    return fit


def fit_minimize_scipy(res, p0=None, in_logp=True, method='Nelder-Mead', **kwargs):
    """
    """
    if p0 is None:
        p0 = res.p0
    else:
        p0 = Series(p0, res.pids)
        
    if in_logp:
        res = res.get_in_logp()
        p0 = p0.log()
    else:
        res = res
        
    _grad = lambda p: np.dot(res.Dr(p).T, res(p))
    out0 = minimize(res.cost, p0, method=method, jac=_grad, **kwargs)
    
    out = Series(OD([('p', Series(out0.x, res.pids)),
                     ('cost', out0.fun),
                     ('message', out0.message),
                     ('nfcall', out0.nfev)]))
    if hasattr(out0, 'nit'):
        out.nstep = out0.nit
    if hasattr(out0, 'njev'):
        out.nDfev = out0.njev 
    
    if in_logp:
        out.p = out.p.exp()
        
    return out
fit_minimize_scipy.__doc__ += minimize.__doc__


def fit(res, p0=None, np0=1, sigma=1, 
        methods=('lm_scipy', 'lm_sloppycell', 'lm_custom'),
        cost_cutoff=0, ret_all=False,
        # 'minimize_scipy'),  # FIXME **: call signature of minimize not fixed yet 
        **kwargs):
    """
    
    Input:
        np0: number of trial p0's
        sigma: perturbation strength
        methods: 
        cost_cutoff:
        
        
    """
    if p0 is None:
        p0 = res.p0
        
    def _fit(method, p0, idx):
        if idx == 0:
            p0 = p0
        else:
            p0 = Series(p0).perturb(sigma=sigma)
            
        try:
            fit = getattr(res, 'fit_'+method)(p0, **kwargs)
        except Exception, e:
            logging.error(e)
            fit = Series({'cost': np.inf, 
                          'p': Series([np.nan]*len(res.pids), res.pids)})
        fit.method = method
        fit.p0 = p0
        return fit

    fits = []
    for method in methods:
        for idx in range(np0):
            fit = _fit(method, p0, idx)
            if fit.cost < cost_cutoff:
                return fit
            else:
                fits.append(fit)
    
    if ret_all:
        return fits
    else:
        return min(fits, key=lambda fit: fit.cost)
        
    """
    # expected cost is ~ len(self.rids)/2
    while outs[-1].cost > 5 * len(self.rids) / 2 and methods:
        

    
    for idx in range(np0 - 1):
        outs.append(_fit(method, p0.perturb(seed=idx, sigma=sigma)))
           
    if not try_methods:
        return min(outs, key=lambda out: out.cost)
    else:
        methods = list(methods)
        methods.remove(method)
        outs = [out]
        
        while out.cost > 10 * len(self.rids) / 2 and methods:  
            out = _fit(methods.pop(0))
            outs.append(out)
        return min(outs, key=lambda out: out.cost) 
    """

def _r2cost(r):
    """
    """
    return np.linalg.norm(r)**2 / 2