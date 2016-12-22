"""
"""

from __future__ import division
import itertools

import numpy as np

from SloppyCell import daskr
from SloppyCell.ReactionNetworks import Dynamics

from util import butil

from infotopo import predict
reload(predict)


TOL = 1e-12

def get_xt(net, times, varids, p=None, tol=None, to_DF=False, use_daeint=False):
    if tol is None:
        tol = TOL
    
    if times[0] != 0:
        times = [0] + list(times)
        prepend_zero = True
    else:
        prepend_zero = False
        
    if p is not None:
        net.update_optimizable_vars(p)
   
    x0 = net.x0.copy()  # integration changes x0, hence the copying
    
    if not hasattr(net, 'res_function'):
        net.compile()
        
    if use_daeint:
        assert varids == net.xids
        out = daskr.daeint(res=net.res_function, t=times, y0=x0, yp0=[0]*net.xdim, 
                           atol=[tol]*net.xdim, rtol=[tol]*net.xdim, 
                           intermediate_output=False, rpar=net.constantVarValues)
        xt = out[0]
        xt[0] = net.x0  # somehow daskr.daeint messes up the first timepoint
    else:
        # Switch to Dynamics.integrate because of presumably easier indexing 
        # of requested varids.
        # Be careful of Dynamics.integrate's handling of times, esp. when 
        # the initial time is not zero.    
        traj = Dynamics.integrate(net, times, params=p, fill_traj=False,
                                  rtol=[tol]*net.xdim, atol=[tol]*net.xdim)
        xt = traj.copy_subset(varids).values
        
    if prepend_zero:
        xt = xt[1:]
        times = times[1:]
    
    if to_DF:
        return butil.DF(xt, index=times, columns=varids)
    else:
        return xt


def get_dxtdp(net, times, varids, p=None, tol=None, to_DF=False):
    """
    Requires the first timepoint to be 0 (because of the assumption of SloppyCell).  
    """
    if tol is None:
        tol = TOL

    if times[0] != 0:
        times = [0] + list(times)
        prepend_zero = True
    else:
        prepend_zero = False
        
    straj0 = Dynamics.integrate_sensitivity(net, times, params=p, rtol=tol)
    straj = straj0.copy_subset(itertools.product(varids, net.pids))

    if prepend_zero:
        dat = np.vstack(np.hsplit(straj.values[1:], len(varids)))
        times = times[1:]
    else:
        dat = np.vstack(np.hsplit(straj.values, len(varids)))
        
    if to_DF:
        return butil.DF(dat, index=itertools.product(varids, times), 
                        columns=net.pids)
    else:
        return dat


def get_predict(net, expts, **kwargs):
    """
    """
    assert expts.conds == [()], 'condition is not just wildtype.'
    #assert butil.flatten(expts['varids']) == net.xids  # the restriction can be relaxed later
    varids = butil.flatten(expts['varids'])
    
    #import ipdb
    #ipdb.set_trace()      
    
    def f(p):
        net.update(t=0)
        return get_xt(net, times=butil.flatten(expts['times']), p=p, varids=varids, **kwargs).T.flatten()
        
    
    def Df(p):
        net.update(t=0)
        return get_dxtdp(net, times=butil.flatten(expts['times']), p=p, varids=varids, **kwargs)

    pred = predict.Predict(f=f, Df=Df, p0=net.p0, pids=net.pids, 
                           yids=expts.yids, expts=expts)
    
    return pred

