"""
This module is currently not used. 


***********************************
A unified interface...

Simplify and improve SloppyCell.ReactionNetworks.Dynamics

- integrate:
    * return ndarray-based objects
    * time
    * efficiently resume integration (right now it starts from scratch rather from where it was left off)
    * move trajectory.py here
    * no events
    * multiple integrators (compare performance)


"""

from __future__ import division

import numpy as np
from scipy.integrate import ode as spODE

from SloppyCell import daskr

from util.butil import Series
from infotopo.models.rxnnet.examples.mmr2 import pids_R1



class ODE(object):
    """
    x' = f(x, p)
    
    Integrator: 
        'vode'
        'lsoda'
        'dopri5'
        'dp853'
        'cvode'
    """
    TMIN = 1e2
    TMAX = 1e8
    TOL_SS = 1e-9
    K = 100

    def __init__(self, f, x0, jac=None, p=None, pids=None, xids=None,
                 integrator='lsoda', atol=1e-6, rtol=1e-6, 
                 callback=None, **kwargs_integrator):
        
        """
        # The common counter decorator, to get the number of times
        # http://stackoverflow.com/questions/21716940/
        # is-there-a-way-to-track-the-number-of-times-a-function-is-called
        def counted(f):
            def wrapped(t, x):
                wrapped.ncall += 1
                return f(t, x)
            wrapped.ncall = 0
            return wrapped
        """

        ode = spODE(f, jac=jac)
        ode.set_initial_value(x0)
        ode.set_integrator(integrator, atol=atol, rtol=rtol, **kwargs_integrator)
        if p is not None:
            ode = ode.set_f_params(p)
        
        
        self.ode = ode
        self.callback = callback
        
        
        self.p = p
        self.pids = pids
        self.xids = xids
        self.x0 = x0
        
    
    
    def __getattr__(self, attr):
        #if attr in ['integrate', 'y', 't']:
        #    return getattr(self.ode, attr)
        #else:
        #    return getattr(self, attr)
        try:
            return getattr(self.ode, attr)
        except AttributeError:
            return getattr(self, attr)
       
    
    def update(self, **kwargs):
        """
        Input: 
            kwargs: p, integrator, atol, rtol
        """
        if 'p' in kwargs:
            p = kwargs.pop('p')
            self.set_f_params(p)
            self.p = p
        if 'x0' in kwargs:
            x0 = kwargs.pop('x0')
            self.ode = self.ode.set_initial_value(x0)
            self.x0 = x0
        if 'integrator' in kwargs:
            integrator = kwargs.pop('integrator')
            self.ode = self.ode.set_integrator(integrator, **kwargs)
            

    def get_x(self, t, to_ser=False):
        if t < self.t:
            self.update(x0=self.x0)  # set self.t = 0
        x = self.integrate(t)
        if to_ser:
            x = Series(x, self.xids)
        return x
    
    
    def get_dxdt(self, x, t):
        return self.f(t, x)
    
    
    def is_ss(self, x, t, tol=None):
        if tol is None:
            tol = type(self).TOL_SS
        return np.max(np.abs(self.f(t, x))) < tol
        

    def get_s(self, tol=None, Tmin=None, Tmax=None, k=None, to_ser=False):
        if Tmin is None:
            Tmin = type(self).TMIN
        if Tmax is None:
            Tmax = type(self).TMAX
        if k is None:
            k = type(self).K
        if tol is None:
            tol = type(self).TOL_SS
        
        #import ipdb
        #ipdb.set_trace()
        
        t = Tmin
        while t <= Tmax:
            x = self.get_x(t, to_ser=to_ser) 
            if self.is_ss(x, t, tol=tol):
                return x
            else:
                t *= k
        raise Exception("Unable to reach steady state")



class DAE(object):
    pass

