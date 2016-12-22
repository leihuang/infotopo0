"""

"""

from __future__ import division
import numpy as np

from util.butil import Series
from util.matrix import Matrix

import predict, sampling, residual
reload(predict)
reload(sampling)
reload(residual)


class Model(object):
    """
    p: two rates theta1 and theta2
    """
    def __init__(self, p0):
        pids = ['theta1', 'theta2']
        self.p0 = Series(p0, pids) 
        self.pids = pids
        
    def __call__(self, t, p=None):
        if p is None:
            p = self.p0
        return p[0] * t + p[1] * t**2
    
    def __repr__(self):
        return 'f(p) = p[0]*t + p[1]*t^2'
    
    def get_predict(self, expts, **kwargs_prior):
        def f(p):
            return np.array([self(t, p) for t in expts])
        
        def Df(p):
            jac = []
            for t in expts:
                jac.append([t, t**2])
            return np.array(jac)
        
        pred = predict.Predict(f=f, Df=Df, p0=self.p0, pids=self.pids, 
                               yids=expts.yids)
        
        if kwargs_prior:
            pred.set_prior(**kwargs_prior)
        return pred
        
            

class Experiments(list):
    """
    """
    @property
    def yids(self):
        return ['f(t=%.1f)'%t for t in self]


if __name__ == '__main__':
    p0 = [0.1, 0.5]
    
    mod = Model(p0=p0)
    expts = Experiments([1,2,3])
    pred = mod.get_predict(expts, prior='jeff')
    dat = pred.get_dat(scheme='sigma', sigma=1)
    res = residual.Residual(pred, dat)
    
    pred_log10p = pred.get_in_log10p()
