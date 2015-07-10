"""

"""

from __future__ import division
import numpy as np

from util.butil import Series

import predict, sampling, residual
from matrix import Matrix

reload(predict)
reload(sampling)
reload(residual)



class Model(object):
    """
    p: two rates r1 and r2
    """
    def __init__(self, p0):
        self.p0 = p0
        self.pids = ['r1','r2']
        
    def __call__(self, t, p=None):
        if p is None:
            p = self.p0
        return np.exp(-p[0]*t) + np.exp(-p[1]*t)
    
    def __repr__(self):
        return 'f(p)=exp(-p[0]*t)+exp(-p[1]*t)'
    
    def get_predict(self, expts, **kwargs_prior):
        def f(p=None):
            y = [self(t, p) for t in expts]
            return Series(y, index=expts.yids)
        
        def Df(p=None):
            if p is None:
                p = self.p0
            jac = []
            for t in expts:
                jac.append([-np.exp(-p[0]*t)*t, -np.exp(-p[1]*t)*t])
            return Matrix(jac, rowvarids=expts.yids, colvarids=self.pids)
        
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
        return ['t=%.1f'%t for t in self]


if __name__ == '__main__':
    p0 = Series([1, 2], index=['r1', 'r2'])
    mod = Model(p0)
    expts = Experiments([1,2,3])
    pred = mod.get_predict(expts, prior='jeff')
    dat = pred.make_dat(scheme='sigma', sigma=1)
    res = residual.Residual(pred, dat)
        
    plot_image = 0
    sample = 0
    plot_contour = 0
    fit = 1
    
    if plot_image:
        pred.plot_image(method='grid', decade=6, npt=100,
                        color='b', alpha=0.2, shade=False, edgecolor='none',  
                        xyzlabels=expts.yids, xyzlims=[(0,2)]*3,
                        filepath='')
    
    if sample:
        ens = sampling.sampling(pred, nstep=2000, in_logp=True, seed=100, 
                                scheme_sampling='eye', stepscale=0.1)
                    
        yens = ens.get_yens(pred)
        yens.scatter_3d(pts=[pred()], xyzlims=[(0,2)]*3)
        
    if plot_contour:
        res.plot_cost_contour(theta1s=np.linspace(0,5,501), theta2s=np.linspace(0,5,501))

    if fit:
        #out = res.fit3(p0=[1,20], ret_steps=True, ret_full=True, tol=1e-9)
        #out = res.fit(p0=[1,20], disp=False, retall=True, full_output=True)
        out = res.fit([1,20])
        #print out