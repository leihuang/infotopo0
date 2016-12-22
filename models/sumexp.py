"""

"""

from __future__ import division
import numpy as np

from util.butil import Series
from util.matrix import Matrix

from infotopo import predict, sampling, residual
reload(predict)
reload(sampling)
reload(residual)


class Model(object):
    """
    p: a vector of rates r's
    """
    def __init__(self, p0):
        pids = ['r%d'%i for i in range(1, len(p0)+1)]
        self.p0 = Series(p0, pids) 
        self.pids = pids
        
    def __call__(self, t, p=None):
        if p is None:
            p = self.p0
        return np.sum([np.exp(-r*t) for r in p])
    
    def __repr__(self):
        return 'f(p)=sum_over_i exp(-r_i*t)'
    
    def get_predict(self, expts, **kwargs_prior):
        def _f(p):
            return np.array([self(t, p) for t in expts])
        
        def _Df(p):
            jac = []
            for t in expts:
                jac.append([-np.exp(-r*t)*t for r in p])
            return np.array(jac)
        
        pred = predict.Predict(f=_f, Df=_Df, p0=self.p0, 
                               pids=self.pids, yids=expts.yids,
                               name='sumexp%d' % len(self.p0))
        
        if kwargs_prior:
            pred.set_prior(**kwargs_prior)
        return pred
        
            

class Experiments(list):
    """
    """
    @property
    def yids(self):
        return ['f(t=%.1f)'%t for t in self]


exp2 = Model([1,2])
exp3 = Model([1,2,3])
expts2 = Experiments([1,2])
expts3 = Experiments([1,2,3])

pred2 = exp2.get_predict(expts3)
pred3 = exp3.get_predict(expts3)

if __name__ == '__main__':
    
    p0 = [1,0.1]
    
    mod = Model(p0=p0)
    expts = Experiments([1,2,3])
    pred = mod.get_predict(expts, prior='jeff')
    dat = pred.get_dat(scheme='sigma', sigma=1)
    res = residual.Residual(pred, dat)
    
    pred_log10p = pred.get_in_log10p()

    gds = pred_log10p.get_geodesic(const_speed='p')
    gds.integrate(tmax=0.5, dt=0.1)
    
    a
    pred.plot_image(p=[1,1], ndecade=6, npt=50,
                    pts=gds.ys, cs=gds.ts, alpha=0.2, linewidth=0.2,  
                    xyzlims=[[0,2]]*3)
                    #pts=[pred(p0), pred([0.5,0.5]), pred([0.1,0.1])], cs=[1,2,3], 
    a
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