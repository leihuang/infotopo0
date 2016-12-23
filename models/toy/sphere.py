"""
Import pred and do analysis elsewhere...

Spherical parametrization of a sphere
Note that the metric is singular when polar angle phi=0.

"""

import numpy as np
from scipy.integrate import odeint

from util.butil import Series

from infotopo import predict, sampling
reload(predict)
reload(sampling)


class Model(object):
        
    def __init__(self, p0, R=1):
        self.pids = ['phi','theta']
        self.p0 = Series(p0, self.pids)
        self.R = R

    def get_predict(self):
        R = self.R
        
        def _f(p):
            y = [R*np.sin(p[0])*np.cos(p[1]), 
                 R*np.sin(p[0])*np.sin(p[1]),
                 R*np.cos(p[0])]
            return np.array(y)
        
        def _Df(p):
            jac = [[R*np.cos(p[0])*np.cos(p[1]), -R*np.sin(p[0])*np.sin(p[1])], 
                   [R*np.cos(p[0])*np.sin(p[1]), R*np.sin(p[0])*np.cos(p[1])],
                   [-R*np.sin(p[0]), 0]]
            return np.array(jac)
        
        pred = predict.Predict(f=_f, p0=self.p0, name='sphere', pids=self.pids,
                               yids=['x','y','z'], Df=_Df)
        return pred

mod = Model(p0=[0,0], R=1)
pred = mod.get_predict()

"""
def func(y, t):
    #y = (p1, p2, v1, v2)
    #return [y[2], y[3], np.cos(y[0])*np.sin(y[0])*y[3], -2*np.cos(y[0])/np.sin(y[0])*y[2]*y[3]]
    return [y[2], y[3], np.cos(y[0])*np.sin(y[0])*y[3]**2, -2*np.cos(y[0])/np.sin(y[0])*y[2]*y[3]]

y0 = [1, 0, np.sqrt(2)/2, np.sqrt(2)/2]
ts = np.arange(0, 6, 0.1)
ys = odeint(func, y0, ts)
ens = sampling.Ensemble(ys, columns=pred.pids+['v1', 'v2'])
pens = ens[pred.pids]

yens = pens.apply(pred, axis=1)

yens.scatter3d()

"""