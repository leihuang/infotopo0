"""
Import pred and do analysis elsewhere...


"""

from collections import OrderedDict as OD

import numpy as np
import pandas as pd

import predict

#import infotopo.prediction as p
#import infotopo.geometry as g
#reload(p)
#reload(g)

import Geodesic_Code2 as g
reload(g)



def f((phi, theta)):
    """
    Inputs:
        phi: (0, pi)
        theta: (0, 2*pi)
    """
    return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


def Df((phi, theta)):
    """
    """
    return np.array([[np.cos(phi)*np.cos(theta), -np.sin(phi)*np.sin(theta)], 
                     [np.cos(phi)*np.sin(theta), np.sin(phi)*np.cos(theta)],
                     [-np.sin(phi), 0]])

pred = predict.Predict(f, Df, pids, dids)

#domain = pd.Series(OD([('phi', (0,np.pi)), ('theta', (0, 2*np.pi))])) 
#pred = p.Prediction(f, Df=Df, domain=domain)


#gd = g.Geodesic(r=f, j=Df, )
#pred.plot(n=200)


geqn = g.Geodesic(r=f, j=Df, M=3, N=2, x=(np.pi/4,np.pi/4), v=np.array([1,1]), lam=0.0001)

geqn.integrate(tmax=50)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_aspect("equal")
ax.plot(*geqn.rs.T, color='b', alpha=0.2)
plt.show()
plt.close()
