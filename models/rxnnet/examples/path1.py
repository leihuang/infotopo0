"""
The simplest pathway:

    R1    R2
X1 <-> S <-> X2

"""

from __future__ import division
import numpy as np
#from matplotlib.mlab import PCA

from infotopo import geodesic, predict, residual
reload(geodesic)
reload(predict)
reload(residual)

from infotopo.model.rxnnet import model
reload(model) 



net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 2, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='X1<->S', ratelaw='k1*(X1-S)', p={'k1':1})
net.add_reaction('R2', stoich_or_eqn='S<->X2', ratelaw='k2*(S-X2)', p={'k2':2})
net.add_ratevars()

#traj = net.get_traj((0,2))
#traj.plot(filepath='tmp.pdf')


#f = lambda (k1, k2): np.array([(k1*2+k2)/(k1+k2), k1*k2/(k1+k2), k1*k2/(k1+k2)])
#pred = predict.Predict(f=f, p0=[1,2], pids=['k1','k2'], dids=['S','v1','v2'])


p = [k1,k2] = [1,2]


r = 2
f = lambda (k1,k2): np.array([1/(1/k1+1/k2), 1/(1/(r*k1)+1/k2), 1/(1/k1+1/(r*k2))])
Df = lambda (k1,k2): np.array([[(1/k1/(1/k1+1/k2))**2, (1/k2/(1/k1+1/k2))**2],
                               [(1/(r*k1)/(1/(r*k1)+1/k2))**2/2, (1/k2/(1/(r*k1)+1/k2))**2],
                               [(1/k1/(1/k1+1/(r*k2)))**2, (1/(r*k2)/(1/k1+1/(r*k2)))**2/2]])
pred = predict.Predict(f, Df=Df, p0=p, pids=['k1','k2'], dids=['J1','J2','J3'])
pred_logp = pred.get_in_logp()

geqn = geodesic.Geodesic(pred=pred_logp, x=np.log(p), lam=1e-9, atol=1e-9, rtol=1e-9, callback=None)
#geqn = geodesic.Geodesic(pred=pred, x=p, lam=1e-9, atol=1e-9, rtol=1e-9, callback=None)
geqn.integrate(tmax=1e3)

#pred.plot_sloppyvs([geqn.xs[0],geqn.xs[-1]], plabels=['start','end'], filepath='sloppyv_concn.pdf')
#pred.plot_spectrums([geqn.xs[0],geqn.xs[-1]], plabels=['start','end'], filepath='spectrums_concn.pdf')
geqn.plot_traj(pid1='log_k1', pid2='log_k2', log10=False, color_vol=True, filepath='ptraj_concn.pdf')

a

"""
f2 = lambda (k,): np.array([k/2, r*k/2, k/2])
Df2 = lambda (k,): np.array([[1/2],
                             [r/2],
                             [1/2]])
pred2 = predict.Predict(f2, Df=Df2, p0=[1], pids=['k'], dids=['J1','J2','J3']) 
""" 

#res = residual.Residual(pred=pred2, dat=pred.make_data(p, cv=0.2))
#print fit.get_best_fit(disp=False, avegtol=1e-9)


def C(k):
    return (k1*k2/(k1+k2)-k/2)**2 + (r*k1*k2/(r*k1+k2)-r*k/2)**2 + (r*k1*k2/(k1+r*k2)-k/2)**2

import scipy as sp
print sp.optimize.fmin(C, [1], xtol=1e-9, ftol=1e-9, disp=False)

print sp.stats.hmean(p)


"""
r = 3
f = lambda (k1,k2): np.array([k1/(k1+k2), r*k1/(r*k1+k2), k1/(k1+r*k2)])
Df = lambda (k1,k2): np.array([[k2/(k1+k2)**2, -k1/(k1+k2)**2],
                               [r*k2/(r*k1+k2)**2, -r*k1/(r*k1+k2)**2],
                               [r*k2/(k1+r*k2)**2, -r*k1/(k1+r*k2)**2]])

p = [1,10]
pred = predict.Predict(f, Df=Df, p0=p, pids=['k1','k2'], dids=['S1','S2','S3'])

res = residual.Residual(pred=pred, dat=pred.make_data(p, cv=0.2))
"""

#ens = res.sampling(p0=p, nstep=1000, in_logp=True, seed=3, cutoff_singval=1e-3,
#                   interval_print_step=100, filepath='ens_tmp.txt')
#pens = ens[:,:2] 
#pens.scatter(pts=[p], log10=True, adjust={'left':0.2,'bottom':0.2}, filepath='')
#ens.scatter(pts=[[1,2]], filepath='tmp.pdf')


pred_logp = pred.get_in_logp()

#pred.plot_image(decade=4, npt=100, xyzlabels=['S1','S2','S3'], 
#                filepath='', 
#                color='b', alpha=0.5, shade=False, edgecolor='none')


#prange = {'k1':np.linspace(1,10,10), 'k2':np.linspace(1,10,10)} 
#pred.plot_sloppyv_field(prange, filepath='sloppyvfield_concn.pdf')


  


#pts = [[ 6.63259937,  4.8627397 ,  4.0540656 ], [ 0.90909091,  0.47619048,  0.83333333]]
pred.plot_image(decade=6, pts=geqn.rs, xyzlabels=['J1','J2','J3'])

import matplotlib.pyplot as plt


a
k1s, k2s = np.logspace(-1,1,100), np.logspace(-1,1,100)
k1ss, k2ss = np.meshgrid(k1s, k2s)

fs = [lambda k1,k2:2/(1/k1+1/k2), lambda k1,k2:2/(2/k1+1/k2), lambda k1,k2:2/(1/k1+2/k2)]
#ys = [f(k1ss, k2ss) for f in fs]

#gs = [lambda k1:2/(1/k1), lambda k1:2/(2/k1), lambda k1:2/(1/k1)]

# make parameter grid


# generate points for data manifold
#fvec = np.vectorize(f)

def g(*p):
    return pred(p) 
#ps = np.dstack((k1ss, k2ss))
ys = g(k1ss, k2ss)





fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.plot_surface(*ys, color='b', alpha=0.5, 
                shade=False, edgecolor='none')
ax.set_xlabel('J(wildtype)')
ax.set_ylabel('J(perturb R1)')
ax.set_zlabel('J(perturb R2)')

pt1 = [[f(9.11162756,5.21400829)] for f in fs]
pt2 = [[f(0.5,5)] for f in fs]
ax.scatter(*pt1, c='r', marker='o')
ax.scatter(*pt2, c='g', marker='o')
#ax.scatter(*yrs, c='k', marker='.')

#plt.show()
plt.close()

