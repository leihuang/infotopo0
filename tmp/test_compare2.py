from __future__ import division
import numpy as np
import time
#from matplotlib.mlab import PCA

import geodesic, predict, residual, compare
reload(geodesic)
reload(predict)
reload(residual)
reload(compare)

from infotopo.model.rxnnet import model, experiments
reload(model) 
reload(experiments)

import test_compare
reload(test_compare)
from test_compare import net, net2


# make data expts
dexpts = experiments.Experiments()
dexpts.add(((), 'S', np.inf))
dexpts.add((('k1',2), 'S', np.inf))
dexpts.add((('k2',2), 'S', np.inf))
dexpts.add(((), 'J_R1', np.inf))


dexpts2 = experiments.Experiments()
dexpts2.add(((), 'S', np.inf))
dexpts2.add((('k',2), 'S', np.inf))
dexpts2.add((('V',2), 'S', np.inf))
dexpts2.add(((), 'J_R1', np.inf))


# make data predict
dpred = net.get_predict(dexpts)
dpred2 = net2.get_predict(dexpts2)

# generate data
p = [1,2]
dat = dpred.make_dat(p, scheme='sigma', sigma=1)
dat2 = dat.rename(dict(zip(dpred.dids, dpred2.dids)))


# fit data to model2
res2 = residual.Residual(dpred2, dat2)
p2 = res2.fit([1,2,1], in_logp=True, disp=1)


# make prediction expts
pexpts = experiments.Experiments()
pexpts.add(((), ('J_R1','v_R1'), np.inf))

ppred = net.get_predict(pexpts)
ppred2 = net2.get_predict(pexpts)

fcc = ppred(p)
fcc2 = ppred2(p2)

print fcc, fcc2


"""
pexpt = model.Experiments()
cmpr = compare.Comparison(net, net2, dexpt, pexpt)
p = [1,2]
z, z2 = cmpr.predict(p)
"""
