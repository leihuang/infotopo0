"""
"""

from collections import OrderedDict as OD

import pandas as pd

from infotopo.model.rxnnet import model as m
from infotopo import fitting as f, sampling as s
reload(m)
reload(f)
reload(s)



mod = m.from_smod('../model/rxnnet/smod/mm1.smod')

traj = mod.integrate((1,2,5))

expt = m.Experiment()
expt.add('', OD([(('S','P','E','X'), [1,2,3])]), 1)
expt.add(('R1',2), OD([('P',[1.5, 5])]), 1)

pred = mod.get_prediction(expt, precision=True)

p = mod.p
dat = pred(p)

fit = f.Fit(pred, dat)

pens1, dens1 = s.sampling_jeffreysian(pred)

pens2, dens2 = s.sampling_bayesian(pred, dat)

dens_3d = dens2.pca(k=3)

dens_3d.scatterplot(pts=None, filepath='plot.pdf')
