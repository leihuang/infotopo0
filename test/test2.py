"""

"""

from collections import OrderedDict as OD

import numpy as np
import pandas as pd

from SloppyCell.ReactionNetworks import *

from util.sloppycell import trajutil
from infotopo.model.rxnnet import model as m
from infotopo import sampling as s
reload(m)
reload(s)

from infotopo import JP_Ensembles as jp 



net1 = m.from_smod('../model/rxnnet/smod/mm1.smod')
net2 = m.from_smod('../model/rxnnet/smod/mm2.smod')
net1.id = 'net'
net2 = Network('net')
net2.add_compartment('cell')
net2.add_species('S', 'cell', 100)
net2.add_species('P', 'cell', 1)
net2.add_parameter('V', 1)
net2.add_parameter('K', 1)
net2.addReaction('R', {'S':-1, 'P':1}, kineticLaw='V*S/(S+K)')
net2.pids = net2.parameters.keys()

trj1 = Dynamics.integrate(net1, np.linspace(0,10,51), fill_traj=False)
trj2 = Dynamics.integrate(net2, np.linspace(0,10,51), fill_traj=False)
expt1 = trajutil.traj2expt(trj1, net1.id, datvarids=['S','P'], sigma=1)
expt2 = trajutil.traj2expt(trj2, net2.id, datvarids=['S','P'], sigma=1)
mod1 = Model([expt1], [net1])
mod2 = Model([expt2], [net2])

pens1, r1 = jp.jeffreys_prior_ensemble_log_params(mod1, mod1.params, steps=200)
pens1 = s.Ensemble(np.array(pens1), varids=net1.pids, r=r1)
f1 = lambda p: pd.Series(mod1.res(p), index=mod1.residuals.keys())
rens1 = pens1.calc(f1)


pens2, r2 = jp.jeffreys_prior_ensemble_log_params(mod2, mod2.params, steps=200)
pens2 = s.Ensemble(np.array(pens2), varids=net2.pids, r=r2)
f2 = lambda p: pd.Series(mod2.res(p), index=mod2.residuals.keys())
rens2 = pens2.calc(f2)


rens = rens1.append(rens2)

rens_pca = rens.pca(k=3)


rens1_pca = rens_pca[:201]
rens2_pca = rens_pca[201:]


#rens_pca_3d = rens_pca.slice(cidxs=[0,1,2])
s.scatter([rens1_pca, rens2_pca])
#dens = [mod.res(p) for p in pens]




