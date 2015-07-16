"""
"""

from __future__ import division

import numpy as np

import geodesic, predict, residual, compare
reload(geodesic)
reload(predict)
reload(residual)
reload(compare)

from model.rxnnet.examples.MAI2 import net as net_mai
from model.rxnnet.examples.MAR2 import net as net_mar
from model.rxnnet.examples.MMI2 import net as net_mmi
from model.rxnnet.examples.MMR2 import net as net_mmr
from model.rxnnet import experiments
reload(experiments)


# make data expts
dexpts_mai = experiments.Experiments()
dexpts_mai.add(((), 'S', np.inf))
#dexpts_mai.add((('kf_R1',2), 'S', np.inf))
#dexpts_mai.add((('kf_R2',2), 'S', np.inf))
dexpts_mai.add(((), 'J_R1', np.inf))

dexpts_mar = experiments.Experiments()
dexpts_mar.add(((), 'S', np.inf))
dexpts_mar.add((('k_R1',[0.25,0.5,2,4]), 'S', np.inf))
dexpts_mar.add((('k_R2',[0.25,0.5,2,4]), 'S', np.inf))
dexpts_mar.add(((), 'J_R1', np.inf))
dexpts_mar.add((('k_R1',[0.25,0.5,2,4]), 'J_R1', np.inf))
dexpts_mar.add((('k_R2',[0.25,0.5,2,4]), 'J_R1', np.inf))

dexpts_mmi = experiments.Experiments()
dexpts_mmi.add(((), 'S', np.inf))
dexpts_mmi.add((('Vf_R1',2), 'S', np.inf))
dexpts_mmi.add((('Vf_R2',2), 'S', np.inf))
dexpts_mmi.add(((), 'J_R1', np.inf))

dexpts_mmr = experiments.Experiments()
dexpts_mmr.add(((), 'S', np.inf))
dexpts_mmr.add((('V_R1',[0.25,0.5,2,4]), 'S', np.inf))
dexpts_mmr.add((('V_R2',[0.25,0.5,2,4]), 'S', np.inf))
dexpts_mmr.add(((), 'J_R1', np.inf))
dexpts_mmr.add((('V_R1',[0.25,0.5,2,4]), 'J_R1', np.inf))
dexpts_mmr.add((('V_R2',[0.25,0.5,2,4]), 'J_R1', np.inf))

# make prediction expts
pexpts = experiments.Experiments()
pexpts.add(((), ('J_R1','v_R1'), np.inf))
pexpts.add(((), ('J_R1','v_R2'), np.inf))
pexpts.add(((), ('v_R1','S'), np.inf))
pexpts.add(((), ('v_R2','S'), np.inf))


cmpn12 = compare.Comparison(net_mai, net_mar, dexpts=dexpts_mai, dexpts2=dexpts_mar, pexpts=pexpts)
cmpn21 = compare.Comparison(net_mar, net_mai, dexpts=dexpts_mar, dexpts2=dexpts_mai, pexpts=pexpts)
cmpn24 = compare.Comparison(net_mar, net_mmr, dexpts=dexpts_mar, dexpts2=dexpts_mmr, pexpts=pexpts)
a
p1 = [1,2]
cost2, p2, z, z2 = cmpn12.cmp_prediction(p=p1, ens=False, p0=[1,1], in_logp=True, disp=0)
energies2, ps2, z, zs2 = cmpn12.cmp_prediction(p=p1, ens=True, in_logp=True,
                                               scheme='sigma', sigma0=0.1, 
                                               p0=[1,1], cutoff_singval=1e-3,
                                               disp=0, nstep=1000, seed=1, 
                                               interval_print_step=100)




