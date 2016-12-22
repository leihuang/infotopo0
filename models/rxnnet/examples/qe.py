"""
Quasi-equilibrium system...

"""

import numpy as np

from util import plotutil

from infotopo.models.rxnnet import model, experiments
reload(model)
reload(experiments)


kf, kr, kc = 1, 1, 1

net = model.Network(id='qe')
net.add_compartment('cell')
#net.add_spp(X=1, Y=0, Z=0)
net.add_species('X', 'cell', 1)
net.add_species('Y', 'cell', 0)
net.add_species('Z', 'cell', 0)
net.add_reaction('Rf', stoich_or_eqn='X->Y', ratelaw='kf*X', p={'kf':kf})
net.add_reaction('Rr', stoich_or_eqn='Y->X', ratelaw='kr*Y', p={'kr':kr})
net.add_reaction('Rc', stoich_or_eqn='Y->Z', ratelaw='kc*Y', p={'kc':kc})
net = net.add_ratevars()

traj = net.get_traj((0,5))

a

expts = experiments.Experiments()
expts.add(((), 'Z', [0.5,1,2]))  #np.arange(0, 20.001, 0.2).tolist())) 

pred = net.get_predict(expts, name='qe')

pred_kfkr = pred.currying(name='qe_kfkr', kc=1)

#pred_kfkr_log10 = pred_kfkr.get_in_log10p()

pred_kfkr.plot_image(ndecade=4, npt=20, alpha=0.2, linewidth=1, rstride=3, cstride=3,
                     xyzlims=[[-0.5,1.5]]*3, xyzlabels=['Z(1)','Z(2)','Z(3)'])
