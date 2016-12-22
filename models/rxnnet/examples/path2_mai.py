"""
"""

import numpy as np

from util import plotutil

from infotopo.models.rxnnet import model
reload(model)


k1, k2 = 1, 10

net = model.Network(id='path2_mai')
net.add_compartment('cell')
net.add_spp(X=1, Y=0, Z=0)
net.add_reaction('R1', stoich_or_eqn='X->Y', ratelaw='k1*X', p={'k1':k1})
net.add_reaction('R2', stoich_or_eqn='Y->Z', ratelaw='k2*Y', p={'k2':k2})

traj = net.get_traj(np.arange(0, 20.001, 0.2).tolist())

plotutil.plot3d(*traj.values.T, marker='o', cs=traj.index/max(traj.index),
                linewidth=1, s=1,
                xyzlabels=['X', 'Y', 'Z'], xyzlims=[[0,1]]*3, 
                title='k1=%d, k2=%d'%(k1, k2))

