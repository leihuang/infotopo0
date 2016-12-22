"""


"""

from __future__ import division

from infotopo.models.rxnnet import model
reload(model)


net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 1)
net.add_species('X2', 'CELL', 0)
net.add_reaction('R1', stoich_or_eqn='X1->X2', ratelaw='k1*X1', p={'k1':1})
net.add_reaction('R2', stoich_or_eqn='X2->X1', ratelaw='k2*X2', p={'k2':2})

