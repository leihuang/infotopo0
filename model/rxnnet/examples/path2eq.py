"""
Testing network specification and MCA calculations of equilibrium assumption.

A simple pathway:

    R1    R2
X1 <-> S <-> X2
"""

import geodesic, predict, residual
reload(geodesic)
reload(predict)
reload(residual)

from model.rxnnet import model
reload(model) 


net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 2, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='X1<->S', ratelaw='k1*(X1-S)', p={'k1':1e6})
net.add_reaction('R2', stoich_or_eqn='S<->X2', ratelaw='k2*(S-X2)', p={'k2':1})
net.add_ratevars()

# net2 uses equilibrium assumption for R1
net2 = model.Network(id='net')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 2, is_constant=True)
net2.add_species('X2', 'CELL', 1, is_constant=True)
net2.add_species('S', 'CELL', 1)
#net2.add_reaction_eq('R1', stoich_or_eqn='X1<->S', KE=1)
net.add_reaction(rxnid='R1', stoich_or_eqn='X1<->S', p={'KE_R1':1}, ratelaw='')
net2.add_reaction('R2', stoich_or_eqn='S<->X2', ratelaw='k2*(S-X2)', p={'k2':1})
net2.add_ratevars()

for rxn in net2.reactions:
    if rxn.kineticLaw == '':
        