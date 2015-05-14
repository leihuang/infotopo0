"""
Testing network specification and MCA calculations of equilibrium assumption.

A simple pathway:

    R1     R2     R3
X1 <-> S1 <-> S2 <-> X2
"""

import geodesic, predict, residual
reload(geodesic)
reload(predict)
reload(residual)

from model.rxnnet import model
reload(model) 


net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 3, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S1', 'CELL', 1)
net.add_species('S2', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='X1<->S1', ratelaw='k1*(X1-S1)', p={'k1':1e6})
net.add_reaction('R2', stoich_or_eqn='S1<->S2', ratelaw='k2*(S1-S2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='S2<->X2', ratelaw='k3*(S2-X2)', p={'k3':1})
net.add_ratevars()

# net2 uses equilibrium assumption for R1
net2 = model.Network(id='net2')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 3, is_constant=True)
net2.add_species('X2', 'CELL', 1, is_constant=True)
net2.add_species('S1', 'CELL', 1)
net2.add_species('S2', 'CELL', 1)
#net2.add_reaction_eq('R1', stoich_or_eqn='X1<->S', KE=1)
net2.add_reaction(rxnid='R1', stoich_or_eqn='X1<->S1', p={'KE_R1':1}, ratelaw='0')
net2.add_reaction('R2', stoich_or_eqn='S1<->S2', ratelaw='k2*(S1-S2)', p={'k2':1})
net2.add_reaction('R3', stoich_or_eqn='S2<->X2', ratelaw='k3*(S2-X2)', p={'k3':1})
#net2.add_algebraic_rule('X1-S1')
#net2.add_rate_rule('S1', '1')
#net2.add_assignment_rule('S1', '%s*KE_%s'%('X1','R1'))
#net2.add_ratevars()




for rxn in net2.reactions:
    if rxn.kineticLaw == '0':
        subids = model._get_substrates(rxn.stoichiometry, multi=True)
        proids = model._get_products(rxn.stoichiometry, multi=True)
        net2.add_algebraic_rule('%s/%s-KE_%s'%('*'.join(proids), 
                                               '*'.join(subids), rxn.id))
        #net2.add_assignment_rule(proids[0], '%s*KE_%s'%(''.join(subids), rxn.id))
        for proid in proids:
            net2.species.get(proid).is_boundary_condition = True



net2 = net2.change_varids({'S1':'Y1','S2':'Y2'})

traj = net.get_traj((0,1))
traj2 = net2.get_traj(traj.times)

traj_all = traj + traj2

