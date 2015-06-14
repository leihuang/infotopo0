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
net.add_species('X1', 'CELL', 2)
net.add_species('X2', 'CELL', 1, is_boundary_condition=True)
net.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
#net.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(S1-S2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='X2->', ratelaw='k3*X2', p={'k3':1})
net.add_parameter('KE2', 2)

net.add_algebraic_rule('X1-X2/KE2')
net.reactions.get('R3').stoichiometry = {'X1':'-1'}




net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 2)
net.add_species('X2', 'CELL', 1, is_boundary_condition=True)
net.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
#net.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(S1-S2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='X2->', ratelaw='k3*X2', p={'k3':1})
net.add_parameter('KE2', 2)

net.add_algebraic_rule('X1-X2/KE2')
net.reactions.get('R3').stoichiometry = {'X1':'-1'}



traj = net.get_traj([0,1,10,100,1000])
print traj

a
"""
net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 3, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S1', 'CELL', 1)
net.add_species('S2', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='X1<->S1', ratelaw='k1*abs(X1-S1*2)', p={'k1':1})
net.add_reaction('R2', stoich_or_eqn='S1<->S2', ratelaw='k2*(S1-S2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='S2<->X2', ratelaw='k3*(S2-X2)', p={'k3':1})
net.add_ratevars()

net.reactions.get('R1').kineticLaw = net.reactions.get('R1').kineticLaw.replace('abs(', 'f(')
net.add_func_def('f', ('x'), 'min(x, -x)')

traj = net.get_traj((0,1))
"""

"""
# net2 uses equilibrium assumption for R1
net2 = model.Network(id='net2')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 3, is_constant=True)
net2.add_species('X2', 'CELL', 1, is_constant=True)
net2.add_species('S1', 'CELL', 1, is_boundary_condition=True)
net2.add_species('S2', 'CELL', 1)
#net2.add_reaction_eq('R1', stoich_or_eqn='X1<->S', KE=1)
net2.add_reaction(rxnid='R1', stoich_or_eqn='X1<->S1', p={'KE_R1':1}, ratelaw='0')
net2.add_reaction('R2', stoich_or_eqn='S1<->S2', ratelaw='k2*(S1-S2)', p={'k2':1})
net2.add_reaction('R3', stoich_or_eqn='S2<->X2', ratelaw='k3*(S2-X2)', p={'k3':1})
net2.add_algebraic_rule('X1-S1')
#net2.algebraicVars.set('S1', net2.variables.get('S1'))

traj = net2.get_traj([0,1,2])
#net2._makeCrossReferences()
print traj
"""

net2 = model.Network(id='net2')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 2)
net2.add_species('X2', 'CELL', 2)
net2.add_species('X3', 'CELL', 1, is_boundary_condition=True)
#net2.add_reaction_eq('R1', stoich_or_eqn='X1<->S', KE=1)
net2.add_reaction(rxnid='R1', stoich_or_eqn='->X1', ratelaw='1')
net2.add_reaction(rxnid='R2', stoich_or_eqn='->X2', ratelaw='1')
net2.add_reaction(rxnid='R4', stoich_or_eqn='X3->', ratelaw='k4*X3', p={'k4':10})
#net2.add_reaction(rxnid='R3', stoich_or_eqn='X1+X2<->X3', ratelaw='k3*(X1*X2-X3)', p={'k3':1})
net2.add_algebraic_rule('X1*X2-X3')

traj = net2.get_traj([0,0.1,100,1000])
print traj



#net2.add_rate_rule('S1', '1')
#net2.add_assignment_rule('S1', '%s*KE_%s'%('X1','R1'))
#net2.add_ratevars()

#net2.to_sbml('tmp.xml')

"""

for rxn in net2.reactions:
    if rxn.kineticLaw == '0':
        subids = model._get_substrates(rxn.stoichiometry, multi=True)
        proids = model._get_products(rxn.stoichiometry, multi=True)
        net2.add_algebraic_rule('%s/%s-KE_%s'%('*'.join(proids), 
                                               '*'.join(subids), rxn.id))
        #net2.add_assignment_rule(proids[0], '%s*KE_%s'%(''.join(subids), rxn.id))
        for proid in proids:
            net2.species.get(proid).is_boundary_condition = True
            net2.algebraicVars.set(proid, net2.variables.get(proid))

#net2 = net2.replace_varids({'S1':'Y1','S2':'Y2'})

traj = net.get_traj((0,1))
traj2 = net2.get_traj(traj.times)

traj_all = traj + traj2

"""