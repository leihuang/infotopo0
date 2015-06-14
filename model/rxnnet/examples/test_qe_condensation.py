"""
Testing quasi-equilibrium assumption for a condensation reaction

R1    R2  -----------  
-> X1 -> | X2 -      |
         |    |R5    | R6    R7
         |    |-> X5 | -> X6 ->
         |    |      |
-> X3 -> | X4 -      |
R3    R4  -----------         
"""

from model.rxnnet import model
reload(model) 


times = [0,0.1,1,10,100,1000]

net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 10)
net.add_species('X2', 'CELL', 5)
net.add_species('X3', 'CELL', 2)
net.add_species('X4', 'CELL', 1)
net.add_species('X5', 'CELL', 1)
net.add_species('X6', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
net.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(X1-X2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='->X3', ratelaw='1')
net.add_reaction('R4', stoich_or_eqn='X3<->X4', ratelaw='k4*(X3-X4)', p={'k4':1})
net.add_reaction('R5', stoich_or_eqn='X2+X4<->X5', ratelaw='k5*(X2*X4-X5/KE5)', p={'k5':1e9,'KE5':2})
net.add_reaction('R6', stoich_or_eqn='X5<->X6', ratelaw='k6*(X5-X6)', p={'k6':1})
net.add_reaction('R7', stoich_or_eqn='X6->', ratelaw='k7*X6', p={'k7':1})
traj = net.get_traj(times)

net2 = model.Network(id='net2')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 10)
net2.add_species('X2', 'CELL', 5, is_boundary_condition=True)
net2.add_species('X3', 'CELL', 2)
net2.add_species('X4', 'CELL', 1)
net2.add_species('X5', 'CELL', 1)
net2.add_species('X6', 'CELL', 1)
net2.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
net2.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(X1-X2)', p={'k2':1})
net2.add_reaction('R3', stoich_or_eqn='->X3', ratelaw='1')
net2.add_reaction('R4', stoich_or_eqn='X3<->X4', ratelaw='k4*(X3-X4)', p={'k4':1})
#net2.add_reaction('R5', stoich_or_eqn='X2+X4<->X5', ratelaw='k5*(X2*X4-X5/KE5)', p={'k5':1e9,'KE5':2})
net2.add_reaction('R6', stoich_or_eqn='X5<->X6', ratelaw='k6*(X5-X6)', p={'k6':1})
net2.add_reaction('R7', stoich_or_eqn='X6->', ratelaw='k7*X6', p={'k7':1})

net2.add_parameter('KE5', 2)
net2.add_algebraic_rule('X2*X4-X5/KE5')
net2.reactions.get('R2').stoichiometry ={'X1':-1, 'X4':1}
traj2 = net2.get_traj(times)

print traj
print traj2
