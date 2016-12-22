"""
Testing quasi-equilibrium assumption for a uni-uni reaction

R1     R2  ---------   R4    R5
-> X1 <-> |X2 <-> X3| <-> X4 ->
           ---------
"""

from models.rxnnet import model
reload(model) 


times = [0,0.1,1,10,100,1000]

net = model.Network(id='net')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 10)
net.add_species('X2', 'CELL', 5)
net.add_species('X3', 'CELL', 2)
net.add_species('X4', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
net.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(X1-X2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='X2<->X3', ratelaw='k3*(X2-X3/KE3)', p={'k3':1e9,'KE3':2})
net.add_reaction('R4', stoich_or_eqn='X3<->X4', ratelaw='k4*(X3-X4)', p={'k4':1})
net.add_reaction('R5', stoich_or_eqn='X4->', ratelaw='k5*X4', p={'k5':1})
traj = net.get_traj(times)

net2 = model.Network(id='net2')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 10)
net2.add_species('X2', 'CELL', 5, is_boundary_condition=True)
net2.add_species('X3', 'CELL', 2)
net2.add_species('X4', 'CELL', 1)
net2.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
net2.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(X1-X2)', p={'k2':1})
#net2.add_reaction('R3', stoich_or_eqn='X2<->X3', ratelaw='k3*(X2-X3/KE3)', p={'k3':1e9,'KE3':2})
net2.add_reaction('R4', stoich_or_eqn='X3<->X4', ratelaw='k4*(X3-X4)', p={'k4':1})
net2.add_reaction('R5', stoich_or_eqn='X4->', ratelaw='k5*X4', p={'k5':1})

net2.add_parameter('KE3', 2)
net2.add_algebraic_rule('X2-X3/KE3')
net2.reactions.get('R2').stoichiometry ={'X1':-1, 'X3':1}
traj2 = net2.get_traj(times)


net3 = model.Network(id='net3')
net3.add_compartment('CELL')
net3.add_species('X1', 'CELL', 10)
net3.add_species('X2', 'CELL', 5)
net3.add_species('X3', 'CELL', 2, is_boundary_condition=True)
net3.add_species('X4', 'CELL', 1)
net3.add_reaction('R1', stoich_or_eqn='->X1', ratelaw='1')
net3.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(X1-X2)', p={'k2':1})
#net3.add_reaction('R3', stoich_or_eqn='X2<->X3', ratelaw='k3*(X2-X3/KE3)', p={'k3':1e9,'KE3':2})
net3.add_reaction('R4', stoich_or_eqn='X3<->X4', ratelaw='k4*(X3-X4)', p={'k4':1})
net3.add_reaction('R5', stoich_or_eqn='X4->', ratelaw='k5*X4', p={'k5':1})

net3.add_parameter('KE3', 2)
net3.add_algebraic_rule('X2-X3/KE3')
net3.reactions.get('R4').stoichiometry ={'X2':-1, 'X4':1}
traj3 = net3.get_traj(times)

print traj
print traj2
print traj3