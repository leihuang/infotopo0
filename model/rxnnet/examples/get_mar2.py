"""
A two-reaction system with reversible mass action kinetics. 
"""

from model.rxnnet import model
reload(model) 


net0 = model.Network(id='MAR2')
net0.add_compartment('CELL')
net0.add_species('X1', 'CELL', 2, is_constant=True)
net0.add_species('X2', 'CELL', 1, is_constant=True)
net0.add_species('S', 'CELL', 1)
net0.add_reaction_ma('R1', stoich_or_eqn='X1<->S', p={'kf':1,'KE':1}, haldane='kf', add_thermo=False)
net0.add_reaction_ma('R2', stoich_or_eqn='S<->X2', p={'kf':1,'KE':1}, haldane='kf', add_thermo=False)
net0 = net0.add_ratevars()
net0.compile()

net = model.Network(id='MAR2_simple')
net.add_compartment('CELL')
net.add_parameter('X1', 2, is_optimizable=False)
net.add_parameter('X2', 1, is_optimizable=False)
net.add_species('S', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='<->S', ratelaw='k1*(X1-S)', p={'k1':1})
net.add_reaction('R2', stoich_or_eqn='S<->', ratelaw='k2*(S-X2)', p={'k2':1})
net = net.add_ratevars()
net.compile()