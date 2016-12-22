"""
A three-reaction system with reversible mass action kinetics. 
"""

from infotopo.models.rxnnet import model
reload(model) 


net0 = model.Network(id='MAR3_standard')
net0.add_compartment('CELL')
net0.add_species('A1', 'CELL', 2, is_constant=True)
net0.add_species('A2', 'CELL', 1, is_constant=True)
net0.add_species('X1', 'CELL', 1)
net0.add_species('X2', 'CELL', 1)
net0.add_reaction_ma('R1', stoich_or_eqn='A1<->X1', p={'kf':1,'KE':1}, haldane='kf', add_thermo=False)
net0.add_reaction_ma('R2', stoich_or_eqn='X1<->X2', p={'kf':1,'KE':1}, haldane='kf', add_thermo=False)
net0.add_reaction_ma('R3', stoich_or_eqn='X2<->A2', p={'kf':1,'KE':1}, haldane='kf', add_thermo=False)
net0 = net0.add_ratevars()
net0.compile()

net = model.Network(id='MAR3')
net.add_compartment('CELL')
net.add_parameter('A1', 2, is_optimizable=False)
net.add_parameter('A2', 1, is_optimizable=False)
net.add_species('X1', 'CELL', 1)
net.add_species('X2', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='<->X1', ratelaw='k1*(A1-X1)', p={'k1':1})
net.add_reaction('R2', stoich_or_eqn='X1<->X2', ratelaw='k2*(X1-X2)', p={'k2':1})
net.add_reaction('R3', stoich_or_eqn='X2<->', ratelaw='k3*(X2-A2)', p={'k3':1})
net = net.add_ratevars()
net.compile()