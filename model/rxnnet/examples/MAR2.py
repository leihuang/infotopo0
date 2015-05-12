"""
A two-reaction system with reversible mass action kinetics. 
"""

from model.rxnnet import model
reload(model) 


net = model.Network(id='MAR2')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 2, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S', 'CELL', 1)
net.add_reaction_ma('R1', stoich_or_eqn='X1<->S', p={'kf':1,'kb':1}, haldane='k')
net.add_reaction_ma('R2', stoich_or_eqn='S<->X2', p={'kf':1,'kb':1}, haldane='k')
net = net.add_ratevars()
net.compile()
