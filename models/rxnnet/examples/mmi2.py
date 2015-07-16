"""
A two-reaction system with irreversible Michaelis-Menten kinetics. 
"""

from models.rxnnet import model
reload(model) 


net = model.Network(id='MMI2')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 2, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S', 'CELL', 1)
net.add_reaction_mm_qe('R1', stoich_or_eqn='X1->S', pM={'Vf':1,'X1':1}, 
                       mechanism='standard')  
net.add_reaction_mm_qe('R2', stoich_or_eqn='S->X2', pM={'Vf':1,'S':1}, 
                       mechanism='standard')  
net = net.add_ratevars()
net.compile()
