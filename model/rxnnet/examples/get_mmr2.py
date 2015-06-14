"""
A two-reaction system with reversible Michaelis-Menten kinetics. 
"""

from model.rxnnet import model
reload(model) 


net = model.Network(id='MMR2')
net.add_compartment('CELL')
net.add_species('X1', 'CELL', 2, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S', 'CELL', 1)
net.add_reaction_mm_qe('R1', stoich_or_eqn='X1<->S', pM={'Vf':1,'X1':1,'S':1,'KE':1}, 
                       haldane='Vf', mechanism='standard', add_thermo=False)
net.add_reaction_mm_qe('R2', stoich_or_eqn='S<->X2', pM={'Vf':1,'S':1,'X2':1,'KE':1}, 
                       haldane='Vf', mechanism='standard', add_thermo=False)
net = net.add_ratevars()
net.compile()
net.to_sbml('model_mmr2.xml')

net2 = model.Network(id='MMR2_simple')
net2.add_compartment('CELL')
net2.add_species('X1', 'CELL', 2, is_constant=True)
net2.add_species('X2', 'CELL', 1, is_constant=True)
net2.add_species('S', 'CELL', 1)
net2.add_reaction('R1', stoich_or_eqn='X1<->S', 
                 ratelaw='V1*(X1/K1-S/K1)/(1+X1/K1+S/K1)', p={'V1':1,'K1':1}) 
net2.add_reaction('R2', stoich_or_eqn='S<->X2', 
                 ratelaw='V2*(S/K2-X2/K2)/(1+S/K2+X2/K2)', p={'V2':1,'K2':1}) 
net2 = net.add_ratevars()
net2.compile()
net.to_sbml('model_mmr2_simple.xml')

