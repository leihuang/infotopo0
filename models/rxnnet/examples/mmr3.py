"""
A three-reaction system with reversible Michaelis-Menten kinetics. 
"""

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model) 


net0 = model.Network(id='MMR3_standard')
net0.add_compartment('CELL')
net0.add_species('A1', 'CELL', 2, is_constant=True)
net0.add_species('A2', 'CELL', 1, is_constant=True)
net0.add_species('X1', 'CELL', 1)
net0.add_species('X2', 'CELL', 1)
net0.add_reaction_mm_qe('R1', stoich_or_eqn='A1<->X1', pM={'Vf':1,'A1':1,'X1':1,'KE':1}, 
                        haldane='Vf', mechanism='standard', add_thermo=False)
net0.add_reaction_mm_qe('R2', stoich_or_eqn='X1<->X2', pM={'Vf':1,'X1':1,'X2':1,'KE':1}, 
                        haldane='Vf', mechanism='standard', add_thermo=False)
net0.add_reaction_mm_qe('R3', stoich_or_eqn='X2<->A2', pM={'Vf':1,'X2':1,'A2':1,'KE':1}, 
                        haldane='Vf', mechanism='standard', add_thermo=False)
net0 = net0.add_ratevars()
net0.compile()
#net0.to_sbml('model_mmr2.xml')


net = model.Network(id='MMR3')
net.add_compartment('CELL')
net.add_species('A1', 'CELL', 2, is_constant=True)
net.add_species('A2', 'CELL', 1, is_constant=True)
#net.add_parameter('X1', 2, is_optimizable=False)
#net.add_parameter('X2', 1, is_optimizable=False)
net.add_species('X1', 'CELL', 1)
net.add_species('X2', 'CELL', 1)
net.add_reaction('R1', stoich_or_eqn='<->X1', 
                 ratelaw='V1/K1*(A1-X1)/(1+A1/K1+X1/K1)', 
                 p=OD([('V1',1),('K1',1)])) 
net.add_reaction('R2', stoich_or_eqn='X1<->X2', 
                 ratelaw='V2/K2*(X1-X2)/(1+X1/K2+X2/K2)', 
                 p=OD([('V2',1),('K2',1)]))
net.add_reaction('R3', stoich_or_eqn='X2<->A2', 
                 ratelaw='V3/K3*(X2-A2)/(1+X2/K3+A2/K3)', 
                 p=OD([('V3',1),('K3',1)])) 
net = net.add_ratevars()
net.compile()

#net2.to_sbml('model_mmr2_simple.xml')



