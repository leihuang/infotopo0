from infotopo.model.rxnnet.model import Network
  

net = Network(id='net')

net.add_compartment('CELL')

net.add_species('X1', 'CELL', 2, is_constant=True)
net.add_species('X2', 'CELL', 1, is_constant=True)
net.add_species('S', 'CELL', 1)

net.add_reaction('R1', stoich_or_eqn='X1<->S', ratelaw='k1*(X1-S)', p={'k1':10})
net.add_reaction('R2', stoich_or_eqn='S<->X2', ratelaw='k2*(S-X2)', p={'k2':1})

net.to_sbml('test.xml')



"""
net.add_reaction_mm_qe('R1', stoich_or_eqn='X1<->S', 
                        pM={'Vf':1, 'Vb':1, 'X1':1, 'S':1, 'dG0':0}, 
                        haldane='', mechanism='standard', 
                        add_thermo=True, T=25)

net.add_reaction_mm_qe('R2', stoich_or_eqn='S<->X2', 
                        pM={'Vf':1, 'Vb':1, 'S':1, 'X2':1, 'dG0':0}, 
                        haldane='', mechanism='standard', 
                        add_thermo=True, T=25)
"""
