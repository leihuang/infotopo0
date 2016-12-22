"""
"""

from __future__ import division

import numpy as np

from infotopo.models.rxnnet import model
reload(model)


S0, E0, r, log_kf, log_kr, log_kc = 1, 0.25, 0.25/1, 3, 2, 1
#log_kf, log_kr, log_kc = 1, 1/2, 3/2

net_ma = model.Network('net_ma')
net_ma.add_parameter('E0', E0)
net_ma.add_parameter('S0', S0)
net_ma.add_species('E', 'E0')
net_ma.add_species('S', 'S0')
net_ma.add_species('C', 0)
net_ma.add_species('P', 0)
net_ma.add_reaction(id='Rf', stoich_or_eqn='E+S->C', ratelaw='kf*E*S', p={'kf':np.exp(log_kf)})
net_ma.add_reaction(id='Rr', stoich_or_eqn='C->E+S', ratelaw='kr*C', p={'kr':np.exp(log_kr)})
net_ma.add_reaction(id='Rc', stoich_or_eqn='C->E+P', ratelaw='kc*C', p={'kc':np.exp(log_kc)})

net2 = model.Network('net_ma_E0,S0_AScoord')
net2.add_compartment('cell')
net2.add_parameter('S0', S0)
net2.add_parameter('r', r)
net2.add_species('S', 'cell', 'S0')
net2.add_species('E', 'cell', 'r*S0')
net2.add_species('C', 'cell', 0)
net2.add_species('P', 'cell', 0)
net2.add_reaction(id='Rf', stoich_or_eqn='E+S->C', ratelaw='kf*E*S', p={'kf':np.exp(log_kf)})
net2.add_reaction(id='Rr', stoich_or_eqn='C->E+S', ratelaw='kr*C', p={'kr':np.exp(log_kr)})
net2.add_reaction(id='Rc', stoich_or_eqn='C->E+P', ratelaw='kc*C', p={'kc':np.exp(log_kc)})
net2.add_parameter('Pn')  # normalized P
net2.add_assignment_rule('Pn', 'P/S0')  
net_ma_ascoord = net2

net_ma2 = net_ma.copy()
net_ma2.id = 'net_ma2'
net_ma2.remove_component('Rr')
net_ma2.remove_component('kr')


net_mm = model.Network('net_mm')
net_mm.add_compartment('cell')
net_mm.add_species('S', 'cell', S0)
net_mm.add_species('P', 'cell', 0)
net_mm.add_reaction(id='R', stoich_or_eqn='S->P', ratelaw='V*S/(K+S)', p={'V':1, 'K':1})

