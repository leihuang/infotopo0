"""
    A1
    |R1   R3
X1 --->X2 --> A4 
   <---
A3/ R2 \A2

"""

from collections import OrderedDict as OD

import numpy as np

from infotopo.models.rxnnet import model
reload(model)


def get_net1():
    net = model.Network('cycle2')
    net.add_compartment('CELL')
    net.add_species('C1', 'CELL', np.exp(2), is_constant=True)
    net.add_species('C2', 'CELL', np.exp(3), is_constant=True)
    net.add_species('C3', 'CELL', np.exp(1), is_constant=True)
    net.add_species('C4', 'CELL', np.exp(1), is_constant=True)
    net.add_species('X1', 'CELL', 1)
    net.add_species('X2', 'CELL', 1)
    net.add_reaction('R1', stoich_or_eqn='X1+C1<->2 X2', 
                     ratelaw='k1*(C1*X1-X2**2/K1)', p={'k1':1, 'K1':1})
    net.add_reaction('R2', stoich_or_eqn='X2+C2<->X1+C3', 
                     ratelaw='k2*(C2*X2-C3*X1/K2)', p={'k2':1, 'K2':1})
    net.add_reaction('R3', stoich_or_eqn='2 X2<->2 C4', 
                     ratelaw='k3*(X2**2-C4**2/K3)', p={'k3':1, 'K3':1})
    net.add_ratevars()
    net.compile()
    return net


def get_net2():
    net = model.Network(id='net')
    net.add_compartment('CELL')
    net.add_species('C1', 'CELL', 2, is_constant=True)
    net.add_species('C2', 'CELL', 2, is_constant=True)
    net.add_species('C3', 'CELL', 1, is_constant=True)
    
    net.add_species('X1', 'CELL', 0)
    net.add_species('X2', 'CELL', 0)
    
    p1 = OD.fromkeys(['k1f','k1b'], 1)
    p2 = OD.fromkeys(['k2f','k2b'], 1)
    p3 = OD.fromkeys(['k3f','k3b'], 1)
    
    net.add_reaction('R1', stoich_or_eqn='X1+C1<->2 X2', ratelaw='k1f*X1*C1-k1b*X2**2', p=p1)
    net.add_reaction('R2', stoich_or_eqn='X2+C2<->X1', ratelaw='k2f*X2*C2-k2b*X1', p=p2)
    net.add_reaction('R3', stoich_or_eqn='X2<->C3', ratelaw='k3f*X2-k3b*C3', p=p3)
    net.add_ratevars()
    net.compile()
    return net


def get_net3():
    net = model.Network(id='net_cycle2_mar2')
    net.add_compartment('CELL')
    net.add_species('C1', 'CELL', 2, is_constant=True)
    net.add_species('C2', 'CELL', 2, is_constant=True)
    net.add_species('C3', 'CELL', 1, is_constant=True)
    net.add_species('C4', 'CELL', 1, is_constant=True)
    
    net.add_species('X1', 'CELL', 0)
    net.add_species('X2', 'CELL', 0)
    
    p1 = OD.fromkeys(['k1','K1'], 1)
    p2 = OD.fromkeys(['k2','K2'], 1)
    p3 = OD.fromkeys(['k3','K3'], 1)
    
    net.add_reaction('R1', stoich_or_eqn='X1+C1<->2 X2', ratelaw='k1*(X1*C1-X2**2/K1)', p=p1)
    net.add_reaction('R2', stoich_or_eqn='X2+C2<->X1+C3', ratelaw='k2*(X2*C2-X1*C3/K2)', p=p2)
    net.add_reaction('R3', stoich_or_eqn='X2<->C4', ratelaw='k3*(X2-C4/K3)', p=p3)
    net.add_ratevars()
    net.compile()
    return net

net_cycle2_mar_ke1 = get_net1()
net_cycle2_mar = get_net2()
net_cycle2_mar2 = get_net3()



