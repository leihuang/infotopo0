"""
"""

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model)

from util import plotutil, butil
reload(plotutil)


  
def make_net(netid, ratelaw1, ratelaw2, ratelaw3,
             pids1, pids2, pids3, 
             eqn1='C1<->X1', eqn2='X1<->X2', eqn3='X2<->C2', 
             p1=None, p2=None, p3=None, C1=2, C2=1, C_as_species=True):
    """
    Input:
        ratelaw1 and ratelaw2: strings for ratelaws,
            or mechanisms (eg, 'mmr_ke1', 'mai')
    """
    net = model.Network(id=netid)
    net.add_compartment('CELL')
    if C_as_species:
        net.add_species('C1', 'CELL', C1, is_constant=True)
        net.add_species('C2', 'CELL', C2, is_constant=True)
    else:
        net.add_parameter('C1', C1)
        net.add_parameter('C2', C2)
    net.add_species('X1', 'CELL', 0)
    net.add_species('X2', 'CELL', 0)
    
    if p1 is None:
        p1 = OD.fromkeys(pids1, 1)
    if p2 is None:
        p2 = OD.fromkeys(pids2, 1)
    if p3 is None:
        p3 = OD.fromkeys(pids3, 1)
    
    def _get_ratelaw(rl, eqn, rxnidx):
        if rl == 'mar_ke1':
            ratelaw = 'k%d*(%s)'%(rxnidx, '-'.join(eqn.replace(' ', '').split('<->')))
        elif rl == 'mar':
            subid, proid = eqn.replace(' ', '').split('<->')  # assuming S<->P
            ratelaw = 'k%df*%s-k%db*%s'%(rxnidx, subid, rxnidx, proid)
        else:
            ratelaw = rl
        return ratelaw
    
    net.add_reaction('R1', stoich_or_eqn=eqn1, p=p1,
                     ratelaw=_get_ratelaw(ratelaw1, eqn1, 1))
    net.add_reaction('R2', stoich_or_eqn=eqn2, p=p2,
                     ratelaw=_get_ratelaw(ratelaw2, eqn2, 2))
    net.add_reaction('R3', stoich_or_eqn=eqn3, p=p3,
                     ratelaw=_get_ratelaw(ratelaw3, eqn3, 3))
    net.add_ratevars()
    
    net.compile()
    return net


net_path3_mar_ke1 = make_net('net_path3_mar_ke1', 
                             ratelaw1='mar_ke1', 
                             ratelaw2='mar_ke1',
                             ratelaw3='mar_ke1',
                             pids1=['k1'], pids2=['k2'], pids3=['k3'])

net_path3_mar = make_net('net_path3_mar', 
                         ratelaw1='mar', 
                         ratelaw2='mar',
                         ratelaw3='mar',
                         pids1=['k1f','k1b'], 
                         pids2=['k2f','k2b'], 
                         pids3=['k3f','k3b'])
