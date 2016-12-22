"""
"""

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model)

from util import plotutil, butil
reload(plotutil)



def make_net(netid, ratelaw1, ratelaw2, ratelaw3, ratelaw4,
             pids1, pids2, pids3, pids4,
             eqn1='X1+C1<->2 X2', eqn2='X2+X3<->X1+X4', 
             eqn3='X2<->C3', eqn4='X4+C2<->X3', 
             p1=None, p2=None, p3=None, p4=None,              
             C_as_species=True, C1=2, C2=2, C3=1):
    """
    Input:
        ratelaw1 and ratelaw2: strings for ratelaws or mechanisms (eg, 'mmr', 'mai')
    """
    net = model.Network(id=netid)
    net.add_compartment('CELL')
    if C_as_species:
        net.add_species('C1', 'CELL', C1, is_constant=True)
        net.add_species('C2', 'CELL', C2, is_constant=True)
        net.add_species('C3', 'CELL', C3, is_constant=True)
    else:
        net.add_parameter('C1', C1)
        net.add_parameter('C2', C2)
        net.add_species('C3', C3)
    
    net.add_species('X1', 'CELL', 1)
    net.add_species('X2', 'CELL', 1)
    net.add_species('X3', 'CELL', 1)
    net.add_species('X4', 'CELL', 1)
    
    if p1 is None:
        p1 = OD.fromkeys(pids1, 1)
    if p2 is None:
        p2 = OD.fromkeys(pids2, 1)
    if p3 is None:
        p3 = OD.fromkeys(pids3, 1)
    if p4 is None:
        p4 = OD.fromkeys(pids4, 1)
        
    def _get_ratelaw(rl, eqn, rxnidx):
        # problematic...
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
    net.add_reaction('R4', stoich_or_eqn=eqn4, p=p4,
                     ratelaw=_get_ratelaw(ratelaw4, eqn4, 4))
    
    net.add_ratevars()
    net.compile()
    
    return net


net_cycle4_mar_ke1 = make_net('net_cycle4_mar_ke1', 
                              ratelaw1='k1*(C1*X1-X2**2)',
                              ratelaw2='k2*(X2*X3-X1*X4)',
                              ratelaw3='k3*(X2-C3)',
                              ratelaw4='k4*(X4*C2-X3)',
                              pids1=['k1'],
                              pids2=['k2'],
                              pids3=['k3'],
                              pids4=['k4'])

net_cycle4_mar = make_net('net_cycle4_mar', 
                          ratelaw1='k1f*C1*X1-k1b*X2**2',
                          ratelaw2='k2f*X2*X3-k2b*X1*X4',
                          ratelaw3='k3f*X2-k3b*C3',
                          ratelaw4='k4f*X4*C2-k4b*X3',
                          pids1=['k1f','k1b'],
                          pids2=['k2f','k2b'],
                          pids3=['k3f','k3b'],
                          pids4=['k4f','k4b'])

net_cycle4_mmr = make_net('net_cycle4_mmr', 
                          ratelaw1='(k1f*C1*X1-k1b*X2**2)/(1+b11*C1+b12*X1+b11*C1*b12*X1+2*b13*X2+(b13*X2)**2)',
                          ratelaw2='(k2f*X2*X3-k2b*X1*X4)/(1+b21*X2+b22*X3+b21*X2*b22*X3+b23*X1+b24*X4+b23*X1*b24*X4)',
                          ratelaw3='(k3f*X2-k3b*C3)/(1+b31*X2+b32*C3)',
                          ratelaw4='(k4f*X4*C2-k4b*X3)/(1+b41*X4+b42*C2+b41*X4*b42*C2+b43*X3)',
                          pids1=['k1f','k1b','b11','b12','b13'],
                          pids2=['k2f','k2b','b21','b22','b23','b24'],
                          pids3=['k3f','k3b','b31','b32'],
                          pids4=['k4f','k4b','b41','b42','b43'])

net_cycle4_mmr2 = make_net('net_cycle4_mmr2', 
                           ratelaw1='(k1f*C1*X1-k1b*X2**2)/(1+b1c1*C1+b1x1*X1+b1c1*C1*b1x1*X1+2*b1x2*X2+(b1x2*X2)**2)',
                           ratelaw2='(k2f*X2*X3-k2b*X1*X4)/(1+b2x2*X2+b2x3*X3+b2x2*X2*b2x3*X3+b2x1*X1+b2x4*X4+b2x1*X1*b2x4*X4)',
                           ratelaw3='(k3f*X2-k3b*C3)/(1+b3x2*X2+b3c3*C3)',
                           ratelaw4='(k4f*X4*C2-k4b*X3)/(1+b4x4*X4+b4c2*C2+b4x4*X4*b4c2*C2+b4x3*X3)',
                           pids1=['k1f','k1b','b1c1','b1x1','b1x2'],
                           pids2=['k2f','k2b','b2x1','b2x2','b2x3','b2x4'],
                           pids3=['k3f','k3b','b3x2','b3c3'],
                           pids4=['k4f','k4b','b4x4','b4c2','b4x3'])

net_cycle4_mmr3 = make_net('net_cycle4_mmr3', 
                           ratelaw1='(k1f*C1*X1-k1r*X2**2)/(1+b1f1*C1+b1f2*X1+b1f1*C1*b1f2*X1+2*b1r*X2+(b1r*X2)**2)',
                           ratelaw2='(k2f*X2*X3-k2r*X1*X4)/(1+b2f1*X2+b2f2*X3+b2f1*X2*b2f2*X3+b2r1*X1+b2r2*X4+b2r1*X1*b2r2*X4)',
                           ratelaw3='(k3f*X2-k3r*C3)/(1+b3f*X2+b3r*C3)',
                           ratelaw4='(k4f*C2*X4-k4r*X3)/(1+b4f1*C2+b4f2*X4+b4f1*C2*b4f2*X4+b4r*X3)',
                           pids1=['k1f','k1r','b1f1','b1f2','b1r'],
                           pids2=['k2f','k2r','b2f1','b2f2','b2r1','b2r2'],
                           pids3=['k3f','k3r','b3f','b3r'],
                           pids4=['k4f','k4r','b4f1','b4f2','b4r'])



