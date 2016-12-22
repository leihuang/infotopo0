"""
"""

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model)

from util import plotutil, butil
reload(plotutil)


  
def make_net(netid, ratelaw1, ratelaw2, pids1, pids2, 
             eqn1='C1<->X', eqn2='X<->C2', 
             p1=None, p2=None, C1=2, C2=1, C_as_species=True):
    """
    Input:
        ratelaw1 and ratelaw2: strings for ratelaws or mechanisms (eg, 'mm', 'ma')
    """
    net = model.Network(id=netid)
    net.add_compartment('CELL')
    if C_as_species:
        net.add_species('C1', 'CELL', C1, is_constant=True)
        net.add_species('C2', 'CELL', C2, is_constant=True)
    else:
        net.add_parameter('C1', C1)
        net.add_parameter('C2', C2)
    net.add_species('X', 'CELL', 1.0)
    
    if p1 is None:
        p1 = OD.fromkeys(pids1, 1.0)
    if p2 is None:
        p2 = OD.fromkeys(pids2, 1.0)
    net.add_reaction('R1', stoich_or_eqn=eqn1, ratelaw=ratelaw1, p=p1)
    net.add_reaction('R2', stoich_or_eqn=eqn2, ratelaw=ratelaw2, p=p2)
    net.add_ratevars()
    
    return net


net_path2_mar = make_net('net_path2_mar', 
                         ratelaw1='k1*(C1-X)', ratelaw2='k2*(X-C2)',
                         pids1=['k1'], pids2=['k2'])

net_path2_mar2 = make_net('net_path2_mar2', 
                          ratelaw1='k1*C1-k2*X', ratelaw2='k3*X-k4*C2',
                          pids1=['k1','k2'], pids2=['k3','k4'])

net_path2_mar3 = make_net('net_path2_mar3', 
                         ratelaw1='k1*(C1-X/KE1)', ratelaw2='k2*(X-C2/KE2)',
                         pids1=['k1','KE1'], pids2=['k2','KE2'])

net_path2_mmr1 = make_net('net_path2_mmr1',
                          ratelaw1='V1/K1*(C1-X)/(1+C1/K1+X/K1)', 
                          ratelaw2='V2/K2*(X-C2)/(1+X/K2+C2/K2)',
                          pids1=['V1','K1'], pids2=['V2','K2'])

net_path2_mmr2 = make_net('net_path2_mmr2',
                          ratelaw1='(V1f*C1/K1C1-V1b*X/K1X)/(1+C1/K1C1+X/K1X)', 
                          ratelaw2='(V2f*X/K2X-V2b*C2/K2C2)/(1+X/K2X+C2/K2C2)',
                          pids1=['V1f','V1b','K1C1','K1X'], 
                          pids2=['V2f','V2b','K2X','K2C2'])

net_path2_mmr6 = make_net('net_path2_mmr6',
                          ratelaw1='(V1*C1/K1-V2*X/K2)/(1+C1/K1+X/K2)', 
                          ratelaw2='(V3*X/K3-V4*C2/K4)/(1+X/K3+C2/K4)',
                          pids1=['V1','V2','K1','K2'], 
                          pids2=['V3','V4','K3','K4'])

net_path2_mmr7 = make_net('net_path2_mmr7',
                          ratelaw1='(k1*C1-V2*X/K2)/(1+X/K2)', 
                          ratelaw2='(V3*X/K3-V4*C2/K4)/(1+X/K3+C2/K4)',
                          pids1=['k1','V2','K2'], pids2=['V3','V4','K3','K4'])

net_path2_mmr8 = make_net('net_path2_mmr8',
                          ratelaw1='(k1*C1-k2*X)/(1+b1*C1+b2*X)', 
                          ratelaw2='(k3*X-k4*C2)/(1+b3*X+b4*C2)',
                          pids1=['k1','k2','b1','b2'], 
                          pids2=['k3','k4','b3','b4'])

net_path2_mmr9 = make_net('net_path2_mmr9',
                          ratelaw1='(k1*C1-k2*X)/(1+b1*C1+b2*X)', 
                          ratelaw2='(k3*X-k4*C2)/(1+b3*X)',
                          pids1=['k1','k2','b1','b2'], 
                          pids2=['k3','k4','b3'])

net_path2_mmr10 = make_net('net_path2_mmr10',
                           ratelaw1='(k1f*C1-k1b*X)/(1+b1c1*C1+b1x*X)', 
                           ratelaw2='(k2f*X-k2b*C2)/(1+b2x*X+b2c2*C2)',
                           pids1=['k1f','k1b','b1c1','b1x'], 
                           pids2=['k2f','k2b','b2x','b2c2'])

net_path2_mmr11 = make_net('net_path2_mmr11',
                           ratelaw1='(k1f*C1-k1r*X)/(1+b1f*C1+b1r*X)', 
                           ratelaw2='(k2f*X-k2r*C2)/(1+b2f*X+b2r*C2)',
                           pids1=['k1f','k1r','b1f','b1r'], 
                           pids2=['k2f','k2r','b2f','b2r'])

net_path2_mmr12 = make_net('net_path2_mmr12',
                           ratelaw1='(k1f*(C1-X/KE1))/(1+b1f*C1+b1r*X)', 
                           ratelaw2='(k2f*(X-C2/KE2))/(1+b2f*X+b2r*C2)',
                           pids1=['k1f','b1f','b1r','KE1'], 
                           pids2=['k2f','b2f','b2r','KE2'])

net_path2_mmr5 = make_net('net_path2_mmr5',
                          ratelaw1='(k1f*C1-k1b*X)/(1+C1/K1C1+X/K1X)', 
                          ratelaw2='(k2f*X-k2b*C2)/(1+X/K2X+C2/K2C2)',
                          pids1=['k1f','k1b','K1C1','K1X'], 
                          pids2=['k2f','k2b','K2X','K2C2'])

net_path2_mmr3 = make_net('net_path2_mmr3',
                          eqn1='C1+X<->2 X', eqn2='2 X<->X+C2', 
                          ratelaw1='V1/K1**2*(C1*X-X**2)/(1+C1*X/K1**2+X**2/K1**2)', 
                          ratelaw2='V2/K2**2*(X**2-X*C2)/(1+X**2/K2**2+X*C2/K2**2)',
                          pids1=['V1','K1'], pids2=['V2','K2'])


"""
net_path2_mmr4 = make_net('net_path2_mmr4',
                          eqn1='C1+X<->2 X', eqn2='2 X<->X+C2', 
                          ratelaw1='V1/K1**2*(C1*X-X**2)/(1+C1*X/K1**2+X**2/K1**2)', 
                          ratelaw2='V2/K2**2*(X**2-X*C2)/(1+X**2/K2**2+X*C2/K2**2)',
                          pids1=[], pids2=[])
"""

if __name__ == '__main__':
    net = net_path2_mar
    polys = net.get_polynomials(varid2rid=[('C1','r1'), ('C2','r2')])
    
    
