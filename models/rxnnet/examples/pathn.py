"""
"""

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model)

from util import plotutil, butil
reload(plotutil)


def make_path2(rl1, rl2, KE=True):
    net = model.Network()
    net.add_species('C1', 2, is_constant=True)
    net.add_species('C2', 1, is_constant=True)
    net.add_species('X', 1)
    if KE:
        net.add_parameter('KE1', 1, is_optimizable=False)
        net.add_parameter('KE2', 2, is_optimizable=False)
    net.add_reaction('R1', stoich_or_eqn='C1<->X', ratelaw=rl1.s, p=OD.fromkeys(rl1.pids, 1))
    net.add_reaction('R2', stoich_or_eqn='X<->C2', ratelaw=rl2.s, p=OD.fromkeys(rl2.pids, 1))
    net.add_ratevars()
    net.compile()
    return net

def make_path0(ratelaws=None, #n=None, ma_or_mm=None, KE=True, 
              r=False, 
              add_ratevars=True, netid=''):
    """
    To make a pathway: 
        - of n reactions with reaction ids 'R1', 'R2', ..., etc.;
        - of boundary species of fixed concentration with ids 'C1' and 'C2'
        - with ratelaws as provided (in either ratelaw instances or str short codes)
    
    Examples:
    
    net = make_path()
         
    Input:
        ratelaws: a list of ratelaw.Ratelaw's or str's ('ma'/'mm' + 'ke'/''),
            eg, ['ma', 'mmke']
        r: if True, add rid to each k of V (for studying perturbation data)
        add_ratevars: bool; if the net is used for studying XxT data, having 
            ratevars can slow down the integration (esp. sens. integration)
        netid: str
        
        #n: int; number of reactions
        #ma_or_mm: str; 'ma' or 'mm'
        #KE: bool; if True, KE is assumed known
    """
    m = len(ratelaws)  # number of reactions
    
    if netid == '':
        netid = 'path%d' % m
    net = model.Network(id=netid)
    net.add_compartment('CELL')
    net.add_species('C1', 'CELL', 2, is_constant=True)
    net.add_species('C2', 'CELL', 1, is_constant=True)
    
    if m == 2:
        spids = ['X']
    else:
        spids = ['X%d'%i for i in range(1, m)] 
    for spid in spids:
        net.add_species(spid, 'CELL', 1.0)
    
    from infotopo.models.rxnnet import ratelaw
    rls = []
    for j, rl_ in enumerate(ratelaws):
        if isinstance(rl_, ratelaw.RateLaw):
            rl = rl_.facelift()
            rls.append(rl)
        elif isinstance(rl, str):
            #rl = f(rl_)
            pass
        else:
            raise ValueError
    
    for rl in rls:
        if r: 
            rid = 'r%d'%i
            net.add_parameter(rid, 1, is_optimizable=False)
            ratelaw = '%s * (%s)'%(rid, ratelaw)
            
        net.add_reaction('R%d'%i, stoich_or_eqn=eqn, ratelaw=ratelaw, 
                         p=OD.fromkeys(pids, 1))
    
    for pid in net.pids:
        if pid.startswith('KE'):
            net.set_var_optimizable(pid, False)
    if add_ratevars:
        net.add_ratevars()

        
    for i in range(1, n+1):
        if i == 1:
            S, P = 'C1', 'X1'
        elif i == n:
            S, P = 'X%d'%(i-1), 'C2'
        else:
            S, P = 'X%d'%(i-1), 'X%d'%i
        eqn = '%s <-> %s'%(S, P)
        
        if ma_or_mm == 'ma':
            if KE:
                kfid, KEid = 'k%df'%i, 'KE%d'%i
                ratelaw = '%s * (%s - %s / %s)'%(kfid, S, P, KEid)
                pids = [kfid, KEid]
            else:
                kfid, krid = 'k%df'%i, 'k%dr'%i
                ratelaw = '%s * %s - %s * %s'%(kfid, S, krid, P)
                pids = [kfid, krid]
        elif ma_or_mm == 'mm':
            if KE:
                kfid, KEid, bfid, brid = 'k%df'%i, 'KE%d'%i, 'b%df'%i, 'b%dr'%i
                ratelaw = '%s * (%s - %s / %s) / (1 + %s * %s + %s * %s)'%\
                    (kfid, S, P, KEid, bfid, S, brid, P)
                pids = [kfid, bfid, brid, KEid]
            else:
                kfid, krid, bfid, brid = 'k%df'%i, 'k%dr'%i, 'b%df'%i, 'b%dr'%i
                ratelaw = '(%s * %s - %s * %s) / (1 + %s * %s + %s * %s)'%\
                    (kfid, S, krid, P, bfid, S, brid, P)
                pids = [kfid, krid, bfid, brid]
        else:
            pass
        
            return net    
    
