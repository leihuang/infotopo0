"""
Perhaps incorporate add_reaction_mmqe method from model.py...

A <-> P
A + B <-> P
A + B <-> P + Q
A <-> P + Q  (is it just symmetric to A + B <-> P?)
A + B + C <-> P + Q  (eg, GAPDH)

- mm vs ma
- KE vs not KE
- V, K vs k, b
- 1, 2 vs A, B
(not sure if the time would be worth it)

Purposes:
- Explore the local boundaries
- Expedite network construction by ratelaw composition
- Compare different ratelaws

- Develop a naming schemes for the ratelaws and their parameters
v = kf*S/(1+bf*S+br*P)
kf = Vf/Kf
bf = 1/Kf
br = 1/Kr

bf/kf = 1/Vf = 
br/kf = br/bf bf/kf = (1/Kr)/(1/Kf) 1/Vf = Kf/Kr 1/Vf

"""

from __future__ import division
from collections import OrderedDict as OD
import itertools
import re

import sympy 
import numpy as np

from SloppyCell import ExprManip as exprmanip
 
from util import butil

from infotopo import predict, hasse
reload(predict)
reload(hasse)

#from infotopo.models.rxnnet import model
#reload(model)


class RateLaw(object):
    """
    2016-07-16:
    - Only distinguish between xids and pids. cids (eg, C1 and KE) is taken to be pids. 
    - It is left to systems-level modeling (that is, at the level of net) where
    cids are decided and fixed. 
    - When coverted to a pred, cids can be decided using currying.
    
    2016-07-22:
    - What about doing ratelaw reduction? KE should be kept fixed right??
    - Or if I want to do a reduction starting from the interior...
    - Yes, a ratelaw should have cids... 
    
    2016-08-21:
    I can always fix certain params if I want to later, by coding some methods 
    if necessary. Now I think a more economical way is perhaps only xids and
    pids. 
    
    In essence a function mapping from x and p to v. Independent of stoichiometry.  
    """
    
    def __init__(self, s, xids, pids=None, cids=None, info='', **kwargs):
        """
        If pids is not given, it is inferred from s; in this case, cids, if any,
        must be given.
        """
        self.s = s
        self.xids = xids
        
        if cids is None:
            cids = []
        self.cids = cids
        self.info = info
        
        if pids is None:
            pids_set = set(exprmanip.extract_vars(s)) - set(xids+cids)  # a set, unordered
            items = [(pid, s.index(pid)) for pid in pids_set]
            pids = [item[0] for item in sorted(items, key=lambda item: item[1])]
            if 'inf' in pids:
                pids.remove('inf')
        self.pids = pids
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        def v(x, p, c=None):
            subs = dict(zip(xids+pids, list(x)+list(p)))
            if cids:
                subs.update(**dict(zip(cids, c)))
            return eval(s, subs)
        self.v = v
    
        
    def __repr__(self):
        return self.s
        
    
    @property
    def pdim(self):
        return len(self.pids)
    
        
    def facelift(self, xids_new=None, xmap=None,
                 pids_new=None, cids_new=None,
                 pcmap=None, rxnidx=None,
                 #cmap=None, 
                 **kwargs):
        """Change the look by substituting new variable names.
        
        Input:
            xmap: a function that changes xids
            pcmap: a function that changes pids and cids or a str indicating 
                facelifting scheme:
                - 'num': insert numbering index; eg, k2r, KE1, b2X1, b3GAP
                - 'rxnid': use underscore; eg, kf_R1, kr_GAPDH, KE_PK, b_GAPDH_GAP
            rxnidx: an int (numbering index) or str (rxnid)
        """
        # get new xids
        if xids_new is not None:
            pass
        elif xmap:
            xids_new = map(xmap, self.xids) 
        else:
            xids_new = self.xids
        xmap = lambda xid: dict(zip(self.xids, xids_new))[xid]
        
        # get new pids
        if pids_new is not None and cids_new is not None:
            pass
        elif pcmap:
            # get pcmap
            if pcmap == 'num':
                def pcmap(varid):
                    if varid in ['KE', 'rK']:
                        return varid + str(rxnidx)
                    elif varid[0] in ['V', 'k','Q']:  # eg, k1f, V2r
                        return varid[0] + str(rxnidx) + varid[1:]  
                    elif varid[0] in ['K', 'b']:  # eg, b2X1, K1C1 (assuming KM takes the form of K)
                        return varid[0] + str(rxnidx) + xmap(varid[1:])  
                    else:
                        #raise ValueError("pid: %s" % varid)
                        return varid + str(rxnidx)
            if pcmap == 'rxnid':
                def pcmap(varid):
                    if varid in ['KE', 'rK']:
                        return varid + '_' + rxnidx
                    elif varid[0] in ['k', 'V']:  # eg, kf_R1, Vr_GAPDH
                        return varid + '_' + rxnidx  
                    elif varid[0] in ['b', 'K']:  # eg, b_R1_X1, K_GAPDH_GAP 
                        return '%s_%s_%s'%(varid[0], rxnidx, xmap(varid[1:]))  
                    else:
                        #raise ValueError("varid: %s" % varid)
                        return varid + '_' + rxnidx

            pids_new = map(pcmap, self.pids)
            cids_new = map(pcmap, self.cids)
        else:
            pids_new = self.pids
            cids_new = self.cids
            
        """
        elif pmap or add_idx is not None:
            if add_idx is not None:
                def pmap(pid):
                    if pid == 'KE':
                        return pid + str(add_idx)
                    else:
                        return pid[0] + str(add_idx) + pid[1:]
            pids_new = map(pmap, self.pids)
        else:
            pids_new = self.pids
        """
        
        # get new s
        subs = dict(zip(self.xids+self.pids+self.cids, xids_new+pids_new+cids_new))
        s_new = exprmanip.sub_for_vars(self.s, subs)
        
        # copy the attributes, such as 'info'
        attrs = self.__dict__.copy()
        for attrname in ['s', 'xids', 'pids', 'cids']:
            del attrs[attrname]
        kwargs.update(attrs)
        
        return RateLaw(s=s_new, xids=xids_new, pids=pids_new, cids=cids_new, 
                       **kwargs)
            
            
    def get_predict(self, xs, c=None, p0=None):
        """
        """
        if c is not None and not isinstance(c, dict):
            c = dict(zip(self.cids, c))
        return predict.str2predict(s=self.s, pids=self.pids, uids=self.xids, 
                                   c=c, us=xs, p0=p0)
        


class RateLaw0(object):
    def __init__(self, s, xids, pids, cids=None, id='', info='', texmap=None,
                 eqn='', 
                 **kwargs):
        """
        s_tex: need to be able to strip the \left( and \right) when they're unnecessary
        should take texmap, s_tex, etc. away. Users should provide such a func
        when using graph.change_labels (rename it to set_labels?).   
        
        Input:
            s: a str; eg, kf*(S-P/KE)
            xids:
            pids:
            cids:  
            
        """
        def v(x, p, c=None):
            subs = dict(zip(xids+pids, list(x)+list(p)))
            if cids:
                subs.update(dict(zip(cids, c)))
            return eval(s, subs) 
        
        self.s = s
        self.v = v
        self.xids = xids
        self.pids = pids
        if cids is None:
            cids = []
        self.cids = cids
        self.id = id
        if texmap is None:
            texmap = {}
        self.s_tex = '\displaystyle '+exprmanip.expr2TeX(s, texmap).replace('\\', '\\\\')
        self.info = info
        for k, v in kwargs.items():
            setattr(self, k, v)
    
        
    def __repr__(self):
        return self.s
    
    
    @property
    def pdim(self):
        return len(self.pids)
    
    
    def facelift(self, xids_new=None, pids_new=None, cids_new=None, pmap=None,
                 add_idx=None, **kwargs):
        """Change the look by substituting new variable names.
        
        Input:
            pmap: a function that changes pids
            add_idx: None or int
        """
        s_new = self.s
        
        if xids_new is None:
            xids_new = self.xids
        else:
            s_new = exprmanip.sub_for_vars(s_new, dict(zip(self.xids, xids_new)))
            
        if pids_new is not None or pmap or add_idx is not None:
            if add_idx is not None:
                pmap = lambda pid: pid[0] + str(add_idx) + pid[1]
            if pmap:
                pids_new = map(pmap, self.pids)
            s_new = exprmanip.sub_for_vars(s_new, dict(zip(self.pids, pids_new)))
        else:
            pids_new = self.pids
            
        if cids_new is None:
            cids_new = self.cids
        else:
            s_new = exprmanip.sub_for_vars(s_new, dict(zip(self.cids, cids_new)))
            
        return RateLaw(s=s_new, xids=xids_new, pids=pids_new, cids=cids_new, 
                       id=self.id)
        
    
    def get_predict(self, xs, c=None, p0=None):
        from collections import Mapping
        if c is not None and not isinstance(c, Mapping):
            c = dict(zip(self.cids, c))
        return predict.str2pred(s=self.s, pids=self.pids, uids=self.xids, 
                                us=xs, c=c, p0=p0)
        
        
    def currying(self):
        pass


def str2tex(s, texmap):
    """A more elaborate version of SloppyCell's ExprManip.expr2Tex, allowing 
    for preserving the "structural look" of the expression, among other things.
    """
    _str2texstr = lambda s: exprmanip.expr2TeX(s, texmap).\
        replace('\\cdot ', '').replace('\\frac', '\\\\frac') .\
        replace('\\left', '\\\\left').replace('\\right', '\\\\right')    
    _str2ast = lambda s: exprmanip.AST.strip_parse(s)
    _ast2str = lambda a: exprmanip.ast2str(a)
    
    ast = _str2ast(s)
        
    # The code block below is used to keep the fractional structure in
    # latex: eg, (V/K)/(1+X/K) would be changed to V/(K(1+X/K)) by default.
    if isinstance(ast, exprmanip.AST.Div):
        if isinstance(ast.left, exprmanip.AST.Mul):
            numtexstr = '%s \\\\left( %s \\\\right)' %\
                (_str2texstr(_ast2str(ast.left.left)), 
                 _str2texstr(_ast2str(ast.left.right)))
        else:
            numtexstr = _str2texstr(_ast2str(ast.left))
        denomtexstr = _str2texstr(_ast2str(ast.right))
        texstr = '\\\\displaystyle \\\\frac{%s}{%s}' % (numtexstr, denomtexstr) 
        #texstr = '\\\\frac{%s}{%s}' % (numtexstr, denomtexstr) 
    else:
        texstr =  '\\\\displaystyle %s' % _str2texstr(s)
        #texstr =  '%s' % _str2texstr(s)
        
    # remove parentheses in some numerators and denominators
    #if '\\\\right)}{' in texstr:
    #    texstr = re.sub(r'\\\\right\)}{', '}{', texstr)
    #    texstr = re.sub(r'\\\\left(', '', texstr)
    if '}{\\\\left(' in texstr:
        texstr = re.sub(r'}{\\\\left\(', '}{', texstr)
        texstr = re.sub(r'\\\\right\)}$', '}', texstr)
        
    return texstr


def _has_edge_is_singular(rl1, rl2, rxn):
    """Returns two bools: 
        - is there an edge between rl1 and rl2
        - is the edge singular
    """
    pdim1, pdim2 = rl1.pdim, rl2.pdim
    info1, info2 = rl1.info.lstrip('mm11'), rl2.info.lstrip('mm11')
    
    assert pdim1 == pdim2 + 1
    
    if pdim2 == 0:
        if info1[0] == 'f' and info2 == '0':
            return True, False
        if info1[0] == 'f' and info2 == 'inff':
            return True, True
        elif info1[0] == 'r' and info2 =='0':
            return True, False
        elif info1[0] == 'r' and info2 == 'infr':
            return True, True
        elif info1 == 'APlininf' and info2 in ['inff', 'infr']:
            return True, True
        else:
            return False, False
    else:
                
        def _info2limits(info):
            limits = butil.Series(['', set(), set(), False], 
                                  ['dir', 'lin', 'sat', 'inf'])
            if info == '':
                return limits
            if info[0] in ['f', 'r']:
                limits.dir = info[0]
                info = info[1:]
            if info.endswith('inf'):  # specifically for mm11APlininf; any other solution?
                limits.inf = True
                info = info.rstrip('inf')
            info = info.replace('lin', 'lin_').replace('sat', 'sat_')
            parts = filter(None, info.split('_'))
            for part in parts:
                if part.endswith('lin'):
                    limits.lin.update(list(part.rstrip('lin')))
                if part.endswith('sat'):
                    limits.sat.update(list(part.rstrip('sat')))
            return limits
        
        l1 = _info2limits(info1)
        l2 = _info2limits(info2)
        
        # unidirectionalization
        if len(l2.dir)==len(l1.dir)+1 and l2.lin==l1.lin and l2.sat==l1.sat:
            return True, False
        # single linearization
        elif l2.dir==l1.dir and len(l2.lin-l1.lin)==1 and l2.sat==l1.sat:
            return True, False
        # double saturation
        elif l2.dir==l1.dir and l2.lin==l1.lin==set() and len(l2.sat-l1.sat)==2:
            return True, False
        # final saturation
        elif l2.dir==l1.dir and l2.lin==l1.lin and len(l2.lin)==len(rl.xids)-1 and len(l2.sat-l1.sat)==1:
            return True, False
        # saturation-turn-linearization
        elif l2.dir==l1.dir and len(l2.lin-l1.lin)==1 and len(l1.sat-l2.sat)==1:
            return True, False
        # infinitization
        elif l1.dir=='' and l2.inf:
            return True, True
        else:
            return False, False


def get_hasse_mm11():
    """
    """
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl4 = _get_rl('(Vf * A/KA - Vr * P/KP) / (1 + A/KA + P/KP)', 'mm11') 
    
    rl3s = [_get_rl('(Vf * A/KA) / (1 + A/KA + P/KP)', 'f'),
            _get_rl('(kf*A - Vr * P/KP) / (1 + P/KP)', 'Alin'),
            _get_rl('(Vf*A - Vr*r*P) / (A + r*P)', 'APsat'),
            _get_rl('(Vfp * A/KA - kr*P) / (1 + A/KA)', 'Plin'),
            _get_rl('(- Vr * P/KP) / (1 + A/KA + P/KP)', 'r')]
    
    rl2s = [_get_rl('(Vf * A/KA) / (1 + A/KA)', 'fPlin'),
            _get_rl('(kf*A) / (1 + P/KP)', 'fAlin'), 
            _get_rl('Vf*A / (A + r*P)', 'fAPsat'),
            _get_rl('(Vfp*A - Vr*P) / P', 'AlinPsat'),
            _get_rl('kf*A - kr*P', 'APlin'),
            _get_rl('(Vf*A - Vrp*P) / A', 'PlinAsat'),
            _get_rl('(- Vr*r*P) / (A + r*P)', 'rAPsat'),
            _get_rl('(- kr*P) / (1 + A/KA)', 'rPlin'),
            _get_rl('(- Vr * P/KP) / (1 + P/KP)', 'rAlin')]
    
    rl1s = [_get_rl('Vf', 'fPlinAsat'),  # Vf, Asat
            _get_rl('kf*A', 'fAPlin'),  # ma11f
            _get_rl('Vfp*A/P', 'fAlinPsat'),  # Qf
            _get_rl('inf * (A - P/KE)', 'infAPlin', style='dashed'),
            _get_rl('- Vrp*P/A', 'rPlinAsat'),  # Qr
            _get_rl('- kr*P', 'rAPlin'),  # ma11r
            _get_rl('- Vr', 'rAlinPsat')]  # Vr, Psat  
    
    rl0s = [_get_rl('inf', 'inff', style='dashed'),
            _get_rl('0', '0'),
            _get_rl('-inf', 'infr', style='dashed')]
    
    hd = hasse.HasseDiagram(rank=4)
    
    rls = OD([(4, [rl4]), (3, rl3s), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
            
    # between corank 0 and 1
    hd.add_edge('mm11', 'f', info='Vr->0')
    hd.add_edge('mm11', 'Alin', info='Vf,KA->inf')
    hd.add_edge('mm11', 'APsat', info='KA,KP->0')
    hd.add_edge('mm11', 'Plin', info='Vr,KP->inf')
    hd.add_edge('mm11', 'r', info='Vf->0')
    
    # between corank 1 and 2
    hd.add_edge('f', 'fPlin', info='KP->inf')
    hd.add_edge('f', 'fAlin', info='Vf,KA->inf')
    hd.add_edge('f', 'fAPsat', info='KA,KP->0')
    
    hd.add_edge('Alin', 'fAlin', info='Vr->0')
    hd.add_edge('Alin', 'AlinPsat', info='kf->inf,KP->0')
    hd.add_edge('Alin', 'APlin', info='Vr,KP->inf')
    hd.add_edge('Alin', 'rAlin', info='kf->0')
    
    hd.add_edge('APsat', 'fAPsat')
    hd.add_edge('APsat', 'AlinPsat')
    hd.add_edge('APsat', 'PlinAsat')
    hd.add_edge('APsat', 'rAPsat')
    
    hd.add_edge('Plin', 'fPlin')
    hd.add_edge('Plin', 'APlin')
    hd.add_edge('Plin', 'PlinAsat')
    hd.add_edge('Plin', 'rPlin')
    
    hd.add_edge('r', 'rAlin')
    hd.add_edge('r', 'rPlin')
    hd.add_edge('r', 'rAPsat')
    
    # between corank 2 and 3
    hd.add_edge('fPlin', 'fPlinAsat')
    hd.add_edge('fPlin', 'fAPlin')
    hd.add_edge('fAlin', 'fAPlin')
    hd.add_edge('fAlin', 'fAlinPsat')
    hd.add_edge('fAPsat', 'fPlinAsat')
    hd.add_edge('fAPsat', 'fAlinPsat')
    hd.add_edge('AlinPsat', 'fAlinPsat')
    hd.add_edge('AlinPsat', 'rAlinPsat')
    hd.add_edge('APlin', 'fAPlin')
    hd.add_edge('APlin', 'rAPlin')
    hd.add_edge('PlinAsat', 'fPlinAsat')
    hd.add_edge('PlinAsat', 'rPlinAsat')
    hd.add_edge('rAPsat', 'rPlinAsat')
    hd.add_edge('rAPsat', 'rAlinPsat')
    hd.add_edge('rPlin', 'rPlinAsat')
    hd.add_edge('rPlin', 'rAPlin')
    hd.add_edge('rAlin', 'rAPlin')
    hd.add_edge('rAlin', 'rAlinPsat')
    hd.add_edge('APlin', 'infAPlin', style='dashed')
    
    # between corank 3 and 4
    hd.add_edge('fPlinAsat', 'inff', style='dashed')
    hd.add_edge('fPlinAsat', '0')
    hd.add_edge('fAPlin', 'inff', style='dashed')
    hd.add_edge('fAPlin', '0')
    hd.add_edge('fAlinPsat', 'inff', style='dashed')
    hd.add_edge('fAlinPsat', '0')
    hd.add_edge('rPlinAsat', 'infr', style='dashed')
    hd.add_edge('rPlinAsat', '0')
    hd.add_edge('rAPlin', 'infr', style='dashed')
    hd.add_edge('rAPlin', '0')
    hd.add_edge('rAlinPsat', 'infr', style='dashed')
    hd.add_edge('rAlinPsat', '0')
    hd.add_edge('infAPlin', 'inff', style='dashed')
    hd.add_edge('infAPlin', 'infr', style='dashed')

    texmap = OD([('Vfp',"V_f"), ('Vrp',"V_r"),
                 (' r*','\\\\rho '), ('*r*','\\\\rho '), 
                 ('Vf','V_f'), ('Vr','V_r'), ('KA','K_A'), ('KP','K_P'), 
                 ('KE','K_E'), ('kf','k_f'), ('kr','k_r'), 
                 ('inf','\\\\infty '), ('*','')])
    
    def _str2tex(s):  
        def _repl(s):
            return s.replace('A/KA', '\\\\frac{A}{KA}').\
                replace('P/KP', '\\\\frac{P}{KP}').\
                replace('(P/A)', '\\\\frac{P}{A}').\
                replace('P/A', '\\\\frac{P}{A}').\
                replace('A/P', '\\\\frac{A}{P}').\
                replace('P/KE', '\\\\frac{P}{KE}')
                
        if ' / ' in s:  # central /
            n, d = s.split(' / ')
            tex = '\\\\frac{%s}{%s}' % (_repl(n).strip('()'), _repl(d).strip('()'))
        else:
            tex = _repl(s)
        for k, v in texmap.items():  # sequential replacement so order matters
            tex = tex.replace(k, v)
        return '\\\\displaystyle ' + tex
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        # Not using str2tex which uses SloppyCell's exprmanip as it disrupts
        # the expression structure (like moving a denominator in the numerator
        # to the denominator)
        # Not much work (only two ratelaw hds would be shown the formula) 
        # so an ad-hoc solution suffices here
        #texstr = str2tex(s, texmap)  
        texstr = _str2tex(s)
        hd.get_node(nodeid)['texstr'] = texstr
        
    return hd


def get_hasse_mm11_kb():
    """
    """
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl4 = _get_rl('(kf*A - kr*P) / (1 + bA*A + bP*P)', 'mm11') 
    
    rl3s = [_get_rl('kf*A / (1 + bA*A + bP*P)', 'f'),
            _get_rl('(kf*A - kr*P) / (1 + bP*P)', 'Alin'),
            _get_rl('(Vf*A - Vr*r*P) / (A + r*P)', 'APsat'),
            _get_rl('(kf*A - kr*P) / (1 + bA*A)', 'Plin'),
            _get_rl('-kr*P / (1 + bA*A + bP*P)', 'r')]
    
    rl2s = [_get_rl('kf*A / (1 + bA*A)', 'fPlin'),
            _get_rl('kf*A / (1 + bP*P)', 'fAlin'), 
            _get_rl('Vf*A / (A + r*P)', 'fAPsat'),
            _get_rl('(Vfp*A - Vr*P) / P', 'AlinPsat'),
            _get_rl('kf*A - kr*P', 'APlin'),
            _get_rl('(Vf*A - Vrp*P) / A', 'PlinAsat'),
            _get_rl('(- Vr*r*P) / (A + r*P)', 'rAPsat'),
            _get_rl('-kr*P / (1 + bA*A)', 'rPlin'),
            _get_rl('-kr*P / (1 + bP*P)', 'rAlin')]
    
    rl1s = [_get_rl('Vf', 'fPlinAsat'),  # Vf, Asat
            _get_rl('kf*A', 'fAPlin'),  # ma11f
            _get_rl('Vfp*A/P', 'fAlinPsat'),  # Qf
            _get_rl('inf * (A - P/KE)', 'infAPlin', style='dashed'),
            _get_rl('- Vrp*P/A', 'rPlinAsat'),  # Qr
            _get_rl('- kr*P', 'rAPlin'),  # ma11r
            _get_rl('- Vr', 'rAlinPsat')]  # Vr, Psat  
    
    rl0s = [_get_rl('inf', 'inff', style='dashed'),
            _get_rl('0', '0'),
            _get_rl('-inf', 'infr', style='dashed')]
    
    hd = hasse.HasseDiagram(rank=4)
    
    rls = OD([(4, [rl4]), (3, rl3s), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
            
    # between corank 0 and 1
    hd.add_edge('mm11', 'f')
    hd.add_edge('mm11', 'Alin')
    hd.add_edge('mm11', 'APsat')
    hd.add_edge('mm11', 'Plin')
    hd.add_edge('mm11', 'r')
    
    # between corank 1 and 2
    hd.add_edge('f', 'fPlin')
    hd.add_edge('f', 'fAlin')
    hd.add_edge('f', 'fAPsat')
    
    hd.add_edge('Alin', 'fAlin')
    hd.add_edge('Alin', 'AlinPsat')
    hd.add_edge('Alin', 'APlin')
    hd.add_edge('Alin', 'rAlin')
    
    hd.add_edge('APsat', 'fAPsat')
    hd.add_edge('APsat', 'AlinPsat')
    hd.add_edge('APsat', 'PlinAsat')
    hd.add_edge('APsat', 'rAPsat')
    
    hd.add_edge('Plin', 'fPlin')
    hd.add_edge('Plin', 'APlin')
    hd.add_edge('Plin', 'PlinAsat')
    hd.add_edge('Plin', 'rPlin')
    
    hd.add_edge('r', 'rAlin')
    hd.add_edge('r', 'rPlin')
    hd.add_edge('r', 'rAPsat')
    
    # between corank 2 and 3
    hd.add_edge('fPlin', 'fPlinAsat')
    hd.add_edge('fPlin', 'fAPlin')
    hd.add_edge('fAlin', 'fAPlin')
    hd.add_edge('fAlin', 'fAlinPsat')
    hd.add_edge('fAPsat', 'fPlinAsat')
    hd.add_edge('fAPsat', 'fAlinPsat')
    hd.add_edge('AlinPsat', 'fAlinPsat')
    hd.add_edge('AlinPsat', 'rAlinPsat')
    hd.add_edge('APlin', 'fAPlin')
    hd.add_edge('APlin', 'rAPlin')
    hd.add_edge('PlinAsat', 'fPlinAsat')
    hd.add_edge('PlinAsat', 'rPlinAsat')
    hd.add_edge('rAPsat', 'rPlinAsat')
    hd.add_edge('rAPsat', 'rAlinPsat')
    hd.add_edge('rPlin', 'rPlinAsat')
    hd.add_edge('rPlin', 'rAPlin')
    hd.add_edge('rAlin', 'rAPlin')
    hd.add_edge('rAlin', 'rAlinPsat')
    hd.add_edge('APlin', 'infAPlin', style='dashed')
    
    # between corank 3 and 4
    hd.add_edge('fPlinAsat', 'inff', style='dashed')
    hd.add_edge('fPlinAsat', '0')
    hd.add_edge('fAPlin', 'inff', style='dashed')
    hd.add_edge('fAPlin', '0')
    hd.add_edge('fAlinPsat', 'inff', style='dashed')
    hd.add_edge('fAlinPsat', '0')
    hd.add_edge('rPlinAsat', 'infr', style='dashed')
    hd.add_edge('rPlinAsat', '0')
    hd.add_edge('rAPlin', 'infr', style='dashed')
    hd.add_edge('rAPlin', '0')
    hd.add_edge('rAlinPsat', 'infr', style='dashed')
    hd.add_edge('rAlinPsat', '0')
    hd.add_edge('infAPlin', 'inff', style='dashed')
    hd.add_edge('infAPlin', 'infr', style='dashed')

    texmap = OD([('Vfp',"V_f"), ('Vrp',"V_r"),
                 (' r*','\\\\rho '), ('*r*','\\\\rho '), 
                 ('Vf','V_f'), ('Vr','V_r'), ('KA','K_A'), ('KP','K_P'), 
                 ('KE','K_E'), ('kf','k_f'), ('kr','k_r'), 
                 ('bA','b_A'), ('bP','b_P'),
                 ('inf','\\\\infty '), ('*','')])
    
    def _str2tex(s):  
        def _repl(s):
            return s.replace('A/KA', '\\\\frac{A}{KA}').\
                replace('P/KP', '\\\\frac{P}{KP}').\
                replace('(P/A)', '\\\\frac{P}{A}').\
                replace('P/A', '\\\\frac{P}{A}').\
                replace('A/P', '\\\\frac{A}{P}').\
                replace('P/KE', '\\\\frac{P}{KE}')
                
        if ' / ' in s:  # central /
            n, d = s.split(' / ')
            tex = '\\\\frac{%s}{%s}' % (_repl(n).strip('()'), _repl(d).strip('()'))
        else:
            tex = _repl(s)
        for k, v in texmap.items():  # sequential replacement so order matters
            tex = tex.replace(k, v)
        return '\\\\displaystyle ' + tex
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        # Not using str2tex which uses SloppyCell's exprmanip as it disrupts
        # the expression structure (like moving a denominator in the numerator
        # to the denominator)
        # Not much work (only two ratelaw hds would be shown the formula) 
        # so an ad-hoc solution suffices here
        #texstr = str2tex(s, texmap)  
        texstr = _str2tex(s)
        hd.get_node(nodeid)['texstr'] = texstr
        
    return hd



def get_hasse_mm11Vf1():
    """
    """
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl3 = _get_rl('(1 * A/KA - Vr * P/KP) / (1 + A/KA + P/KP)', 'mm11Vf1') 
    
    rl2s = [_get_rl('(1 * A/KA) / (1 + A/KA + P/KP)', 'Vf1f'),
            _get_rl('(1 * A - Vr*r*P) / (A + r*P)', 'Vf1APsat'),
            _get_rl('(1 * A/KA - kr*P) / (1 + A/KA)', 'Vf1Plin'),
            _get_rl('(- Vr * P/KP) / (1 + P/KP)', 'rAlin')]
    
    rl1s = [_get_rl('(1 * A/KA) / (1 + A/KA)', 'Vf1fPlin'),
            _get_rl('1 * A / (A + r*P)', 'Vf1fAPsat'),
            _get_rl('(1 * A - Vr*P) / A', 'Vf1PlinAsat'),
            _get_rl('-kr * P', 'rAPlin'),
            _get_rl('-Vr', 'rAlinPsat')]
    
    rl0s = [_get_rl('1', '1'),
            _get_rl('0', '0'),
            _get_rl('-inf', 'infr', style='dashed')]
    
    hd = hasse.HasseDiagram(rank=3)
    
    rls = OD([(3, [rl3]), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
            
    # between corank 0 and 1
    hd.add_edge('mm11Vf1', 'Vf1f', info='Vr->0')
    hd.add_edge('mm11Vf1', 'Vf1APsat', info='KA,KP->0')
    hd.add_edge('mm11Vf1', 'Vf1Plin', info='Vr,KP->inf')
    hd.add_edge('mm11Vf1', 'rAlin', info='KA->inf')
    
    # between corank 1 and 2
    hd.add_edge('Vf1f', 'Vf1fPlin', info='')
    hd.add_edge('Vf1f', 'Vf1fAPsat', info='')
        
    hd.add_edge('Vf1APsat', 'Vf1fAPsat', info='Vr->0')
    hd.add_edge('Vf1APsat', 'Vf1PlinAsat', info='Vf->0,r->inf')
    hd.add_edge('Vf1APsat', 'rAlinPsat', info='r->inf')
    
    hd.add_edge('Vf1Plin', 'rAPlin', info='KA->inf')
    hd.add_edge('Vf1Plin', 'Vf1PlinAsat', info='kr->inf,KA->0')
    hd.add_edge('Vf1Plin', 'Vf1fPlin', info='kr->0')
    
    hd.add_edge('rAlin', 'rAPlin')
    hd.add_edge('rAlin', 'rAlinPsat')
    
    # between corank 2 and 3
    hd.add_edge('rAPlin', '0')
    hd.add_edge('rAPlin', 'infr', style='dashed')
    hd.add_edge('rAlinPsat', '0')
    hd.add_edge('rAlinPsat', 'infr', style='dashed')
    hd.add_edge('Vf1PlinAsat', '1')
    hd.add_edge('Vf1PlinAsat', 'infr', style='dashed')
    hd.add_edge('Vf1fPlin', '0')
    hd.add_edge('Vf1fPlin', '1')
    hd.add_edge('Vf1fAPsat', '0')
    hd.add_edge('Vf1fAPsat', '1')
    

    texmap = OD([('Vfp',"V_f"), ('Vrp',"V_r"),
                 (' r*','\\\\rho '), ('*r*','\\\\rho '), 
                 ('Vf','V_f'), ('Vr','V_r'), ('KA','K_A'), ('KP','K_P'), 
                 ('KE','K_E'), ('kf','k_f'), ('kr','k_r'), 
                 ('inf','\\\\infty '), ('*','')])
    
    def _str2tex(s):  
        def _repl(s):
            return s.replace('A/KA', '\\\\frac{A}{KA}').\
                replace('P/KP', '\\\\frac{P}{KP}').\
                replace('(P/A)', '\\\\frac{P}{A}').\
                replace('P/A', '\\\\frac{P}{A}').\
                replace('A/P', '\\\\frac{A}{P}')
                
        if ' / ' in s:  # central /
            n, d = s.split(' / ')
            tex = '\\\\frac{%s}{%s}' % (_repl(n).strip('()'), _repl(d).strip('()'))
        else:
            tex = _repl(s)
        for k, v in texmap.items():  # sequential replacement so order matters
            tex = tex.replace(k, v)
        return '\\\\displaystyle ' + tex
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        # Not using str2tex which uses SloppyCell's exprmanip as it disrupts
        # the expression structure (like moving a denominator in the numerator
        # to the denominator)
        # Not much work (only two ratelaw hds would be shown the formula) 
        # so an ad-hoc solution suffices here
        #texstr = str2tex(s, texmap)  
        texstr = _str2tex(s)
        hd.get_node(nodeid)['texstr'] = texstr
        
    return hd


def get_hasse_mm11Vf1_kb():
    """Undone yet...
    """
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl3 = _get_rl('(1 * A - kr * P) / (1 + bA*A + bP*P)', 'mm11kf1') 
    
    rl2s = [_get_rl('A / (1 + bA*A + bP*P)', 'kf1f'),
            _get_rl('(1 * A - kr * P) / (1 + bA*A)', 'kf1Plin'),
            _get_rl('(1 * A - kr * P) / (1 + bP*P)', 'kf1Alin'),
            _get_rl('(- Vr*P) / (A + r*P)', 'kf1APsat')]
    
    rl1s = [_get_rl('A / (1 + bA*A)', 'kf1fPlin'),
            _get_rl('A / (1 + bP*P)', 'kf1fAlin'),
            
            ### below undone...
            
            _get_rl('(- Vr*P) / A', 'kf1'),
            
            _get_rl('1 * A / (A + r*P)', 'kf1'),
            
            _get_rl('(1 * A - Vr*P) / A', 'mm11Vf1PlinAsat'),
            _get_rl('-kr * P', 'ma11r'),
            _get_rl('-Vr', 'Vr')]
    
    rl0s = [_get_rl('1', '1'),
            _get_rl('0', '0'),
            _get_rl('-inf', 'infr', style='dashed')]
    
    hd = hasse.HasseDiagram(rank=3)
    
    rls = OD([(3, [rl3]), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
            
    # between corank 0 and 1
    hd.add_edge('mm11Vf1', 'mm11Vf1f', info='Vr->0')
    hd.add_edge('mm11Vf1', 'mm11Vf1APsat', info='KA,KP->0')
    hd.add_edge('mm11Vf1', 'mm11Vf1Plin', info='Vr,KP->inf')
    hd.add_edge('mm11Vf1', 'mm11rAlin', info='KA->inf')
    
    # between corank 1 and 2
    hd.add_edge('mm11Vf1f', 'mm11Vf1fPlin', info='')
    hd.add_edge('mm11Vf1f', 'mm11Vf1fAPsat', info='')
        
    hd.add_edge('mm11Vf1APsat', 'mm11Vf1fAPsat', info='Vr->0')
    hd.add_edge('mm11Vf1APsat', 'mm11Vf1PlinAsat', info='Vf->0,r->inf')
    hd.add_edge('mm11Vf1APsat', 'Vr', info='r->inf')
    
    hd.add_edge('mm11Vf1Plin', 'ma11r', info='KA->inf')
    hd.add_edge('mm11Vf1Plin', 'mm11Vf1PlinAsat', info='kr->inf,KA->0')
    hd.add_edge('mm11Vf1Plin', 'mm11Vf1fPlin', info='kr->0')
    
    hd.add_edge('mm11rAlin', 'ma11r')
    hd.add_edge('mm11rAlin', 'Vr')
    
    # between corank 2 and 3
    hd.add_edge('ma11r', '0')
    hd.add_edge('ma11r', 'infr', style='dashed')
    hd.add_edge('Vr', '0')
    hd.add_edge('Vr', 'infr', style='dashed')
    hd.add_edge('mm11Vf1PlinAsat', '1')
    hd.add_edge('mm11Vf1PlinAsat', 'infr', style='dashed')
    hd.add_edge('mm11Vf1fPlin', '0')
    hd.add_edge('mm11Vf1fPlin', '1')
    hd.add_edge('mm11Vf1fAPsat', '0')
    hd.add_edge('mm11Vf1fAPsat', '1')
    

    texmap = OD([('Vfp',"V_f"), ('Vrp',"V_r"),
                 (' r*','\\\\rho '), ('*r*','\\\\rho '), 
                 ('Vf','V_f'), ('Vr','V_r'), ('KA','K_A'), ('KP','K_P'), 
                 ('KE','K_E'), ('kf','k_f'), ('kr','k_r'), 
                 ('inf','\\\\infty '), ('*','')])
    
    def _str2tex(s):  
        def _repl(s):
            return s.replace('A/KA', '\\\\frac{A}{KA}').\
                replace('P/KP', '\\\\frac{P}{KP}').\
                replace('(P/A)', '\\\\frac{P}{A}').\
                replace('P/A', '\\\\frac{P}{A}').\
                replace('A/P', '\\\\frac{A}{P}')
                
        if ' / ' in s:  # central /
            n, d = s.split(' / ')
            tex = '\\\\frac{%s}{%s}' % (_repl(n).strip('()'), _repl(d).strip('()'))
        else:
            tex = _repl(s)
        for k, v in texmap.items():  # sequential replacement so order matters
            tex = tex.replace(k, v)
        return '\\\\displaystyle ' + tex
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        # Not using str2tex which uses SloppyCell's exprmanip as it disrupts
        # the expression structure (like moving a denominator in the numerator
        # to the denominator)
        # Not much work (only two ratelaw hds would be shown the formula) 
        # so an ad-hoc solution suffices here
        #texstr = str2tex(s, texmap)  
        texstr = _str2tex(s)
        hd.get_node(nodeid)['texstr'] = texstr
        
    return hd


def get_hasse_mm11KA1():
    """
    """
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl3 = _get_rl('(Vf * A/1 - Vr * P/KP) / (1 + A/1 + P/KP)', 'mm11KA1') 
    
    rl2s = [_get_rl('(Vf * A/1) / (1 + A/1 + P/KP)', 'mm11KA1f'),
            _get_rl('(Vf * A - Vr * P) / P', 'mm11AlinPsat'),
            _get_rl('(Vf * A/1 - kr*P) / (1 + A/1)', 'mm11KA1Plin'),
            _get_rl('(- Vr * P/KP) / (1 + A/1 + P/KP)', 'mm11KA1r')]
    
    rl1s = [_get_rl('Vf * A / P', 'Qf'),
            _get_rl('(Vf * A/1) / (1 + A/1)', 'mm11KA1fPlin'),
            _get_rl('-Vr', 'Vr'),
            _get_rl('-kr * P / (1 + A/1)', 'mm11KA1rPlin')]
    
    rl0s = [_get_rl('inf', 'inff', style='dashed'),
            _get_rl('0', '0'),
            _get_rl('-inf', 'infr', style='dashed')]
    
    hd = hasse.HasseDiagram(rank=3)
    
    rls = OD([(3, [rl3]), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
            
    # between corank 0 and 1
    hd.add_edge('mm11KA1', 'mm11KA1f', info='Vr->0')
    hd.add_edge('mm11KA1', 'mm11AlinPsat', info='Vf->inf,KP->0')
    hd.add_edge('mm11KA1', 'mm11KA1Plin', info='Vr,KP->inf')
    hd.add_edge('mm11KA1', 'mm11KA1r', info='Vf->0')
    
    # between corank 1 and 2
    hd.add_edge('mm11KA1f', 'mm11KA1fPlin', info='KP->inf')
    hd.add_edge('mm11KA1f', 'Qf', info='Vf->inf,KP->0')
        
    hd.add_edge('mm11AlinPsat', 'Qf', info='Vr->0')
    hd.add_edge('mm11AlinPsat', 'Vr', info='Vf->0')
    
    hd.add_edge('mm11KA1Plin', 'mm11KA1fPlin', info='kr->0')
    hd.add_edge('mm11KA1Plin', 'mm11KA1rPlin', info='Vf->0')
    
    hd.add_edge('mm11KA1r', 'mm11KA1rPlin', 'Vf,KP->inf')
    hd.add_edge('mm11KA1r', 'Vr', info='KP->0')
    
    # between corank 2 and 3
    hd.add_edge('mm11KA1fPlin', '0')
    hd.add_edge('mm11KA1fPlin', 'inff', style='dashed')
    hd.add_edge('Qf', '0')
    hd.add_edge('Qf', 'inff', style='dashed')
    hd.add_edge('mm11KA1rPlin', '0')
    hd.add_edge('mm11KA1rPlin', 'infr', style='dashed')
    hd.add_edge('mm11KA1rPlin', '0')
    hd.add_edge('mm11KA1rPlin', 'infr', style='dashed')
    hd.add_edge('Vr', '0')
    hd.add_edge('Vr', 'infr', style='dashed')
    

    texmap = OD([('Vfp',"V_f"), ('Vrp',"V_r"),
                 (' r*','\\\\rho '), ('*r*','\\\\rho '), 
                 ('Vf','V_f'), ('Vr','V_r'), ('KA','K_A'), ('KP','K_P'), 
                 ('KE','K_E'), ('kf','k_f'), ('kr','k_r'), 
                 ('inf','\\\\infty '), ('*','')])
    
    def _str2tex(s):  
        def _repl(s):
            return s.replace('A/KA', '\\\\frac{A}{KA}').\
                replace('P/KP', '\\\\frac{P}{KP}').\
                replace('(P/A)', '\\\\frac{P}{A}').\
                replace('P/A', '\\\\frac{P}{A}').\
                replace('A/P', '\\\\frac{A}{P}').\
                replace('A/1', '\\\\frac{A}{1}')
                
        if ' / ' in s:  # central /
            n, d = s.split(' / ')
            tex = '\\\\frac{%s}{%s}' % (_repl(n).strip('()'), _repl(d).strip('()'))
        else:
            tex = _repl(s)
        for k, v in texmap.items():  # sequential replacement so order matters
            tex = tex.replace(k, v)
        return '\\\\displaystyle ' + tex
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        # Not using str2tex which uses SloppyCell's exprmanip as it disrupts
        # the expression structure (like moving a denominator in the numerator
        # to the denominator)
        # Not much work (only two ratelaw hds would be shown the formula) 
        # so an ad-hoc solution suffices here
        #texstr = str2tex(s, texmap)  
        texstr = _str2tex(s)
        hd.get_node(nodeid)['texstr'] = texstr
        
    return hd


def get_hasse_mmh11():
    
    #rl = RateLaw(s='kf * (A - P/KE) / (1 + bA*A + bP*P)', **kwargs)  # mmke11
    #rl_a1 = RateLaw(s='kf * (A - P/KE) / (1 + bP*P)', **kwargs)  # mmke11bA0
    #rl_a2 = RateLaw(s='kf * (A - P/KE) / (1 + bA*A)', **kwargs)  # mmke11bP0
    
    # just realized that:
    # - k and b are not independent, but linked through K
    # - V and K are much easier to interpret
    # - When bP->0, it means KP and Vr -> inf and it should be kr * KE
    
    # Commenting out pids=['KE']: see the comment under RateLaw.__init__ added
    # on 2016-08-21
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl3 = _get_rl('Vf/KA * (A - P/KE) / (1 + A/KA + P/KP)', info='mmh11')
    
    rl2s = [_get_rl('Vf/KA * (A - P/KE) / (1 + A/KA)', 'Plin'),  # mmh11VrKPinf
            _get_rl('Vf * (A - P/KE) / (A + rK*P)', 'APsat'),  # mmh11KAKP0
            _get_rl('kf * (A - P/KE) / (1 + P/KP)', 'Alin')]  # mmh11VfKAinf
            
    rl1s = [_get_rl('Vf * (A - P/KE) / A', 'PlinAsat'),  # mmh11KA0
            _get_rl('kf * (A - P/KE)', 'APlin'),  # mah11 
            _get_rl('Vfp * (A - P/KE) / P', 'AlinPsat')]  # Vfp: Vf'=Vr*KE 
    
    rl0s = [#RateLaw(s='0', xids=['A','P']),  # 110
            _get_rl('0', '0'),
            _get_rl('inf * (A - P/KE)', 'infAPlin', style='dashed')]
    
    info = dict(kf='Vf/KA', kr='Vr/KP', rK='KP/KA', Vfp='Vr*KE',
                KE='(Vf/KA)/(Vr/KP)')
    
    hd = hasse.HasseDiagram(rank=3, info=info)
    
    rank2rls = OD([(3, [rl3]), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rank2rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
    
    hd.add_edge('mmh11', 'Plin', info='Vr,KP->inf')
    hd.add_edge('mmh11', 'APsat', info='KA,KP->0')
    hd.add_edge('mmh11', 'Alin', info='Vf,KA->inf')
    hd.add_edge('Plin', 'PlinAsat')
    hd.add_edge('Plin', 'APlin')
    hd.add_edge('APsat', 'PlinAsat')
    hd.add_edge('APsat', 'AlinPsat')
    hd.add_edge('Alin', 'AlinPsat')
    hd.add_edge('Alin', 'APlin')
    hd.add_edge('PlinAsat', '0')
    hd.add_edge('PlinAsat', 'infAPlin', style='dashed')
    hd.add_edge('APlin', '0')
    hd.add_edge('APlin', 'infAPlin', style='dashed')
    hd.add_edge('AlinPsat', '0')
    hd.add_edge('AlinPsat', 'infAPlin', style='dashed')
    
    texmap = OD([('Vfp',"V_f'"), ('Vrp',"V_r'"), ('rK','\\\\rho '), 
                 ('Vf','V_f'), ('Vr','V_r'), ('KA','K_A'), ('KP','K_P'), 
                 ('KE','K_E'), ('kf','k_f'), ('kr','k_r'), 
                 ('inf','\\\\infty')])
    
    def _str2tex(s):
        def _repl(s):
            return s.replace('A/KA', '\\\\frac{A}{KA}').\
                replace('P/KP', '\\\\frac{P}{KP}').\
                replace('Vf/KA', '\\\\frac{Vf}{KA}').\
                replace('P/KE', '\\\\frac{P}{KE}').\
                replace('*', '')
                
        if ' / ' in s:  # central /
            n, d = s.split(' / ')
            tex = '\\\\frac{%s}{%s}' % (_repl(n), _repl(d).strip('()'))
        else:
            tex = _repl(s)
        for k, v in texmap.items():
            tex = tex.replace(k, v)
        return '\\\\displaystyle ' + tex
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        #texstr = str2tex(s, texmap)
        texstr = _str2tex(s)
        hd.get_node(nodeid)['texstr'] = texstr
    
    return hd


def get_hasse_mmh11_kb():
    
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=['A','P']), style
    
    rl3 = _get_rl('kf * (A - P/KE) / (1 + bA*A + bP*P)', info='mmh11')
    
    rl2s = [_get_rl('kf * (A - P/KE) / (1 + bP*P)', 'Alin'),
            _get_rl('kf * (A - P/KE) / (1 + bA*A)', 'Plin'),
            _get_rl('kf * (A - P/KE) / (A + rK*P)', 'APsat')]
            
    rl1s = [_get_rl('Vf * (A - P/KE) / A', 'PlinAsat'),
            _get_rl('kf * (A - P/KE)', 'APlin'),
            _get_rl('Vf * (A - P/KE) / P', 'AlinPsat')]
    
    rl0s = [_get_rl('0', '0'),
            _get_rl('inf * (A - P/KE)', 'inf', style='dashed')]
    
    hd = hasse.HasseDiagram(rank=3)
    
    rank2rls = OD([(3, [rl3]), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rank2rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
    
    hd.add_edge('mmh11', 'Plin')
    hd.add_edge('mmh11', 'APsat')
    hd.add_edge('mmh11', 'Alin')
    hd.add_edge('Plin', 'PlinAsat')
    hd.add_edge('Plin', 'APlin')
    hd.add_edge('APsat', 'PlinAsat')
    hd.add_edge('APsat', 'AlinPsat')
    hd.add_edge('Alin', 'AlinPsat')
    hd.add_edge('Alin', 'APlin')
    hd.add_edge('PlinAsat', '0')
    hd.add_edge('PlinAsat', 'inf', style='dashed')
    hd.add_edge('APlin', '0')
    hd.add_edge('APlin', 'inf', style='dashed')
    hd.add_edge('AlinPsat', '0')
    hd.add_edge('AlinPsat', 'inf', style='dashed')
    
    return hd


"""
func_nodeattr = lambda nodeattrs: {'ratelaws': 
                                   tuple([d['ratelaw'] for d in nodeattrs])}
    
hd2 = hd_2mmh11

for nodeid in hd2.nodeids:
    hd2.node[nodeid]['style'] = 'filled'

def feq_mmh11(nodeid):
    v1, v2 = nodeid
    xmap = {'A':'P', 'P':'A', '':''}
    opmap = {'lin':'lin', 'sat':'sat', '':''}
    def _flip(vid):
        if vid in ['mmh11', 'APsat', 'APlin', '0', 'infAPlin']:
            return vid
        elif len(vid) == 4:
            xid, op = vid[0], vid[1:4]
            return xmap[xid] + opmap[op]
        elif len(vid) == 8:
            xid1, op1, xid2, op2 = vid[0], vid[1:4], vid[4], vid[5:]
            return xmap[xid1] + opmap[op1] + xmap[xid2] + opmap[op2]
        else:
            pass
    return [(_flip(v2), _flip(v1))]
"""


"""
hd.order[4] = [('b1', 'mmke11'),
               ('b2', 'mmke11'),
               ('b3', 'mmke11'), 
               ('a1', 'a2'), #
               ('a2', 'a1'), #
               ('a1', 'a1'), #

               [('a2', 'a2'),
                ('a3', 'a1'),  
                ('a1', 'a3')],  

               ('a3', 'a3'), #
               ('a3', 'a2'), #
               ('a2', 'a3'), #
               ('mmke11', 'b1'),
               ('mmke11', 'b2'),
               ('mmke11', 'b3')]     

hd.order[3] = [('c1', 'mmke11'),
               ('c2', 'mmke11'),
               ('b1', 'a2'), #
               ('b2', 'a3'), #

               ('a1', 'b1'), #
               ('b1', 'a1'), #
               
               ('b3', 'a2'), #
               ('b2', 'a1'), #
               
               ('b1', 'a3'),
               ('b3', 'a1'),
               
               ('b2', 'a2'),
               ('a2', 'b2'),
               
               ('a3', 'b1'),
               ('a1', 'b3'),
               
               
               ('a3', 'b2'), #
               ('a2', 'b1'), #
               
               ('a3', 'b3'), #
               ('b3', 'a3'), #

               
               ('a1', 'b2'), #
               ('a2', 'b3'), #
               
               ('mmke11', 'c2'),
               ('mmke11', 'c1')
               ]

hd.order[2] = [('c1', 'a1'),
               ('c1', 'a2'),
               ('c1', 'a3'),
               ('c2', 'a1'),
               ('c2', 'a2'),
               ('c2', 'a3'),
               
               ('b1', 'b1'),
               ('b1', 'b2'),
               ('b2', 'b1'),
               
               [('b3', 'b1'),
                ('b2', 'b2'),
                ('b1', 'b3')],
               
               ('b3', 'b2'),
               ('b2', 'b3'),
               ('b3', 'b3'),
               
               ('a1', 'c2'),
               ('a2', 'c2'),
               ('a3', 'c2'),
               ('a1', 'c1'),
               ('a2', 'c1'),
               ('a3', 'c1')
               ]

hd.order[1] = [('c1', 'b1'),
               ('c1', 'b2'),
               ('c1', 'b3'),
               
               ('c2', 'b1'),
               ('c2', 'b2'),
               ('c2', 'b3'),
               
               ('b1', 'c2'),
               ('b2', 'c2'),               
               ('b3', 'c2'),
               
               ('b1', 'c1'),
               ('b2', 'c1'),
               ('b3', 'c1'),
               ]

hd.order[0] = [('c1', 'c2'), 
               [('c1', 'c1'), 
                ('c2', 'c2')],
               ('c2', 'c1')]

"""

# All the eqclss info below doesn't make general sense as it depends on data;
# should move it elsewhere

"""
hd_2mmke11.eqclss[6] = [(('mmke11', 'mmke11'),)]

hd_2mmke11.eqclss[5] = [(('a1', 'mmke11'), ('mmke11', 'a3')),
                        (('a2', 'mmke11'), ('mmke11', 'a2')),
                        (('a3', 'mmke11'), ('mmke11', 'a1'))]

hd_2mmke11.eqclss[4] = [(('b1', 'mmke11'), ('mmke11', 'b3')),
                        (('b2', 'mmke11'), ('mmke11', 'b2')),
                        (('b3', 'mmke11'), ('mmke11', 'b1')), 
                        (('a1', 'a2'), ('a2', 'a3')),
                        (('a2', 'a1'), ('a3', 'a2')),
                        (('a1', 'a1'), ('a3', 'a3')),
                        (('a2', 'a2'),),
                        (('a3', 'a1'),),  
                        (('a1', 'a3'),)]

hd_2mmke11.eqclss[3] = [(('c1', 'mmke11'), ('mmke11', 'c1')),
                        (('c2', 'mmke11'), ('mmke11', 'c2')),
                        (('b1', 'a2'), ('a2', 'b3')),
                        (('b2', 'a3'), ('a1', 'b2')),
                        (('a1', 'b1'), ('b3', 'a3')),
                        (('b1', 'a1'), ('a3', 'b3')),
                        (('b3', 'a2'), ('a2', 'b1')),
                        (('b2', 'a1'), ('a3', 'b2')),
                        (('b1', 'a3'), ('a1', 'b3')),
                        (('b3', 'a1'), ('a3', 'b1')),
                        (('b2', 'a2'), ('a2', 'b2'))]

hd_2mmke11.eqclss[2] = [(('c1', 'a1'), ('a3', 'c1')),
                        (('c1', 'a2'), ('a2', 'c1')),
                        (('c1', 'a3'), ('a1', 'c1')),
                        (('c2', 'a1'), ('a3', 'c2')),
                        (('c2', 'a2'), ('a2', 'c2')),
                        (('c2', 'a3'), ('a1', 'c2')),
                        (('b1', 'b1'), ('b3', 'b3')),
                        (('b1', 'b2'), ('b2', 'b3')),
                        (('b2', 'b1'), ('b3', 'b2')),
                        (('b3', 'b1'),),
                        (('b2', 'b2'),),
                        (('b1', 'b3'),)]

hd_2mmke11.eqclss[1] = [(('c1', 'b1'), ('b3', 'c1')),
                        (('c1', 'b2'), ('b2', 'c1')),
                        (('c1', 'b3'), ('b1', 'c1')),
                        (('c2', 'b1'), ('b3', 'c2')),
                        (('c2', 'b2'), ('b2', 'c2')), 
                        (('c2', 'b3'), ('b1', 'c2'))]

hd_2mmke11.eqclss[0] = [(('c1', 'c2'), ('c2', 'c1')), 
                        (('c1', 'c1'),),
                        (('c2', 'c2'),)]

"""


    
"""    
def eqn2stoich(eqn):
    
    Convert reaction equation (a string) to stoichiometry (a dictionary).

    def unpack(s):
        # an example of s: ' 2 ATP '
        l = filter(None, s.split(' '))
        if len(l) == 1:
            # sc: stoichcoef
            sc_unsigned, spid = '1', l[0]
        if len(l) == 2:
            sc_unsigned, spid = l
        sc_unsigned = int(sc_unsigned)
        return spid, sc_unsigned
    
    # remove annotating species
    # eg, '(ST+)P->(ST+)G1P', where 'ST' (starch) is kept there to better
    # represent the chemistry
    eqn = re.sub('\(.*?\)', '', eqn)
    
    # re: '<?': 0 or 1 '<'; '[-|=]': '-' or '=' 
    subs, pros = re.split('<?[-|=]>', eqn)
    stoich = OD()
    
    if subs:
        for sub in subs.split('+'):
            subid, sc_unsigned = unpack(sub)
            stoich[subid] = -1 * sc_unsigned
    if pros:
        for pro in pros.split('+'):
            proid, sc_unsigned = unpack(pro)
            if proid in stoich:
                stoich[proid] = stoich[proid] + sc_unsigned
            else:
                stoich[proid] = sc_unsigned
        
    return stoich


def get_substrates(stoich_or_eqn, multi=False):

    Input: 
        stoich_or_eqn: a mapping, from species ids to stoich coefs which can be
                an int, a float, or a string; or a str
        multi: a bool; if True, return a multiset by repeating
            for stoichcoef times
    
    Output:
        a list of substrate ids

    if isinstance(stoich_or_eqn, str):
        eqn = stoich_or_eqn
        stoich = eqn2stoich(eqn)
    else:
        stoich = stoich_or_eqn
    subids = []
    for spid, stoichcoef in stoich.items():
        try:
            stoichcoef = int(float(stoichcoef))
        except ValueError:
            stoichcoef = int(stoichcoef.lstrip()[0]+'1')
        if stoichcoef < 0:
            if multi:
                subids.extend([spid]*(-stoichcoef))
            else:
                subids.append(spid)
    return subids


def get_products(stoich_or_eqn, multi=False):
    
    Input: 
        stoich_or_eqn: a mapping, from species ids to stoich coefs which can be
                an int, a float, or a string; or a str
        multi: a bool; if True, return a multiset by repeating
            for stoichcoef times
    
    Output:
        a list of product ids
    
    if isinstance(stoich_or_eqn, str):
        eqn = stoich_or_eqn
        stoich = eqn2stoich(eqn)
    else:
        stoich = stoich_or_eqn
    proids = []
    for spid, stoichcoef in stoich.items():
        try:
            stoichcoef = int(float(stoichcoef))
        except ValueError:
            stoichcoef = 1
        if stoichcoef > 0:
            if multi:
                proids.extend([spid]*stoichcoef)
            else:
                proids.append(spid)
    return proids



def get_reactants(stoich_or_eqn, multi=False):
    return get_substrates(stoich_or_eqn, multi=multi) +\
        get_products(stoich_or_eqn, multi=multi)

"""



def get_hasse_mmh21():
    """
    - Plot part of the nodes by first create a subdiagram
    """
    
    #rl = RateLaw(s='kf * (A - P/KE) / (1 + bA*A + bP*P)', **kwargs)  # mmke11
    #rl_a1 = RateLaw(s='kf * (A - P/KE) / (1 + bP*P)', **kwargs)  # mmke11bA0
    #rl_a2 = RateLaw(s='kf * (A - P/KE) / (1 + bA*A)', **kwargs)  # mmke11bP0
    
    # just realized that:
    # - k and b are not independent, but linked through K
    # - V and K are much easier to interpret
    # - When bP->0, it means KP and Vr -> inf and it should be kr * KE
    
    _get_rl = lambda s, info: RateLaw(s=s, info=info, xids=['A','B','P'])
    
    rl = _get_rl('Vf/(KA*KB) * (A*B - P/KE) / (1 + A/KA + B/KB + A*B/(KA*KB) + P/KP)', 
                 info='mmke21')
    
    rl1s = [_get_rl('kB/KA * (A*B - P/KE) / (1 + A/KA + P/KP)', 'mmke21Blin'),
            _get_rl('Vf/KB * (A*B - P/KE) / (A + A*B/KB + P*rAP)', 'mmke21APsat'),
            _get_rl('Vf/(KA*KB) * (A*B - P/KE) / (1 + A/KA + B/KB + A*B/(KA*KB))', 'mmke21Plin'), 
            _get_rl('Vf/KA * (A*B - P/KE) / (B + A*B/KA + P*rBP)', 'mmke21BPsat'),
            _get_rl('kA/KB * (A*B - P/KE) / (1 + B/KB + P/KP)', 'mmke21Alin')]
            

    rl2s = [_get_rl('kB/KA * (A*B - P/KE) / (1 + A/KA)', 'mmke21BPlin'),
            _get_rl('Vf/KB * (A*B - P/KE) / (A + A*B/KB)', 'mmke21PlinAsat'),
            _get_rl('kf * (A*B - P/KE) / (A + P*rAP)', 'mmke21BlinAPsat'),
            
            _get_rl('kf * (A*B - P/KE) / (1 + P/KP)', 'mmke21ABlin'),
            _get_rl('Vf * (A*B - P/KE) / (A*B + P*rABP)', 'mmke21ABPsat'),
                
            _get_rl('kf * (A*B - P/KE) / (B + P*rBP)', 'mmke21AlinBPsat'),
            _get_rl('Vf/KA * (A*B - P/KE) / (B + A*B/KA)', 'mmke21PlinBsat'),
            _get_rl('kA/KB * (A*B - P/KE) / (1 + B/KB)', 'mmke21APlin')]

    
    rl3s = [_get_rl('kB * (A*B - P/KE) / A', 'mmke21BPlinAsat'),
    
            _get_rl('kf * (A*B - P/KE)', 'make21'),
            _get_rl('k * (A*B - P/KE) / P', 'mmke21ABlinPsat'),
            _get_rl('Vf * (A*B - P/KE) / (A*B)', 'mmke21PlinABsat'),
            
                        
            _get_rl('kA * (A*B - P/KE) / B', 'mmke21APlinBsat')]


    rl4s = [_get_rl('0', '0'),
            _get_rl('inf * (A*B - P/KE)', 'inf')]

        
    info = dict(kf='Vf/(KA*KB)', kA='Vf/KA', kB='Vf/KB', 
                kr='', rK='', Vfp='',
                KE='')
    
    hd = hasse.HasseDiagram(rank=4, info=info)
    rls = OD([(0, [rl]), (1, rl1s), (2, rl2s), (3, rl3s), (4, rl4s)])
    for corank, rls_corank in rls.items():
        for rl in rls_corank:
            hd.add_node(rl.info, corank=corank, ratelaw=rl)
    
    """
    for idx, rl1 in enumerate(rl1s):
        nodeid = 'a%d' % (idx+1)
        hd.add_node(nodeid, corank=1, ratelaw=rl1)
        
    for idx, rl2 in enumerate(rl2s):
        nodeid = 'b%d' % (idx+1)
        hd.add_node(nodeid, corank=2, ratelaw=rl2)
        
    for idx, rl3 in enumerate(rl3s):
        nodeid = 'c%d' % (idx+1)
        hd.add_node(nodeid, corank=3, ratelaw=rl3)

    for idx, rl4 in enumerate(rl4s):
        nodeid = 'd%d' % (idx+1)
        hd.add_node(nodeid, corank=4, ratelaw=rl4)
    """
    
    hd.add_edge('mmke21', 'mmke21Alin', info='')
    hd.add_edge('mmke21', 'mmke21Blin', info='')
    hd.add_edge('mmke21', 'mmke21Plin', info='')
    hd.add_edge('mmke21', 'mmke21APsat', info='')
    hd.add_edge('mmke21', 'mmke21BPsat', info='')
    
    ############################################################################
    
    hd.add_edge('mmke21Alin', 'mmke21ABlin')
    hd.add_edge('mmke21Alin', 'mmke21APlin')
    hd.add_edge('mmke21Alin', 'mmke21AlinBPsat')
    
    hd.add_edge('mmke21Blin', 'mmke21BPlin')
    hd.add_edge('mmke21Blin', 'mmke21BlinAPsat')
    hd.add_edge('mmke21Blin', 'mmke21ABlin')
    
    hd.add_edge('mmke21Plin', 'mmke21APlin')
    hd.add_edge('mmke21Plin', 'mmke21BPlin')
    hd.add_edge('mmke21Plin', 'mmke21PlinAsat')
    hd.add_edge('mmke21Plin', 'mmke21PlinBsat')
    
    hd.add_edge('mmke21APsat', 'mmke21ABPsat')
    hd.add_edge('mmke21APsat', 'mmke21BlinAPsat')
    
    hd.add_edge('mmke21BPsat', 'mmke21ABPsat')
    hd.add_edge('mmke21BPsat', 'mmke21AlinBPsat')
    
    ############################################################################
    
    hd.add_edge('mmke21ABlin', 'make21')
    hd.add_edge('mmke21ABlin', 'mmke21ABlinPsat')
    
    hd.add_edge('mmke21APlin', 'make21')
    hd.add_edge('mmke21APlin', 'mmke21APlinBsat')
    
    hd.add_edge('mmke21BPlin', 'make21')
    hd.add_edge('mmke21BPlin', 'mmke21BPlinAsat')
    
    hd.add_edge('mmke21AlinBPsat', 'mmke21ABlinPsat')
    hd.add_edge('mmke21AlinBPsat', 'mmke21APlinBsat')
    
    hd.add_edge('mmke21BlinAPsat', 'mmke21ABlinPsat')
    hd.add_edge('mmke21BlinAPsat', 'mmke21BPlinAsat')
    
    hd.add_edge('mmke21PlinAsat', 'mmke21PlinABsat')
    hd.add_edge('mmke21PlinAsat', 'mmke21BPlinAsat')
    
    hd.add_edge('mmke21PlinBsat', 'mmke21PlinABsat')
    hd.add_edge('mmke21PlinBsat', 'mmke21APlinBsat')
    
    hd.add_edge('mmke21ABPsat', 'mmke21PlinABsat')
    hd.add_edge('mmke21ABPsat', 'mmke21ABlinPsat')
    
    ############################################################################
    
    hd.add_edge('make21', 'inf')
    hd.add_edge('make21', '0')
    
    hd.add_edge('mmke21ABlinPsat', 'inf')
    hd.add_edge('mmke21ABlinPsat', '0')
    hd.add_edge('mmke21APlinBsat', 'inf')
    hd.add_edge('mmke21APlinBsat', '0')
    hd.add_edge('mmke21BPlinAsat', 'inf')
    hd.add_edge('mmke21BPlinAsat', '0')
    hd.add_edge('mmke21PlinABsat', 'inf')
    hd.add_edge('mmke21PlinABsat', '0')

    
    
    #hd.add_edge('mmke21ABlin', 'mmke21')
    #hd.add_edge('mmke21', 'mmke21')

    
    texmap = dict(Vf='V_f', Vr='V_r', KA='K_A', KB='K_B', KP='K_P', KE='K_E',  
                  kf='k_f', kA='k_A', kB='k_B', 
                  rAP='\\\\rho_{_{AP}}', rBP='\\\\rho_{_{BP}}', rABP='\\\\rho_{_{ABP}}',
                  rK='\\\\rho', inf='\\\\infty',
                  Vfp="V_f'", Vrp="V_r'", **{'->':'\\\\rightarrow'})

    _str2texstr = lambda s: exprmanip.expr2TeX(s, texmap).\
        replace('\\cdot ', '').replace('\\frac', '\\\\frac') .\
        replace('\\left', '\\\\left').replace('\\right', '\\\\right')
        
    _str2ast = lambda s: exprmanip.AST.strip_parse(s)
    _ast2str = lambda a: exprmanip.ast2str(a)
    
    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        ast = _str2ast(s)
        
        # The code block below is used to keep the fractional structure in
        # latex: eg, (V/K)/(1+X/K) would be changed to V/(K(1+X/K)) by default.
        if isinstance(ast, exprmanip.AST.Div):
            if isinstance(ast.left, exprmanip.AST.Mul):
                numtexstr = '%s \\\\left( %s \\\\right)' %\
                    (_str2texstr(_ast2str(ast.left.left)), 
                     _str2texstr(_ast2str(ast.left.right)))
            else:
                numtexstr = _str2texstr(_ast2str(ast.left))
            denomtexstr = _str2texstr(_ast2str(ast.right))
            texstr = '\\\\displaystyle \\\\frac{%s}{%s}' % (numtexstr, denomtexstr) 
        else:
            texstr =  '\\\\displaystyle %s' % _str2texstr(s)
            
        # remove parentheses in some denominators
        if '}{\\\\left(' in texstr:  
            texstr = re.sub(r'}{\\\\left\(', '}{', texstr)
            texstr = re.sub(r'\\\\right\)}$', '}', texstr)
        hd.get_node(nodeid)['texstr'] = texstr
    
    # manually change the look of 
    #hd.get_node(rl.info)['texstr'] =\
    #    '\\\\displaystyle \\\\frac{\\\\frac{V_f}{K_A} \\\\left(A - \\\\frac{P}{K_E}\\\\right)}{1 + \\\\frac{A}{K_A} + \\\\frac{P}{K_P}}'
    
    return hd


def get_hasse_mm21():
    """
    """
    _get_rl = lambda s, info: RateLaw(s=s, info=info, xids=['A','B','P'])
    
    rl5 = _get_rl('(Vf*(A/KA)*(B/KB) - Vr*(P/KP)) / (1 + A/KA + B/KB + (A/KA)*(B/KB) + P/KP)', 'mm21')
    
    rl4s = [_get_rl('Vf*(A/KA)*(B/KB) / (1 + A/KA + B/KB + (A/KA)*(B/KB) + P/KP)', 'mm21f'),
            _get_rl('(kB*B*(A/KA) - Vr*(P/KP)) / (1 + A/KA + P/KP)', 'mm21Blin'),
            _get_rl('(Vf*B/KB - Vr*(P/A)*rAP) / (1 + B/KB + (P/A)*rAP)', 'mm21APsat'),
            _get_rl('(Vf*(A/KA)*(B/KB) - kr*P) / (1 + A/KA + B/KB + (A/KA)*(B/KB))', 'mm21Plin'),
            _get_rl('(Vf*A/KA - Vr*(P/B)*rBP) / (1 + A/KA + (P/B)*rBP)', 'mm21BPsat'),
            _get_rl('(kA*A*(B/KB) - Vr*(P/KP)) / (1 + B/KB + P/KP)', 'mm21Alin'),
            _get_rl('- Vr*(P/KP) / (1 + A/KA + B/KB + (A/KA)*(B/KB) + P/KP)', 'mm21r')]
            
            
    rl3s = [_get_rl('kB*B*(A/KA) / (1 + A/KA + P/KP)', 'mm21fBlin'),
            _get_rl('(kB*B*(A/KA) - kr*P) / (1 + A/KA)', 'mm21BPlin'),
            _get_rl('(Vfp*A*B - Vr*(P*rAP)) / (A + P*rAP)', 'mm21BlinAPsat'),
            
            _get_rl('Vf*(A*B/KB) / (A + A*B/KB + P*rAP)', 'mm21fAPsat'),
            _get_rl('- Vr*(P/KP) / (1 + A/KA + P/KP)', 'mm21rBlin'),
            _get_rl('- Vr*(P*rAP) / (A + A*B/KB + P*rAP)', 'mm21rAPsat'),
            _get_rl('(Vf*(A*B/KB) - Vr*P) / (A + A*B/KB)', 'mm21PlinAsat'),
            
            _get_rl('Vf*(A/KA)*(B/KB) / (1 + A/KA + B/KB + (A/KA)*(B/KB))', 'mm21fPlin'),
            _get_rl('(kf*A*B - Vr*(P/KP)) / (1 + P/KP)', 'mm21ABlin'),
            _get_rl('(Vf*A*B - Vr*(P*rABP)) / (A*B + P*rABP)', 'mm21ABPsat'),
            _get_rl('- kr*P / (1 + A/KA + B/KB + (A/KA)*(B/KB))', 'mm21rPlin'),
            
            _get_rl('(Vf*(A*B/KA) - Vr*P) / (B + A*B/KA)', 'mm21PlinBsat'),
            _get_rl('- Vr*(P*rBP) / (B + A*B/KA + P*rBP)', 'mm21rBPsat'),
            _get_rl('- Vr*(P/KP) / (1 + B/KB + P/KP)', 'mm21rAlin'),
            _get_rl('Vf*(A*B/KA) / (B + A*B/KA + P*rBP)', 'mm21fBPsat'),
            
            _get_rl('(Vfp*A*B - Vr*(P*rBP)) / (B + P*rBP)', 'mm21AlinBPsat'),
            _get_rl('(kA*A*(B/KB) - kr*P) / (1 + B/KB)', 'mm21APlin'),
            _get_rl('kA*A*(B/KB) / (1 + B/KB + P/KP)', 'mm21fAlin')]
    
    
    rl2s = [_get_rl('- Vr*(P/A)*rAP / (1 + (P/A)*rAP)', 'mm21rBlinAPsat'),
            _get_rl('- Vr*(P/B)*rBP / (1 + (P/B)*rBP)', 'mm21rAlinBPsat'),
            _get_rl('kB*B / (1 + (P/A)*rAP)', 'mm21fBlinAPsat'),
            _get_rl('kA*A / (1 + (P/B)*rBP)', 'mm21fAlinBPsat'),
            
            _get_rl('kf*A*B / (1 + P/KP)', 'mm21fABlin'),
            _get_rl('- kr*P / (1 + P/KP)', 'mm21rABlin'),
            _get_rl('(Vfp * A * B/KB) / (1 + B/KB)', 'mm21fAPlin'),
            _get_rl('(- kr * P) / (1 + B/KB)', 'mm21rAPlin'),
            _get_rl('(Vfp * B * A/KA) / (1 + A/KA)', 'mm21fBPlin'),
            _get_rl('(- kr * P) / (1 + A/KA)', 'mm21rBPlin'),
            _get_rl('kf * A*B - kr * P', 'mm21ABPlin'),
            
            _get_rl('(Vfp * A*B - Vrp * P) / A', 'mm21BPlinAsat'),
            _get_rl('(Vfp * A*B - Vrp * P) / P', 'mm21ABlinPsat'),
            _get_rl('(Vfp * A*B - Vrp * P) / B', 'mm21APlinBsat'),

            _get_rl('Vf * B/KB / (1 + B/KB)', 'mm21Asat'),
            _get_rl('Vf * A/KA / (1 + A/KA)', 'mm21Bsat'),
            
            _get_rl('(Vf*A*B) / (A*B + P*rABP)', 'mm21fABPsat'),
            _get_rl('(- Vr*(P*rABP)) / (A*B + P*rABP)', 'mm21rABPsat')]
    
    
    rl1s = [_get_rl('Vf', 'mm21ABsat'),
            _get_rl('kf * A*B', 'mm21fABPlin'),
            _get_rl('- kr * P', 'mm21rABPlin'),
            _get_rl('- Vr', 'mm21Psat'),
            _get_rl('kB * B', 'mm21BlinAsat'),
            _get_rl('kA * A', 'mm21AlinBsat'),
            _get_rl('kf * A*B / P', 'mm21fABlinPsat'),
            _get_rl('- kr * P / (A*B)', 'mm21rPlinABsat'),
            _get_rl('inf * (A*B - P/KE)', 'mm21ABPlininf')]  # dashed
    
    
    rl0s = [_get_rl('inf', 'inf'),  # dashed
            _get_rl('0', '0'),
            _get_rl('-inf', '-inf')]  # dashed
    
     
    info = dict(kf='Vf/(KA*KB)', kA='Vf/KA', kB='Vf/KB', 
                kr='', rK='', Vfp='',
                KE='')
    
    hd = hasse.HasseDiagram(rank=5, info=info)
    rls = OD([(5, [rl5]), (4, rl4s), (3, rl3s), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl)
    
    """
    def _has_edge_is_singular(rl1, rl2):
        Return two bools: 
            - is there an edge between rl1 and rl2
            - is the edge singular
        
        pdim1, pdim2 = rl1.pdim, rl2.pdim
        info1, info2 = rl1.info.lstrip('mm21'), rl2.info.lstrip('mm21')
        
        assert pdim1 == pdim2 + 1
        
        if pdim2 == 0:
            if info1[0] == 'f' and info2 == '0':
                return True, False
            if info1[0] == 'f' and info2 == 'inff':
                return True, True
            elif info1[0] == 'r' and info2 =='0':
                return True, False
            elif info1[0] == 'r' and info2 == 'infr':
                return True, True
            elif info1 == 'ABPlininf' and info2 in ['inff', 'infr']:
                return True, True
            else:
                return False, False
        else:
                    
            def _info2limits(info):
                limits = butil.Series(['', set(), set(), False], 
                                      ['dir', 'lin', 'sat', 'inf'])
                if info == '':
                    return limits
                if info[0] in ['f', 'r']:
                    limits.dir = info[0]
                    info = info[1:]
                if info.endswith('inf'):  # specifically for mm11APlininf; any other solution?
                    limits.inf = True
                    info = info.rstrip('inf')
                info = info.replace('lin', 'lin_').replace('sat', 'sat_')
                parts = filter(None, info.split('_'))
                for part in parts:
                    if part.endswith('lin'):
                        limits.lin.update(list(part.rstrip('lin')))
                    if part.endswith('sat'):
                        limits.sat.update(list(part.rstrip('sat')))
                return limits
            
            l1 = _info2limits(info1)
            l2 = _info2limits(info2)
            
            #if info2 == 'APlininf':
            #    import ipdb
            #    ipdb.set_trace()
            
            # unidirectionalization
            if len(l2.dir)==len(l1.dir)+1 and l2.lin==l1.lin and l2.sat==l1.sat:
                return True, False
            # single linearization
            elif l2.dir==l1.dir and len(l2.lin-l1.lin)==1 and l2.sat==l1.sat:
                return True, False
            # double saturation
            elif l2.dir==l1.dir and l2.lin==l1.lin==set() and len(l2.sat-l1.sat)==2:
                return True, False
            # final saturation
            elif l2.dir==l1.dir and l2.lin==l1.lin and len(l2.lin)==len(rl.xids)-1 and len(l2.sat-l1.sat)==1:
                return True, False
            # saturation-turn-linearization
            elif l2.dir==l1.dir and len(l2.lin-l1.lin)==1 and len(l1.sat-l2.sat)==1:
                return True, False
            # infinitization
            elif l1.dir=='' and l2.inf:
                return True, True
            else:
                return False, False
    
    """        
    
    rankpairs = [(4,3), (3,2), (2,1), (1,0)]
    
    """
    
    for rankpair in rankpairs:
        rank_, rank_succ = rankpair
        rls_, rls_succ = rls[rank_], rls[rank_succ]
        for rl_ in rls_:
            for rl_succ in rls_succ:
                if _has_edge_is_singular(rl_, rl_succ)[0]:
                    hd.add_edge(rl_.info, rl_succ.info)
    
    
    hd.add_edge('mm21', 'mm21Alin', info='')
    hd.add_edge('mm21', 'mm21Blin', info='')
    hd.add_edge('mm21', 'mm21Plin', info='')
    hd.add_edge('mm21', 'mm21APsat', info='')
    hd.add_edge('mm21', 'mm21BPsat', info='')
    hd.add_edge('mm21', 'mm21f', info='')
    hd.add_edge('mm21', 'mm21r', info='')
    
    ############################################################################
    
    hd.add_edge('mm21Alin', 'mm21ABlin')
    hd.add_edge('mm21Alin', 'mm21APlin')
    hd.add_edge('mm21Alin', 'mm21AlinBPsat')
    hd.add_edge('mm21Alin', 'mm21fAlin')
    hd.add_edge('mm21Alin', 'mm21rAlin')
    
    hd.add_edge('mm21Blin', 'mm21BPlin')
    hd.add_edge('mm21Blin', 'mm21BlinAPsat')
    hd.add_edge('mm21Blin', 'mm21ABlin')
    hd.add_edge('mm21Blin', 'mm21fBlin')
    hd.add_edge('mm21Blin', 'mm21rBlin')
    
    hd.add_edge('mm21Plin', 'mm21APlin')
    hd.add_edge('mm21Plin', 'mm21BPlin')
    hd.add_edge('mm21Plin', 'mm21PlinAsat')
    hd.add_edge('mm21Plin', 'mm21PlinBsat')
    hd.add_edge('mm21Plin', 'mm21fPlin')
    hd.add_edge('mm21Plin', 'mm21rPlin')
    
    hd.add_edge('mm21APsat', 'mm21ABPsat')
    hd.add_edge('mm21APsat', 'mm21BlinAPsat')
    hd.add_edge('mm21APsat', 'mm21fAPsat')
    hd.add_edge('mm21APsat', 'mm21rAPsat')
    
    hd.add_edge('mm21BPsat', 'mm21ABPsat')
    hd.add_edge('mm21BPsat', 'mm21AlinBPsat')
    hd.add_edge('mm21BPsat', 'mm21fBPsat')
    hd.add_edge('mm21BPsat', 'mm21rBPsat')
    
    hd.add_edge('mm21f', 'mm21fAlin')
    hd.add_edge('mm21f', 'mm21fBlin')
    hd.add_edge('mm21f', 'mm21fPlin')
    hd.add_edge('mm21f', 'mm21fAPsat')
    hd.add_edge('mm21f', 'mm21fBPsat')
    
    hd.add_edge('mm21r', 'mm21rAlin')
    hd.add_edge('mm21r', 'mm21rBlin')
    hd.add_edge('mm21r', 'mm21rPlin')
    hd.add_edge('mm21r', 'mm21rAPsat')
    hd.add_edge('mm21r', 'mm21rBPsat')
    
    ############################################################################
    
    hd.add_edge('mm21fPlin', 'mm21Asat')
    hd.add_edge('mm21fPlin', 'mm21Bsat')
    
    hd.add_edge('mm21BPlin', 'mm21Asat')
    hd.add_edge('mm21BPlin', 'ma21')
    hd.add_edge('mm21BPlin', 'mm21BPlinAsat')
    
    hd.add_edge('mm21BlinAPsat', 'mm21fBlinAPsat')
    hd.add_edge('mm21BlinAPsat', 'mm21rBlinAPsat')
    hd.add_edge('mm21BlinAPsat', 'mm21BPlinAsat')
    hd.add_edge('mm21BlinAPsat', 'mm21ABlinPsat')
    
    
    ############################################################################
    
    hd.add_edge('mm21Asat', 'Vf')
    hd.add_edge('mm21Bsat', 'Vf')
    hd.add_edge('ma21', 'maf2')
    hd.add_edge('ma21', 'mar1')
    
    ############################################################################
    
    hd.add_edge('maf2', 'inf')
    hd.add_edge('maf2', '0')
    hd.add_edge('mar1', '-inf')
    hd.add_edge('mar1', '0')
    hd.add_edge('Vf', 'inf')
    hd.add_edge('Vf', '0')
    hd.add_edge('Vr', '-inf')
    hd.add_edge('Vr', '0')
    
    ############################################################################
    
    #hd.add_edge('mmke21ABlin', 'mmke21')
    #hd.add_edge('mmke21', 'mmke21')

    """
    
    texmap = dict(Vf='V_f', Vr='V_r', KA='K_A', KB='K_B', KP='K_P', KE='K_E',  
                  kf='k_f', kr='k_r', kA='k_A', kB='k_B',
                  rAP='\\\\rho_{_{AP}}', rBP='\\\\rho_{_{BP}}', rABP='\\\\rho_{_{ABP}}',
                  rK='\\\\rho', inf='\\\\infty',
                  Vfp="V_f'", Vrp="V_r'", **{'->':'\\\\rightarrow'})

    for nodeid in hd.nodeids:
        s = hd.get_node(nodeid)['ratelaw'].s
        texstr = str2tex(s, texmap)
        hd.get_node(nodeid)['texstr'] = texstr
        
    # manually change the look of 
    #hd.get_node(rl.info)['texstr'] =\
    #    '\\\\displaystyle \\\\frac{\\\\frac{V_f}{K_A} \\\\left(A - \\\\frac{P}{K_E}\\\\right)}{1 + \\\\frac{A}{K_A} + \\\\frac{P}{K_P}}'
    
    return hd


def get_hasse_mmh22_ABPP():
    """
    """
    def _get_rl(s, info, style='filled'): 
        return RateLaw(s=s, info=info, xids=list('ABP')), style
    
    rl4 = _get_rl('kf*(A*B - P*P/KE) / (1 + bA*A + bB*B + bA*bB*A*B + 2*bP*P + bP**2*P**2)', 
                  info='mmh22ABPP')
    
    rl3s = [_get_rl('kf*(A*B - P*P/KE) / (1 + bB*B + 2*bP*P + bP**2*P**2)', 'Alin'),
            _get_rl('kf*(A*B - P*P/KE) / (1 + bA*A + 2*bP*P + bP**2*P**2)', 'Blin'),
            _get_rl('kf*(A*B - P*P/KE) / (1 + bA*A + bB*B + bA*bB*A*B)', 'Plin'),
            _get_rl('Vf*(A*B - P*P/KE) / (A + bB*A*B + bP*P**2)', 'APsat'), 
            _get_rl('Vf*(A*B - P*P/KE) / (B + bA*A*B + bP*P**2)', 'BPsat')
            ]
            
    rl2s = [_get_rl('kf*(A*B - P*P/KE) / (1 + 2*bP*P + bP**2*P**2)', 'ABlin'),
            _get_rl('kf*(A*B - P*P/KE) / (1 + bB*B)', 'APlin'),
            _get_rl('kf*(A*B - P*P/KE) / (1 + bA*A)', 'BPlin'),
            _get_rl('Vf*(A*B - P*P/KE) / (B + bP*P**2)', 'AlinBPsat'),
            _get_rl('Vf*(A*B - P*P/KE) / (A + bP*P**2)', 'BlinAPsat'),
            _get_rl('Vf*(A*B - P*P/KE) / (A + bB*A*B)', 'PlinAsat'),
            _get_rl('Vf*(A*B - P*P/KE) / (B + bA*A*B)', 'PlinBsat'),
            _get_rl('Vf*(A*B - P*P/KE) / (A*B + bP*P**2)', 'ABPsat')]

    rl1s = [_get_rl('kf*(A*B - P*P/KE)', 'ABPlin'),
            _get_rl('Vf*(A*B - P*P/KE) / B', 'APlinBsat'),
            _get_rl('Vf*(A*B - P*P/KE) / P**2', 'ABlinPsat'),
            _get_rl('Vf*(A*B - P*P/KE) / A', 'BPlinAsat'),
            _get_rl('Vf*(A*B - P*P/KE) / A*B', 'PlinABsat')]

    rl0s = [_get_rl('0', '0'),
            _get_rl('inf * (A*B - P/KE)', 'inf', style='dashed')]
        
    
    hd = hasse.HasseDiagram(rank=4)
    rls = OD([(4, [rl4]), (3, rl3s), (2, rl2s), (1, rl1s), (0, rl0s)])
    for rank, rls_rank in rls.items():
        for rl, style in rls_rank:
            hd.add_node(rl.info, rank=rank, ratelaw=rl, style=style)
    
    hd.add_edge('mmh22ABPP', 'Alin', info='')
    hd.add_edge('mmh22ABPP', 'Blin', info='')
    hd.add_edge('mmh22ABPP', 'Plin', info='')
    hd.add_edge('mmh22ABPP', 'APsat', info='')
    hd.add_edge('mmh22ABPP', 'BPsat', info='')
    
    ############################################################################
    
    hd.add_edge('Alin', 'APlin', info='')
    hd.add_edge('Alin', 'ABlin', info='')
    hd.add_edge('Alin', 'AlinBPsat', info='')
    
    hd.add_edge('Blin', 'BPlin', info='')
    hd.add_edge('Blin', 'ABlin', info='')
    hd.add_edge('Blin', 'BlinAPsat', info='')
    
    hd.add_edge('Plin', 'APlin', info='')
    hd.add_edge('Plin', 'BPlin', info='')
    hd.add_edge('Plin', 'PlinAsat', info='')
    hd.add_edge('Plin', 'PlinBsat', info='')
    
    hd.add_edge('APsat', 'BlinAPsat', info='')
    hd.add_edge('APsat', 'PlinAsat', info='')
    hd.add_edge('APsat', 'ABPsat', info='')
    
    hd.add_edge('BPsat', 'AlinBPsat', info='')
    hd.add_edge('BPsat', 'PlinBsat', info='')
    hd.add_edge('BPsat', 'ABPsat', info='')

    ############################################################################
    
    hd.add_edge('ABlin', 'ABlinPsat', info='')
    hd.add_edge('ABlin', 'ABPlin', info='')
    
    hd.add_edge('APlin', 'APlinBsat', info='')
    hd.add_edge('APlin', 'ABPlin', info='')
    
    hd.add_edge('BPlin', 'BPlinAsat', info='')
    hd.add_edge('BPlin', 'ABPlin', info='')
    
    hd.add_edge('AlinBPsat', 'APlinBsat', info='')
    hd.add_edge('AlinBPsat', 'ABlinPsat', info='')
    
    hd.add_edge('BlinAPsat', 'BPlinAsat', info='')
    hd.add_edge('BlinAPsat', 'ABlinPsat', info='')
    
    hd.add_edge('PlinAsat', 'BPlinAsat', info='')
    hd.add_edge('PlinAsat', 'PlinABsat', info='')
    
    hd.add_edge('PlinBsat', 'APlinBsat', info='')
    hd.add_edge('PlinBsat', 'PlinABsat', info='')
    
    hd.add_edge('ABPsat', 'PlinABsat', info='')
    hd.add_edge('ABPsat', 'ABlinPsat', info='')
        
    ############################################################################
    
    hd.add_edge('APlinBsat', '0', info='')
    hd.add_edge('APlinBsat', 'inf', info='', style='dashed')
    
    hd.add_edge('BPlinAsat', '0', info='')
    hd.add_edge('BPlinAsat', 'inf', info='', style='dashed')
    
    hd.add_edge('ABlinPsat', '0', info='')
    hd.add_edge('ABlinPsat', 'inf', info='', style='dashed')
    
    hd.add_edge('ABPlin', '0', info='')
    hd.add_edge('ABPlin', 'inf', info='', style='dashed')
    
    hd.add_edge('PlinABsat', '0', info='')
    hd.add_edge('PlinABsat', 'inf', info='', style='dashed')
        
    return hd


def get_ratelaw(id, parametrization='VK'):
        """
        Input:
            id: a string indicating ratelaw type and the number of reactants;
                - 'mm21', 'mmh11', 'mmi2', 'ma11', 'mah22', 'mai1', 'qe12'
                - 'mmh21_AAP', to be found here in this module
        """
        if '_' not in id:  # is the presence of '_' the sole criterion?
            rltype = re.match('\D+(?=\d)', id).group()
            ns = id.lstrip(rltype)
            if len(ns) == 2:
                nsub, npro = int(ns[0]), int(ns[1])
            if len(ns) == 1:
                nsub, npro = int(ns), 0
                
            subids = list('ABCDEFG'[:nsub])
            proids = list('PQRSTUV'[:npro])
            xids = subids + proids
            
            masubstr, maprostr = '*'.join(subids), '*'.join(proids)
            # mafstr, marstr = '*'.join(subids), '*'.join(proids)
            # XK: X over K
            XKsubstrs = ['(%s/K%s)'%(subid, subid) for subid in subids]
            XKprostrs = ['(%s/K%s)'%(proid, proid) for proid in proids]
            
            Xbsubstrs = ['(%s*b%s)'%(subid, subid) for subid in subids]
            Xbprostrs = ['(%s*b%s)'%(proid, proid) for proid in proids]
                
            Ksubstr = '*'.join(['K%s'%subid for subid in subids])
            Kprostr = '*'.join(['K%s'%proid for proid in proids])
            
            XKsubstr, XKprostr = '*'.join(XKsubstrs), '*'.join(XKprostrs)
            XKdenstr = '(1+%s+%s)' % ('+'.join(['*'.join(e) for e in butil.powerset(XKsubstrs)][1:]),
                                      '+'.join(['*'.join(e) for e in butil.powerset(XKprostrs)][1:]))
            Xbdenstr = '(1+%s+%s)' % ('+'.join(['*'.join(e) for e in butil.powerset(Xbsubstrs)][1:]),
                                      '+'.join(['*'.join(e) for e in butil.powerset(Xbprostrs)][1:]))

            if rltype == 'mah':
                rl = RateLaw('kf*((%s)-(%s)/KE)'%(masubstr, maprostr), xids=xids)
            elif rltype == 'ma':
                rl = RateLaw('kf*(%s)-kr*(%s)'%(masubstr, maprostr), xids=xids)
            elif rltype == 'mai':
                rl = RateLaw('kf*(%s)'%masubstr, xids=xids)
            elif rltype == 'qe':
                rl = RateLaw('inf*((%s)-(%s)/KE)'%(masubstr, maprostr), xids=xids)
            elif rltype == 'mm':
                if parametrization == 'VK':
                    rl = RateLaw('(Vf*%s/(%s)-Vr*%s/(%s))/%s'%(masubstr, Ksubstr, maprostr, Kprostr, XKdenstr), 
                                 xids=xids)
                if parametrization == 'kb':
                    rl = RateLaw('(kf*%s-kr*%s)/%s'%(masubstr, maprostr, Xbdenstr), 
                                 xids=xids)
            elif rltype == 'mmh':
                if parametrization == 'VK':
                    rl = RateLaw('Vf/(%s)*(%s-%s/KE)/%s'%(Ksubstr, masubstr, maprostr, XKdenstr),
                                 xids=xids)
                if parametrization == 'kb':
                    rl = RateLaw('kf*(%s-%s/KE)/%s'%(masubstr, maprostr, Xbdenstr),
                                 xids=xids)
            elif rltype == 'mmi':
                if parametrization == 'VK':
                    rl = RateLaw('(Vf*%s/(%s))/%s'%(masubstr, Ksubstr, XKdenstr), xids=xids)
                if parametrization == 'kb':
                    rl = RateLaw('(kf*%s)/%s'%(masubstr, Xbdenstr), xids=xids)
            else:
                raise ValueError("Unrecognized ratelaw type: %s" % rltype)
            return rl
        else:
            return rls[id]


ma11 = get_ratelaw('ma11')
ma21 = get_ratelaw('ma21')
ma22 = get_ratelaw('ma22')
ma22_ABPP = RateLaw(s='kf * A*B - kr * P*P', xids=['A','B','P'])

mah11 = get_ratelaw('mah11')
mah21 = get_ratelaw('mah21')
mah22 = get_ratelaw('mah22')
mah22_ABPP = RateLaw(s='kf * (A*B - P*P/KE)', xids=['A','B','P'])

mm11 = get_ratelaw('mm11')
mm21 = get_ratelaw('mm21')
mm22 = get_ratelaw('mm22')
mm22_ABPP = RateLaw(s='(Vf*A*B/(KA*KB) - Vr*P*P/(KP*KP)) / (1 + A/KA + B/KB + A*B/(KA*KB) + 2*P/KP + P*P/(KP*KP))',
                    xids=['A','B','P'])

mmh11 = get_ratelaw('mmh11')
mmh21 = get_ratelaw('mmh21')
mmh22 = get_ratelaw('mmh22')
mmh22_ABPP = RateLaw(s='Vf/(KA*KB) * (A*B - P*P/KE) / (1 + A/KA + B/KB + A*B/(KA*KB) + 2*P/KP + P*P/(KP*KP))',
                    xids=['A','B','P'])

mm11_kb = get_ratelaw('mm11', 'kb')
mm21_kb = get_ratelaw('mm21', 'kb')
mm22_kb = get_ratelaw('mm22', 'kb')
mm22_ABPP_kb = RateLaw(s='(kf*A*B - kr*P*P) / (1 + bA*A + bB*B + bA*A*bB*B + 2*bP*P + bP*P*bP*P)',
                       xids=['A','B','P'])

mmh11_kb = get_ratelaw('mmh11', 'kb')
mmh21_kb = get_ratelaw('mmh21', 'kb')
mmh22_kb = get_ratelaw('mmh22', 'kb')
mmh22_ABPP_kb = RateLaw(s='kf*(A*B - P*P/KE) / (1 + bA*A + bB*B + bA*A*bB*B + 2*bP*P + bP*P*bP*P)',
                        xids=['A','B','P'])

hd_mmh11 = get_hasse_mmh11()

hd_mm11 = get_hasse_mm11()

hd_mm11Vf1 = get_hasse_mm11Vf1()

hd_mm11KA1 = get_hasse_mm11KA1()

hd_mmh22_ABPP = get_hasse_mmh22_ABPP()

hd_mmh11_kb = get_hasse_mmh11_kb()

hd_mm11_kb = get_hasse_mm11_kb()

#hd_2mmh11 = hasse.get_product([hd_mmh11]*2)


# todo's (in principle):
# - clean up latex: fractional structure; remove parentheses
# - clean up dot2tex: control fig size; make box appear; dashed boxes and edges
# automate edge creation? requires a good understanding of physics and naming scheme
if __name__ == '__main__':

    #hd = get_hasse_mm21()
    #hd.draw(nodeid2label=lambda nodeid: hd.get_node(nodeid)['texstr'], 
    #        width=60, height=60, nodeid2color='white', 
    #        rank2size={5:(2.5,0.8), 4:(2,0.8), 3:(1.5,0.7), 2:(1.2,0.5), 1:(1.2,0.5), 0:(1.2,0.5)},
    #        filepath='hasse_mm21_tmp.pdf')

    #subhd = hd.get_subdiagram(nodeids=hd.order[4]+hd.order[3][:-1]+hd.order[2][:-2]+hd.order[1][:-3])
    #subhd.draw(nodeid2label=lambda nodeid: subhd.get_node(nodeid)['texstr'], 
    #            width=60, height=60, nodeid2color='white', 
    #            rank2size={4:(2.5,0.8), 3:(2,0.8), 2:(1.5,0.7), 1:(1.2,0.5), 0:(1.2,0.5)},
    #            filepath='subhasse_mmke21.pdf')
    
    hd_mm11.draw(nodeid2label=lambda nodeid: '\\\\begin{tabular}{c} \\\\textcolor{red}{\\\\textsf{%s}} \\\\\\\\  \\\\\\\\ $%s$ \\\\end{tabular}'%(nodeid,hd_mm11.get_node(nodeid)['texstr']),
                 width=600, height=450, nodeid2color='white',  
                 rank2size={4:(1.2,0.8), 3:(1.2,0.8), 2:(1.1,0.8), 1:(1.1,0.8), 0:(0.7,0.6)},
                 filepath='hasse_mm11.pdf')
    
    #hd_mm11_kb.draw(nodeid2label=lambda nodeid: '\\\\begin{tabular}{c} \\\\textcolor{red}{\\\\textsf{%s}} \\\\\\\\  \\\\\\\\ $%s$ \\\\end{tabular}'%(nodeid,hd_mm11_kb.get_node(nodeid)['texstr']),
    #             width=600, height=450, nodeid2color='white',  
    #             rank2size={4:(1.2,0.8), 3:(1.2,0.8), 2:(1.1,0.8), 1:(1.1,0.8), 0:(0.7,0.6)},
    #             filepath='hasse_mm11_kb.pdf')
    
    
    #hd_mm11Vf1.draw(nodeid2label=lambda nodeid: '\\\\begin{tabular}{c} \\\\textcolor{red}{\\\\textsf{%s}} \\\\\\\\  \\\\\\\\ $%s$ \\\\end{tabular}'%(nodeid,hd_mm11Vf1.get_node(nodeid)['texstr']),
    #                width=600, height=450, nodeid2color='white',  
    #                rank2size={3:(1.2,0.8), 2:(1.1,0.8), 1:(1.1,0.8), 0:(0.7,0.6)},
    #                filepath='hasse_mm11Vf1.pdf')
    
    
    #hd_mm11KA1.draw(nodeid2label=lambda nodeid: '\\\\begin{tabular}{c} \\\\textcolor{red}{\\\\textsf{%s}} \\\\\\\\  \\\\\\\\ $%s$ \\\\end{tabular}'%(nodeid,hd_mm11KA1.get_node(nodeid)['texstr']),
    #                width=600, height=450, nodeid2color='white',  
    #                rank2size={3:(1.2,0.8), 2:(1.1,0.8), 1:(1.1,0.8), 0:(0.7,0.6)},
    #                filepath='hasse_mm11KA1.pdf')
    
    
    #hd_mmh11.draw(nodeid2label=lambda nodeid: '\\\\begin{tabular}{c} \\\\textcolor{red}{\\\\textsf{%s}} \\\\\\\\  \\\\\\\\ $%s$ \\\\end{tabular}'%(nodeid,hd_mmh11.get_node(nodeid)['texstr']),
    #              width=500, height=500, nodeid2color='white', 
    #              rank2size={3:(1.3,1), 2:(1,0.9), 1:(1,0.8), 0:(1,0.8)},
    #              filepath='hasse_mmh11.pdf')
    
    
    
    #hd_mmh22ABPP.draw(nodeid2label=lambda nodeid: '\\\\textsf{%s}'%nodeid,
    #                  width=100, height=100, nodeid2color='white', 
    #              rank2size={4:(0.9,0.5),3:(0.7,0.4), 2:(0.65,0.4), 1:(0.65,0.4), 0:(0.5,0.3)},
    #              filepath='hasse_mmh22ABPP.pdf')
    
    
            
    """
    rl_make11 = RateLaw(s='kf*(A-P/KE)', xids=['A','P'], cids=['KE'], info='make11')
    rl_ma11 = RateLaw(s='kf*A-kr*P', xids=['A','P'], info='ma11')
    rl_mai1 = RateLaw(s='kf*A', xids=['A'], info='mai1')
    
    rl_mmke11 = RateLaw(s='kf * (A - P/KE) / (1 + bA*A + bP*P)', 
                        xids=['A','P'], cids=['KE'], info='mmke11')
    rl_mm11 = RateLaw(s='(kf*A - kr*P) / (1 + bA*A + bP*P)', xids=['A','P'], info='mm11')
    rl_mmi1 = RateLaw(s='(kf*A) / (1 + bA*A)', xids=['A'], info='mmi1')
    
    
    rl_mmke21 = RateLaw(s='kf * (A*B - P/KE) / (1 + bA*A + bB*B + bA*bB*A*B + bP*P)',
                        xids=['A','B','P'], cids=['KE'], info='mmke21')
    
    rl_mmke21_full = RateLaw(s='kf * (A*B - P/KE) / (1 + bA*A + bB*B + 2*bA*bB*rho*A*B + bP*P)',
                             xids=['A','B','P'], cids=['KE'], info='mmke21_full')
    
    rl_mmke22 = RateLaw(s='kf * (A*B - P*Q/KE) / (1 + bA*A + bB*B + bA*bB*A*B + bP*P + bQ*Q + bP*bQ*P*Q)',
                        xids=['A','B','P','Q'], info='mmke22')
    """
    
    #rl_ = rl_ma11.facelift(eqn_new='C1<->X', pmap='num', rxnidx=1)
    #rl2_ = rl2.facelift(eqn_new='C2+X4<->X3', add_idx=4)
