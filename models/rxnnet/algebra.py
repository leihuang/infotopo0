"""
Polynomials

N v(x, p, u, C) = 0, where:
    x: dynamic variables
    p: parameters
    u: control variables (tunables)
    C: contants
<=> f(x, p, u, C) = 0, where f is a set of polynomials
<=> f(x, a) = 0, where a = g(u, c), nested polynomials
<=> f(x, u, c) = 0, where c = h(p, C)
    
=> x = x(u, p) = x(u, c) = x(a)
    
Implicit function theorem: if dfdx is invertible, then x = x(a)
Implicit differentiation: dxda = -(dfdx)^(-1) * dfda

Decomposition: 
    p |-> x(u)
 => p |-> c |-> x(u)

Multiple representations:
    - Polynomial
    - Polynomial function/equation
    - Coefficients
    - String
    - Latex string

Computation: 
    - Convert between representations
    - Verify rootfinding is the same as steady-state calculation
    - Calculate dim(p) = n, dim(c) = m, dim(image(h)) = k, dim(X) = k?

Different representations:
    - sympy object representation: for symbolic manipulations
    - string representation: for general inspection and manipulations
    - other representations: for numerical evaluations
    
"""

from __future__ import division
from collections import OrderedDict as OD
import re

import sympy 
import numpy as np
from scipy.optimize import fsolve

from SloppyCell import ExprManip as exprmanip

from util import butil
from infotopo import predict


def sym2str(sympyobj):
    """Convert a sympy object to string.
    """
    return str(sympyobj.as_expr())


def monom2str(m, xids):
    """Monomials to strings, where monomials are represented as tuples in
    sympy.
    >>> monom2str((0, 1, 1), ['X1','X2','X3'])
    'X2*X3'
    >>> monom2str((1, 2, 0), ['X1','X2','X3'])
    'X1*X2**2'
    """
    return sym2str(sympy.Monomial(m, sympy.symbols(xids)))


def str2poly(s, varids, subvarids=None, pids=None, p0=None, polyid='', 
             convarvals=None):
    """
    """
    poly = Polynomial(s, sympy.symbols(varids))
    
    #super(Polynomial, self).__init__(s, vars_)
    poly.varids = varids
    poly.subvarids = subvarids
    poly.pids = pids
    poly.p0 = butil.Series(p0, pids)
    if convarvals is None:
        convarvals = butil.Series([])
    poly.convarvals = convarvals
    poly.polyid = polyid
    return poly



class Polynomial(sympy.Poly):
    """
    """
    
    @classmethod
    def from_str(cls, s, varids, subvarids=None, pids=None, p0=None, polyid='', 
                 convarvals=None):
        vars_ = sympy.symbols(varids)
        poly = Polynomial(s, vars_)
        #super(Polynomial, self).__init__(s, vars_)
        poly.varids = varids
        poly.subvarids = subvarids
        poly.pids = pids
        poly.p0 = butil.Series(p0, pids)
        if convarvals is None:
            convarvals = butil.Series([])
        poly.convarvals = convarvals
        poly.polyid = polyid
        return poly
    
    
    
        
        
    def divide_by_lc(self):
        """
        """
        
        pass

    
    def get_coefs(self, norm=False):
        """
        """
        
        for m, a in self.terms():
            pass
        
        
        if self.uids is not None: 
            sublc = str2poly(self.coeffs()[0], self.uids).coeffs()[0]
            
            subcoefids = []
            subcoefstrs = []
            for m, coef in self.terms():
                subpoly = str2poly(sympy.simplify(coef, ratio=1), self.uids)
                subcoefids.extend([(monom2str(m, self.varids),
                                    monom2str(subm, self.uids))
                                   for subm in subpoly.monoms()])
                subcoefstrs.extend([str(sympy.simplify(subcoef/sublc)) 
                                    for subcoef in subpoly.coeffs()])
            return OD(zip(subcoefids, subcoefstrs)) 
                
        else:
            pass

    
    
    def get_h(self, keep1=True):
        """
        """
        coefs = self.get_coefs()
        if not keep1: 
            coefs = butil.get_submapping(coefs, f_value=lambda s: s != '1')
        subcoefids, subcoefstrs = coefs.keys(), coefs.values()
        yids = ['%s, %s, %s'%((self.polyid,)+subcoefid) 
                for subcoefid in subcoefids]
        
        senstrs = [[exprmanip.diff_expr(subcoefstr, pid)
                    for pid in self.pids] for subcoefstr in subcoefstrs]
            
        hstr = str(subcoefstrs).replace("'", "") 
        Dhstr = str(senstrs).replace("'", "") 
                
        subs0 = self.convarvals.to_dict()
        
        def h(p):
            subs = subs0.copy()
            subs.update(dict(zip(self.pids, p)))
            return np.array(eval(hstr, subs))
        
        def Dh(p):
            subs = subs0.copy()
            subs.update(dict(zip(self.pids, p)))
            return np.array(eval(Dhstr, subs))
        
        h = predict.Predict(f=h, Df=Dh, pids=self.pids, p0=self.p0, 
                            yids=yids, frepr=butil.Series(subcoefstrs, yids),
                            Dfrepr=butil.DF(senstrs, yids, self.pids)) 
        return h
        
        
    def __call__(self, varvals, subvarvals, p=None):
        if p is None:
            p = self.p0
        subs = dict(zip(self.pids, p) + zip(self.varids, varvals) +
                    zip(self.subvarids, subvarvals) + self.convarvals.items())
        return self.evalf(subs=subs)

            
class Polynomials(list):
    def __init__(self, polys):
        list.__init__(self, polys)
        #assert all([poly.pids == polys[0].pids for poly in polys]) 
        #assert all([(poly.p0 == polys[0].p0).all() for poly in polys])
        #assert all([poly.varids == polys[0].varids for poly in polys])
        #assert all([poly.subvarids == polys[0].subvarids for poly in polys])
        
        self.pids = polys[0].pids
        self.p0 = polys[0].p0
        self.varids = polys[0].varids
        self.subvarids = polys[0].subvarids
        self.polyids = [poly.polyid for poly in polys]
        
        self.xids = self.varids
        self.uids = self.subvarids
        
        
    def get_h(self, keep1=True):
        """
        c = h(p)
        """
        hs = [poly.get_h(keep1=keep1) for poly in self]
        yids = butil.flatten([h_.yids for h_ in hs])
        coefstrs = butil.flatten([h_.frepr.tolist() for h_ in hs])
        senstrs = [[exprmanip.simplify_expr(exprmanip.diff_expr(coefstr, pid))
                    for pid in self.pids] for coefstr in coefstrs]
        
        hstr = str(coefstrs).replace("'", "")
        Dhstr = str(senstrs).replace("'", "") 
        
        subs0 = dict(butil.flatten([poly.convarvals.items() for poly in self], 
                                   depth=1))
        def f(p):
            subs = subs0.copy()
            subs.update(dict(zip(self.pids, p)))
            return np.array(eval(hstr, subs))
        
        def Df(p):
            subs = subs0.copy()
            subs.update(dict(zip(self.pids, p)))
            return np.array(eval(Dhstr, subs))

        h = predict.Predict(f=f, Df=Df, pids=self.pids, p0=self.p0, 
                            yids=yids, frepr=butil.Series(coefstrs, yids),
                            Dfrepr=butil.DF(senstrs, yids, self.pids))
        return h
    
    
    def get_cpolys(self):
        cpolys = []
        for poly in self:
            for m, a in poly.terms():
                poly_a = Polynomial(a, self.uids) 
        pass
        
    def cu2a(self, c, u):
        pass
    
    @property
    def xdim(self):
        return len(self.xids) 
    
    
    @property
    def udim(self):
        return len(self.uids) 
    
    @property
    def fstr(self):
        return str([sym2str(poly) for poly in self]).replace("'","")
    
    
    #def eval(self, s=None, to_arr=True, **kwargs):
    #    if s is None:
    #        s = self.fstr
    #    return np.array(eval(s, kwargs))
    
    
    def get_root(self, namespace, x0=None, warning_negative=True, **kwargs):
        """
        Rootfinding. 
        varvalmap: a dict of variable values other than x's
        """
        nspace = namespace.copy()
        fstr = self.fstr
        
        def _f(x):
            nspace.update(dict(zip(self.xids, x)))
            return np.array(eval(fstr, nspace))

        if x0 is None:
            x0 = [1] * self.xdim
        out = fsolve(_f, x0, **kwargs)
        if warning_negative and any(out < 0):
            print "The found root has negative numbers: %s" % str(out)
        return out
    
    # cu2a, a2x -> get_root_cu
    # p2c (h), cu2a, a2x -> get_root_pu
    
    @property
    def aids(self):
        pass
    
    
    def dfdxstr(self):
        pass
    
    
    def get_dxdc_func(self, us, x0=None, namespace=None, **kwargs):
        """
        dxdc = -(dfdx)^(-1) * dfda * dadc
        
        Input:
            us: a list of u (u can a vector)
            x0: initial guess for rootfinding
        """
        def dxdc_func(c):
            dxdc = []
            for u in us:
                a_u = self.cu2a(c, u)  # polynomial evalution
                
                #x_u = self.a2x(a_u)  # root-finding
                if varvalmap is None:
                    varvalmap = {}
                varvalmap.update(dict(zip(self.cids, c) + zip(self.uids, u)))
                x_u = self.get_root(varvalmap, x0=x0, **kwargs)
                
                varvalmap_u = dict(zip(self.aids, a_u) + zip(self.xids, x_u) +\
                                   zip(self.uids, u))
                dfdx_u = np.matrix(eval(self.dfdxstr, varvalmap_u))
                dfda_u = np.matrix(eval(self.dfdastr, varvalmap_u))
                dadc_u = np.matrix(eval(self.dadcstr, varvalmap_u))
                dxdc_u = -dfdx_u.I * dfda_u, dadc_u
                dxdc.extend(dxdc_u.tolist()) 
            return np.array(dxdc)
        return dxdc_func
        
    
    def __add__(self, other):
        pass
        
    
    def __call__(self, p):
        pass
    
    
    
    
    def to_tex(self, d_tex=None, eqn=True, 
               filepath='', landscape=True, margin=2):
        """
        
        Input:
            d_tex: a mapping...
        """
        
        _repl = exprmanip.sub_for_vars
        _raisepower = lambda tu: tu[0] ** tu[1]
        
        def _2tex_pid(pid):
            if pid.startswith('Vf_') or pid.startswith('Vb_'):
                pid = '%s^{%s}' % tuple(pid.split('_'))
            if pid.count('_') == 2:
                pid = '%s^{%s}_{%s}' % tuple(pid.split('_'))
            return pid
        d_tex = dict(zip(self.pids, [_2tex_pid(pid) for pid in self.pids]) +\
                     d_tex.items())
        
        _2tex = lambda sympyexpr:\
            sympy.latex(sympyexpr, mode='plain', long_frac_ratio=10, mul_symbol='dot',
                        symbol_names=butil.chkeys(d_tex, lambda k: sympy.symbols(k))) 
        _rm1pt0 = lambda expr: re.sub('(?<![0-9])1.0\s*\\\cdot', '', expr)

        
        lines = []
        lines.append(r'\documentclass{article}') 
        lines.append(r'\usepackage{amsmath,fullpage,longtable,array,calc,mathastext,breqn,xcolor}') 
        if landscape == True:
            lines.append(r'\usepackage[a4paper,landscape,margin=1in]{geometry}')
        else:
            lines.append(r'\usepackage[a4paper,margin=%fin]{geometry}'%margin)
        lines.append(r'\begin{document}')
        
        coefs_r = []
        yids = []
        for poly in self:
            termstrs = []
            
            leadingcoef_r = sympy.Poly(poly.coeffs()[0], r).coeffs()[0]
            
            for monom_X, coef_X in poly.terms():
                coef_X = sympy.simplify(coef_X, ratio=1)
                poly_r = sympy.Poly(coef_X, r)
                
                coefs_r.extend([coef_r/leadingcoef_r for coef_r in poly_r.coeffs()])
                
                monom_X = sympy.prod(map(_raisepower, zip(X, monom_X)))
                monomstr_X = _2tex(monom_X)
                if monomstr_X == '1':
                    monomstr_X = ''
                monomstr_X = '\\textcolor{red}{%s}' % monomstr_X
                
                termstrs_r = []
                for monom_r, coef_r in poly_r.terms():
                    
                    coefstr_r = _rm1pt0(_2tex(coef_r))
                    if coef_r.is_Add:
                        coefstr_r = '\left('+ coefstr_r +'\\right)'
                    
                    monom_r = sympy.prod(map(_raisepower, zip(r, monom_r)))
                    monomstr_r = _2tex(monom_r)
                    if monomstr_r == '1':
                        monomstr_r = ''
                    monomstr_r = '\\textcolor{blue}{%s}' % monomstr_r
                        
                    termstrs_r.append(coefstr_r + '\t' + monomstr_r)
                    coefstr_X = '\\left(' + '+'.join(termstrs_r) + '\\right)'
                    
                    yids.append((ixid, str(monom_X), str(monom_r)))
                
                termstrs.append(coefstr_X + '\t' + monomstr_X)
        
            linestr = '\\begin{dmath} \n' + '+'.join(termstrs) + '=0\n\end{dmath} \n\n'
            lines.append(linestr.replace('+-', '-'))
        
        lines.append('\\end{document}') 
        
        if filepath:
            fh = file(filepath, 'w') 
            fh.write(os.linesep.join(lines)) 
            fh.close()
        
        coefs_r = [_rm1pt0(str(coef)) for coef in coefs_r]
        
        str_h = str(coefs_r).replace("'", "")
        str_Dh = str([[exprmanip.diff_expr(coef, pid) for pid in self.pids] 
                      for coef in coefs_r]).replace("'", "")
        
        def h(p):
            self.update(p=p)
            return np.array(eval(str_h, self.varvals.to_dict()))
        
        def Dh(p):
            self.update(p=p)
            return np.array(eval(str_Dh, self.varvals.to_dict()))
        
        coefs = predict.Predict(f=h, Df=Dh, pids=self.pids, p0=self.p0, 
                                yids=yids, funcform=coefs_r) 
        
        return coefs

    
    
if __name__ == '__main__':
    polystr1 = 'k1*k2*r*X1**2 + k1^2*X2*X3 + X3*k2*X1*X2'
    polystr2 = 'k1**2*r*X2**2 + 2*k1^2*X1*X3'
    
    pids = ['k1','k2','Vf1']
    varids = ['X1','X2','RuBP']
    p0 = [1,2,3]
    
        
    poly1 = Polynomial.from_str(polystr1, varids, pids=pids, p0=p0, polyid='X1', 
                                subvarids=['r'], convarvals=butil.Series([1], ['X3']))
    poly2 = Polynomial.from_str(polystr2, varids, pids=pids, p0=p0, polyid='X1', 
                                subvarids=['r'], convarvals=butil.Series([1], ['X3']))
    
        
    polys = Polynomials([poly1, poly2])
