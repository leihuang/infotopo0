"""
Model comparison:

            M
 -- Theta  --> 
 |          M'     B
 |  Theta' -->
 |    |            |
 |f   |f'          | X
 |    |            V
 |    |--------> 
 |                 D
 |------------->   

Compare f and f'.

Terminology: 
    dexpt: experiment for data
    pexpt: experiment for predict
    pred: predict
    y: data
    z: predict
    
    
Compare: 
    structure: 
        - FIXME **: migrate functions in rxnnet.model to here
        - FIXME **: in this case, can allow dexpts and pexpts to be None
        p: 
        custom object: eg, ratelaws (latex)
    behavior:
        fit: ensemble (sampling)
        prediction: ensemble (sampling)
    
"""

from collections import OrderedDict as OD

import pandas as pd

from util import butil

import residual
reload(residual)


class Comparison(object):
    """
    """
    def __init__(self, mod, mod2, dexpts, dexpts2=None, pexpts=None):
        """
        Input: 
            mod: reference/true model
            mod2: approximating model
            dexpts:
            dexpts2: sometimes different from dexpts in 'conditions'; indices have to match
            pexpts: 
        """
        self.mod = mod
        self.mod2 = mod2
        self.dexpts = dexpts
        self.dexpts2 = dexpts2
        self.dpred = mod.get_predict(dexpts)
        if dexpts2 is None:
            dexpts2 = self.dpred.expts
        else:
            dexpts2 = dexpts2.loc[self.dpred.expts.index]
        self.dpred2 = mod2.get_predict(dexpts2)
        if pexpts:
            self.pexpt = pexpts
            self.ppred = mod.get_predict(pexpts)
            self.ppred2 = mod2.get_predict(pexpts)
        
        
    def switch(self):
        pexpts = getattr(self, 'pexpts', None)
        return Comparison(self.mod2, self.mod, self.dexpts2, self.dexpts, pexpts)
    
    
    def get_residual2(self, p=None, **kwargs_dat):
        """
        
        """
        dat = self.dpred.make_dat(p=p, **kwargs_dat)
        # conditions may have different names
        dat2 = dat.rename(dict(zip(self.dpred.dids, self.dpred2.dids)))
        res2 = residual.Residual(self.dpred2, dat2)
        return res2
    
    
    def fit(self, p=None, **kwargs):
        """
        Input:
            kwargs: kwargs for predict.Predict.make_dat & residual.Residual.fit; 
                    usually include the following:
                        p0: initial guess for fitting dat to dpred2
        """
        res2 = self.get_residual2(p=p, **kwargs)
        cost2, p2 = res2.fit(**kwargs)
        return cost2, p2
    
    
    def sampling(self, nstep, p=None, **kwargs):
        """
        
        Input:
            nstep: 
            kwargs: kwargs for predict.Predict.make_dat, 
                               residual.Residual.fit,
                               residual.Residual.sampling.
        """
        res2 = self.get_residual2(p, **kwargs)
        cost2, p2 = res2.fit(**kwargs)
        if 'p0' in kwargs:
            del kwargs['p0']
        ens2 = res2.sampling(p0=p2, nstep=nstep, **kwargs)
        return ens2
    
    
    def cmp_prediction(self, p, ens=False, **kwargs):
        """
        """
        z = self.ppred(p)
        if ens:
            ens2 = self.sampling(p=p, **kwargs)
            eens2, pens2 = ens2.split()
            zs2 = pens2.apply(self.ppred2)
            return eens2, pens2, z, zs2
        else:
            cost2, p2 = self.fit(p, **kwargs)
            z2 = self.ppred2(p2)
            return cost2, p2, z, z2
    
    
    def plot_manifolds(self):
        """
        """
        pass
    
    
    def save(self):
        pass
        
        
    
    def cmp_p(self):
        pass
    
    
    def cmp(self, func, varids=None, to_table=False, tex=True, filepath='', **kwargs_tex):
        """
        FIXME ***
        
        Input:
            func: a function that takes a model and outputs something to be
                compared
            varids: a list
            to_table: if True...
            kwargs_tex: margin... 
        """
        df = butil.DF(OD([(self.mod.id, func(self.mod)), 
                          (self.mod2.id, func(self.mod2))]), index=varids)
        return df


