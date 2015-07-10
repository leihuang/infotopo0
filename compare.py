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
    dexpts: experiment for data
    pexpts: experiment for prediction
    dpred: predict for data
    ppred: predict for prediction
    y: data
    z: prediction
    
    
Compare: 
    structure: 
        - FIXME **: migrate functions in rxnnet.model to here
        - FIXME **: in this case, can allow dexpts and pexpts to be None
        p: 
        custom object: eg, ratelaws (latex)
    behavior:
        fit: sampling (sampling)
        prediction: sampling (sampling)
        

compare:
 - cmp: 
 - cmp_p: 
 
 - cmp_fit: p -> y -> p2 -> cost2
 - cmp_fits: pens -> yens -> pens2 -> eens2;  sample M(P), get the distribution of cost: eens2
 - cmp_prediction: p -> y -> p2/pens2
                   |            |
                   V            V
                   z         z2/zens2
 - cmp_predictions: pens -> yens -> pens2 -> eens2
                     |                |
                     V                V
                    zens            zens2
 

    
"""

from collections import OrderedDict as OD

import numpy as np
import pandas as pd

from util import butil
Series, DF = butil.Series, butil.DF

import residual, sampling
reload(residual)
reload(sampling)




class Comparison(object):
    """
    """
    def __init__(self, mod, mod2, dexpts=None, dexpts2=None, pexpts=None, pexpts2=None):
        """
        Input: 
            mod: reference model
            mod2: alternative model
            dexpts: experiments for collecting data
            dexpts2: sometimes different from dexpts in 'conditions';
                eg, ('kf',2) vs ('Vf',2);
                depxts.yids has to match dexpts2.yids            
            pexpts: experiments for making predictions;
                pepxts.yids has to match pexpts2.yids            
        """
        self.mod = mod
        self.mod2 = mod2
        self.dexpts = dexpts
        self.dexpts2 = dexpts2
        self.pexpts = pexpts
        self.pexpts2 = pexpts2
        if dexpts is not None:
            self.dpred = mod.get_predict(dexpts)
        if dexpts2 is not None:
            self.dpred2 = mod2.get_predict(dexpts2)
        if pexpts is not None:
            self.ppred = mod.get_predict(pexpts)
        if pexpts2 is not None:
            self.ppred2 = mod2.get_predict(pexpts2)
            
        self.pids = mod.pids
        self.pids2 = mod2.pids
        
        self.p0 = self.dpred.p0
    
    def add_dexpts(self, dexpts, dexpts2):
        """
        """
        self.dexpts = dexpts
        self.dexpts2 = dexpts2
        self.dpred = self.mod.get_predict(dexpts)
        self.dpred2 = self.mod2.get_predict(dexpts2)
        
    
    def add_pexpts(self, pexpts, pexpts2):
        pass
        
        
    @property    
    def yids(self):
        return self.dexpts.yids

    @property    
    def yids2(self):
        return self.dexpts2.yids

    
    def switch(self):
        """
        """
        return Comparison(mod=self.mod2, mod2=self.mod, 
                          dexpts=self.dexpts2, dexpts2=self.dexpts, 
                          pexpts=self.pexpts2, pexpts2=self.pexpts)
    
    
    def get_residual2(self, p=None, **kwargs_dat):
        """
        
        """
        dat = self.dpred.make_dat(p=p, **kwargs_dat)
        # conditions may have different names
        dat2 = dat.rename(dict(zip(self.dpred.yids, self.dpred2.yids)))
        res2 = residual.Residual(self.dpred2, dat2)
        return res2
    
    
    def cmp_fit(self, p=None, sampling=False, initguess=None, **kwargs):
        """
        Input:
            sample: if True, return the ensemble fit; otherwise return just the 
                best fit
            initguess: a function that guesses p0 for fitting residual2 from p
            kwargs: kwargs for predict.Predict.make_dat & residual.Residual.fit; 
                    usually include the following:
                        p0: initial guess for fitting dat to dpred2
        
        Output:
            If 'full_output' is True in kwargs, then returns a tuple of
                (cost, p_fit, nfcall, nDfcall, convergence, lamb, Df);
            else returns (cost, p_fit)
            See the doc of residual.Residual.fit for details.
            
            (p, y, e, p2, y2, e2)
        """
        res2 = self.get_residual2(p=p, **kwargs)
                    
        if initguess: 
            kwargs['p0'] = initguess(p)

        out = res2.fit(**kwargs)
        
        if sampling:
            ens2 = sampling.sampling(res2, p0=out.p, **kwargs)
            return ens2
        else:
            return out
    
    
    def cmp_fits(self, **kwargs):
        """
        
        Input:
            kwargs: predict.Predict.set_prior: dim, codim;
                    sampling.sampling: nstep, p0, in_log, 
                    residual.Residual.fit: 
        
        """
        if self.dpred.prior is None:
            self.dpred.set_prior('jeff', **kwargs)
        ens = sampling.sampling(self.dpred, **kwargs)
        def _f(p):
            tuples = [('p2', pid) for pid in self.pids2] + [('e2', 'cost')]
            index = pd.MultiIndex.from_tuples(tuples)
            try:
                out = self.cmp_fit(p=p, disp=0, full_output=1, **kwargs)                
                return Series(out.p.tolist() + [out.cost], index=index)
            except:
                return Series([np.nan]*len(self.pids2) + [np.inf], index=index)
        ens2 = ens.p.apply(_f, axis=1)
        return ens.add_ens(ens2)
    
        
    """
    def sampling(self, nstep, p=None, **kwargs):

        Input:
            nstep: 
            kwargs: kwargs for predict.Predict.make_dat, 
                               residual.Residual.fit,
                               residual.Residual.sampling.

        res2 = self.get_residual2(p, **kwargs)
        cost2, p2 = res2.fit(**kwargs)
        if 'p0' in kwargs:
            del kwargs['p0']
        ens2 = res2.sampling(p0=p2, nstep=nstep, **kwargs)
        return ens2
    """    
        
    
    def cmp_prediction(self, p=None, sampling=False, **kwargs):
        """
        """
        if p is None:
            p = self.p0
            
        z = self.ppred(p)
        
        res2 = self.get_residual2(p=p)
        
        if 'initguess' in kwargs:
            p2_init = kwargs['initguess'](p)
        p2, cost2 = res2.fit(p=p2_init, sample=False, **kwargs)[['p', 'cost']]
        
        if sampling:
            ens2 = res2.sampling(p0=p2, **kwargs)
            ens2.vartypes = ['p2', 'e2']
            zens2 = ens2.p2.apply(self.ppred2, axis=1)
            ens2 = ens2.add_ens(z2=zens2)
            return z, ens2  # ens2: p2, e2, z2
        else:
            z2 = self.ppred2(p2)
            dat = p.tolist() + z.tolist() + p2.tolist() + [cost2] + z2.tolist()
            tus = [('p', pid) for pid in p.index] +\
                  [('z', zid) for zid in z.index] +\
                  [('p2', pid) for pid in p2.index] +\
                  [('e2', 'cost')] +\
                  [('z2', zid) for zid in z2.index]
            out = Series(dat, index=pd.MultiIndex.from_tuples(tus))
            return out
        
        """
        if sample:
            ens2 = self.sampling(p=p, **kwargs)
            eens2, pens2 = ens2.split()
            zs2 = pens2.apply(self.ppred2)
            return eens2, pens2, z, zs2
        else:
            cost2, p2 = self.fit(p, **kwargs)
            z2 = self.ppred2(p2)
            return cost2, p2, z, z2
        """
    
    def cmp_predictions(self, ens=None, **kwargs):
        """
        """
        if ens is not None:
            assert hasattr(ens, 'p'), "ens does not have p"
            assert hasattr(ens, 'p2'), "ens does not have p2"
        else:
            ens = self.fits(**kwargs)
        zens = ens.p.apply(self.ppred, axis=1)
        zens2 = ens.p2.apply(self.ppred2, axis=1)
        ens = ens.add_ens(z=zens, z2=zens2)
        return ens
        
        
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
    
    
    def cmp_p(self):
        pass
    
    
    def plot_manifolds(self):
        """
        """
        pass
    
