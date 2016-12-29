"""
(u, p) |-> z
f: p |-> y=z(u)

"""

from __future__ import division
from collections import OrderedDict as OD, Counter
import logging
import copy
import itertools
import cPickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SloppyCell import ExprManip as exprmanip

from util import butil
from util.butil import Series, DF
from util.matrix import Matrix
from util import plotutil

from infotopo import residual, geodesic
reload(residual)
reload(geodesic)


class Predict(object):
    """
    """
    
    def __init__(self, f=None, p0=None, name='', pids=None, yids=None, Df=None, rank=None,
                 domain=None, prior=None, ptype='', rdeltap=0.01, pred=None, **kwargs):
        """
        Input:
            f: a function, optimized for performance, outputting **np.array**
            Df: differential of f, optimized for performance, outputting
                **np.array**;  if not given, use finite difference
            p0:  reference parameter value
            pids: parameter ids
            yids: prediction ids
            domain: parameter space 
            prior: prior distribution on parameter space
            rdelta: used for getting _Df through finite difference
            
            
        
        """
        # use np.array() because when passing a pd object to the object's
        # __init__ with index also given, the values would be messed up
        # Eg, ser = pd.Series([1,2], index=['a','b'])
        # pd.Series(ser, index=['A','B']) would return:
        # A   NaN
        # B   NaN
        # dtype: float64
        # for details, one can check out .../pandas/core/series.py
            
        ## FIXME ***: fix the constructor
        
        #def _f(p=None):
        #    if p is None:
        #        p = p0
        #    #if isinstance(p, list) or isinstance(p, tuple) or isinstance(p, np.ndarray):
        #    #p = Series(p, index=pids)
        #    return Series(f(p), index=yids)
        if pred is not None:  # FIXME **: add other attributes as well; clean things up a bit
            f = pred.f
            Df = pred.Df
            p0 = pred.p0
            pids = pred.pids
            yids = pred.yids
            
        _f, _Df = f, Df
        
        if _Df is None:
            logging.warn("Df not provided; calculated using finite difference.")
            _Df = get_Df_fd(_f, rdeltap=rdeltap)
        
        def f(p=None, to_ser=False):
            if p is None:
                p = p0
            y = _f(p)
            if to_ser:
                y = Series(y, yids)
            return y
        
        def Df(p=None, to_mat=False):
            if p is None:
                p = p0
            jac = _Df(p)
            if to_mat:
                jac = Matrix(jac, index=yids, columns=pids) 
            return jac
        
        self.f = f
        self.Df = Df
        self._f = _f
        self._Df = _Df
        self.pids = pids
        self.yids = yids
        self.p0 = Series(p0, pids)
        self.name = name
        self.N = len(pids)  # to be deprecated; use pdim
        self.M = len(yids)  # to be deprecated; use ydim
        self.pdim = len(pids)
        self.ydim = len(yids)
        self.rank = rank
        self.domain = domain
        self.prior = prior
        self.ptype = ptype
        
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)
                
    
    # necessary??    
    #def __getattr__(self, attr):
    #    return getattr(self.f, attr)
           
            
    def __call__(self, p=None):
        return self.f(p=p)
    
    """
    def __repr__(self):
        return "pids: %s\nyids: %s\np0:\n%s"%\
            (str(self.pids), str(self.yids), str(self.p0))
    """
    
    def __getitem__(self, keys):
        """A convenience function for using only part of the data. 
        Since it still computes all the data and only takes a subset afterwards, 
        one should code the sub-predict separately if performance is important. 
         
        Input:
            keys: a slice object or a list of indices 
                (treated similarly by numpy)
        
        Output: 
            a sub-predict object
        """
        _fsub = lambda p: self._f(p)[keys] 
        _Dfsub = lambda p: self._Df(p)[keys]
        predsub = Predict(f=_fsub, Df=_Dfsub, p0=self.p0,
                          pids=self.p0.varids, yids=self.yids[keys], 
                          domain=None, prior=None, ptype=self.ptype)
        return predsub
    
        
    def get_in_logp(self):
        """
        Get a Prediction object in log parameters.
        """
        assert self.ptype == '', "predict not in bare parametrization."
        
        def _f_logp(logp):
            p = np.exp(np.array(logp)) 
            return self.f(p)
        
        def _Df_logp(logp):
            # d y/d logp = d y/(d p/p) = (d y/d p) * p
            p = np.exp(np.array(logp))
            return self.Df(p) * p

        pred_logp = Predict(f=_f_logp, Df=_Df_logp, p0=self.p0.log(),
                            pids=self.p0.logvarids, yids=self.yids, 
                            domain=None, prior=None, ptype='logp')
        return pred_logp
    
    
    def get_in_log10p(self):
        """Get a predict in log10 parameters.
        """
        assert self.ptype == '', "predict not in bare parametrization."
        
        def _f_log10p(log10p):
            p = np.power(10, np.array(log10p))
            return self.f(p)
        
        def _Df_log10p(log10p):
            p = np.power(10, np.array(log10p))
            # d y/d log10p = d y/(d p/(p*log10)) = (d y/d p) * p/log10
            return self.Df(p) * p / np.log(10)
        
        log10pids = map(lambda pid: 'log10_'+pid, self.pids)
        log10p0 = Series(np.log10(self.p0.values), log10pids)
        pred_log10p = Predict(f=_f_log10p, Df=_Df_log10p, p0=log10p0, 
                              pids=log10pids, yids=self.yids, 
                              domain=None, prior=None, ptype='log10p')
        return pred_log10p
    
    
    def set_prior(self, prior=None, dim=None, codim=None, p0=None, **kwargs):
        """
        Input:
            prior: a string
            dim, codim: an int
            kwargs: a placeholder
        """
        if prior == 'jeff':
            if dim is None:
                dim = len(self.p0)
            if codim:
                dim -= codim
            self.prior = lambda p: self.Df(p).svd(to_mat=False)[1][:dim].prod()
        if prior == 'lognormal':
            pass
    
        
    def get_errorbar(self, p=None, errormodel='sigma', cv=0.1, sigma0=1, 
                  to_ser=False, **kwargs):
        """Calculate the sigmas of data from the specified *error model*; 
        the default setting amounts to unweighted least square.
        
        Input:
            errormodel: 'sigma': constant sigma of sigma0 (default)
                        'cv': proportional to y by cv
                        'mixed': the max of scheme 'sigma' and 'cv'
            cv: coefficient of variation
            sigma0: constant sigma (default is 1)
            
        """
        y = self(p)
        if errormodel == 'sigma':
            sigma = np.array([sigma0] * len(y))
        if errormodel == 'cv':
            sigma = y * cv
        if errormodel == 'mixed':
            sigma = np.max((y*cv, [sigma0]*len(y)), axis=0)
        if to_ser:
            sigma = Series(sigma, self.yids) 
        return sigma
    get_sigma = get_errorbar  # deprecation warning
        
        
    def get_dat(self, p=None, **kwargs):
        """
        Input:
            kwargs: kwargs of Predict.get_sigma, whose docstring is 
                attached below: \n
        """
        y = self(p)
        sigma = self.get_sigma(p=p, **kwargs)
        dat = DF(OD([('Y',y), ('sigma',sigma)]), index=self.yids)
        return dat
    get_dat.__doc__ += get_sigma.__doc__
    
    
    def scale(self, p=None, sigmas=None, **kwargs):
        """Return a new predict whose output is scaled by sigma."""
        if p is None:
            p = self.p0
        if sigmas is None:
            sigmas = self.get_sigma(p=p, **kwargs)
        else:
            sigmas = np.array(sigmas)
        f = lambda p: self.f(p) / sigmas
        Df = lambda p: (self.Df(p).T/sigmas).T
        #yids = map(lambda yid: '%s/sigma' % yid, self.yids)
        if 'yids' in kwargs:
            yids = kwargs['yids']
        else:
            yids = ['%s / %f'%(yid, sigma) for yid, sigma in zip(self.yids, sigmas)]
        return Predict(f=f, Df=Df, p0=self.p0, pids=self.pids, yids=yids)
    
    
    def to_residual(self, dat=None, **kwargs_dat):
        """
        Input:
            dat:
            kwargs_dat: kwargs for 
                self.make_dat(p=None, scheme='sigma', cv=0.2, sigma0=1, 
                              sigma_min=1)
        """
        if dat is None:
            dat = self.make_dat(**kwargs_dat)
        return residual.Residual(pred=self, dat=dat)
    
    
    def currying(self, name='', **kwargs):
        """
        Fix part of the arguments, keep the order of arguments
        
        Input:
            kwargs: parameter id = fixed parameter value
        
        https://en.wikipedia.org/wiki/Currying 
        """
        pred = copy.deepcopy(self)
        
        pids = [pid for pid in pred.pids if pid not in kwargs.keys()]
        idxs = [pred.pids.index(pid) for pid in pids]
        p0 = pred.p0[pids]
        
        p_template = pred.p0.copy()
        p_template[kwargs.keys()] = kwargs.values()
        
        def _augment(p):
            p_full = p_template.copy() 
            p_full[pids] = p  # not making a copy
            return p_full
        
        def _f(p):
            return pred._f(_augment(p))
        
        def _Df(p):
            return pred.Df(_augment(p))[:,idxs]
        
        return Predict(f=_f, Df=_Df, p0=p0, name=name, pids=pids, yids=pred.yids)
    
    
    def __add__(self, other):
        """
        Concatenate the output:
        (f+g)(p) = (f(p),g(p))
        """
        assert self.pids == other.pids, "pids not the same"
        assert all(self.p0 == other.p0), "p0 not the same"
        
        def _f(p):
            return np.concatenate((self._f(p), other._f(p)))
        
        def _Df(p):
            return np.concatenate((self._Df(p), other._Df(p)))
            

        pred = Predict(f=_f, Df=_Df, p0=self.p0,
                       pids=self.pids, yids=self.yids+other.yids, 
                       domain=None, prior=None, ptype='')
        return pred

    
    '''    
    def plot(self, n=100, pts=None, show=True, filepath=''):

        if self.domain is not None:
            ps = self.domain.apply(lambda interval: 
                                   np.linspace(interval[0], interval[1], n+1))
        pgrids = np.meshgrid(*ps)  
        ygrids = self.f(*pgrids)
        #ys = [ygrid.flatten() for ygrid in ygrids]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        #import ipdb
        #ipdb.set_trace()
        
        ax.set_aspect("equal")
        ax.plot_surface(*ygrids, color='b', alpha=0.2,
                        shade=False, edgecolor='none')
        #ax.set_xlim(0,2)
        #ax.set_ylim(0,2)
        #ax.set_zlim(0,2)
        
        if pts is not None:
            ax.plot3D(pts, color='r')
        
        if show:
            plt.show()
        plt.savefig(filepath)
        plt.close()
    '''
        
    ############################################################################
    # svd
    
    def svd(self, p=None):
        """returns U, S, Vh (note that it is not V), such that Df = U * S * Vh. 
        """
        return np.linalg.svd(self.Df(p))
        
    
    def get_spectrum(self, p=None):
        return np.linalg.svd(self.Df(p), compute_uv=False)
    
    
    def get_rank(self, ntrial=3, sigma=1, tol=None, ndiff_allowed=0):
        """
        Input:
            ntrial: number of parameter points to try 
            sigma:
            tol:
            ndiff_allowed: number of different ranks allowed in the random trials
        """
        if self.ptype == '':
            kwargs = dict(distribution='lognormal', sigma=sigma)
        else:  # log p
            kwargs = dict(distribution='normal', scale=sigma)
            
        if self.rank is not None:
            return self.rank
        else:
            # http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.linalg.matrix_rank.html
            # http://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon
            if self.pdim == 0:
                return 0
            
            seeds = np.random.randint(low=0, high=1000, size=ntrial)
            ps = [self.p0.randomize(seed=seed, **kwargs) for seed in seeds]
            
            ranks = []
            for p in ps:
                try:
                    singvals = self.get_spectrum(p=p)
                    if tol is None:
                        ## **** FIXME, sometimes this is too stringent, 
                        ## esp. for jacobian with two parameters
                        ## not accurate; shouldn't be trusted
                        tol = singvals[0] * max(self.N, self.M) *\
                            np.finfo(singvals.dtype).eps
                    rank = np.sum(singvals > tol)
                    ranks.append(rank)
                except:  # FIXME *: what exceptions to accept?
                    pass
            
            
            rank2cnt = Counter(ranks)
            rank_major = max(rank2cnt.items(), key=lambda item: item[1])[0]
            ndiff = len(ranks) - rank2cnt[rank_major]
            #print ndiff, ranks
            if ndiff == 0:
                return rank_major
            elif ndiff <= ndiff_allowed:
                logging.warning('Ranks of different trials are not all the same: %s' % str(ranks))
                return rank_major
            else:
                raise ValueError('More ranks are different than allowed: %s' % str(ranks))
    
    
    def get_volume(self, p, rank=None):
        #if self.rank is None:
        #    # http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.linalg.matrix_rank.html
        #    # http://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon
        #    tol = sigma[0] * max(self.N, self.M) * np.finfo(sigma.dtype).eps
        #    rank = np.sum(sigma > tol)
        if rank is None:
            rank = self.pdim
        return np.prod(self.get_spectrum(p)[:rank])
    
    
    def get_eigvec(self, p=None, idx=-1):
        """
        """
        return np.linalg.svd(self.Df(p), full_matrices=False)[-1][idx]
    get_eigenv = get_eigvec  # deprecation warning
    
    
    def get_sloppyv(self, p=None, dt=0.1):
        """Deprecation warning  FIXME **
        
        Should be in in logp? Otherwise it does not make sense to compare 
        the spectrums at p+deltap and p-deltap. (eg, when p0=[1], and 0 and inf are two limits.)
        """
        if p is None:
            p = self.p0
        sloppyv_f = np.linalg.svd(self.Df(p))[-1][-1,:]
        sloppyv_b = -sloppyv_f
        p_f = p + sloppyv_f * dt
        p_b = p + sloppyv_b * dt
        vol_f = self.get_volume(p_f)
        vol_b = self.get_volume(p_b)
        if vol_f < vol_b:
            sloppyv = sloppyv_f
        else:
            sloppyv = sloppyv_b
        
        # The following codes implement the selection method mentioned 
        # in the second paragraph of Transtrum & Qiu 14, suppl doc., 
        # which is based on the speed;
        # But they do not yield satisfying results, hence commented off. 
        """
        speed_f = np.linalg.norm(self.svd(p_f)[-1][:,-1]) 
        speed_b = np.linalg.norm(self.svd(p_b)[-1][:,-1]) 
        if speed_f > speed_b:
            sloppyv = sloppyv_f
        else:
            sloppyv = sloppyv_b
        """
        return -sloppyv
    
    ############################################################################
    
    def get_geodesic(self, p0=None, ptype='logp', v0=None,
                     idx_eigvec=None, uturn=False, yidxs=None, **kwargs):
        """
        Input:
            p0: initial parameter, in bare parametrization
            ptype: str
            v0: in parametrization given in **ptype**
            idx_eigvec:
            uturn: 
            yidxs: a subset of data indices; if given, only part of the data 
                would be used to calculate eigvec (useful for SN)
            kwargs: kwargs of geodesic.Geodesic.__init__, whose docstring 
                is appended below. \n
        """
        assert self.ptype == '', "pred is not in bare parametrization." 
        
        if p0 is None:
            p0 = self.p0.values
            
        if ptype == '':
            pred = self
        elif ptype == 'logp':
            pred = self.get_in_logp()
            p0 = np.log(p0)
        elif ptype == 'log10p':
            pred = self.get_in_log10p()
            p0 = np.log10(p0)
        else:
            raise ValueError("ptype is invalid.")
        
        # get v0
        if v0 is None:
            if idx_eigvec is not None:
                if yidxs is not None:
                    predsub = pred[yidxs]
                    v0 = predsub.get_eigvec(p0, idx=idx_eigvec)
                else:
                    v0 = pred.get_eigvec(p0, idx=idx_eigvec)
                if uturn:
                    v0 = -v0
            else:
                v0 = pred.get_sloppyv(p0)
            
        if 'rank' not in kwargs and kwargs.get('inv', '') == 'pseudo':
            kwargs['rank'] = pred.get_rank() 
            
        gds = geodesic.Geodesic(f=pred.f, Df=pred.Df, p0=p0, v0=v0,  
                                pids=pred.pids, yids=pred.yids, 
                                ptype=ptype, pred=pred, **kwargs)
        return gds
    get_geodesic.__doc__ += geodesic.Geodesic.__init__.__doc__
    
    
    def get_geodesics(self, p0=None, ptype='logp', v0s=None, v0idxs=None,
                      seeds=None, sigma=1,
                      **kwargs):
        """Return geodesic.Geodesics (a Series type of object). If v0s is not 
        provided (usually the case), by default uses directions along all 
        *eigenpredictions* (both forward and reverse directions). 
        # If seeds and/or sigmas are provided, then also iterate over different
        # p0's generated using the seeds and sigmas, and in this case the index
        # of the returned gdss would be a multiindex...
        
        Input:
            v0idxs:
            v0s: almost never used... but I should keep it here... Commented 
                out for now for simplicity.  FIXME ** 
        """
        #if v0s is None and v0idxs is None:
        if v0idxs is None:
            v0idxs = butil.get_product(range(-1, -self.pdim-1, -1), 
                                       [True, False])
        
        
        _p02gdss = lambda p0: [self.get_geodesic(p0=p0, ptype=ptype, v0=None,
                                                 idx_eigvec=idx_eigvec, 
                                                 uturn=uturn, **kwargs)
                               for idx_eigvec, uturn in v0idxs]
    
        if seeds is not None:
            _gdss = []
            seeds_list = list(seeds)
            for seed in seeds:
                _p0 = self.p0.randomize(seed=seed, sigma=sigma)
                try:
                    _gdss_p0 = _p02gdss(_p0)
                # sometimes the given p0 does not have a well-defined y 
                # (eg, blowup rather than steady state) and an exception is
                # thrown out  
                except:
                    seeds_list.remove(seed)
                    _gdss_p0 = []
                _gdss.extend(_gdss_p0)
            index = pd.MultiIndex.from_product([seeds_list, v0idxs], 
                                               names=['seed','v0idx'])    
        else:
            _gdss = _p02gdss(p0)
            index = v0idxs
        
        # almost never used...
        #else:
        #    v0idxs = range(1, len(v0s)+1)
        #    for v0 in v0s:
        #        gds = self.get_geodesic(p0=p0, ptype=ptype, v0=v0, **kwargs)
        #        _gdss.append(gds)
        
        gdss = geodesic.Geodesics(_gdss, index=index)
        return gdss
    get_geodesics.__doc__ += geodesic.Geodesic.__init__.__doc__
    
    ############################################################################
    # plotting
    
    def plot_volume(self, theta1s=None, theta2s=None, p0=None, ndecade=4, npt=10,
                    **kwargs_heatmap):
        """Deprecation warning? ##
        """
        if theta1s is None and theta2s is None:
            if p0 is None:
                p0 = self.p0
            theta1, theta2 = list(p0)
            theta1s = np.logspace(np.log10(theta1)-ndecade/2, np.log10(theta1)-ndecade/2, npt)
            theta2s = np.logspace(np.log10(theta2)-ndecade/2, np.log10(theta2)-ndecade/2, npt)
        
        reload(plotutil)
        plotutil.plot_heatmap(xs=theta1s, ys=theta2s, f=lambda p: np.log10(self.get_volume(p)), 
                              **kwargs_heatmap)
        
        
    def plot_pspace(self, p=None, ndecade=4, npt=10, cfunc=None, **kwargs_scatter):
        """Deprecation warning? ##
        """
        if not all([varid.startswith('log10_') for varid in self.pids]) and\
            not any([varid.startswith('log_') for varid in self.pids]):
            pred = self.get_in_log10p()
            if p is not None:
                p = np.log10(np.array(p))
        else:
            pred = self
            
        if p is None:
            p = pred.p0
            
        thetas = [np.linspace(theta-ndecade/2, theta+ndecade/2, npt+1) for theta in p]
        ps = zip(*[thetass.flatten() for thetass in np.meshgrid(*thetas)])
        if cfunc is None:
            cfunc = lambda p: 0
        cs = map(cfunc, ps)
        reload(plotutil)
        plotutil.scatter3d(*np.transpose(ps), cs=cs, **kwargs_scatter)
        
        
    
    def plot_image(self, theta1s=None, theta2s=None, 
                   p0=None, 
                   ndecade=6, npt=30,  # parameters for grid 
                   #nstep=1000,  # parameters for sampling
                   pts=None, cs=None, 
                   #xyzlabels=None, xyzlims=None, 
                   #filepath='', 
                   #color='b', alpha=0.2, shade=False, edgecolor='none', 
                   **kwargs_surface):
        """Plot the image of predict, aka "model manifold".
        
        Input:
            p0: the center of grid or starting point of sampling
            method: 'grid' or 'sampling' (using Jeffrey's prior)
            decade: how many decades to cover
            npt: number of points for each parameter
            pts: a list of 3-tuples for the points to be marked
        """
        #import ipdb
        #ipdb.set_trace()
        
        if theta1s is None and theta2s is None:
            if p0 is None:
                p0 = self.p0
        
            assert len(p0) == 2, "Dimension of parameter space is larger than 2."
        
            theta1, theta2 = list(p0)
            theta1s = np.linspace(theta1-ndecade/2, theta1+ndecade/2, npt+1)
            theta2s = np.linspace(theta2-ndecade/2, theta2+ndecade/2, npt+1)
            
        reload(plotutil)
        plotutil.plot_surface(self.f, theta1s, theta2s, pts=pts, cs_pt=cs, 
                              **kwargs_surface)
    plot_image.__doc__ += plotutil.plot_surface.__doc__
    
    '''    
        if method == 'grid':
            
            pss = np.meshgrid(*ps)
            
            # make a dummy function that takes in the elements of an input vector 
            # as separate arguments
            
            pss_flat = [thetass.flatten() for thetass in pss] 
            ps = zip(*pss_flat)
            ys = map(self.f, ps)
            yss_flat = zip(*ys)
            yss = [np.reshape(yiss, thetass.shape) for yiss in yss_flat]
            
        if method == 'sampling':
            pass
        
        if xyzlabels is None:
            xyzlabels = self.yids    
        
        if len(yss) > 3:
            #yss = pca(yss, k=3)
            xyzlabels = ['PC1', 'PC2', 'PC3']
        
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")

        ax.plot_surface(*yss, color=color, alpha=alpha, shade=shade, 
                        edgecolor=edgecolor, **kwargs_surface)
        #ax.plot_trisurf(yss[0].flatten(), yss[1].flatten(), yss[2].flatten())
        #ax.scatter(yss[0].flatten(), yss[1].flatten(), yss[2].flatten())
                        
        if pts is not None:
            for idx, pt in enumerate(pts):
                if idx == 0:
                    color = 'r'
                else:
                    color = 'y'    
                ax.scatter(*pt, color=color, alpha=1./(idx+1))  # s=(idx+1)*30) #, alpha=1./(idx+1))
            #ax.scatter(*np.array(pts).T, color='r', alpha=1)
            
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])
        
        if xyzlims:
            ax.set_xlim(xyzlims[0])
            ax.set_ylim(xyzlims[1])
            ax.set_zlim(xyzlims[2])

        
        plt.show()
        plt.savefig(filepath)
        plt.close()
    '''
    
    def plot_image2(self, theta1s=None, theta2s=None, y2c=None, p2c=None,
                    **kwargs_plot):
        """Plot the image of predict, aka "model manifold".
        
        Input:
            p0: the center of grid or starting point of sampling
            method: 'grid' or 'sampling' (using Jeffrey's prior)
            decade: how many decades to cover
            npt: number of points for each parameter
            pts: a list of 3-tuples for the points to be marked
        """
        pts, cs = [], []
        if y2c is not None:
            if not hasattr(y2c, 'keys'):
                pts.extend([item[0] for item in y2c])
                cs.extend([item[1] for item in y2c])
            else:
                pts.extend(y2c.keys())
                cs.extend(y2c.values())
        if p2c is not None:
            if not hasattr(p2c, 'keys'):
                pts.extend([self.f(item[0]) for item in p2c])
                cs.extend([item[1] for item in p2c])
            else:
                pts.extend([self.f(p) for p in p2c.keys()])
                cs.extend(p2c.values())
            
        reload(plotutil)
        if self.ydim == 3:
            plotutil.plot_surface(self.f, theta1s, theta2s, pts=pts, cs_pt=cs, 
                                  **kwargs_plot)
        if self.ydim == 2:
            xs, ys = [], []
            for theta1 in theta1s:
                xys = [self.f([theta1, theta2]) for theta2 in theta2s]
                xs.append([xy[0] for xy in xys])
                ys.append([xy[1] for xy in xys])
            for theta2 in theta2s:
                xys = [self.f([theta1, theta2]) for theta1 in theta1s]
                xs.append([xy[0] for xy in xys])
                ys.append([xy[1] for xy in xys])
            if pts:
                xs.append([pt[0] for pt in pts])
                ys.append([pt[1] for pt in pts])
            plotutil.plot(xs, ys, **kwargs_plot)
            
    plot_image.__doc__ += plotutil.plot_surface.__doc__
    
    
    def plot_homotopy(self):
        pass
    
     
    def scatter_image(self, pgrid, **kwargs_scatter):
        """
        Input:
            pgrid: a map from pids to a list of parameter values
        """
        reload(plotutil)
        
        #if not all([varid.startswith('log10_') for varid in self.pids]) and\
        #    not any([varid.startswith('log_') for varid in self.pids]):
        #    pred = self.get_in_log10p()
        #    if p is not None:
        #        p = np.log10(np.array(p))
        #else:
        #    pred = self
        
        thetas = [pgrid[pid] for pid in self.pids]
        ps = zip(*[thetass.flatten() for thetass in np.meshgrid(*thetas)])
        
        ys = np.array([self(p) for p in ps])
        
        if ys.shape[1] > 3:
            ys = plotutil.pca(ys, k=3)
            xyzlabels = ['PC1', 'PC2', 'PC3']
        else:
            xyzlabels = self.yids
        
        plotutil.scatter3d(*np.array(ys).T, xyzlabels=xyzlabels, 
                           **kwargs_scatter)
        #**kwargs_scatter)
         

    
    def plot_sloppyv_field(self, pid2range, filepath=''):
        """
        Input:
            pid2range: a dict mapping from _two_ pids to their corresponding values
        """
        # order the dict
        pids, pranges = zip(*[(pid, pid2range[pid]) for pid in self.pids 
                             if pid in pid2range])
        
        p1ss, p2ss = np.meshgrid(*pranges)
        shape = p1ss.shape
        vxss, vyss = np.zeros(shape), np.zeros(shape)        
        for i,j in np.ndindex(shape):
            p = [p1ss[i,j], p2ss[i,j]]
            v = self.get_sloppyv(p)
            vxss[i,j] = v[0]
            vyss[i,j] = v[1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.quiver(p1ss, p2ss, vxss, vyss, pivot='middle', 
                  headwidth=3, headlength=3)
        ax.set_xlabel(pids[0])
        ax.set_ylabel(pids[1])
        ax.set_xlim(p1ss.min()*0.8, p1ss.max()*1.1)
        ax.set_ylim(p2ss.min()*0.8, p2ss.max()*1.1)

        plt.savefig(filepath)
        plt.show()
        plt.close()
        
        
    def plot_sloppyvs(self, ps, plabels=None, filepath=''):
        """
        Input:
            ps: a list of parameter vectors
            plabels: a list of labels 
        """
        m, n = len(ps), len(ps[0])
        colors = ['b','g','r','c','m','y','k']
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 1/(m+2)
        for idx, p in enumerate(ps):
            v = self.get_sloppyv(p)
            xs = np.arange(n)
            ax.bar(xs+(idx+1)*width, v, width=width, color=colors[idx], 
                   edgecolor='none')
        if plabels:
            ax.legend(plabels, loc='lower right')
        ax.set_xticks([0]+(np.arange(n)+0.5).tolist()+[n])
        ax.set_xticklabels(['']+self.pids+[''])
        ax.set_ylabel('Components')
        ax.set_ylim(-1,1)
        plt.subplots_adjust(left=0.2)
        plt.savefig(filepath)
        plt.show()
        plt.close()
    
        
    def plot_spectra(self, ps=None, interval=1, filepath='', figsize=None, figtitle='',
                     xylims=None, xylabels=None, subplots_adjust=None, plot_tol=False,
                     **kwargs_plabel):
        """
        Input:
            ps: a list of parameter vectors
            interval: 
            kwargs_plabel: 'labels', 'rotation', 'ha', 'position', etc.
                docstring attached below.
        """
        if ps is None:
            ps = [self.p0]
        ps = ps[::interval]
        
        if kwargs_plabel:
            kwargs_plabel['labels'] = kwargs_plabel['labels'][::interval]
            
        m, n = len(ps), len(ps[0])
        
        if figsize is None:
            figsize = (2*m, 2*n**0.8)
            
        fig = plt.figure(figsize=figsize)  # need to be tuned
        ax = fig.add_subplot(111)
        for idx, p in enumerate(ps):
            sigmas = self.get_spectrum(p)
            for sigma in sigmas:
                #y = np.log10(sigma)
                ax.plot([idx+0.1, idx+0.9], [sigma, sigma], c='k')
            if plot_tol:
                tol = sigmas[0] * max(len(self.pids), len(self.yids)) *\
                    np.finfo(sigmas.dtype).eps
                ax.plot([idx+0.1, idx+0.9], [tol, tol], c='r')
            ax.set_yscale('log')
            if kwargs_plabel:
                ax.set_xticks(np.arange(0.5, m+1, 1), minor=False)
                ax.set_xticklabels(**kwargs_plabel)
                ax.set_xticks(np.arange(0, m, 1), minor=True)
                ax.grid(which='major', alpha=0)
                ax.grid(which='minor', alpha=1, linewidth=1)
                
            else:
                ax.set_xticks([])
        if xylims:
            ax.set_xlim(xylims[0])
            ax.set_ylim(xylims[1])
        else:
            ax.set_xlim(0, m)
        
        if xylabels:
            ax.set_xlabel(xylabels[0])
            ax.set_ylabel(xylabels[1])
        
        if subplots_adjust:
            plt.subplots_adjust(**subplots_adjust)
        
        plt.title(figtitle) 
        
        plt.savefig(filepath)
        plt.show()
        plt.close()
    
    plot_spectra.__doc__ += plt.Axes.set_xticklabels.__doc__
    
    
    def plot_eigvec(self, p=None, idx=None, ax=None, figsize=None, 
                    filepath='', show=True, **kwargs):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            plot_fig = True
        else:
            plot_fig = False
        
        eigvec = self.get_eigenv(p, idx=idx)
        plotutil.barplot(ax=ax, lefts=np.arange(self.N)-0.5, heights=eigvec, 
                         widths=1, cmapname='jet', 
                         xylims=[[-0.5, self.N-0.5],[-1,1]], 
                         xyticks=[[],[-1,0,1]], **kwargs)
        
        if plot_fig:
            if filepath:
                pass
            if show:
                pass
            plotutil.plt.close()
    
        
    def plot_eigvecs(self, ps=None, interval=1, idx_eigvec=-1, 
                     figsize=None, colorscheme='standard',
                     xylims=None, xylabels=None, subplots_adjust=None,
                     plabels=None,
                     
                     pids=None, show_pids='legend', ax=None,
                     xloc_title=0.5,
                     figtitle='', filepath='', show=True, 
                     **kwargs_legend_subplot):
        """
        Input:
            ps: a list of parameter vectors
            interval: 
            show_pids: 'xticklabel' or 'legend'
            plabels: 
        """
        if ps is None:
            ps = [self.p0]
        ps = ps[::interval]
        
        if plabels is not None:
            plabels = plabels[::interval]
            
        m, n = len(ps), len(ps[0])
        
        if figsize is None:
            figsize = (2*m, 2*n**0.8)
        #width = 1/(m+2)  # bar width
        
        if m > 1 and show_pids == 'legend':
            _add_legend_subplot = True
            nsubplot = m + 1
        else:
            _add_legend_subplot = False
            nsubplot = m
        
        if pids is None:
            pids = self.pids
        
        colors = plotutil.get_colors(n, scheme=colorscheme)
        
        fig = plt.figure(figsize=figsize)  # need to be tuned
        
        for i, p in enumerate(ps):
            ax = fig.add_subplot(1, nsubplot, i+1)
            
            ax.bar(np.arange(n)+0.1, self.get_eigenv(p, idx=idx_eigvec), 
                   color=colors, edgecolor='none')
            
            ax.set_ylim(-1,1)
            
            if show_pids == 'xticklabel':
                ax.set_xticks([0]+(np.arange(n)+0.5).tolist()+[n])
                ax.set_xticklabels(['']+pids+[''], rotation=70)
            elif show_pids == 'legend':
                ax.set_xticklabels([])
                if _add_legend_subplot == False:
                    ax.legend(pids, loc='best')
            else:
                raise ValueError("")
        
            if i == m-1:
                ax.yaxis.tick_right()
            else:
                if i != 0:
                    ax.set_yticklabels([])
            
            if plabels is not None:
                ax.set_xlabel(plabels[i])
        
        if _add_legend_subplot:
            legends = kwargs_legend_subplot.get('legends', pids)
            linewidth = kwargs_legend_subplot.get('linewidth', 5)
            loc = kwargs_legend_subplot.get('loc', (1,0.1))
            fontsize = kwargs_legend_subplot.get('fontsize', None)
            
            ax = fig.add_subplot(1, nsubplot, nsubplot)
            for color in colors:
                ax.plot([0],[0], color=color, linewidth=linewidth)
            ax.set_axis_off()
            ax.legend(legends, loc=loc, fontsize=fontsize)
        
        if subplots_adjust:
            plt.subplots_adjust(**subplots_adjust)
        plt.suptitle(figtitle, x=xloc_title)
        plt.savefig(filepath)    
        if show:
            plt.show()
        plt.close()
    
    plot_eigenvs = plot_eigvecs
        
    
    def plot_isocurve(self, p, nstep, dt, uturn=False,  
                      print_step=False, ret=False, **kwargs):
        """1d level set. 2d called 'isosurface'.  
        It doesn't work for isosurfaces (eigenvectors are random).
        
        Input:
            nstep: number of steps
            dt: time interval for one step (unit speed)
        
        """
        assert self.get_rank() == self.N - 1, "Degree of degeneracy has to be 1."

        p0 = p    
        v0 = self.get_eigenv(p0, idx=-1)
        if uturn:
            v0 = -v0
        ps = [p0]
        
        for istep in range(nstep):
            p = p0 + v0 * dt
            ps.append(p)
            
            ## update p and v
            p0 = p
            v = self.get_eigenv(p, idx=-1)
            # the sign of eigenv can randombly flip 
            d_plus = np.linalg.norm(v0-v)
            d_minus = np.linalg.norm(v0-(-v))
            if d_plus < d_minus:
                v0 = v
            else:
                v0 = -v
            
            if print_step:
                print istep
        
        ts = np.arange(nstep+1) * dt
        ps = np.array(ps)
        
        plotutil.plot(ts, ps.T, **kwargs)
        
        if ret:
            return ts, ps
    
    ############################################################################
    # io
    
    def to_pickle(self, filepath):
        from cloud.serialization.cloudpickle import dump
        fh = open(filepath, 'w')
        dump(self, fh)
        fh.close() 
        

    @staticmethod
    def from_pickle(filepath):
        fh = open(filepath, 'r')
        pred = cPickle.load(fh)
        fh.close()
        return pred
        
        

def get_Df_fd(f, rdeltap=1e-4):
    """Get jacobian of f through symmetric finite difference
        
    Input:
        rdelta: relative delta
    """
    
    '''
    def _DF(p):
        jacT = []  # jacobian matrix transpose
        for i, p_i in enumerate(p):
            deltap = np.zeros(self.N)
            deltap_i = p_i * rdelta
            deltap[i] = deltap_i
            p_minus = p - deltap
            p_plus = p + deltap
            jacT.append((self(p_plus)-self(p_minus))/2/deltap_i)
        return np.transpose(jacT)
    '''
    
    def _Df(p):
        jacT = []  # jacobian matrix transpose
        for i, p_i in enumerate(p):
            deltap_i = max(p_i * rdeltap, rdeltap)
            deltap = np.zeros(len(p))
            deltap[i] = deltap_i
            p_plus = p + deltap
            p_minus = p - deltap
            jacT.append((f(p_plus) - f(p_minus))/ 2 / deltap_i)
        jac = np.transpose(jacT)
        return jac
    return _Df

mathsubs = dict(sqrt=np.sqrt, exp=np.exp, log=np.log,
                Sqrt=np.sqrt, Exp=np.exp, Log=np.log,
                arctan=np.arctan, atan=np.arctan,
                sin=np.sin, cos=np.cos,
                pi=np.pi)

def list2predict(l, pids, uids=None, us=None, yids=None, c=None, p0=None):
    """
    pred = list2predict(['exp(-p1*1)+exp(-p2*1)', 'exp(-p1*2)+exp(-p2*2)', 'exp(-p1*1)-exp(-p2*1)'],
                        pids=['p1','p2'], p0=None)
    
    pred = list2predict(['(k1f*C1-k1r*X1)-(k2f*X1-k2r*X2)', 
                         '(k2f*X1-k2r*X2)-(k3f*X2-k3r*C2)'],
                        uids=['X1','X2'],
                        us=butil.get_product([1,2,3],[1,2,3]),
                        pids=['k1f','k1r','k2f','k2r','k3f','k3r'], 
                        c={'C1':2,'C2':1})
                        
    Input:
        c: a mapping
    """
    if c is not None:
        l = [exprmanip.sub_for_vars(s, c) for s in l]
    
    if us is not None:
        l = butil.flatten([[exprmanip.sub_for_vars(s, dict(zip(uids, u))) for u in us] 
                           for s in l])
    
    ystr = str(l).replace("'", "")
    ycode = compile(ystr, '', 'eval')
    
    def f(p):
        return np.array(eval(ycode, dict(zip(pids, p)), mathsubs))
    
    jaclist = []
    for s in l:
        jacrow = [exprmanip.simplify_expr(exprmanip.diff_expr(s, pid))
                  for pid in pids]
        jaclist.append(jacrow)
    jacstr = str(jaclist).replace("'", "")
    jaccode = compile(jacstr, '', 'eval')

    def Df(p):
        return np.array(eval(jaccode, dict(zip(pids, p)), mathsubs))
    
    if p0 is None:
        p0 = [1] * len(pids)
        
    if yids is None:
        yids = ['y%d'%i for i in range(1, len(l)+1)]
        
    return Predict(f=f, Df=Df, p0=p0, pids=pids, yids=yids)


def list2predict2(l, pids, uids=None, us=None, yids=None, c=None, p0=None):
    """
    pred = list2predict(['exp(-p1*1)+exp(-p2*1)', 'exp(-p1*2)+exp(-p2*2)', 'exp(-p1*1)-exp(-p2*1)'],
                        pids=['p1','p2'], p0=None)
    
    pred = list2predict(['(k1f*C1-k1r*X1)-(k2f*X1-k2r*X2)', 
                         '(k2f*X1-k2r*X2)-(k3f*X2-k3r*C2)'],
                        uids=['X1','X2'],
                        us=butil.get_product([1,2,3],[1,2,3]),
                        pids=['k1f','k1r','k2f','k2r','k3f','k3r'], 
                        c={'C1':2,'C2':1})
                        
    Input:
        c: a mapping
    """
    if c is not None:
        l = [exprmanip.sub_for_vars(s, c) for s in l]
    
    ystr = str(l).replace("'", "")
    ycode = compile(ystr, '', 'eval')
    
    def f(p):
        return np.array(eval(ycode, dict(zip(pids, p)), mathsubs))
    
    jaclist = []
    for s in l:
        jacrow = [exprmanip.simplify_expr(exprmanip.diff_expr(s, pid))
                  for pid in pids]
        jaclist.append(jacrow)
        
    if us is not None:
        jaclist = [[[exprmanip.sub_for_vars(jacentry, dict(zip(uids, u))) 
                     for jacentry in jacrow] 
                    for jacrow in jaclist]
                   for u in us]
        jaclist = butil.flatten(jaclist, depth=1)       

    jacstr = str(jaclist).replace("'", "")
    jaccode = compile(jacstr, '', 'eval')

    def Df(p):
        return np.array(eval(jaccode, dict(zip(pids, p)), mathsubs))
    
    if p0 is None:
        p0 = [1] * len(pids)
        
    if yids is None:
        yids = ['y%d'%i for i in range(1, len(l)+1)]
        if us is not None:
            uids = ['u%d'%i for i in range(1, len(us)+1)]
            yids = butil.get_product(yids, uids)
        
    return Predict(f=f, Df=Df, p0=p0, pids=pids, yids=yids)    
    

def str2predict(s, pids, uids, us, c=None, p0=None, yids=None):
    """
    Input:
        us: a list of u's where each u has the same length as uids and has the 
            same order
        c: if given, a mapping from convarid to convarval 
    """
    if c is not None:
        s = exprmanip.sub_for_vars(s, c)
    
    ystr = str([exprmanip.sub_for_vars(s, dict(zip(uids, u))) for u in us]).\
        replace("'", "")
    ycode = compile(ystr, '', 'eval')
    
    def f(p):
        return np.array(eval(ycode, dict(zip(pids, p)), mathsubs))
    
    jaclist = []
    for u in us:
        s_u = exprmanip.sub_for_vars(s, dict(zip(uids, u)))
        jacrow = [exprmanip.simplify_expr(exprmanip.diff_expr(s_u, pid))
                  for pid in pids]
        jaclist.append(jacrow)
    jacstr = str(jaclist).replace("'", "")
    jaccode = compile(jacstr, '', 'eval')

    def Df(p):
        return np.array(eval(jaccode, dict(zip(pids, p)), mathsubs))
    
    if p0 is None:
        p0 = [1] * len(pids)
    
    if yids is None:
        yids = ['u=%s'%str(list(u)) for u in us]
        
    return Predict(f=f, Df=Df, p0=p0, pids=pids, yids=yids)
    
"""    
def stack_mat(matstr, pids, uids, us, c=None, v_or_h='v'):
    #Return a function that takes in p and outputs a 2D numpy array. 
    
    #Input:
    #    matstr: string of a mat (can be expanded to include func)
    #    v_or_h: vertical stack or horizontal stack
    
    #p2u...
    if c is not None:
        matstr = exprmanip.sub_for_vars(matstr, c)
    
    def _replace(matstr, uids, u):
        for uid, u in zip(uids, u):
            matstr = matstr.replace(uid, str(u))
        return matstr
    
    matsstr = str([_replace(matstr, uids, u) for u in us]).replace("'", "")
    matscode = compile(matsstr, '', 'eval')
    
    def Df(p):
        mats = eval(matscode, dict(zip(pids, p)))
        if v_or_h == 'v':
            return np.vstack(mats)
        else:
            return np.hstack(mats)
    return Df
        

from infotopo.models.rxnnet.examples import pathn 

net = pathn.make_net('', n=2, ma_or_mm='mm', KE=False, r=False, add_ratevars=True)
net.set_var_optimizable('b2r', False)
net.compile()

net.get_Ep_str()
net.get_Ex_str()

M = 20

p = np.random.lognormal(size=8)
Cs = np.random.lognormal(size=(M,2))


def Cp2x(C, p):
    net2 = net.copy()
    net2.update(p=p, C1=C[0], C2=C[1])
    return net2.get_s(tol_ss=1e-15).values[0]
xs = [Cp2x(C, p) for C in Cs]
us = np.hstack((np.reshape(xs, (M,1)), Cs))
get_Eps = stack_mat(net.Ep_str, pids=net.pids, uids=['X','C1','C2'], 
                    us=us, v_or_h='v')


nets = []
for C in Cs:
    net2 = net.copy()
    net2.update(p=p, C1=C[0], C2=C[1])
    nets.append(net2)

Eps = np.vstack([net.Ep for net in nets])
CCMs0 = [net.Cs for net in nets]
CCMs = np.zeros((M, net.vdim*M))
for idx, CCM in enumerate(CCMs0):
    CCMs[idx, net.vdim*idx:net.vdim*(idx+1)] = CCM.values.flatten()
"""    
    
# net.FCM, net.CCM, net.FRM, net.CRM, net.CEM, net.PEM
