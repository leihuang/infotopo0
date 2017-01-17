"""
Need to rewrite SloppyCell Bayesian sampling codes because the codes are 
parameter-centric (only preserve the parameter information), and the new perspectives
are data-centric. 

Plus different sampling schemes (such as Jeffreysian and continuous interpolation) 
come into light.

Return Ensemble objects (pens, dens).

A few minor improvements:

print steps
record costs
recalculate hessian?
"""

from __future__ import division
import time
import logging
from collections import OrderedDict as OD, Mapping

import numpy as np
import pandas as pd
#from matplotlib.mlab import PCA  # write my own pca codes?
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from more_itertools import unique_everseen

from util import butil, plotutil
Series, DF = butil.Series, butil.DF

from util.matrix import Matrix



    
class Ensemble(DF):
    
    @property
    def _constructor(self):
        return Ensemble
    
    _metadata = ['ratio']
    
    # specify the commons variable types and their order
    _vartypes = ['p', 'y', 'e', 'z', 'p2', 'y2', 'e2', 'z2']  
    
    def __init__(self, data=None, index=None, columns=None, dtype='float', copy=False):
        """Make a *simple* or *composite* ensemble.
        
        Simple ensembles have ``pandas.Index`` as columns, while
        composite ensembles have ``pandas.MultiIndex`` as columns.
         
        Input:
            varids: {list-like, map-like}
        """
        if isinstance(columns, OD):
            columns = [[(vartype, varid) for varid in varids] for
                       vartype, varids in columns.items()]
            columns = butil.flatten(columns, depth=1)
            columns = pd.MultiIndex.from_tuples(columns)
        super(DF, self).__init__(data=data, index=index, columns=columns, 
                                 dtype='float', copy=copy)
        self.index.name = 'step'
    
    def get_vartypes(self):
        """
        """
        assert self.columns.nlevels == 2  # a composite ensemble
        return list(unique_everseen(self.columns.get_level_values(0)))
    
    def set_vartypes(self, vartypes):
        """
        """
        vartypemap = OD(zip(self.vartypes, vartypes))
        tuples = [(vartypemap(tu[0]), tu[1]) for tu in self.columns.tolist()]
        self.columns = pd.MultiIndex.from_tuples(tuples)
    
    vartypes = property(fget=get_vartypes, fset=set_vartypes)
    
    
    def add(self, row=None, **kwargs):
        """
        Use sparingly as it makes a copy each time ? 
        
        Time it: FIXME **
        
        Input:
            row: {list-like, map-like}
        """
        if row is None:
            row = kwargs
        if isinstance(row, Mapping):
            row = butil.flatten([row[vartype] for vartype in self.vartypes])
        self.loc[self.nrow] = row


    def to_od(self):
        return OD([(vartype, getattr(self, vartype)) 
                   for vartype in self.vartypes])
        

    def set_vartype(self, vartype):
        """In-place convert a simple ensemble to a composite ensemble 
        by making the columns a ``MultiIndex``.
        
        Input:
            vartype: str
            
        Output:
            None (in-place changes)
        """
        tuples = [(vartype, varid) for varid in self.columns]
        columns_new = pd.MultiIndex.from_tuples(tuples)                                        
        self.columns = columns_new
    
        
    def add_ens(self, other=None, **others):
        """
        Input:
            others: 
        """
        ensmap = OD.fromkeys(self._vartypes, Ensemble(index=self.index)) 
        ensmap.update(self.to_od())
        
        if other is not None:
            ensmap.update(other.to_od())
        
        if others:
            for vartype, ens_vartype in others.items():
                ensmap[vartype] = ens_vartype
            
        # remove empty ens
        ensmap = butil.get_submapping(ensmap, f_value=lambda ens: ens.size)
        ens = Ensemble(pd.concat(ensmap.values(), axis=1).values, 
                       index=self.index,
                       columns=butil.chvals(ensmap, lambda ens: ens.columns))
        return ens
    
    
    def uniquify(self, vartype='p'):
        """
        """
        if self.columns.nlevels == 2:
            ens = getattr(self, vartype)
        else:
            ens = self
        return self.iloc[ens.drop_duplicates().index]
        
        
    def predict(self, f, yids=None):
        """
        """
        yids = getattr(f, 'yids', yids)
        assert yids is not None, "yids is None"
            
        nans = Series([np.nan]*len(yids), yids)
        # skip p's that has nan
        _f = lambda p: Series(f(p), yids) if not p.hasnans() else nans
            
        ens_uniq = self.uniquify()
        yens = ens_uniq.apply(_f, axis=1)
        yens.columns.name = None  # it is 'step' otherwise
        for idx in self.index:
            if idx in ens_uniq.index:
                idx_uniq = idx
            else:
                if self.loc[idx].hasnans():
                    yens.loc[idx] = nans
                else:
                    yens.loc[idx] = yens.loc[idx_uniq]
        return yens.sort_index()
    
    # ??
    #def append(self, row):
    #    """
    #    """
    #    self.loc[self.size] = row
        
    
    #def apply(self, pred):
    #    """
    #    """
    #    if set(pred.pids) > set(self.varids):
    #        raise ValueError("...")
    #    else:
    #        def f(row):
    #            vvals = pred(row[pred.pids])
    #            # entra data in self such as energies and costs
    #            misc = row[~row.index.isin(pred.pids)]  
    #            return pd.concat((vvals, misc))
    #        vens = Ensemble(self._.apply(f, axis=1))
    #        return vens
        
        
    #def split(self):
    #    """
    #    Separate energies from parameters. 
    #    """
    #    eids = ['nlprior', 'cost', 'energy']
    #    eens = Ensemble(self.ix[:, eids])
    #    pens = Ensemble(self.ix[:, ~self.columns.isin(eids)])
    #    return eens, pens
    
    
    #@classmethod
    #def join(eens, pens):
    #    pass
        
        
    #def save(self, filepath='', fmt='readable'):
    #    """
    #    need to improve... 
    #    right now can only write 60 lines (pd.options.display.max_rows = 60)
    #    
    #    Input:
    #        fmt: format, 'readable' or 'pickle'
    #    """
    #    if fmt == 'readable':
    #       fh = open(filepath, 'w')
    #        fh.write(self.dat.__str__())
    #        fh.close()
    #    else:  # pickle
    #        pass
    
    
    #@staticmethod
    #def read(filepath='', fmt='readable'):
    #    """
    #    Input:
    #        format: 'readable' or 'pickle'
    #    """
    #    pass
    
    
    #def calc(self, f, varids=None, **kwargs):
    #    return Ensemble(dat=self.apply(f, axis=1), varids=varids, **kwargs)
    
    
    
    def exp(self):
        """
        Exponentiate an ensemble of log-parameters to get one of (bare) parameters.
        
        """
        def _exp(row):
            data, tuples = [], []
            for k, v in row.iteritems():
                if k[0] in ['p', 'p2']:
                    data.append(np.exp(v))
                    tuples.append((k[0], k[1].lstrip('log_')))
                else:
                    data.append(v)
                    tuples.append(k)
            return Series(data, index=pd.MultiIndex.from_tuples(tuples))
        return self.apply(_exp, axis=1)

        
    def pca(self, k=None, plot_eigvals=False, filepath=''):
        """
        """
        # doc for function PCA: https://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml
        p = PCA(np.array(self), standardize=False)
        dat_pca = p.Y + np.array(self.mean())
        if k is not None:
            dat_pca = dat_pca[:,:k]
        else:
            k = dat_pca.shape[1]
        ens_pca = Ensemble(dat_pca, 
                           varids=['pc%d'%i for i in range(1, k+1)]) 
        #zs = [p.Y.T[i].reshape(ys[i].shape)+ys[i].mean() for i in range(len(ys))]
        return ens_pca
    
    
    
    def hist(self, log10=False, figshape=None, filepath='', adjust=None, **kwargs_hist):
        """
        """
        fig = plt.figure()
        if figshape is None:
            figshape = (self.nvar, 1)
        for i in range(self.nvar):
            ax = fig.add_subplot(figshape[0], figshape[1], i+1)
            if log10:
                dat = np.log10(self.iloc[:,i])
                xlabel = 'log10(%s)'%self.columns[i] 
            else:
                dat = self.iloc[:,i]
                xlabel = self.columns[i]
            ax.hist(dat, **kwargs_hist)
            ax.set_xlabel(xlabel)
            ax.set_yticklabels([])
            
        kwargs_adjust = {'wspace':0, 'hspace':0, 'top':0.9, 'bottom':0.1, 
                         'left':0.1, 'right':0.9}    
        if adjust:
            kwargs_adjust.update(adjust)
        plt.subplots_adjust(**kwargs_adjust)
        plt.savefig(filepath)
        plt.show()
        plt.close()
    

    def scatter3d(self, **kwargs):
        """See the doc of plotutil.scatter3d:
        """
        xs, ys, zs = self.iloc[:,0], self.iloc[:,1], self.iloc[:,2]
        plotutil.scatter3d(xs, ys, zs, **kwargs)
        
    scatter3d.__doc__ += plotutil.scatter3d.__doc__
        
        
    def scatter_3d(self, pts=None, xyzlabels=None, xyzlims=None, filepath=''):
        """
        """
        assert self.ncol == 3, "Ensemble is not 3-dimensional"
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")

        ax.scatter(self.iloc[:,0], self.iloc[:,1], self.iloc[:,2], 
                   s=5, color='b', alpha=0.2, edgecolor='none')
                        
        if pts is not None:
            ax.scatter(*np.array(pts).T, color='r', alpha=1)
        
        if xyzlabels is None:
            xyzlabels = self.colvarids
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




    def scatter(self, hist=False, log10=False, pts=None, colors=None,
                figsize=None, adjust=None, labels=None, labelsize=6, filepath='',
                nodiag=True, lims=None):
        """
        Input:
            hist: if True, also plot histograms for the marginal distributions
            filepath:
        """
        #import ipdb
        #ipdb.set_trace()
        n = self.ncol
        assert n > 1, "Cannot do scatterplot with 1d data."
        
        if figsize is None:
            figsize = (n*2, n*2)
        fig = plt.figure(figsize=figsize)
        if n == 2:
            ax = fig.add_subplot(111)
            xs, ys = self.iloc[:,0], self.iloc[:,1]
            ax.scatter(xs, ys, s=1)
            if pts is not None:
                for pt in pts:
                    ax.scatter(*pt, marker='o', color='r', s=10)  # can change the color for diff pts
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())            
            if log10:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel(self.colvarids[0], fontsize=10)
            ax.set_ylabel(self.colvarids[1], fontsize=10)
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            
        if n >= 3:
            if colors is None:
                colors = 'k'
            if labels is None:
                labels = self.colvarids
            for i, j in np.ndindex((n, n)):
                ens_i = self.iloc[:, i]
                ens_j = self.iloc[:, j]
                varid_i = labels[i]
                varid_j = labels[j]
                ax = fig.add_subplot(n, n, i*n+j+1)
                if nodiag:
                    if i == j:
                        ens_i = []
                        ens_j = []
                ax.scatter(ens_j, ens_i, s=2, marker='o', facecolor=colors, lw=0)
                if pts is not None:
                    for pt in pts:
                        ax.scatter([pt[i]],[pt[j]], marker='o', color='r', s=3)  # can change the color for diff pts
                if log10:
                    ax.set_xscale('log', basex=10)
                    ax.set_yscale('log', basey=10)

                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_xlabel(varid_j, fontsize=labelsize)
                    ax.xaxis.set_label_position('top')
                if i == n-1:
                    ax.set_xlabel(varid_j, fontsize=labelsize)
                if j == 0:
                    ax.set_ylabel(varid_i, fontsize=labelsize)
                if j == n-1:
                    ax.set_ylabel(varid_i, fontsize=labelsize, rotation=270)
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.labelpad = 20
                
                if lims is not None:
                    ax.set_xlim(lims[j])
                    ax.set_ylim(lims[i])
        
        kwargs = {'wspace':0, 'hspace':0, 'top':0.9, 'bottom':0.1, 
                  'left':0.1, 'right':0.9}    
        if adjust:
            kwargs.update(adjust)
        plt.subplots_adjust(**kwargs)
        plt.savefig(filepath)
        plt.show()
        plt.close()
    
    
    @classmethod
    def from_csv(cls, *args, **kwargs):
        """Signature is the same as pd.DataFrame.from_csv.
        Use pickling for lossless data persistence.
        """
        ens = DF.from_csv(*args, **kwargs)
        data = ens.values[2:,:].astype(np.float)
        index = pd.Index([int(idx) for idx in ens.index[2:]], name='step')
        vartypes = [vartype.split('.')[0] for vartype in ens.columns]
        varids = []
        for varid in ens.iloc[0]:
            varid = varid.replace('inf', 'np.inf')
            if '(' in varid:
                varids.append(eval(varid))
            else:
                varids.append(varid)
        columns = pd.MultiIndex.from_tuples(zip(vartypes, varids))
        ens = Ensemble(data=data, index=index, columns=columns) 
        return ens
    
    #def concat(self, other, inplace=False):
    #    if inplace:
    #        pass
    #    else:
    #        return Ensemble(dat=pd.concat([self, other], axis=1), varids=self.varids)
    

def pgrid2pens(pids, plists=None, **pid2list):
    """Convert parameter lists to parameter ensembles through gridding.
    
    Input:
        pids: specify the order
        plists: a sequence or a dict
        
    >>>pens = pgrid2pens(['k1', 'k2'], k1=[0.1,1,10], k2=[0.5,1,2])
    >>>pens.values.tolist()
    >>>[[0.5, 0.1],
    >>> [1.0, 0.1],
    >>> [2.0, 0.1],
    >>> [0.5, 1.0],
    >>> [1.0, 1.0],
    >>> [2.0, 1.0],
    >>> [0.5, 10.0],
    >>> [1.0, 10.0],
    >>> [2.0, 10.0]]
    
    
    There might be bug here:
    >>>pens = sampling.pgrid2pens(list('ab'), plists=[[1,2],[1]])
    >>>pens.set_vartype('p')
    >>>hasattr(pens, 'p')  # True
    >>>hasattr(pens, 'p')  # True
    
    >>>pens = sampling.pgrid2pens(list('ab'), plists=[[1,2.],[1]])  # a float, mixed type
    >>>pens.set_vartype('p')
    >>>hasattr(pens, 'p')  # True
    >>>hasattr(pens, 'p')  # False
    """
    if plists is not None:
        pid2list = OD(zip(pids, plists))
    else:
        pid2list = butil.get_submapping(OD(pid2list), pids)
    pgrid = np.meshgrid(*pid2list.values())
    return Ensemble(zip(*[_.flatten() for _ in pgrid]), columns=pids)     


def sampling(func, nstep, p0=None, in_logp=True, seed=None, scheme_sampling='jtj', 
             w1=1, w2=1, temperature=1, stepscale=1, 
             cutoff_singval=0, recalc_sampling_mat=False, interval_print_step=np.inf,
             maxhour=np.inf, filepath='', **kwargs):
    """Perform sampling for predict by prior (a common one is Jeffreys prior), 
    or for residual by posterior with tunable weights of prior and data. 
        
    predict: prior (Jeffrey)
    residual: prior + posterior

    detailed balance:
    Pr(a)Pr(a->b) = Pr(b)Pr(b->a)
    => Pr(a)/Pr(b) = Pr(b->a)/Pr(a->b) 
    = T(b->a)A(b->a)/(T(a->b)A(a->b))

    If using a fixed candidate generating Gaussian density (usually the case),
    T(b->a) = T(a->b), then
    Pr(a)/Pr(b) = A(b->a)/A(a->b)

    Metropolis choice of A(a->b): 
    https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    A(a->b) = min(1, Pr(b)/Pr(a))

    Input:
        func: predict or residual
        p0: 
        sampmat: either 'JtJ' or 'I' or a matrix
        w1 and w2: floats between 0 and 1; weights on the prior and data
            components of posterior distribution respectively;
            Pr(p|d) \propto Pr(p) Pr(d|p) \propto Pr(p) exp(-C(p))
            = exp(log(Pr(p)) - C(p)) = exp(w1 log(Pr(p)) - w2 C(p)), where
            w1 and w2 are 1; 
            when w1 goes from 1 to 0, prior gets increasingly flat (just data); 
            when w2 goes from 1 to 0, data gets increasingly uncertain (just prior);
            so, (w1, w2) = (1, 0) means prior sampling,
                (w1, w2) = (0, 1) means data sampling,
                (w1, w2) = (1, 1) means bayesian sampling (default).
        kwargs: a waste basket
    """
    ## calculate energies (negative log-prior, cost, and their sum)
    def _get_energies(p):  # nstep_tried?
        try:
            if func.prior is None:
                nlprior = 0
            else:
                #print p[0], func.prior(p)
                nlprior = -np.log(func.prior(p))
            cost = _get_cost(p)
        except:
            print "Error in step %d with parameter:\n%s" % (nstep_tried, str(p))
            nlprior, cost = np.inf, np.inf
        energy = nlprior + cost
        return [nlprior, cost, energy]

    ## calculate sampling matrix
    def _get_smat(p, scheme, cutoff_singval, stepscale, temperature):
        if scheme == 'jtj':
            jac = Dfunc(p)
            jtj = np.dot(jac.T, jac)
            smat = _hess2smat(jtj, cutoff_singval, stepscale, temperature)
        if scheme == 'eye':
            smat = Matrix.eye(func.pids) * stepscale

        return smat
    
    ## initializations
    # func
    if in_logp:
        func = func.get_in_logp()
    else:
        func = func

    # Dfunc
    if hasattr(func, 'yids'):  # predict
        Dfunc = func.Df
        _get_cost = lambda p: 0
    if hasattr(func, 'rids'):  # residual
        Dfunc = func.Dr
        _get_cost = lambda p: func.cost(p)

    # seed
    if seed is None:
        seed = int(time.time() % 1e6)
    np.random.seed(seed)
    
    # p
    if p0 is None:
        p0 = func.p0
    else:
        if in_logp:
            p0 = p0.log()

    nstep_accepted, nstep_tried = 0, 0

    # energies
    e0 = _get_energies(p0)

    # ens
    ens = Ensemble(data=[p0.tolist()+e0], 
                   columns=OD([('p', func.pids), 
                               ('e', ['nlprior', 'cost', 'energy'])]))

    t0 = time.time()
    smat = _get_smat(p0, scheme_sampling, cutoff_singval, stepscale, temperature)
    p, e = p0, e0
    
    ## start sampling
    while nstep_tried < nstep and (time.time()-t0) < maxhour*3600:

        if nstep_tried % interval_print_step == 0 and nstep_tried != 0:
            print nstep_tried

        if recalc_sampling_mat:
            smat = _get_smat(p, scheme_sampling, cutoff_singval, stepscale, 
                             temperature)
            
        # trial move
        p2 = p + _smat2deltap(smat)
        e2 = _get_energies(p2)
        nstep_tried += 1

        # basic Metropolis accept/reject step                                                                                                                                 
        if recalc_sampling_mat:
            pass  # not implemented yet; see Gutenkunst's thesis
        else:
            accept = np.random.rand() < np.exp((e[2]-e2[2])/temperature)
        # add p to ens
        if accept:
            ens.add(p=p2, e=e2)
            p, e = p2, e2
            nstep_accepted += 1
        else:
            ens.add(p=p, e=e)

    ens.ratio = nstep_accepted / nstep

    if in_logp:
        ens = ens.exp()
        # ens now has no attribute ratio

    if filepath:
        ens.save(filepath)

    return ens


def _hess2smat(hess, cutoff_singval, stepscale, temperature):
    """
    Convert Hessian to sampling matrix, where sampling matrix is the 
    square root of covariance matrix for random Gaussian vector generation.
    
    Hessian is d^2 C / d p_i d p_j, where C = r^T r / 2. 
    
    Geometrically, the anisotropic elliptical parameter cloud is represented
    by Hessian: its eigenvectors correspond to the axis directions, and the 
    reciprocals of square roots of its eigenvalues correpond to the axis lengths. 
    Columns of sampling matrix correspond to the axis directions with 
    appropriate lengths.
    """
    eigvals, eigvecs = np.linalg.eig(hess)
    singvals = np.sqrt(eigvals)
    singval_min = cutoff_singval * max(singvals)
    lengths = 1.0 / np.maximum(singvals, singval_min)

    ## now fill in the sampling matrix ("square root" of the Hessian)                                                                                                         
    smat = eigvecs * lengths

    # Divide the sampling matrix by an additional factor such                                                                                                                 
    # that the expected quadratic increase in cost will be about 1.                                                                                                           
    # LH: need to understand what is going on in the following block
    cutoff_vals = np.compress(singvals < cutoff_singval, singvals)
    if len(cutoff_vals):
        scale = np.sqrt(len(singvals) - len(cutoff_vals)
                        + sum(cutoff_vals)/cutoff_singval)
    else:
        scale = np.sqrt(len(singvals))
    #print scale
    
    smat /= scale
    smat *= stepscale
    smat *= np.sqrt(temperature)

    return smat


def _smat2deltap(smat):
    randvec = np.random.randn(len(smat))
    deltap = np.dot(smat, randvec)
    return deltap




    
    
    
    

"""
        Generate a Bayesian ensemble of parameter sets consistent with the data in 
        the models. The sampling is done in terms of the logarithm of the parameters. 
       
        Inputs: 
            p0: -- Initial parameter KeyedList to start from  
            hess: -- Hessian of the models 
            nstep: -- Maximum number of Monte Carlo steps to attempt 
     75       max_run_hours -- Maximum number of hours to run 
     76       temperature -- Temperature of the ensemble 
     77       step_scale -- Additional scale applied to each step taken. step_scale < 1 
     78                     results in steps shorter than those dictated by the quadratic 
     79                     approximation and may be useful if acceptance is low. 
     80       sing_val_cutoff -- Truncate the quadratic approximation at eigenvalues 
     81                          smaller than this fraction of the largest. 
     82       seeds -- A tuple of two integers to seed the random number generator 
     83       recalc_hess_alg --- If True, the Monte-Carlo is done by recalculating the 
     84                           hessian matrix every timestep. This signficantly 
     85                           increases the computation requirements for each step, 
     86                           but it may be worth it if it improves convergence. 
     87       recalc_func --- Function used to calculate the hessian matrix. It should 
     88                       take only a log parameters argument and return the matrix. 
     89                       If this is None, default is to use  
     90                       m.GetJandJtJInLogParameteters 
     91       save_hours --- If save_to is not None, the ensemble will be saved to 
     92                         that file every 'save_hours' hours. 
     93       save_to --- Filename to save ensemble to. 
     94       skip_elems --- If non-zero, skip_elems are skipped between each included  
     95                      step in the returned ensemble. For example, skip_elems=1 
     96                      will return every other member. Using this option can 
     97                      reduce memory consumption. 
     98   
     99      Outputs: 
    100       ens, ens_fes, ratio 
    101       ens -- List of KeyedList parameter sets in the ensemble 
           ens_fes -- List of free energies for each parameter set 
           ratio -- Fraction of attempted moves that were accepted 
      
          The sampling is done by Markov Chain Monte Carlo, with a Metropolis-Hasting 
          update scheme. The canidate-generating density is a gaussian centered on the 
          current point, with axes determined by the hessian. For a useful  
          introduction see: 
           Chib and Greenberg. "Understanding the Metropolis-Hastings Algorithm"  
           _The_American_Statistician_ 49(4), 327-335 
           


def scatter(enss, pts=None, show=True, filepath='', **kwargs_scatter):

    Input:
        enss: a list of ens of the same dimension
        dim: 2 or 3; the dimension of the plot
        pts: a list of points to be highlighted
        show: a bool; 
        

    dim = enss[0].nvar
    colors = ['b','g','r','c','m','y','k']
    
    fig = plt.figure()

    if dim == 2:
        ax = fig.add_subplot(111)
        for ens in enss:
            color = colors.pop(0)
            ax.scatter(ens[:,0], ens[:,1], color=color, **kwargs_scatter)
        ax.set_xlabel(ens.varids[0])
        ax.set_ylabel(ens.varids[1])
    
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        for ens in enss: 
            color = colors.pop(0)
            ax.scatter(ens[:,0], ens[:,1], ens[:,2], color=color, **kwargs_scatter)
        ax.set_xlabel(ens.varids[0])
        ax.set_ylabel(ens.varids[1])
        ax.set_zlabel(ens.varids[2])
 
    if show:
        plt.show()
        
    plt.savefig(filepath)



def sampling0(p0, nstep, get_trial_move, get_prior=None, get_cost=None, pids=None, seed=None, 
             maxhour=np.inf, nchain=1, temperature=1, filepath='', save_realtime=False,
             interval_print_step=None):

    Input:
        pdf: probability density function; a function that takes in p, and 
            outputs the probability density or something proportional
        p0:
        get_trial_move: a function that takes in p and number of steps, and
            outputs deltap
    
    Output:
        ens or metaens (MetaEnsemble, if nchain>1)

    if seed is None: 
        seed = int(time.time()%1e6) 
    np.random.seed(seed)
    
    def _get_energies(p, steps_tried):
        try:
            if get_prior is not None:
                energy_pr = -np.log(get_prior(p)) 
            else:
                energy_pr = 0
            if get_cost is not None:
                cost = get_cost(p)
            else:
                cost = 0
            #energy = logprior + cost
        except:
            #import ipdb
            #ipdb.set_trace()
            print "Error in step %d with parameter:\n%s"%(steps_tried, str(p))
            energy_pr, cost = np.inf, np.inf
        return energy_pr, cost, energy_pr+cost
    
    if pids is None:
        pids = list(p0.index)
    ens = Ensemble(varids=pids, energyids=['energy_pr','cost','energy'])
    
    steps_accepted, steps_tried = 0, 0
    t0 = time.time()
        
    p = copy.copy(p0)
    energy_pr, cost, energy = _get_energies(p, steps_tried)
    ens.append(p, [energy_pr, cost, energy])
    
    while steps_tried < nstep and (time.time()-t0) < maxhour*3600:
        if steps_tried % interval_print_step == 0:
            print steps_tried
            
        steps_tried += 1
        
        p2 = get_trial_move(p)
        energy_pr2, cost2, energy2 = _get_energies(p2, steps_tried)    
        
        # Basic Metropolis accept/reject step
        accept = np.random.rand() < np.exp((energy-energy2)/temperature)
        
        if accept:
            ens.append(p2, [energy_pr2, cost2, energy2])
            p, energy_pr, cost, energy = p2, energy_pr2, cost2, energy2 
            steps_accepted += 1
        else:
            ens.append(p, [energy_pr, cost, energy])
        
    ens.ratio = steps_accepted / nstep
        
    if filepath:
        ens.save(filepath)
    
    return ens
 

    if recalc_func is None and log_params: 
        recalc_func = lambda p: m.GetJandJtJInLogParameters(scipy.log(p))[1] 
    else: 
        recalc_func = lambda p: m.GetJandJtJ(p)[1] 
   
    accepted_moves, attempt_exceptions, ratio = 0, 0, scipy.nan 
    start_time = last_save_time = time.time() 
   
    # Calculate our first hessian if necessary 
    if hess is None: 
          hess = recalc_func(curr_params) 
      # Generate the sampling matrix used to generate candidate moves 
    samp_mat = _sampling_matrix(hess, sing_val_cutoff, temperature, step_scale) 
   
    steps_attempted = 0 
    while steps_attempted < steps: 
          # Have we run too long? 
        if (time.time() - start_time) >= max_run_hours*3600: 
            break 
   
          # Generate the trial move from the quadratic approximation 
          deltaParams = _trial_move(samp_mat) 
          # Scale the trial move by the step_scale and the temperature 
          #scaled_step = step_scale * scipy.sqrt(temperature) * deltaParams 
          scaled_step = deltaParams 
   
          if log_params: 
              next_params = curr_params * scipy.exp(scaled_step) 
          else: 
              next_params = curr_params + scaled_step 
   
          try: 
              next_F = m.free_energy(next_params, temperature) 
          except Utility.SloppyCellException, X: 
              logger.warn('SloppyCellException in free energy evaluation at step ' 
                          '%i, free energy set to infinity.' % len(ens)) 
              logger.warn('Parameters tried: %s.' % str(next_params)) 
              attempt_exceptions += 1 
              next_F = scipy.inf 
   
          if recalc_hess_alg and not scipy.isinf(next_F): 
              try: 
                  next_hess = recalc_func(next_params) 
                  next_samp_mat = _sampling_matrix(next_hess, sing_val_cutoff, 
                                                   temperature, step_scale) 
                  accepted = _accept_move_recalc_alg(curr_F, samp_mat,  
                                                     next_F, next_samp_mat,  
                                                     deltaParams, temperature) 
              except Utility.SloppyCellException, X: 
                  logger.warn('SloppyCellException in JtJ evaluation at step ' 
                              '%i, move not accepted.' % len(ens)) 
                  logger.warn('Parameters tried: %s.' % str(next_params)) 
                  attempt_exceptions += 1 
                  next_F = scipy.inf 
                  accepted = False 
          else: 
              accepted = _accept_move(next_F - curr_F, temperature) 
   
          steps_attempted += 1 
          if accepted: 
              accepted_moves += 1. 
              curr_params = next_params 
              curr_F = next_F 
              if recalc_hess_alg: 
                  hess = next_hess 
                  samp_mat = next_samp_mat 
   
          if steps_attempted % (skip_elems + 1) == 0: 
              ens_Fs.append(curr_F) 
              if isinstance(params, KeyedList): 
                  ens.append(KeyedList(zip(param_keys, curr_params))) 
              else: 
                  ens.append(curr_params) 
          ratio = accepted_moves/steps_attempted 
   
          # Save to a file 
          if save_to is not None\ 
             and time.time() >= last_save_time + save_hours * 3600: 
              _save_ens(ens, ens_Fs, ratio, save_to, attempt_exceptions, 
                        steps_attempted) 
              last_save_time = time.time() 
    if save_to is not None: 
          _save_ens(ens, ens_Fs, ratio, save_to, attempt_exceptions,  
                   steps_attempted) 
  
    return ens, ens_Fs, ratio 




"""