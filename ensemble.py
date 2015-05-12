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
import copy

import numpy as np
import pandas as pd
from matplotlib.mlab import PCA  # write my own pca codes?
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



"""
def sampling_jeffreysian(pred, ):

    pass



def sampling_bayesian(fit):

    pass




def sampling_old(fit, nstep, w1=1, w2=1):

    Input:
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
            

    pass
"""


class Ensemble(object):
    """
    """
    
    def __init__(self, dat=None, varids=None, ens=None, **kwargs):
        """
        Need to think about a better way to name and store the energies... 
        
        Input:
            dat: ensemble of variables
        """
        if ens is not None:
            dat = ens._.values
            varids = ens.varids
            
        if not isinstance(dat, pd.DataFrame):
            dat = pd.DataFrame(dat, columns=varids, dtype='float')
            dat.index.name = 'step'
        else:
            # data.columns.dtype would be 'int64' if 
            # the columns are automatically created (eg, [0,1,2])
            if dat.columns.dtype != int and varids is None:
                varids = list(dat.columns)
                
        self._ = dat
        self.varids = varids
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)
        
        
    def __getattr__(self, attr):
        return getattr(self._, attr)
        

    def __repr__(self):
        return self._.__repr__()
    
    
    def __getitem__(self, key):
        """
        Slicing/indexing first rows then columns
        """
        return Ensemble(self._.ix[key])

        
    @property
    def nvar(self):
        return len(self.varids)
    
    
    @property
    def size(self):
        return self.shape[0]
    
    
    # ??
    def append(self, row):
        """
        """
        self.loc[self.size] = row
        
    
    def apply(self, pred):
        """
        """
        if set(pred.pids) > set(self.varids):
            raise ValueError("...")
        else:
            def f(row):
                vvals = pred(row[pred.pids])
                # entra data in self such as energies and costs
                misc = row[~row.index.isin(pred.pids)]  
                return pd.concat((vvals, misc))
            vens = Ensemble(self._.apply(f, axis=1))
            return vens
        
        
    def split(self):
        """
        Separate energies from parameters. 
        """
        eids = ['nlprior', 'cost', 'energy']
        eens = Ensemble(self.ix[:, eids])
        pens = Ensemble(self.ix[:, ~self.columns.isin(eids)])
        return eens, pens
    
    
    @classmethod
    def join(eens, pens):
        pass
        
        
    def save(self, filepath='', fmt='readable'):
        """
        need to improve... 
        right now can only write 60 lines (pd.options.display.max_rows = 60)
        
        Input:
            fmt: format, 'readable' or 'pickle'
        """
        if fmt == 'readable':
            fh = open(filepath, 'w')
            fh.write(self.dat.__str__())
            fh.close()
        else:  # pickle
            pass
    
    
    @staticmethod
    def read(filepath='', fmt='readable'):
        """
        Input:
            format: 'readable' or 'pickle'
        """
        pass
    
    
    def calc(self, f, varids=None, **kwargs):
        return Ensemble(dat=self.apply(f, axis=1), varids=varids, **kwargs)
        
        
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
    

    def scatter(self, hist=False, log10=False, pts=None, adjust=None, filepath=''):
        """
        Input:
            hist: if True, also plot histograms for the marginal distributions
            filepath:
        """
        #import ipdb
        #ipdb.set_trace()
        n = self.nvar
        fig = plt.figure(figsize=(n*2, n*2))
        if n == 1:
            raise ValueError("Cannot do scatterplot with 1d data.")
        if n == 2:
            ax = fig.add_subplot(111)
            xs, ys = self.iloc[:,0], self.iloc[:,1]
            ax.scatter(xs, ys, s=1)
            if pts is not None:
                for pt in pts:
                    ax.scatter(*pt, marker='o', color='r', s=5)  # can change the color for diff pts
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(ys.min(), ys.max())            
            if log10:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel(self.varids[0], fontsize=10)
            ax.set_ylabel(self.varids[1], fontsize=10)
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
        if n >= 3:
            for i, j in np.ndindex((n, n)):
                ens_i = self.iloc[:, i]
                ens_j = self.iloc[:, j]
                varid_i = self.varids[i]
                varid_j = self.varids[j]
                ax = fig.add_subplot(n, n, i*n+j+1)
                ax.scatter(ens_j, ens_i, s=1, marker='o', facecolor='k', lw=0)
                if pts is not None:
                    for pt in pts:
                        ax.scatter([pt[i]],[pt[j]], marker='o', color='r', s=3)  # can change the color for diff pts
                if log10:
                    ax.set_xscale('log', basex=10)
                    ax.set_yscale('log', basey=10)
    
                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_xlabel(varid_j, fontsize=6)
                    ax.xaxis.set_label_position('top')
                if i == n-1:
                    ax.set_xlabel(varid_j, fontsize=6)
                if j == 0:
                    ax.set_ylabel(varid_i, fontsize=6)
                if j == n-1:
                    ax.set_ylabel(varid_i, fontsize=6, rotation=180)
                    ax.yaxis.set_label_position('right')
        
        kwargs = {'wspace':0, 'hspace':0, 'top':0.9, 'bottom':0.1, 
                  'left':0.1, 'right':0.9}    
        if adjust:
            kwargs.update(adjust)
        plt.subplots_adjust(**kwargs)
        plt.savefig(filepath)
        plt.show()
        plt.close()
    
    
    
    def concat(self, other, inplace=False):
        if inplace:
            pass
        else:
            return Ensemble(dat=pd.concat([self, other], axis=1), varids=self.varids)
    
    
    def exp(self):
        """
        """
        idxs_log = [idx for idx, varid in enumerate(self.varids) 
                    if varid.startswith('log_')]
        dat_unlogged = np.array(self._)
        dat_unlogged[:,idxs_log] = np.exp(dat_unlogged[:,idxs_log])
        varids_unlogged = [varid.lstrip('log_') for varid in self.varids]
        ens_unlogged = Ensemble(dat=dat_unlogged, varids=varids_unlogged, 
                                ratio=self.ratio)
        return ens_unlogged
    
    
    
def scatter(enss, pts=None, show=True, filepath='', **kwargs_scatter):
    """
    Input:
        enss: a list of ens of the same dimension
        dim: 2 or 3; the dimension of the plot
        pts: a list of points to be highlighted
        show: a bool; 
        
    """
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
    
    

"""
        Generate a Bayesian ensemble of parameter sets consistent with the data in 
        the model. The sampling is done in terms of the logarithm of the parameters. 
       
        Inputs: 
            p0: -- Initial parameter KeyedList to start from  
            hess: -- Hessian of the model 
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
"""

def sampling(p0, nstep, get_trial_move, get_prior=None, get_cost=None, pids=None, seed=None, 
             maxhour=np.inf, nchain=1, temperature=1, filepath='', save_realtime=False,
             interval_print_step=None):
    """
    Input:
        pdf: probability density function; a function that takes in p, and 
            outputs the probability density or something proportional
        p0:
        get_trial_move: a function that takes in p and number of steps, and
            outputs deltap
    
    Output:
        ens or metaens (MetaEnsemble, if nchain>1)
    """
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
 
    """
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