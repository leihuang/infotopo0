"""
Add Mark's geometry-accelerated LM algorithm?
"""

from __future__ import division
import time
import copy

import numpy as np
import pandas as pd

from SloppyCell import lmopt

import butil
reload(butil)
 
import parameter
import ensemble
reload(parameter)
reload(ensemble)


"""
class Fitting(object):

    def __init__(self, pred, dat):
        self.pred = pred
        self.dat = dat
        
    
    def get_best_fit(self, p0=None, in_logp=True, *args, **kwargs):

        LM algorithm.
        
        Input:
            in_logp: optimizing in log parameters
            *args & **kwargs: additional parmeters to be passed to 
                SloppyCell.lm_opt.fmin_lm, whose docstring is appended below: 
        

        if in_logp:
            #f = lambda lp: self.pred(np.exp(lp))
            #Df = lambda lp: self.pred.Df(np.exp(lp))*np.exp(lp)
            #p0 = np.log(p0)
            pred = self.pred.get_in_logp()
        else:
            #f = lambda p: self.pred(p)
            #Df = lambda p: self.pred.Df(p)
            pred = self.pred
        
        # divided by the uncertainties?
        # Df has to change as well...
        res = lambda p: pred(p) - self.dat  
            
        if p0 is None:
            p0 = pred.p0
        elif in_logp:
            p0 = np.log(p0)
        
        p_opt = lmopt.fmin_lm(f=res, x0=p0, fprime=pred.Df, *args, **kwargs)
         
        if in_logp:
            p_opt = np.exp(p_opt)
        p_opt = pd.Series(p_opt, index=self.pred.pids)
        
        return p_opt 
    
    get_best_fit.__doc__ += lmopt.fmin_lm.__doc__     
    
    
    def sampling(self, p0=None, hess=None, in_logp=True, 
                 nstep=np.inf, max_run_hours=np.inf, 
                 temperature=1.0, stepscale=1.0, 
                 cutoff_singval=0, seed=None, 
                 recalc_hess_alg = False, recalc_func=None, 
                 save_hours=np.inf, save_to=None, 
                 skip_elems = 0, log_params=True): 
        
        pass
    
    
    def cost(self, p=None):

        return np.linalg.norm(self.pred(p)-self.dat)/2
"""
    
    
class Residual(object):
    def __init__(self, pred=None, dat=None, res=None, prior=None):
        """
        Input:
            prior: if given, a function
        """
        if res is not None:
            pred, dat = res.pred, res.dat
            
        def r(p=None):
            if p is None:
                p = pred.p0
            y = pred(p)
            Y = dat['mean'][y.index]
            sigma = dat['sigma'][y.index]
            return (y-Y)/sigma
        
        def Dr(p=None):
            if p is None:
                p = pred.p0
            jac = pred.Df(p)
            sigma = dat['sigma'][jac.index]
            jac_r = jac.divide(sigma, axis=0)
            return jac_r
        
        self.r = r
        self.Dr = Dr
        self.pids = pred.pids
        self.rids = pred.dids
        self.p0 = pred.p0
        self.prior = prior
        self.pred = pred
        self.dat = dat
        
    
    def __call__(self, p):
        return self.r(p)
    
    
    def __repr__(self):
        return "Parameter ids: %s\nResidual ids: %s\np0:\n%s\nData:\n%s"%\
            (str(self.pids), str(self.rids), str(self.p0), str(self.dat))
    
    
    def get_in_logp(self):
        """
        """
        pred, dat = self.pred, self.dat
        pred_logp = pred.get_in_logp()
        return Residual(pred_logp, dat)
    
    
    def cost(self, p):
        return np.linalg.norm(self(p))**2/2
    
    
    def set_prior_gaussian(self, means, sigmas, log=True):
        """
        """
        #_prior = lambda p: sp.stats....
        #self.prior = prior
        pass
    
    
    def scale_prior_gaussian(self, k):
        """
        Scale prior distribution...
        """
        pass 
    
    
    def scale_sigma(self, k):
        """
        Scale posterior distribution...
        """
        pass
    
    
    def fit(self, p0=None, in_logp=True, *args, **kwargs):
        """
        Get the best fit using LM algorithm.
        
        Input:
            in_logp: optimizing in log parameters
            *args and **kwargs: additional parmeters to be passed to 
                SloppyCell.lm_opt.fmin_lm, whose docstring is appended below: 
                
        """
        if p0 is None:
            p0 = self.p0
        else:
            p0 = parameter.Parameter(p0, self.pids)
        
        if in_logp:
            res = self.get_in_logp()
            p0 = p0.log()
        else:
            res = self
        
        keys = ['args', 'avegtol', 'epsilon', 'maxiter', 'full_output', 'disp', 
                'retall', 'lambdainit', 'jinit', 'trustradius']
        kwargs = butil.get_submapping(kwargs, f_key=lambda key: key in keys) 
        p_opt = lmopt.fmin_lm(f=res.r, x0=p0, fprime=res.Dr, *args, **kwargs)
        
        if in_logp:
            p_opt = np.exp(p_opt)
        p_opt = parameter.Parameter(p_opt, self.pids)
         
        return self.cost(p_opt), p_opt
    fit.__doc__ += lmopt.fmin_lm.__doc__     
    
    
    @staticmethod
    def _hess_to_sampling_mat(hess, cutoff_singval=0, temperature=1, 
                              stepscale=1):
        """
        What is a sampling matrix?
        """
        U, singvals, Vt = np.linalg.svd(0.5*hess) 

        singval_min = cutoff_singval * max(singvals) 

        D = 1.0 / np.maximum(singvals, singval_min) 
        
        ## now fill in the ensemble matrix ("square root" of the Hessian) 
        sampmat = Vt.T * np.sqrt(D) 

        # Divide the ensemble matrix by an additional factor such 
        # that the expected quadratic increase in cost will be about 1.
        # LH: need to understand what is going on in the following block 
        cutoff_vals = np.compress(singvals < cutoff_singval, singvals) 
        if len(cutoff_vals): 
            scale = np.sqrt(len(singvals) - len(cutoff_vals) 
                            + sum(cutoff_vals)/cutoff_singval) 
        else: 
            scale = np.sqrt(len(singvals)) 
        sampmat /= scale
         
        sampmat *= stepscale 
        sampmat *= np.sqrt(temperature) 

        return sampmat
    
    
    @staticmethod
    def _get_trial_move(p, sampmat):
        randvec = np.random.randn(len(sampmat))
        deltap = np.dot(sampmat, randvec)
        p2 = p + deltap
        return p2 
    
    
    def sampling(self, p0, nstep, jtj0=None, in_logp=True, 
                 recalc_sampling_mat=False,
                 seed=None,  
                 maxhour=np.inf, temperature=1,
                 
                 cutoff_singval=0, stepscale=1, 
                 interval_print_step=None, filepath='',
                 **kwargs):
        """
        Bayesian ensemble...
        
        Explain sampling matrix...
        
        Input:
            p0: 
            nstep:
            jtj0: either evaluated at the best fit or from pca of a preliminary ensemble
            recalc_sampling_mat: if True...
            in_logp: do I want it?
            kwargs: a placeholder
        """
        #import ipdb
        #ipdb.set_trace()

        if seed is None: 
            seed = int(time.time() % 1e6) 
        np.random.seed(seed)
        
        if in_logp and not self.pids[0].startswith('log_'):
            res = self.get_in_logp()
        else:
            res = self
            
        def _get_energies(p, steps_tried):
            try:
                if res.prior is None:
                    nlprior = 0
                else:
                    nlprior = -np.log(res.prior(p)) 
                cost = res.cost(p)
            except:
                print "Error in step %d with parameter:\n%s"%(steps_tried, str(p))
                nlprior, cost = np.inf, np.inf
            energy = nlprior + cost
            return nlprior, cost, energy

        ens = ensemble.Ensemble(varids=res.pids+['nlprior','cost','energy'])
        
        steps_accepted, steps_tried = 0, 0
        t0 = time.time()

        if p0 is None:
            p0 = res.p0
        else:
            if in_logp:
                p0 = p0.log()
        
        nlprior0, cost0, energy0 = _get_energies(p0, steps_tried)
        print p0.tolist()+[nlprior0, cost0, energy0]
        ens.append(p0.tolist()+[nlprior0, cost0, energy0])
        
        if jtj0 is None:
            jac = res.Dr(p0)
            jtj = np.dot(jac.T, jac)
        else:
            jtj = jtj0
        sampmat = Residual._hess_to_sampling_mat(jtj, cutoff_singval=cutoff_singval, 
                                                 temperature=temperature, 
                                                 stepscale=stepscale)
        
        p, nlprior, cost, energy = p0, nlprior0, cost0, energy0 
        
        while steps_tried < nstep and (time.time()-t0) < maxhour*3600:
            if interval_print_step is not None and\
                steps_tried % interval_print_step == 0:
                print steps_tried
            
            if recalc_sampling_mat:
                jac = res.Dr(p)
                jtj = np.dot(jac.T, jac)
                sampmat = Residual._hess_to_sampling_mat(jtj)
            
            p2 = Residual._get_trial_move(p, sampmat)    
            steps_tried += 1
            nlprior2, cost2, energy2 = _get_energies(p2, steps_tried)    
            
            # Basic Metropolis accept/reject step
            if recalc_sampling_mat:
                pass  # not implemented yet
            else:
                accept = np.random.rand() < np.exp((energy-energy2)/temperature)
            
            if accept:
                ens.append(p2.tolist()+[nlprior2, cost2, energy2])
                p, nlprior, cost, energy = p2, nlprior2, cost2, energy2 
                steps_accepted += 1
            else:
                ens.append(p.tolist()+[nlprior, cost, energy])
            
        ens.ratio = steps_accepted / nstep
        
        if in_logp:
            ens = ens.exp()
            
        if filepath:
            ens.save(filepath)
        
        return ens
            
    #    return ensemble.ensemble(p0=p0, get_prior=get_prior, get_cost=get_cost,
    #                             trialmove=trialmove, pids=self.pids,
    #                             nstep=nstep, seed=seed, 
    #                             interval_print_step=interval_print_step,
    #                             filepath=filepath)
        
        
        
    