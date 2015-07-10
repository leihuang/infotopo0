"""
Add Mark's geometry-accelerated LM algorithm?
"""

from __future__ import division
from collections import OrderedDict as OD

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from SloppyCell import lmopt

from util import butil
Series, DF = butil.Series, butil.DF

from matrix import Matrix

import sampling
reload(sampling)


class Residual(object):
    def __init__(self, pred=None, dat=None, prior=None):
        """
        Input:
            prior: if given, a function
        """
        def _r(p=None):
            if p is None:
                p = pred.p0
            y = pred(p)
            Y = dat['mean'][y.index]
            sigma = dat['sigma'][y.index]
            return (y-Y)/sigma
        
        def _Dr(p=None):
            if p is None:
                p = pred.p0
            jac = pred.Df(p)
            sigma = dat['sigma'][jac.index]
            jac_r = jac.divide(sigma, axis=0)
            return jac_r
        
        self.r = _r
        self.Dr = _Dr
        self.pids = pred.pids
        self.rids = pred.yids
        self.p0 = pred.p0
        self.prior = prior
        self.pred = pred
        self.dat = dat
        
    
    def __call__(self, p=None):
        return self.r(p=p)
    
    
    def __repr__(self):
        return "pids: %s\nrids: %s\np0:\n%s\ny:\n%s"%\
            (str(self.pids), str(self.rids), str(self.p0), str(self.dat))
    
    
    def get_in_logp(self):
        """
        """
        pred, dat = self.pred, self.dat
        pred_logp = pred.get_in_logp()
        return Residual(pred_logp, dat)
    
    
    def cost(self, p=None):
        return _r2cost(self(p))
    
    
    def set_prior_gaussian(self, means, sigmas, log=True):
        """
        """
        #_prior = lambda p: sp.stats....
        #self.prior = prior
        pass
    
    
    def scale_prior_gaussian(self, k):
        """Scale prior distribution."""
        pass 
    
    
    def scale_sigma(self, k):
        """Scale posterior distribution."""
        pass
    
    
    def fit_scipy(self, p0=None, in_logp=True, *args, **kwargs):
        """
        """
        if p0 is None:
            p0 = self.p0
        else:
            p0 = Series(p0, self.pids)
            
        if in_logp:
            res = self.get_in_logp()
            p0 = p0.log()
        else:
            res = self
        
        kwargs = butil.get_submapping(kwargs, keys=['full_output', 'col_deriv', 
            'ftol', 'xtol', 'gtol', 'maxfev', 'epsfcn', 'factor', 'diag'])
        kwargs_ls = kwargs.copy()
        kwargs_ls['full_output'] = True

        p, cov, infodict, mesg, ier = leastsq(res, p0, Dfun=res.Dr, 
                                              **kwargs_ls)
        
        out = Series(OD([('p', Series(p, res.pids)),
                         ('cost', np.linalg.norm(infodict['fvec'])**2/2),
                         ('covmat', Matrix(cov, rowvarids=res.pids, colvarids=res.pids)), 
                         ('nfev', infodict['nfev']),
                         ('r', Series(infodict['fvec'], res.rids)), 
                         ('mesg', mesg), ('ier', ier)]))
        
        if in_logp:
            out['p'] = out.p.exp()
            # out['covmat'] = 
        
        if not kwargs.get('full_output', True):
            out = out[['p', 'ier']] 
        
        return out
    fit_scipy.__doc__ += leastsq.__doc__
                     
    
    
    def fit_sloppycell(self, p0=None, in_logp=True, *args, **kwargs):
        """Get the best fit using Levenberg-Marquardt algorithm.
        
        Input:
            p0: initial guess
            in_logp: optimizing in log parameters
            *args and **kwargs: additional parmeters to be passed to 
                SloppyCell.lm_opt.fmin_lm, whose docstring is appended below: 
        
        Output:
            out: a Series
        """
        if p0 is None:
            p0 = self.p0
        else:
            p0 = Series(p0, self.pids)
        
        if in_logp:
            res = self.get_in_logp()
            p0 = p0.log()
        else:
            res = self
        
        keys = ['args', 'avegtol', 'epsilon', 'maxiter', 'full_output', 'disp', 
                'retall', 'lambdainit', 'jinit', 'trustradius']
        kwargs_lm = butil.get_submapping(kwargs, f_key=lambda k: k in keys)
        kwargs_lm['full_output'] = True
        kwargs_lm['retall'] = True
        p, cost, nfev, nDfev, convergence, lamb, Df, ps =\
            lmopt.fmin_lm(f=res.r, x0=p0, fprime=res.Dr, *args, **kwargs_lm)
            #lmopt.fmin_lm(f=res.r, x0=p0, fprime=None, *args, **kwargs_lm)
        
        if in_logp:
            p = np.exp(p)
            ps = np.exp(np.array(ps))
        
        p = Series(p, self.pids)
        ps = DF(ps, columns=self.pids)
        ps.index.name = 'step'
        
        out = Series(OD([('p', p), ('cost', cost)]))
        if kwargs.get('full_output', False):
            out['nfev'] = nfev
            out['nDfev'] = nDfev
            out['convergence'] = convergence
            out['lamb'] = lamb
            out['Df'] = Df
        if kwargs.get('retall', False):
            out['ps'] = ps
        return out
    fit_sloppycell.__doc__ += lmopt.fmin_lm.__doc__    


    def fit_custom(self, p0=None, in_logp=True,
                   maxnstep=1000, ret_full=False, ret_steps=False, disp=False, 
                   lamb0=0.001, tol=1e-6, k_up=10, k_down=10, ndone=5, **kwargs):
        """
        
        Input:
            k_up and k_down: parameters used in tuning lamb at each step;
                in the traditional scheme, typically 
                    k_up = k_down = 10;
                in the delayed gratification scheme, typically 
                    k_up = 2, k_down = 10 (see, [1])
        
        grad C = Jt * r
        J = U * S * Vt
         ______     ______  
        |      |   |      |  ______   ______
        |      |   |      | |      | |      |
        |   J  | = |   U  | |   S  | |  Vt  |
        |      |   |      | |______| |______|
        |______|   |______|
        
        Vt * V = V * Vt = I
        Ut* U = I =/= U * Ut
        JtJ = (V * S * Ut) * (U * S * Vt) = V * S^2 * Vt
        
         ______     ____________   ______
        |      |   |            | |      |  ______
        |      |   |            | |      | |      |
        |      | = |            | |      | |      |
        |      |   |            | |      | |______|
        |______|   |____________| |______|
        
        
        
        Gradient Descent step: 
            delta p = - grad C   
            
        Gauss Newton step:
            delta p = - (JtJ).I * grad C

        Levenberg Marquardt step:
            delta p = - (JtJ + lamb * I).inv * grad C
                    = - (V * (S^2 + lamb * I) * Vt).inv * Jt * r
                    = - (V * (S^2 + lamb * I).inv * Vt) * V * S * Ut * r
                    = - V * (S^2 + lamb * I).inv * S * Ut * r
        
        References:
        [1] Transtrum
        [2] Numerical Recipes
        """
        if p0 is None:
            p0 = self.p0
        else:
            p0 = Series(p0, self.pids)

        if in_logp:
            res = self.get_in_logp()
            p0 = p0.log()
        else:
            res = self
        
        if maxnstep is None :
            maxnstep = len(res.pids) * 100

        nstep = 0
        nfev = 0
        nDfev = 0  

        p = p0
        lamb = lamb0
        done = 0
        accept = True
        convergence = False
        
        r = res(p0)
        cost = _r2cost(r)        
        nfev += 1

        if ret_steps:
            ps = DF([p0], columns=res.pids)
            deltaps = DF(columns=res.pids)
            costs = Series([cost], name='cost')
            lambs = Series([lamb], name='lamb')
            ps.index.name = 'step'
            costs.index.name = 'step'
            lambs.index.name = 'step'
        
        while not convergence and nstep < maxnstep:
            
            if accept:
                jac = res.Dr(p)
                U, S, Vt = jac.svd(to_mat=True)
                nDfev += 1            
            
            
            deltap = - Vt.T * (S**2 + lamb * Matrix.eye(res.pids)).I * S * U.T * r
            deltap = deltap[0]  # convert 1-d DF to series
            p2 = p + deltap
            nstep += 1
            
            r2 = res(p2)
            cost2 = _r2cost(r2)
            nfev += 1
            
            if np.abs(cost - cost2) < max(tol, cost * tol):
                done += 1
                
            if cost2 < cost:
                accept = True
                lamb /= k_down
                p = p2
                r = r2
                cost = cost2    
            else:
                accept = False
                lamb *= k_up
            
            if ret_steps:
                ps.loc[ps.nrow] = p
                deltaps.loc[deltaps.nrow] = deltap
                costs.loc[costs.size] = cost
                lambs.loc[lambs.size] = lamb
                
            if done == ndone:
                convergence = True
                # lamb = 0
                
        if in_logp:
            p = p.exp()
            if ret_steps:
                ps = np.exp(ps)
                ps.columns = self.pids
                
            
        out = Series(OD([('p', p), ('cost', cost)]))
        if ret_full:
            out['nfcall'] = nfev
            out['nDfcall'] = nDfev
            out['convergence'] = convergence
            out['nstep'] = nstep
        if ret_steps:
            out['ps'] = ps
            out['deltaps'] = deltaps
            out['costs'] = costs
            out['lambs'] = lambs

        return out


    fit = fit_scipy
    
    
    def sampling(self, p0=None, **kwargs):
        """
        Input:
            p0: starting point of sampling, usually the best fit
            
        """
        ens = sampling.sampling(self, p0=p0, **kwargs)
        return ens
    
    
    def plot_cost_contour(self, theta1s=None, theta2s=None, ndecade=4, npt=100, 
                          show=True, filepath=''):
        """
        """
        if theta1s is None:
            theta1s = np.logspace(np.log10(self.p0[0])-ndecade/2, 
                                  np.log10(self.p0[0])+ndecade/2, npt)
        if theta2s is None:
            theta2s = np.logspace(np.log10(self.p0[1])-ndecade/2, 
                                  np.log10(self.p0[1])+ndecade/2, npt)
         
        theta1ss, theta2ss = np.meshgrid(theta1s, theta2s)
        def _get_cost(*p):
            return self.cost(p)
        costss = _get_cost(theta1ss, theta2ss)
        
        print costss.min(), costss.max()
        
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        plt.contourf(theta1ss, theta2ss, costss, levels=np.logspace(-8,1,20))
        
        if show:
            plt.show()
        if filepath:
            plt.savefig(filepath)
        plt.close()
   
def _r2cost(r):
    """
    """
    return np.linalg.norm(r)**2 / 2
        
         
    """
                p_fit = np.exp([0]), np.exp(np.array(p_fit[1]))
            else:
                p_fit = np.exp([p_fit])
        
        
        if kwargs.get('full_output', False):
            p_fit, 
        else:
            p_fit = out
        
        if in_logp:
            if kwargs.get('retall', False):
                p_fit = np.exp(p_fit[0]), np.exp(np.array(p_fit[1]))
            else:
                p_fit = np.exp([p_fit])
        p_fit = Series(p_fit, self.pids)
        
        if kwargs.get('full_output', False):
            return cost, p_fit, nfcall, nDfcall, convg, lamb, Df
        else:  
            if 
            return self.cost(p_fit), p_fit
    
    @staticmethod
    def _hess_to_sampling_mat(hess, cutoff_singval=0, temperature=1, 
                              stepscale=1):

        What is a sampling matrix?

        U, singvals, Vt = np.linalg.svd(0.5*hess) 

        singval_min = cutoff_singval * max(singvals) 

        D = 1.0 / np.maximum(singvals, singval_min) 
        
        ## now fill in the sampling matrix ("square root" of the Hessian) 
        sampmat = Vt.T * np.sqrt(D) 

        # Divide the sampling matrix by an additional factor such 
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
        
        Bayesian sampling...
        
        Explain sampling matrix...
        
        Input:
            p0: 
            nstep:
            jtj0: either evaluated at the best fit or from pca of a preliminary sampling
            recalc_sampling_mat: if True...
            in_logp: do I want it?
            kwargs: a placeholder
        
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

        ens = sampling.Ensemble(varids=res.pids+['nlprior','cost','energy'])
        
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
            
    #    return sampling.sampling(p0=p0, get_prior=get_prior, get_cost=get_cost,
    #                             trialmove=trialmove, pids=self.pids,
    #                             nstep=nstep, seed=seed, 
    #                             interval_print_step=interval_print_step,
    #                             filepath=filepath)
        
        
        
    """