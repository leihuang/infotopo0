# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#### THIS PART IS ALREADY IN SLOPPY CELL ENSEMBLES.PY ####
import logging
logger = logging.getLogger('Ensembles')
import copy
import shutil
import sys
import time

import scipy
import scipy.linalg
import scipy.stats
import scipy.fftpack

import SloppyCell.KeyedList_mod as KeyedList_mod
KeyedList = KeyedList_mod.KeyedList
import SloppyCell.Utility as Utility

from SloppyCell import HAVE_PYPAR, my_rank, my_host, num_procs
if HAVE_PYPAR:
    import pypar

# <codecell>

from scipy.linalg import svdvals
from scipy.optimize import leastsq

#### CAN I USE NUMPY??? ####
import numpy as np

def jeffreys_prior_ensemble(m, params, jac=None, 
             steps=scipy.inf, max_run_hours=scipy.inf,
             step_scale=1.0, seeds=None, jac_func=None,
             skip_elems=0):
    """
    Generate an ensemble of parameter sets consistent with the jeffrey's prior weighting in
    the model manifold. The sampling is done in terms of the bare parameters.

    Inputs:
     (All not listed here are identical to those of jeffreys_prior_ensemble_log_params.)
     jac_func --- Function used to calculate the jacobian matrix. It should
                     take only a parameters argument and return the matrix.
                     If this is None, default is to use m.GetJandJtJ.
    """
    return jeffreys_prior_ensemble_log_params(m, params, jac, steps, max_run_hours, 
                                              step_scale, seeds, jac_func, 
                                              skip_elems, log_params=False)
    
def jeffreys_prior_ensemble_log_params(m, params, jac=None, 
                        steps=scipy.inf, max_run_hours=scipy.inf,
                        step_scale=1.0, seeds=None, jac_func=None,
                        skip_elems = 0, log_params=True):
    """
    Generate an ensemble of parameter sets consistent with the jeffrey's prior weighting in
    the model manifold. The sampling is done in terms of the logarithm of the parameters.

    Inputs:
     m -- Model to generate the ensemble for
     params -- Initial parameter KeyedList to start from 
     jac -- Jacobian of the model
     steps -- Maximum number of Monte Carlo steps to attempt
     max_run_hours -- Maximum number of hours to run
     step_scale -- Additional scale applied to each step taken. step_scale < 1
                   results in steps shorter than those dictated by the quadratic
                   approximation and may be useful if acceptance is low.
     seeds -- A tuple of two integers to seed the random number generator
     jac_func --- Function used to calculate the jacobian matrix. It should
                     take only a log parameters argument and return the matrix.
                     If this is None, default is to use 
                     m.GetJandJtJInLogParameteters 
     skip_elems --- If non-zero, skip_elems are skipped between each included 
                    step in the returned ensemble. For example, skip_elems=1
                    will return every other member. Using this option can
                    reduce memory consumption.

    Outputs:
     ens, ratio
     ens -- List of KeyedList parameter sets in the ensemble
     ratio -- Fraction of attempted moves that were accepted

    The sampling is done by Markov Chain Monte Carlo, with a Metropolis-Hasting
    update scheme. The canidate-generating density is a gaussian centered on the
    current point. For a useful introduction see:
     Chib and Greenberg. "Understanding the Metropolis-Hastings Algorithm" 
     _The_American_Statistician_ 49(4), 327-335
    """
    
    # CHECK RUNTIME AND SET SEEDS
    if scipy.isinf(steps) and scipy.isinf(max_run_hours):
        logger.warn('Both steps and max_run_hours are infinite! '
                    'Code will not stop by itself!')

    if seeds is None:
        seeds = int(time.time()%1 * 1e6)
        logger.debug('Seeding random number generator based on system time.')
        logger.debug('Seed used: %s' % str(seeds))
    scipy.random.seed(seeds)
    if isinstance(params, KeyedList):
        param_keys = params.keys()
    
    # INITIATE VARIABLES - PARAMS, JAC, ENS and DIMENSION
    curr_params = copy.deepcopy(params)
    ens = [curr_params]
    curr_params = scipy.array(curr_params)
    parameter_dimension = len(curr_params)

    if jac_func is None and log_params: # What if jac_func != None but log_params = False??
        jac_func = lambda p: m.GetJandJtJInLogParameters(scipy.log(p))[0]
    else:
        jac_func = lambda p: m.GetJandJtJ(p)[0]

    accepted_moves, attempt_exceptions, ratio = 0, 0, scipy.nan
    start_time = time.time()

    if jac is None:
        jac = jac_func(curr_params)
    
    curr_prob=_find_probability_jp(jac)
    steps_attempted = 0
    
    # PERFORM THE SAMPLING
    while steps_attempted < steps:
        #Have we run too long?
        if (time.time() - start_time) >= max_run_hours*3600:
            break
        # Generate the trial move from a guassian distribution
        scaled_step = _trial_move_jp(parameter_dimension, step_scale)
        if log_params:
            next_params = curr_params * scipy.exp(scaled_step)
        else:
            next_params = curr_params + scaled_step

        next_jac = jac_func(next_params)
        # Calculate the new probability and decide whether to accept/reject
        next_prob=_find_probability_jp(next_jac)
        accepted = _accept_move_jp(curr_prob, next_prob)
        steps_attempted += 1
        
        if accepted:
            accepted_moves += 1.
            curr_params = next_params
            curr_sf = m.internalVars['scaleFactors'].copy()
            jac = next_jac
        # Append new point to the ens 
        if steps_attempted % (skip_elems + 1) == 0:
            if isinstance(params, KeyedList):
                ens.append(KeyedList(zip(param_keys, curr_params)))
            else:
                ens.append(curr_params)
        ratio = accepted_moves/steps_attempted
        
    return ens, ratio

def _trial_move_jp(parameter_dimension, step_scale=1):
    trialMove = step_scale*scipy.randn(parameter_dimension)
    return trialMove

def _find_probability_jp(jac):
    """find the probabilities for parameters with jacobian - jac"""
    probability = np.prod(svdvals(jac))
    return probability

def _accept_move_jp(curr_prob, next_prob):
    """ decide whether to accept the proposed step """
    qual1 = next_prob> curr_prob
    qual2=(next_prob/curr_prob)> np.random.random()
    accepted=(qual1|qual2)
    return accepted

def find_step_size_jp(m, params, jac=None, test_step_sizes=None,
                        steps=scipy.inf, step_scale=1.0, 
                        seeds=None, jac_func=None,
                        skip_elems = 0, log_params=True):
    """find the optimum step size"""
    if test_step_sizes==None:
        test_step_sizes=(np.arange(5)+1.)
    val=False
    while val==False:
        slopes=get_slopes(m, params, jac, test_step_sizes,
                        steps=scipy.inf, step_scale=1.0, 
                        seeds=None, jac_func=None,
                        skip_elems = 0, log_params=True)
        if slopes.argmax()==0:
            test_step_sizes=test_step_sizes/2
        elif slopes.argmax()==4:
            test_step_sizes=test_step_sizes*2
        else:
            best_step_size = test_step_sizes[slopes.argmax()]
            val=True
            D=slopes.max()
    print "The optimum step size is: "+str(best_step_size)
    print "The diffusion constant is: "+str(D)
    return best_step_size    

def get_slopes(m, params, jac=None, test_step_sizes=None,
                        steps=scipy.inf, step_scale=1.0, 
                        seeds=None, jac_func=None,
                        skip_elems = 0, log_params=True):
    """find the slopes for different test step sizes"""
    slope_list=[]
    for a_step_size in test_step_sizes:
        ens, r = jeffreys_prior_ensemble_log_params(m, asarray(params),steps=a_step_size)
        
        #### THE NEXT LINE WILL NOT WORK AS IS ####
        #### HOW DO YOU MAP FROM PARAMETERS TO DATA IN SLOPPYCELL? ####
        manifold=SdA.sampling_map_down(manifold_h, layer_num) #Needs to be step x data_dimension array
        
        distance_pts=[rsqrd_vs_t(manifold,i+1) for i in xrange(len(manifold)-1)]
        slope=get_fit_linear(distance_pts)
        slope_list.append(slope)
    slope_list=np.asarray(slope_list)
    return slope_list

def rsqrd_vs_t(walk,t):
    """makes an array of r**2 distance vs time for recorded walk"""
    totalsum=0
    RL=walk[t:]-walk[:-t]
    totalsum+=sum((RL*RL).sum(axis=1))/(len(walk)-t)
    return totalsum/len(walks)

def get_fit_linear(distance_pts):
    """Find the slope of the r**2 vs t of a walk"""
    t=np.arange(len(distance_pts))
    fitfunc = lambda params, t: params[0] * t 
    errfunc = lambda p, t, distance_pts: fitfunc(p, t) - distance_pts 

    p1, success = leastsq(errfunc, 0.5, args = (t, distance_pts))
    return p1 

