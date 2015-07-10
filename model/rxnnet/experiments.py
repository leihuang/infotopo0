"""
FIXME ***: 
Experiments -> butil.DF
methods: 
    regularize
    __iter__ 
    get_conds
    
properties:
    conds: 
    varids:
    times:  
    dids: 

Doc: (varids, times): measurements
"""

import collections
OD = collections.OrderedDict
import itertools

import numpy as np
import pandas as pd

from util import butil


class Experiments0(object):
    """
    Make it a subclass of DataFrame? I really don't know what to do in this case: 
    self.expts = expts? (and 'self' would usually be called expts as well)
    
    what other metaattributes would I need?
    
    condition, variable, time
    
    condition: 
        None: wildtype
        ('R', k): capacity of reaction R changed by k fold
        ('X', k): fixed concentration of X changed by k fold
        ('R', 'I', 'c', KI, I): adding concentration I of competitive
                                inhibitor I of reaction R with 
                                inhibition constant KI
    
        OD([('X1', (1,2)), ('X2', [1,2]), (('X1','X3'), [3,4])]) 
        
    """
    def __init__(self, expts=None):
        if expts is None:
            expts = pd.DataFrame(columns=['condition','varids','times'])
        elif hasattr(expts, '_'):  # expts is an Experiments object
            expts = expts._
        expts.index.name = 'experiment'
        self._ = expts
    
    
    def __getattr__(self, attr):
        out = getattr(self._, attr)
        if callable(out):
            out = _wrap(out)
        return out
        
    
    def __getitem__(self, key):
        out = self._.ix[key]
        try:
            return Experiments0(out)
        except ValueError: 
            return out
    
    
    def __repr__(self):
        return self._.__repr__()

    
    @property
    def size(self):
        return self.shape[0]
    
    
    def to_dids(self):
        """
        Get the ids of data variables.
        """
        dids = []
        for expts_cond in self.separate_conditions().values():
            for idx, row in expts_cond.iterrows():
                condition, varids, times = row
                if not isinstance(varids, list):
                    varids = [varids]
                if not isinstance(times, list):
                    times = [times]
                for did in itertools.product([condition], varids, times):
                    dids.append(tuple(did))
        return dids    

    def add(self, expt):
        """
        Input:
            expt: a 3-tuple or a mapping
                (None, 'A', 1)
                (('k1',0.5), ['X','Y'], [1,2,3])
        
        """
        if isinstance(expt, collections.Mapping):
            expt = butil.get_values(expt, self.columns)
            
        condition, varids, times = expt
        # refine times
        times = np.float_(times)
        if isinstance(times, np.ndarray):
            times = list(times)
        
        if len(condition)>=2 and isinstance(condition[-1], list):
            conditions = [condition[:-1]+(r,) for r in condition[-1]]
            for condition in conditions:
                self.loc[self.size+1] = (condition, varids, times)
        else:
            self.loc[self.size+1] = (condition, varids, times)
            
    
    def delete(self, condition=None):
        """
        """
        if condition is not None:
            return Experiments0(self[self.condition != condition])
    
    
    def add_perturbation_series(self, pid, series, varids, times, mode=None):
        """
        Input:
            varids: ids of measured variables
            times: measured times
            mode: perturbation mode: '+', '=', etc.
        """
        for pert in series:
            if mode is None:
                cond = (pid, pert)
            else:
                cond = (pid, mode, pert)
            expt = (cond,) + tuple([varids, times])
            self.add(expt)
    
    
    def separate_conditions(self):
        """
        """
        cond2expts = OD()
        for cond, expts_cond in self.groupby('condition', sort=False):
            cond2expts[cond] = Experiments0(expts_cond)
        return cond2expts
    

    def get_measurements(self):
        """
        """
        if len(set(self.condition)) > 1:
            raise ValueError("There are more than one conditions.")
        else:
            return [did[1:] for did in self.get_dids()]
    
    
def _wrap(f):
    def f_wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        try:
            return Experiments0(out)
        except:
            return out
    return f_wrapped





class Experiments(butil.DF):
    """
    """
    
    @property
    def _constructor(self):
        return Experiments
    
    
    def __init__(self, *args, **kwargs):
        """
        FIXME **: fix this constructor...
        """
        if len(args) >= 3:
            args[2] = ['condition','varids','times']
        kwargs.update({'columns':['condition','varids','times']})
        super(Experiments, self).__init__(*args, **kwargs)
        self.index.name = 'experiment'
        
        
    @property
    def size(self):
        return self.shape[0]
    
    
    def to_dids(self):
        """
        Get the ids of data variables.
        """
        dids = []
        for expts_cond in self.separate_conditions().values():
            for idx, row in expts_cond.iterrows():
                condition, varids, times = row
                if not isinstance(varids, list):
                    varids = [varids]
                if not isinstance(times, list):
                    times = [times]
                for did in itertools.product([condition], varids, times):
                    dids.append(tuple(did))
        return dids
    

    def add(self, expt):
        """
        Input:
            expt: a 3-tuple or a mapping
                (None, 'A', 1)
                (('k1',0.5), ['X','Y'], [1,2,3])
        
        """
        if isinstance(expt, collections.Mapping):
            expt = butil.get_values(expt, self.columns)
            
        condition, varids, times = expt
        # refine times
        times = np.float_(times)
        if isinstance(times, np.ndarray):
            times = list(times)
        
        if len(condition)>=2 and isinstance(condition[-1], list):
            conditions = [condition[:-1]+(r,) for r in condition[-1]]
            for condition in conditions:
                self.loc[self.size+1] = (condition, varids, times)
        else:
            self.loc[self.size+1] = (condition, varids, times)
    
    to_yids = to_dids  # FIXME **
    
    @property
    def yids(self):
        return self.to_yids()
      
    
    def delete(self, condition=None):
        """
        """
        if condition is not None:
            return self[self.condition != condition]
    
    
    def add_perturbation_series(self, pid, series, varids, times, mode=None):
        """
        Input:
            varids: ids of measured variables
            times: measured times
            mode: perturbation mode: '+', '=', etc.
        """
        for pert in series:
            if mode is None:
                cond = (pid, pert)
            else:
                cond = (pid, mode, pert)
            expt = (cond,) + tuple([varids, times])
            self.add(expt)
    
    
    def separate_conditions(self):
        """
        """
        cond2expts = OD()
        for cond, expts_cond in self.groupby('condition', sort=False):
            cond2expts[cond] = expts_cond
        return cond2expts
    

    def get_measurements(self):
        """
        """
        if len(set(self.condition)) > 1:
            raise ValueError("There are more than one conditions.")
        else:
            return [did[1:] for did in self.get_dids()]
    
    
    
    