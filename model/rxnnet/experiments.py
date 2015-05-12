"""
"""

import collections
OD = collections.OrderedDict
import itertools

import numpy as np
import pandas as pd

import butil


class Experiments(object):
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
        return getattr(self._, attr)
    
    
    def __repr__(self):
        return self._.__repr__()

    
    @property
    def size(self):
        return self.shape[0]
    

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
    
    
    def sep_conditions(self):
        """
        """
        cond2expts = OD()
        for cond, expts_cond in self.groupby('condition', sort=False):
            cond2expts[cond] = Experiments(expts_cond)
        return cond2expts


    def to_dids(self):
        """
        Get the ids of data variables.
        """
        dids = []
        for expts_cond in self.sep_conditions().values():
            for idx, row in expts_cond.iterrows():
                condition, varids, times = row
                if not isinstance(varids, list):
                    varids = [varids]
                if not isinstance(times, list):
                    times = [times]
                for did in itertools.product([condition], varids, times):
                    dids.append(tuple(did))
        return dids
    

    def get_measurements(self):
        """
        """
        if len(set(self.condition)) > 1:
            raise ValueError("There are more than one conditions.")
        else:
            return [did[1:] for did in self.get_dids()]
    
    
    