"""
"""

import pandas as pd
import numpy as np


class Parameter(object):
    """
    """
    
    def __init__(self, vals=None, pids=None, p=None):
        if p is not None:
            vals, pids = p.values, p.pids
            
        if pids is None:
            if isinstance(vals, pd.Series):
                pids = vals.index.tolist()
            else:
                pids = ['theta%d'%i for i in range(1, len(vals)+1)]
        p = pd.Series(vals, index=pids)
        self._ = p
        self.pids = pids
        self.logpids = ['log_'+pid for pid in pids]
        
        
    def __getattr__(self, attr):
        return getattr(self._, attr)
    
    
    def __repr__(self):
        return self._.__repr__()
    
    
    def __getitem__(self, key):
        return self._.__getitem__(key)
    
    
    def log(self):
        return Parameter(np.log(np.array(self, dtype='float')),
                         pids=['log_'+pid for pid in self.pids]) 
    
    
    def exp(self):
        if not all([pid.startswith('log_') for pid in self.pids]):
            raise ValueError("The parameters are not in log-scale.")
        return Parameter(np.exp(np.array(self, dtype='float')),
                         pids=[pid.lstrip('log_') for pid in self.pids])
    
    
    def reorder(self, pids):
        """
        """
        if set(self.pids) != set(pids):
            raise ValueError("The parameter ids are different:\n%s\n%s"%\
                             (str(self.pids), str(pids)))
        return Parameter(self, pids)
    
    
    def mutate(self, seed=None, distribution='lognormal', **kwargs):
        if seed:
            np.random.seed(seed)
        
        