"""
FIXME: ****
Make it more general: Parameter->Variables
Subclass pd.Series
mutate->perturb
Apply it to all model.Network shorthands, eg, dynvars
???dynvars, dynvarids, dynvarvals??? 
"""

from collections import OrderedDict as OD

import pandas as pd
import numpy as np


class Parameter(object):
    """
    """
    
    def __init__(self, vals=None, pids=None):
        """
        vals: four possibilities: sequence, mapping, series, a Parameter instance
        """
        if hasattr(vals, 'items'):
            pids = vals.keys()
            vals = vals.values()
            
        if isinstance(vals, pd.Series):
            pids = vals.index.tolist()
            vals = vals.values
            
        if hasattr(vals, '_'):
            pids = vals._.index.tolist()
            vals = vals._.values
            
        if pids is None:
            raise ValueError("pids are not given.")

        
        self._ = pd.Series(vals, index=pids) 

    @property
    def pids(self):
        return self._.index.tolist()
    
    @property
    def logpids(self):
        return ['log_'+pid for pid in self.pids]
        
        
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
        
    def to_od(self):
        return OD(self._)