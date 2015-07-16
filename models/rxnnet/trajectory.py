"""
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def sort_times(times):
    """
    """
    times_sorted = []
    for t in times:
        if isinstance(t, list) or isinstance(t, tuple):
            times_sorted.extend(t)
        else:
            times_sorted.append(t)
    times_sorted = sorted(set(times_sorted))
    return times_sorted


class Trajectory(object):
    """
    dat: one of the following five options 
        - a 2d array (row indexed by time, and column indexed by varid); 
            varids has to be provided in this case
        - a pd.DataFrame
        - a pd.Series
        - a SloppyCell traj
        - a Trajectory instance
    """
    def __init__(self, dat=None, times=None, varids=None, **kwargs):
        # a pd.DataFrame
        if isinstance(dat, pd.DataFrame):
            times, varids = dat.index, dat.columns
        # a pd.Series
        if isinstance(dat, pd.Series):
            if times is not None:
                varids = dat.index     
            elif varids is not None:
                times = dat.index
            else:
                raise ValueError("One of times and varids has to be given.")
            dat = dat.tolist()
        # a SloppyCell traj
        if hasattr(dat, 'timepoints'):  
            times, varids = dat.timepoints, dat.key_column.keys()
            dat = dat.values
        # a Trajectory instance
        if hasattr(dat, 'times'):
            times, varids = dat.times, dat.varids
            dat = dat.values
        
        if dat is None:
            dat = []
        if times is None:
            times = []
        
        # enforcing providing a list
        #if not isinstance(times, list):
        #    times = [times]
        #if not isinstance(varids, list):
        #    varids = [varids]
        traj = pd.DataFrame(np.reshape(dat, (len(times),len(varids))), 
                            index=times, columns=varids)
        traj.index.name = 'time'
        
        self._ = traj
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
        return self._[key]
    
    
    def __add__(self, other):
        """
        """
        if self.varids == other.varids:
            times = self.times + other.times  # list addition
            varids = self.varids
            values = np.vstack((self.values, other.values))
        elif self.times == other.times:
            times = self.times
            varids = self.varids + other.varids
            values = np.hstack((self.values, other.values))      
        else:
            raise ValueError("...")
        return Trajectory(dat=values, times=times, varids=varids)


    @property
    def times(self):
        return self.index.tolist()

    
    @property
    def varids(self):
        return self.columns.tolist()
    
    
    @property
    def ntime(self):
        return len(self.times)


    @property
    def nvar(self):
        return len(self.varids)    

    
    def flatten(self):
        """
        """
        if self.ntime == 1 and self.nvar == 1:
            return self.iloc[0,0]
        if self.ntime == 1 and self.nvar > 1:
            return self.iloc[0,:]

    
    def get_subset(self, times=None, varids=None):
        """
        Returns a Trajectory instance (if using [], which invokes __getitem__,
        returns a pandas object (DataFrame, Series or scalar).)
        
        Interpolation?
        """
        if times is None:
            dat = self.ix[:,varids]
            times = self.times
        elif varids is None:
            dat = self.ix[times,:]
            varids = self.varids
        else:
            dat = self.ix[times,varids]
        subtraj = Trajectory(dat=dat, times=times, varids=varids) 
        return subtraj        
        
    
    def get_diff_traj(self, other):
        pass
    
    
    def plot(self, plotvarids=None, filepath=''):
        """
        """
        nvar = len(self.varids)
        nsubplot = int(np.ceil(nvar/7.)) 
        
        fig = plt.figure(figsize=(8, 3*nsubplot))  # 3 is the height of each subplot; make it an argument?
        for i in range(nsubplot):
            ax = fig.add_subplot(nsubplot, 1, i)
            ax.plot(self.times, self.ix[:,i*7:(i+1)*7])
            ax.legend(self.varids[i*7:(i+1)*7], fontsize=10)
        
        plt.savefig(filepath)
        plt.close()
        
        