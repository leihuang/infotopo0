"""
"""

import itertools
import cPickle 
import os
import copy
import sympy
from collections import OrderedDict as OD

import numpy as np
import pandas as pd



class Series(pd.Series):
    """
    Reference:
    http://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures
    """
    @property
    def _constructor(self):
        return Series


    @property
    def _constructor_expanddim(self):
        return DF
        
    
    def __init__(self, data, index=None, **kwargs):
        """
        Overwrite the constructor...
        
        >>> ser = pd.Series([1,2], index=['A','B'])
        >>> ser2 = pd.Series(ser, index=['a','b'])
        >>> ser2
            a   NaN
            b   NaN
            dtype: float64
        >>> ser3 = Series(ser, index=['a','b'])
        >>> ser3
            a   1
            b   2
        """
        if index is None and hasattr(data, 'index') and\
                not callable(data.index):
            index = data.index
        if hasattr(data, 'values') and not callable(data.values):
            data = data.values
        # FIXME ***
        super(Series, self).__init__(data, index=index, **kwargs)
        
    
    def copy(self, **kwargs):
        """
        Copy custom attributes as well.
        """
        ser_cp = super(Series, self).copy(**kwargs)
        ser_cp.__dict__ = self.__dict__
        return ser_cp
    
    
    @property
    def varids(self):
        return self.index.tolist()
    
    @property
    def logvarids(self):
        return map(lambda varid: 'log_'+varid, self.varids)
    
        
    def log(self):
        return Series(np.log(self), self.logvarids)
    
    
    def exp(self):
        if not all([varid.startswith('log_') for varid in self.varids]):
            raise ValueError("The values are not in log-scale.")
        return Series(np.exp(self), 
                      map(lambda varid: varid.lstrip('log_'), self.varids))


    def perturb(self, seed=None, distribution='lognormal', **kwargs):
        """
        Input:
            kwargs: sigma
        """
        if seed:
            np.random.seed(seed)
        if distribution == 'lognormal':
            return self * np.random.lognormal(size=self.size, **kwargs)
        if distribution == 'normal':
            return self * np.random.normal(size=self.size, **kwargs)
    
    # self[varids] 
    #def reorder(self, varids):
    #    """
    #    """
    #    if set(self.varids) != set(varids):
    #        raise ValueError("The parameter ids are different:\n%s\n%s"%\
    #                         (str(self.varids), str(varids)))
    #    return Series(self, varids)
    
        
    def to_od(self):
        return OD(self)
    

    # FIXME *: why returning pd.Series instead?
    #def append(self, other, **kwargs):
    #    return Series(super(Series, self).append(other, **kwargs))

    
class DF(pd.DataFrame):
    """
    Reference:
    
    http://stackoverflow.com/questions/13460889/how-to-redirect-all-methods-of-a-contained-class-in-python
    
    http://stackoverflow.com/questions/22155951/how-to-subclass-pandas-dataframe
    http://stackoverflow.com/questions/29569005/error-in-copying-a-composite-object-consisting-mostly-of-pandas-dataframe
    
    http://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures
    """
    
    @property
    def _constructor(self, **kwargs):
        return DF
    
    
    @property
    def _constructor_sliced(self):
        return Series
    
    """
    def __init__(self, data, index=None, columns=None, **kwargs):
        if index is None and hasattr(data, 'index') and\
                not callable(data.index):
            index = data.index
        if columns is None and hasattr(data, 'columns'):
            columns = data.columns
        if hasattr(data, 'values') and not callable(data.values):
            data = data.values
        # FIXME ***
        #print id(self)
        #print id(self.__class__)
        #print id(DF)
        super(DF, self).__init__(data, index=index, columns=columns, **kwargs)
    """
    
    def get_rowvarids(self):
        return self.index.tolist()
    
    def set_rowvarids(self, rowvarids):
        self.index = pd.Index(rowvarids)
    
    rowvarids = property(get_rowvarids, set_rowvarids)
    
    def get_colvarids(self):
        return self.columns.tolist()
    
    def set_colvarids(self, colvarids):
        self.index = pd.Index(colvarids)
    
    colvarids = property(get_colvarids, set_colvarids)
    
    @property
    def nrow(self):
        return self.shape[0]
        
    @property
    def ncol(self):
        return self.shape[1]
    
    
    def copy(self):
        """
        Copy custom attributes as well.
        """
        df_cp = self.copy()
        df_cp.__dict__ = self.__dict__
        return df_cp
    
    
    #def append(self, ser, varid, axis=0):
    #    """
    #    """
    #    pass
    
    
    #def extend(self, other, axis=0):
    #    """
    #    """
    #    pass
    
    
    def dump(self, filepath):
        """
        Dump custom attributes as well.
        """
        fh = open(filepath, 'w')
        cPickle.dump((self, self.__dict__), fh)
        fh.close()
    
    
    def load(self, filepath):
        # dat, attrs = cPickle.load(fh)
        # mydf = self.__class__(dat)  -- does it work??
        # mydf.__dict__ = attrs 
        pass
    
    
    # self.to_csv
    #def save(self):
    #    """
    #    Just data (no metadata)
    #    Human readable
    #    """
    #    pass
    
    
    # self.iloc[::-1]
    # self.iloc[:, ::-1]
    #def flip(self, axis=0):
    #    """
    #    Input:
    #        axis: 0 (up-down) or 1 (left-right)
    #    """
    #    if axis == 0:
    #        return DF(np.flipud(self), self.rowvarids[::-1], self.colvarids)
    #    if axis == 1:
    #        return DF(np.fliplr(self), self.rowvarids, self.colvarids[::-1])
    
    
    def to_series(self):
        """
        """
        dat = self.values.flatten()
        index = list(itertools.product(self.rowvarids, self.colvarids))
        return Series(dat, index=index)
    
    
    #def pca(self):
    #    pass
    
    
    def plot(self, fmt='heatmap', orientation=''):
        """
        heatmap or table
        """
        pass
    
    
    

class Matrix(DF):
    """
    A possibly useful conceptual distinction: 
        - DF is mostly a book-keeping table
        - Matrix represents linear transformation
    
    """
    @property
    def _constructor(self):
        return Matrix
    
    def get_rank(self, tol):
        pass

    @property
    def rank(self):
        return self.get_rank()
    
    def rref(self):
        """
        row-reduced echolon form
        """
        pass
    
    def svd(self):
        pass
    
    @property
    def T(self):
        pass
    
    @property
    def I(self):
        pass
    
    def normalize(self, y, x):
        """
        M = dy / dx
        M_normed = diag(1/y) * M * diag(x)
        
        Used much in rxnnet mca calculations.
        """
        return Matrix.diag(1/y) * self * Matrix.diag(x)
    
    @staticmethod
    def eye():
        pass
    
    @staticmethod
    def diag():
        pass
    

def check_filepath(filepath):
    """
    The function performs two tasks:
        1. Check if the directory exists, and create it if not
        2. Check if the filepath exists, and ask for permission if yes
    """
    # task 1
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # task 2
    if os.path.isfile(filepath):
        return raw_input("%s exists: Proceed? "%filepath) in\
            ['y', 'Y', 'yes', 'Yes', 'YES']
    else:
        return True