"""
"""

import itertools 

import pandas as pd
import numpy as np
import sympy

# FIXME ****
#from util import butil
#reload(butil)
from util import butil 
#reload(butil)



class Matrix(butil.DF):
    """
    """
    
    @property
    def _constructor(self):
        return Matrix
    
    
    def __mul__(self, other):
        """
        A customization of np.matrix.__mul__ so that if two Matrix instances
        are passed in, their attributes of rowvarids and colvarids are kept.
        """
        if hasattr(other, 'colvarids'):
            if self.colvarids != other.rowvarids:
                raise ValueError("...")
            return Matrix(np.matrix(self)*np.matrix(other), 
                          rowvarids=self.rowvarids, colvarids=other.colvarids)
        else:
            return np.matrix.__mul__(np.matrix(self), np.matrix(other)) 

            
    def get_rank(self, tol=None):
        return np.linalg.matrix_rank(self, tol=tol)
    
    @property
    def rank(self):
        return self.get_rank()
            
   
    @property
    def T(self):
        return Matrix(self.T, index=self.colvarids, columns=self.rowvarids) 
        
    
    @property
    def I(self):
        return Matrix(np.linalg.inv(self), index=self.colvarids, columns=self.rowvarids)
    
    
    def rref(self):
        """
        Get the reduced row echelon form (rref) of the matrix using sympy.
        """
        # symmat's dtype is sympy.core.numbers.Integer/Zero/One, and 
        # when converted to np.matrix the dtype becomes 'object' which
        # slows down the matrix computation a lot
        symmat = sympy.Matrix.rref(sympy.Matrix(self))[0]
        return Matrix(np.asarray(symmat.tolist(), dtype='float'), 
                      self.rowvarids, self.colvarids)
        
    
    # define matrix concatenation and slicing operations,
    # especially rowvarids and colvarids attributes
    
    def is_zero(self):
        if (np.array(self) == np.zeros((self.nrow, self.ncol))).all():
            return True
        else:
            return False
    

    def diff(self, other):
        """
        """
        if set(other.rowvarids) != set(self.rowvarids) or\
            set(other.colvarids) != set(self.colvarids):
            raise ValueError("rowvarids or colvarids not the same.")
        return self - other
    
    
    def rdiff(self, other):
        """
        """
        if set(other.rowvarids) != set(self.rowvarids) or\
            set(other.colvarids) != set(self.colvarids):
            raise ValueError("rowvarids or colvarids not the same.")
        return (self - other) / self
        
        
    def normalize(self, y, x):
        """
        M = dy / dx
        M_normed = diag(1/y) * M * diag(x)
        """
        return Matrix.diag(1/y) * self * Matrix.diag(x)
        # mat = np.matrix(np.diag(1/y)) * np.matrix(self) * np.matrix(np.diag(x))
        # return Matrix(mat, self.rowvarids, self.colvarids)
    
    
    
    @staticmethod
    def diag(x):
        """
        Return the diagonal Matrix of vector x. 
        
        D(x) = diag(x)
        """
        return Matrix(np.diag(x), x.index, x.index)
        
    
    @staticmethod
    def eye(rowvarids, colvarids=None):
        """
        """
        if colvarids is None:
            colvarids = rowvarids
        return Matrix(np.eye(len(rowvarids)), rowvarids, colvarids)
        
