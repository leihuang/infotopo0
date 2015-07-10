"""
"""

import itertools 

import sympy

import pandas as pd
import numpy as np


from util.butil import Series, DF



class Matrix(DF):
    """
    A possibly useful conceptual distinction: 
        - DF is mostly a book-keeping table
        - Matrix represents linear transformation between v. spaces over a field
    
    """
    @property
    def _constructor(self):
        return Matrix
    
    
    def __init__(self, data, rowvarids=None, colvarids=None, **kwargs):
        if rowvarids is not None:
            kwargs.update({'index': rowvarids})
        if colvarids is not None:
            kwargs.update({'columns': colvarids})
        super(Matrix, self).__init__(data, **kwargs)
        
    
    def __mul__(self, other):
        """
        A customization of np.matrix.__mul__ so that if two Matrix instances
        are passed in, their attributes of rowvarids and colvarids are kept.
        """
        if hasattr(other, 'colvarids'):
            assert self.colvarids == other.rowvarids, "varids do not match"
            colvarids = other.colvarids
        elif isinstance(other, float) or isinstance(other, int):
            colvarids = self.colvarids
        else:
            colvarids = None
        return Matrix(np.dot(self, other), 
                      rowvarids=self.rowvarids, colvarids=colvarids)
        
                
    def __pow__(self, exponent):
        return Matrix(np.matrix(self)**exponent, 
                      rowvarids=self.rowvarids, colvarids=self.colvarids)
        
    
    
    def ch_rowvarids(self, rowvarids_new):
        self.rowvarids = rowvarids_new
        return self
    
    def ch_colvarids(self, colvarids_new):
        self.colvarids = colvarids_new
        return self
        
    
    def get_rank(self, tol=None):
        return np.linalg.matrix_rank(self, tol=tol)

    @property
    def rank(self):
        return self.get_rank()
    
    def rref(self):
        """
        row-reduced echolon form
        
        Get the reduced row echelon form (rref) of the matrix using sympy.
        """
        # symmat's dtype is sympy.core.numbers.Integer/Zero/One, and 
        # when converted to np.matrix the dtype becomes 'object' which
        # slows down the matrix computation a lot
        symmat = sympy.Matrix.rref(sympy.Matrix(self))[0]
        return Matrix(np.asarray(symmat.tolist(), dtype='float'), 
                      self.rowvarids, self.colvarids)
    
    
    def svd(self, to_mat=True):
        """
        Input:
            to_mat: 
        """
        U, S, Vt = np.linalg.svd(self)
        if to_mat:
            # http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.linalg.matrix_rank.html
            rank = np.sum(S > S.max() * max(self.shape) * 1e-16)
            U = Matrix(U[:,:rank], rowvarids=self.rowvarids, colvarids=self.colvarids)
            S = Matrix(np.diag(S), rowvarids=self.colvarids, colvarids=self.colvarids)
            Vt = Matrix(Vt, rowvarids=self.colvarids, colvarids=self.colvarids)
            #assert np.allclose(U*S*Vt, self)   
        return U, S, Vt
    
    
    @property
    def I(self):
        return Matrix(np.linalg.inv(self), rowvarids=self.colvarids, 
                      colvarids=self.rowvarids)
    
    
    def normalize(self, y=None, x=None):
        """
        M = dy / dx
        M_normed = d logy/d logx = diag(1/y) * M * diag(x)
        
        Used much in rxnnet mca calculations.
        """
        mat = self
        if y is not None:
            mat = Matrix.diag(1/y) * self
        if x is not None:
            mat = mat * Matrix.diag(x)
        return mat 
    
    
    @staticmethod
    def eye(rowvarids, colvarids=None):
        """
        """
        if colvarids is None:
            colvarids = rowvarids
        return Matrix(np.eye(len(rowvarids)), rowvarids=rowvarids, 
                      colvarids=colvarids)
    
    
    @staticmethod
    def diag(ser):
        """
        Return the diagonal Matrix of vector ser. 
        
        D(x) = diag(x)
        """
        ser = Series(ser)
        return Matrix(np.diag(ser), rowvarids=ser.index, colvarids=ser.index)
    
    """
    def flipud(self):
        return Matrix(np.flipud(self), self.rowvarids[::-1], self.colvarids)
    
    def fliplr(self):
        return Matrix(np.fliplr(self), self.rowvarids, self.colvarids[::-1])
    """
    

class Matrix0(object):
    """
    """
    
    def __init__(self, mat, rowvarids=None, colvarids=None, **kwargs):
        """
        Input:
            mat: three possibilities: 2d array of numbers, a pd.DataFrame or 
                a Matrix object
        """
        if hasattr(mat, '_'):
            #attrs = mat.__dict__
            #del attrs['_']
            mat = mat._ 
            dat = mat.values
        elif isinstance(mat, pd.DataFrame):
            dat = mat.values
            #attrs = {}
        else:
            dat = mat
            
        try:
            if rowvarids is None:
                rowvarids = mat.index.tolist()
            if colvarids is None:
                colvarids = mat.columns.tolist()
        except AttributeError:
            raise ValueError("rowvarids and colvarids are not provided and cannot be inferred")
            
        df = pd.DataFrame(dat, index=rowvarids, columns=colvarids)
        self._ = df
        
        #for attrid, attrval in attrs.items():
        #    setattr(self, attrid, attrval)
        
        
    def __getattr__(self, attr):
        out = getattr(self._, attr)
        if callable(out):
            out = _wrap(out)
        return out
        
    
    def __getitem__(self, key):
        out = self._.ix[key]
        try:
            return Matrix(out)
        except ValueError: 
            return out
    
    
    def __setitem__(self, key, value):
        self._.ix[key] = value
        

    def __repr__(self):
        return self._.__repr__()
    
    
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


    def __neg__(self):
        return Matrix(-self._)
        
    
    def __add__(self, other):
        if self.rowvarids != other.rowvarids or self.colvarids != other.colvarids:
            raise ValueError("...") 
        return Matrix(self._ + other._)
    
    
    def __sub__(self, other):
        if self.rowvarids != other.rowvarids or self.colvarids != other.colvarids:
            raise ValueError("...") 
        return Matrix(self._ - other._)
    
        
    @property
    def rowvarids(self):
        return self.index.tolist()

    @rowvarids.setter
    def rowvarids(self, rowvarids):
        self._.index = rowvarids
        
    
    @property
    def colvarids(self):
        return self.columns.tolist()
    
    @colvarids.setter
    def colvarids(self, colvarids):
        self._.columns = colvarids
        
        
    def ch_rowvarids(self, rowvarids):
        return Matrix(self, rowvarids, self.colvarids)
        
    
    def ch_colvarids(self, colvarids):
        return Matrix(self, self.rowvarids, colvarids)
        
    
    @property
    def nrow(self):
        return self.shape[0]
        
        
    @property
    def ncol(self):
        return self.shape[1]


    def slice(self, rowvarids=None, colvarids=None):
        submat = self._
        if rowvarids is not None:
            submat = submat.loc[rowvarids]
        if colvarids is not None:
            submat = submat.loc[:, colvarids]
        return Matrix(submat)
        

    def to_series(self):
        """
        What is the point? FIXME
        """
        dat = self.values.flatten()
        index = list(itertools.product(self.index, self.columns))
        ser = pd.Series(dat, index=index)
        return ser
    

    def vstack(self, other):
        """
        Stack vertically.
        """
        mat = Matrix(np.vstack((self.values, other.values)), 
                     rowvarids=self.rowvarids+other.rowvarids,
                     colvarids=self.colvarids)
        return mat
    
    
    def flipud(self):
        """
        """
        return Matrix(np.flipud(self), self.rowvarids[::-1], self.colvarids)
    
    def fliplr(self):
        return Matrix(np.fliplr(self), self.rowvarids, self.colvarids[::-1])
        
        
    def get_rank(self, tol=None):
        return np.linalg.matrix_rank(self, tol=tol)
    
    @property
    def rank(self):
        return self.get_rank()
            
   
    @property
    def T(self):
        return Matrix(self.T, rowvarids=self.colvarids, colvarids=self.rowvarids) 
        
    
    @property
    def I(self):
        return Matrix(np.linalg.inv(self), rowvarids=self.colvarids, colvarids=self.rowvarids)
    
    
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
        return Matrix(self._ - other._)
    
    
    def rdiff(self, other):
        """
        """
        if set(other.rowvarids) != set(self.rowvarids) or\
            set(other.colvarids) != set(self.colvarids):
            raise ValueError("rowvarids or colvarids not the same.")
        return Matrix((self._ - other._) / self._)
        
        
    def normalize(self, y, x):
        """
        M = dy / dx
        M_normed = diag(1/y) * M * diag(x)
        """
        return Matrix.diag(1/y) * self * Matrix.diag(x)
        # mat = np.matrix(np.diag(1/y)) * np.matrix(self) * np.matrix(np.diag(x))
        # return Matrix(mat, self.rowvarids, self.colvarids)
    
    
    def plot(self, filepath=''):
        pass
    
    
    @staticmethod
    def diag(x):
        """
        Return the diagonal Matrix of vector x. 
        
        D(x) = diag(x)
        """
        return Matrix(np.diag(x), rowvarids=x.index.tolist(), 
                      colvarids=x.index.tolist())
        
    
    @staticmethod
    def eye(rowvarids, colvarids=None):
        """
        """
        if colvarids is None:
            colvarids = rowvarids
        return Matrix(np.eye(len(rowvarids)), rowvarids=rowvarids, 
                      colvarids=colvarids)
        

    
def _wrap(f):
    def f_wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        try:
            return Matrix(out)
        except:
            return out
    return f_wrapped