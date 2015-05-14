"""
This is a library of class and functions for computations related to
steady states and Metabolic Control Analysis (MCA).
MCA is essentially first-order sensitivity analysis of a metabolic model at
steady-state. For this reason, a network instance passed to any function in
this library is checked for steady state.
"""

from __future__ import division
import re
import subprocess
import fractions
from collections import OrderedDict as OD
import time
import itertools

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from SloppyCell.ReactionNetworks import *
from SloppyCell import ExprManip as expr

from util import butil
reload(butil)



class MCAMatrix(object):
    """
    stoichiometry matrix,
    elasticity matrix of concentrations,
    elasticity matrix of parameters,
    concentration control matrix 
    flux control matrix
    concentration response matrix
    flux response matrix

    reduced stoichiometry matrix
    link matrix L
    L0    
    """
    def __init__(self, mat, rowvarids=None, colvarids=None, **kwargs):
        """
        Input:
            mat: three possibilities: 2d array of numbers, a pd.DataFrame or 
                an MCAMatrix object
        """
        if hasattr(mat, '_'):
            if rowvarids is None:
                rowvarids = mat.index.tolist()
            if colvarids is None:
                colvarids = mat.columns.tolist()
            attrs = mat.__dict__
            del attrs['_']
            mat = mat._
        elif isinstance(mat, pd.DataFrame):
            if rowvarids is None:
                rowvarids = mat.index.tolist()
            if colvarids is None:
                colvarids = mat.columns.tolist()
            attrs = {}
        else:
            attrs = {}
        
        attrs.update(kwargs)
                    
        if rowvarids is None or colvarids is None:
            raise ValueError("rowvarids and colvarids are not provided and cannot be inferred")
            
        mat = pd.DataFrame(mat, index=rowvarids, columns=colvarids)
        self._ = mat
        for attrid, attrval in attrs.items():
            setattr(self, attrid, attrval)
        
        
    def __getattr__(self, attr):
        return getattr(self._, attr)
        
    
    def __getitem__(self, key):
        return self._.ix[key]
    
    
    def __setitem__(self, key, value):
        self._.ix[key] = value
        

    def __repr__(self):
        return self._.__repr__()

    
    @property
    def rowvarids(self):
        return self.index.tolist()
    
    
    @property
    def colvarids(self):
        return self.columns.tolist()
    
    
    @property
    def nrow(self):
        return self.shape[0]
        
        
    @property
    def ncol(self):
        return self.shape[1]


    def to_series(self):
        dat = self.values.flatten()
        index = list(itertools.product(self.index, self.columns))
        ser = pd.Series(dat, index=index)
        return ser
    

    def vstack(self, other):
        """
        Stack vertically.
        """
        mat = MCAMatrix(np.vstack((self, other)), 
                        rowvarids=self.rowvarids+other.rowvarids,
                        colvarids=self.colvarids)
        return mat
    
    
    def rank(self, tol=None):
        return np.linalg.matrix_rank(self, tol=tol)
    
    
    def __mul__(self, other):
        """
        A customization of np.matrix.__mul__ so that if two MCAMatrix instances
        are passed in, their meta-info of rowvarids and colvarids are kept.
        """
        rowvarids = self.rowvarids
        colvarids = other.colvarids
        return MCAMatrix(np.matrix.__mul__(self, other), 
                         rowvarids=rowvarids, colvarids=colvarids)
        
   
    @property
    def T(self):
        return MCAMatrix(self.T, rowvarids=self.colvarids, colvarids=self.rowvarids) 
        
    
    @property
    def I(self):
        return MCAMatrix(np.linalg.inv(self), rowvarids=self.colvarids, colvarids=self.rowvarids) 
            
    # define matrix concatenation and slicing operations,
    # especially rowvarids and colvarids attributes


def get_reordered_net(net):
    """
    This functions returns a new network instance with dynamic variables 
    reordered so that dependent dynamic variables come last. It should be
    called before calling any of the following functions in this module. 
    """
    
    """
    N = get_stoich_mat(net)
    if N.rank() == N.shape[0]: 
        # rank of N equals the number of rows
        # net has no conserved pools or dependent dynamic variables
        return net.copy()
    else:
        # rank of N smaller than the number of rows
        # net has conserved pools and dependent dynamic variables
        ddynvarids = get_dep_dyn_var_ids(net)  # d for dependent
        net2 = Network(net.id, name=net.name)
        ## Make new variables: move the dependent dynamic variables to 
        # the end of keyedlist net.variables, and call 
        # method _makeCrossReferences later so that
        # attributes of the network reflect the new order.
        variables2 = net.variables.copy()
        for ddynvarid in ddynvarids:
            idx = variables2.keys().index(ddynvarid)
            var = variables2.pop(idx)
            variables2.set(ddynvarid, var)
        ## attach attributes
        net2.variables = variables2
        net2.reactions = net.reactions.copy()
        net2.assignmentRules = net.assignmentRules.copy()
        net2.algebraicRules = net.algebraicRules.copy()
        net2.rateRules = net.rateRules.copy()
        net2.events = net.events.copy()
        ## final processings
        # method _makeCrossReferences will take care of at least
        # the following attributes:
        # assignedVars, constantVars, optimizableVars, dynamicVars, 
        # algebraicVars
        net2._makeCrossReferences()
        net2.reordered = True
        net2.ddynvarids = ddynvarids
        return net2
    """
    pass

def is_ss(net, tol=1e-6):
    """
    Output:
        True or False
    """
    vels = net.get_velocities()
    if vels.abs().max() < tol:
        return True
    else:
        return False


def set_ss(net, tol=1e-6, method='integration', T0=1e3, Tmax=1e6):
    """
    Input:
    
    Output: 
        No output

    Return the steady state values of dynamic variables found by 
    the (adaptive) integration method, which may or may not represent
    the true steady state.
    Dynvarvals of the network get updated.
    """
    if method == 'integration':
        t = T0
        while not net.is_ss() and t < Tmax:
            net.update(t=t)
            t *= 10
        if net.is_ss():
            pass
        else:
            raise Exception("not being able to reach steady state")
    if method == 'rootfinding':
        pass
        

def get_concn_ctrl_mat(net):
    pass


