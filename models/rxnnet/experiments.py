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


Each of **experiments** consists of a **condition** and **measurements**. 

"""

import collections
OD = collections.OrderedDict
import itertools

import numpy as np

from util import butil

'''
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



'''


class Condition(butil.DF):
    """
    A representation of condition
    """
    def __init__(self, data=None, index=None):
        columns = ['target', 'type', 'strength']  # perturbation
        super(Experiments, self).__init__(data=data, index=index, columns=columns)
    
    def __repr__(self):
        return self.values.tolist()    
    

class Experiments(butil.DF):
    """**Experiments** is a collection where each line consists of 
    an individual **experiment**, which is represented as a 3-tuple:
        (condition, varids, times), where 
            - condition itself is a 3-tuple of
            (perturbation target, perturbation type, perturbation strength);
                if 'perturbation type' is neglected, interpreted as multiplicative
            - varids
            - times  
            
            eg, (None, 'A', 1)
                (('k1',0.5), ['X','Y'], [1,2,3])
                ((), ['P'], [0.1, 0.5])
                ('', [('X1','k1'), ('X1','k2')], np.inf)
                (('V_R1','+',2), 'J_R2', np.inf) 

    """
    
    @property
    def _constructor(self):
        return Experiments
    
    
    def __init__(self, data=None, index=None, order_varid=None, group=True, 
                 info=''):
        """
        FIXME **: fix this constructor...
        
        Input:
            group: determine how the experiments are performed (grouped or not)
        """
        columns = ['condition','varids','times']
        super(Experiments, self).__init__(data=data, index=index, columns=columns)
        self.index.name = 'experiment'
        self.order_varid = order_varid
        self.group = group
        self.info = info
        

    '''
    def to_dids(self):
        Get the ids of data variables.

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
    '''

    def add(self, expt):
        """
        Input:
            expt: a 3-tuple or a mapping
        """
        if isinstance(expt, collections.Mapping):
            expt = butil.get_values(expt, self.columns)
            
        condition, varids, times = expt
        
        if condition in [(), None, '']:  # wildtype
            condition = ()
            
        if isinstance(varids, str):
            varids = [varids]
        else:
            varids = list(varids)
            
        times = np.float_(times)
        if isinstance(times, float):
            times = [times]
        else:
            times = list(times)
        
        self.loc[self.nrow+1] = (condition, varids, times)
        
    '''
    def delete(self, condition=None):
        if condition is not None:
            return self[self.condition != condition]
    
    def add_perturbation_series(self, pid, series, varids, times, mode=None):
        Input:
            varids: ids of measured variables
            times: measured times
            mode: perturbation mode: '+', '=', etc.
        
        for pert in series:
            if mode is None:
                cond = (pid, pert)
            else:
                cond = (pid, mode, pert)
            expt = (cond,) + tuple([varids, times])
            self.add(expt)
    
    def separate_conditions(self):
        cond2expts = OD()
        for cond, expts_cond in self.groupby('condition', sort=False):
            cond2expts[cond] = expts_cond
        return cond2expts
    
    
    
    def get_measurements(self):
        if len(set(self.condition)) > 1:
            raise ValueError("There are more than one conditions.")
        else:
            return [did[1:] for did in self.get_dids()]
    '''
        
    def get_condmsrmts_items(self):
        """
        """
        items = []
        if self.group:
            for cond, expts_cond in self.groupby('condition', sort=False):
                msrmts = []
                for varids, times in expts_cond.values[:, 1:]:
                    msrmts.extend(itertools.product(varids, times))
                items.append((cond, Measurements(msrmts, self.order_varid)))
        else:
            for cond, varids, times in self.values:
                msrmts = Measurements(list(itertools.product(varids, times)), 
                                      order_varid=self.order_varid)
                items.append((cond, msrmts))
        return items


    @property
    def yids(self):
        _yids = []
        for cond, msrmts in self.get_condmsrmts_items():
            _yids.extend([_get_yid(cond, msrmt) for msrmt in msrmts])
        return _yids


    @property
    def conds(self):
        # always the same as cond2msrmts.keys() ??
        return self.condition.drop_duplicates().tolist()  
        
    @property
    def varids(self):
        pass
    
    @property
    def times(self):
        pass
    
    
class Measurements(list):
    """
    """
    
    def __init__(self, msrmts, order_varid=None):
        """
        Three levels of information:
            - Coarsest level: varids, times
            - Intermediate level: varid2times
            - Finest level: msrmts
        
        All levels are useful:
            - Coarse level for guiding integration and quick inspection
            - Intermediate level for speedy quering of integration results
            - Finest level for generating yids
        
        Input:
            order_varid: a list of varids specifying the order of varids
                (as expts can be scrambled)
        """
        varids, times = zip(*msrmts)
        
        if order_varid is not None:
            varids = [varid for varid in order_varid if varid in varids]
        else:
            varids = butil.Series(varids).drop_duplicates().tolist()
        
        times = sorted(set(times))
        
        varid2times = OD()
        for varid in varids:
            varid2times[varid] = sorted([time_ for varid_, time_ in msrmts 
                                         if varid_==varid])
        
        msrmts = []
        for varid, times_ in varid2times.items():
            msrmts.extend([(varid, time) for time in times_])
        
        list.__init__(self, msrmts)
        
        self.varids = varids
        self.times = times
        self.varid2times = varid2times
        
    
    
def _get_yid(cond, msrmt):
    """
    """
    ## get condid
    #condid = ''.join([str(_) for _ in cond])
    condid = str(cond)
    
    ## get msrmtid
    varid, time = msrmt
    if isinstance(varid, str):
        pass
    elif isinstance(varid, tuple) and len(varid) == 2:
        varid = 'd %s/d %s'%varid
    else:
        raise ValueError
    # rstrip: get rid of the appending 0's and the decimal point
    msrmtid = ('%s, %f'%(varid, time)).rstrip('0').rstrip('.')
    
    # lstrip: get rid of the prepending ',' and whitespace (when condition
    # is wildtype)
    yid = ('%s, %s'%(condid, msrmtid)).lstrip(' ,')
    
    # very ad-hoc solution for now  FIXME ***
    yid = yid.replace("'","").replace(', =, ','=').replace('((','(').\
        replace(')),','),').replace('), (',', ')
    return yid


def get_experiments(zids, uids, us=None, grid=None, axis=None, info=''):
    """A convenience function for making experiments for reaction networks. 
    
    Input:
        zids: 
        uids:
        us: 
        grid:
        axis:
    """
    expts = Experiments(info=info)
    if uids in ['t', ['t'], 'T', ['T']]:
        expts.add(('', zids, us))
    else:
        if axis:
            conds = [((uid,'=',u),) for uid, u in butil.get_product(uids, axis)]
        else:
            if grid:
                us = butil.get_product(*[grid]*len(uids))
            conds = [tuple(zip(uids, ['=']*len(uids), u)) for u in us]
        for cond in conds:
            expts.add((cond, zids, np.inf))
    return expts

#expts_jc = get_expts(zids='J_R1', uids=['C1','C2'], grid=[1,2,3], info='JxC')
