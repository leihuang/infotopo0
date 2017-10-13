"""

FIXME ***: 
try to convert functions in modules into methods by binding the functions to the class? 
modules: structure and mca...

               
+------------+------------+-----------+----------+-----------+
| variables  |   values   |    ids    |  logids? |  comment  |
+============+============+===========+==========+===========+
|    vars    |  varvals   |   varids  |          | 
+------------+------------+-----------+----------+
|  optvars   |   p, p0    |   pids    |   yes    |
+------------+------------+-----------+----------+
|   params   |            |           |          |
+------------+------------+-----------+----------+
|  dynvars   |  x, x0, s  |   xids    |   yes    | 
+------------+------------+-----------+----------+
|            |            |   ixids   |          | 
+------------+------------+-----------+----------+
|            |            |   dxids   |          | 
+------------+------------+-----------+----------+
|    spp     |            |   spids   |          |
+------------+------------+-----------+----------+
|  asgvars   | asgvarvals | asgvarids |          | 
+------------+------------+-----------+----------+
|  algvars   |            | algvarids |          |
+------------+------------+-----------+----------+
|  ratevars  |            | ratevarids|          |
+------------+------------+-----------+----------+
|   ncvars   |            |           |          | dynvars + asgvars + algvars + ratevars
+------------+------------+-----------+----------+
|  convars   | convarvals | convarids |          |
+------------+------------+-----------+----------+
|            |     v      |   vids    |   yes    | 
+------------+------------+-----------+----------+
|            |     J      |   Jids    |   yes    | 
+------------+------------+-----------+----------+


+------------+------------+-----------+ 
|   rules    |    ids     |  comment  | 
+============+============+===========+ 
|    rxns    |   rxnids   |           |
+------------+------------+-----------+
|  ratelaws  |   rxnids   |           |
+------------+------------+-----------+
|  asgrules  | asgvarids  |           | 
+------------+------------+-----------+ 
|  algrules  | algvarids  |           |
+------------+------------+-----------+
|  raterules | ratevarids |           |
+------------+------------+-----------+


"""

from __future__ import division
from collections import OrderedDict as OD, Mapping
import copy
import re
import itertools
import os
import sympy
import logging

import numpy as np
import pandas as pd

from SloppyCell.ReactionNetworks import Network as Network0,\
    Dynamics, IO, KeyedList
from SloppyCell import ExprManip as exprmanip
from SloppyCell.daskr import daeintException

# FIXME ****
from util import butil
Series, DF = butil.Series, butil.DF

from util.matrix import Matrix

from infotopo import predict
reload(predict)

from infotopo.models.rxnnet import trajectory, structure, mca, algebra, experiments, ratelaw
reload(trajectory)
reload(mca)
reload(algebra)
reload(experiments)
reload(ratelaw)




class Network(object, Network0):
    """Turn Network into a new-style class as ``property`` is used extensively.
    
    Two levels of specifications:
        - biological: species, reactions, rates, parameters, etc.
        - mathematical: dynamical, assigned, algebraic, constant, optimizable, etc.
    
    """
    def __init__(self, id='', name='', net=None):
        if net is None:
            Network0.__init__(self, id, name)
            self.t = 0
        else:
            for attrid, attrval in net.__dict__.items():
                setattr(self, attrid, attrval)
            if not hasattr(self, 't'):
                self.t = 0
        
        #['id', 'compartments', 'parameters', 'species', 'reactions', 
        # 'assignmentRules', 'algebraicRules', 'rateRules', 'constraints', 'events', 'functionDefinitions']:
        
    @property
    def vars(self):
        return Series(OD(self.variables.items()), dtype=object)
    
    @property
    def varvals(self):
        return self.vars.apply(lambda var: var.value)
    #varvals = vals
    
    @property
    def varids(self):
        return self.variables.keys()
    
    @property
    def xdim(self):
        return len(self.x)
    
    @property
    def vdim(self):
        return len(self.rxns)
    
    @property
    def pdim(self):
        return len(self.p)
    
    @property
    def cdim(self):
        return len(self.convarids)
    
    @property  ## FIXME **: what's the use of it? remove it?
    def optvars(self):
        return Series(OD(self.optimizableVars.items()), dtype=object)
    
    @property
    def p(self):
        return Series([var.value for var in self.optimizableVars], self.pids, 
                      dtype=np.float) 

    @p.setter
    def p(self, p2):
        self.update_optimizable_vars(p2)
    
    @property
    def p0(self):
        #return self.optvars.apply(lambda var: var.initialValue)
        return Series([var.initialValue for var in self.optimizableVars], self.pids)
        
    @property
    def pids(self):
        return self.optimizableVars.keys()
    
    @property
    def logpids(self):
        return map(lambda pid: 'log_'+pid, self.pids)
    
    
    @property
    def pvars(self):
        return Series(OD(self.optimizableVars.items()), dtype=object)
    params = pvars  # FIXME **: backward compatibility; deprecation warning
    
    
    @property
    def xvars(self):
        return Series(OD(self.dynamicVars.items()), dtype=object)
    dynvars = xvars  # FIXME **: backward compatibility; deprecation warning
    
    
    def get_x(self, p=None, t=None, to_ser=False):
        if p is not None:
            self.update(p=p)
        self.update(t=t)
        return self.dynvars.apply(lambda var: var.value)
    x = property(get_x)
    
    
    @property
    def x0(self):
        return self.dynvars.apply(lambda var: self.evaluate_expr(var.initialValue))


    @x0.setter   ## FIXME ***: test it
    def x0(self, xinit2):
        for xid_, x0_ in xinit2.items():
            self.dynvars[xid_].initialValue = x0_
    
    
    @property
    def xids(self):
        return self.dynamicVars.keys() 
    
    
    @property
    def logxids(self):
        return map(lambda xid: 'log_'+xid, self.xids)
    
    @property
    def spvars(self):
        return Series(OD(self.species.items()), dtype=object)
    spp = spvars  # FIXME **: backward compatibility; deprecation warning
    
    @property
    def spids(self):
        return self.species.keys()
    
    @property
    def asgvars(self):
        return Series(OD(self.assignedVars.items()), dtype=object)
    
    @property
    def asgvarvals(self):
        return self.asgvars.apply(lambda var: var.value)
        
    @property
    def asgvarids(self):
        return self.assignedVars.keys()
    
    @property
    def algvars(self):
        return Series(OD(self.algebraicVars.items()), dtype=object)
    
    @property
    def algvarids(self):
        return self.algebraicVars.keys()
    
    @property
    def vvars(self):
        return self.vars[self.ratevarids]
    ratevars = vvars  # FIXME **: backward compatibility; deprecation warning
    
    @property
    def ratevarids(self):
        return self.rateRules.keys()
    
    @property
    def ncvars(self):  # non-constant variables  ## FIXME ***
        return Series.append(Series.append(Series.append(self.dynvars, self.asgvars), 
                                           self.algvars), self.ratevars)
    
    @property
    def ncids(self):
        return self.xids + self.asgvarids + self.algvarids #+ self.ratevarids
    ncvarids = ncids  # FIXME **: backward compatibility; deprecation warning
    ## FIXME ****: when net.add_species('X', 1, is_boundary_condition=1), 
    ## 'X' appears in both xids and algvarids
    
    @property
    def cvars(self):
        return Series(OD([(var.id, var) for var in self.constantVars if
                          var not in self.compartments and 
                          not var.is_optimizable]), dtype=object)
    convars = cvars  # FIXME **: backward compatibility; deprecation warning
        
    @property
    def c(self):
        return Series(OD([(var.id, var.value) for var in self.constantVars if 
                          var not in self.compartments and 
                          not var.is_optimizable]), dtype=np.float)
        ##self.convars.apply(lambda var: var.value)
    
    #convarvals = c  # FIXME **: backward compatibility; deprecation warning
    
    @property
    def convarvals(self):
        return self.constantVarValues
    
    
    @property
    def cids(self):
        return [var.id for var in self.constantVars if 
                var not in self.compartments and 
                not var.is_optimizable]
    convarids = cids  # FIXME **: backward compatibility; deprecation warning
    
    
    #def get_v(self, x=None, p=None, **varmap):
    #    self.update(x=x, p=p, **varmap)
        
    def get_v(self, to_ser=False):
        v = [self.evaluate_expr(rxn.kineticLaw) for rxn in self.reactions]
        if to_ser:
            return Series(v, self.vids)
        else:
            return np.array(v)
        
    
    @property
    def v(self):
        return self.get_v(to_ser=True)
    #rates = v


    @property
    def vids(self):
        return map(lambda _: 'v_'+_, self.rxnids)
    #rateids = vids

    
    @property
    def logvids(self):
        return map(lambda vid: 'log_'+vid, self.vids)

    
    @property
    def Jids(self):
        return map(lambda _: 'J_'+_, self.rxnids)
    #fluxids = Jids

    
    @property
    def logJids(self):
        return map(lambda Jid: 'log_'+Jid, self.Jids)

    @property
    def rxns(self):
        return Series(OD(self.reactions.items()), dtype=object)
    
    @property
    def rxnids(self):
        return self.reactions.keys()
    
    @property
    def ratelaws(self):
        return self.rxns.apply(lambda rxn: rxn.kineticLaw)

    @property
    def asgrules(self):
        return Series(OD(self.assignmentRules.items()), dtype=object)
        
    @property
    def algrules(self):
        return Series(OD(self.algebraicRules.items()), dtype=object)

    @property
    def raterules(self):
        return Series(OD(self.rateRules.items()), dtype=object)

    @property    
    def funcdefs(self):
        return Series(OD(self.functionDefinitions.items()), dtype=object)
        
    # FIXME ***:
    # call it theta? theta: parameters; p: optimizable parameters??
    # when cmp, we should compare theta. 
    
    
    def copy(self):
        """
        Why does copy.deepcopy miss the functional definitions in namespace?
        """
        net = copy.deepcopy(self)
        # the following block is taken from method 'change_varid'
        for fid, f in net.functionDefinitions.items():
            fstr = 'lambda %s: %s' % (','.join(f.variables), f.math)
            net.namespace[fid] = eval(fstr, net.namespace)
        return net
        
    
    def domain(self):
        """
        Output:
            domain: a pd.DataFrame 
        """
        pass
    
    
    def regularize(self):
        """Remove unnecessarily heterogeneous model specifications such as
        function definitions, assignment rules for conserved moieties...
        
        v = v(x, y, p)
        dx/dt = N v
        y = f(x, p)
        
        asgrules: 
        1) Conserved moiety, eg, ADP=3-ATP: y->x (make asgvar to be dynvar)
        2) Shorthand, eg, a=sqrt(k1/k2): y->p (make asgvar to be parameter)
        3) Monitor, eg, v_R1=k1*S: remove y (y does not feed back into system)
        4) True DAE, eg, from QE: X1=X/(1+KE): conceptually, it is k->infinity singular perturbation 
        
        =>
        
        v = v(x, p)
        dx/dt = N v
        
        Obselete:
        Regularize the network so that all parameters have ranges between 0
        and infinity.
        """
        net = self.remove_function_definitions()
        # how to systematically remove asgrules for conserved moieties?
        return net
         

    def clean(self):
        """Standardize the data structures and clean up the methods...
        Might break people's codes...
        
        rxn.ratelaw? (KineticLaw in SBML)
        pd.Series replacing KeyedList?
        Name conventions such as "rateRules"...
        
        FIXME *
        """
        net = self.copy()
        for rxn in net.reactions:
            if hasattr(rxn, 'kineticLaw'):
                rxn.ratelaw = rxn.kineticLaw
                
        # delete SloppyCell obsolete methods
        del net.get_eqn_structure
        del net.get_initial_velocities
        return net
    
    
    def get_eqns(self):
        """
        v = v(x, y, p)  ratelaws
        dx/dt = N v  odes
        y = f(x, p)  asgrules/algrules
        """
        eqns = Series([])
        eqns['ratelaws'] = self.ratelaws
        eqns['N'] = self.get_stoich_mat(only_dynvar=True)
        eqns['asgrules'] = self.asgrules.filt(f_key=lambda varid: not varid.startswith('v_'))
        eqns['algrules'] = self.algrules
        eqns['p'] = self.p
        eqns = eqns.filt(f_val=lambda _: len(_) > 0)
        return eqns
    
    
    def get_eqn(self, p=None, **kwargs):
        from infotopo import dynamics
        reload(dynamics)
        
        #consts = self.convarvals.tolist()[:-len(self.optimizableVars)]
        
        #if p is None:
        #    p = self.p
            
        f = lambda t, x:\
            self.res_function(t, x, [0]*len(x), self.convarvals)  #consts+list(p))

        eqn = dynamics.ODE(f, x0=self.x0, **kwargs) #, p=p, pids=self.pids, xids=self.xids)
        return eqn
    
    
    def get_eqns2(self):
        """
        FIXME ****
        """
        def _mul(row, vids):
            items = []
            for n, vid in zip(row, vids):
                if int(n) == 0:
                    continue
                elif n > 0:
                    items.append('%d*%s' % (n,vid))
                else:
                    items.append('(%d*%s)' % (n,vid))
            expr = '+'.join(items)
            return exprmanip.simplify_expr(expr)
        eqns = Series([_mul(row, self.vids)+' = 0' for row in self.N.values], 
                      ['d%s/dt=0'%xid for xid in self.xids])
        return eqns


    def get_ode_strs(self):
        """Doesn't work for DAE for now. 
        """
        _repl_ratelaw = lambda eqn: exprmanip(eqn, self.ratelaws.to_dict())
        eqns = [exprmanip.sub_for_vars(eqn.rstrip(' = 0').replace('v_',''),
                                       self.ratelaws.to_dict()) 
                for eqn in self.get_eqns2().values.tolist()]
        return eqns

    
    def add_reaction_ma(self, rxnid, stoich_or_eqn, p, reversible=True, 
                        haldane='kf', add_thermo=False, T=25):
        """
        Add a reaction with mass-action kinetics to a network in-place.
        
        Parameters
        ----------
        rxnid: a str; id of the reaction
        stoich_or_eqn: a mapping (stoich, eg, {'S':-1, 'P':1}) or 
            a str (eqn, eg, 'S<->P'); if an eqn is provided, 
            bidirectional arrow ('<->') denotes reversible reaction and 
            unidirectional arrow ('->') denotes irreversible reaction
        p: a dict or float
            If dict, it maps from pid to pinfo, where pinfo can be 
            a float (pval) or a tuple (pval, is_optimizable); 
            eg, p = {'kf':1, 'kb':2, 'KE':(1, False)}.
            If float, the default value of whatever parameters.
        reversible: a bool; specify whether the reaction is reversible
        haldane: a str; specify whether to constrain the kinetic parameters
            and hence reduce the number of independent parameters using 
            Haldane's relationship, which describes the thermodynamic 
            dependence of kinetic parameters: kf/kb = KE
            - '': not use haldane relationship 
                (both kf and kb are free parameters)
            - 'kf': use haldane relationship with 'kf' as the 
                independent parameter: kb = kf/KE
            - 'k': use haldane relationship with 'k' as the independent 
                parameter (aka parameter balancing by Liebermeister et al.):
                kf/kb=KE, kf*kb=k^2 => 
                    kf=k*sqrt(KE)
                    kb=k/sqrt(KE)
        add_thermo: a bool; specify whether to add thermodynamic variables
            if dG0 or KE is provided in p: dG=dG0+RT*log(Q), g=dG/RT
        T: a float; temperature in celsius, used for RT in thermodynamic 
            relationships
        """
        ## get stoich
        if isinstance(stoich_or_eqn, str):
            eqn = stoich_or_eqn
            stoich = _eqn2stoich(eqn)
            if '<' in eqn:
                reversible = True
            else:
                reversible = False
        else:
            stoich = stoich_or_eqn
        stoich = butil.chkeys(stoich, _format)
            
        ## get reactants (with multiplicity)
        subids = _get_substrates(stoich, multi=True)
        proids = _get_products(stoich, multi=True)
        
        ## add parameters
        RT = 8.31446 * (T + 273.15) / 1000
        if not isinstance(p, Mapping):
            p0 = p
            if not add_thermo:
                if haldane == '':
                    p = dict.fromkeys(['kf', 'kb'], p0)
                elif haldane == 'kf':
                    p = {'kf':(p0, True), 'KE':(p0, False)}
                else:
                    pass
            
        for pid, pinfo in p.items():
            if isinstance(pinfo, tuple):
                pval, is_optimizable = pinfo
            else:
                pval, is_optimizable = pinfo, True
            if pid == 'dG0':
                self.add_parameter('dG0_'+rxnid, pval, is_optimizable=False)
                self.add_parameter('KE_'+rxnid, np.exp(-pval/RT), 
                                   is_optimizable=False)
            elif pid == 'KE':
                self.add_parameter('KE_'+rxnid, pval, is_optimizable=False)
                self.add_parameter('dG0_'+rxnid, -np.log(pval)*RT, 
                                   is_optimizable=False)
            else:
                self.add_parameter('%s_%s'%(pid, rxnid), pval, 
                                   is_optimizable=is_optimizable)
        
        ## get ratelaw
        # get kbid
        kfid = 'kf_' + rxnid
        if haldane == '':
            kbid = 'kb_' + rxnid
        elif haldane == 'kf':
            kbid = '%s/KE_%s' % (kfid, rxnid)
        elif haldane == 'k':
            kbid, kid, KEid = 'kb_'+rxnid, 'k_'+rxnid, 'KE_'+rxnid
            if 'kf' in p and 'kb' in p:
                kfval, kbval = p['kf'], p['kb']
                self.add_parameter(KEid, kfval/kbval, is_optimizable=False)
                self.add_parameter(kid, np.sqrt(kfval*kbval), is_optimizable=True)
                self.add_assignment_rule(kfid, '%s*sqrt(%s)'%(kid, KEid))
                self.add_assignment_rule(kbid, '%s/sqrt(%s)'%(kid, KEid))
            elif 'dG0' in p or 'KE' in p:
                pass  # not implemented
            else:
                raise ValueError("haldane cannot be 'k' as p misses parameters")
        else:
            raise ValueError("not recognizable value of haldane")
        
        if reversible:
            ratelaw = '%s*%s-%s*%s' % (kfid, '*'.join(subids), 
                                       kbid, '*'.join(proids))
        else:
            ratelaw = '%s*%s' % (kfid, '*'.join(subids))
            
        ## add thermodynamic variables
        if add_thermo and ('dG0' in p or 'KE' in p):
            if 'RT' not in self.parameters.keys():
                self.add_parameter('RT', RT, is_optimizable=False)
            dGid, gid = 'dG_'+rxnid, 'g_'+rxnid
            self.add_parameter(dGid, 0, is_optimizable=False)
            self.add_assignment_rule(dGid, '%s+RT*log(%s/(%s))'%\
                ('dG0_'+rxnid, '*'.join(proids), '*'.join(subids)))
            self.add_parameter(gid, 0, is_optimizable=False)
            self.add_assignment_rule(gid, '%s/RT+log(%s/(%s))'%\
                ('dG0_'+rxnid, '*'.join(proids), '*'.join(subids)))
            
        ## add the reaction
        self.addReaction(rxnid, stoichiometry=stoich, kineticLaw=ratelaw)


    def add_reaction_qe(self, id, stoich_or_eqn, KE):
        """
        Add a reaction that is assumed to be at quasi-equilibrium (qe). 
        """
        self.add_reaction(id=id, stoich_or_eqn=stoich_or_eqn, 
                          p={'KE_'+id:(KE,False)}, ratelaw='0')
        # add algebraic rules
        
    
    def add_reaction_mm_qe(self, id, stoich_or_eqn, 
                           pM, pI=None, pA=None, 
                           reversible=True, haldane='Vf', 
                           mechanism='standard', 
                           states=None, states_plus=None, states_minus=None,
                           add_thermo=False, T=25):
        """
        Add a reaction assuming Michaelis-Menten (MM) kinetics with 
        quasi-equilibrium (QE) approximation.
        
        MMQE derives the ratelaw by thinking in terms of binding states of 
        an enzyme: an enzyme can be 1) free/unbound, 2) bound to one or more 
        reactants, 3) bound to one or more modifiers (inhibitors/activators).
        
        Once all binding states of an enzyme are specified, MMQE assumes them 
        to be in equilibrium (hence Michaelis constants in this formalism are
        dissociation constants), except the slow conversion between the 
        all-substrate-bound state (called the forward state, or state_f) and 
        all-product-bound state (called the backward state, or state_b). 
        Hence the ratelaw looks like:
        v = kf * [enzyme in state_f] - kb * [enzyme in state_b]
          = kf * E * prop(state_f) - kb * E * prop(state_b)
          = Vf * prop(state_f) - Vb * prop(state_b),
        where prop(state_i) is the proportion of enzyme in state_i and is given
        in the classical stat mech form (a term corresponding to the state
        divided by the partition function which is the sum of terms 
        corresponding to all states). 
        
        For example, S <-> P:
        v = (Vf * S/KM_S - Vb * P/KM_P) / (1 + S/KM_S + P/KM_P + I/KI), where
        S/KM_S, P/KM_P and I/KI represent the S-bound, P-bound and I-bound 
        states, respectively.
        
        States are represented as tuples here: for example, () is the 
        free state, ('S',) is the state bound to S, ('S1','S2','I') is the
        state bound to S1, S2 and I. 
        
        Abbreviations:
            sub: substrate
            pro: product
            rxt: reactant
            mod: modifier
            inh: inhibitor
            act: activator
            st: state
            f: forward
            b: backward
        
        Input:
            rxnid: a str; id of the reaction
            stoich_or_eqn: a mapping (stoich; eg, {'S':-1, 'P':1}) or 
                a str (eqn; eg, 'S<->P'); if an eqn is provided, 
                bidirectional arrow ('<->') denotes reversible reaction and 
                unidirectional arrow ('->') denotes irreversible reaction
            pM: a dict; maps from pid to pval; contains the necessary 
                parameters as minimally required by MM:
                    Vf, Vb, dG0, KE, KMs for reactants
            pI: a dict; maps from the id of an inhibitor to all the actions 
                the inhibitor exerts, represented by a dict mapping from 
                the states to which it binds to the corresponding KI 
                (dissociation constant)
            pA: a dict; maps from actid to ... (not implemented yet)
            reversible: a bool; specify whether the reaction is reversible; 
                can be inferred from eqn if given
            add_thermo: a bool; specify whether to add thermodynamic variables
                if dG0 or KE is provided in p: 
                dG=dG0+RT*log(Q), g=dG/RT (g: see Noor et al. 2014 PLoS Comp. Bio.)
            haldane: a str; for a _reversible_ reaction, specify whether to 
                constrain the kinetic parameters and hence reduce the number of
                independent parameters using Haldane's relationship, which 
                describes the thermodynamic dependence of kinetic parameters: 
                Vf/KMf/(Vb/KMb) = KE
                - '': not use Haldane's relationship
                - 'Vf': use Haldane's relationship and let Vf be the independent
                    parameter: Vb=(Vf/KMf)*(KMb/KE), hence numerator becomes
                    something like Vf/KMf*(S-P/KE)   
                - 'V': use Haldane's relationship and let V be the independent
                    parameter (aka parameter balancing by Liebermeister et al.):
                    Vf/KMf/(Vb/KMb)=KE => Vf/Vb=KE*KMf/KMb  
                    Vf*Vb=V^2                               
                    => Vf=V*sqrt(KE*KMf/KMb)
                       Vb=V/sqrt(KE*KMf/KMb)
            mechanism: a str; specify the reaction mechanism (see Liebermeister
                et al. 2010 Bioinformatics for details)
                - 'standard'/'CM':  
                - 'onestep'/'DB':
                - 'sequential'/'SB':
            states: a list of tuples; represent the bound states of enzyme;
                eg, [('S1',), ('S2',), ('S1','S2'), ('P1',), ('P2',), ('P1','P2')]
            states_plus/states_minus: a list of tuples; sometimes states 
                are specified by mechanism with minor variations, and the two
                parameters allow for such minor modifications for convenience
            T: a float; temperature in celsius, used for RT in thermodynamic 
                relationships
        """
        rxnid = id  ## FIXME **
        
        ## get states (from stoich, p (with modifier info), states, mechanism)
        # get stoich and reversible
        if isinstance(stoich_or_eqn, str):
            eqn = stoich_or_eqn
            stoich = _eqn2stoich(eqn)
            if '<' in eqn:
                reversible = True
            else:
                reversible = False
        else:
            stoich = stoich_or_eqn
        stoich = butil.chkeys(stoich, func=_format)
        
        # get reactants (with multiplicity)
        subids = _get_substrates(stoich, multi=True)
        proids = _get_products(stoich, multi=True)
        
        _get_Kid = lambda state: '*'.join(['KM_%s_%s'%(rxnid, spid) for spid in state])\
            if state != () else '1' 

        # get states involving only reactants and the corresponding K terms
        if mechanism in ['onestep', 'DB'] and states is None:
            if reversible:
                states = [(), tuple(subids), tuple(proids)]
                Kids = ['1', 'KMf_'+rxnid, 'KMb_'+rxnid]
            else:
                states = [(), tuple(subids)]
                Kids = ['1', 'KMf_'+rxnid]
        else:
            if states is not None:
                states = [tuple(map(_format, st)) for st in states]
            elif mechanism in ['standard', 'CM']:
                if reversible:  
                    states = butil.powerset(subids) + butil.powerset(proids)[1:]
                else:
                    states = butil.powerset(subids)
            elif mechanism in ['sequential', 'SB']:
                # not checked yet
                if reversible:
                    states = [tuple(subids)[:i] for i in range(1,len(subids)+1)] +\
                             [tuple(proids)[:i] for i in range(1,len(proids)+1)] + [()]
                else:
                    states = [tuple(subids)[:i] for i in range(1,len(subids)+1)] + [()]
            else:
                raise ValueError("Wrong value of mechanism: %s" % mechanism)
            Kids = map(_get_Kid, states)
        state2Kid = dict(zip(states, Kids))

        ## add parameters and get states involving modifiers and 
        ## the corresponding Kids
        RT = 8.31446 * (T + 273.15) / 1000
        
        # pM: Vf, Vb, dG0, KE, KM
        for pid, pval in pM.items():
            pid = _format(pid)
            if pid in ['Vf', 'Vb']:
                self.add_parameter('%s_%s'%(pid, rxnid), pval, is_optimizable=True)
            elif pid == 'dG0':
                self.add_parameter('dG0_'+rxnid, pval, is_optimizable=False)
                self.add_parameter('KE_'+rxnid, np.exp(-pval/RT), is_optimizable=False)
            elif pid == 'KE':
                self.add_parameter('KE_'+rxnid, pval, is_optimizable=False)
                if add_thermo:
                    self.add_parameter('dG0_'+rxnid, -np.log(pval)*RT, is_optimizable=False)
            elif pid in subids+proids:
                self.add_parameter('KM_%s_%s'%(rxnid, pid), pval, is_optimizable=True)
            else:
                raise ValueError("Wrong parameter id: %s" % pid)
            
        # pI = {'I1':{state: KI1}, 'I2':{(state1, state2): KI2}}
        if pI is None:
            pI = {}
        if pA is None:
            pA = {}    
        
        for inhid, info in pI.items():
            idx = 1
            for sts_bnd, pval in info.items():
                # the inhibitor has only one action, hence no indexing
                if len(info) == 1:
                    KIid = 'KI_%s_%s' % (rxnid, inhid)
                else:
                    KIid = 'KI%d_%s_%s' % (idx, rxnid, inhid)
                self.add_parameter(KIid, pval, is_optimizable=True)
                idx += 1
                _get_Kid_inh = lambda st_bnd, KIid: KIid if st_bnd == () else\
                    '%s*%s' % (state2Kid[st_bnd], KIid)
                # sts_bnd is a tuple of bound states
                if sts_bnd != () and isinstance(sts_bnd[0], tuple):
                    for st_bnd in sts_bnd:
                        state = st_bnd + tuple([inhid])
                        Kid = _get_Kid_inh(st_bnd, KIid) 
                        state2Kid[state] = Kid
                # sts_bnd is a single bound state 
                else:
                    st_bnd = sts_bnd
                    state = st_bnd + tuple([inhid])
                    Kid = _get_Kid_inh(st_bnd, KIid) 
                    state2Kid[state] = Kid
        
        # pA = ...
        for actid, info in pA.items():
            pass
        
        if states_plus:
            for state in states_plus:
                state2Kid[state] = _get_Kid(state)
        if states_minus:
            for state in states_minus:
                del state2Kid[state]
        
        ## get rate law
        Vfid, Vbid = 'Vf_'+rxnid, 'Vb_'+rxnid
        actids = pA.keys()
        state_f = tuple(subids) + tuple(actids) 
        state_b = tuple(proids) + tuple(actids)
        
        # get numerator
        KMfid = state2Kid[state_f]
        if reversible:
            KMbid = state2Kid[state_b]
            KEid, Vid = 'KE_'+rxnid, 'V_'+rxnid
            if haldane in ['','V']:
                if haldane == 'V':
                    self.add_parameter(Vid, np.sqrt(pM['Vf']*pM['Vb']))
                    self.add_assignment_rule(Vfid, '%s*sqrt(%s*%s/%s)'%(Vid,KEid,KMfid, KMbid))
                    self.add_assignment_rule(Vbid, '%s/sqrt(%s*%s/%s)'%(Vid,KEid,KMfid, KMbid))
                numerator = '%s*%s/(%s)-%s*%s/(%s)' %\
                    (Vfid, '*'.join(state_f), KMfid, Vbid, '*'.join(state_b), KMbid)
            elif haldane == 'Vf':
                numerator = '%s/(%s)*(%s-%s/%s)' %\
                    (Vfid, KMfid, '*'.join(state_f), '*'.join(state_b), KEid)
            else:
                raise ValueError("Wrong value of haldane: %s" % haldane)
        else:
            numerator = '%s*%s/(%s)' % (Vfid, '*'.join(state_f), KMfid)

        # get denominator
        denominator = '1+' + '+'.join(['%s/(%s)'%('*'.join(st), Kid) if '*' in Kid 
                                       else '%s/%s'%('*'.join(st), Kid) 
                                       for st, Kid in state2Kid.items() if st!=()])
        ratelaw = '(%s)/(%s)' % (numerator, denominator)
        
        ## add thermodynamic variables: dG, g
        if add_thermo and ('dG0' in pM or 'KE' in pM):
            if 'RT' not in self.parameters.keys():
                self.add_parameter('RT', RT, is_optimizable=False)
            dGid, gid = 'dG_'+rxnid, 'g_'+rxnid
            self.add_parameter(dGid, is_optimizable=False)
            self.add_assignment_rule(dGid, '%s+RT*log((%s)/(%s))'%\
                                     ('dG0_'+rxnid, '*'.join(proids), '*'.join(subids)))
            self.add_parameter(gid, is_optimizable=False)
            self.add_assignment_rule(gid, '%s/RT+log((%s)/(%s))'%\
                                     ('dG0_'+rxnid, '*'.join(proids), '*'.join(subids)))
            
        ## add modifiers to stoich 
        # otherwise some applications, eg, Copasi and JWS online, 
        # would complain (I do not know a good reason for that yet); 
        # here, modifiers are 1) are defined as variables in ratelaw that are 
        # neither reactants nor parameters;
        # 2) represented as species in stoich with stoichcoef of 0 (when 
        # exported to SBML, they would be picked up and specified)
        modids = [varid for varid in exprmanip.extract_vars(ratelaw) 
                  if varid not in stoich.keys()+self.parameters.keys()]
        for modid in modids:
            stoich[modid] = 0

        ## add reaction
        self.addReaction(rxnid, stoichiometry=stoich, kineticLaw=ratelaw)
        
            
    def add_reaction(self, id, stoich_or_eqn, ratelaw, p=None, **kwargs):
        """
        Add a reaction with all the given information, in a sense a wrapper
        of the SloppyCell.ReactionNetworks.Network.addReaction.
        
        Input:
            rxnid: a str; id of the reaction
            stoich_or_eqn: a mapping (stoich, eg, {'S':-1, 'P':1}) or 
                a str (eqn, eg, 'S<->P'); if an eqn is provided, 
                bidirectional arrow ('<->') denotes reversible reaction and 
                unidirectional arrow ('->') denotes irreversible reaction
            ratelaw:
            p: a dict; map from pid to pinfo, where pinfo can be 
                a float (pval) or a tuple (pval, is_optimizable); 
                eg, p={'V_R1':1, 'KM_S':2, 'KE_R1':(1, False)}
        """
        rxnid = id  ## FIXME **
        
        ## get stoich
        if isinstance(stoich_or_eqn, str):
            eqn = stoich_or_eqn
            stoich = _eqn2stoich(eqn)
        else:
            stoich = stoich_or_eqn
        stoich = butil.chkeys(stoich, func=_format)  
        
        ## add parameters
        if p is not None:
            for pid, pinfo in p.items():
                if isinstance(pinfo, tuple):
                    pval, is_optimizable = pinfo
                else:
                    pval, is_optimizable = pinfo, True
                if pid in self.parameters.keys():
                    if self.parameters.get(pid).value != pval:
                        raise ValueError("Value of parameter %s in reaction %s different."%\
                                         (pid, rxnid)) 
                else:
                    self.add_parameter(pid, pval, is_optimizable=is_optimizable)
                    
        ## add modifiers 
        # otherwise some applications, eg, Copasi and JWS online, 
        # would complain - I do not know a good reason for that yet; 
        # here, modifiers are 1) are defined as variables in ratelaw that are 
        # neither reactants nor parameters;
        # 2) represented as species in stoich with stoichcoef of 0 (when 
        # exported to SBML, they would be picked up and specified)
        modids = [varid for varid in exprmanip.extract_vars(ratelaw) 
                  if varid not in stoich.keys()+self.parameters.keys()]
        for modid in modids:
            stoich[modid] = 0
        
        # add reaction
        self.addReaction(id=rxnid, stoichiometry=stoich, kineticLaw=ratelaw, **kwargs)
    

    def add_ratevars(self):
        """
        Add rate variables.
        """
        for rxn in self.reactions:
            rateid = 'v_' + rxn.id
            try:
                self.add_parameter(rateid, is_constant=False, is_optimizable=False)
                self.add_assignment_rule(rateid, rxn.kineticLaw)
            except ValueError:
                pass
    
    
    def print_details(self):
        """
        """
        print "Species:"
        for sp in self.species:
            print '\t', (sp.id, sp.name, sp.value)
            
        print "\nReactions:"
        for rxn in self.reactions:
            print '\t', (rxn.id, rxn.name, rxn.stoichiometry, rxn.kineticLaw)
            
        print "Optimizable Parameters:\n\t",\
            [(v.id, v.value) for v in self.optimizableVars]
        print "Non-optimizable Parameters:\n\t",\
            [(p.id, p.value) for p in self.parameters if not p.is_optimizable]
        print "Assignment Rules:\n\t",\
            self.assignmentRules.items()
        print "Rate Rules:\n\t",\
            self.rateRules.items()
    
    
    def get_uses(self, varid):
        """
        """
        uses, uses_rxn, uses_asgrule = OD(), OD(), OD()
    
        for rxn in self.reactions:
            if varid in exprmanip.extract_vars(rxn.kineticLaw):
                uses_rxn[rxn.id] = rxn.kineticLaw
        uses['rxn'] = uses_rxn
        
        for asgvarid, asgrule in self.asgrules.iteritems():
            if varid in exprmanip.extract_vars(asgrule):
                uses_asgrule[asgvarid] = asgrule
        uses['asgrule'] = uses_asgrule
        """
        for algrule in self.algrules:
            pass
        for raterule in self.raterules:
            pass
        """
        return uses
        
    
    def del_varid(self, varid):
        """
        """
        self.remove_component(varid)
        for rxn in self.reactions:
            if varid in rxn.stoichiometry:
                del rxn.stoichiometry[varid]                
            if varid in rxn.parameters:
                rxn.parameters.remove(varid)
                
                
    def del_varids(self, varids):
        """
        """
        for varid in varids:
            self.del_varid(varid)

    
    def replace_varid(self, varid_old, varid_new, only_expr=False):
        """
        Change id of rxn, species, or parameter.
        Ratelaws, assignmentRules, rateRules
        
        Input:
            only_expr: if True, only replace varid in expressions such as
                reaction ratelaws, assignment rules, etc.; 
                useful when varid_new is like 'varid_old * r'
        """
        vid, vid2 = varid_old, varid_new
        if only_expr:
            f = lambda _: _
        else:
            f = lambda _: vid2 if _ == vid else _
        
        netid2 = f(self.id)
        net2 = self.__class__(netid2, name=self.name)
        
        if hasattr(self, 't'):
            net2.t = self.t
        
        vars2 = KeyedList()
        for var in self.variables:
            var2 = copy.deepcopy(var)
            var2.id = f(var.id)
            vars2.set(var2.id, var2)
        net2.variables = vars2
        
        rxns2 = KeyedList()
        for rxn in self.reactions:
            rxn2 = copy.deepcopy(rxn)
            rxn2.id = f(rxn.id)
            rxn2.stoichiometry = butil.chkeys(rxn.stoichiometry, f)
            try:
                rxn2.reactant_stoichiometry = butil.chkeys(rxn.reactant_stoichiometry, f)
                rxn2.product_stoichiometry = butil.chkeys(rxn.product_stoichiometry, f)
            except AttributeError:  # some reactions have no reactant/product stoich
                pass
            rxn2.parameters = set(map(f, rxn.parameters))
            rxn2.kineticLaw = exprmanip.sub_for_var(rxn.kineticLaw, vid, vid2)
            rxns2.set(rxn2.id, rxn2)
        net2.reactions = rxns2
        
        asgrules2 = KeyedList()  # assignment rules
        for varid, rule in self.assignmentRules.items():
            varid2 = f(varid)
            rule2 = exprmanip.sub_for_var(rule, vid, vid2)
            asgrules2.set(varid2, rule2)
        net2.assignmentRules = asgrules2
        
        algrules2 = KeyedList()  # algebraic rules
        for varid, rule in self.algebraicRules.items():
            varid2 = exprmanip.sub_for_var(varid, vid, vid2)
            rule2 = exprmanip.sub_for_var(rule, vid, vid2)
            algrules2.set(varid2, rule2)
        net2.algebraicRules = algrules2
         
        raterules2 = KeyedList()  # rate rules
        for varid, rule in self.rateRules.items():
            varid2 = f(varid)
            rule2 = exprmanip.sub_for_var(rule, vid, vid2)
            raterules2.set(varid2, rule2)
        net2.rateRules = raterules2    

        for eid, event in self.events.items():
            eid2 = f(eid)
            trigger2 = exprmanip.sub_for_var(event.trigger, vid, vid2)
            assignments2 = butil.chkeys(event.event_assignments, f)
            net2.add_event(eid2, trigger2, assignments2)
            
        net2.functionDefinitions = self.functionDefinitions.copy()
        for fid, f in net2.functionDefinitions.items():
            fstr = 'lambda %s: %s' % (','.join(f.variables), f.math)
            net2.namespace[fid] = eval(fstr, net2.namespace)

        ## final processings
        # method _makeCrossReferences will take care of at least
        # the following attributes:
        # assignedVars, constantVars, optimizableVars, dynamicVars, 
        # algebraicVars
        net2._makeCrossReferences()
        return net2
    
    
    def replace_varids(self, varidmap):
        """
        Input:
            varidmap: a mapping from old varids to new varids 
        """
        net = self.copy()
        for vid, vid2 in varidmap.items():
            net = net.replace_varid(vid, vid2)
        return net
    
    
    def replace_ratelaw(self, rxnid, rl, facelift=False):
        net = self.copy()
        
        rxn = copy.deepcopy(net.rxns[rxnid])
        rxnids = net.rxnids
                
        for pid in rxn.parameters:
            # FIXME***: need to test the uses of the pid; some are shared 
            # by other reactions as well
            net.remove_component(pid)
        net.remove_component(rxnid)
        
        if facelift:
            rl = rl.facelift(xids_new=rxn.stoichiometry.keys(),  
                             pcmap='rxnid', rxnidx=rxnid)

        net.add_reaction(rxnid, stoich_or_eqn=rxn.stoichiometry, 
                         ratelaw=rl.s, p=OD.fromkeys(rl.pids, 1))
        
        net = net.reorder_reactions(rxnids)
        
        vid = 'v_' + rxnid 
        if vid in net.asgvarids:
            net.assignmentRules.set(vid, net.ratelaws[rxnid])
        
        net.compile()  # FIXME ****: it does NOT seem to update!!
        return net
        
        
         
    def remove_function_definitions(self):
        """
        Only replace ratelaws of reactions so far...
        
        Check out exprmanip.Substitution.sub_for_func... Prabably more elegant.
        """
        net2 = self.copy()
        for fid, f in self.functionDefinitions.items():
            for rxnid, rxn in self.reactions.items():
                ratelaw = rxn.kineticLaw
                fids_rxn = [_[0] for _ in exprmanip.extract_funcs(ratelaw)] 
                if fid in fids_rxn:
                    # an example of fstr: 'function_1(Vf_R1,KM_R1_X1,X1)'
                    fstr = re.search('%s\(.*?\)'%fid, ratelaw).group()
                    # an example of varids2: ['Vf_R1','KM_R1_X1','X1']
                    varids2 = [_.strip() for _ in 
                               re.search('(?<=\().*(?=\))', fstr).group().split(',')]
                    # an example of fstr2: 'Vf_R1*X1/(KM_R1_X1+X1)'
                    fstr2 = exprmanip.sub_for_vars(f.math, 
                                              dict(zip(f.variables, varids2)))
                    ratelaw2 = ratelaw.replace(fstr, '(%s)'%fstr2)
                    net2.reactions.get(rxnid).kineticLaw = ratelaw2
        net2.functionDefinitions = KeyedList()
        return net2
    
    
    def add_names(self, id2name=None):
        """
        """
        if id2name is None:
            id2name = OD(zip(self.varids+self.rxnids, self.varids+self.rxnids))
        for _id, name in id2name.items():
            if _id in self.varids:
                self.vars[_id].name = name
            if _id in self.rxnids:
                self.rxns[_id].name = name
    
    def add_compartment(self, id, *args, **kwargs):
        """A wrapper of SloppyCell that accepts ...
        """
        if id in self.compartments.keys():
            pass
        else:
            Network0.add_compartment(self, id, *args, **kwargs)


    def add_species(self, *args, **kwargs):
        """A wrapper of the SloppyCell method so that compartment does not
        have to be passed in if there is only one compartment.
        """
        cmptids = self.compartments.keys()
        
        # input has compartment info
        if any([cmptid in list(args)+kwargs.values() for cmptid in cmptids]):
            Network0.add_species(self, *args, **kwargs)
        # input has no compartment info
        else:
            if len(cmptids) == 0:
                cmptid = 'cell'
                self.add_compartment(cmptid)
            elif len(cmptids) == 1:
                cmptid = cmptids[0]
            else:
                raise ValueError("Compartment has to be provided.")
            args = list(args)
            args.insert(1, cmptid)    
            Network0.add_species(self, *args, **kwargs)
        
    
    def add_spp(self, **kwargs):
        """
        """
        assert len(self.compartments) == 1, "network has more than one compartments."
        cmptid = self.compartments.keys()[0]
        for spid, concn0 in kwargs.items():
            self.add_species(spid, cmptid, concn0)
    
    def add_p(self, **kwargs):
        """
        """
        for pid, pval in kwargs.items():
            self.add_parameter(pid, pval, is_optimizable=True)
    
    
    def get_eq_pools(self):
        """
        Get the list of equilibrium pools, where an equilibrium pool is a set
        of complexes that are at equilibrium with each other (a complex is 
        a multiset of species).  
        
        Output:
            a list of sets
        """
        eqpools = []
        for rxn in self.reactions:
            if rxn.kineticLaw == '0':
                subids = tuple(_get_substrates(rxn.stoichiometry, multi=False))
                proids = tuple(_get_products(rxn.stoichiometry, multi=False))
                for eqpool in eqpools:
                    if subids in eqpool or proids in eqpool:
                        eqpool.add(subids)
                        eqpool.add(proids)
                        break
                eqpools.append(set([subids, proids]))
        return eqpools
        
        
    def resolve_eq_pools(self, scheme='kinetics', share_k=False, kf=1e6, 
                         eqpoolids=None):
        """
        Input:
            scheme: 'kinetics' or 'algebra'; 'algebra' only works for uni-uni
                reactions
            share_k: 
        """
        net = self.copy()
        if scheme == 'kinetics':
            net.add_parameter('kf', kf, is_optimizable=False)
            for rxn in net.rxns:
                if rxn.kineticLaw == '0':
                    if share_k:
                        kfid = 'kf'
                    else:
                        kfid = 'kf_' + rxn.id
                        net.add_parameter(kfid)
                        net.add_assignment_rule(kfid, 'kf')
                    subids = _get_substrates(rxn.stoichiometry, multi=True)
                    proids = _get_products(rxn.stoichiometry, multi=True)
                    rxn.kineticLaw = '%s*(%s-%s/KE_%s)'%(kfid,
                                                         '*'.join(subids),
                                                         '*'.join(proids), 
                                                         rxn.id)
        elif scheme == 'algebra':
            eqpools = net.get_eq_pools()
            if eqpoolids is None:
                eqpoolids = ['pool%d'%idx for idx in range(1, len(eqpools)+1)]
            for eqpoolid, eqpool in zip(eqpoolids, eqpools):
                if all([len(rxtids)==1 for rxtids in eqpool]):
                    spids = [rxtids[0] for rxtids in eqpool]
                    net.add_parameter(eqpoolid, is_optimizable=False)
                    net.add_assignment_rule(eqpoolid, '+'.join(spids))
                    net.add_rate_rule(eqpoolid, )
                    for spid in spids:
                        net.species.get(spid).is_boundary_condition = True
                        net.add_assignment_rule(spid, )
                else:
                    # add kinetics?
                    raise Exception("Pool %s has complexes."%str(eqpool))
        return net
        
        
    def set_KE_nonoptimizable(self):
        """
        sbml import loses the optimizability information
        """
        for pid in self.pids:
            if pid.startswith('KE'):
                self.set_var_optimizable(pid, False)
        self._makeCrossReferences()
        
        
    def make_abs_differentiable(self, scheme='max', eps=1e-6):
        """
        FIXME *: 
        make it a function?
        
        SloppyCell cannot do differentiation for 'abs' function, so replace it
        with a differentiable equivalent. 
        So far it is sort of a hack (one of the costs is to have to define
        an addition function 'dabs'). 
        A systematic way of doing it is to replace 'abs' by the appropriate 
        SloppyCell-differentiable function through ast (abstract syntax tree) 
        utilities offered in exprmanip. 
        
        Input: 
            - scheme: 'max' or 'sqrt'
                'max: max(x, -x) = abs(x)
                'sqrt': lim_{eps->0} sqrt(x^2+eps) -> abs(x)
            - eps: used for scheme of 'sqrt'
        """
        net = self.copy()
        # 'dabs' for differentiable abs
        if scheme == 'max':
            net.add_func_def('dabs', ('x'), 'max(x, -x)')
        if scheme == 'sqrt':
            net.add_func_def('dabs', ('x'), 'sqrt(x**2+%f)'%eps)
        
        for rxn in net.reactions:
            rxn.kineticLaw = rxn.kineticLaw.replace('abs(', 'dabs(')
        for asgvarid, asgrule in net.assignmentRules.items():
            net.assignmentRules.set(asgvarid, asgrule.replace('abs(', 'dabs('))
        for ratevarid, raterule in net.rateRules.items():
            net.rateRules.set(ratevarid, raterule.replace('abs(', 'dabs('))
        
        return net
    
    
    def perturb0(self, condition):
        """
        """
        if condition in [(), None, '']:
            return self
        else:
            net = self.copy()
            if len(condition) == 2:
                pid, mode = condition[0], '*'  # default
            else:
                pid, mode = condition[:2]
                
            if pid in self.rxnids:
                if 'Vf_%s' % pid in self.pids:
                    pid = 'Vf_%s' % pid
                elif 'kf_%s' % pid in self.pids:
                    pid = 'kf_%s' % pid
                else:
                    raise ValueError("neither Vf_%s nor kf_%s are pids." % (pid, pid))
            
            if mode in ['*', '/', '+', '-']:
                change = condition[-1]
                #pid2 = pid + '_new'
                #net.add_parameter(pid2, is_optimizable=False)
                pid2 = '(%s%s%s)'%(pid, mode, change)
                # ratelaw, assignmentrules, raterules
                net = net.replace_varid(pid, pid2, only_expr=True)  
                #net.add_assignment_rule(pid2, '%s%s%s'%(pid, mode, str(change)), 0)
            if mode == '=':
                pval_new = condition[-1]
                net.set_var_val(pid, pval_new)  # need to verify...
        return net
        
        
    
    def perturb(self, condition):
        """
        """
        if condition == ():  # wildtype  
            return self
        else:  # perturbation
            net = self.copy()
            for perturbation in condition:
                varid, mode, strength = perturbation
                if mode in ['*', '/', '+', '-']:
                    varid2 = '(%s%s%s)'%(varid, mode, strength)
                    net = net.replace_varid(varid, varid2, only_expr=True)  
                if mode == '=':
                    net.set_var_val(varid, strength)  # need to verify...
            return net
        
        
    def measure(self, msrmts, to_ser=False, **kwargs):
        """
        Input:
            msrmts: measurements, a list of (varid, time) with attributes
                eg, [('A',1), ('J_R1',np.inf)]
        
        """
        traj = self.get_traj(msrmts.times, varids=msrmts.varids, **kwargs)
        y = []
        for varid, times in msrmts.varid2times.items():
            y.extend(traj[varid][times].tolist())
        if to_ser:
            y = Series(y, index=msrmts)
        return y    
        
    
    def get_psens(self, msrmts, to_mat=False, **kwargs):
        """Get the *parameter* sensitivities of the quantities in measurements 
        (msrmts). 
        
        Input:
            msrmts: measurements, a list of (varid, time)
                eg, [('A',1), ('J_R1',np.inf)]
        
        Output:

        """
        _get_derivids = lambda varids: list(itertools.product(varids, self.pids))
        
        traj = self.get_traj(msrmts.times, varids=_get_derivids(msrmts.varids),
                             **kwargs)
        jac = []
        for varid, times in msrmts.varid2times.items():
            jac.extend(traj.loc[times, _get_derivids([varid])].values.tolist())   
        if to_mat:
            jac = Matrix(jac, msrmts, self.pids)
        return jac    

    
    """
    def get_predict0(self, expts, **kwargs_prior):
        
        Returns a predict object, essentially f = X*M, where M is the models and
        X is the design variable.
        
        Input:
            expts: eg,
                condition variable time
            1        ()        S    np.inf
            2   (k1, 2)        S    [1,np.inf]
            3   (k2, 2)        S    [2,10]
        
        
        expts_worked = expts.copy()
        
        condmap = OD()
        for cond, expts_cond in expts.separate_conditions().items():
            net = self.perturb(cond)
            net.id = net.id + '__' + '__'.join([str(_) for _ in cond])
            net.compile()
            try:
                net.set_ss()
            except Exception:
                expts_worked = expts_worked.delete(cond)
                continue
            dids_cond = expts_cond.to_dids()
            msrmts_cond = [did[1:] for did in dids_cond]
            condmap[cond] = (net, msrmts_cond)
        cond2net = butil.chvals(condmap, lambda tu: tu[0])
            
        pids = self.pids
        yids = expts_worked.to_yids()
        
        def f(p):
            y = []
            for net, msrmts in condmap.values():
                net.update(p)
                y_cond = net.measure(msrmts, to_ser=False)
                y.extend(y_cond)
            y = Series(y, yids)
            return y
        
        def Df(p):
            jac = []
            for net, msrmts in condmap.values():
                net.update(p)
                jac_cond = net.get_psens(msrmts, to_mat=False)
                jac.extend(jac_cond)
            jac = Matrix(jac, yids, pids)
            return jac
                
        pred = predict.Predict(f=f, Df=Df, p0=self.p, pids=pids, yids=yids,
                               expts=expts_worked, cond2net=cond2net)
        if kwargs_prior:
            pred.set_prior(**kwargs_prior)
        return pred
    """


    def get_predict(self, expts, **kwargs): 
        """Returns a predict object, essentially f = X * F, 
        where M is the models and X is the design variable.
        
        Input:
            expts: eg,
                    condition variable time
                1        ()        S    np.inf
                2   (k1, 2)        S    [1,np.inf]
                3   (k2, 2)        S    [2,10]
            name: 
            kwargs: kwargs for dynamic and steady-state calculation
        """
        import dynlite, mcalite
        reload(dynlite)
        reload(mcalite)
        if np.inf in butil.flatten(expts['times']):
            return mcalite.get_predict(self, expts, **kwargs)
        else:
            return dynlite.get_predict(self, expts, **kwargs)
            
        
        """
        if test:
            for cond in expts.condset:
                try:
                    self.perturb(cond, inplace=True)
                    self.set_ss()
                except Exception:
                    expts.rm_condition(cond)
                    logging.warn("Remove condition: %s"%str(cond))

        
        # this will ...
        #expts = expts.regularize()
        
        ## FIXME ***: 
        # When get y, one often needs to integrate
        # When get jac, one often needs to sens-integrate, which is a superset
        # of integrate (?): repetition of labor 
        # Something like this:
        # varids = yids + sensids, which are product(yids, pids) 
        # traj = net.get_traj(times, varids)
        # y = traj[yids]
        # jac = traj[sensids]
        net0 = self.copy()
        
        items = expts.get_condmsrmts_items()
        items2 = [(net0.perturb(cond), msrmts) for cond, msrmts in items]
        
        def f(p):
            y = []
            for net, msrmts in items2:
                net.update(p=p)
                y_cond = net.measure(msrmts, **kwargs)
                y.extend(y_cond)
            return np.array(y)
        
        def Df(p):
            jac = []
            for net, msrmts in items2:
                net.update(p=p)
                jac_cond = net.get_psens(msrmts, **kwargs)
                jac.extend(jac_cond)
            return np.array(jac)

        if p0 is None:
            p0 = net0.p0
        pred = predict.Predict(f=f, Df=Df, p0=net0.p, name=name, pids=net0.pids, 
                               yids=expts.yids, expts=expts)
        return pred
        """        
        
        
        """
            for cond, expts_cond in expts.sep_conditions().items():
                net = self.perturb(cond)
                net.update(p)
                dids_cond = expts_cond.to_dids()
                msrmts_cond = [did[1:] for did in dids_cond]
                y_cond = net.measure(msrmts_cond)
                # y_cond is a pd.Series instance
                y.extend(y_cond.tolist())
            y = pd.Series(y, index=dids)
            return y
        
        def Df(p):
            jac = []
            for cond, expts_cond in expts.sep_conditions().items():
                net = self.perturb(cond)
                net.update(p)
                dids_cond = expts_cond.to_dids()
                msrmts_cond = [did[1:] for did in dids_cond]
                jac_cond = net.get_sensitivities(msrmts_cond)
                # jac_cond is an MCAMatrix instance
                jac.extend(jac_cond.values.tolist())
            jac = pd.DataFrame(jac, index=dids, columns=pids)
            return jac
        """
    
    def get_predict_vfield(self, ts):
        pass
        

###############################################################################
###############################################################################

    ## calculating methods
    #  dynamics
    def get_traj(self, times, p=None, varids=None, copy_net=False, **kwargs):
        """
        Input:
            times: a list of numbers (time series), or
                   a tuple of two numbers (an interval, dense sampling)
            copy: really necessary??? check the code behaviors... FIXME **
             
        Output:
            traj
        """
        if copy_net:
            net = self.copy()
        else:
            net = self
        
        if p is not None:
            net.update(p=p)
        
        # assuming traj_sc has ncvarids by default; needs to check...
        ncvarids = self.ncvarids  # FIXME ****: brittle here
        if varids is None:
            # only normal traj (no sens traj)
            varids = ncvarids
            copy_traj_sc = False
        elif varids == ncvarids:
            copy_traj_sc = False
        else:
            copy_traj_sc = True
        
        if any([isinstance(varid, tuple) for varid in varids]):
            calc_sens = True
        else:
            calc_sens = False
        
        # sort times; return a list of sorted floats
        #times_sorted = trajectory.sort_times(times)
        if isinstance(times, tuple):
            times_sorted = tuple(sorted(times))
        else:
            times_sorted = sorted(times)
        
        # see if steady state needs to be calculated
        if times_sorted[-1] == np.inf:
            times_intgr = times_sorted[:-1]
            calc_ss = True
        else:
            times_intgr = copy.copy(times_sorted)
            calc_ss = False
        
        ## integrate to get traj_int
        # make an empty traj_int
        if times_intgr == []:
            traj_intgr = trajectory.Trajectory(varids=varids)
        elif times_intgr == [0.0]:
            data = [[self.evaluate_expr(varid, time=0) for varid in varids]]
            traj_intgr = trajectory.Trajectory(data=data, times=[0], varids=varids)
        else:
            # see if there are time intervals
            if 'fill_traj' not in kwargs:
                if isinstance(times, tuple) or any([isinstance(t, tuple) for t in times]):  
                    fill_traj = True
                else:
                    fill_traj = False
                kwargs['fill_traj'] = fill_traj    
        
            # fix time: when supplying a time not starting from 0, say, [1,2], 
            # SloppyCell starts from the current state of the network, 
            # even if the current state does not correspond to t=1.    
            # http://sourceforge.net/p/sloppycell/mailman/message/31806741/
            times_intgr = list(times_intgr)
            
            if float(times_intgr[0]) != 0.0:
                times_intgr.insert(0, 0)
            #net.x = net.x0
            
            if calc_sens:
                integrate = Dynamics.integrate_sensitivity
            else:
                integrate = Dynamics.integrate
            
            try:
                kwargs_intgr = butil.get_submapping(kwargs, f_key=lambda k: k in\
                    ['rtol', 'atol', 'fill_traj', 'return_events', 
                     'return_derivs', 'redirect_msgs', 'calculate_ic', 
                     'include_extra_event_info', 'use_constraints'])
                traj_intgr_sc = integrate(net, times=times_intgr, **kwargs_intgr)
            except daeintException:
                # It seems SloppyCell automatically sets net.x = net.x0, so:
                net.t = 0
                # rethrow the exception: 
                # http://nedbatchelder.com/blog/200711/rethrowing_exceptions_in_python.html
                raise
            
            # indexing creates a copy and slows things down
            if copy_traj_sc:      
                traj_intgr = trajectory.Trajectory(traj_intgr_sc)[varids]
            else:
                traj_intgr = trajectory.Trajectory(traj_intgr_sc)
                
        ## perform MCA to get traj_ss
        if calc_ss:
            #f = lambda vid: vid.replace('v_','J_') if isinstance(vid, str) else vid 
            #varids_ss = map(f, varids)
            varssvals = net.get_ssvals(varids=varids, **kwargs)
            traj_ss = trajectory.Trajectory(data=[varssvals.tolist()], 
                                            times=[np.inf], varids=varids)
        else:
            traj_ss = trajectory.Trajectory(varids=varids)
        
        traj = traj_intgr + traj_ss
        
        net.t = times_sorted[-1]
        
        # comment out the following line because if fill_traj is True then
        # we want all the times...
        
        #traj = traj_all.get_subset(times=times_sorted)  
        return traj
    
    """
    structure (parameter-independent)
    """
    
    def reorder_xids(self):
        return structure.reorder_xids(self)
 
   
    def get_stoich_mat(self, **kwargs):
        """
        """
        return structure.get_stoich_mat(self, **kwargs)
    get_stoich_mat.__doc__ += structure.get_stoich_mat.__doc__
                                     
    
    def get_reduced_stoich_mat(self, **kwargs):
        return structure.get_reduced_stoich_mat(self, **kwargs)
    
    
    def get_reduced_link_mat(self):
        return structure.get_reduced_link_mat(self)
    
    
    def get_link_mat(self):
        return structure.get_link_mat(self)
    
    
    def get_pool_mul_mat(self):
        return structure.get_pool_mul_mat(self)
    
    
    def get_ss_flux_mat(self):
        return structure.get_ss_flux_mat(self)
    
    
    def get_ddynvarids(self):
        return structure.get_ddynvarids(self)

    
    def get_idynvarids(self):
        return structure.get_idynvarids(self)                                     
    
    
    # shorthands
    @property
    def N(self):
        return self.get_stoich_mat()
    
    @property
    def Nr(self):
        return self.get_reduced_stoich_mat()
    
    @property
    def L0(self):
        return self.get_reduced_link_mat()
    
    @property
    def L(self):
        return self.get_link_mat()
    
    @property
    def P(self):
        return self.get_pool_mul_mat()
    
    @property
    def K(self):
        return self.get_ss_flux_mat()
    
    @property
    def ddynvarids(self):
        return self.get_ddynvarids()
    dxids = ddynvarids

    @property
    def idynvarids(self):
        return self.get_idynvarids()
    ixids = idynvarids
    
###############################################################################

    #def get_x(self, t):
    #    traj = self.get_traj([0,t], copy_net=True)
    #    return traj.loc[t, self.dynvarids]
    
    
    def get_dxdt(self, x=None, to_ser=False):
        """
        Velocities: velocities of species dynamics, dx/dt. 
        (Another way of getting it is N*v.)  
        
        Input:
            t: time
        """
        if x is None:
            x = [var.value for var in self.dynamicVars]
        
        if not hasattr(self, 'res_function'):
            self.compile()
        
        # SloppyCell Network doesn't seem to update self.constantVarValues
        # >>> net.set_var_val('p1', 1)
        # >>> print net.constantVarValues
        # >>> net.set_var_val('p1', 100)
        # >>> print net.constantVarValues
        dxdt = self.res_function(self.t, x, np.zeros(len(x)), self.constantVarValues)
        if to_ser:
            dxdt = Series(dxdt, index=self.xids)
        return dxdt
    
    
    def is_ss(self, tol=1e-6):
        return mca.is_ss(self, tol=tol)
    
    
    #def set_ss(self, tol=1e-6, method='integration', T0=1e2, Tmax=1e6):
    #    return mca.set_ss(self, tol=tol, method=method, T0=T0, Tmax=Tmax)
    def set_ss(self, **kwargs):
        return mca.set_ss(self, **kwargs)
    
            
###############################################################################
    """
    def get_ssvals_type(self, vartype, **kwargs_ss):
        self.set_ss(**kwargs_ss)
        if vartype == 'concn':
            return self.x
        elif vartype == 'flux':
            return self.v.rename(dict(zip(self.rateids, self.fluxids)))
        elif vartype == 'param_elas':
            return self.get_param_elas_mat().to_series()
        elif vartype == 'concn_elas':
            return self.get_concn_elas_mat().to_series()
        elif vartype == 'concn_ctrl':
            return self.get_concn_ctrl_mat().to_series()
        elif vartype == 'flux_ctrl':
            return self.get_flux_ctrl_mat().to_series()
        elif vartype == 'concn_resp':
            return self.get_concn_resp_mat().to_series()
        elif vartype == 'flux_resp':
            return self.get_flux_resp_mat().to_series()
        else:
            raise ValueError("Unrecognized variable type: %s"%vartype)
    """
    
    
    def get_s(self, *args, **kwargs):
        """Provide mca.get_s below for convenience. 
        """
        return mca.get_s(self, *args, **kwargs)
    get_s.__doc__ += mca.get_s.__doc__
    
    
    def get_s_integration(self, *args, **kwargs):
        """Provide mca.get_s_integration below for convenience. 
        """
        return mca.get_s_integration(self, *args, **kwargs)
    get_s_integration.__doc__ += mca.get_s_integration.__doc__
    
    
    def get_s_rootfinding(self, *args, **kwargs):
        """Provide mca.get_s_rootfinding below for convenience. 
        """
        return mca.get_s_rootfinding(self, *args, **kwargs)
    get_s_rootfinding.__doc__ += mca.get_s_rootfinding.__doc__
     
    
    def get_J(self, *args, **kwargs):
        """Signature the same as mca.get_s, provided below for convenience.
        """
        mca.set_ss(self, *args, **kwargs)
        return self.v.rename(OD(zip(self.vids, self.Jids)))
    get_J.__doc__ += mca.get_s.__doc__
    
    
    @property
    def s(self):
        return self.get_s(to_ser=True)
    
    
    @property
    def J(self):
        return self.get_J(to_ser=True)
    
    
    # steady-state
    def get_ssvals(self, varids=None, **kwargs_ss):
        """
        """
        if varids is None:
            varids = self.ncvarids
        self.set_ss(**kwargs_ss)
        
        def _calc_n_update(vartype, vartypes, varid2val):
            if vartype in vartypes:
                pass
            else:
                varid2val.update(getattr(self, vartype).to_series().to_dict())
                vartypes.append(vartype)
        
        varid2val = self.varvals.append(self.J).to_dict()
        vartypes = []
                
        for varid in varids:
            if isinstance(varid, tuple):
                if varid[0] in self.vids and varid[1] in self.xids:
                    _calc_n_update('Es', vartypes, varid2val)
                elif varid[0] in self.vids and varid[1] in self.pids:
                    _calc_n_update('Ep', vartypes, varid2val)
                elif varid[0] in self.xids and varid[1] in self.vids:
                    _calc_n_update('Cs', vartypes, varid2val)
                elif varid[0] in self.Jids and varid[1] in self.vids:
                    _calc_n_update('CJ', vartypes, varid2val)
                elif varid[0] in self.xids and varid[1] in self.pids:
                    _calc_n_update('Rs', vartypes, varid2val)
                elif varid[0] in self.Jids and varid[1] in self.pids:
                    _calc_n_update('RJ', vartypes, varid2val)
                elif varid[0] in self.logvids and varid[1] in self.logxids:
                    _calc_n_update('nEs', vartypes, varid2val)
                elif varid[0] in self.logvids and varid[1] in self.logpids:
                    _calc_n_update('nEp', vartypes, varid2val)
                elif varid[0] in self.logxids and varid[1] in self.logvids:
                    _calc_n_update('nCs', vartypes, varid2val)
                elif varid[0] in self.logJids and varid[1] in self.logvids:
                    _calc_n_update('nCJ', vartypes, varid2val)
                elif varid[0] in self.logxids and varid[1] in self.logpids:
                    _calc_n_update('nRs', vartypes, varid2val)
                elif varid[0] in self.logJids and varid[1] in self.logpids:
                    _calc_n_update('nRJ', vartypes, varid2val)
                else:
                    raise ValueError("Unrecognized value of varid: %s"%str(varid))
                
        ssvals = Series(varid2val).loc[varids]
        
        if ssvals.isnull().any():
            raise ValueError("ssvals has nan:\n%s"%str(ssvals))
        
        return ssvals
    
    
    def get_Ep_str(self):
        return mca.get_Ep_str(self)
    
    
    def get_Ex_str(self):
        return mca.get_Ex_str(self)
    
    
    def get_concn_elas_mat(self, **kwargs):
        """
        """
        return mca.get_concn_elas_mat(self, **kwargs) 
    
    
    def get_param_elas_mat(self, **kwargs):
        """
        """
        return mca.get_param_elas_mat(self, **kwargs)
    
    
    def get_jac_mat(self, **kwargs):
        return mca.get_jac_mat(self, **kwargs)
    
    
    def get_concn_ctrl_mat(self, **kwargs):
        return mca.get_concn_ctrl_mat(self, **kwargs) 
    
            
    def get_flux_ctrl_mat(self, **kwargs):
        return mca.get_flux_ctrl_mat(self, **kwargs)


    def get_concn_resp_mat(self, **kwargs):
        return mca.get_concn_resp_mat(self, **kwargs) 
    
    
    def get_flux_resp_mat(self, **kwargs):
        return mca.get_flux_resp_mat(self, **kwargs)     

    
    # unnormalized
    @property
    def Es(self):
        return self.get_concn_elas_mat(normed=False)
    
    @property
    def Ep(self):
        return self.get_param_elas_mat(normed=False)
    
    @property
    def M(self):
        return self.get_jac_mat()
    
    @property
    def Cs(self):
        return self.get_concn_ctrl_mat(normed=False)
    
    @property
    def CJ(self):
        return self.get_flux_ctrl_mat(normed=False)

    @property
    def Rs(self):
        return self.get_concn_resp_mat(normed=False)
    
    @property
    def RJ(self):
        return self.get_flux_resp_mat(normed=False)

    # normalized
    @property
    def nEs(self):
        return self.get_concn_elas_mat(normed=True)
    
    @property
    def nEp(self):
        return self.get_param_elas_mat(normed=True)
    
    @property
    def nCs(self):
        return self.get_concn_ctrl_mat(normed=True)
    
    @property
    def nCJ(self):
        return self.get_flux_ctrl_mat(normed=True)

    @property
    def nRs(self):
        return self.get_concn_resp_mat(normed=True)
    
    @property
    def nRJ(self):
        return self.get_flux_resp_mat(normed=True)
    
    @property
    def Exids(self):
        return list(itertools.product(self.vids, self.xids))
    
    @property
    def Epids(self):
        return list(itertools.product(self.vids, self.pids))
    
    @property
    def Csids(self):
        return list(itertools.product(self.xids, self.vids))
    
    @property
    def CJids(self):
        return list(itertools.product(self.Jids, self.vids))
    
    @property
    def Rsids(self):
        return list(itertools.product(self.xids, self.pids))
    
    @property
    def RJids(self):
        return list(itertools.product(self.Jids, self.pids))
    
    @property
    def nCsids(self):
        return list(itertools.product(self.logxids, self.logvids))
    
    @property
    def nCJids(self):
        return list(itertools.product(self.logJids, self.logvids))
###############################################################################
    
    @staticmethod
    def from_sbml(filepath, **kwargs):
        # note that reaction _names_ are not extracted properly (FIXME *)
        # optimizability attributes are not kept (FIXME *)
        net = IO.from_SBML_file(filepath, **kwargs)
        return Network(net=net)
    
        
    def to_sbml(self, filepath):
        # setting the following two attributes to None because otherwise 
        # _modifiers_ with stoichcoefs of 0 are not recognized 
        # when exporting the net to SBML, which would cause problems
        # when using the exported SBML in platforms such as Copasi or JWS
        for rxn in self.reactions:
            rxn.reactant_stoichiometry = None
            rxn.product_stoichiometry = None
        IO.to_SBML_file(self, filepath)
        
        
    def to_tex(self, filepath='', landscape=False):
        """
        """
        if butil.check_filepath(filepath):
            IO.eqns_TeX_file(self, filename=filepath, simpleTeX=False, 
                             landscape=landscape)
    
    
    def update(self, p=None, t=None, x=None, t_x=None, **varmap):
        """
        Update the state of network. 
        
        Input:
            p: parameter
            t: time 
            x: 
            t_x: 
            varmap: kwargs for individual variable values, eg, Vf_R1=2  (FIXME *: bad design?)
        """
        if p is not None:
            self.update_optimizable_vars(p)
            
        if varmap:
            for varid, varval in varmap.items():
                self.set_var_ic(varid, varval)  # FIXME **: this suffices?
                
        if t is not None:
            if np.isclose(t, 0):
                self.t = 0
                for idx, dynvar in enumerate(self.dynvars):
                    dynvar.value = self.x0[idx]  ## FIXME **
            if not np.isclose(self.t, t):
                self.get_traj(times=[t])  # FIXME **: this suffices?
         
        if x is not None:
            assert t_x is not None, "t_x has to be provided."
            self.t = t_x
            self.updateVariablesFromDynamicVars(x, t_x)
        
    
    def reorder_species(self, spids2):
        """
        Return a new network with the order of species given in spids2, 
        which only needs to be partially overlapping with the existing spids.
        """
        net2 = self.copy()
        for spid in reversed(spids2):
            if spid in net2.varids:
                sp = net2.vars[spid]
                net2.variables.del_by_key(spid)
                net2.variables.insert_item(0, spid, sp)
        net2._makeCrossReferences()
        return net2

    
    def reorder_reactions(self, rxnids2):
        """
        Return a new network with the order of reactions given in rxnids2, 
        which only needs to be partially overlapping with the existing rxnids.
        """
        net2 = self.copy()
        for rxnid in reversed(rxnids2):
            if rxnid in net2.rxnids:
                rxn = net2.rxns[rxnid]
                net2.reactions.del_by_key(rxnid)
                net2.reactions.insert_item(0, rxnid, rxn)
        net2._makeCrossReferences()
        return net2
    
    
    def reorder_parameters(self, pids2):
        """
        """
        net2 = self.copy()
        for pid in reversed(pids2):
            if pid in net2.varids:
                param = net2.vars[pid]
                net2.variables.del_by_key(pid)
                net2.variables.insert_item(0, pid, param)
        net2._makeCrossReferences()
        # needed to recompile 
        # because SloppyCell's 'structure' does not concern the order, but it
        # matters (took me an hour to find this bug...)
        net2._last_structure = None  
        net2.compile()
        return net2
    
    
    def remove_unused_vars(self):
        """
        """
        for varid in self.varids:
            if varid not in self.xids and varid not in self.asgvarids: 
                uses_varid = self.get_uses(varid)
                if uses_varid['rxn'] == OD() and uses_varid['asgrule'] == OD()\
                    and varid not in self.compartments.keys():
                    self.del_varid(varid)
    
    
    def reduce(self, id=None, **limits):
        """
        Input:
            limits: in the form of pid=limiting_value; 
                eg, k1=np.inf, k2=0
            
        """
        def take_limit(expr, limits):
            for pid, pval_limit in limits.items():
                expr = str(sympy.limit(expr, pid, pval_limit))
            return expr
            
        net = self.copy()
        if id is None:
            net.id = self.id + '_reduced'
        
        for rxn in net.rxns:
            rxn.kineticLaw = take_limit(rxn.kineticLaw, limits)
        for asgvarid, asgrule in net.asgrules.items():
            net.assignmentRules.set(asgvarid, take_limit(asgrule, limits))
        
        net.remove_unused_vars()
        net.compile()
        return net
        

    def draw_nx(self, pos=None, jsonpath=None, figsize=None, arrows=True, show=True, filepath=''):
        """
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        if pos is None:
            pos = _json2pos(jsonpath)

        spids, rxnids = self.spids, self.rxnids
        edges = butil.flatten([[(spid, rxn.id) for spid, stoichcoef
                                in rxn.stoichiometry.items() if stoichcoef<0]+\
                               [(rxn.id, spid) for spid, stoichcoef
                                in rxn.stoichiometry.items() if stoichcoef>0] 
                               for rxn in self.rxns], D=1)

        G = nx.DiGraph()
        G.add_nodes_from(spids, bipartite='sp')
        G.add_nodes_from(rxnids, bipartite='rxn')
        G.add_edges_from(edges)
        
        node2color = OD(zip(spids, ['b']*len(spids))+zip(rxnids, ['g']*len(rxnids)))
        ncolors = butil.get_values(node2color, G.nodes())
        ecolors = ['r']*len(edges)
        nlabels = OD(zip(G.nodes(), G.nodes()))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        nx.draw_networkx(G, pos, ax=ax, node_color=ncolors, labels=nlabels)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ecolors, width=3, arrows=arrows)
        plt.savefig(filepath)
        if show:
            plt.show()
        plt.close() 
    

    def draw_pgv(self, jsonpath=None, pos=None, rxnlabels=None,
                 arrowsize=0.5,
                 shape_sp='ellipse', shape_rxn='box',
                 spid2rgba=(0,255,0,100), rxnid2rgba=(255,0,0,100),
                 labelfontsize_sp=8, labelfontsize_rxn=8,
                 spid2shapescale=0.2, rxnid2shapescale=0.2,
                 insert_images=False, imagepath='', filepath=''):
        """
        Doc of node and edges attributes in graphviz:
            http://www.graphviz.org/content/attrs
            
        Input:
            rxnlabels: if given, a mapping from rxnid to a str (to be appended
                after rxnid in the plot)
        """
        import pygraphviz as pgv
        
        if jsonpath is not None:
            pos = _json2pos(jsonpath)
        else:
            assert pos is not None, "either jsonpath or pos has to be provided" 
            
        nodeids = pos.keys()
        
        edges_rxn = []
        for rxnid, rxn in self.rxns.items():
            for spid, stoichcoef in rxn.stoichiometry.items():
                nodeids_sp = [nodeid for nodeid in nodeids if 
                              spid == nodeid.split('_')[0] and
                              rxnid in nodeid.split('_')[1:]]
                if len(nodeids_sp) == 1:
                    nodeid = nodeids_sp[0]
                elif len(nodeids_sp) > 1:
                    raise ValueError('more than one nodeid matches spid: %s'%\
                                     str(nodeids_sp))
                else:
                    nodeid = spid 
                if stoichcoef < 0:  # substrate
                    edges_rxn.append((nodeid, rxnid))
                if stoichcoef > 0:  # product
                    edges_rxn.append((rxnid, nodeid))
                
        ## some preprossessing...
        spids, rxnids = self.spids, self.rxnids
        if not isinstance(spid2rgba, Mapping):
            spid2rgba = OD.fromkeys(spids, spid2rgba)
        if not isinstance(rxnid2rgba, Mapping):
            rxnid2rgba = OD.fromkeys(rxnids, rxnid2rgba)
        if not isinstance(spid2shapescale, Mapping):
            spid2shapescale = OD.fromkeys(spids, spid2shapescale)
        if not isinstance(rxnid2shapescale, Mapping):
            rxnid2shapescale = OD.fromkeys(rxnids, rxnid2shapescale)
        
        G = pgv.AGraph(strict=False, directed=True)
        
        if insert_images:
            figids = map(lambda rxnid:'fig'+rxnid, rxnids)
            edges_fig = zip(rxnids, figids)
        
            G.add_nodes_from(nodeids+figids)
            G.add_edges_from(edges_rxn, arrowsize=arrowsize)
            G.add_edges_from(edges_fig, arrowsize=0, style='dotted')
        else:
            G.add_nodes_from(nodeids+rxnids)
            G.add_edges_from(edges_rxn, arrowsize=arrowsize)
        #G.graph_attr = {'label':figtitle, 'labelfontsize':figtitlefontsize}
        
        for nodeid, nodepos in pos.items():
            if nodeid.split('_')[0] in spids:
                spnode = G.get_node(nodeid)
                spnode.attr['pos'] = '%f, %f'%tuple(nodepos)
                spnode.attr['label'] = nodeid.split('_')[0]
                spnode.attr['shape'] = shape_sp
                #node_spid.attr['fillcolor'] = 'green'
                spnode.attr['fillcolor'] = '#%02x%02x%02x%02x' % spid2rgba[spid]
                spnode.attr['style'] = 'filled'
                #node_spid.attr['size'] = 2.
                spnode.attr['fontsize'] = labelfontsize_sp
                spnode.attr['width'] = spid2shapescale[spid]
                spnode.attr['height'] = spid2shapescale[spid]
            else:
                rxnnode = G.get_node(nodeid)
                rxnnode.attr['pos'] = '%f, %f'%tuple(nodepos)
                if rxnlabels is not None:
                    label = '%s (%s)' % (nodeid, rxnlabels[nodeid])
                else:
                    label = nodeid
                rxnnode.attr['label'] = label
                rxnnode.attr['shape'] = shape_rxn
                rxnnode.attr['fillcolor'] = '#%02x%02x%02x%02x' % rxnid2rgba[rxnid]
                rxnnode.attr['style'] = 'filled'
                rxnnode.attr['fontsize'] = labelfontsize_rxn
                rxnnode.attr['width'] = rxnid2shapescale[rxnid]
                rxnnode.attr['height'] = rxnid2shapescale[rxnid]
        
        if insert_images:
            for rxnid in rxnids:
                figid = 'fig' + rxnid
                node_fig = G.get_node(figid)
                node_fig.attr['pos'] = '%f, %f'%tuple(pos[figid])
                node_fig.attr['shape'] = 'box'
                node_fig.attr['label'] = ''
                node_fig.attr['fillcolor'] = ''
                node_fig.attr['style'] = 'filled'
                node_fig.attr['imagepath'] = imagepath
                node_fig.attr['image'] = 'hist.pdf'  
            
        G.draw(filepath, prog='neato', args='-n2')
        
        
    def to_tex_polynomials(self, varids_r, d_tex=None, filepath='', landscape=True, margin=2):
        """
        Input:
            d_latex:
        """
        #def _replace(expr, d):
        #    for s, s2 in d.items():
        #        expr = expr.replace(s, s2)
        #    return expr
        
        
        _repl = exprmanip.sub_for_vars
        _raisepower = lambda tu: tu[0] ** tu[1]
        
        def _2tex_pid(pid):
            if pid.startswith('Vf_') or pid.startswith('Vb_'):
                pid = '%s^{%s}' % tuple(pid.split('_'))
            if pid.count('_') == 2:
                pid = '%s^{%s}_{%s}' % tuple(pid.split('_'))
            return pid
        d_tex = dict(zip(self.pids, [_2tex_pid(pid) for pid in self.pids]) +\
                     d_tex.items())
        
        _2tex = lambda sympyexpr:\
            sympy.latex(sympyexpr, mode='plain', long_frac_ratio=10, mul_symbol='dot',
                        symbol_names=butil.chkeys(d_tex, lambda k: sympy.symbols(k))) 
        _rm1pt0 = lambda expr: re.sub('(?<![0-9])1.0\s*\\\cdot', '', expr)

        if len(varids_r) > 1:
            rids = ['r%d'%i for i in range(1, len(varids_r)+1)]
        else:
            rids = ['r']
        varids_r2 = ['*'.join(tu) for tu in zip(varids_r, rids)]
        d = OD(zip(varids_r, varids_r2))
        ratelaws = np.array([str(sympy.simplify(_repl(ratelaw, d))) for ratelaw in self.ratelaws])
        denoms = np.array([str(sympy.fraction(rl)[1]) for rl in ratelaws])
        
        X = sympy.symbols(self.xids)
        r = sympy.symbols(rids)
        
        polys = []
        for ixid, stoichcoefs in zip(self.ixids, self.Nr.values):
            idxs_nonzero = [idx for idx, stoichcoef in enumerate(stoichcoefs) 
                            if not np.isclose(stoichcoef, 0)]
            stoichcoefstrs_nonzero = [str(coef).rstrip('.0') for coef in stoichcoefs[idxs_nonzero]]
            str_ratelawsum = _rm1pt0('+'.join(['%s*%s'%tu for tu in zip(stoichcoefstrs_nonzero, ratelaws[idxs_nonzero])]))
            str_denomprod = '*'.join(['(%s)'%denom for denom in denoms[idxs_nonzero]])
            poly = sympy.Poly(sympy.simplify('(%s)*%s'%(str_ratelawsum, str_denomprod)), X)
            polys.append(poly)
            
        lines = []
        lines.append(r'\documentclass{article}') 
        lines.append(r'\usepackage{amsmath,fullpage,longtable,array,calc,mathastext,breqn,xcolor}') 
        if landscape == True:
            lines.append(r'\usepackage[a4paper,landscape,margin=1in]{geometry}')
        else:
            lines.append(r'\usepackage[a4paper,margin=%fin]{geometry}'%margin)
        lines.append(r'\begin{document}') 
       
        coefs_r = []
        yids = []
        for poly in polys:
            termstrs = []
            
            leadingcoef_r = sympy.Poly(poly.coeffs()[0], r).coeffs()[0]
            
            for monom_X, coef_X in poly.terms():
                coef_X = sympy.simplify(coef_X, ratio=1)
                poly_r = sympy.Poly(coef_X, r)
                
                coefs_r.extend([coef_r/leadingcoef_r for coef_r in poly_r.coeffs()])
                
                monom_X = sympy.prod(map(_raisepower, zip(X, monom_X)))
                monomstr_X = _2tex(monom_X)
                if monomstr_X == '1':
                    monomstr_X = ''
                monomstr_X = '\\textcolor{red}{%s}' % monomstr_X
                
                termstrs_r = []
                for monom_r, coef_r in poly_r.terms():
                    
                    coefstr_r = _rm1pt0(_2tex(coef_r))
                    if coef_r.is_Add:
                        coefstr_r = '\left('+ coefstr_r +'\\right)'
                    
                    monom_r = sympy.prod(map(_raisepower, zip(r, monom_r)))
                    monomstr_r = _2tex(monom_r)
                    if monomstr_r == '1':
                        monomstr_r = ''
                    monomstr_r = '\\textcolor{blue}{%s}' % monomstr_r
                        
                    termstrs_r.append(coefstr_r + '\t' + monomstr_r)
                    coefstr_X = '\\left(' + '+'.join(termstrs_r) + '\\right)'
                    
                    yids.append((ixid, str(monom_X), str(monom_r)))
                
                termstrs.append(coefstr_X + '\t' + monomstr_X)
        
            linestr = '\\begin{dmath} \n' + '+'.join(termstrs) + '=0\n\end{dmath} \n\n'
            lines.append(linestr.replace('+-', '-'))
        
        lines.append('\\end{document}') 
        
        if filepath:
            fh = file(filepath, 'w') 
            fh.write(os.linesep.join(lines)) 
            fh.close()
        
        coefs_r = [_rm1pt0(str(coef)) for coef in coefs_r]
        
        str_h = str(coefs_r).replace("'", "")
        str_Dh = str([[exprmanip.diff_expr(coef, pid) for pid in self.pids] 
                      for coef in coefs_r]).replace("'", "")
        
        def h(p):
            self.update(p=p)
            return np.array(eval(str_h, self.varvals.to_dict()))
        
        def Dh(p):
            self.update(p=p)
            return np.array(eval(str_Dh, self.varvals.to_dict()))
        
        coefs = predict.Predict(f=h, Df=Dh, pids=self.pids, p0=self.p0, 
                                yids=yids, funcform=coefs_r) 
        
        return coefs

        
        
    def get_polynomials(self, varid2rid=None, N=None):
        """
        FIXME ****: I don't need the r thing. Just use C...
        Input:
            varid2rid: optional; 
            N: different arrangements of N yield equivalent polynomials
                of different complexities;
                eg, N=net.Nr[[0,1,3,2]].rref()[[0,1,3,2]]
        """
        
        _repl = exprmanip.sub_for_vars
        #_raisepower = lambda tu: tu[0] ** tu[1]
        
        if varid2rid is None:
            ratelaws = self.ratelaws
            rids = None
        else:
            varid2rid = OD(varid2rid)
            varids_r, rids = varid2rid.keys(), varid2rid.values()
            varids_r2 = ['*'.join(tu) for tu in zip(varids_r, rids)]
            d = OD(zip(varids_r, varids_r2))
            ratelaws = np.array([str(sympy.simplify(_repl(ratelaw, d))) 
                                 for ratelaw in self.ratelaws])
            
        denoms = np.array([str(sympy.fraction(rl)[1]) for rl in ratelaws])
        
        polys = []
        if N is None:
            N = self.Nr.values
        else:
            N = np.array(N)
        for ixid, scoefs in zip(self.ixids, N):
            idxs_nonzero = [idx for idx, scoef in enumerate(scoefs) 
                            if not np.isclose(scoef, 0)]
            scoefstrs_nonzero = [str(scoef).rstrip('.0') for scoef in 
                                 scoefs[idxs_nonzero]]
            rlsumstr = '+'.join(['%s*%s'%tu for tu in 
                                 zip(scoefstrs_nonzero, ratelaws[idxs_nonzero])])
            denomprodstr = '*'.join(['(%s)'%denom for denom in denoms[idxs_nonzero]])
            polystr = sympy.simplify('(%s)*%s'%(rlsumstr, denomprodstr))
            poly = algebra.Polynomial.from_str(polystr, self.xids, 
                                               pids=self.pids, p0=self.p0, polyid=ixid, 
                                               subvarids=rids, convarvals=self.convarvals) 
            polys.append(poly)
        
        for row in self.P:
            pass
        
        polys = algebra.Polynomials(polys)
        return polys
    
    def get_polynomials2(self, varids=None, N=None):
        """
        FIXME ****: I don't need the r thing. Just use C...
        Input:
            varid2rid: optional; 
            N: different arrangements of N yield equivalent polynomials
                of different complexities;
                eg, N=net.Nr[[0,1,3,2]].rref()[[0,1,3,2]]
        """
        ratelaws = self.ratelaws
        denoms = np.array([str(sympy.fraction(rl)[1]) for rl in ratelaws])
        
        if hasattr(N, 'colvarids'):
            N = N.loc[:, self.rxnids].values
        elif N is None:
            N = self.Nr.values
        else:
            N = np.array(N)
            
        polys = []    
        for ixid, scoefs in zip(self.ixids, N):
            idxs_nonzero = [idx for idx, scoef in enumerate(scoefs) 
                            if not np.isclose(scoef, 0)]
            scoefstrs_nonzero = [str(scoef).rstrip('.0') for scoef in 
                                 scoefs[idxs_nonzero]]
            rlsumstr = '+'.join(['%s*%s'%tu for tu in 
                                 zip(scoefstrs_nonzero, ratelaws[idxs_nonzero])])
            denomprodstr = '*'.join(['(%s)'%denom for denom in denoms[idxs_nonzero]])
            polystr = sympy.simplify('(%s)*%s'%(rlsumstr, denomprodstr))
            poly = algebra.Polynomial.from_str(polystr, self.xids, 
                                               pids=self.pids, p0=self.p0, 
                                               polyid=ixid, subvarids=varids, 
                                               convarvals=self.convarvals) 
            polys.append(poly)
        
        for row in self.P:
            pass
        
        polys = algebra.Polynomials(polys)
        return polys

        
def _json2pos(filepath):
    """
    
    To make json file:
        1) Open Cytoscape, import sbml file
        2) Visualize the network, arrange the nodes
        3) One can color them according to "sbml type" (say, species red and
            reactions blue)
        4) Create new (species) nodes to declutter if needed; in this case, 
            all nodes corresponding to the same species have to be named in
            the spid_rxnid(1_rxnid2...) convention (eg, ATP_PGAK)
        5) Export it through "Export -> Network and View"
    
    pos: a mapping from nodeid to pos, where nodeid has a particular naming 
        scheme: if it is duplicated for visualization purpose, then it is 
        spid_rxnid (duplicate spid for the particular rxnid to avoid cluttering)
    """
    import json
    fh = open(filepath)
    out = json.load(fh)
    fh.close()
    pos = Series([])
    for node in out['elements']['nodes']:
        nodeid = node['data']['sbml_id']        
        pos_node = butil.get_values(node['position'], ['x','y'])
        pos_node[1] *= -1
        pos[nodeid] = pos_node
    return pos
    
    
def from_smod(filepath):
    """
    """
    def format(input):
        # remove whitespace
        # add a preceding underscore if the species id starts with a number
        if isinstance(input, str):
            return _format(input)
        if isinstance(input, list):
            return map(_format, input)
        
    fh = open(filepath)
    string = ''.join(fh.readlines())
    fh.close()
    
    mod, spp, rxns = filter(None, string.split('@'))
    
    # re: preceded by '=', followed by some whitespace and '(', nongreedy (?) 
    netid = re.search('(?<=\=).*?(?=\s*\()', mod).group()
    # re: preceded by '(', followed by ')'
    note = re.search('(?<=\().*(?=\))', mod).group()
    
    net = Network(netid)
    net.note = note
    net.add_compartment('Cell')
    
    for sp in filter(None, spp.split('\n'))[1:]:
        if '(' in sp:
            spid, concn = sp.split('(')[0].split(':')
            if '1' in sp.split('(')[1]:  # constancy flag: constant=1
                is_constant = True
            else:
                is_constant = False
        else:
            spid, concn = sp.split(':')
            is_constant = False
        spid = format(spid)
        net.add_species(spid, 'Cell', float(concn), is_constant=is_constant)
        
    for rxn in filter(None, rxns.split('\n'))[1:]:
        rxnid = format(rxn.split(':')[0])
        # re: get what comes after the first ':'
        eqn, rate, pstr = re.split('^.*?:', rxn)[1].split(';')
        p = Series()
        for _ in pstr.split(','):
            pid, pval = _.split('=')
            p[format(pid)] = float(pval)
        
        """
        ## modifiers
        
        activators = []
        inhibitors = []
        if '(' in eqn:
            # re: between '(' and ')'
            for modifiers in re.search('(?<=\().*(?=\))', eqn).group().split('|'):
                if modifiers.replace(' ', '')[0] == '+':
                    activators.extend(format(modifiers.split(':')[1].split(',')))
                if modifiers.replace(' ', '')[0] == '-':
                    inhibitors.extend(format(modifiers.split(':')[1].split(',')))
            eqn = eqn.split('(')[0]
        """
        
        if '<->' in eqn:
            rev = True
        else:
            rev = False
        
        ratelaw = rate.split('v=')[1]
        
        if ratelaw == 'SMM':
            net.add_reaction_mm_qe(rxnid, stoich_or_eqn=eqn, p=p, scheme='standard')
        
        if ratelaw == 'MA':
            net.add_reaction_ma(rxnid, stoich_or_eqn=eqn, p=p, reversible=rev)
    
    return net 

# move to ratelaw?  FIXME ***
def _get_substrates(stoich_or_eqn, multi=False):
    """
    Input: 
        stoich_or_eqn: a mapping, from species ids to stoich coefs which can be
                an int, a float, or a string; or a str
        multi: a bool; if True, return a multiset by repeating
            for stoichcoef times
    
    Output:
        a list of substrate ids
    """
    if isinstance(stoich_or_eqn, str):
        eqn = stoich_or_eqn
        stoich = _eqn2stoich(eqn)
    else:
        stoich = stoich_or_eqn
    subids = []
    for spid, stoichcoef in stoich.items():
        try:
            stoichcoef = int(float(stoichcoef))
        except ValueError:
            stoichcoef = int(stoichcoef.lstrip()[0]+'1')
        if stoichcoef < 0:
            if multi:
                subids.extend([spid]*(-stoichcoef))
            else:
                subids.append(spid)
    return subids
get_substrates = _get_substrates


# move to ratelaw?  FIXME ***
def _get_products(stoich_or_eqn, multi=False):
    """
    Input: 
        stoich_or_eqn: a mapping, from species ids to stoich coefs which can be
                an int, a float, or a string
        multi: a bool; if True, return a multiset by repeating
            for stoichcoef times
    
    Output:
        a list of product ids
    """
    if isinstance(stoich_or_eqn, str):
        eqn = stoich_or_eqn
        stoich = _eqn2stoich(eqn)
    else:
        stoich = stoich_or_eqn
    proids = []
    for spid, stoichcoef in stoich.items():
        try:
            #stoichcoef = int(float(stoichcoef))
            stoichcoef = float(stoichcoef)
        except ValueError:
            print "ValueError: stoichcoef = ", stoichcoef
        if stoichcoef > 0:
            if not np.isclose(stoichcoef, int(stoichcoef)):
                multi = False
            else:
                stoichcoef = int(stoichcoef)
                
            if multi:
                proids.extend([spid]*stoichcoef)
            else:
                proids.append(spid)
    return proids
get_products = _get_products


# move to ratelaw?  FIXME ***
def _get_reactants(stoich, multi=False):
    """
    """
    return _get_substrates(stoich, multi=multi) + _get_products(stoich, multi=multi)
get_reactants = _get_reactants


# move to ratelaw?  FIXME ***
def _eqn2stoich(eqn):
    """
    Convert reaction equation (a string) to stoichiometry (a dictionary).
    """
    def unpack(s):
        # an example of s: ' 2 ATP '
        l = filter(None, s.split(' '))
        if len(l) == 1:
            # sc: stoichcoef
            sc_unsigned, spid = '1', l[0]
        if len(l) == 2:
            sc_unsigned, spid = l
        """
        if np.isclose(sc_unsigned, int(sc_unsigned)):
            sc_unsigned = int(sc_unsigned)
        else:
            pass
        """
        sc_unsigned = eval(sc_unsigned)
        return spid, sc_unsigned
    
    # remove annotating species
    # eg, '(ST+)P->(ST+)G1P', where 'ST' (starch) is kept there to better
    # represent the chemistry
    eqn = re.sub('\(.*?\)', '', eqn)
    
    # re: '<?': 0 or 1 '<'; '[-|=]': '-' or '=' 
    subs, pros = re.split('<?[-|=]>', eqn)
    stoich = OD()
    
    if subs:
        for sub in subs.split('+'):
            subid, sc_unsigned = unpack(sub)
            stoich[subid] = -1 * sc_unsigned
    if pros:
        for pro in pros.split('+'):
            proid, sc_unsigned = unpack(pro)
            if proid in stoich:
                stoich[proid] = stoich[proid] + sc_unsigned
            else:
                stoich[proid] = sc_unsigned
        
    return stoich


def _format(varid):
    """
    """
    varid2 = varid.replace(' ', '')
    if varid[0] in '0123456789':
        varid2 = '_' + varid2
    return varid2


def cmp_p(nets, only_common=False, only_diff=False, eps=1e-3):
    """
    Input:
        nets: a list of nets
        only_common: if True, only outputs the intersection of parameter sets
        only_diff: if True, only outputs the different parameter values 
        
    Output:
        a df
    """
    
    # not using set.intersection to preserve the order
    pids_common = [pid for pid in nets[0].pids 
                   if all([pid in net.pids for net in nets])]
    ps = DF([net.p[pids_common] for net in nets], 
              index=[net.id for net in nets], columns=pids_common).T
    if not only_common:
        for net in nets:
            p_net = DF({net.id: net.p[~net.p.index.isin(pids_common)]})
            ps = pd.concat((ps, p_net))
            
    if only_diff:
        ps = ps.loc[ps.apply(lambda row: not row.std()/row.mean()<eps, axis=1)]
        
    return ps
    
    
def cmp_ratelaw(nets, only_common=False, filepath='', landscape=True,
                paperheight=19, paperwidth=12, margin=0.2,
                colwidth=9, colwidth2=8.5):
    """
    Input:
        nets:
        
    Output:
        
    """
    rateids_common = [vid for vid in nets[0].vids 
                      if all([vid in net.vids for net in nets])]
    ratelaws = DF([net.asgrules[rateids_common] for net in nets],
                        index=[net.id for net in nets], 
                        columns=rateids_common).T
    if not only_common:
        for net in nets:
            rateids_net = [_ for _ in net.vids if _ not in rateids_common]
            ratelaws_net = DF({net.id: net.asgrules[rateids_net]})
            ratelaws = pd.concat((ratelaws, ratelaws_net))
    # tex things up    
    if filepath:
        butil.check_filepath(filepath)
        
        xratelaws = ratelaws.applymap(lambda s: exprmanip.expr2TeX(s).replace('_','\_'))         
        
        lines = []
        lines.append(r'\documentclass{article}') 
        lines.append(r'\usepackage{amsmath,fullpage,longtable,array,calc,mathastext}') 
        if landscape == True:
            lines.append(r'\usepackage[a4paper,landscape,margin=1in]{geometry}')
        else:
            lines.append(r'\usepackage[a4paper,margin=1in]{geometry}')
        ## to be customized
        lines[-1] = r'\usepackage[paperheight=%.1fin,paperwidth=%.1fin,landscape,margin=%.1fin]{geometry}'%\
            (paperheight,paperwidth,margin)
        
        lines.append(r'\begin{document}') 
        lines.append(r'\begin{center}')
        ## to be customized
        ## \arraybackslash is often messed up
        lines.append(r'\begin{tabular}{|>{\centering\arraybackslash}m{0.7in}'+\
                      '|>{\centering\\arraybackslash}m{%.1fin}'%colwidth+\
                      '|>{\centering\\arraybackslash}m{%.1fin}'%colwidth2+\
                      '|'+'}')
        lines.append(r'\hline')
        # vertically align the last column
        # http://tex.stackexchange.com/questions/127050/vertically-align-text-in-table-in-latex
        lines.append(r'\textbf{Reaction} & '+\
                     ' & '.join(['\\textbf{\LARGE %s}'%netid.replace('_','\_') 
                                 for netid in ratelaws.columns])+\
                     ' \parbox{0pt}{\\rule{0pt}{5ex+\\baselineskip}} \\\\ \hline')
        for rateid, xratelaws_rxn in xratelaws.iterrows():
            ## to be customized
            lines.append(r'\textbf{%s} & $'%rateid.lstrip('v_')+\
                          ' $ & $ '.join(xratelaws_rxn).replace('\cdot','\,\cdot\,')+\
                          ' $ \parbox{0pt}{\\rule{0pt}{5ex+\\baselineskip}} \\\\ \hline')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{center}')
        lines.append(r'\end{document}') 

        fh = file(filepath, 'w') 
        fh.write(os.linesep.join(lines)) 
        fh.close()        
    
    return ratelaws


def extract_varids(expr):
    """
    """
    return set(exprmanip.extract_vars(expr))
    
    
def make_net(eqns, ratelaws, 
             facelift=True, scheme='num', parametrization='VK',
             rxnids=None, cids=None, r=False, add_ratevars=True, 
             netid='', compile=False, **kwargs):
    """
    FIXME **: need to be able to add inhibitors and activators...
    
    Input:
        eqns: a list of equation strings
        ratelaws: 
            - a list, of:
                * ratelaw.RateLaw instances, #(or ratelaws strings,) 
                * ratelaw string short codes ('ma'/'mm' + 'ke'/'');
            when it is standard ratelaw.RateLaw instances to be 
            facelifted, which comes in two schemes ('num' and 'rxnid')
            - a str: 'mm', 'ma', 'mmke', 'make'
        facelift:
        scheme: 'num' or 'rxnid'               
        rxnids:
        cids:
        r: 
        add_ratevars:
        netid:
        
        kwargs: varid, varval to set in net
    """
    
    ## get all the information ready
    
    # get rxnids
    if rxnids is None:
        rxnids = ['R%d'%i for i in range(1, len(eqns)+1)]
    
    # get ratelaws
    if isinstance(ratelaws, str):
        rltype = ratelaws
        ratelaws = []
        for eqn in eqns:
            rlid = '%s%d%d' % (rltype, 
                               len(get_substrates(eqn)),
                               len(get_products(eqn)))
            ratelaws.append(ratelaw.get_ratelaw(rlid, parametrization))
    else: # a list
        pass
        
    """
    if ratelaws in ['ma', 'mm', 'make', 'mmke', 'mai', 'mmi', 'qe']:
        rltype = ratelaws
        subs = _get_substrates(eqn, multi=True)
        pros = _get_products(eqn, multi=True)
        
        
        ratelaws = [_get_ratelaw(eqn, rltype) for eqn in eqns]
    
    def _eqn2rlstr(eqn, rltype):  
        subs = _get_substrates(eqn, multi=True)
        pros = _get_products(eqn, multi=True)
        if len(set(subs)) < len(subs) or len(set(pros)) < len(pros):
            # a makeshift solution below  FIXME **
            s = ''
            if len(subs) == 2:
                if len(set(subs)) == 1:
                    s += 'AA'
                else:
                    s += 'AB'
            if len(pros) == 2:
                if len(set(pros)) == 1:
                    s += 'PP'
                else:
                    s += 'PQ'
            return 'rl_%s%d%d_%s' % (rltype, len(subs), len(pros), s)
        else:
            return 'rl_%s%d%d' % (rltype, len(subs), len(pros))
    
    
        else:
            return ratelaw.RateLaw(s, xids=xids)
    """        
    
    net = Network(id=netid)
    
    # add species
    for eqn in eqns:
        spids = get_reactants(eqn)
        for spid in spids:
            if spid not in net.spids:
                net.add_species(spid, 1.)
    
    # add reactions and parameters
    for idx, (rxnid, eqn, rl) in enumerate(zip(rxnids, eqns, ratelaws)):
        if scheme == 'num':
            rxnidx = idx + 1
        if scheme == 'rxnid':
            rxnidx = rxnid
        
        if facelift:
            # slicing because of irreversible reactions 
            rl = rl.facelift(xids_new=get_reactants(eqn)[:len(rl.xids)],  
                             pcmap=scheme, rxnidx=rxnidx)
            
        if r:
            rid = 'r%d' % (idx+1)
            s = '%s*(%s)' % (rid, rl.s)
            net.add_parameter(rid, is_optimizable=False)
        else:
            s = rl.s

        if 'inf' not in s:
            net.add_reaction(rxnid, stoich_or_eqn=eqn, ratelaw=s, 
                             p=OD.fromkeys(rl.pids, 1))
        else:
            net.add_algebraic_rule(rl.s.replace('inf','1'))
            for xid in rl.xids:
                net.variables.get(xid).is_boundary_condition = True  
        
    
    if cids is not None:    
        for cid in cids:
            if cid not in net.varids:
                net.add_parameter(cid)
            net.set_var_constant(cid, True)
            net.set_var_optimizable(cid, False)
            
    net.update(**kwargs)
    
    if add_ratevars:
        net.add_ratevars()
    
    if compile:
        net.compile()
    
    return net


def make_path(ratelaws, **kwargs):
    """Make a linear pathway: C1 <-> X1 <-> ... <-> C2. 
    
    Docstring of make_net is attached for convenience. 
    """
    m = len(ratelaws)  # number of reactions
    if 'eqns' not in kwargs:
        if m == 2:
            xids = ['X']
        else:
            xids = ['X%d'%i for i in range(1, m)]
        xids = ['C1'] + xids + ['C2']
        tus = zip(xids[:-1], xids[1:])
        eqns = ['<->'.join(tu) for tu in tus]
        #print eqns
        #eqns = ['C1<->X', 'X<->C2']
    #if 'netid' not in kwargs:
    #    kwargs['netid'] = 'path%d'%len(ratelaws)
    return make_net(eqns, ratelaws, **kwargs)
make_path.__doc__ += make_net.__doc__

#from ratelaw import v_mmke11, v_mmke22, v_mmke21


"""
net = make_net(eqns=['X1+C1<->2 X2', 'X2+X3<->X1+X4', 'X2<->C3', 'X4+C2<->X3'],
               #ratelaws=[v_mmke22, v_mmke22, v_mmke11, v_mmke21],
               ratelaws=['mmke']*4,
               cids=['KE1','KE2','KE3','KE4','C3'], scheme_pid='num',
               )

net2 = make_net(eqns=['RuBP+CO2<->2 PGA', 'PGA+ATP<->BPGA+ADP', 'BPGA<->GAP',
                      '5 GAP<->3 Ru5P', 'Ru5P+ATP<->RuBP+ADP', 'PGA<->PGAc'], 
                rxnids=['RBCO', 'PGAK', 'GAPDH', 'REGN', 'RK', 'PGAT'],
                #ratelaws=[v_mmke22, v_mmke22, v_mmke11, v_mmke11, v_mmke22, v_mmke11],
                ratelaws=['mmke']*6,
                cids=['CO2', 'PGAc'], scheme_pid='rxnid')

net3 = make_path(ratelaws=['ma','mm'])
"""