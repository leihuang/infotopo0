"""

FIXME ***: 
try to convert functions in modules into methods by binding the functions to the class? 
modules: structure and mca...

"""

from __future__ import division
from collections import OrderedDict as OD, Mapping
import copy
import re
import itertools
import os
import logging

import numpy as np
import pandas as pd

from SloppyCell.ReactionNetworks import Network as Network0, Dynamics, IO, KeyedList
from SloppyCell import ExprManip as exprmanip

# FIXME ****
from util import butil
reload(butil)
#from butil import Series, DF
#from util.butil import Series, DF

import predict
reload(predict)

import trajectory, structure as struct, mca  # FIXME **: struct is a module in the standard library
reload(trajectory)
reload(struct)
reload(mca)



class Network(Network0):
    """
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
    
    
    def set_p(self, p):
        self.update_optimizable_vars(p[self.pids])
    
    
    @property
    def vars(self):
        return butil.Series(OD(self.variables.items()))


    @property
    def p(self):
        return butil.Series(OD([(var.id, var.value) for var 
                            in self.optimizableVars]))
        
    
    @property
    def dynvars(self):
        return butil.Series(OD(self.dynamicVars.items()))
    
    
    @property
    def asgvars(self):
        return butil.Series(OD(self.assignedVars.items()))
    
    
    @property
    def rxns(self):
        return butil.Series(OD(self.reactions.items()))
    

    @property
    def asgrules(self):
        return butil.Series(OD(self.assignmentRules.items()))
    
    
    @property
    def algrules(self):
        return butil.Series(OD(self.algebraicRules.items()))


    @property
    def raterules(self):
        return butil.Series(OD(self.rateRules.items()))
    
    
    """
    funcdefs = function_definitions
    asgvars = assigned_vars
    algvars = algebraic_vars
    convars = constant_vars
    dynvars = dynamic_vars
    optvars = optimizable_vars
    """
    
    # vids? dynvids, asgvids, ratevids, algvids?
    @property
    def varids(self):
        return self.variables.keys()
    
    @property
    def pids(self):
        return self.optimizableVars.keys()
    
    @property
    def dynvarids(self):
        return self.dynamicVars.keys() 
    
    @property
    def asgvarids(self):
        return self.assignedVars.keys() 
    
    @property
    def ncvarids(self):  # non-constant variables
        return self.dynvarids + self.asgvarids
    
    @property
    def constvarids(self):
        return self.constantVars.keys()
    
    @property
    def spids(self):
        return self.species.keys()
    
    @property
    def rxnids(self):
        return self.reactions.keys()
    
    
    @property
    def rateids(self):
        return map(lambda _: 'v_'+_, self.rxnids)
    
    
    @property
    def fluxids(self):
        return map(lambda _: 'J_'+_, self.rxnids)
    
    
    @property
    def ratelaws(self):
        return butil.Series(OD([(rxn.id, rxn.kineticLaw) for rxn in self.rxns]))
    
    
    @property
    def varvals(self):
        return butil.Series(OD([(var.id, var.value) for var in self.variables]))
    
    @property
    def dynvarvals(self):
        return butil.Series(OD([(var.id, var.value) for var in self.dynamicVars]))
    x = dynvarvals

    @property
    def dynvarvals_init(self):
        return butil.Series(OD([(var.id, var.initialValue) for var in self.dynamicVars]))
    x0 = dynvarvals_init
    

    def set_initial_condition(self, x0):
        pass
    set_x0 = set_initial_condition
    
    
    # FIXME ***:
    # call it theta? theta: parameters; p: optimizable parameters??
    # when cmp, we should compare theta. 
    @property
    def constvarvals(self):
        return butil.Series(OD([(var.id, var.value) for var in self.constantVars]))
    
    
    @property
    def rates(self):
        return butil.Series([self.evaluate_expr(rateid) for rateid in self.rateids],
                         index=self.rateids)
    v = rates
    
    
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
        """
        Regularize the network so that all parameters have ranges between 0
        and infinity.
        """
        pass
        
    
    def add_reaction_ma(self, rxnid, stoich_or_eqn, p, reversible=True, 
                        haldane='kf', add_thermo=False, T=25):
        """
        Add a reaction assuming mass action kinetics.
        
        Input:
            rxnid: a str; id of the reaction
            stoich_or_eqn: a mapping (stoich, eg, {'S':-1, 'P':1}) or 
                a str (eqn, eg, 'S<->P'); if an eqn is provided, 
                bidirectional arrow ('<->') denotes reversible reaction and 
                unidirectional arrow ('->') denotes irreversible reaction
            p: a dict; mapping from pid to pinfo, where pinfo can be 
                a float (pval) or a tuple (pval, is_optimizable); 
                eg, p = {'kf':1, 'kb':2, 'KE':(1, False)}
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


    def add_reaction_qe(self, rxnid, stoich_or_eqn, KE):
        """
        Add a reaction that is assumed to be at quasi-equilibrium (qe). 
        """
        self.add_reaction(rxnid=rxnid, stoich_or_eqn=stoich_or_eqn, 
                          p={'KE_'+rxnid:(KE,False)}, ratelaw='0')
        # add algebraic rules
        
    
    def add_reaction_mm_qe(self, rxnid, stoich_or_eqn, 
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
        
            
    def add_reaction(self, rxnid, stoich_or_eqn, ratelaw, p=None, **kwargs):
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
        #import ipdb
        #ipdb.set_trace()
        net = self.copy()
        for rxn in net.reactions:
            rateid = 'v_' + rxn.id
            try:
                net.add_parameter(rateid, is_constant=False, is_optimizable=False)
                net.add_assignment_rule(rateid, rxn.kineticLaw)
            except ValueError:
                pass
        return net
    
    
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

    
    def standardize(self):
        """
        rxn.ratelaw? (KineticLaw in SBML)
        pd.Series replacing KeyedList?
        Name conventions such as "rateRules"...
        
        Very low priority...
        """
        net = self.copy()
        for rxn in net.reactions:
            if hasattr(rxn, 'kineticLaw'):
                rxn.ratelaw = rxn.kineticLaw
        return net
    
    
    def get_uses(self, varid):
        """
        """
        uses = OD()
        uses_rxn = OD([])
        for rxn in self.reactions:
            if varid in exprmanip.extract_vars(rxn.kineticLaw):
                uses_rxn[rxn.id] = rxn.kineticLaw
        uses['rxn'] = uses_rxn
        uses_asgrule = OD()
        for asgvarid, asgrule in self.asgrules.iteritems():
            if varid in exprmanip.extract_vars(asgrule):
                uses_rxn[asgvarid] = asgrule
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
                reaction ratelaws, assignment rules, etc.
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
            if pid.startswith('KE_'):
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
    
    
    def perturb(self, condition):
        """
        """
        if condition == ():
            net = self.copy() 
        else:
            if len(condition) == 2:
                pid, mode = condition[0], '*'  # default
            else:
                pid, mode = condition[:2]
            
            if mode in ['*', '/', '+', '-']:
                change = condition[-1]
                #pid2 = pid + '_new'
                #net.add_parameter(pid2, is_optimizable=False)
                pid2 = '(%s%s%s)'%(pid, mode, change)
                # ratelaw, assignmentrules, raterules
                net = self.replace_varid(pid, pid2, only_expr=True)  
                #net.add_assignment_rule(pid2, '%s%s%s'%(pid, mode, str(change)), 0)
            if mode == '=':
                pval_new = condition[-1]
                net = self.set_var_val(pid, pval_new)  # need to verify...
        return net
        
        
    def measure(self, msrmts):
        """
        Input:
            msrmts: measurements, a list of (varid, time)
                eg, [('A',1), ('J_R1',np.inf)]
        
        """
        varids, times = zip(*msrmts)
        varids, times = list(set(varids)), sorted(set(times))  # zip returns a tuple
        traj = self.get_traj(times, varids=varids)
        y = []
        for varid, time in msrmts:
            y.append(traj[varid].loc[time])
        y = butil.Series(y, index=msrmts)
        return y    
        
    
    def get_parameter_sensitivities(self, msrmts):
        """
        Get the _parameter_ sensitivities of the quantities in measurements 
        (msrmts). 
        
        Input:
            msrmts: measurements, a list of (varid, time)
                eg, [('A',1), ('J_R1',np.inf)]
        
        Output:
            a pd.DataFrame
        """
        #import ipdb
        #ipdb.set_trace()
        varids0, times = zip(*msrmts)
        varids =  list(itertools.product(set(varids0), self.pids))
        times = sorted(set(times))
        traj = self.get_traj(times, varids=varids)
        jac = []
        for varid0, time in msrmts:
            jac.append(traj.loc[time, [(varid0,pid) for pid in self.pids]].tolist())
        jac = butil.DF(jac, index=msrmts, columns=self.pids)
        return jac
    
    get_psens = get_parameter_sensitivities
    
    
    def get_predict(self, expts):
        """
        Returns a predict object, essentially f = X*M, where M is the model and
        X is the design variable.
        
        Input:
            expts: eg,
                condition variable time
            1        ()        S    np.inf
            2   (k1, 2)        S    [1,np.inf]
            3   (k2, 2)        S    [2,10]
        
        """
        #import ipdb
        #ipdb.set_trace()
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
        dids = expts_worked.to_dids()
        
        def f(p):
            y = []
            for net, msrmts in condmap.values():
                net.update(p)
                y_cond = net.measure(msrmts)  # a series
                y.extend(y_cond.tolist())
            y = butil.Series(y, index=dids)
            return y
        
        def Df(p):
            jac = []
            for net, msrmts in condmap.values():
                net.update(p)
                jac_cond = net.get_psens(msrmts)  # a df
                jac.extend(jac_cond.values.tolist())
            jac = butil.DF(jac, index=dids, columns=pids)
            return jac

        pred = predict.Predict(f=f, Df=Df, p0=self.p, pids=pids, dids=dids,
                               expts=expts_worked, cond2net=cond2net)        
        return pred
        

    def get_predict2(self, expts, allow_fail=True):  # FIXME *: better name than allow_fail?
        """
        Returns a predict object, essentially f = X*M, where M is the model and
        X is the design variable.
        
        Input:
            expts: eg,
                    condition variable time
                1        ()        S    np.inf
                2   (k1, 2)        S    [1,np.inf]
                3   (k2, 2)        S    [2,10]
            test: 
        
        """
        #import ipdb
        #ipdb.set_trace()
        
        # this will ...
        expts = expts.regularize()
        """
        if test:
            for cond in expts.condset:
                try:
                    self.perturb(cond, inplace=True)
                    self.set_ss()
                except Exception:
                    expts.rm_condition(cond)
                    logging.warn("Remove condition: %s"%str(cond))
        """
        def f(p):
            y = Series(conds=expts.conds)
            for cond, msrmts in expts:
                self.perturb(cond, inplace=True)
                try:
                    y_cond = self.measure(msrmts, cond=cond)  # a series
                    y.append(y_cond)
                except (Exception, daeintException):
                    if allow_fail:
                        y.conds.remove(cond)
                        logging.warn("...")
                    else:
                        raise
            return y
        
        def Df(p):
            jac = Matrix(conds=expts.conds)
            for cond, msrmts in expts:
                self.perturb(cond, inplace=True)
                try:
                    jac_cond = self.get_psens(msrmts, cond=cond)  # a df
                    jac.append(jac_cond)
                except (Exception, daeintException):
                    if allow_fail:
                        jac.conds.remove(cond)
                        logging.warn("...")
                    else:
                        raise
            #jac = butil.DF(jac, index=expts.dids, columns=self.pids)
            return jac

        pred = predict.Predict(f=f, Df=Df, p0=self.p, pids=self.pids, 
                               dids=expts.dids, expts=expts, allow_fail=allow_fail)
        return pred
        
        
        
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
        

###############################################################################
###############################################################################

    ## calculating methods
    #  dynamics
    def get_traj(self, times, varids=None, copy=False, **kwargs_int):
        """
        Input:
            times: a list of floats, lists or tuples
            copy: really necessary??? check the code behaviors... FIXME **
             
        Output:
            traj
        """
        if copy:
            net = self.copy()
        else:
            net = self
            
        #import ipdb
        #ipdb.set_trace()

        if varids is None:
            # only normal traj (no sens traj)
            varids = self.ncvarids
        if any([isinstance(varid, tuple) for varid in varids]):
            calc_sens = True
        else:
            calc_sens = False
        
        # sort times
        times_sorted = trajectory.sort_times(times)
        
        # see if steady state needs to be calculated
        if times_sorted[-1] == np.inf:
            times_int = times_sorted[:-1]
            calc_ss = True
        else:
            times_int = times_sorted
            calc_ss = False
        
        ## integrate to get traj_int
        # make an empty traj_int
        if times_int == []:
            traj_int = trajectory.Trajectory(varids=varids)
        elif times_int == [0] or times_int == [0.0]:
            dat = [self.evaluate_expr(varid, time=0) for varid in varids]
            traj_int = trajectory.Trajectory(dat=dat, times=[0], varids=varids)
        else:
            # see if there are time intervals
            if 'fill_traj' not in kwargs_int:
                if isinstance(times, tuple) or any([isinstance(t, tuple) for t in times]):
                    fill_traj = True
                else:
                    fill_traj = False
                kwargs_int['fill_traj'] = fill_traj    
        
            # fix time: when supplying a time not starting from 0, say, [1,2], 
            # SloppyCell starts from the current state of the network, 
            # even if the current state does not correspond to t=1.    
            # http://sourceforge.net/p/sloppycell/mailman/message/31806741/
            if float(times_int[0]) != 0.0:
                times_int.insert(0, 0)
            if calc_sens:
                traj_int_sc = Dynamics.integrate_sensitivity(net, times=times_int, **kwargs_int)
            else:
                traj_int_sc = Dynamics.integrate(net, times=times_int, **kwargs_int)
            traj_int = trajectory.Trajectory(traj_int_sc.copy_subset(varids))      
            
        ## perform MCA to get traj_ss
        if calc_ss:
            #f = lambda vid: vid.replace('v_','J_') if isinstance(vid, str) else vid 
            #varids_ss = map(f, varids)
            varssvals = net.get_ssvals(varids=varids)
            traj_ss = trajectory.Trajectory(dat=varssvals.tolist(), 
                                            times=[np.inf], varids=varids)
        else:
            traj_ss = trajectory.Trajectory(varids=varids)
        
        traj_all = traj_int + traj_ss
        
        net.t = times_sorted[-1]
        
        # comment out the following line because if fill_traj is True then
        # we want all the times...
        #traj = traj_all.get_subset(times=times_sorted)  
        return traj_all
    
    """
    structure (parameter-independent)
    """
    
    def reorder_dynvarids(self):
        return struct.reorder_dynvarids(self)
 
   
    def get_stoich_mat(self, **kwargs):
        return struct.get_stoich_mat(self, **kwargs)
                                     
    
    def get_reduced_stoich_mat(self, **kwargs):
        return struct.get_reduced_stoich_mat(self, **kwargs)
    
    
    def get_reduced_link_mat(self):
        return struct.get_reduced_link_mat(self)
    
    
    def get_link_mat(self):
        return struct.get_link_mat(self)
    
    
    def get_pool_mul_mat(self):
        return struct.get_pool_mul_mat(self)
    
    
    def get_ss_flux_mat(self):
        return struct.get_ss_flux_mat(self)
    
    
    def get_ddynvarids(self):
        return struct.get_ddynvarids(self)

    
    def get_idynvarids(self):
        return struct.get_idynvarids(self)                                     
    
    
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

    @property
    def idynvarids(self):
        return self.get_idynvarids()
    
###############################################################################

    def get_dynvarvals_t(self, t):
        traj = self.get_traj([0,t], copy=True)
        return traj.loc[t, self.dynvarids]
    x_t = get_dynvarvals_t
    
    
    def get_velocities(self, t=None):
        """
        Velocities: velocities of species dynamics, dx/dt. 
        
        Input:
            t: time, for non-autonomous dynamics
        """
        self.compile()
        
        if t is None:
            t = self.t
            x = self.x
        else:
            x = self.x_t(t)

        # SloppyCell Network doesn't seem to update self.constantVarValues
        # >>> net.set_var_val('p1', 1)
        # >>> print net.constantVarValues
        # >>> net.set_var_val('p1', 100)
        # >>> print net.constantVarValues
        vels = self.res_function(t, x, np.zeros(len(x)), self.constvarvals)
        vels = butil.Series(vels, index=self.dynvarids)
        return vels
    
    
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
    
    
    def get_s(self, **kwargs_ss):
        self.set_ss(**kwargs_ss)
        return self.x
    
    
    def get_J(self, **kwargs_ss):
        self.set_ss(**kwargs_ss)
        return self.v.rename(OD(zip(self.rateids, self.fluxids)))
    
    
    @property
    def s(self):
        return self.get_s()
    
    
    @property
    def J(self):
        return self.get_J()
    
    
    # steady-state
    def get_ssvals(self, varids=None, **kwargs_ss):
        """
        """
        if varids is None:
            varids = self.ncvarids
        self.set_ss(**kwargs_ss)
        
        varid2val = self.varvals.append(self.J).to_dict()
        vartypes = []
        
        def _calc_n_update(vartype, vartypes, varid2val):
            if vartype in vartypes:
                pass
            else:
                varid2val.update(getattr(self, vartype).to_series().to_dict())
                vartypes.append(vartype)
                
        for varid in varids:
            if isinstance(varid, tuple):
                if varid[0] in self.rateids and varid[1] in self.dynvarids:
                    _calc_n_update('Es', vartypes, varid2val)
                elif varid[0] in self.rateids and varid[1] in self.pids:
                    _calc_n_update('Ep', vartypes, varid2val)
                elif varid[0] in self.dynvarids and varid[1] in self.rateids:
                    _calc_n_update('Cs', vartypes, varid2val)
                elif varid[0] in self.fluxids and varid[1] in self.rateids:
                    _calc_n_update('CJ', vartypes, varid2val)
                elif varid[0] in self.dynvarids and varid[1] in self.pids:
                    _calc_n_update('Rs', vartypes, varid2val)
                elif varid[0] in self.fluxids and varid[1] in self.pids:
                    _calc_n_update('RJ', vartypes, varid2val)
                elif varid[0] in self.lograteids and varid[1] in self.logdynvarids:
                    _calc_n_update('nEs', vartypes, varid2val)
                elif varid[0] in self.lograteids and varid[1] in self.logpids:
                    _calc_n_update('nEp', vartypes, varid2val)
                elif varid[0] in self.logdynvarids and varid[1] in self.lograteids:
                    _calc_n_update('nCs', vartypes, varid2val)
                elif varid[0] in self.logfluxids and varid[1] in self.lograteids:
                    _calc_n_update('nCJ', vartypes, varid2val)
                elif varid[0] in self.logdynvarids and varid[1] in self.logpids:
                    _calc_n_update('nRs', vartypes, varid2val)
                elif varid[0] in self.logfluxids and varid[1] in self.logpids:
                    _calc_n_update('nRJ', vartypes, varid2val)
                else:
                    raise ValueError("Unrecognized value of varid: %s"%str(varid))
                
        ssvals = butil.Series(varid2val).loc[varids]
        
        if ssvals.isnull().any():
            raise ValueError("ssvals has nan:\n%s"%str(ssvals))
        
        return ssvals
    
    
    def get_E_strs(self):
        return mca.get_E_strs(self)
    
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
    
    
    def update(self, p=None, t=None, **thetas):
        """
        Update the state of network. 
        
        Input:
            p: parameter
            t: time 
            thetas: kwargs for individual parameter values, eg, Vf_R1=2
        """
        if p is not None:
            self.update_optimizable_vars(p[self.pids])
        if thetas:
            self.update_optimizable_vars(thetas)
        if t is not None:
            #traj = self.get_traj(times=[0,t])
            #dynvarvals = traj.get_var_vals(varids=self.dynvarids, times=[t])
            #self.updateVariablesFromDynamicVars(self, dynvarvals, t)
            _ = self.get_traj(times=[0,t])  # this suffices? FIXME **
    
    
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
    

    def draw_pgv(self, pos=None, jsonpath=None,
                 arrowsize=0.5,
                 shape_sp='ellipse', shape_rxn='box',
                 spid2rgba=(0,255,0,100), rxnid2rgba=(255,0,0,100),
                 labelfontsize_sp=8, labelfontsize_rxn=8,
                 spid2shapescale=0.2, rxnid2shapescale=0.2,
                 insert_images=True, imagepath='', filepath=''):
        """
        """
        import pygraphviz as pgv
                
        ## some preprossessing...
        
        spids, rxnids = self.spids, self.rxnids
        figids = map(lambda rxnid:'fig'+rxnid, rxnids)
        edges_rxn = butil.flatten([[(spid, rxn.id) for spid, stoichcoef
                                    in rxn.stoichiometry.items() if stoichcoef<0]+\
                                   [(rxn.id, spid) for spid, stoichcoef
                                    in rxn.stoichiometry.items() if stoichcoef>0] 
                                   for rxn in self.rxns], D=1)
        edges_fig = zip(rxnids, figids)
        
        if not isinstance(spid2rgba, Mapping):
            spid2rgba = OD.fromkeys(spids, spid2rgba)
        if not isinstance(rxnid2rgba, Mapping):
            rxnid2rgba = OD.fromkeys(rxnids, rxnid2rgba)
        if not isinstance(spid2shapescale, Mapping):
            spid2shapescale = OD.fromkeys(spids, spid2shapescale)
        if not isinstance(rxnid2shapescale, Mapping):
            rxnid2shapescale = OD.fromkeys(rxnids, rxnid2shapescale)
                  
        pos = _json2pos(jsonpath)
        
        G = pgv.AGraph(strict=False, directed=True)
        
        G.add_nodes_from(spids+rxnids+figids)
        G.add_edges_from(edges_rxn, arrowsize=arrowsize)
        G.add_edges_from(edges_fig, arrowsize=0, style='dotted')
        
        #G.graph_attr = {'label':figtitle, 'labelfontsize':figtitlefontsize}
        
        for spid in spids:
            node_spid = G.get_node(spid)
            node_spid.attr['pos'] = '%f, %f'%tuple(pos[spid])
            node_spid.attr['shape'] = shape_sp
            #node_spid.attr['fillcolor'] = 'green'
            node_spid.attr['fillcolor'] = '#%02x%02x%02x%02x' % spid2rgba[spid]
            node_spid.attr['style'] = 'filled'
            #node_spid.attr['size'] = 2.
            node_spid.attr['fontsize'] = labelfontsize_sp
            node_spid.attr['width'] = spid2shapescale[spid]
            node_spid.attr['height'] = spid2shapescale[spid]
            
        for rxnid in rxnids:
            node_rxnid = G.get_node(rxnid)
            node_rxnid.attr['pos'] = '%f, %f'%tuple(pos[rxnid])
            node_rxnid.attr['shape'] = shape_rxn
            node_rxnid.attr['fillcolor'] = '#%02x%02x%02x%02x' % rxnid2rgba[rxnid]
            node_rxnid.attr['style'] = 'filled'
            node_rxnid.attr['label'] = rxnid
            node_rxnid.attr['fontsize'] = labelfontsize_rxn
            node_rxnid.attr['width'] = rxnid2shapescale[rxnid]
            node_rxnid.attr['height'] = rxnid2shapescale[rxnid]
        
        for rxnid in rxnids:
            figid = 'fig' + rxnid
            node_fig = G.get_node(figid)
            node_fig.attr['pos'] = '%f, %f'%tuple(pos[figid])
            node_fig.attr['shape'] = 'box'
            node_fig.attr['label'] = ''
            node_fig.attr['fillcolor'] = ''
            node_fig.attr['style'] = 'filled'
            if insert_images:
                #node_rxnid.attr['image'] = 'hist_%s.png'%rxnid
                node_fig.attr['imagepath'] = imagepath
                node_fig.attr['image'] = 'hist.pdf'  
        
        G.draw(filepath, prog='neato', args='-n2')
        
    
def _json2pos(filepath):
    """
    """
    import json
    fh = open(filepath)
    out = json.load(fh)
    fh.close()
    pos = OD()
    for node in out['elements']['nodes']:
        pos_node = butil.get_values(node['position'], ['x','y'])
        pos_node[1] *= -1
        pos[node['data']['sbml_id']] = pos_node
    
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
        p = butil.Series()
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


def _get_substrates(stoich, multi=False):
    """
    Input: 
        stoich: a mapping, from species ids to stoich coefs which can be
                an int, a float, or a string.
        multi: a bool; if True, return a multiset by repeating
            for stoichcoef times
    
    Output:
        a list of substrate ids
    """
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


def _get_products(stoich, multi=False):
    """
    Input: 
        stoich: a mapping, from species ids to stoich coefs which can be
                an int, a float, or a string.
        multi: a bool; if True, return a multiset by repeating
            for stoichcoef times
    
    Output:
        a list of product ids
    """
    proids = []
    for spid, stoichcoef in stoich.items():
        try:
            stoichcoef = int(float(stoichcoef))
        except ValueError:
            stoichcoef = 1
        if stoichcoef > 0:
            if multi:
                proids.extend([spid]*stoichcoef)
            else:
                proids.append(spid)
    return proids


def _get_reactants(stoich, multi=False):
    """
    """
    return _get_substrates(stoich, multi=multi) + _get_products(stoich, multi=multi)


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
        sc_unsigned = int(sc_unsigned)
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
    ps = butil.DF([net.p[pids_common] for net in nets], 
              index=[net.id for net in nets], columns=pids_common).T
    if not only_common:
        for net in nets:
            p_net = butil.DF({net.id: net.p[~net.p.index.isin(pids_common)]})
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
    rateids_common = [rateid for rateid in nets[0].rateids 
                      if all([rateid in net.rateids for net in nets])]
    ratelaws = butil.DF([net.asgrules[rateids_common] for net in nets],
                        index=[net.id for net in nets], 
                        columns=rateids_common).T
    if not only_common:
        for net in nets:
            rateids_net = [_ for _ in net.rateids if _ not in rateids_common]
            ratelaws_net = butil.DF({net.id: net.asgrules[rateids_net]})
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
