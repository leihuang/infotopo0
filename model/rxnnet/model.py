"""
"""

from __future__ import division
import copy
import re
import itertools
from collections import OrderedDict as OD

import numpy as np
import pandas as pd

from SloppyCell.ReactionNetworks import Network as Network0, Dynamics, IO, KeyedList
from SloppyCell import ExprManip as expr

from util import butil
reload(butil)

import predict  ##?
reload(predict)

import trajectory
reload(trajectory)

import mca
reload(mca)

# temporary
from util.sloppycell.mca import mcautil
reload(mcautil) 




class Network(Network0):
    """
    """
    def __init__(self, id='', name='', net=None):
        if net is None:
            Network0.__init__(self, id, name)
        else:
            for attrid, attrval in net.__dict__.items():
                setattr(self, attrid, attrval)
        #['id', 'compartments', 'parameters', 'species', 'reactions', 
        # 'assignmentRules', 'algebraicRules', 'rateRules', 'constraints', 'events', 'functionDefinitions']:
                
    @property
    def pids(self):
        return self.optimizableVars.keys()
    
    
    @property
    def p(self):
        return pd.Series([v.value for v in self.optimizableVars],
                         index=self.pids)
    
    
    @property
    def dynvarids(self):
        return self.dynamicVars.keys() 
    
    
    @property
    def dynvarvals(self):
        return pd.Series([v.value for v in self.dynamicVars],
                         index=self.dynvarids)
    x = dynvarvals
    
    
    @property
    def varids(self):
        """
        Ids of variables (non-constants)
        """
        return self.dynvarids + self.assignedVars.keys()
    
    
    @property
    def rateids(self):
        return ['v_'+rxn.id for rxn in self.reactions]
    
    
    @property
    def fluxids(self):
        return ['J_'+rxn.id for rxn in self.reactions]
    
    
    @property
    def rates(self):
        return pd.Series([self.evaluate_expr(rateid) for rateid in self.rateids],
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
                        haldane='kf', add_thermo=True, T=25):
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


    def add_reaction_eq(self, rxnid, stoich_or_eqn, KE):
        """
        Add a reaction that is assumed to be always at equilibrium.
        """
        self.add_reaction(rxnid=rxnid, stoich_or_eqn=stoich_or_eqn, 
                          p={'KE_'+rxnid:KE}, ratelaw='0')
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
        modids = [varid for varid in expr.extract_vars(ratelaw) 
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
            for pinfo, pval in p.items():
                if isinstance(pinfo, tuple):
                    pid, is_optimizable = pinfo
                else:
                    pid, is_optimizable = pinfo, True
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
        modids = [varid for varid in expr.extract_vars(ratelaw) 
                  if varid not in stoich.keys()+self.parameters.keys()]
        for modid in modids:
            stoich[modid] = 0
        
        # add reaction
        self.addReaction(id=rxnid, stoichiometry=stoich, kineticLaw=ratelaw, **kwargs)
    

    def add_ratevars(self):
        """
        Add rate variables.
        """
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
        pass
    
    
    def change_varid(self, varid_old, varid_new):
        """
        Change id of rxn, species, or parameter.
        Ratelaws, assignmentRules, rateRules
        """
        vid, vid2 = varid_old, varid_new
        f = lambda _: vid2 if _ == vid else _
        
        netid2 = f(self.id)
        net2 = self.__class__(netid2, name=self.name)

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
            rxn2.kineticLaw = expr.sub_for_var(rxn.kineticLaw, vid, vid2)
            rxns2.set(rxn2.id, rxn2)
        net2.reactions = rxns2
        
        asrules2 = KeyedList()  # assignment rules
        for varid, rule in self.assignmentRules.items():
            varid2 = f(varid)
            rule2 = expr.sub_for_var(rule, vid, vid2)
            asrules2.set(varid2, rule2)
        net2.assignmentRules = asrules2
        
        alrules2 = KeyedList()  # algebraic rules
        for varid, rule in self.algebraicRules.items():
            varid2 = expr.sub_for_var(varid, vid, vid2)
            rule2 = expr.sub_for_var(rule, vid, vid2)
            alrules2.set(varid2, rule2)
        net2.algebraicRules = alrules2
         
        rrules2 = KeyedList()  # rate rules
        for varid, rule in self.rateRules.items():
            varid2 = f(varid)
            rule2 = expr.sub_for_var(rule, vid, vid2)
            rrules2.set(varid2, rule2)
        net2.rateRules = rrules2    

        for eid, event in self.events.items():
            eid2 = f(eid)
            trigger2 = expr.sub_for_var(event.trigger, vid, vid2)
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
    
    
    def change_varids(self, varidmap):
        """
        Input:
            varidmap: a mapping from old varids to new varids 
        """
        net = self.copy()
        for vid, vid2 in varidmap.items():
            net = net.change_varid(vid, vid2)
        return net
    
    
    def perturb(self, condition):
        """
        """
        #net = self.cosmetics()
        net = self.copy()
        
        if condition == ():
            pass
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
                net.replace_pid(pid, pid2)  # ratelaw, assignmentrules, raterules
                #net.add_assignment_rule(pid2, '%s%s%s'%(pid, mode, str(change)), 0)
            if mode == '=':
                pval_new = condition[-1]
                net.set_var_val(pid, pval_new)  # need to verify...
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
        y = pd.Series(y, index=msrmts)
        return y    
        
    
    def get_sensitivities(self, msrmts):
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
        jac = pd.DataFrame(jac, index=msrmts, columns=self.pids)
        return jac
    
    
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
        pids = self.pids
        dids = expts.to_dids()
        
        net2msrmts = OD()
        for cond, expts_cond in expts.sep_conditions().items():
            net = self.perturb(cond)
            net.compile()
            dids_cond = expts_cond.to_dids()
            msrmts_cond = [did[1:] for did in dids_cond]
            net2msrmts[net] = msrmts_cond
            
        def f(p):
            y = []
            for net, msrmts in net2msrmts.items():
                net.update(p)
                y_cond = net.measure(msrmts)  # a pd.Series
                y.extend(y_cond.tolist())
            y = pd.Series(y, index=dids)
            return y
        
        def Df(p):
            #import ipdb
            #ipdb.set_trace()
            jac = []
            for net, msrmts in net2msrmts.items():
                net.update(p)
                jac_cond = net.get_sensitivities(msrmts)  # a pd.DataFrame
                jac.extend(jac_cond.values.tolist())
            jac = pd.DataFrame(jac, index=dids, columns=pids)
            return jac

            
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
        pred = predict.Predict(f=f, Df=Df, p0=self.p, pids=pids, dids=dids)        
        return pred

###############################################################################
###############################################################################

    ## calculating methods
    #  dynamics
    def get_traj(self, times, varids=None, copy=False, **kwargs_int):
        """
        Input:
            times: a list of floats, lists or tuples
            copy: really necessary??? check the code behaviors...
             
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
            varids = self.varids
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
        
        # integrate to get traj_int
        if times_int != []:
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
        # make an empty traj_int
        else:
            traj_int = trajectory.Trajectory(varids=varids)
            
        # perform MCA to get traj_ss
        if calc_ss:
            #f = lambda vid: vid.replace('v_','J_') if isinstance(vid, str) else vid 
            #varids_ss = map(f, varids)
            varssvals = net.get_ssvals(varids=varids)
            traj_ss = trajectory.Trajectory(dat=varssvals.tolist(), 
                                            times=[np.inf], varids=varids)
        else:
            traj_ss = trajectory.Trajectory(varids=varids)
        
        traj_all = traj_int + traj_ss
        
        # comment out the following line because if fill_traj is True then
        # we want all the times...
        #traj = traj_all.get_subset(times=times_sorted)  
        return traj_all
    
###############################################################################

    # structure (parameter-independent)
    def get_stoich_mat(self):
        """
        """
        pass
    
###############################################################################
    
    def get_velocities(self, dynvarvals=None, t=None):
        """
        Input:
            t: time, for non-autonomous dynamics
        """
        if dynvarvals is None:
            dynvarvals = self.dynvarvals
        if t is None:
            t = 0
        self.compile()
        vels = self.res_function(t, dynvarvals, np.zeros(len(dynvarvals)),
                                 self.constantVarValues)
        vels = pd.Series(vels, index=self.dynvarids)
        return vels
    
    
    def is_ss(self, tol=1e-6):
        return mca.is_ss(self, tol=tol)
    
    
    def set_ss(self, tol=1e-6, method='integration', T0=1e2, Tmax=1e6):
        return mca.set_ss(self, tol=tol, method=method, T0=T0, Tmax=Tmax)
    

    def get_param_elas_mat(self, **kwargs):
        """
        """
        return mca.MCAMatrix(mcautil.get_param_elas_mat(self, **kwargs).tolist(),
                             rowvarids=self.rateids, colvarids=self.pids)
    

    def get_concn_elas_mat(self, **kwargs):
        """
        """
        return mca.MCAMatrix(mcautil.get_concn_elas_mat(self, **kwargs).tolist(),
                             rowvarids=self.rateids, colvarids=self.dynvarids)
    
    
    def get_concn_resp_mat(self, **kwargs):
        return mca.MCAMatrix(mcautil.get_concn_resp_mat(self, **kwargs).tolist(),
                             rowvarids=self.dynvarids, colvarids=self.pids)
    
    
    def get_flux_resp_mat(self, **kwargs):
        return mca.MCAMatrix(mcautil.get_flux_resp_mat(self, **kwargs).tolist(),
                             rowvarids=self.fluxids, colvarids=self.pids)
    
    
    def get_concn_ctrl_mat(self, **kwargs):
        return mca.MCAMatrix(mcautil.get_concn_ctrl_mat(self, **kwargs).tolist(),
                             rowvarids=self.dynvarids, colvarids=self.rateids)
            
            
    def get_flux_ctrl_mat(self, **kwargs):
        return mca.MCAMatrix(mcautil.get_flux_ctrl_mat(self, **kwargs).tolist(),
                             rowvarids=self.fluxids, colvarids=self.rateids)
    
    
###############################################################################
    
    def get_ssvals_type(self, vartype, **kwargs_ss):
        """
        """
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
    
    
    def get_s(self, **kwargs_ss):
        return self.get_ssvals_type(vartype='concn', **kwargs_ss)    
    
    
    def get_J(self, **kwargs_ss):
        return self.get_ssvals_type(vartype='flux', **kwargs_ss)
    
    
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
            varids = self.varids
        self.set_ss(**kwargs_ss)
        
        ssvals_all = pd.Series()
        vartypes = []
        for varid in varids:
            if varid in self.dynvarids:
                vartype = 'concn'
            elif varid in self.fluxids:
                vartype = 'flux'
            elif isinstance(varid, tuple):
                if varid[0] in self.dynvarids and varid[1] in self.rateids:
                    vartype = 'concn_ctrl'  
                elif varid[0] in self.fluxids and varid[1] in self.rateids:
                    vartype = 'flux_ctrl'
                elif varid[0] in self.dynvarids and varid[1] in self.pids:
                    vartype = 'concn_resp'
                elif varid[0] in self.fluxids and varid[1] in self.pids:
                    vartype = 'flux_resp'
                elif varid[0] in self.rateids and varid[1] in self.pids:
                    vartype = 'param_elas'
                elif varid[0] in self.rateids and varid[1] in self.dynvarids:
                    vartype = 'concn_elas'
                else:
                    raise ValueError("Unrecognized value of varid: %s"%str(varid))
            else:
                raise ValueError("Unrecognized value of varid: %s"%str(varid))
            
            if vartype in vartypes:
                continue
            else:
                ssvals_type = self.get_ssvals_type(vartype)
                ssvals_all = ssvals_all.append(ssvals_type)
                vartypes.append(vartype)
                
        ssvals = ssvals_all.loc[varids]
        return ssvals
    
###############################################################################
    
    
    @staticmethod
    def from_sbml(filepath, **kwargs):
        net = IO.from_SBML_file(filepath, **kwargs)
        return Network(net=net)
    
        
    def to_sbml(self, filepath):
        IO.to_SBML_file(self, filepath)
    
    
    def update(self, p=None, t=None):
        """
        """
        if p is not None:
            self.update_optimizable_vars(p)
        if t is not None:
            #traj = self.get_traj(times=[0,t])
            #dynvarvals = traj.get_var_vals(varids=self.dynvarids, times=[t])
            #self.updateVariablesFromDynamicVars(self, dynvarvals, t)
            _ = self.get_traj(times=[0,t])  # this suffices?
    



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
        p = pd.Series()
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
            stoichcoef, spid = '1', l[0]
        if len(l) == 2:
            stoichcoef, spid = l
        return spid, stoichcoef
    
    # re: '<?': 0 or 1 '<'; '[-|=]': '-' or '=' 
    subs, pros = re.split('<?[-|=]>', eqn)
    stoich = OD()
    
    if subs:
        for sub in subs.split('+'):
            subid, stoichcoef = unpack(sub)
            stoich[subid] = '-' + stoichcoef
    if pros:
        for pro in pros.split('+'):
            proid, stoichcoef = unpack(pro)
            stoich[proid] = stoichcoef
        
    return stoich


def _format(varid):
    """
    """
    varid2 = varid.replace(' ', '')
    if varid[0] in '0123456789':
        varid2 = '_' + varid2
    return varid2

