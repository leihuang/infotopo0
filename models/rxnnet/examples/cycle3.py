"""

"""

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model)


def get_net1():
    """
       C
       |R1
       V
    X1--->X2
      \   |R2
     R3 \ V  R4
          X3 -->

    """
    net = model.Network('cycle')
    net.add_compartment('CELL')
    net.add_species('C', 'CELL', 1, is_constant=True)
    net.add_species('X1', 'CELL', 1)
    net.add_species('X2', 'CELL', 1)
    net.add_species('X3', 'CELL', 1)
    #net.add_reaction_ma('R1', stoich_or_eqn='X1+X0->2 X2', p={'kf':1})
    #net.add_reaction_ma('R2', stoich_or_eqn='X2<->X3', p={'kf':1,'dG0':0}, haldane='kf')
    #net.add_reaction_ma('R3', stoich_or_eqn='X3<->X1', p={'kf':1,'dG0':0}, haldane='kf')
    #net.add_reaction_ma('R4', stoich_or_eqn='X3->', p={'kf':0.5})
    net.add_reaction('R1', stoich_or_eqn='X1+C1->2 X2', ratelaw='k1*C*X1', p={'k1':1})
    net.add_reaction('R2', stoich_or_eqn='X2<->X3', ratelaw='k2f*X2-k2b*X3', p={'k2f':1, 'k2b':1})
    net.add_reaction('R3', stoich_or_eqn='X3<->X1', ratelaw='k3f*X3-k3b*X1', p={'k3f':1, 'k3b':1})
    net.add_reaction('R4', stoich_or_eqn='X3->', ratelaw='k4*X3', p={'k4':1})
    net.compile()
    return net


def get_net2():
    """
       C
       |R1
       V     R4
    X1--->X2 -->
      \   |R2
     R3 \ V  
          X3

    """
    net = model.Network('cycle')
    net.add_compartment('CELL')
    net.add_species('C', 'CELL', 1, is_constant=True)
    net.add_species('X1', 'CELL', 1)
    net.add_species('X2', 'CELL', 1)
    net.add_species('X3', 'CELL', 1)
    net.add_reaction('R1', stoich_or_eqn='X1+C->2 X2', ratelaw='k1f*C*X1-k1b*X2**2', p=OD.fromkeys(['k1f','k1b'],1))
    net.add_reaction('R2', stoich_or_eqn='X2<->X3', ratelaw='k2f*X2-k2b*X3', p={'k2f':1, 'k2b':1})
    net.add_reaction('R3', stoich_or_eqn='X3<->X1', ratelaw='k3f*X3-k3b*X1', p={'k3f':1, 'k3b':1})
    net.add_reaction('R4', stoich_or_eqn='X2->', ratelaw='V4*X2/(X2+K4)', p=OD.fromkeys(['V4','K4'],1))
    net.add_ratevars()
    net.compile()
    return net

net_cycle3_mar = get_net2()


def get_net3():
    """
    """
    net = model.Network('cycle')
    net.add_compartment('CELL')
    net.add_species('C1', 'CELL', 2, is_constant=True)
    net.add_species('C2', 'CELL', 1, is_constant=True)
    net.add_species('X1', 'CELL', 1)
    net.add_species('X2', 'CELL', 1)
    net.add_species('X3', 'CELL', 1)
    net.add_reaction('R1', stoich_or_eqn='X1+C1->2 X2', ratelaw='k1f*C1*X1-k1b*X2**2', p=OD.fromkeys(['k1f','k1b'],1))
    net.add_reaction('R2', stoich_or_eqn='X2<->X3', ratelaw='k2f*X2-k2b*X3', p={'k2f':1, 'k2b':1})
    net.add_reaction('R3', stoich_or_eqn='X3<->X1', ratelaw='k3f*X3-k3b*X1', p={'k3f':1, 'k3b':1})
    net.add_reaction('R4', stoich_or_eqn='X2->C2', ratelaw='k4f*X2-k4b*C2', p=OD.fromkeys(['k4f','k4b'],1))
    net.add_ratevars()
    net.compile()
    return net

net_cycle3_mar2 = get_net3()

"""

Es_str = net.get_E_strs()[0]
varids = list(set(butil.flatten([model.exprmanip.extract_vars(ex) for ex in 
                                 re.sub('[\[|\]]', '', Es_str).split(',')])))
for varid in varids:
    locals()[varid] = sympy.Symbol(varid)

symEs = sympy.Matrix(eval(Es_str))
symN = sympy.Matrix(net.N)

p = model.parameter.Parameter([1,1,1,1], net.pids)
net.update(p=p)
symM = symN * symEs
symM2 = symM.subs({'kf_R1':'k1', 'kf_R2':'k2', 'kf_R3':'k3', 'kf_R4':'k4', 'KE_R2':1, 'KE_R3':1, '1.0':1, '2.0':2})
symM3 = symM2.subs({'X0':1, 'k1':1, 'k2':'a', 'k3':'b', 'k4':'c'})
#symM3 = Matrix([[-b-1,  0,      b],
#                [   2, -a,      a],
#                [   b,  a, -a-b-c]])
M = symM.evalf(subs=dict(net.varvals))
evals = M.eigenvals().keys()
"""

