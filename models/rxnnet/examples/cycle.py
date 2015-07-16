"""
   X0
   |R1
X1--->X2
  \   |R2
 R3 \ V  R4
      X3 -->

"""

from infotopo.models.rxnnet import model
reload(model)


net = model.Network('cycle')
net.add_compartment('CELL')
net.add_species('X0', 'CELL', 1, is_constant=True)
net.add_species('X1', 'CELL', 1)
net.add_species('X2', 'CELL', 1)
net.add_species('X3', 'CELL', 1)
#net.add_reaction_ma('R1', stoich_or_eqn='X1+X0->2 X2', p={'kf':1})
#net.add_reaction_ma('R2', stoich_or_eqn='X2<->X3', p={'kf':1,'dG0':0}, haldane='kf')
#net.add_reaction_ma('R3', stoich_or_eqn='X3<->X1', p={'kf':1,'dG0':0}, haldane='kf')
#net.add_reaction_ma('R4', stoich_or_eqn='X3->', p={'kf':0.5})
net.add_reaction('R1', stoich_or_eqn='X1+X0->2 X2', ratelaw='k1*X0*X1', p={'k1':1})
net.add_reaction('R2', stoich_or_eqn='X2<->X3', ratelaw='k2f*X2-k2b*X3', p={'k2f':1, 'k2b':1})
net.add_reaction('R3', stoich_or_eqn='X3<->X1', ratelaw='k3f*X3-k3b*X1', p={'k3f':1, 'k3b':1})
net.add_reaction('R4', stoich_or_eqn='X3->', ratelaw='k4*X3', p={'k4':1})
#net.compile()

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

