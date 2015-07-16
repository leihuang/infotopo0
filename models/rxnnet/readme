Chemical reaction networks:
net:
compartment: 
species (sp): could be dynamic or constant (boundary)
reaction (rxn):
parameter (param): could be dynamic (assigned) or constant 
dynamic variable (dynvar): dynamic species 
constant variable (convar): constant species and parameter
nonconstant variable (ncvar): dynamic and assigned variable
algebraic variable (algvar):
assigned variable (asgrule): 
assignment rule (asgrule):
algebraic rule (algrule):
raterule (rrule):
 
Dynamic quantities:
t: time
p: parameter vector
x: dynamic species concentration vector, function of t (values of dynvars)
xi: independent dynamic species concentration vector, function of t
xd: dependent dynamic species concentration vector, function of t
dxdt: velocities
v: reaction rate vector, function of t
s: steady-state species concentration vector; s = x(inf)
J: steady-state reaction rate vector; J = v(inf)

pids:
xids:
vids: eg, ['v_R1', 'v_R2']
Jids: eg, ['J_R1', 'J_R2']


Structrue: 
N: stoichiometry matrix
P:
L0:  
L: link matrix
Nr: 
ixids: 
dxids:

MCA: 
Ex: concentration elasticity; Ex = d v/d x
Ep: parameter elasticity; Ep = d v/d p
Cs: 
CJ: 
Rs: 
RJ: 
nCs: 
nCJ:

Epids: (vid, pid); eg, [('v_R1', 'theta1'), ('v_R1', 'theta2')]
Exids: (vid, xid); eg, [('v_R1', 'X1'), ('v_R1', 'X2')]
Csids: (xid, vid); eg, [('X1', 'v_R1'), ('X2', 'v_R1')]
CJids: (Jid, vid); eg, [('J_R1', 'v_R1'), ('J_R2', 'v_R1')]
Rsids: (xid, pid); eg, [('X1', 'theta1'), ('X2', 'theta2')]
RJids: (Jid, pid); eg, [('J_R1', 'theta1'), ('J_R2', 'theta2')]
nCsids: (log_xid, log_vid); eg, [('log_X1', 'log_v_R1'), ('log_X2', 'log_v_R1')]
nCJids: (log_Jid, log_vid); eg, [('log_J_R1', 'log_v_R1'), ('log_J_R2', 'log_v_R1')]

# write MCA tests
# numpy c api deprecation warnings
# 
