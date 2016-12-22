"""

         -------------
C1, C2 -|  ToyCalvin  |-> C3
         -------------
         

R1, Fixation: X1 + C1 <-> 2 X2
R2, Regeneration: X2 + X3 <-> X1 + X4
R3, Transport: X2 <-> C3
R4, Energization:  X4 + C2 <-> X3

"""

from __future__ import division

from collections import OrderedDict as OD

from infotopo.models.rxnnet import model
reload(model)


ratelaw1 = '(V1f*%(rX1)s*%(rC1)s-V1b*%(rX2)s*%(rX2)s)/(1+%(rX1)s+%(rC1)s+%(rX1)s*%(rC1)s+2*%(rX2)s+%(rX2)s*%(rX2)s)'%\
    {'rX1':'X1/K1_X1', 'rC1':'C1/K1_C1', 'rX2':'X2/K1_X2'}
ratelaw2 = '(V2f*%(rX2)s*%(rX3)s-V2b*%(rX1)s*%(rX4)s)/(1+%(rX2)s+%(rX3)s+%(rX2)s*%(rX3)s+%(rX1)s+%(rX4)s+%(rX1)s*%(rX4)s)'%\
    {'rX2':'X2/K2_X2', 'rX3':'X3/K2_X3', 'rX1':'X1/K2_X1', 'rX4':'X4/K2_X4'}
ratelaw3 = '(V3f*%(rX2)s-V3b*%(rC3)s)/(1+%(rX2)s+%(rC3)s)'%\
    {'rX2':'X2/K3_X2', 'rC3':'C3/K3_C3'}
ratelaw4 = '(V4f*%(rX4)s*%(rC2)s-V4b*%(rX3)s)/(1+%(rX4)s+%(rC2)s+%(rX4)s*%(rC2)s+%(rX3)s)'%\
    {'rX4':'X4/K4_X4', 'rC2':'C2/K4_C2', 'rX3':'X3/K4_X3'}
pids_R1 = ['V1f','V1b','K1_X1','K1_C1','K1_X2']
pids_R2 = ['V2f','V2b','K2_X2','K2_X3','K2_X1','K2_X4']
pids_R3 = ['V3f','V3b','K3_X2','K3_C3']
pids_R4 = ['V4f','V4b','K4_X4','K4_C2','K4_X3']

'''
ratelaw1 = 'k1*(X1*C1-X22/K1)'
ratelaw2 = 'k2*(X2*X3-X1*X4/K2)'
ratelaw3 = 'k3*(X2-C3/K3)'
ratelaw4 = 'k4*(X4*C2-X3/K4)'
pids_R1 = ['k1','K1']
pids_R2 = ['k2','K2']
pids_R3 = ['k3','K3']
pids_R4 = ['k4','K4']
'''

net = model.Network('ToyCalvin')
net.add_species('X1', 1)
net.add_species('X2', 1)
net.add_species('X3', 1)
net.add_species('X4', 1)
net.add_species('C1', 2, is_constant=True)
net.add_species('C2', 2, is_constant=True)
net.add_species('C3', 1, is_constant=True)
net.add_reaction(id='R1', stoich_or_eqn='X1+C1<->2 X2', ratelaw=ratelaw1, 
                 name='fixation', p=OD.fromkeys(pids_R1,1))
net.add_reaction(id='R2', stoich_or_eqn='X2+X3<->X1+X4', ratelaw=ratelaw2, 
                 name='regeneration', p=OD.fromkeys(pids_R2,1))
net.add_reaction(id='R3', stoich_or_eqn='X2<->C3', ratelaw=ratelaw3, 
                 name='transport', p=OD.fromkeys(pids_R3,1))
net.add_reaction(id='R4', stoich_or_eqn='X4+C2<->X3', ratelaw=ratelaw4, 
                 name='energization', p=OD.fromkeys(pids_R4,1))

net.add_ratevars()
net.compile()
