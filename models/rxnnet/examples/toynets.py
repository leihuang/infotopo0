"""A collection of toy networks.

A few dimensions of change: 
    - ma or mm
    - with haldane or not
    - VK or kb
    - with r or not
"""

from infotopo.models.rxnnet import model
from infotopo.models.rxnnet.ratelaw import *


Cids2 = ['C1', 'C2']
Cids3 = ['C1', 'C2', 'C3']
CKEids22 = ['C1', 'C2', 'KE1', 'KE2']
CKEids23 = ['C1', 'C2', 'KE1', 'KE2', 'KE3']
CKEids33 = ['C1', 'C2', 'C3', 'KE1', 'KE2', 'KE3']
CKEids34 = ['C1', 'C2', 'C3', 'KE1', 'KE2', 'KE3', 'KE4']
C = {'C1':2, 'C2':0.5}

path2ma = model.make_path([ma11]*2, cids=Cids2, **C)
path2mm = model.make_path([mm11]*2, cids=Cids2, **C)
path2mmr = model.make_path([mm11]*2, cids=Cids2, r=True, **C)
path2mmkb = model.make_path([mm11_kb]*2, cids=Cids2, **C)
path2mmkbr = model.make_path([mm11_kb]*2, cids=Cids2, r=True, **C)
path2mah = model.make_path([mah11]*2, cids=CKEids22, **C)
path2mmh = model.make_path([mmh11]*2, cids=CKEids22, **C)
path2mmhr = model.make_path([mmh11]*2, cids=CKEids22, r=True, **C)
path2mmhkb = model.make_path([mmh11_kb]*2, cids=CKEids22, **C)
path2mmhkbr = model.make_path([mmh11_kb]*2, cids=CKEids22, r=True, **C)

path3ma = model.make_path([ma11]*3, cids=Cids2, **C)
path3mm = model.make_path([mm11]*3, cids=Cids2, **C)
path3mmkb = model.make_path([mm11_kb]*3,  cids=Cids2, **C)
path3mah = model.make_path([mah11]*3, cids=CKEids23, **C)
path3mmh = model.make_path([mmh11]*3, cids=CKEids23, **C)
path3mmhkb = model.make_path([mmh11_kb]*3, cids=CKEids23, **C)

eqns3 = ['X1+C1<->2 X2', 'X2+C2<->X1', 'X2<->C3']
cycle3ma = model.make_net(eqns=eqns3, ratelaws=[ma22_ABPP,ma21,ma11], cids=Cids3)
cycle3mm = model.make_net(eqns=eqns3, ratelaws=[mm22_ABPP,mm21,mm11], cids=Cids3)
cycle3mmr = model.make_net(eqns=eqns3, ratelaws=[mm22_ABPP,mm21,mm11], r=True, cids=Cids3)
cycle3mmkb = model.make_net(eqns=eqns3, ratelaws=[mm22_ABPP_kb,mm21_kb,mm11_kb], cids=Cids3)
cycle3mmkbr = model.make_net(eqns=eqns3, ratelaws=[mm22_ABPP_kb,mm21_kb,mm11_kb], r=True, cids=Cids3)
cycle3mah = model.make_net(eqns=eqns3, ratelaws=[mah22_ABPP,mah21,mah11], cids=CKEids33)
cycle3mahr = model.make_net(eqns=eqns3, ratelaws=[mah22_ABPP,mah21,mah11], r=True, cids=CKEids33)
cycle3mmh = model.make_net(eqns=eqns3, ratelaws=[mmh22_ABPP,mmh21,mmh11], cids=CKEids33)
cycle3mmhr = model.make_net(eqns=eqns3, ratelaws=[mmh22_ABPP,mmh21,mmh11], r=True, cids=CKEids33)
cycle3mmhkb = model.make_net(eqns=eqns3, ratelaws=[mmh22_ABPP_kb,mmh21_kb,mmh11_kb], cids=CKEids33)
cycle3mmhkbr = model.make_net(eqns=eqns3, ratelaws=[mmh22_ABPP_kb,mmh21_kb,mmh11_kb], r=True, cids=CKEids33)
  
eqns4 = ['X1+C1<->2 X2', 'X2+X3<->X4+X1', 'X4+C2<->X3', 'X2<->C3']
cycle4ma = model.make_net(eqns=eqns4, ratelaws=[ma22_ABPP,ma22,ma21,ma11], cids=Cids3)
cycle4mm = model.make_net(eqns=eqns4, ratelaws=[mm22_ABPP,mm22,mm21,mm11], cids=Cids3)
cycle4mmr = model.make_net(eqns=eqns4, ratelaws=[mm22_ABPP,mm22,mm21,mm11], r=True, cids=Cids3)
cycle4mmkb = model.make_net(eqns=eqns4, ratelaws=[mm22_ABPP_kb,mm22_kb,mm21_kb,mm11_kb], cids=Cids3)
cycle4mmkbr = model.make_net(eqns=eqns4, ratelaws=[mm22_ABPP_kb,mm22_kb,mm21_kb,mm11_kb], r=True, cids=Cids3)
cycle4mah = model.make_net(eqns=eqns4, ratelaws=[mah22_ABPP,mah22,mah21,mah11], cids=CKEids34)
cycle4mmh = model.make_net(eqns=eqns4, ratelaws=[mmh22_ABPP,mmh22,mmh21,mmh11], cids=Cids3)
cycle4mmhr = model.make_net(eqns=eqns4, ratelaws=[mmh22_ABPP,mmh22,mmh21,mmh11], r=True, cids=Cids3)
cycle4mmhkb = model.make_net(eqns=eqns4, ratelaws=[mmh22_ABPP_kb,mmh22_kb,mmh21_kb,mmh11_kb], cids=Cids3)
cycle4mmhkbr = model.make_net(eqns=eqns4, ratelaws=[mmh22_ABPP_kb,mmh22_kb,mmh21_kb,mmh11_kb], r=True, cids=Cids3)

