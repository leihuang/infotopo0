"""
"""

from __future__ import division

import numpy as np

from util import butil

from infotopo import hasse, predict  #, residual
from infotopo.models.rxnnet import experiments, mcalite, dynlite, ratelaw, model 
from infotopo.models.rxnnet.examples.toynets import path2mmkb as net
reload(mcalite)


hd_mm11 = ratelaw.hd_mm11_kb
hd = hasse.get_product([hd_mm11]*2, f_nodeattr=lambda nodeattrs: 
                       {'ratelaws': [d['ratelaw'] for d in nodeattrs]})
nodeids = hd.order[8] #+ hd.order[7]
#nodeids.remove(('mm11', 'Alin'))
#nodeids.remove(('Plin', 'mm11'))
hd = hd.get_subdiagram(nodeids=nodeids)


def get_net(node):
    nodeattr = node[1]
    rls = nodeattr['ratelaws']
    net = model.make_path(ratelaws=rls, cids=['C1','C2'], C1=2)
    net.compile()
    nodeattr['net'] = net
    
    
def get_preds1(node):
    nodeattr = node[1]
    net = nodeattr['net']
    expts_xt = experiments.get_experiments(net.xids, uids=['t'], us=np.linspace(0.5,5,10))
    expts_xc = experiments.get_experiments(net.xids, uids=['C1','C2'], 
                                           us=butil.get_product([1,2],[1,2]))
    pred_xt = dynlite.get_predict(net, expts_xt, tol=1e-13)
    pred_xc = mcalite.get_predict(net, expts_xc, tol=1e-13)
    pred = pred_xt + pred_xc
    nodeattr['pred_xt'] = pred_xt
    nodeattr['pred_xc'] = pred_xc
    nodeattr['pred'] = pred


def get_preds2(node):
    nodeattr = node[1]
    net = nodeattr['net']
    s_xt = '-'.join(net.ratelaws.tolist())
    pred_xt = predict.str2predict(s_xt, pids=net.pids, uids=net.xids, 
                                  us=np.linspace(0.5,5,10).reshape(10,1),
                                  c=net.c.to_dict())
    s_xc = mcalite.solve_path2(*net.ratelaws)[0]         
    pred_xc = predict.str2predict(s_xc, pids=net.pids, uids=net.cids, 
                                  us=butil.get_product([1,2],[1,2]))
    pred = pred_xt + pred_xc
    nodeattr['pred_xt'] = pred_xt
    nodeattr['pred_xc'] = pred_xc
    nodeattr['pred'] = pred
    

def deltay_ok(gds):
    ytraj = gds._ytraj.iloc[:,:10]
    y0 = ytraj.iloc[0]
    yt = ytraj.iloc[-1]
    deltay = (yt - y0) / y0
    if np.max(np.abs(deltay)) < 0.005:
        return True
    else:
        return False
    
    

def gexplore(node):
    nodeattr = node[1]
    pred, corank = nodeattr['pred'], nodeattr['corank']
    if corank == 0:
        v0idxs = butil.get_product([-1,-2,-3], [True,False])
        nseed = 200
    if corank == 1:
        v0idxs = butil.get_product([-1,-2], [True,False])
        nseed = 100
    seeds = [i for i in range(nseed) if
             pred.get_spectrum(pred.p0.randomize(seed=i))[-1]>1e-7]
    gdss = pred.get_geodesics(seeds=seeds, v0idxs=v0idxs, yidxs=slice(10),
                              atol=1e-3, rtol=1e-3)
    gdss.integrate(tmax=20, dt=0.01, pass_exception=True, maxncall=500)
    nodeattr['gdss'] = gdss
    


def get_edges(node):
    nodeattr = node[1]
    gdss, corank = nodeattr['gdss'], nodeattr['corank']

def _eigvec2boundary(eigvec, tol):
    v = np.abs(eigvec)
    if np.isclose(v[0], 1, atol=tol):
        return '1r'
    elif np.isclose(v[1], 1, atol=tol):
        return '1f'
    elif np.isclose(v[2], 1, atol=tol):
        return '1Alin'
    elif np.isclose(v[3], 1, atol=tol):
        return '1Plin'
    elif np.isclose(v[4], 1, atol=tol):
        return '2r'
    elif np.isclose(v[5], 1, atol=tol):
        return '2f'
    elif np.isclose(v[6], 1, atol=tol):
        return '2Alin'
    elif np.isclose(v[7], 1, atol=tol):
        return '2Plin'
    elif np.all(np.isclose(v[:4], 0.5, atol=tol)):
        return '1APsat'
    elif np.all(np.isclose(v[-4:], 0.5, atol=tol)):
        return '2APsat'
    else:
        raise ValueError("Unrecognized eigvec signature.")
    
bools = gdss.apply(lambda gds: gds.is_boundary(tol_singval=0.02))
idxs = np.where(np.all(bools.reshape(bools.size/2,2), axis=1))[0] * 2    
edges = []
for idx in idxs:
    try:
        boundary1 = _eigvec2boundary(gdss.iloc[idx].get_eigvec(), tol=0.05)
        boundary2 = _eigvec2boundary(gdss.iloc[idx+1].get_eigvec(), tol=0.05)
        edges.append((gdss.index[idx], gdss.index[idx+1], 
                      tuple(sorted([boundary1, boundary2]))))
    except ValueError:
        pass

nodeattr['edges'] = edges


hd.apply(get_net)
hd.apply(get_preds2)
hd.apply(gexplore, print_nodeid=True)
hd.apply(get_edges)

#hd.to_pickle('hd_sn_path2mm_xt.pkl')

