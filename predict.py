"""
"""

from __future__ import division
from collections import OrderedDict as OD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as p3
#from mpl_toolkits.mplot3d import proj3d

import parameter
import residual, geodesic
reload(residual)
reload(parameter)
reload(geodesic)


class Predict(object):
    """
    - Make p0 compulsory and default??  Yes 
    - Make output of f a pd.Series and Df a pd.DataFrame?  Yes
    """
    
    def __init__(self, f, pids, dids, p0, Df=None, domain=None, prior=None, **kwargs):
        """
        Input:
            f: 
            pids: parameter ids
            dids: data(/residual?) ids
            p0:  
            
        
        """
        # use np.array() because when passing a pd object to the object's
        # __init__ with index also given, the values would be messed up
        # Eg, ser = pd.Series([1,2], index=['a','b'])
        # pd.Series(ser, index=['A','B']) would return:
        # A   NaN
        # B   NaN
        # dtype: float64
        # for details, one can check out .../pandas/core/series.py
        p0 = parameter.Parameter(p0, pids)
        
        def _f(p=None):
            if p is None:
                p = p0
            return pd.Series(np.array(f(p)), index=dids)
        
        if Df is None:
            pass  # finite difference?
        
        def _Df(p=None):
            if p is None:
                p = p0
            return pd.DataFrame(np.array(Df(p)), index=dids, columns=pids)
                    
        self.f = _f
        self.pids = pids
        self.dids = dids
        self.p0 = p0
        self.Df = _Df
        # domain and prior: necessary?
        self.domain = domain
        self.prior = prior
        
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)
        

    # necessary??    
    def __getattr__(self, attr):
        return getattr(self.f, attr)


    def __call__(self, p=None):
        return self.f(p)
    
    
    def __repr__(self):
        return "Parameter ids: %s\nData ids: %s\np0:\n%s"%\
            (str(self.pids), str(self.dids), str(self.p0))
        
    
    def get_in_logp(self):
        """
        Get a Prediction object in log parameters.
        """
        def f_logp(lp):
            lp = parameter.Parameter(lp, self.p0.logpids)
            return self.f(lp.exp())
        
        def Df_logp(lp):
            lp = parameter.Parameter(lp, self.p0.logpids)
            return self.Df(lp.exp()) * lp.exp()  # needs testing

        pred_logp = Predict(f_logp, pids=self.p0.logpids, dids=self.dids, 
                            p0=self.p0.log(), Df=Df_logp, 
                            domain=None, prior=None)
        return pred_logp
    
    
    def plot_image(self, p0=None, decade=6, npt=100, pts=None, xyzlabels=['','',''], 
                   filepath='', 
                   color='b', alpha=0.5, shade=False, edgecolor='none', 
                   **kwargs_surface):
        """
        Plot the image of predict, aka "model manifold".
        
        Input:
            decade: how many decades to cover
            npt: number of points for each parameter
            pts: a list of 3-tuples for the points to be marked
        """
        #import ipdb
        #ipdb.set_trace()
        if p0 is None:
            p0 = self.p0
            
        ps = [np.logspace(np.log10(p0_i)-decade/2, np.log10(p0_i)+decade/2, npt) 
              for p0_i in p0]
        pss = np.meshgrid(*ps)
        
        # make a dummy function that takes in the elements of an input vector 
        # as separate arguments
        def _f(*p):
            return self(p) 

        yss = _f(*pss)
        if len(yss) > 3:
            #yss = pca(yss, k=3)
            xyzlabels = ['PC1', 'PC2', 'PC3']
            pass
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        ax.plot_surface(*yss, color=color, alpha=alpha, shade=shade, 
                        edgecolor=edgecolor, **kwargs_surface)
                        
        if pts is not None:
            ax.scatter(*np.array(pts).T, color='r', alpha=1)
            
        ax.set_xlabel(xyzlabels[0])
        ax.set_ylabel(xyzlabels[1])
        ax.set_zlabel(xyzlabels[2])

        plt.show()
        plt.savefig(filepath)
        plt.close()
        
    
    def get_Df_fd(self, rdelta=0.01):
        """
        Get jacobian of f through symmetric finite difference
        
        """
        def _Df(p=None):
            if p is None and self.p0 is not None:
                p = self.p0
            jacT = []  # jacobian matrix transpose
            for i, p_i in enumerate(p):
                deltap_i = p_i * rdelta
                deltap = np.zeros(len(p))
                deltap[i] = deltap_i
                p_plus = p + deltap
                p_minus = p - deltap
                jacT.append((self(p_plus)-self(p_minus))/2/deltap_i)
            jac = np.transpose(jacT)
            jac = pd.DataFrame(jac, index=self.dids, columns=self.pids) 
            return jac
        
        if self.Df == None:
            self.Df = _Df
        return _Df
    
    
    def make_dat(self, p=None, scheme='sigma', cv=0.2, sigma0=1, sigma_min=1,
                 **kwargs):
        """
        Input:
            scheme: 'sigma': constant sigma
                    'cv': proportional to dat by cv
                    'mixed': the max of scheme 'sigma' and 'cv'
            kwargs: a placeholder
        """
        y = self(p)
        if scheme == 'sigma':
            sigmas = [sigma0]*len(y)
        if scheme == 'cv':
            sigmas = y*cv
        if scheme == 'mixed':
            sigmas = np.max((y*cv, [sigma0]*len*y), axis=1)
        dat = pd.DataFrame(OD([('mean',y),('sigma',sigmas)]))
        return dat
    
    
    def get_geodesic_eqn(self, p0=None, rank_sloppy=1, lam=1e-3):
        """
        Input:
            p0: initial parameter
            v0: initial parameter velocity
            rank_sloppy: 
        """
        if p0 is None:
            p0 = self.p0
        jac = self.Df(p0)
        U, S, Vh = np.linalg.svd(jac)
        v0 = Vh[:,-rank_sloppy]
        
        geqn = geodesic.Geodesic(r=self.f, j=self.Df, 
                                 M=len(self.dids), N=len(self.pids), 
                                 x=p0, v=v0, lam=lam)
        return geqn
    
    
    def to_residual(self, dat=None, **kwargs_dat):
        """
        Input:
            dat:
            kwargs_dat: kwargs for 
                self.make_dat(p=None, scheme='sigma', cv=0.2, sigma0=1, 
                              sigma_min=1)
        """
        if dat is None:
            dat = self.make_dat(**kwargs_dat)
        return residual.Residual(pred=self, dat=dat)
    
    
    def get_dresdp(self, dat):
        pass
    
    
    #def linearize_fd(self, p, rdelta=0.01):
    #    pass
    
    
    def plot(self, n=100, pts=None, show=True, filepath=''):
        """
        """
        if self.domain is not None:
            ps = self.domain.apply(lambda interval: 
                                   np.linspace(interval[0], interval[1], n+1))
        pgrids = np.meshgrid(*ps)  
        ygrids = self.f(*pgrids)
        #ys = [ygrid.flatten() for ygrid in ygrids]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        #import ipdb
        #ipdb.set_trace()
        
        ax.set_aspect("equal")
        ax.plot_surface(*ygrids, color='b', alpha=0.2,
                        shade=False, edgecolor='none')
        #ax.set_xlim(0,2)
        #ax.set_ylim(0,2)
        #ax.set_zlim(0,2)
        
        if pts is not None:
            ax.plot3D(pts, color='r')
        
        if show:
            plt.show()
        plt.savefig(filepath)
        plt.close()
        
        
    def svd(self, p):
        """
        U, S, Vh = svd(jac), return U, S, _Vh_ (not V)
        """
        jac = self.Df(p)
        return np.linalg.svd(jac)
    
    
    def get_spectrum(self, p):
        sigmas = self.svd(p)[1]
        return sigmas
    
    
    def get_sloppyv(self, p=None, eps=0.01):
        """
        Should be in in logp? Otherwise it does not make sense to compare 
        the spectrums at p+deltap and p-deltap. (eg, when p0=[1], and 0 and inf are two limits.)
        """
        sloppyv_f = self.svd(p)[2][-1,:]
        sloppyv_b = -sloppyv_f
        p_f = p + sloppyv_f*eps
        p_b = p + sloppyv_b*eps
        vol_f = np.prod(self.get_spectrum(p_f))
        vol_b = np.prod(self.get_spectrum(p_b))
        if vol_f < vol_b:
            sloppyv = sloppyv_f
        else:
            sloppyv = sloppyv_b
        
        # The following codes implement the selection method mentioned 
        # in the second paragraph of Transtrum & Qiu 14, suppl doc., 
        # which is based on the speed;
        # But they do not yield satisfying results, hence commented off. 
        """
        speed_f = np.linalg.norm(self.svd(p_f)[-1][:,-1]) 
        speed_b = np.linalg.norm(self.svd(p_b)[-1][:,-1]) 
        if speed_f > speed_b:
            sloppyv = sloppyv_f
        else:
            sloppyv = sloppyv_b
        """
        return sloppyv
    
    
    def plot_sloppyv_field(self, pid2range, filepath=''):
        """
        Input:
            pid2range: a dict mapping from _two_ pids to their corresponding values
        """
        # order the dict
        pids, pranges = zip(*[(pid, pid2range[pid]) for pid in self.pids 
                             if pid in pid2range])
        
        p1ss, p2ss = np.meshgrid(*pranges)
        shape = p1ss.shape
        vxss, vyss = np.zeros(shape), np.zeros(shape)        
        for i,j in np.ndindex(shape):
            p = [p1ss[i,j], p2ss[i,j]]
            v = self.get_sloppyv(p)
            vxss[i,j] = v[0]
            vyss[i,j] = v[1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.quiver(p1ss, p2ss, vxss, vyss, pivot='middle', 
                  headwidth=3, headlength=3)
        ax.set_xlabel(pids[0])
        ax.set_ylabel(pids[1])
        ax.set_xlim(p1ss.min()*0.8, p1ss.max()*1.1)
        ax.set_ylim(p2ss.min()*0.8, p2ss.max()*1.1)

        plt.savefig(filepath)
        plt.show()
        plt.close()
        
        
    def plot_sloppyvs(self, ps, plabels=None, filepath=''):
        """
        Input:
            ps: a list of parameter vectors
            plabels: a list of labels 
        """
        m, n = len(ps), len(ps[0])
        colors = ['b','g','r','c','m','y','k']
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 1/(m+2)
        for idx, p in enumerate(ps):
            v = self.get_sloppyv(p)
            xs = np.arange(n)
            ax.bar(xs+(idx+1)*width, v, width=width, color=colors[idx], 
                   edgecolor='none')
        if plabels:
            ax.legend(plabels, loc='lower right')
        ax.set_xticks([0]+(np.arange(n)+0.5).tolist()+[n])
        ax.set_xticklabels(['']+self.pids+[''])
        ax.set_ylabel('Components')
        ax.set_ylim(-1,1)
        plt.subplots_adjust(left=0.2)
        plt.savefig(filepath)
        plt.show()
        plt.close()
    
        
    def plot_spectrums(self, ps, plabels=None, filepath=''):
        """
        Input:
            ps: a list of parameter vectors
            plabels: a list of labels 
        """
        m, n = len(ps), len(ps[0])
        
        fig = plt.figure(figsize=(2*m, 2*n**0.8))  # need to be tuned
        ax = fig.add_subplot(111)
        for idx, p in enumerate(ps):
            sigmas = self.get_spectrum(p)
            for sigma in sigmas:
                y = np.log10(sigma)
                ax.plot([idx+0.1, idx+0.9], [y, y], c='k')
            if plabels:
                ax.set_xticks([0]+(np.arange(m)+0.5).tolist()+[m])
                ax.set_xticklabels(['']+plabels+[''])
        ax.set_ylabel(r'$\log_{10} (\sigma)$')
        plt.subplots_adjust(left=0.2)
        plt.savefig(filepath)
        plt.show()
        plt.close()
        
        
    def currying(self, ):
        """
        Fix part of the arguments
        
        https://en.wikipedia.org/wiki/Currying 
        """
        pass
    
    
    def __add__(self, other):
        """
        Concatenate the output:
        (f+g)(p) = (f(p),g(p))
        """
        pass
    
    