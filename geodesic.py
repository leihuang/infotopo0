# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

def callback_func(t, x, v):
    return True
    


    
class Geodesic(ode):

    def __init__(self, r=None, j=None, M=None, N=None, x=None, v=None, pred=None, 
                 Avv=None, lam = 0.0, dtd = None, 
                 atol = 1e-6, rtol = 1e-6, callback=None, parameterspacenorm=False,
                 pids=None, in_logp=False):
        '''
        Initiate variables for calculating the geodesic 
        inputs:
            r - residual function
            j - jacobian function
            Avv - second directional derivative function
            M - data space dimension
            N - parameter space dimension
            x - starting position
            v - starting direction
            lam - parameter which scales the addition to g (ie g = j^T j + lam *dtd, addition helps with small singular values)
            dtd - identity matrix of size N
            atol - absolute tolerance
            rtol - relative tolerance
            callback - optional function to determine whether to stop geodesic
            parameterspacenorm - whether to restrict a to be perpendicular to v (ie. no acceleration along v, |v|=1)
        outputs:
            set state of ode for finding the geodesic
        '''
        if pred:
            r, j, M, N, pids = pred.f, pred.Df, len(pred.dids), len(pred.pids), pred.pids
            if x is None:
                x = pred.p0
            if v is None:
                v = pred.get_sloppyv(x)
            
        if Avv is None:
            Avv = lambda x, v: (r(x+0.1*v) + r(x-0.1*v) - 2.0*r(x))/0.01

        self.r, self.j, self.Avv = r, j, Avv
        self.M, self.N = M, N
        self.lam = lam
        if dtd is None:
            self.dtd = np.eye(N)
        else:
            self.dtd = dtd
        self.atol = atol
        self.rtol = rtol
        ode.__init__(self, self.geodesic_rhs, jac = None)
        self.set_initial_value(x, v)
        ode.set_integrator(self, 'vode', atol = atol, rtol = rtol)
        if callback is None:
            self.callback = callback_func
        elif callback == 'volume':
            self.callback = lambda t,x,v: self.vols[-1] > self.vols[0] * 0.01**N
        else:
            self.callback = callback
        self.parameterspacenorm = parameterspacenorm
        self.pids = pids
        self.in_logp = in_logp


    def geodesic_rhs(self, t, xv):
        ''' Calculate the rhs of the geodesic eqn'''
        x = xv[:self.N]
        v = xv[self.N:]
        j = self.j(x)
        g = np.dot(j.T, j) + self.lam*self.dtd
        Avv = self.Avv(x, v)
        a = -np.linalg.solve(g, np.dot(j.T, Avv) )
        if self.parameterspacenorm:
            a -= np.dot(a,v)*v/np.dot(v,v)
        return np.append(v, a)
    
    
    def svd(self, x):
        jac = self.j(x)
        U, S, Vh = np.linalg.svd(jac)
        return U, S, Vh
    
        
    def get_vol_sloppyv(self, x, eps=0.01):
        U, S, Vh = self.svd(x)
        sloppyvf = Vh[:,-1]
        sloppyvb = -sloppyvf
        xf = x + sloppyvf*eps
        xb = x + sloppyvb*eps
        volf = np.prod(self.svd(xf)[1])
        volb = np.prod(self.svd(xb)[1])
        if volf < volb:
            sloppyv = sloppyvf
        else:
            sloppyv = sloppyvb
        
        # The following codes implement the selection method mentioned 
        # in the second paragraph of Transtrum & Qiu 14, suppl doc., 
        # which is based on the speed;
        # But they do not yield satisfying results, hence commented off. 
        """
        speedf = np.linalg.norm(self.svd(xf)[-1][:,-1]) 
        speedb = np.linalg.norm(self.svd(xb)[-1][:,-1]) 
        if speedf > speedb:
            sloppyv = sloppyvf
        else:
            sloppyv = sloppyvb
        """
        return np.prod(S), sloppyv

       
    def set_initial_value(self, x, v):
        ''' set initial value of the integrator'''
        self.xs = np.array([x])
        self.vs = np.array([v])
        self.ts = np.array([0.0])
        self.rs = np.array([ self.r(x) ] )
        self.vels = np.array([ np.dot(self.j(x), v) ] )
        vol, sloppyv = self.get_vol_sloppyv(x)
        self.vols = np.array([vol])  # volumes
        self.sloppyvs = np.array([sloppyv])  # sloppy direction
        ode.set_initial_value( self, np.append(x, v), 0.0 )


    def step(self, dt = 1.0):
        ''' take a step along the geodesic'''
        ode.integrate(self, self.t + dt, step = 1)
        self.xs = np.append(self.xs, [self.y[:self.N]], axis = 0)
        self.vs = np.append(self.vs, [self.y[self.N:]], axis = 0 )
        self.rs = np.append(self.rs, [self.r(self.xs[-1])], axis = 0)
        self.vels = np.append(self.vels, [np.dot(self.j(self.xs[-1]), self.vs[-1])], axis = 0)
        self.ts = np.append(self.ts, self.t)
        vol, sloppyv = self.get_vol_sloppyv(self.xs[-1])
        self.vols = np.append(self.vols, vol)
        self.sloppyvs = np.append(self.sloppyvs, [sloppyv], axis=0) 
        
        

    def integrate(self, tmax, maxsteps = 500):
        '''take steps along the full geodesic'''
        cont = True
        while self.successful() and len(self.xs) < maxsteps and self.t < tmax and cont:
            self.step(tmax - self.t)
            cont = self.callback(self.t, self.y[:self.N], self.y[self.N:])
            
            
    def integrate2(self, tmax=10, maxsteps=500, callback=None):
        """
        Integrate...
        """
        cont = True
        while cont:
            self.step(tmax - self.t)  # ??
            if callback:
                cont = callback(self.t, self.y[:self.N], self.y[self.N])
            if tmax:
                cont = cont and self.t < tmax
            if maxsteps:
                cont = cont and len(self.ts) < maxsteps
        
        #while self.successful() and len(self.xs) < maxsteps and self.t < tmax and cont:
            #self.step(tmax - self.t)
            #cont = self.callback(self.t, self.y[:self.N], self.y[self.N:])
            

    def plot_traj(self, pid1, pid2, log10=True, color_vol=False, filepath=''):
        """
        Input:
            color_vol: color map the volume
        """
        xs = self.xs[:,self.pids.index(pid1)]
        ys = self.xs[:,self.pids.index(pid2)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if color_vol:
            cax = ax.scatter(xs, ys, c=self.vols, cmap=cm.jet, edgecolor='none')
            fig.colorbar(cax)
        else:
            ax.scatter(xs, ys, edgecolor='none')
        if log10:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(pid1)
        ax.set_ylabel(pid2)
        plt.savefig(filepath)
        plt.show()
        plt.close()