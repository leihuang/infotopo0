"""
"""

from __future__ import division
import copy
import itertools

from scipy.integrate import ode as ODE
import numpy as np
#import matplotlib.cm as cm
    
from util import butil, plotutil
reload(plotutil)
from util.butil import Series, DF
from util.trajectory import Trajectory

"""    
class Geodesic0(ode):

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
            self.callback = None  #callback_func
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
        '''
        speedf = np.linalg.norm(self.svd(xf)[-1][:,-1]) 
        speedb = np.linalg.norm(self.svd(xb)[-1][:,-1]) 
        if speedf > speedb:
            sloppyv = sloppyvf
        else:
            sloppyv = sloppyvb
        '''
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
        '''
        Integrate...
        '''
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
        '''
        Input:
            color_vol: color map the volume
        '''
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
"""

'''
class Geodesic(ode):
    """Geodesic equation...
    
    Naming convention (a bit different from rest of the package, with prefix underscore):
        p: parameter, parameter space coordinate
        _y: _y = _f(p), data space coordinate; prediction or residual
        _f: predict or residual
        _Df: differential of f
        v: p'(t), parameter space velocity
        a: p''(t), parameter space acceleration
        _v: _y'(t), data space velocity
        _a: _y''(t), data space acceleration
        t: time
        
    Caution: scipy.integrate.ode instances has the following built-in attributes:
        y: (p, v)
        f: y' = f(t, y)
        _y: (p, v)
        _integrator: can be set by calling ode.set_integrator; 
            options: vode, zvode, lsoda, dopri5, dop853
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    
    Geodesic equation:
        p_k'' + Gamma^k_ij p_i' p_j' = 0, where Gamma is the connection coefficient
        
    However, using extrinsic geometry one can derive geodesic equation
        in a different and computational less intensive way:
            _y''(t) = _v'(p) = _a_parallel + _a_perpendicular 
                = _a_parallel = _a_prl
                (_a_perpendicular = _a_ppd = 0)
                
        Projection matrix in data space: P_prl = J * (J.T * J).I * J.T
        
        _a = _Avv(v) = _y''(t) 
        => _a_prl = P_prl * _a = J * (J.T * J).inv * J.T * _a
        => a_prl = (J.T * J).inv * J.T * _a
        
        J = U * S * V.T
        => J.T = V * S * U.T
        => J.T * J = (V * S * U.T) * (U * S * V.T) = V * S^2 * V.T
        
        => g = J.T * J + lamb * I = V * (S^2 + lamb * I) * V.T
        => g.inv = (V * (S^2 + lamb * I) * V.T).inv = V * diag(1/(S^2+lamb)) * V.T
        => g.inv * J.T = V * diag(1/(S^2+lamb)) * V.T * (V * S * U.T) = V * diag(S/(S^2+lamb)) * U.T
        
        
            
    """
    def __init__(self, func, Df, p0, v0, pids, yids,
                 Avv=None, lamb=0, integrator='vode', #diag=None, 
                 atol=1e-6, rtol=1e-6, callback=None, constpspeed=False):
        """Initialize variables for 
        
        Input:
            f: coordinate function
            Df: differential of f
            p0: initial parameter/coordinate
            v0: initial velocity in parameter space
            Avv: second directional derivative function, really d2f/dt2
            #diag: diagonal matrix of size N
            lamb: parameter which scales the addition to g; 
                ie, g = J.T*J + lamb*diag, addition helps with small singular values
            
            atol: absolute tolerance
            rtol: relative tolerance
            callback: optional function to determine whether to stop geodesic
            constpspeed: keep the speed in parameter space constant 
                (the default is a constant speed in data space)
        """
        
        if Avv is None:
            Avv = lambda p, v: (pred._f(p+0.1*v) + pred._f(p-0.1*v) - 2.0*pred._f(p)) / 0.01
        
        #if diag is None:
        #    diag = np.eye(len(pids))

        self.pred, self.Avv = pred, Avv
        self.pids, self.yids, self.N, self.M = pids, yids, len(pids), len(yids)
        self.lamb, self.atol, self.rtol = lamb, atol, rtol
        # self.diag = diag
        self.p0, self.v0 = p0, v0

        ode.__init__(self, self.rhs, jac=None)
        self.set_initial_value(np.array(p0), np.array(v0))
        ode.set_integrator(self, integrator, atol=atol, rtol=rtol)
        
        if callback == 'volume':
            self.callback = None  #lambda t,x,v: self.vols[-1] > self.vols[0] * 0.01**self.N
        else:
            self.callback = lambda t, x, v: True

        self.constpspeed = constpspeed


    def rhs(self, t, _y):
        """Calculate the rhs of the geodesic equation."""
        p = _y[:self.N]
        v = _y[self.N:]
        jac = self.pred._Df(p)
        
        # p
        # v
        # a
        # y
        # V
        # A

        U, S, Vt = np.linalg.svd(jac, full_matrices=False)
        A = self.Avv(p, v)
        
        a = -np.dot(np.dot(np.dot(Vt.T, np.diag(S/(S**2+self.lamb))), U.T), A)
        
        if self.constpspeed:
            a -= np.dot(a,v)*v / np.dot(v,v)
        
        return np.append(v, a)
    
    
    def set_initial_value(self, p0, v0):
        """Set initial value of the integrator."""
        ode.set_initial_value(self, np.append(p0, v0), 0.0)
        self.p0, self.v0 = p0, v0
        self._ts = [0]
        self._ps = [p0]
        self._vs = [v0]
        #self._ys = [self._f(p0)]
        #self._vs = [np.dot(self._Df(p0), v0)]
        #vol, sloppyv = self.get_vol_sloppyv(p0)
        #self.vols = np.array([vol])  # volumes
        #self.sloppyvs = np.array([sloppyv])  # sloppy direction
        
    
    def step(self, dt=1.0):
        """Take a step along the geodesic"""
        ode.integrate(self, self.t+dt, step=False)  # FIXME **: step=??
        p, v = self._y[:self.N], self._y[self.N:]
        self._ts.append(self.t)
        self._ps.append(p)
        self._vs.append(v)
        #self._ys.append(self._f(p))
        #self._vs.append(np.dot(self._Df(p), v))
        #vol, sloppyv = self.get_vol_sloppyv(self.xs[-1])
        #self.vols = np.append(self.vols, vol)
        #self.sloppyvs = np.append(self.sloppyvs, [sloppyv], axis=0) 
    
    
    def integrate(self, tmax, dt, maxsteps=1e4):
        """Integrate...
        """
        #if self.t != 0:
        #    self.ts, self.ps, self.vs =\
        #        np.array(self.ts).tolist(), np.array(self.ps).tolist(), np.array(self.vs).tolist()
                #self._ys, self._vs =\
                #np.array(self._ys).tolist(), np.array(self._vs).tolist()
        
        if self.t >= tmax:
            cont = False
        else:
            cont = True
        while cont:
            self.step(dt=dt)
            if self.callback:
                cont = self.callback(self.t, self._ps[-1], self._vs[-1])
            if tmax:
                cont = cont and self.t < tmax
            if maxsteps:
                cont = cont and len(self._ts) < maxsteps
        # clean up the states
        #self.ts = Series(self.ts, name='time')
        #self.ps = Trajectory(np.array(self.ps), index=self.ts, columns=self.pids)
        #self.vs = Trajectory(np.array(self.vs), index=self.ts, 
        #                     columns=map(lambda pid: 'd_%s/d_t'%pid, self.pids))
        #self._ys = Trajectory(np.array(self._ys), index=self.ts, columns=self.yids)
        #self._vs = Trajectory(np.array(self._vs), index=self.ts, 
        #                      columns=map(lambda yid: 'd_%s/d_t'%yid, self.yids))
        
        ## FIXME **: 
        #traj = Trajectory()
    
    @property
    def ts(self):
        return Series(self._ts, name='time')
    
    @property
    def ps(self):
        return Trajectory(self._ps, index=self.ts, columns=self.pids)
    
    @property
    def vs(self):
        return Trajectory(self._vs, index=self.ts, 
                          columns=map(lambda pid: 'd_%s/d_t'%pid, self.pids))
    
    @property
    def ys(self):
        return Trajectory([self.pred._f(p) for p in self._ps], index=self.ts, columns=self.yids)
    
    #self._ys = Trajectory([], index=self.ts, columns=self.yids)
        
    def plot_p_traj0(self, pid1, pid2, log10=True, color_vol=False, filepath=''):
        """
        Input:
            color_vol: color map the volume
        
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
        """
     
    def plot_p_traj(self, **kwargs):
        """
        """
        plotutil.plot(self.ts, self.ps.T, **kwargs)
    plot_p_traj.__doc__ += plotutil.plot.__doc__
    
    
    def plot_y_traj(self):
        pass
    
    
    def plot_y(self):
        """
        """
        pass
'''    

callback_dummy = lambda p, v: True
    

class Geodesic(object):
    """Geodesic equation. 
    
    The most core attribute is 'ode', which is an instance of 
    scipy.integrate.ode. 
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    
    There is also scipy.integrate.odeint, which is just a wrapper function of 
    ODEPACK.lsoda and has far more limited capability.
    http://stackoverflow.com/questions/22850908/what-is-the-difference-between-scipy-integrate-odeint-and-scipy-integrate-ode
    
    Caution: scipy.integrate.ode instances have the following built-in 
            attributes:
        t: current time
        y: (p, v)
        f: y' = f(t, y)
        _y: (p, v)
        _integrator: can be set by calling ode.set_integrator; 
            'vode': part of Sundials
            'zvode': 
            'lsoda': part of ODEPACK
            'dopri5': RK5(4) with Dormand-Prince stepsize control
            'dop853': RK8(5,3) with Dormand-Prince stepsize control
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
            
            solver: algorithms, performance, stiff/nonstiff, quirks...
        
        set_solout: used only in solvers dopri5 and dop853; used for returning 
                intermediate results (aka *dense output*) or terminating 
                an integration.
            * return intermediate results: 
                - http://stackoverflow.com/questions/12926393/using-adaptive-step-sizes-with-scipy-integrate-ode
                - Don't really need it
            * terminate integration:
                - http://scicomp.stackexchange.com/questions/16325/dynamically-ending-ode-integration-in-scipy
                - Do it with callback; more sparsely and less overhead
            
        integrate(t, step, relax): what does step do??
        
        Continue integration?
        
        Store calculations?
        
    
    Naming convention (a bit different from rest of the package, with prefix underscore):
        p: parameter, parameter space coordinate
        _y: _y = _f(p), data space coordinate; prediction or residual
        _f: predict or residual
        _Df: differential of f
        v: p'(t), parameter space velocity
        a: p''(t), parameter space acceleration
        _v: _y'(t), data space velocity
        _a: _y''(t), data space acceleration
        t: time
        
    
    Geodesic equation:
        p_k'' + Gamma^k_ij p_i' p_j' = 0, where Gamma is the connection coefficient
        
    However, using extrinsic geometry one can derive geodesic equation
        in a different and computational less intensive way:
            _y''(t) = _v'(p) = _a_parallel + _a_perpendicular 
                = _a_parallel = _a_prl
                (_a_perpendicular = _a_ppd = 0)
                
        Projection matrix in data space: P_prl = J * (J.T * J).I * J.T
        
        _a = _Avv(v) = _y''(t) 
        => _a_prl = P_prl * _a = J * (J.T * J).inv * J.T * _a
        => a_prl = (J.T * J).inv * J.T * _a
        
        J = U * S * V.T
        => J.T = V * S * U.T
        => J.T * J = (V * S * U.T) * (U * S * V.T) = V * S^2 * V.T
        
        => g = J.T * J + lamb * I = V * (S^2 + lamb * I) * V.T
        => g.inv = (V * (S^2 + lamb * I) * V.T).inv = V * diag(1/(S^2+lamb)) * V.T
        => g.inv * J.T = V * diag(1/(S^2+lamb)) * V.T * (V * S * U.T) = V * diag(S/(S^2+lamb)) * U.T
        
        
            
    """
    def __init__(self, f, Df, p0, v0, pids, yids,
                 Avv=None, dt=1e-2, inv='lam', lam=0, rank=None, const_speed='p', 
                 integrator='vode', atol=1e-2, rtol=1e-2, 
                 callback=None, param_cb=None, **kwargs):
    #             ptype=None, pred=None):
        """Initialize variables for 
        
        Input:
            f: coordinate function
            Df: differential of f
            p0: initial parameter/coordinate
            v0: initial velocity in parameter space
            Avv: second directional derivative function, ie, d2f/dt2
            dt: step size in calculating Avv using finite difference
            #diag: diagonal matrix of size N
            lam: parameter which scales the addition to g; 
                ie, g = J.T*J + lam*I, addition helps with small singular values
            integrator: 
                'vode': scipy default, should be used as default here as well;
                    in many cases has comparable performances compared to 
                    'lsoda', but scales reasonably with atol and rtol
                'lsoda': can scale really badly with atol and rtol
                'dopri5': about 10 times slower
                'dop853': about 20 times slower
            atol: absolute tolerance (can be generous)
            rtol: relative tolerance (can be generous)
            callback: optional function to determine when to stop geodesic;
                '', 'singular', 'singval', 'pspeed'
            param_cb: 
            const_speed: 'p' or 'y'; keep the speed in parameter (default) or
                data space constant
        """
        if Avv is None:
            def Avv(p, v, return_y=False):
                """Get directional second derivative using finite difference. 
                Three f evaluations; y can be returned and stored.
                """
                yp = f(p + v*dt)
                ym = f(p - v*dt)
                y = f(p)
                Afd = (yp + ym - 2*y) / dt**2
                if return_y:
                    return Afd, y
                else:
                    return Afd
            
        N, M = len(pids), len(yids)
        
        self._varvals = []
        self._varids = ['t', 'p', 'v', 'y', 'J', 's', 'eigvecs', 'A', #'a_paral', 'a']
                        'a']
        
        # The common counter decorator, to get the number of times
        # rhs is evaluated in integration; call "gds.ode.f.ncall"  
        # http://stackoverflow.com/questions/21716940/
        # is-there-a-way-to-track-the-number-of-times-a-function-is-called
        def counted(f):
            def wrapped(t, pv):
                wrapped.ncall += 1
                return f(t, pv)
            wrapped.ncall = 0
            return wrapped
        
        @counted
        def get_ode_rhs(t, pv):
            """Calculate the rhs of the geodesic equation.
            """
            p, v = pv[:N], pv[N:]
            
            J = Df(p)
            U, s, Vt = np.linalg.svd(J, full_matrices=False)

            # Derivation here:
            # https://onepieceatime.wordpress.com/2016/03/22/geodesic-equation/
            A, y = Avv(p, v, return_y=True)
            if inv == '':
                sinv = 1 / s
            elif inv == 'pseudo':
                sinv = np.append(1/s[:rank], [0]*(N-rank))
            elif inv == 'lam':
                sinv = s / (s**2 + lam)
            else:
                raise ValueError("Invalid value of inv.")
            a = -np.dot(np.dot(np.dot(Vt.T, np.diag(sinv)), U.T), A)    
            
            if const_speed == 'p':
                
                a_paral = np.dot(a,v) / np.dot(v,v) * v 
                a -= a_paral
                
            #if self.ode.successful():
            self._varvals.append((t, p.copy(), v.copy(), y, J, s, Vt.T, A, 
                                  a))  #    a_paral, a))

            return np.append(v, a)

        ode = ODE(get_ode_rhs, jac=None)
        ode.set_initial_value(np.append(p0, v0))
        ode.set_integrator(integrator, atol=atol, rtol=rtol)
        
        self.f, self.Df, self.Avv, self.dt = f, Df, Avv, dt
        self.pids, self.yids = pids, yids
        self.N, self.M, self.pdim, self.ydim = N, M, N, M
        self.lam, self.const_speed, self.rank, self.inv = lam, const_speed, rank, inv
        self.integrator, self.atol, self.rtol = integrator, atol, rtol
        
        self.ode = ode
        self._ts, self._ps, self._vs = [0], [p0], [v0]
        self.p0, self.v0 = Series(p0, pids), Series(v0, pids)
        
        
        if isinstance(callback, str):
            name_cb = callback
        elif callback is None:
            name_cb = 'dummy'
        else:
            name_cb = 'custom'
            
        if name_cb == 'singular':
            callback = lambda p, v: np.linalg.matrix_rank(Df(p)) == N
        elif name_cb in ['pspeed', 'v']:  # used by Transtrum
            assert const_speed == 'y', 'should be constant speed in y'
            if param_cb is None:
                param_cb = 10  # the number Transtrum uses; it seems eigenv signature isn't very clear by 10
            callback = lambda p, v: np.linalg.norm(v) < param_cb
        elif name_cb == 'singval':
            if param_cb is None:
                param_cb = 1e-12
            callback = lambda p, v: np.linalg.svd(Df(p), compute_uv=False)[-1] > param_cb
        elif name_cb == 'dummy' or name_cb == '':
            callback = callback_dummy
        elif name_cb == 'custom':
            pass
        else:
            raise ValueError
        
        callback.name = name_cb
        self.callback = callback
        self.param_cb = param_cb
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        #self.ptype = ptype
        #self.pred = pred

    
    def set_initial_value(self, p0, v0=None, **kwargs):
        """Set initial value of the integrator and remove all previously stored
        integration results."""
        if v0 is None:
            v0 = self.pred.get_eigenv(**kwargs)  # FIXME **: logp?
        self.ode.set_initial_value(np.append(p0, v0), t=0.0)
        self._ts, self._ps, self._vs = [0], [p0], [v0] 
        self.p0, self.v0 = Series(p0, self.pids), Series(v0, self.pids)
        self.ode.f.ncall = 0
        
        
    def reset(self, **kwargs):
        """
        """
        import inspect
        # get the names of all the parameters
        for attrname in inspect.getargspec(Geodesic.__init__).args:
            if attrname not in kwargs and attrname not in ['self', 'Avv']:
                kwargs[attrname] = getattr(self, attrname)
        return Geodesic(**kwargs)
    
    
    def take_step(self, dt, dense_output=False):
        """Take a step along the geodesic, where the step is not internal
        but nominal."""
        self.ode.integrate(self.ode.t+dt, step=dense_output)
        self._ts.append(self.ode.t)
        self._ps.append(self.ode.y[:self.N])
        self._vs.append(self.ode.y[self.N:])
    
    
    def integrate(self, tmax, dt, maxncall=500, dense_output=False, 
                  print_s0=False, print_step=False, pass_exception=True):
        """Integrate.
        
        pass_exception: use logging?  FIXME ***
        
        Input:
            print_step:
            print_s0: print initial spectrum (to prevent from starting too 
                close to boundaries)
            pass_exception:
        """
        def _print_step():
            try:
                ndigit = len(str(dt).split('.')[1])
            except IndexError:
                ndigit = 0  # dt is an integer and has no decimal point
            fmt_t = '%.' + str(ndigit) + 'f'
            print fmt_t % self.t, '\t', np.round(self.p, 2)

        if print_s0:
            print np.linalg.svd(self.Df(self.p), compute_uv=0)
        
        if print_step:
            print "time \t parameters"
            _print_step()
        
        
        cont = self.t < tmax            
        while cont:
            try:
                self.take_step(dt=dt, dense_output=dense_output)
                
                if not self.callback(self.p, self.v):
                    cont = False
                    self.stop = 'callback_' + self.callback.name
                elif self.t > tmax: 
                    cont = False
                    self.stop = 'tmax'
                elif self.ncall > maxncall:
                    cont = False
                    self.stop = 'maxncall'
                else:
                    cont = True
                    
                if print_step:
                    _print_step()
            except:
                if pass_exception:
                    break
                else:
                    raise
    
    def get_subset(self, interval=None, subtimes=None):
        """
        """
        subgds = copy.deepcopy(self) 
        if interval:
            subgds._ts = self._ts[::interval]
            subgds._ps = self._ps[::interval]
            subgds._vs = self._vs[::interval]
        if subtimes:
            pass
        return subgds
    
    # current t
    @property
    def t(self):
        return self.ode.t
    
    # current p
    @property
    def p(self):
        return self.ode.y[:self.N]
    
    # current v
    @property
    def v(self):
        return self.ode.y[self.N:]
    
    @property
    def ts(self):
        return Series(self._ts, name='time')
    
    @property
    def ptraj(self):  
        return Trajectory(np.array(self._ps), index=self.ts, columns=self.pids)
    ps = ptraj
    
    @property
    def vtraj(self):
        return Trajectory(np.array(self._vs), index=self.ts, columns=self.pids)
    vs = vtraj
    
    @property
    def ytraj(self):
        return Trajectory([self.f(p) for p in self._ps], 
                          index=self.ts, columns=self.yids)
    
    @property
    def atraj(self):
        return Trajectory([self.ode.f(0, np.append(p,v))[self.N:] 
                           for p, v in zip(self._ps, self._vs)], 
                          index=self.ts, columns=self.pids)
    
    @property
    def Vtraj(self):
        return Trajectory([np.dot(self.Df(p), v) 
                           for p, v in zip(self._ps, self._vs)], 
                          index=self.ts, columns=self.yids)
    
    @property
    def Atraj(self):
        return Trajectory([self.Avv(p, v) for p, v in zip(self._ps, self._vs)], 
                          index=self.ts, columns=self.yids)
    
    @property
    def straj(self):
        columns = ['sigma%d'%i for i in range(1, len(self.pids)+1)]
        return Trajectory([np.linalg.svd(self.Df(p), compute_uv=0) for p in self._ps], 
                          index=self.ts, columns=columns)
        
    
    def get_svtraj(self, idx=-1, regularize=False):
        """Get the trajectory of singular/sloppy vector.
        
        Input:
            idx:
            regularize:
        """
        svtraj = Trajectory([np.linalg.svd(self.Df(p))[2][idx] 
                             for p in self._ps],
                            index=self.ts, columns=self.pids)
        if regularize:
            svtraj_reg = []
            for sv in svtraj.values:
                svtraj_reg.append(max([sv, -sv], 
                                      key=lambda v: np.dot(v, svtraj.iloc[0])))
            svtraj = Trajectory(svtraj_reg, index=self.ts, columns=self.pids)
        return svtraj
    
    
    @property
    def ncall(self):
        return self.ode.f.ncall
        
    def _get_vartraj(self, vartype, varids=None):
        """vartype: 'p', 'v', 'y', 'A', 'a', 's'
        """
        idx = self._varids.index(vartype)
        return Trajectory([vals[idx] for vals in self._varvals], 
                          index=Series([vals[0] for vals in self._varvals], 
                                       name='time'), columns=varids)
        
    @property
    def _ptraj(self):
        return self._get_vartraj('p', self.pids)
        
    @property
    def _vtraj(self):
        return self._get_vartraj('v', self.pids)
    
    @property
    def _atraj(self):
        return self._get_vartraj('a', self.pids)
    
    @property
    def _ytraj(self):
        return self._get_vartraj('y', self.yids)
    
    @property
    def _Atraj(self):
        return self._get_vartraj('A', self.yids)
    
    @property
    def _straj(self):
        return self._get_vartraj('s', ['sigma%d'%i for i in range(1, self.N+1)])
    
    @property
    def sloppyvs(self):
        return Trajectory(np.array([np.linalg.svd(self._Df(p))[-1][-1,:] 
                                    for p in self._ps]), 
                          index=self.ts, columns=self.pids)
    
    @property
    def sigmas(self):
        _p2sigma = lambda p: np.linalg.svd(self._Df(p), compute_uv=0)
        return Trajectory(np.array([_p2sigma(p) for p in self._ps]), 
                          index=self.ts, 
                          columns=['sigma%d'%idx for idx in range(1, self.N+1)])
    
    
    def get_stats(self):
        """
        """
        S0 = np.linalg.svd(self.Df(self.p0), compute_uv=False)
        # VTt: terminal VT (V transposed)
        St, VTt = np.linalg.svd(self.Df(self.p))[1:]
        evt =  VTt[-1]  # terminal eigvec
        return S0, St, evt
        
        
    def is_boundary(self, stats=None, tol_singval=1e-2):
        """
        """
        if stats is None:
            stats = self.get_stats()
        S0, St = stats[:2]
        if St[-1] / S0[-1] < tol_singval:
            return True
        else:
            return False
    
        
    def is_global_boundary(self, part, tol_singval=1e-2, tol_eigvec=1e-2):
        """A crude filter to get the global boundaries.
        
        FIXME ****: Or maybe I should check the diff between initial and terminal singvals??
        Changed. Need to update the doc below... 
        
        - First, it has to be a boundary and so far only the terminal singular
            values are used: if the smallest singular value is smaller than 
            a threshold then it is considered to be a boundary (for example,
            to exclude the kind of limiting behavior going to infinity);
        - Second, it has to be global and so far only the terminal eigenvector
            is used: if the element in the eigenvector is greater than 
            a threshold than the corresponding parameter is considered to be
            involved in the limiting behavior.  
        The set of parameters involved in the limiting behavior is then 
        compared to the partition of parameter vector corresponding to 
        different system components to determine if the boundary is global.
        
        (If one day my geodesic integration is more stable and almost always 
        behaves sensibly -- it would require some deeper understanding
        of the diff geom involved -- I may want to code a class "Boundary" 
        or "Limit" to encode the limiting behaviors of a geodesic run. 
        Then it'd be more sensible to move this code there.)
        
        Input:
            part: a list of integers, partition of parameters
            tol_singval: threshold of the smallest singular value to determine
                if it is a boundary (the *larger* the more geodesics 
                considered boundaries)
            tol_eigvec: threshold of elements of eigenvector to determine 
                whether a parameter is involved in the limiting behavior 
                (the *smaller* the more boundaries considered global)
        """            
        def _is_global(stats):
            """
            An example:
            part = (2, 3)  # partition of parameters among component models
            compidxs = [0,0,1,1,1]  # component model indices
            pidx2compidx = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}
            """
            S0, St, evt = stats
            compidxs = butil.flatten([[compidx]*rep for compidx, rep in 
                                      zip(range(len(part)), part)])
            pidxs = range(self.pdim)
            pidx2compidx = dict(zip(pidxs, compidxs))
            pidxs_boundary = np.where(np.abs(evt) > tol_eigvec)[0]
            compidxs_boundary = [pidx2compidx[pidx] for pidx in pidxs_boundary]
            if len(set(compidxs_boundary)) > 1:
                return True
            else:
                return False
        
            """
            rxnidxs = range(1, len(part)+1)
            _part = [0] + list(part)
            pidxss = [range(sum(_part))[start:end] for start, end in 
                      zip(np.cumsum(_part)[:-1], np.cumsum(_part)[1:])]
            rxnidx2pidxs = dict(zip(rxnidxs, pidxss))
            pidxs = np.where(np.abs(evt) > tol_eigvec)[0]
            # FIXME **: nameclash, there is a rxnidxs above
            rxnidxs = [[rxnidx_ for rxnidx_, pidxs_ in rxnidx2pidxs.items() 
                        if pidx in pidxs_] for pidx in pidxs]
            if len(set(butil.flatten(rxnidxs))) > 1:
                return True
            else:
                return False
            """
        stats = self.get_stats()
        if self.is_boundary(stats, tol_singval) and _is_global(stats):
            return True
        else:
            return False 

    
    
    def plot_traj(self, vartype='p', ax=None, **kwargs):
        """
        Input:
            vartype: str, one of the following: 'p', 'v', 'y', 'V', 'sigma'
            kwargs: docstring of plotutil.plot is provided below for convenience
        """
        reload(plotutil)
        plotutil.plot(self.ts, getattr(self, vartype+'s').T, ax=ax, **kwargs)
    plot_traj.__doc__ += plotutil.plot.__doc__
    
    
    def plot_trace(self, vartype='p', varid1=None, varid2=None, **kwargs):
        """
        """
        traj = getattr(self, vartype+'traj')
        if varid1 is None and varid2 is None:
            assert len(traj.columns) == 2, "not exactly two variables"
            varid1, varid2 = traj.columns
            
        xs, ys = traj[varid1], traj[varid2]

        reload(plotutil)
        plotutil.scatterplot(xs, ys, c=self.ts, cmap='jet', **kwargs)
    plot_traj.__doc__ += plotutil.plot.__doc__
    
    
    def plot(self, ):
        """ptraj, ytraj, spectra, eigenv
        plot_gds(gds, pred, xlim=None, **kwargs):
        """
        pass
        """
        axslices = [(0,slice(4)), (1,slice(4)), (2,slice(4)), (3,0), (3,1), (3,2), (3,3)]
        ax_p, ax_y, ax_sigma, ax_eigenv0, ax_eigenvt1, ax_eigenvt2, ax_eigenvt3 =\
        plotutil.get_axs(4, 4, axslices, figsize=(8,8),
                         subplots_adjust={'top':0.8, 'right':0.8, 'left':0.1,
                                          'hspace':0.3})
    
        gds.ptraj.plot(ax=ax_p, markers='', colorscheme='jet', 
                       xylims=[xlim,None], xyticks=[[],None], legends=gds.pids,
                       legendloc=(1,-1.5), xylabels=['', r'$p$'])
        gds._ytraj.plot(ax=ax_y, markers='', colorscheme='standard',
                        xylims=[xlim,None], xyticks=[[],None], xylabels=['', r'$y$'])
        np.log10(gds._straj).plot(ax=ax_sigma, markers='', colorscheme='brg',
                                  xylims=[xlim,None], xylabels=['', r'$\sigma$'])
        
        eigenv0 = pred.get_in_logp().get_eigenv(gds.ps.iloc[0], idx=-1)
        plotutil.barplot(ax=ax_eigenv0, lefts=np.arange(pred.N)-0.5, heights=eigenv0, 
                         widths=1, cmapname='jet', 
                         xylims=[None,[-1,1]], xyticks=[[],[-1,0,1]])
        
        
        eigenvt1 = pred.get_in_logp().get_eigenv(gds.ps.iloc[-1], idx=-1)
        plotutil.barplot(ax=ax_eigenvt1, lefts=np.arange(pred.N)-0.5, heights=eigenvt1, 
                         widths=1, cmapname='jet', 
                         xylims=[None,[-1,1]], xyticks=[[],[-1,0,1]])
        eigenvt2 = pred.get_in_logp().get_eigenv(gds.ps.iloc[-1], idx=-2)
        plotutil.barplot(ax=ax_eigenvt2, lefts=np.arange(pred.N)-0.5, heights=eigenvt2, 
                         widths=1, cmapname='jet', 
                         xylims=[None,[-1,1]], xyticks=[[],[-1,0,1]])
        eigenvt3 = pred.get_in_logp().get_eigenv(gds.ps.iloc[-1], idx=-3)
        plotutil.barplot(ax=ax_eigenvt3, lefts=np.arange(pred.N)-0.5, heights=eigenvt3, 
                         widths=1, cmapname='jet', 
                         xylims=[None,[-1,1]], xyticks=[[],[-1,0,1]])
        
        plotutil.plt.show()
        plotutil.plt.close()
        """

    def get_length(self, metric='riemmanian'):
        """
        Check singularity of the terminal pt? 
        
        Input:
            metric: 'riemmannian' or 'euclidean'
        """
        pass

    def get_eigvec(self, idx_t=-1, idx_eigvec=-1):
        """        
        Input:
            idx_t: int, the index of time (not the actual time)
            idx_eigv: int, the index of eigenvector
        """
        p = self._ps[idx_t]
        eigvec = np.linalg.svd(self.Df(p))[-1][idx_eigvec]
        return butil.Series(eigvec, self.pids)
    
    
    def refresh(self):
        """Used for after updating the methods. 
        """
        return Geodesic(**self.__dict__)
    
        
class Geodesics(butil.Series):
    """A collection of geodesics starting at the same point with different 
    velocities.  FIXME ***: should be relaxed... Just a bunch of geodesics...
    Then I should change the constructor, no p0...
    
    
    cond: condition, (idx_eigenv, uturn)
    conds and v0s: only one is given
    
    No modification (addition, removal of gds, change p0, etc.) has been 
    implemented yet.
    
    Shall I just make it a subclass of butil.Series?  FIXME ****
    - Allow tuples as indices
    - stops, ts, ncalls, etc. (all butil.Series)
    - filter(func_key, func_value)  # call it f_key can be confusing as f is special in infotopo
    - map (diff w apply?)
    - plot...
    
    
    Add in Geodesic:
    - get_length (or another name)
    
    """
        
    """
    def __init__(self, ):
    
        Input:
            gdss: a series (pandas.Series or butil.Series)
        
        #self.p0 = p0  # FIXME ***: probably not necessary as each gds has it; or can be inferred; add N and M?
        self.gdss = gdss
        
        #assert all([gdss[i].pids==gdss[0].pids for i in range(len(gdss))]),\
        #    "the geodesics do not agree on pids"
        #self.pids = gdss[0].pids
        
        #assert all([gdss[i].ptype==gdss[0].ptype for i in range(len(gdss))]),\
        #    "the geodesics do not agree on ptype"
        #self.ptype = gdss[0].ptype
    """

    # does not seem to work for append; but it works for butil.Series.append
    # don't know why...
    @property
    def _constructor(self): 
        return Geodesics
    
    #def __init__(self, data=None, index=None, **kwargs):
    #    super(Geodesics, self).__init__(data, index=index, **kwargs)
        

    @property    
    def ts(self):
        return self.apply(lambda gds: gds.t)
    
    @property
    def stops(self):
        return self.apply(lambda gds: gds.stop)
    
    @property
    def ncalls(self):
        return self.apply(lambda gds: gds.ncall)
    
    # p0s and v0s: return a DF
    # May not be homogeneous as the geodesics may not share pids
    @property
    def p0s(self):
        return self.apply(lambda gds: butil.Series(gds.p0, gds.pids))
    
    @property
    def v0s(self):  
        return self.apply(lambda gds: butil.Series(gds.v0, gds.pids))
    

    def get_eigvecs(self, idx_t=-1, idx_eigvec=-1):
        return self.apply(lambda gds: gds.get_eigvec(idx_t, idx_eigvec))
    
    
    @property
    def eigvecs(self):
        return self.apply(lambda gds: gds.get_eigvec(idx_t=-1, idx_eigvec=-1))
    
                  
    def integrate(self, print_key=True, **kwargs):
        """Calling signature is the same as Geodesic.integrate, pasted below:
        """
        for key, gds in self.items():
            if print_key:
                print key
            gds.integrate(**kwargs)
    integrate.__doc__ += Geodesic.integrate.__doc__
    
    """
    def plot_eigvts(self, idx_eigv=-1, nrow=None, ncol=None, cmapname=None, 
                    subplots_adjust=None, figtitle='', figsize=None,
                    show=True, filepath='', axs=None, **kwargs):
        if axs is None:
            if nrow is None and ncol is None:
                nrow, ncol = int(np.ceil(len(self.conds)/2)), 2 
            axidxs = list(itertools.product(range(nrow), range(ncol)))[:len(self.gdss)]
            axs = plotutil.get_axs(nrow, ncol, axidxs, figsize=figsize,
                                   subplots_adjust=subplots_adjust)
            plot_fig = True
        else:
            plot_fig = False

        for ax, gds, cond in zip(axs, self.gdss, self.conds):
            eigvt = np.linalg.svd(gds.Df(gds.p))[-1][idx_eigv]
            plotutil.barplot(ax=ax, lefts=np.arange(len(self.p0))-0.5, 
                             heights=eigvt, widths=1, cmapname=cmapname, 
                             xylims=[[-0.5,len(self.p0)-0.5],[-1,1]], xyticks=[[],[-1,0,1]],
                             title=cond)
        
        if plot_fig: 
            if figtitle:
                plotutil.plt.suptitle(figtitle, fontsize=16)
            if show:                                              
                plotutil.plt.show()
            if filepath:
                plotutil.plt.savefig(filepath)
            plotutil.plt.close()
    """
    
    
    def plot(self, vartypes=('_p','_y','_s','eigvec'), ncol=2,
             varlabels=(r'$p(t)$', r'$y(t)$', r'$\log_{10} \sigma(t)$', r'$v$'),
             figsize=(18,10), subplots_adjust={'wspace':0.8,'hspace':0.8,'right':0.95}, 
             figtitle='', show=True, filepath='', keymap=None, 
             **kwargs):
        """
        
        Overwrites pandas.Series.plot; if need to use that, cast it back to 
        the type. 
        
        Goals: 
        
        1. Layout:
                 major column 1                  major column 2
            vtype1  vtype2 ... vtypeN |     vtype1  vtype2 ... vtypeN
        key                           | key
        
        Eg, 
                            p   y  s  eigvec |                      p  y  s  eigvec
        ((m,m),2,(-1,True))                   ((m,a1),1,(-3,False))  
        
        2. Eigvecs have number indices 1, 2, 3...
        
        3. More space between major columns? Not quite necessary at the moment.
        
                
        Input: 
            vartypes: a tuple of strings, such that each element, when appended
                with 'traj', (eg, '_ptraj', 'ytraj') is an attribute of gds.
                Exceptions? eigvec...; if 'eigvec' is one of the vartypes,
                kwargs can provide 'idx_t' and 'idx_eigvec' to specify which
                eigenvector. 
            ncol: int, number of *major columns*; the geodesics are placed 
                row-wise, eg, if ncol==2,  
                gds1 gds2
                gds3 gds4
                            
             
        """
        
        #_vartype2label = dict(_p=r'$p$', _y=r'$y$', _s='$\log_{10}\sigma$',
        #                      eigvec='eigenvector')
        _vartype2cmapid = dict(_p='jet', _y='standard', _s='brg', eigvec='jet',
                                 p='jet', y='standard', s='brg')
        #labels = butil.get_values(_vartype2label, vartypes)
        #cmapnames = butil.get_values(_vartype2cmapname, vartypes)
        
        # some numbers for internal use
        _ngds = len(self)
        _nvartype = len(vartypes)
        _nrow = int(np.ceil(_ngds / 2))
        _ncol = ncol * _nvartype   
        # augmented column number; + (ncol-1) to add the middle dividers
        _ncol_aug = _ncol + (ncol - 1)
        
        # _colidxs are the indices of the columns that are used for plotting
        # (as opposed to the divider columns). 
        # The following three lines are more readable than the "one-liner"
        # that is equivalent but uses some tricky logic: 
        # _colidxs = butil.flatten(
        #    [np.arange((idx-1)*(_nvartype+1), idx*(_nvartype+1))[:-1] 
        #    for idx in range(1,ncol+1)])
        _colidxs = range(_ncol_aug)
        for colidx in _colidxs[_nvartype::_nvartype+1]:
            _colidxs.remove(colidx)
        
        axidxs = butil.get_product(range(_nrow), _colidxs)
        axs = plotutil.get_axs(_nrow, _ncol_aug, axidxs, figsize=figsize,
                               subplots_adjust=subplots_adjust)
        axmat = np.reshape(axs, (_nrow, _ncol))
        
        # idx_col is used to plot key on the left of the first column
        def _plot_vartype(vartype, axs_vartype, idx_col):   
            for ax, key, gds in zip(axs_vartype, self.keys(), self):
                if vartype == 'eigvec':
                    idx_t = kwargs.pop('idx_t', -1)
                    idx_eigvec = kwargs.pop('idx_eigvec', -1)
                    eigvec = gds.get_eigvec(idx_t=idx_t, idx_eigvec=idx_eigvec)
                    plotutil.barplot(ax=ax, lefts=np.arange(gds.pdim)+1, 
                                     heights=eigvec, widths=1, 
                                     cmapname=_vartype2cmapid[vartype], 
                                     xylims=[[0.5, gds.pdim+0.5],[-1,1]],
                                     xyticks=[np.arange(gds.pdim)+1,[-1,0,1]])
                    # The following four lines generate my own grid lines
                    # (the default grid lines are determined by the xticks
                    # and would go through the bars)
                    ax.grid(b=False)
                    ax.axhline(0, c='w', linewidth=1)
                    for x in np.arange(1, gds.pdim) + 0.5:
                        ax.axvline(x, c='w', linewidth=1)
                else:
                    traj = getattr(gds, vartype+'traj')
                    if vartype in ['_s', 's']:
                        traj = np.log10(traj)
                    traj.plot(ax=ax, colorscheme=_vartype2cmapid[vartype], **kwargs)
                
                # plot key  (also plot vartype based on idx_row? not quite necessary)
                if idx_col == 0:
                    if keymap is not None:
                        key = keymap(key)
                    ax.set_ylabel(key, fontsize=10)
                    ax.yaxis.set_label_coords(-1.3, 0.5)
        
        for idx, vartype in enumerate(vartypes):
            # ravel returns a view while flatten returns a copy
            axs_vartype = axmat[:, idx::_nvartype].ravel()
            _plot_vartype(vartype, axs_vartype, idx)
        
        if varlabels is not None:
            for idx_col, ax in enumerate(axmat[0]):
                ax.set_title(varlabels[idx_col % len(varlabels)])
            
        if figtitle:
            plotutil.plt.suptitle(figtitle, fontsize=13)
        if filepath:
            plotutil.plt.savefig(filepath)
        if show:  
            plotutil.plt.show()
        plotutil.plt.close()
            
    
    def save(self, filepath=''):
        pass
        
        

def gds_explore(pred, seed=0, filepath='', show=False, tmax=10):
    p0 = pred.p0.randomize(sigma=1, seed=seed)
    gdss = pred.get_geodesics(p0=p0, ptype='logp',
                              lam=0, inv='lam', atol=1e-2, rtol=1e-2, 
                              dt=1e-2, const_speed='p',
                              callback='singval', param_cb=1e-10)
    print 'Sloppiest singval: ', pred.get_in_logp().get_spectrum(p0.log())[-1]
    gdss.integrate(tmax=tmax, dt=0.1, print_cond=1, pass_exception=True, 
                   print_s0=False)
    gdss.plot(filepath=filepath, show=show)
    return gdss

