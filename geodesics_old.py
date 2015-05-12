import scipy
import scipy.linalg
from scipy.integrate import ode
from leastsq import leastsq

def AvvFD(x,v,func,args):
    return (func(x + 0.1*v,*args) + func(x - 0.1*v,*args) - 2.0*func(x,*args))/0.01

def geodesic_rhs_ode(t, xv, lam, dtd, func, jacobian, Avv, args, j, Acc): ## note t, xv need to be switched to use odeint/ode
    M,N = j.shape
    ans = scipy.empty(2*N)
    ans[:N] = xv[N:]
    j[:,:] = jacobian(xv[:N],*args)
    g = (scipy.dot(j.T,j) + lam*dtd)
    if Avv is not None:
        Acc[:] = Avv(xv[:N], xv[N:], *args)
    else:
        Acc[:] = AvvFD(xv[:N], xv[N:], func, args)
    ans[N:] = -scipy.linalg.solve(g,scipy.dot(j.T,Acc))
    return ans

def geodesic(x, v, tmax, func, jacobian, Avv, args = (), lam = 0, dtd = None,
             rtol = 1e-6, atol = 1e-6, maxsteps = 500, callback = None):

    N = len(x)
    y = scipy.empty((2*N,))
    y[:N] = x[:]
    y[N:] = v[:]

    if dtd is None:
        dtd = scipy.eye(N)
    
    j = jacobian(x,*args)
    M,N = j.shape
    Acc = scipy.empty((M,))
    
    r = ode(geodesic_rhs_ode,jac=None).set_f_params(lam, dtd, func, jacobian, Avv, args, j, Acc).set_integrator('vode',atol = atol, rtol=rtol).set_initial_value(y,0.0)

    steps = 0
    xs = []
    vs = []
    ts = []
    stop = False
    while r.successful() and steps < maxsteps and r.t < tmax and not(scipy.any(scipy.isnan(r.y))) and not stop:
        try:
            r.integrate(tmax,step = 1)
            xs.append(r.y[:N])
            vs.append(r.y[N:])
            ts.append(r.t)
            steps += 1
            if callback is not None:
                stop = callback(r.y[:N], r.y[N:], r.t, j, Acc, dtd)
        except:
            stop = True
    return scipy.array(xs), scipy.array(vs), scipy.array(ts)
