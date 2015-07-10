"""
"""

import numpy as np

from util import butil
Series, DF = butil.Series, butil.DF


def lmfit(self, p0=None, Dr=None, args=(), avegtol=1e-5, eps=_epsilon,
            maxnstep=None, ret_all=False, ret_steps=False, disp=True, lamb0=None,
	    Df0=None, trustradius=1.0):
	

	#full_output=0, disp=1, retall=0, lambdainit = None, 
        #      jinit = None, trustradius = 1.0):
    """Minimizer for a nonlinear least squares problem. Allowed to
    have more residuals than parameters or vice versa.

    f : residual function (function of parameters)
    fprime : derivative of residual function with respect to parameters.
             Should return a matrix (J) with dimensions  number of residuals
             by number of parameters.
    x0 : initial parameter set
    avegtol : convergence tolerance on the gradient vector
    epsilon : size of steps to use for finite differencing of f (if fprime
              not passed in)
    maxiter : maximum number of iterations 
    full_output : 0 to get only the minimum set of parameters back
                  1 if you also want the best parameter set, the 
                  lowest value of f, the number of function calls, 
                  the number of gradient calls, the convergence flag,
                  the last Marquardt parameter used (lambda), and the 
                  last evaluation of fprime (J matrix)
    disp : 0 for no display, 1 to give cost at each iteration and convergence
           conditions at the end
    retall : 0 for nothing extra to be returned, 1 for all the parameter 
             sets during the optimization to be returned 
    lambdainit : initial value of the Marquardt parameter to use (useful if
                 continuing from an old optimization run
    jinit : initial evaluation of the residual sensitivity matrix (J).
    trustradius : set this to the maximum move you want to allow in a single
                  parameter direction. 
                  If you are using log parameters, then setting this
                  to 1.0, for example, corresponds to a multiplicative 
                  change of exp(1) = 2.718
    """

    app_fprime = 0
    if fprime is None:
        app_fprime = 1

    xcopy = copy.copy(x0)
    if isinstance(x0,KeyedList) :
        x0 = asarray(x0.values())
    else :
        x0 = asarray(x0)

    if lamb0 is not None:
        lamb = lamb0
    else :
        lamb = 1.0e-2

    mult = 10.0

    n, m = self.np, self.nr

    nfcall = 0
    ngradcall = 0

    r = self((p0,)+args)

    if maxnstep is None :
        maxnstep = n * 200
    nstep = 0
    p = p0

    gtol = n * avegtol

    if ret_steps:
        ps = DF([p0])
	ps.index.name = 'step'

    p1 = p0
    p2 = p0

    d = zeros(n, scipy.float_)
    move = zeros(n, scipy.float_)

    convergence = False

    if Dr0 is not None:
        Dr = Dr0

    else:
        if app_fprime:
            j = asarray(apply(approx_fprime2,(x,f,epsilon)+args))
            func_calls = func_calls + 2*len(x)
        else :
            j = asarray(apply(fprime,(x,)+args))
            grad_calls+=1

    res = asarray(apply(f,(x,)+args))
    nfcall += 1
    # NOTE: Below is actually *half* the gradient (because
    # we define the cost as the sum of squares of residuals)
    # However the equations defining the optimization move, dp, 
    # are  2.0*J^tJ dp = -2.0*J^t r, where r is the residual
    # vector; therefore, the twos cancel both sides 
    grad = mat(res)*mat(j)

    while not convergence and nstep < maxnstep:

    # note: grad, res and j will be available from the end of the
    # last iteration. They just need to be computed the zeroth
    # time as well (above)

        lmh = mat(transpose(j))*mat(j)
        # use more accurate way to get e-vals/dirns
        #[u,s,v] = scipy.linalg.svd(lmh)
        [u,ssqrt,vt] = scipy.linalg.svd(j)
        # want n singular values even if m<n and we have
        # more parameters than data points.
        if (len(ssqrt) == n) :
            s = ssqrt**2
        elif (len(ssqrt)<n) :
            s = zeros((n,),scipy.float_)
            s[0:len(ssqrt)] = ssqrt**2
        #print "s is (in original) ", s
        #rhsvect = -mat(transpose(u))*mat(transpose(grad))

        rhsvect = -mat(vt)*mat(transpose(grad))
        rhsvect = asarray(rhsvect)[:,0]
        move = abs(rhsvect)/(s+Lambda*scipy.ones(n)+1.0e-30*scipy.ones(n))
        move = list(move)
        maxindex = move.index(max(move))
        move = asarray(move)

        if max(move) > trustradius :
            Lambda = Mult*(1.0/trustradius*abs(rhsvect[maxindex])-s[maxindex])
            #print " Increasing lambda to ", Lambda
        # now do the matrix inversion

        for i in range(0,n) :
            if (s[i]+Lambda) < 1.0e-30 :
                d[i] = 0.0
            else :
                d[i] = 1.0/(s[i]+Lambda)
            move[i] = d[i]*rhsvect[i]
        move = asarray(move)
        # move = asarray(mat(transpose(v))*mat(transpose(mat(move))))[:,0]
        move = asarray(mat(transpose(vt))*mat(transpose(mat(move))))[:,0]
        # print move
        p1 = p + deltap
        moveold = move[:]
        
        for i in range(0,n) :
            if (s[i]+Lambda/Mult) < 1.0e-30 :
                d[i] = 0.0
            else :
                d[i] = 1.0/(s[i]+Lambda/Mult)
            move[i] = d[i]*rhsvect[i]
        move = asarray(mat(transpose(vt))*mat(transpose(mat(move))))[:,0]

        p2 = p + deltap

        currentcost = sum(asarray(apply(f,(x,)+args))**2)
        func_calls+=1
        try:
            res2 = asarray(apply(f,(x2,)+args))
            costlambdasmaller = sum(res2**2)
        except SloppyCell.Utility.SloppyCellException:
            costlambdasmaller = scipy.inf
        func_calls+=1
        try:
            res1 = asarray(apply(f,(x1,)+args))
            costlambda = sum(res1**2)
        except SloppyCell.Utility.SloppyCellException:
            costlambda = scipy.inf
        func_calls+=1

        if disp :
            print 'Iteration number', niters
            print 'Current cost', currentcost
            print "Move 1 gives cost of" , costlambda
            print "Move 2 gives cost of ", costlambdasmaller
            #fp = open('LMoutfile','a')
            #fp.write('Iteration number ' + niters.__str__() + '\n')
            #fp.write('Current cost ' + currentcost.__str__() + '\n')
            #fp.write('Move 1 gives cost of ' + costlambda.__str__() + '\n')
            #fp.write('Move 2 gives cost of ' + costlambdasmaller.__str__() + '\n')
            #fp.close()

        oldcost = currentcost
        oldres = res
        oldjac = j

        if costlambdasmaller <= currentcost :
            Lambda = Lambda/Mult
            x = x2[:]
            if retall:
                allvecs.append(x)
            currentcost = costlambdasmaller
            if app_fprime :
                j = asarray(apply(approx_fprime2,(x2,f,epsilon)+args))
                func_calls = func_calls + 2*len(x2)
            else :
                j = asarray(apply(fprime,(x2,)+args))
                grad_calls+=1
            grad = mat(res2)*mat(j)
            if sum(abs(2.0*grad), axis=None) < gtol :
                finish = 2
        elif costlambda <= currentcost :
            currentcost = costlambda
            x = x1[:]
            move = moveold[:]
            if retall:
                allvecs.append(x)
            if app_fprime :
                j = asarray(apply(approx_fprime2,(x1,f,epsilon)+args))
                func_calls = func_calls + 2*len(x1)
            else :
                j = asarray(apply(fprime,(x1,)+args))
                grad_calls+=1

            grad = mat(res1)*mat(j)
            if sum(abs(2.0*grad), axis=None) < gtol :
                finish = 2
        else :
            Lambdamult = Lambda
            costmult = costlambda
            piOverFour = .78539816339744825
            NTrials = 0
            NTrials2 = 0
            move = moveold[:]
            while (costmult > currentcost) and (NTrials < 10) :
                num = -scipy.dot(grad,move)[0]
                den = scipy.linalg.norm(grad)*scipy.linalg.norm(move)
                gamma = scipy.arccos(num/den)
                NTrials = NTrials+1
                # was (gamma>piOverFour) below but that doens't
                # make much sense to me. I don't think you should
                # cut back on a given step, I think the trust
                # region strategy is more successful
                if (gamma > 0) :
                    Lambdamult = Lambdamult*Mult
                    for i in range(0,n) :
                        if s[i]+Lambdamult < 1.0e-30 :
                            d[i] = 0.0
                        else :
                            d[i] = 1.0/(s[i]+Lambdamult)
                        move[i] = d[i]*rhsvect[i]
                    move = asarray(mat(transpose(vt))*mat(transpose(mat(move))))[:,0]
                    x1 = x + move
                    res1 = asarray(apply(f,(x1,)+args))
                    func_calls+=1
                    costmult = sum(res1**2)

                else :
                    NTrials2 = 0
                    while (costmult > currentcost) and (NTrials2 < 10) :
                        NTrials2 = NTrials2 + 1
                        if disp == 1:
                            print " Decreasing stepsize "
                        move = (.5)**NTrials2*moveold
                        x1 = x + asarray(move)
                        res1 = asarray(apply(f,(x1,)+args))
                        func_calls+=1
                        costmult = sum(res1**2)

            if (NTrials==10) or (NTrials2==10) :
                if disp == 1:
                    print " Failed to converge"
                finish = 1
            else :
                x = x1[:]
                if retall:
                    allvecs.append(x)
                Lambda = Lambdamult
                if app_fprime :
                    j = asarray(apply(approx_fprime2,(x,f,epsilon)+args))
                    func_calls = func_calls + 2*len(x)
                else :
                    j = asarray(apply(fprime,(x,)+args))
                    grad_calls+=1

                grad = mat(res1)*mat(j)
                currentcost = costmult
                if sum(abs(2.0*grad), axis=None) < gtol :
                    finish = 2
        niters = niters + 1

        # see if we need to reduce the trust region
        newmodelval = oldres+asarray(mat(oldjac)*mat(transpose(mat(move))))[:,0]
        oldmodelval = oldres
        #print oldcost-sum(newmodelval**2)
        #print trustradius
        if ((oldcost-sum(newmodelval**2))>1.0e-16) :
            ratio = (oldcost-currentcost)/(oldcost-sum(newmodelval**2))
            if ratio < .25 :
                trustradius = trustradius/2.0
            if ratio >.25 and ratio<=.75 :
                trustradius = trustradius
            if ratio > .75 and trustradius<10.0 :
                trustradius = 2.0*trustradius

        #save(x,'currentParamsLM')

    if disp :
        if (niters>=maxiter) and (finish != 2) :
            print " Current function value: %f" % currentcost
            print " Iterations: %d" % niters
            print " Function evaluations: %d" % func_calls
            print " Gradient evaluations: %d" % grad_calls
            print " Maximum number of iterations exceeded with no convergence "
        if (finish == 2) :
            print " Optimization terminated successfully."
            print " Current function value: %f" % currentcost
            print " Iterations: %d" % niters
            print " Function evaluations: %d" % func_calls
            print " Gradient evaluations: %d" % grad_calls

    if isinstance(xcopy,KeyedList) :
        xcopy.update(x)
    else :
        xcopy = x

    if full_output:
        retlist = xcopy, currentcost, func_calls, grad_calls, finish, Lambda, j
        if retall:
            retlist += (allvecs,)
    else :
        retlist = xcopy
        if retall :
            retlist = (xcopy,allvecs)

    return retlist
