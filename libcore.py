#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
-----------------------------
Library of core functions for Mercury model computation.
"""

import numpy as np
import time
import scipy
import coreEos as eos
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from scipy.constants import G
from scipy.constants import R as RGas

def shoot_mercmodel(v,ricb,rhocr,rh,param,scale):
    
    """ 
    Shoots to find one solution from a set of initial conditions 
    specified in the vector v

    Here, the 5 unknowns are 
    1) P(r=0) 2) T(r=0) 3) rcmb  4) rhom 5) chiSicb

    The roots of the system are specified by 3 conditions:
        matching P, g at cmb, as well as the melting T at ICB.
        The success is measured in the 3-element vector f

    output: r= radial points of integration (non-dimensional)
         yy(:,1) = pressure vs radius (non-dimensional)
         yy(:,2) = g vs radius (non-dimensional)
         yy(:,3) = temperature vs radius (non-dimensional)
         yy(:,4) = adiabatic temperature vs radius (non-dimensional)
         yy(:,5) = density vs radius (non-dimensional)
         yy(:,6) = chiS vs radius (non-dimensional)
         fout(1) = P at icb (dimensional)
         fout(2) = T at cmb (dimensional)
         fout(3) = Cm/C
         fout(4) = C/MR^2
         fout(5) = xi
         fout(6) = k2
         fout(7) = isnow  (0,1,2 = no, layer, deep snow)
         fout(8) = isnowcmb  (0,1 = snow at CMB (no,yes))
         fout(9) = chiSin (initial sulfur content in core)
         fout(10)= gradTa (adiabatic temp gradient at CMB)

     shoot from set of conditions at r=small
     """


    # Mercury parameters
    M=param['GM']/G
    rm = param['rm']
    rhomean = 3*M/(4*np.pi*rm**3)

    # scales
    a=scale['a']
    ga=scale['ga']
    P=scale['P']
    T=scale['T']

    # limits of integration
    r0=0.0001/a
    rcmb=v[2]
    rhom=v[3]*rhomean

    # calculate Pcmb, gcmb
    rc=a*rcmb
    rhd=rh*a
    Pcmb,gcmb=getPgcmb_crust(rhom,rc,rhocr,rhd,param,scale)
    Pcmb=Pcmb/P
    gcmb=gcmb/ga
    CmC=param['CmC']
    CMR2=param['CMR2']

    # get gruneisan, bulk and density for P,T at r0 
    P1=P*v[0]
    T1=T*v[1]
    rho = eos.solidFccFe(P1/1E+9,T1,param)[1]
    #boundary values at r0 for integration
    gr0=4*np.pi*G*rho*(r0*a)/(3*ga)
    y0 = [v[0], gr0, v[1]]

    #Shoot In solid inner core 
    sol = scipy.integrate.solve_ivp(lambda t,y: rhs_PTrhog_solid_snow(t,y,ricb,scale,param),
                                    [r0, ricb], y0, method='LSODA',
                                    rtol=5e-5)
    rs = sol.t
    ys = sol.y
    ns=len(rs)
    
    # calculate melting temperature at ICB
    chiSicb=v[4]
    P1=P*ys[0,-1]
    Tmicb = getmelt_anzellini(chiSicb,P1,param,0)/T
    # boundary values for FOC integration: 
    # continuity of P, g, and T=Tm
    yicb = [ys[0,-1],ys[1,-1],Tmicb,Tmicb]

    # Shoot In Fluid core
    nc=51
    h=(rcmb-ricb)/(nc-1)   
    rc,yc,rhof,chiS = odeRK4_snow('rhs_fluid_snow',ricb,rcmb,h,yicb,chiSicb,
                                  scale,param)
    # Calculate moments of inertia:
    # First build polynomials of density
    rhos = np.empty(ns)
    if param['li_el'] == 'S':
        for i in range(0,ns):
            rhos[i]=eos.solidFccFe(P*ys[0,i]/1E+9,T*ys[2,i],param)[1]/rhomean
    elif param['li_el'] == 'Si':
        for i in range(0,ns):
            rhos[i]=eos.solidFccFeSi(chiS[0],P*ys[0,i]/1E+9,T*ys[2,i],param)[1]/rhomean
    sols=np.polyfit(rs,rhos,3)

    for i in range(0,nc):
        rhof[i]=rhof[i]/rhomean

    solf=np.polyfit(rc,rhof,3)
    
    # ... then multiply by r4 and integrate
    rhor4=np.convolve(sols,[1,0,0,0,0])
    bigIo=np.polyval(np.polyint(rhor4),ricb)
    rhor4=np.convolve(solf,[1,0,0,0,0])
    bigIo=bigIo + np.polyval(np.polyint(rhor4),rcmb)\
          -np.polyval(np.polyint(rhor4),ricb)


    bigCm = 0.2*rhocr*(1-rh**5)/rhomean + 0.2*rhom*(rh**5-rcmb**5)/rhomean
    bigIo = bigIo + bigCm   # To get dimensional moment of inertia, * 8pi/3*rhomean*a^5
    CmCtry=bigCm/bigIo #Cm/C 
    CMR2try=2*bigIo #C/MR^2 


    chisrc2=chiS*rc**2
    chiSin = 3/(rcmb**3)*simpsonDat(rc,chisrc2)
    
    # snow state
    isnow=0  # snow index: default is no snow
    
    if ((chiS[-1]-chiSicb) > 1e-10):
        isnow=1
        # get dimensional P and T at second point in FOC
        i=0
        T1=T*yc[i,2]
        P1=P*yc[i,0]
        Tm=getmelt_anzellini(chiS[i],P1,param,0)  # get Tliquidus

        if abs(T1-Tm)<1e-8: # if adiabat temperature = Liquidus 
            isnow=2    
  
    isnowcmb=0;  # snow at cmb index: default no
    if isnow==1 or isnow==2:
        i=-1
        # get dimensional P and T
        T1=T*yc[i,2]
        P1=P*yc[i,0]
        Tm=getmelt_anzellini(chiS[i],P1,param,0)  # get Tliquidus

        if abs(T1-Tm)<1e-6:  #if adiabat temperature = Liquidus
            isnowcmb=1
    
    # Calculate k2 and xi
    rhoml=rhom/rhomean

    k2,xi=getk2(ricb,rcmb,rhoml,sols,solf,param,scale)

        
    # adiabatic temp gradient at CMB
    gradTa=T/a*(yc[nc-1,3]-yc[nc-2,3])/(rc[nc-1]-rc[nc-2])
    fout=[P*ys[0,-1],T*yc[nc-1,2],isnow,isnowcmb,chiSin,gradTa]

    # include the (1+xi) factor on Cm/C
    CmCtry = CmCtry*(1+xi)
    
    # function to minimize (roots)
    f=[yc[-1,0]-Pcmb,  # match P at cmb
       yc[-1,1]-gcmb,  # match g at cmb
       ys[2,-1]-Tmicb, # match Tm at icb
       CmCtry-CmC,
       CMR2try-CMR2]

    # concatenate solution
    r=np.concatenate((rs,rc))
    ytemp=np.vstack((ys,ys[2]))
    y = np.hstack((ytemp,np.transpose(yc)))
    rho = np.concatenate((rhos,rhof))
    chi = np.concatenate((np.zeros(ns),chiS))
    if param['li_el']=='Si': 
        chi[0:ns] = chiS[0]
    yy=np.vstack((y[0],y[1],y[2],y[3],rho,chi))
    
    return f,r,yy,fout

def J_mercmodel(v,args):
    """
      computes the Jacobian and function evaluation for our 
      interior model system
    
     input vinit = variable vinit (5 element vector)
    
     output f = function evaluation (5 function)
            J = Jacobian matrix of derivatives
    """

    #initialize
    ricb,rhocr,rh,param,scale = args
    n=len(v)
    f = np.zeros(n) # f must be defined as a column vector
    f2 = np.zeros(n) # f must be defined as a column vector
    J = np.zeros((n,n))  
    
    # compute the function f, 
    f=shoot_mercmodel(v,ricb,rhocr,rh,param,scale)[0]

    eps=1.e-6
    for j in range(n):
        temp=v[j]
        h=eps*abs(temp)
        if (h==0):
            h=eps
        v[j]=temp+h
        h=v[j]-temp
        f2=shoot_mercmodel(v,ricb,rhocr,rh,param,scale)[0]
        v[j]=temp
        for i in range(n):
            J[i,j]=(f2[i]-f[i])/h

    return J,f

def mynewtonSys(Jfun,x0,varargin,
                xtol=5e-5,ftol=5e-5,maxit=15,verbose=False):
    """
     newtonSys  Newton's method for systems of nonlinear equations.
    
     Synopsis:  x = newtonSys(Jfun,x0)
                x = newtonSys(Jfun,x0,xtol)
                x = newtonSys(Jfun,x0,xtol,ftol)
                x = newtonSys(Jfun,x0,xtol,ftol,maxit,verbose)
                x = newtonSys(Jfun,x0,xtol,ftol,maxit,verbose,arg1,arg2,...)
    
     Input:  Jfun = (string) name of mfile that returns matrix J and vector f
             x0   = initial guess at solution vector, x
             xtol = (optional) tolerance on norm(dx).  Default: xtol=5e-5
             ftol = (optional) tolerance on norm(f).   Default: ftol=5e-5
    
             verbose = (optional) flag.  Default: verbose=0, no printing.
             arg1,arg2,... = (optional) optional arguments that are passed                   
             through to the mfile defined by the 'Jfun' argument
    
     Note:  Use [] to request default value of an optional input.  For example,
            x = newtonSys('JFun',x0,[],[],[],arg1,arg2) passes arg1 and arg2 to
            'JFun', while using the default values for xtol, ftol, and verbose
    
     Output:  x = solution vector;  x is returned after k iterations if 
                  tolerances are met, or after maxit iterations if 
                  tolerances are not met.
    """

    
    xeps = xtol
    feps = ftol #   %  Smallest tols are 5*eps

    if verbose:
        print('\nNewton iterations\n  k     norm(f)      norm(dx)    time(s)\n')

    x = x0
    k = 0        #  Initial guess and current number of iterations
    
    
    while k <= maxit:
      start = time.time()
      k = k + 1
      J,f = eval(Jfun)(x,varargin)   #   Returns Jacobian matrix and f vector
      dx = np.dot(np.linalg.inv(J),f)
      x = x - dx
      if verbose:     
          end = time.time()
          print(k,np.linalg.norm(f),np.linalg.norm(dx),end-start)
      if (np.linalg.norm(f) < feps) or (np.linalg.norm(dx) < xeps):  
          return x

    print('Solution not found within tolerance after %d iterations\n',k)
    return None



def odeRK4_snow(diffeq,ricb,rcmb,h,y0,chiSicb,scale,param):
    """
    
     odeRK4_snow: ode solver for our system of equations  
     modified from NMM, odeRK4sysv.  Customization is such that it is possible
     to track changes of chiS vs radius as well as integration of other
     variables.
    
    
     odeRK4sysv  Fourth order Runge-Kutta method for systems of first order ODEs
                 Vectorized version with pass-through parameters.
    
     Input:     diffeq = (string) name of the m-file that evaluates the right
                          hand side of the ODE system written in standard
                          form.
                ricb,rcmb = icb,cmb radius
                h       = stepsize for advancing the independent variable
                y0      = vector of the dependent variable values at icb
                chiSicb = chiS at icb
    
     Output:    r = vector of independent variable values:  r(j) = ricb + j*h
                y = matrix of dependent variables values, one column for each
                    state variable.  Each row is from a different time step.
                rhof = density at each radius
                chiS = Sulfur concentartion at each radius
    """

    r = np.arange(ricb,rcmb+h/2,h)#  Column vector of elements with spacing h
    nt = len(r)                             #  number of steps (+1 for the initial conditions)
    neq = len(y0)                           #  number of equations simultaneously advanced
    y = np.zeros((nt,neq))                  #  Preallocate y for speed
    y[0,:] = y0                             #  Assign IC. y0(:) is column, y0(:)' is row vector
    rhof=np.zeros(nt)
    chiS=np.zeros(nt)
    
    #  Avoid repeated evaluation of constants    
    h2 = h/2
    h3 = h/3
    h6 = h/6   
    k1 = np.zeros(neq)
    k2 = k1
    # Preallocate memory for the Runge-Kutta
    k3 = k1  
    k4 = k1
    # coefficients and a temporary vector
    ytemp = k1  
    
    # Outer loop for all steps:  j = time step index;  k = equation number index
    # Note use of transpose on definition of yold, and in formula for y(j,:) 
    
    res = getchiSgrun(y0[2],y0[0],chiSicb,scale,param)
    chiS[0],rhof[0] = res[0:2]

    for j in range(1,nt): 
        rold = r[j-1]        
        yold = y[j-1,:]
        chiSold=chiS[j-1]       #  Temp variables
        chiStemp,rhoftemp,grun,KS=getchiSgrun(yold[2],yold[0],chiSold,scale,param)
        k1 = eval(diffeq + '(rold,yold,ricb,rcmb,rhoftemp,grun,KS,scale)') #  Slopes at the start
        k1 = np.array(k1)
        ytemp = yold + h2*k1
        
        chiStemp,rhoftemp,grun,KS=getchiSgrun(ytemp[2],ytemp[0],chiSold,scale,param)
        k2 = eval(diffeq + '(rold+h2,ytemp,ricb,rcmb,rhoftemp,grun,KS,scale)') # 1st slope at midpoint
        k2 = np.array(k2)
        
        ytemp = yold + h2*k2
        chiStemp,rhoftemp,grun,KS=getchiSgrun(ytemp[2],ytemp[0],chiSold,scale,param)
        k3 = eval(diffeq + '(rold+h2,ytemp,ricb,rcmb,rhoftemp,grun,KS,scale)') #  2nd slope at midpoint
        k3 = np.array(k3)
        
        ytemp = yold + h*k3
        chiStemp,rhoftemp,grun,KS=getchiSgrun(ytemp[2],ytemp[0],chiSold,scale,param)
        k4 = eval(diffeq + '(rold+h,ytemp,ricb,rcmb,rhoftemp,grun,KS,scale)')  #  Slope at endpoint
        k4 = np.array(k4)
        
        y[j,:] = ( yold + h6*(k1+k4) + h3*(k2+k3) )  #  Advance all equations
        res = getchiSgrun(y[j,2],y[j,0],chiSold,scale,param)
        chiS[j],rhof[j] = res[0:2]

    return r,y,rhof,chiS
    
def rhs_PTrhog_solid_snow(r,y,ricb,scale,param):
    """
    rhs_PTrhog  Right-hand sides of coupled ODEs for interior model equations

    Input:    r      = radius, the independent variable 
              y      = vector (length 3) of dependent variables
              ricb   = ICB radius

    Output:   dydr = column vector of dy(i)/dr values
    """

    # scales
    a=scale['a']
    ga=scale['ga']
    P=scale['P']
    T=scale['T']
    
    # get dimensional P and T
    T1=T*y[2]
    P1=P*y[0]

    out = eos.solidFccFe(P1/1E+9,T1,param)
    rho = out[1]
    grun = out[6]
    KS = out[4]*1e+9
    dydr = [-(a*ga/P)*rho*y[1],
            (a/ga)*4*np.pi*G*rho-2*y[1]/r,
            -(a*ga)*grun*rho*y[1]*y[2]/KS]
    
    return dydr

def getPgcmb_crust(rhom,rc,rhocr,rh,param,scale):
    """
    determines the Pressure and grav acc at cmb for a given choice of 
    mantle density (rhom), cmb radius (rcmb)
    crustal density (rhocr), crust-mantle boundary radius (rh)
    """
    # Mercury parameters
    GM=param['GM']
    M=GM/G
    rm = param['rm']

    # scales
    a=scale['a']
    P=scale['P']

    # mass of crust
    Mh=4*np.pi*rhocr*(rm**3-rh**3)/3
    # mass of mantle
    Mm=4*np.pi*rhom*(rh**3-rc**3)/3
    # mass of core
    Mcore=M-Mm-Mh
    # grav acc at CMB
    gcmb=Mcore*G/rc**2

    # Pressure at CMB: Shoot In crust + mantle 
    y0 = [0,1]
    sol = scipy.integrate.solve_ivp(lambda t,y: rhs_Pgz(t,y,rhocr,scale), [1,rh/a],
                                    y0, method='RK45')

    yh = sol.y[:,-1]

    sol = scipy.integrate.solve_ivp(lambda t,y: rhs_Pgz(t,y,rhom,scale), 
                                    [rh/a,rc/a],
                                    yh, method='RK45')

    Pcmb=P*sol.y[0,-1] # dimensional

    return Pcmb,gcmb

def getmelt_anzellini(chis,P,param,To):
    """
    Determines melting temperature of FeS mixture as a function 
    of chis and P.  
    Here, the melting point of pure Fe is determined according to Anzellini,
    Science 2013
    
    From Eq 2 of Anzellini et al 2013
    but reformulated as a 3rd order polynomial
    """

    el = param['li_el']
    
    if el == 'S':
        P1=P*1e-9;  
        # parametrization for Anzellini
        TmFe= 495.4969600595926*(22.19 + P1)**0.42016806722689076
        
        if P1 < 14:
            Te0=1265.4
            b1=-11.15
            Pe0=3
        elif P1 < 21:   
            Te0=1142.7
            b1=29
            Pe0=14
        else:    
            Te0=1345.72
            b1=12.9975
            Pe0=21
            
        Te=Te0+b1*(P1-Pe0)
        chiSeut=0.11+0.187*np.exp(-0.065*P1)
        
        Tm = TmFe -(TmFe - Te)*chis/chiSeut
        return Tm-To
    
    elif el=='Si':
        P1=P*1e-9;  
        # parametrization for Anzellini
        TmFe= 495.4969600595926*(22.19 + P1)**0.42016806722689076
        Tm15 = 1478 *(P1/10+1)**(1/3)        
        Tm =(chis/0.15)*Tm15+(1-chis/0.15)*TmFe
        return Tm-To
    
    else: 
        print('Error: Light element',el,' not defined!')
        return 0


def getchiSgrun(yT,yP,chiSold,scale,param):
    # scales
    P=scale['P']
    T=scale['T']  
    el = param['li_el']
    

    # get dimensional P and T
    T1=T*yT
    P1=P*yP
    # chiSold
    Tm=getmelt_anzellini(chiSold,P1,param,0)

    if T1>Tm: # if adiabat temperature larger than Liquidus
        # get chiS on basis of previous radius         
        chiS=chiSold
    else:
        # get chiS that matches melting
        chiSeut=0.11+0.187*np.exp(-0.065*P1*1e-9)
        sol = scipy.optimize.root(getmelt_anzellini, chiSold, 
                                  tol=1e-6, args=(P1, param, T1))          
        chiS = min(sol.x[0],chiSeut)    

    # Updated Equation of State from Rivoldini
    if el == 'S': 
        out = eos.liquidNonIdalFeS(chiS,P1/1E+9,T1,param)
    elif el == 'Si':
        out = eos.liquidNonIdalFeSi(chiS,P1/1E+9,T1,param)
    else:
        print('WARNING: invalid light element')
    rho = out[1]
    KS = out[4]*1E+9
    grun = out[6]
    return chiS,rho,grun,KS

def rhs_fluid_snow(r,y,ricb,rcmb,rho,grun,KS,scale):
    """
    # Right-hand sides of coupled ODEs for interior model equations
    # This version includes a stratified layer at CMB
    # Here, we also track the adiabatic Temperature
    # THIS VERSION to be used with odeRK4_snow.m
    
    # Input:    r      = radius, the independent variable 
    #           y      = vector (length 4) of dependent variables
    #           ricb   = ICB radius
    #           rcmb   = CMB radius
    #           rho    = density
    #           grun   = gruneisan
    #           KS     = adiabatic bulk modulus
    #
    # Output:   dydr = column vector of dy(i)/dr values
    """
    
    # scales
    a=scale['a']
    ga=scale['ga']
    P=scale['P']

    # Size of thermally stratifued layer
    # Default rst=ricb + (rcmb-ricb)/2
    rst = ricb + (rcmb-ricb)/2

    if r<rst: 
        dydr = [ -(a*ga/P)*rho*y[1],
                (a/ga)*4*np.pi*G*rho-2*y[1]/r,
                -(a*ga)*grun*rho*y[1]*y[2]/KS,
                -(a*ga)*grun*rho*y[1]*y[2]/KS]
    else:
        dydr = [-(a*ga/P)*rho*y[1],
                (a/ga)*4*np.pi*G*rho-2*y[1]/r,
                -(a*ga)*grun*rho*y[1]*y[2]*(1 -0.95*(r-rst)/(rcmb-rst))/KS,
                -(a*ga)*grun*rho*y[1]*y[2]/KS]
        
    return dydr

def simpsonDat(x,f):
    """
    simpsonDat  Integration by Composite Simpson's rule
                adapted from nmm package, here for a function f evaluated at
                equally spaced points x

    Synopsis:  I = simpson(fun,a,b,npanel)

    Input:     x = equally spaced points (number of points n must be odd)
               f = integrand at these x points
    Output:    I = approximate value of the integral from x(1) to x(n) of f(x)*dx
    
    """

    h=x[1]-x[0]
    I = (h/3)*(f[0]+4*np.sum(f[1::2]) + 2*np.sum((f[2::2])[0:-1]) + f[-1])
    return I

def CvC(theta_T):
    # heat capacity at constant volume, T and theta in K
    f=3*RGas*(4*debye3(theta_T)-theta_T*3/(np.exp(theta_T)-1))
    return f


def debye3(x, maxdeg=7): # truncated to save computation time
    """
    %DEBYE3 ThirdF order Debye function.
    %   Y = DEBYE3(X) returns the third order Debye function, evaluated at X.
    %   X is a scalar.  For positive X, this is defined as
    %
    %      (3/x^3) * integral from 0 to x of (t^3/(exp(t)-1)) dt
    
    %   Based on the FORTRAN implementation of this function available in the
    %   MISCFUN Package written by Alan MacLead, available as TOMS Algorithm 757
    %   in ACM Transactions of Mathematical Software (1996), 22(3):288-301.
    """

    adeb3=[2.70773706832744094526,
           0.34006813521109175100,
           -0.1294515018444086863e-1,
           0.79637553801738164e-3,
           -0.5463600095908238e-4,
           0.392430195988049e-5,
           -0.28940328235386e-6,
           0.2173176139625e-7,
           -0.165420999498e-8,
           0.12727961892e-9,
           -0.987963459e-11,
           0.77250740e-12,
           -0.6077972e-13,
           0.480759e-14,
           -0.38204e-15,
           0.3048e-16,
           -0.244e-17,
           0.20e-18,
           -0.2e-19]

    if x < 0: 
        print('error in debye3: negative input value')
        return 0
    elif (x < 3.e-8): 
        print('low input value in debye3: check input value')
        D3 = ((x - 7.5 ) * x + 20.0)/20.0
    elif x <= 4:
        # routine only accurate within these limits
        # but should be OK for typical x values in our
        # models

        t = ((x**2/8)-0.5)-0.5
        D3 = cheval(adeb3[0:maxdeg],t)-0.375*x
    else:
        print('error in debye3: input value should be smaller than 4');
        return 0

    return D3

def cheval(a, t):
    """
    CHEVAL evaluates a Chebyshev series.
    modified by MD for Matlab 
    
      Discussion:
    
        This function evaluates a Chebyshev series, using the
        Clenshaw method with Reinsch modification, as analysed
        in the paper by Oliver.
    
      Author:
    
        Allan McLeod,
        Department of Mathematics and Statistics,
        Paisley University, High Street, Paisley, Scotland, PA12BE
        macl_ms0@paisley.ac.uk
    
      Reference:
    
        Allan McLeod,
        Algorithm 757, MISCFUN: A software package to compute uncommon
          special functions,
        ACM Transactions on Mathematical Software,
        Volume 22, Number 3, September 1996, pages 288-301.
    
        J Oliver,
        An error analysis of the modified Clenshaw method for
        evaluating Chebyshev and Fourier series,
        Journal of the IMA,
        Volume 20, 1977, pages 379-391.
    
      Parameters:
    
        Input, A(1:N), the coefficients of the Chebyshev series.
    
        Input,  T, the value at which the series is
        to be evaluated.
    
        Output, CHEV, the value of the Chebyshev series at T.
    
    """
    n=len(a)-1
    u1 = 0.0
    #  T <= -0.6, Reinsch modification.
    # -0.6 < T < 0.6, Standard Clenshaw method.
    #  T > 0.6 Reinsch
    if t <= -0.6 or t>=0.6:
        d1 = 0.0
        tt = (t+0.5) + 0.5
        tt = tt+tt
        for i in range(n,-1,-1):
            d2 = d1
            u2 = u1
            d1 = tt * u2 + a[i] - d2
            u1 = d1 - u2
        chev = 0.5*( d1-d2 )


    else:
        u0 = 0.0
        tt = t + t
        for i in range(n,-1,-1):
            u2 = u1
            u1 = u0
            u0 = tt * u1 + a[i] - u2

        chev = 0.5*( u0 - u2 ) 

    return chev

def Eth(T,theta):
    #internal thermal energy (without not relevant term linear in theta),
    # T and theta in K
    RGas=8.3144621
    f=3.*RGas*(3*theta/8 + T*debye3(theta/T))
    
    return f

def gammaC(eta,gamma0,q0):
    # Grueneisen parameter
    # eta=V0/V
    return gamma0*eta**(-q0)

def thetaC(eta,theta0,gamma0,q0):
    # Debye temperature, eta=V0/V
    return theta0*np.exp((gamma0-gammaC(eta,gamma0,q0))/q0)


def rhs_Pgz(r, y, rho, scale):
    """
    rhs_Pgz  Right-hand sides of coupled ODEs for hydrostatic pressure


    Input:    z      = depth, the independent variable 
              y      = vector (length 2) of dependent variables
              rho    = density (=constant)

    Output:   dydr = column vector of dy(i)/dz values
    """
    # scales
    a=scale['a']
    ga=scale['ga']
    P=scale['P']

    dydr = [-(a*ga/P)*rho*y[1],
            (a/ga)*4*np.pi*G*rho-2*y[1]/r]
    
    return dydr


#### The k2 stuff ####
    
def getk2(rs,rf,rhoml,rhos,rhof,param,scale):
    """
    %
    % Calculates k2, xi = [(Bs-As) - (Bs'-As')]/ (Bmf -Amf)
    % for given density structure of mercury and c22
    %
    %
    """
    # Mercury parameters
    M=param['GM']/G
    rm = param['rm']
    rhomean = 3*M/(4*np.pi*rm**3)
    c22=param['c22']

    # scales
    a=scale['a']
    ga=scale['ga']

    bigGscale=ga/(rhomean*a)
    bigGnd=G/bigGscale

    # Define radial grid points
    nr=400
    nrs=int(round(nr*(rs/rf)))
    nrf=nr-nrs
    
    r = np.empty(nr)
    rho = np.empty(nr)
    g = np.empty(nr)
    
    for k in range(nrs):
        r[k]=rs*(k+1)/nrs


    for k in range(nrf):
        r[k+nrs]=rs+(rf-rs)*k/nrf
    
    drf=(rf-rs)/nrf

    # Build density (rho) and gravitational acceleration (g)
    f4piG=4*np.pi*bigGnd/3

    #  in inner core
    avgr=0.5*r[0]
    rho[0]=np.polyval(rhos, avgr)
    g[0]=f4piG*rho[0]*r[0]
  
    for k in range(1,nrs):
        avgr=0.5*(r[k]+r[k-1])
        rho[k]=np.polyval(rhos, avgr)
        g[k]= f4piG*rho[k]*(r[k]**3-r[k-1]**3)/r[k]**2 +  g[k-1]*(r[k-1]/r[k])**2
    
    #  in fluid core
    avgr=r[nrs]+0.5*drf
    rho[1+nrs]=np.polyval(rhof, avgr)
    g[1+nrs]=f4piG*rho[1+nrs]*(r[1+nrs]**3-r[nrs]**3)/r[1+nrs]**2\
             +g[nrs]*(r[nrs]/r[1+nrs])**2
             
    for k in range(0,nrf):
        avgr=0.5*(r[k+nrs]+r[k+nrs-1])
        rho[k+nrs]=np.polyval(rhof, avgr)
        g[k+nrs]=f4piG*rho[k+nrs]*(r[k+nrs]**3-r[k+nrs-1]**3)/r[k+nrs]**2\
                 +g[k+nrs-1]*(r[k+nrs-1]/r[k+nrs])**2

    # solution
    pot = getpotvsr(nr,bigGnd,r,rho,g)
    k2 = pot[-1]-1;
    
    # get ellipticity
    fell=1.0
    drhocmb = rho[-1]-rhoml
    ell = c22*10.0/ (rhoml*(1 + k2*rf**5)+ fell*(1+k2)*drhocmb*rf**5)
    # calculate Bs-As
    factrho=rhoml+fell*drhocmb
    BsAs=0

    for k in range(nrs):
        BsAs=BsAs+(4*np.pi/5)*bigGnd*rf*rf*(rho[k]-rho[k+1])*(r[k]**4)*pot[k]/g[k]

    BsAs=BsAs*(8*np.pi/15)*ell*factrho

    # calculate Bmf-Amf
    sum3=0
    for k in range(nrs,nr-1):
        sum3=sum3+(4*np.pi/5)*bigGnd*rf*rf*(rho[k]-rho[k+1])*(r[k]**4)*pot[k]/g[k]

    ffrho=rhoml + fell*drhocmb*rf**5
    BmAm=(8*np.pi/15)*ell*(ffrho+sum3*factrho)

    # calculate xi
    xi=BsAs/BmAm

    return k2,xi

def getpotvsr(nr,bigGnd,rnd,rhond,gnd):
    # This function calculates the total potential vs radius in core

    l=2 # %spherical harmonic degree

    # build matrix A (sparse) element by element
    # specifying row, column and numerical value of all non-zero

    ndim=2*nr-1
    k=0
  
    row = np.zeros(3191, dtype=int)
    col = np.zeros(3191, dtype=int)
    s = np.zeros(3191)
    
    kk=0
    row[kk]=k
    col[kk]=k
    s[kk]=rnd[0]**(2*l+1)
  
    kk=kk+1
    row[kk]=k
    col[kk]=k+1
    s[kk]=-1
    
    kk=kk+1
    row[kk]=k
    col[kk]=k+2
    s[kk]=-rnd[0]**(2*l+1)

    alpha = 4.0*np.pi*bigGnd*rnd[0]*(rhond[0]-rhond[1])/gnd[0]

    kk=kk+1
    row[kk]=k+1
    col[kk]=k
    s[kk]=(l - alpha)*rnd[0]**(2*l+1)

    kk=kk+1
    row[kk]=k+1
    col[kk]=k+1
    s[kk]=l+1

    kk=kk+1
    row[kk]=k+1
    col[kk]=k+2
    s[kk]=-l*rnd[0]**(2*l+1)

    for j in range(1,nr-1):
    
        k=2*(j+1)-2
        
        kk=kk+1
        row[kk]=k
        col[kk]=k
        s[kk]=rnd[j]**(2*l+1)

        kk=kk+1
        row[kk]=k
        col[kk]=k-1
        s[kk]=1

        kk=kk+1
        row[kk]=k
        col[kk]=k+1
        s[kk]=-1
    
        kk=kk+1
        row[kk]=k
        col[kk]=k+2
        s[kk]=-rnd[j]**(2*l+1)
        
        alpha = 4.0*np.pi*bigGnd*rnd[j]*(rhond[j]-rhond[j+1])/gnd[j]

        kk=kk+1
        row[kk]=k+1
        col[kk]=k-1
        s[kk]=-(l+1 + alpha)
    
        kk=kk+1
        row[kk]=k+1
        col[kk]=k
        s[kk]=(l - alpha)*rnd[j]**(2*l+1)
    
        kk=kk+1
        row[kk]=k+1
        col[kk]=k+1
        s[kk]=l+1
    
        kk=kk+1
        row[kk]=k+1
        col[kk]=k+2
        s[kk]=-l*rnd[j]**(2*l+1)

    k=2*nr-2
    kk=kk+1
    row[kk]=k
    col[kk]=k
    s[kk]=rnd[nr-1]**l

    # build sparse matrix A
    A=np.zeros([ndim, ndim])
    A[row,col] = s

    #A = np.array(row,col,s,ndim,ndim)

    rhs=np.zeros(2*nr-1)
    rhs[2*nr-2]=1.0

    A = csc_matrix(A)
    b = inv(A)*rhs

    # solution
    pot = np.empty(nr)
    pot[0]=b[0]*rnd[0]**l
    
    for k in range(1,nr):
        kk=2*(k+1)-2
        pot[k]=b[kk]*rnd[k]**l + b[kk-1]*rnd[k]**(-l-1)

    return pot