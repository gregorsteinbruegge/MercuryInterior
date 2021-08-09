#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__authors__ = ['Gregor Steinbrügge (Stanford University), gbs@stanford.edu',
               'Mathieu Dumberry',
               'Attilio Rivoldini']     
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'February 02, 2021',
         'author': 'Gregor Steinbrügge, DLR',
         'info': 'Initial Release.'}}

Desciption:
------------------------------------------------------------
Constructs interior model of Mercury. 
Published under MIT license but please cite Steinbrügge et al. 2020.
https://doi.org/10.1029/2020GL089895.

We constrain models to satisfy mean density, Cm/C, and C/Mr^2.
We impose that Tm = Ticb, with an iron snow scenario.
The variables to solve for are:
1) Pressure at r=0
2) Temperature at r=0
3) Core-Mantle boundary radius
4) Mantle density
5) Light element content (S or Si)
"""

import numpy as np
import libcore as lc
import visualization as vis
import coreEos as eos
from scipy.constants import G


# Hard coded constants
MFeS = (55.845+32.065)
MFeSi = (55.845+28.08)
MFe = 55.845

# Initialize the EoS for faster computation
fccFe=eos.eosAndersonGrueneisen(M0=MFe,p0=1.e-5,T0=298,V0=6.82,
                            alpha0=7.e-5,KT0=163.4,KTP0=5.38,
                            deltaT=5.5,kappa=1.4,GibbsE=eos.GibbsfccFe)        
        
liquidFe=eos.eosAndersonGrueneisen(M0=MFe,p0=1E-5,T0=298,V0=6.88,
                               alpha0=9E-5,KT0=148,KTP0=5.8,deltaT=5.1,
                               kappa=0.56,GibbsE=eos.GibbsLiquidFe)

liquidFeS=eos.eosAndersonGrueneisen(M0=MFeS,p0=1E-5,T0=1650,
                                V0=22.956500240757844,alpha0=11.9e-5,
                                KT0=17.01901122392699,
                                KTP0=5.92217679116356,
                                deltaT=5.92217679116356,kappa=1.4,
                                gamma0=1.3,q=0)

liquidFeSi=eos.eosAndersonGrueneisen(M0=MFeSi,p0=1E-5,T0=1723,
                                 V0=16.5839,alpha0=17.6525e-5,KT0=69.0074,
                                 KTP0=7.76007,deltaT=4.07505,kappa=0.56,
                                 gamma0=1.61986,q=0)

# =================================================
# Define the parameter set to be used for the model
# =================================================
param = {'rm':2439360.0,
         'GM':22031.86E+9,
         'c22':0.804151E-05,
         'CmC':0.148/0.333, #<----- Cm/C constraint
         'CMR2':0.333, # <---- MoI constraint
         'MFe':MFe,
         'MS':32.065,
         'MSi':28.08,
         'MFeS':MFeS,
         'MFeSi':(55.845+28.08),
         'lFe':liquidFe,
         'lFeS':liquidFeS,
         'lFeSi':liquidFeSi,
         'fccFe':fccFe,
         'li_el':'S', #<---- Switch to 'Si' for Silicon cores
         'name':'replace_me'}


M = param['GM']/G
rm = param['rm']
#Average density
rhomean = 3*M/(4*np.pi*rm**3)
# scales
scale = {'a':rm,
         'ga':M*G/rm**2,
         'P':rhomean*rm*M*G/rm**2,
         'T':1800}

# build model from shooting method, using multi-directional Newton method 
# for each iteration
xtol=1.e-5
ftol=1.e-5
maxit=6

# crustal density and thickness
rhocr=2974
hcr=26e3
# radius of crust-mantle boundary
rh=(rm-hcr)/scale['a']
# initial guesses
v0=[0.8,1.0,0.8,0.7,0.05] 
# Parameterize by inner core radius
rs=np.arange(1e1, 2e6, 10e+3)
# non-dimensional
ricb=rs/scale['a']
  
# Solve the system 
for k in range(len(rs)):
    try:
        v = lc.mynewtonSys('J_mercmodel',v0,
                        [ricb[k],rhocr,rh,param,scale],
                        xtol=xtol, ftol=ftol, maxit=maxit, verbose=True)
    except:
        print('Aborted Newton run. Note that this happens as well if no solution exists.')
        break

    v0=v # initial guess for next is taken as previous solution.
  
    # final solution
    [f,r,yy,fout] = lc.shoot_mercmodel(v,ricb[k],rhocr,rh,param,scale)
    nr=len(r)
    
    # re-scale
    r=scale['a']*r
    P1=scale['P']*yy[0]
    g1=scale['ga']*yy[1]
    T1=scale['T']*yy[2]
    Tad=scale['T']*yy[3]
    rho1=rhomean*yy[4]
    chiS=yy[5]
    chiSin=fout[4]

    # Plot the output
    vis.plot_isnow(rs[k],r,rh*scale['a'],rm,T1,P1,chiS,rho1,chiSin,param)    
