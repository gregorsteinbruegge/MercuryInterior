#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
------------------------------
The following code contains routines to compute the Equation of States (EoS)
for l-Fe-S and l-Fe-Si based on Terasaki 2019 (10.1029/2019JE005936). 
Wrt. to this article there are two minor changes. The end-members are FeSi and 
FeSi and not 0.5Fe0.5S and 0.5Fe0.5FeSi. For the EoS of l-Fe we use 
Komabayashi 2014 (10.1002/2014JB010980) and not Anderson 1994. 
This has a minor effect on the parameterization of the EoS in the code which 
are hence not the same as in table 5 et 6.    
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate
from scipy import optimize

def VinetEq(x,p,KTP0,KT0):
    vinet=-p+(3*np.exp((3*(-1+KTP0)*(1-x))/2)*KT0*(1-x))/x**2
    return vinet

def GibbsLiquidFe(T):
    return 300-9007.3402+290.29866*T-46*T*np.log(T)

def GibbsfccFe(T):
    return 16300.921-395355.43/T-2476.28*np.sqrt(T)+ 381.47162*T+0.000177578*T**2-52.2754*T*np.log(T)

def VexFeS(chi,p,T):
    W11=-9.91275
    W12=0.731385
    W21=-1.32521
    W22=1.72716
    return (1-chi[1])*chi[1]*np.array([chi[1]*(W11+W12*np.log(1.5+p))+chi[0]*(W21+W22*np.log(1.5+p)),
                                  chi[1]*W12/(1.5+p)+chi[0]*W22/(1.5+p),0])

def VexFeSi(chi,p,T):
    W1=-2.3199284685783192
    W2=-1.2489264297620897  
    return (1-chi[1])*chi[1]*np.array([chi[1]*W1+chi[0]*W2,0,0])
    
class eosAndersonGrueneisen:
    def __init__(self,M0,p0,T0,V0,alpha0,KT0,KTP0,deltaT,kappa,
                 GibbsE=None,gamma0=None,q=None):
        self.pMax=200
        self.nbrPNodes=10001

        if (GibbsE is not None and (gamma0 is not None or q is not None)):
            print("Gibbs function and gamma not supported")
    				
        if (GibbsE is not None):
            self.GibbsFlag=True
            self.gamma0=0
            self.q=0
        else:
            self.GibbsFlag=False
            self.gamma0=gamma0
            self.q=q
    
        self.M0=M0
        self.p0=p0
        self.T0=T0
        self.V0=V0
        self.alpha0=alpha0
        self.KT0=KT0
        self.KTP0=KTP0
        self.deltaT=deltaT
        self.kappa=kappa
        self.GibbsE=GibbsE

        self.zetaA=np.zeros(self.nbrPNodes)
        self.px=np.zeros(self.nbrPNodes)
        self.zetaA[0]=1
        for i in range(1,self.nbrPNodes):
            self.px[i]=i/self.pMax
            self.zetaA[i]=self.compress(self.px[i])

        self.poly = CubicSpline(self.px,self.zetaA)

    def volume(self,x,T):
        # volume/V0
        p=x*self.pMax
        eta=(self.poly.__call__(p))**3
        alpha=self.alpha0*np.exp(-self.deltaT/self.kappa*(1-eta**self.kappa))
        return eta*np.exp(alpha*(T-self.T0))
    
    def Gibbs(self,p,T):
        if (p>self.p0):
            Gp = integrate.quad(lambda x: self.volume(x,T),
                                self.p0/self.pMax,p/self.pMax)[0]
        else :
            Gp=0
        return self.GibbsE(T)+1.e3*Gp*self.V0*self.pMax
        
    def compress(self,p):
        out = optimize.brentq(VinetEq, 0.7, 1.2, 
                                     args = (p,self.KTP0,self.KT0))
        return out

    def eos(self,p,T):
        deltaTemp=1 # temperature step for numerical differentiation, if too small results too noisy
        if (p>self.pMax):
            print("p should be smaller than ",self.pMax)
        T0=self.T0
        V0=self.V0
        alpha0=self.alpha0
        KT0=self.KT0
        KTP0=self.KTP0
        deltaT=self.deltaT
        kappa=self.kappa

        zeta=self.poly.__call__(p)
        eta=zeta**3
        alpha=alpha0*np.exp(-deltaT/kappa*(1-eta**kappa))
        V=V0*eta*np.exp(alpha*(T-T0))

        KT=(KT0*(4+(-5+3*KTP0)*zeta+3*(1-KTP0)*zeta**2))/np.exp((3*(-1+KTP0)*(-1+zeta))/2)
        KT=KT/(2*zeta**2)
        KT=KT/(1+(T-T0)*deltaT*alpha*eta**kappa)

        KTP=0.5*(KTP0-1)*zeta
        KTP=KTP+(8/3+(KTP0-5/3)*zeta)/(3*(4/3 +(KTP0-5/3)*zeta+(1-KTP0)*zeta**2))

        if (self.GibbsFlag):
            Gibbs=self.Gibbs(p,T)
            Cp=-T*(self.Gibbs(p,T+deltaTemp)-2*Gibbs+self.Gibbs(p,T-deltaTemp))/deltaTemp**2 # numerical second derivative of G with respect to T
            gamma=1/(Cp/(alpha*KT*V*1E+3)-alpha*T) # factor 1000 for conversion of GPa and cm^3/mol
            KS=KT*(1+gamma*alpha*T)
        else:
            Gibbs=0
            gamma=self.gamma0*eta**self.q
            KS=KT*(1+gamma*alpha*T)
            Cp=1E+3*alpha*V*KS/gamma

        self.V=V
        self.rho=1.e3*self.M0/V
        self.alpha=alpha
        self.KT=KT
        self.KTP=KTP
        self.KS=KS
        self.gamma=gamma
        self.vp=np.sqrt(1E+9*KS/self.rho)
        self.vs=0
        self.Cp=Cp
        self.CV=1.e3*alpha*V*KT/gamma
        self.GE=Gibbs	
	
class margules2Solution: 
    def __init__(self,chi,p,T,eM1,eM2,Vex):
        eM1.eos(p,T)
        eM2.eos(p,T)

        Vexx=Vex(chi,p,T) #[Vex,dVex/dp,dVex/dT]
        self.Vex=Vex
        self.M0=np.dot([eM1.M0,eM2.M0],chi)
        self.V=np.dot([eM1.V,eM2.V],chi)+Vexx[0]
        self.Cp=np.dot([eM1.Cp,eM2.Cp],chi)
        self.alpha=(np.dot([eM1.V*eM1.alpha,eM2.V*eM2.alpha],chi)+Vexx[2])/self.V
        self.KT=-self.V/(-np.dot([eM1.V/eM1.KT,eM2.V/eM2.KT],chi)+Vexx[1])
        self.gamma=1/(1E-3*self.Cp/(self.alpha*self.KT*self.V)-self.alpha*T)
        self.KS=self.KT*(1+self.alpha*self.gamma*T)
        self.CV=1E+3*self.alpha*self.V*self.KT/self.gamma
        self.rho=1E+3*self.M0/self.V
        self.vp=np.sqrt(1E+9*self.KS/self.rho)	
	     
def liquidNonIdalFeSi(x,p,T,param):
    MolarMassFe = param['MFe']
    MolarMassSi = param['MSi']
    liquidFe = param['lFe']
    liquidFeSi = param['lFeSi']    
    chi = np.zeros(2)
    
    chi[1]=MolarMassFe*x/(MolarMassSi+x*(MolarMassFe-MolarMassSi)) # molar fraction Si
    chi[1]=chi[1]/(1-chi[1]) # convert to molar fraction of FeSi
    chi[0]=1-chi[1]
    
    nonIdealFeFeSi=margules2Solution(chi,p,T,liquidFe,liquidFeSi,VexFeSi)
    liquidNonIdalFeSi=[nonIdealFeFeSi.V,
                       nonIdealFeFeSi.rho,
                       nonIdealFeFeSi.alpha,
                       nonIdealFeFeSi.KT,
                       nonIdealFeFeSi.KS,
                       nonIdealFeFeSi.Cp,
                       nonIdealFeFeSi.gamma,
                       nonIdealFeFeSi.vp]
    
    return liquidNonIdalFeSi

def liquidNonIdalFeS(x,p,T,param):
    chi = np.zeros(2)
    MolarMassFe = param['MFe']
    MolarMassS = param['MS']
    liquidFe = param['lFe']
    liquidFeS = param['lFeS']

    chi[1]=MolarMassFe*x/(MolarMassS+x*(MolarMassFe-MolarMassS)) # convert weight fraction to molar fraction S
    chi[1]=chi[1]/(1-chi[1]) # convert to molar fraction of FeS
    chi[0]=1-chi[1]
    nonIdealFeFeS=margules2Solution(chi,p,T,liquidFe,liquidFeS,VexFeS)
    liquidNonIdalFeS=[nonIdealFeFeS.V,
                      nonIdealFeFeS.rho,
                      nonIdealFeFeS.alpha,
                      nonIdealFeFeS.KT,
                      nonIdealFeFeS.KS,
                      nonIdealFeFeS.Cp,
                      nonIdealFeFeS.gamma,
                      nonIdealFeFeS.vp]
    
    return liquidNonIdalFeS

def solidFccFe(p,T,param):
    fccFe = param['fccFe']
    fccFe.eos(p,T)
    
    solidFccFe=[fccFe.V,
                fccFe.rho,
                fccFe.alpha,
                fccFe.KT,
                fccFe.KS,
                fccFe.Cp,
                fccFe.gamma,
                fccFe.vp]

    return solidFccFe

def solidFccFeSi(x,p,T,param):   
    MolarMassFe = param['MFe']
    MolarMassSi = param['MSi']

    fcc=solidFccFe(p,T,param)  
    
    chi=MolarMassFe*x/(MolarMassSi+x*(MolarMassFe-MolarMassSi)) # molar fraction Si
    xx=(MolarMassFe *(1.-chi)+chi*MolarMassSi)/MolarMassFe                                       
    return [fcc[0],fcc[1]*xx,fcc[2],fcc[3],fcc[4],fcc[5],fcc[6]/xx**2]