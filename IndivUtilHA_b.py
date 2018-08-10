# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:41:16 2017

@author: Kerk
"""

import numpy as np
from LinApp_FindSS import LinApp_FindSS
import pandas as pd

def mdefs(Kp, BSDp, BUDp, K, BSD, BUD, LSD, LUD, LSI, LUI, z, params):
    
    B = BSD + BUD
    NS = HSD*(1-LSD) + HSI*(1-LSI)
    NU = HUD*(1-LUD) + HUI*(1-LUI)
    W = (c*(f*NS)**((d-1)/d) + (1-c)*NU**((d-1)/d))**(d/(d-1))
    Y = (a*K**((b-1)/b) + (1-a)*W**((b-1)/b))**(b/(b-1))
    r = a*(Y/K)**(1/b)
    wS = f*(1-a)*(Y/W)**(1/b)*c*(W/NS)**(1/d)
    wU = (1-a)*(Y/W)**(1/b)*(1-c)*(W/NU)**(1/d)
    sS = s - nu*(BSD/HSD)
    sU = s - nu*(BUD/HUD)
    CSD = wS*(1-LSD) + (1+sS)*BSD/HSD - BSDp/HSD
    CUD = wU*(1-LUD) + (1+sU)*BUD/HUD - BUDp/HUD
    CSI = wS*(1-LSI)
    CUI = wU*(1-LUI)
    if lam==1:
        USD = np.log(CSD) + AS*np.log(LSD)
        UUD = np.log(CUD) + AU*np.log(LUD)
        USI = np.log(CSI) + AS*np.log(LSI)
        UUI = np.log(CUI) + AU*np.log(LUI)
    else:
        USD = np.log(CSD) + AS*(LSD**(1-lam)-1)/(1-lam)
        UUD = np.log(CUD) + AU*(LUD**(1-lam)-1)/(1-lam)
        USI = np.log(CSI) + AS*(LSI**(1-lam)-1)/(1-lam)
        UUI = np.log(CUI) + AU*(LUI**(1-lam)-1)/(1-lam)        
    
    return B, NS, NU, W, Y, r, wS, wU, sS, sU, CSD, CUD, CSI, CUI, USD, UUD, \
        USI, UUI


def mdyn(theta, params):
    # unpack theta
    [Kpp, BSDpp, BUDpp, Kp, BSDp, BUDp, K, BSD, BUD, \
     LSDp, LUDp, LSIp, LUIp, LSD, LUD, LSI, LUI, zp, z] = theta
    
    B, NS, NU, W, Y, r, wS, wU, sS, sU, CSD, CUD, CSI, CUI, USD, UUD, \
        USI, UUI = \
        mdefs(Kp, BSDp, BUDp, K, BSD, BUD, LSD, LUD, LSI, LUI, z, params)
        
    Bp, NSp, NUp, Wp, Yp, rp, wSp, wUp, sSp, sUp, CSDp, CUDp, CSIp, CUIp, \
        USDp, UUDp, USIp, UUIp = \
        mdefs(Kpp, BSDpp, BUDpp, Kp, BSDp, BUDp, LSDp, LUDp, LSIp, LUIp, z, \
        params)
    
    EK = 1 - mu*np.abs(Kp - K ) - (1 +rp - delta - mu*np.abs(Kpp - Kp ))/(1+s)
    ESD1 = (CSD)**(-1) - beta*(CSDp)**(-1)*(1+sSp)
    EUD1 = (CUD)**(-1) - beta*(CUDp)**(-1)*(1+sUp)
    ESD2 = (CSD)**(-1)*wS - AS*LSD**(-lam)
    EUD2 = (CUD)**(-1)*wU - AU*LUD**(-lam)
    ESI2 = (CSI)**(-1)*wS - AS*LSI**(-lam)
    EUI2 = (CUI)**(-1)*wU - AU*LUI**(-lam)
    
    Earray = np.array([EK, ESD1, EUD1, ESD2, EUD2, ESI2, EUI2])
    
    return Earray


# declare model parameters
delta = .1131/4
a = .38
b = 0.7
c = .37
d = 2.0
f = 3.
HSD = .2296
HUD = .7580
HSI = .0029 #.0405
HUI = .0095 #.0471
AS = 0.3745
AU = 0.5764
nu = .01  #3.08934
mu = 0.
s = .006
beta =  1/(1+1.1*.006)
H = HSD + HUD + HSI + HUI
lam = .75
z= 0.
nx = 3
ny = 4
nz = 1
params = (delta, a, b, c, d, f, HSD, HUD, HSI, HUI, AS, AU, nu, mu, s, beta, \
          lam, H)
guessXY = np.ones((1,nx+ny))*.1
Zbar = np.array([z])
XYbar = LinApp_FindSS(mdyn, params, guessXY, Zbar, nx, ny)

Xbar = XYbar[0:nx]
Ybar = XYbar[nx:nx+ny]
In = np.concatenate((Xbar, Xbar, Xbar, Ybar, Ybar, Zbar, Zbar))

check = mdyn(In, params)
print(check)

[Kbar, BSDbar, BUDbar, LSDbar, LUDbar, LSIbar, LUIbar] = XYbar

Bbar, NSbar, NUbar, Wbar, Ybar, rbar, wSbar, wUbar, sSbar, sUbar, CSDbar, \
    CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar = \
    mdefs(Kbar,  BSDbar, BUDbar, Kbar, BSDbar, BUDbar, \
    LSDbar, LUDbar, LSIbar, LUIbar, z, params)

bars = np.array([Kbar, BSDbar, BUDbar, LSDbar, LUDbar, LSIbar, \
    LUIbar, Bbar, NSbar, NUbar, Wbar, Ybar, rbar, wSbar, wUbar, sSbar, sUbar, \
    CSDbar,  CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar])  

# list the variable names in order
varindex = ['Kbar', 'BSDbar', 'BUDbar', 'LSDbar', 'LUDbar', \
    'LSIbar', 'LUIbar', 'Bbar', 'NSbar', 'NUbar', 'Wbar', 'Ybar', \
    'rbar', 'wSbar', 'wUbar', 'sSbar', 'sUbar', 'CSDbar', 'CUDbar', 'CSIbar', \
    'CUIbar', 'USDbar', 'UUDbar', 'USIbar', 'UUIbar'] 

barsdf = pd.DataFrame(bars.T)
barsdf.index = varindex   

print (barsdf.to_latex())

print(1.01*s, Bbar/Ybar)
