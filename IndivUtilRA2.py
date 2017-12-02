# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:41:16 2017

@author: Kerk
"""

import numpy as np
from LinApp_FindSS import LinApp_FindSS
import pandas as pd

def mdefs(Kp, Bp, K, B, LS, LU, z, params):
    
    NS = HSD*(1-LS) + HSI*(1-LS)
    NU = HUD*(1-LU) + HUI*(1-LU)
    W = (c*(f*NS)**((d-1)/d) + (1-c)*NU**((d-1)/d))**(d/(d-1))
    Y = (a*K**((b-1)/b) + (1-a)*W**((b-1)/b))**(b/(b-1))
    r = a*(Y/K)**(1/b)
    wS = f*(1-a)*(Y/W)**(1/b)*c*(W/NS)**(1/d)
    wU = (1-a)*(Y/W)**(1/b)*(1-c)*(W/NU)**(1/d)
    sC = s - nu*(B/H)
    C = wS*((HSD+HSI)/H)*(1-LS) + wU*((HUD+HUI)/H)*(1-LU) + \
        (1+sC)*K/H - Kp/H
    U = np.log(C) + AS*((HSD + HSI)/H)*np.log(LS) +  \
        AU*((HUD + HUI)/H)*np.log(LU)
    CB = (1+sC)*B/H - Bp/H
    CS = wS*(1-LS)
    CU = wU*(1-LU)
    gamS = (wS*(1-LS)/(HSD*wS*(1-LS)+HUD*wU*(1-LU))/H)
    gamU = (wU*(1-LU)/(HSD*wS*(1-LS)+HUD*wU*(1-LU))/H)
    CSD = gamS*CB + CS
    CUD = gamU*CB + CU
    CSI = CS
    CUI = CU
    USD = np.log(CSD) + AS*np.log(LS)
    UUD = np.log(CUD) + AU*np.log(LU)
    USI = np.log(CSI) + AS*np.log(LS)
    UUI = np.log(CUI) + AU*np.log(LU)

    
    return NS, NU, W, Y, r, wS, wU, sC, C, U, CB, CS, CU, CSD, CUD, CSI, CUI, \
        USD, UUD, USI, UUI 


def mdyn(theta, params):
    # unpack theta
    [Kpp, Bpp, Kp, Bp, K, B, LSp, LUp, LS, LU, zp, z] = theta
    
    NS, NU, W, Y, r, wS, wU, sC, C, U, CB, CS, CU, CSD, CUD, CSI, CUI, USD, \
        UUD, USI, UUI = mdefs(Kp, Bp, K, B, LS, LU, z, params)
        
    NSp, NUp, Wp, Yp, rp, wpS, wUp, sCp, Cp, Up, CBp, CSp, CUp, CSDp, CUDp, \
        CSIp, CUIp, USDp, UUDp, USIp, UUIp = mdefs(Kpp, Bpp, Kp, Bp, LSp, \
        LUp, zp, params)
    
    E1 = 1 - mu*np.abs(Kp - K ) - (1 +rp - delta - mu*np.abs(Kpp - Kp ))/(1+s)
    E2 = (C)**(-1) - beta*(Cp)**(-1)*(1+sCp)
    E3 = (C)**(-1)*wS - AS*LS**(-1)   
    E4 = (C)**(-1)*wU - AU*LU**(-1)   

    Earray = np.array([E1, E2, E3, E4])
    
    return Earray


# declare model parameters
delta = .1131/4
a = .38
b = .7
c = .37
d = 2.0
f = 3.
HSD = .2296
HUD = .7580
HSI = .0405 #.0029, .0405
HUI = .0095 #.0095, .0471
AS = 0.3745
AU = 0.5764
nu = .01  #3.08934
mu = 0.
s = .006
beta =  1/(1+1.1*s)
H = HSD + HUD + HSI + HUI
z= 0.
nx = 2
ny = 2
nz = 1

params = (delta, a, b, c, d, f, HSD, HUD, HSI, HUI, AS, AU, nu, mu, s, beta, H)
guessXY = np.ones((1,nx+ny))*.1
Zbar = np.array([z])
XYbar = LinApp_FindSS(mdyn, params, guessXY, Zbar, nx, ny)

Xbar = XYbar[0:nx]
Ybar = XYbar[nx:nx+ny]
In = np.concatenate((Xbar, Xbar, Xbar, Ybar, Ybar, Zbar, Zbar))

check = mdyn(In, params)
print(check)

[Kbar, Bbar, LSbar, LUbar] = XYbar

NSbar, NUbar, Wbar, Ybar, rbar, wSbar, wUbar, sCbar, Cbar, Ubar, CBbar, \
    CSbar, CUbar, CSDbar, CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, \
    UUIbar = mdefs(Kbar, Bbar, Kbar, Bbar, LSbar, LUbar, z, params)

bars = np.array([Kbar, Bbar, LSbar, LUbar, NSbar, NUbar, Wbar, Ybar, rbar, \
    wSbar, wUbar, sCbar, Cbar, Ubar, CBbar, CSbar, CUbar, CSDbar, CUDbar, \
    CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar])  

# list the variable names in order
varindex = ['Kbar', 'Bbar', 'LSbar', 'LUbar', 'NSbar', 'NUbar', 'Wbar', \
    'Ybar', 'rbar', 'wSbar', 'wUbar', 'sCbar', 'Cbar', 'Ubar', 'CBbar', \
    'CSbar', 'CUbar', 'CSDbar', 'CUDbar', 'CSIbar', 'CUIbar', 'USDbar', \
    'UUDbar', 'USIbar', 'UUIbar'] 

barsdf = pd.DataFrame(bars.T)
barsdf.index = varindex   

print (barsdf.to_latex())
