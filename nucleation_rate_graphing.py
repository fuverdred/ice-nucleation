import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

import calorimeter as c

def sigmoid_j(c):
    x,y = c.sigFit[:,0], c.sigFit[:,1]
    x = x[::-1]
    y = y[::-1]
    y = [1-i for i in  y]
    dndT = np.gradient(y,x)
    j = lambda x: -1/(x[0]*c.area*c.dTdt)*x[1]
    J = list(map(j,list(zip(y,dndT))))
    return x,J
    
    

os.chdir("O:/Documents/Summer_Project/Data/Paper/")

K_01 = c.nucleation_curve("peak_data_c_0.1%.csv", 5000, 0.14, 0.1,label = "Crystalline K-Feldspar 0.1%wt")
K_005 = c.nucleation_curve("peak_data_c_0.05%.csv", 5000, 0.14, 0.05,label = "Crystalline K-Feldspar 0.05%wt")
K_0025 = c.nucleation_curve("peak_data_c_0.025%.csv", 5000, 0.14, 0.025,label = "Crystalline K-Feldspar 0.025%wt")
K_00125 = c.nucleation_curve("peak_data_c_0.0125%.csv", 5000, 0.14, 0.0125,label = "Crystalline K-Feldspar 0.0125%wt")

K_005.temps.sort(reverse = True)
K_005.temps = K_005.temps[:-1]
K_005.frac = K_005.frozen_frac(K_005.temps)
K_005.j = K_005.J()
K_005.n = K_005.N_s()
K_005.frac_error()
K_005.jErr = K_005.JError()
K_005.nErr = K_005.NError()

G_1 = c.nucleation_curve("peak_data_glassy_1%.csv", 1800, 0.22, 1, label="Glassy K-Feldspar 1%wt")
G_01 = c.nucleation_curve("peak_data_glassy_0.1%.csv", 1800, 0.22, 0.1, label="Glassy K-Feldspar 0.1%wt")

cryst = (K_01, K_005, K_0025, K_00125)
glass = (G_1, G_01)

cmapCryst = cm.tab20b(np.linspace(0,0.19,4))#4 shades of blue
cmapGlass = cm.tab20b(np.linspace(0.2,0.4,3))#3 shades of red

symCryst = ["v", "^", ">", "<"]
symGlass = ["+", "x"]

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.subplots_adjust(hspace=0)

for i,val in enumerate(cryst):
    ax1.plot(val.frac[:,0], val.frac[:,1], c=cmapCryst[i],label=val.label)
    ax2.plot(val.frac[:,0], val.frac[:,1], c=cmapCryst[i])
    ax1.plot(val.sigFit[:,0], val.sigFit[:,1], c=cmapCryst[i],alpha=0.5,
             linestyle='--')
    x,y = val.frac[:,0],val.frac[:,1]
    x = x[::-1]#x must be monotonic increasing to spline
    y = y[::-1]
    spl = UnivariateSpline(x,y)
    spl.set_smoothing_factor(0.01)#don't smooth much
    xs = np.linspace(x[0],x[-1],100)
    ax2.plot(xs,spl(xs),c=cmapCryst[i],linestyle="--",alpha=0.5)
    
for i,val in enumerate(glass):
    ax1.plot(val.frac[:,0], val.frac[:,1], c=cmapGlass[i],label=val.label)
    ax2.plot(val.frac[:,0], val.frac[:,1], c=cmapGlass[i])
    ax1.plot(val.sigFit[:,0], val.sigFit[:,1], c=cmapGlass[i],alpha=0.5,
             linestyle='--')
    x,y = val.frac[:,0],val.frac[:,1]
    x = x[::-1]#x must be monotonic increasing to spline
    y = y[::-1]
    spl = UnivariateSpline(x,y)
    spl.set_smoothing_factor(0.01)#don't smooth much
    xs = np.linspace(x[0],x[-1],100)
    ax2.plot(xs,spl(xs),c=cmapGlass[i],linestyle="--",alpha=0.5)

ax1.legend(prop={'size': 6})
ax1.set_ylabel("Frozen Fraction")
ax2.set_ylabel("Frozen Fraction")
ax2.set_xlabel("Temperature ($^\circ$C)")
    
plt.show()
