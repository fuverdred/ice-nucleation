import os
import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

class nucleation_curve():
    def __init__(self, filename,areaPerMass,areaErr,conc,volume = 1E-6,dTdt = -1/60, label = "label"):
        self.filename = filename
        self.label = label
        self.dTdt = dTdt
        self.area = conc*0.01*volume*areaPerMass
        self.areaErr = self.area*areaErr
        self.temps = sorted([i if type(i)==np.float64 else i[-1] for
                             i in np.genfromtxt(filename, delimiter=',')], reverse = True)
        self.N = len(self.temps)
        self.frac = self.frozen_frac(temps = self.temps)
        self.sigFit = self.sigmoid_fit()
        self.j = self.J()
        self.n = self.N_s()
        self.frac_error()
        self.jErr = self.JError()
        self.nErr = self.NError()

    def frozen_frac(self, temps):
        """Give frozen proportion at each temperature"""
        N = len(temps)
        n = 0
        frac = []
        while n<N:
            x = temps.count(temps[n])
            frac.append([temps[n], (x+n)/N])
            n += x
        return np.array(frac)
        
    def sigmoid(self,x, x0, k):
        return 1/(1 + np.exp(-k*(x-x0)))

    def dsigmoid(self, x,x0,k):
        return (k*np.exp(-k*(x-x0)))*self.sigmoid(x,x0,k)**2
    
    def sigmoid_fit(self):
        self.sigParams, pcov = curve_fit(self.sigmoid, self.frac[:,0], self.frac[:,1])
        x = np.linspace(self.temps[0],self.temps[-1],100)
        y = self.sigmoid(x, *self.sigParams)
        return np.array(list(zip(x,y)))

    def J(self):
        """
        Calculate the nucleation rate by numerical differentiation of
        the frozen fraction curve. Weighted by surface area.
        """
        def j(frac, dndt):
            return -1/(self.dTdt*self.area*(frac*self.N))*(dndt*self.N)
        x = self.frac[:,0]
        y = self.frac[:,1]#Frozen fraction
        y = [1-i for i in y]#convert frozen fraction to unfrozen fraction
        self.dndT = np.gradient(y,x)#Numerical differential
        y = [j(y[i], val) for i,val in enumerate(self.dndT)]
        return np.array(list(zip(x,y)))

    def N_s(self):
        return list(map(lambda x: -np.log(1-x)/self.area, self.frac[:,1]))
    
    def frac_error(self, remove = 0.2, repeats = 1000):
        """
        Bootstrapping to find error in n(T) and dndT(T). 
        """
        randoms = []
        remove = int(remove*len(self.temps))
        for i in range(repeats):
            random.shuffle(self.temps)
            new = self.temps[:-remove]
            new.sort(reverse = True)
            frac = self.frozen_frac(new)
            grad = np.gradient(frac[:,1], frac[:,0])
            randoms.append(list(zip(frac[:,0], frac[:,1], grad)))#[temperature, fraction, gradient]
        allTemps = list(set(self.temps))
        allTemps.sort(reverse = True)
        randomsSorted = [[T,[],[]] for T in allTemps]
        for lst in randoms:
            for t,f,df in lst:
                randomsSorted[allTemps.index(t)][1].append(f)
                randomsSorted[allTemps.index(t)][2].append(df)
        self.fErr = [np.std(i[1]) for i in randomsSorted]
        self.dfErr = [np.std(i[2]) for i in randomsSorted]

    def JError(self):
        return [x*((self.areaErr/self.area)**2+\
                   (self.fErr[i]/self.frac[:,1][i])**2+\
                   (self.dfErr[i]/self.dndT[i])**2)**0.5\
                for i,x in enumerate(self.j[:,1])]

    def NError(self):
        return [x*((self.areaErr/self.area)**2+\
                   (self.fErr[i]/self.frac[:,1][i])**2)**0.5
                for i,x in enumerate(self.n)]

def strip_data(data, cutTemperature = -10):
    for i,(time,voltage,temperature) in enumerate(data):
        if temperature < cutTemperature:
            return data[i:]
    print("Cut off not reached")#return None

def find_freezes(data, threshVolt):
    detected = False
    freezes = []
    for i,(time, voltage, temperature) in enumerate(data):
        if voltage > threshVolt and detected == False:
            freezes.append([i,time,voltage,temperature])
            detected = True
        elif voltage<threshVolt/10 and detected == True:
            detected = False#Reset the flag
    return freezes

def find_freezes_noisy(data, smooth,threshVolt):
    freezes = []
    detected = False
    for i,(time, volt, temp) in enumerate(data):
        if volt - smooth[i] > threshVolt and detected == False:
            freezes.append([time, volt, temp])
            detected = True
        elif abs(volt-smooth[i]) < threshVolt/10 and detected == True:
            detected = False
    return freezes

def cumulative_freeze(freezes):
    """List must be ordered before insertion"""
    N = len(freezes)
    frac = []
    n = 0
    while n<N:
        x = freezes.count(freezes[n])
        frac.append([freezes[n], (n+x)/N])
        n += x
    return np.array(frac)

def extract_temps(files):
    total_temps = []
    for f in files:
        data = np.genfromtxt(f, delimiter=',')
        data = strip_data(data,-5)
        plt.plot(data[:,0], data[:,1])
        plt.show()
        x = input("Flip data? y/n")
        if x == "y":
            data = np.array([[t,-v,T] for t,v,T in data])
        smooth = savgol_filter(data[:,1], 201, 2)
        x = True
        while x:
            thresh = float(input("Enter cutoff:"))
            temps = find_freezes_noisy(data, smooth, thresh)
            print("len temps:", len(temps))
            plt.plot(data[:,0], data[:,1])
            plt.plot(data[:,0], smooth, c="g")
            plt.scatter([i[0] for i in temps],[i[1] for i in temps], c="r")
            plt.show()
            x = input("Press enter to continue or a char to repeat")
        for i in temps:
            total_temps.append(i[-1])
    return total_temps

def gradient_error_calc(temps, remove = 0.1, repeats = 100):
    """
    Bootstrapping to find error in n(T) and dndT(T). 
    """
    randoms = []
    remove = int(remove*len(temps))
    for i in range(repeats):
        random.shuffle(temps)
        new = temps[:-remove]
        new.sort(reverse = True)
        frac = cumulative_freeze(new)
        grad = np.gradient(frac[:,1], frac[:,0])
        randoms.append(list(zip(frac[:,0], frac[:,1], grad)))#[temperature, fraction, gradient]
    allTemps = list(set(temps))
    allTemps.sort(reverse = True)
    randomsSorted = [[T,[],[]] for T in allTemps]
    for lst in randoms:
        for t,f,df in lst:
            randomsSorted[allTemps.index(t)][1].append(f)
            randomsSorted[allTemps.index(t)][2].append(df)
    T = [i[0] for i in randomsSorted]
    fErr = [np.std(i[1]) for i in randomsSorted]
    dfErr = [np.std(i[2]) for i in randomsSorted]
    return T,fErr,dfErr    

def spline_j(c):
    def j(frac, dndt):
            return -1/(c.dTdt*c.area*(frac*c.N))*(dndt*c.N)
    x,y = c.frac[:,0], c.frac[:,1]
    x.sort()#must be sorted this way
    spl = UnivariateSpline(x,y)
    spl.set_smoothing_factor(0.01)
    xs = np.linspace(x[0], x[-1], 1000)
    n = list(spl(xs))
    dndT = list(spl.derivative(1)(xs))
    y = [j(n[i], val) for i,val in enumerate(dndT)]
    return xs,y
    
    

os.chdir("O:/Documents/Summer_Project/Data/Paper/")

K_01 = nucleation_curve("peak_data_c_0.1%.csv", 5000, 0.14, 0.1,label = "Crystalline K-Feldspar 0.1%wt")
K_005 = nucleation_curve("peak_data_c_0.05%.csv", 5000, 0.14, 0.05,label = "Crystalline K-Feldspar 0.05%wt")
K_0025 = nucleation_curve("peak_data_c_0.025%.csv", 5000, 0.14, 0.025,label = "Crystalline K-Feldspar 0.025%wt")
K_00125 = nucleation_curve("peak_data_c_0.0125%.csv", 5000, 0.14, 0.0125,label = "Crystalline K-Feldspar 0.0125%wt")

K_005.temps.sort(reverse = True)
K_005.temps = K_005.temps[:-1]
K_005.frac = K_005.frozen_frac(K_005.temps)
K_005.j = K_005.J()
K_005.n = K_005.N_s()
K_005.frac_error()
K_005.jErr = K_005.JError()
K_005.nErr = K_005.NError()

G_1 = nucleation_curve("peak_data_glassy_1%.csv", 1800, 0.22, 1, label="Glassy K-Feldspar 1%wt")
G_01 = nucleation_curve("peak_data_glassy_0.1%.csv", 1800, 0.22, 0.1, label="Glassy K-Feldspar 0.1%wt")

pure = np.genfromtxt("peak_data_Pure_water.csv", delimiter=',')
pure_frac = cumulative_freeze(list(pure))

"""
===============
Plotting Graphs
===============
"""

#=====Nucleation Rates========================================================
cryst = [K_01,K_005, K_0025, K_00125]
glass = [G_1, G_01]

cmapCryst = cm.tab20b(np.linspace(0,0.19,4))#4 shades of blue
cmapGlass = cm.tab20b(np.linspace(0.2,0.4,3))#3 shades of red

symCryst = ["v", "^", ">", "<"]
symGlass = ["+", "x"]
cmapPure = cm.tab20b(0.4)

##fig, ax = plt.subplots()
##fig.set_size_inches(10,10)
##ax.set_yscale("log")
##
##for c,curve in enumerate(cryst):
##    ax.errorbar(curve.j[:,0], curve.j[:,1], xerr = 1, yerr = curve.jErr,
##                c = cmapCryst[c], fmt="o", label = curve.label)
##
##for c,curve in enumerate(glass):
##    ax.errorbar(curve.j[:,0], curve.j[:,1],xerr= 1,yerr= curve.jErr,
##                c = cmapGlass[c], fmt="o", label=curve.label)
##plt.xlabel("Temperature ($^\circ$C)")
##plt.ylabel("Nucleation Rate (s$^{-1}$m$^{-2}$)")
##plt.legend()
##plt.show()
#=============================================================================

#=====Murray Paramaterisation=================================================
##fig, ax = plt.subplots()
##fig.set_size_inches(10,10)
##ax.set_yscale("log")
##
##for c,curve in enumerate(cryst):
##    ax.scatter(curve.frac[:,0], curve.n, c = cmapCryst[c], label=curve.label)
##
##for c,curve in enumerate(glass):
##    ax.scatter(curve.frac[:,0], curve.n, c = cmapGlass[c], label=curve.label)
##
##plt.legend()
##plt.xlabel("Temperature ($^\circ$C)")
##plt.ylabel("$\mathregular{n_s}$")
##plt.show()
#=============================================================================

#=====Shared axes=============================================================
##f, (ax1,ax2,ax3) = plt.subplots(3, sharex=True)
##f.subplots_adjust(hspace=0)
##f.set_size_inches(9,7)
##
##ax1.set_ylabel("Frozen Proportion")
##lines = []#Store info for legend
##lines.append(ax1.plot(pure_frac[:,0], pure_frac[:,1], c="k",
##                      label = "Pure water"))
##for c,curve in enumerate(cryst):
##    lines.append(ax1.scatter(curve.frac[:,0], curve.frac[:,1], c = cmapCryst[c],
##                label = curve.label))
##for c,curve in enumerate(glass):
##    lines.append(ax1.scatter(curve.frac[:,0], curve.frac[:,1], c = cmapGlass[c],
##                label = curve.label))
##
##ax2.set_yscale("log")
##ax2.set_xlabel("Nucleation Rate (s$^{-1}$m$^{-2}$)")
##for c,curve in enumerate(cryst):
##    ax2.scatter(curve.frac[:,0], curve.n, c = cmapCryst[c])
##for c,curve in enumerate(glass):
##    ax2.scatter(curve.frac[:,0], curve.n, c = cmapGlass[c])
##
##ax3.set_yscale("log")
##for c,curve in enumerate(cryst):
##    ax3.errorbar(curve.j[:,0], curve.j[:,1], xerr = 1, yerr = curve.jErr,
##                 c = cmapCryst[c], fmt = "o")
##for c,curve in enumerate(glass):
##    ax3.errorbar(curve.j[:,0], curve.j[:,1], xerr = 1, yerr = curve.jErr,
##                 c = cmapGlass[c], fmt = "o")
##labels = ["pure water"]+[i.label for i in cryst]+[i.label for i in glass]
##ax3.legend(lines, labels, prop={'size': 8}, loc = "lower left")
###ret = f.legend(*ax1.get_legend_handles_labels(),"lower left")
###ret.draggable(True)
##plt.xlabel("Temperature ($^\circ$C)")
##ax3.set_ylabel("$\mathregular{n_s}$")
#plt.show() 
#=============================================================================

#=====Triple plot=============================================================
##temp_error = ([0.2]*2, [0.8]*2)
##
##fig2, (ax1, ax2, ax3) = plt.subplots(3)
##fig2.set_size_inches(12,7)
##grid = plt.GridSpec(2,2)
##ax1 = plt.subplot(grid[0, 0:])
##ax1.tick_params(which='both', top='on', right='on', direction="in")
##ax1.set_ylabel("Frozen Proportion")
##ax1.set_xlabel("Temperature ($^\circ$C)")
##ax1.set_xlim((-33,-4))
##ax1.scatter(pure_frac[:,0], pure_frac[:,1], c="k", label = "Pure water",
##            marker = "*")
##for c,curve in enumerate(cryst):
##    ax1.errorbar([curve.frac[:,0][int(len(curve.frac)/2)]],
##                 [curve.frac[:,1][int(len(curve.frac)/2)]],
##                 xerr = [[0.2],[0.8]], fmt= "none",c=cmapCryst[c])
##    ax1.scatter(curve.frac[:,0], curve.frac[:,1], c = cmapCryst[c],
##                label = curve.label, marker = symCryst[c])
##for c,curve in enumerate(glass):
##    ax1.scatter(curve.frac[:,0], curve.frac[:,1], c = cmapGlass[c],
##                label = curve.label, marker=symGlass[c])
##ax1.legend(loc = "upper right", prop={"size": 9})
##
##ax2 = plt.subplot(grid[1,0])
##ax2.set_yscale("log")
##ax2.set_ylabel("$\mathregular{n_s}$ (m$^{-2}$)")
##ax2.set_xlabel("Temperature ($^\circ$C)")
##ax2.tick_params(which='both', top='on', right='on', direction="in")
##ax2.locator_params(axis = "x", nticks=5)
##for c,curve in enumerate(cryst):
##    ax2.errorbar([curve.frac[:,0][0], curve.frac[:,0][-2]],[curve.n[0],
##                  curve.n[-2]], xerr = temp_error, yerr = [curve.nErr[0], curve.nErr[-2]],
##                  fmt = "none", c = cmapCryst[c])
##    ax2.scatter(curve.frac[:,0], curve.n, c = cmapCryst[c],
##                marker=symCryst[c], s=20)
##for c,curve in enumerate(glass):
##    ax2.errorbar([curve.frac[:,0][0], curve.frac[:,0][-2]],[curve.n[0], curve.n[-2]], xerr = temp_error,
##                yerr = [curve.nErr[0], curve.nErr[-2]], fmt = "none", c = cmapGlass[c])
##    ax2.scatter(curve.frac[:,0], curve.n, c = cmapGlass[c],
##                marker=symGlass[c], s=20)
##
##ax3 = plt.subplot(grid[1,1])
##ax3.set_yscale("log")
##ax3.set_ylabel("Nucleation Rate (s$^{-1}$m$^{-2}$)")
##ax3.set_xlabel("Temperature ($^\circ$C)")
##ax3.tick_params(which='both', top='on', right='on', direction="in")
##ax3.locator_params(axis = "x", nticks=5)
##for c,curve in enumerate(cryst):
##    ax3.errorbar([curve.j[:,0][0], curve.j[:,0][-2]], [curve.j[:,1][0],
##                  curve.j[:,1][-2]], xerr = temp_error,
##                  yerr = [curve.jErr[0],curve.jErr[-2]],
##                  c = cmapCryst[c], fmt = 'none')
##    ax3.scatter(curve.j[:,0], curve.j[:,1], c = cmapCryst[c],
##                marker = symCryst[c], s=20)
##for c,curve in enumerate(glass):
##    ax3.errorbar([curve.j[:,0][0], curve.j[:,0][-2]], [curve.j[:,1][0],
##                  curve.j[:,1][-2]], xerr = temp_error, yerr = [curve.jErr[0],
##                    curve.jErr[-2]],c = cmapGlass[c], fmt = 'none')
##    ax3.scatter(curve.j[:,0], curve.j[:,1], c = cmapGlass[c],
##                marker = symGlass[c], s=20)
##
##ax1.annotate("(a)", xy = (0,0), xycoords = "axes fraction", fontsize=30,
##             xytext=(10,10), textcoords="offset pixels", ha="left", va="bottom")
##ax2.annotate("(b)", xy = (0,0), xycoords = "axes fraction", fontsize=30,
##             xytext=(10,10), textcoords="offset pixels", ha="left", va="bottom")
##ax3.annotate("(c)", xy = (0,0), xycoords = "axes fraction", fontsize=30,
##             xytext=(10,10), textcoords="offset pixels", ha="left", va="bottom")
##plt.show()
#=============================================================================

#=====Final Plots=============================================================
##fig1, ax = plt.subplots(1)
##fig1.set_size_inches(14,7)
##ax.set_ylabel("Frozen Proportion")
##ax.set_xlabel("Temperature ($^\circ$C)")
##
##ax.plot(pure_frac[:,0], pure_frac[:,1], c="k", label = "Pure water")
##for c,curve in enumerate(cryst):
##    ax.scatter(curve.frac[:,0], curve.frac[:,1], c = cmapCryst[c],
##                label = curve.label, s=20)
##for c,curve in enumerate(glass):
##    ax.scatter(curve.frac[:,0], curve.frac[:,1], c = cmapGlass[c],
##                label = curve.label, marker="^", s=20)
##ax.legend(loc = 1, prop={"size": 8.5})
##
##fig2, ax = plt.subplots(1)
##fig2.set_size_inches(7,7)
##ax.set_yscale("log")
##ax.set_ylabel("$\mathregular{n_s}$ (m$^{-2}$)")
##ax.set_xlabel("Temperature ($^\circ$C)")
##ax.locator_params(axis = "x", nticks=5)
##
##for c,curve in enumerate(cryst):
##    ax.errorbar([curve.frac[:,0][0], curve.frac[:,0][-2]],[curve.n[0], curve.n[-2]], xerr = 1,
##                yerr = [curve.nErr[0], curve.nErr[-2]], fmt = "none", c = cmapCryst[c])
##    ax.scatter(curve.frac[:,0], curve.n, c = cmapCryst[c], marker="o", s=20)
##for c,curve in enumerate(glass):
##    ax.errorbar([curve.frac[:,0][0], curve.frac[:,0][-2]],[curve.n[0], curve.n[-2]], xerr = 1,
##                yerr = [curve.nErr[0], curve.nErr[-2]], fmt = "none", c = cmapGlass[c])
##    ax.scatter(curve.frac[:,0], curve.n, c = cmapGlass[c], marker="^", s=20)
##
##fig3, ax = plt.subplots(1)
##fig3.set_size_inches(7,7)
##ax.set_yscale("log")
##ax.set_ylabel("Nucleation Rate (s$^{-1}$m$^{-2}$)")
##ax.set_xlabel("Temperature ($^\circ$C)")
##for c,curve in enumerate(cryst):
##    ax.errorbar([curve.j[:,0][0], curve.j[:,0][-2]], [curve.j[:,1][0], curve.j[:,1][-2]],
##                xerr = 1, yerr = [curve.jErr[0],curve.jErr[-2]],c = cmapCryst[c], fmt = 'none')
##    ax.scatter(curve.j[:,0], curve.j[:,1], c = cmapCryst[c], marker = "o", s=20)
##for c,curve in enumerate(glass):
##    ax.errorbar([curve.j[:,0][0], curve.j[:,0][-2]], [curve.j[:,1][0], curve.j[:,1][-2]],
##                xerr = 1, yerr = [curve.jErr[0],curve.jErr[-2]],c = cmapGlass[c], fmt = 'none')
##    ax.scatter(curve.j[:,0], curve.j[:,1], c = cmapGlass[c], marker = "^", s=20)
##
##plt.show()
#=============================================================================

def survival_curve(frac):
    def super_cooling(T):
        return 0 - T
    x = super_cooling(frac[:,0])
    y = [1-i for i in frac[:,1]]
    return x,y

x,y = survival_curve(K_0025.frac)
y2 = np.log(y)
plt.subplot(311)
plt.plot(x,y)
plt.subplot(312)
plt.plot(x,y2)
plt.subplot(313)
np.polyfit(x,y2, 3)

plt.plot(x,np.gradient(y2,x))
plt.show()
