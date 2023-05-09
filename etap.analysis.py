#eta p analysis

import numpy as np
import LT.box as B
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

#getting the data
F = B.get_file('etap2009.py')
w = B.get_data(F, 'W')
cos = B.get_data(F, 'cos') 
dsigdomega = B.get_data(F, 'dsigdomega')
sig_omega = B.get_data(F, 'sig')

#mass in GeV
mp = 0.938272081 #proton mass which is a byron
me = 0.547862 #eta meson mass

#converting dsigdomega to dsigdcos
dsigdcos = 2*np.pi*dsigdomega
sig = sig_omega*2*np.pi

#Energies in GeV at cm
s = w**2
Eg_cm = (s-mp**2)/(2*w) #energy of photon at cm ... check!
Eb_cm = (s+mp**2-me**2)/(2*w) #energy of byron at cm (proton)
Em_cm = (s-mp**2+me**2)/(2*w) #energy of meson at cm (eta)

#momentom at cm
Pg_cm = Eg_cm #momentum of gamma k=E1
Pb_cm = np.sqrt((Eb_cm)**2 - (mp)**2)  #momentum of byron (proton) at cm
Pm_cm = np.sqrt((Em_cm)**2 - (me)**2) #momentum of meson (proton) at cm

t = (2.*Pg_cm*Pm_cm*cos+me**2-2.*Eg_cm*Em_cm)
dsigdt = dsigdcos/ (2.*Pg_cm*Pm_cm)
sig_dt = sig/(2.*Pg_cm*Pm_cm)

#calculating transverse momentum
pt2_1 = (s-mp**2)**2/(4*s)
pt2_2 = (((s+me**2-mp**2)**2)/(4*s)) - me**2
pt2_3 = ((1/(4*s)*(s-mp**2)*(s+me**2-mp**2))+(t-me**2)/2)**2
pt2_4 = ((s-mp**2)**2)/(4*s)
pt2 = (((pt2_1)*(pt2_2)-(pt2_3))/pt2_4)

#cosines close to zero
mx = 0.15
mn = -0.15

#exclusion of cosines values
cospt1 = cos[(cos <= mx)&(cos >= mn)] #cosine of angle 85 to 90 but in rad
s1 = s[(cos <= mx)&(cos >= mn)]
dsigdt1 = dsigdt[(cos <= mx)&(cos >= mn)]
sig1 = sig_dt[(cos <= mx)&(cos >= mn)] #is it necessary?
t1 = t[(cos <= mx)&(cos >= mn)]
pt21 = pt2[(cos <= mx)&(cos >= mn)]
#d_dsigdt1 = d_dsigdt[(cos <= mx)&(cos >= mn)]

#array for making cuts 
alpha = np.arange(0.0*max(pt21), 0.8*max(pt21), max(pt21)/100) #max(pt21) is the max value of the transverse momentum
second = np.arange(0.8*max(pt21), 0.99*max(pt21), (max(pt21)-0.8*max(pt21))/6)
alpha = np.append(alpha, second)

def expanded_fit(x, A, C, N):
    return (A + C*x[0])*x[1]**(-N)

popt, pcov = curve_fit(expanded_fit, (cospt1, s1), dsigdt1, sigma=sig1, maxfev=5000)
plt.errorbar(s1, dsigdt1, yerr=sig1, fmt='o', marker='v', color='g')
plt.yscale('log')
plt.ylabel(r'$\frac{d\sigma}{dt}$', size=30)
plt.xlabel('s', size=30)

redchi0 = np.array([])
Nres0 = np.array([])
Nerr0 = np.array([])
cut = np.array([])
perc = np.array([])

for j in alpha[:-4]:
    
    per = j/max(alpha)
    perc = np.append(perc, per)
    cut = np.append(cut, j)
    pt2min = j
    
    #exclude values lower than the minimum transverse momentum
    coss= cospt1[pt21 >= pt2min]
    dsigdts = dsigdt1[pt21 >= pt2min]
    sigs = sig1[pt21 >= pt2min]
    pts2 = pt21[pt21 >= pt2min]
    ts = -t1[pt21 >= pt2min]
    ss = s1[pt21 >= pt2min]

    #fit
    popt, pcov = curve_fit(expanded_fit, (coss, ss), dsigdts, sigma=sigs, maxfev= 5000)
    plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'v', marker = 'o', color = 'g')
    plt.yscale('log')
    plt.title(r'$\frac{d\sigma}{dt} = (A+Bcos\theta)s^-N; \gamma  p \rightarrow \eta p$', size = 30)
    plt.ylabel(r'$\frac{d\sigma}{dt}$', size=30)
    plt.xlabel(r'$s (GeV^2)$', size=30)
    plt.show()
    
    y_pred = expanded_fit((coss, ss), popt[0],popt[1], popt[2])
    chi_squared = np.sum(((dsigdts-y_pred)/sigs)**2)
    redchi = (chi_squared)/(len(coss)-len(popt))     
    redchi0 = np.append(redchi0, redchi)
    N = np.abs(popt[2])
    N_err = pcov[2,2]**0.5
    Nres0 = np.append(Nres0, N)
    Nerr0 = np.append(Nerr0, N_err)

#%%
#Reduced Chi Squared and N vs Cut

fig, ax1 = plt.subplots()
plt.title(r'$\gamma  p \rightarrow \eta p$', size =30)
color = 'tab:red'
ax1.set_xlabel(r"$p_T ^2$ cut $Gev^2$", size=30)
ax1.set_ylabel(r"$\chi^2/df$", size=30, color='tab:red')
ax1.plot(cut, redchi0, color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axvline(x=0.79, linestyle='--', color ='r') #constant dotted line
#ax1.axvline(x=1.1, linestyle='--', color ='r') #constant dotted line
plt.rc("ytick", labelsize=25)
plt.rc("xtick", labelsize=25)
ax2 = ax1.twinx() #instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r"$N$", size=30, color='tab:blue')
ax2.errorbar(cut, Nres0, Nerr0, color=color)
ax2.tick_params(axis='y', labelcolor=color)
#ax2.axhline(y=popt[0], color = 'b', linestyle='--', linewidth = 3) #N value
#ax2.fill_between(cut, popt[0]+np.sqrt(pcov[0]), popt[0]-np.sqrt(pcov[0])) #N error
fig.tight_layout()
plt.rc("xtick", labelsize=25)
plt.rc("ytick", labelsize=30) 
plt.show()

#%%
#loop to have cosines in the same graph

cosines = np.array([cospt1[0], cospt1[1], cospt1[2], cospt1[3]])

def fit(x, A, C, N):
    return (A + C*i)*x**(-N)

for i in cosines:
    if (i == cospt1[0]):
        coss= cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
        popt, pcov = curve_fit(fit, (ss), dsigdts, p0=None, sigma=sigs, maxfev= 5000)
        plt.semilogy(ss, fit((ss), *popt), color = 'g', linestyle = '--')
    elif (i == cospt1[1]):
        coss= cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = 2*dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = 2*sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'b')
        popt, pcov = curve_fit(fit, (ss), dsigdts, p0=None, sigma=sigs, maxfev= 5000)
        plt.semilogy(ss, fit((ss), *popt), color = 'b', linestyle = '--')
    elif (i == cospt1[2]):
        coss= cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = 4*dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = 4*sig1[[(cospt1 == i)&(pt21>j)]]              
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[[(cospt1 == i)&(pt21>j)]]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'r')
        popt, pcov = curve_fit(fit, (ss), dsigdts, p0=None, sigma=sigs, maxfev= 5000)
        plt.semilogy(ss, fit((ss), *popt), color = 'r', linestyle = '--')
    elif (i == cospt1[3]):
        coss= cospt1[(cospt1 == i)&(pt21>j)]
        dsigdts = 8*dsigdt1[(cospt1 == i)&(pt21>j)]
        sigs = 8*sig1[(cospt1 == i)&(pt21>j)]
        ss = s1[(cospt1 == i)&(pt21>j)]
        pts = pt21[(cospt1 == i)&(pt21>j)]
        plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'orange')
        popt, pcov = curve_fit(fit, (ss), dsigdts, p0=None, sigma=sigs, maxfev= 5000)
        plt.semilogy(ss, fit((ss), *popt), color = 'orange', linestyle = '--')
        plt.legend(['$\cos \Theta = -0.15$','$\cos \Theta = -0.05$','$\cos \Theta = 0.05$' ,'$\cos \Theta = 0.15$' ],loc = 'lower left', fontsize = 19)
    
    plt.title(r'$\gamma p \rightarrow \eta p$', size=35)
    plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =30)
    plt.yscale('log')
    plt.xlabel('s [$GeV^2$]', size = 30)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=30, width=2.5, length=10)

#%%

PDF = PdfPages('allcutsetap1.pdf')

for j in alpha[:-5]:
    
    def fit(x, A, C, N):
        return (A+ C*i)*x**(-N)
    
    plt.figure(figsize=(18,9))
    for i in cosines:
        if (i == cospt1[0]):
            coss= cospt1[(cospt1 == i)&(pt21>j)]
            dsigdts = dsigdt1[(cospt1 == i)&(pt21>j)]
            sigs = sig1[(cospt1 == i)&(pt21>j)]
            ss = s1[(cospt1 == i)&(pt21>j)]
            pts = pt21[(cospt1 == i)&(pt21>j)]
            plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'g')
            popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
            plt.semilogy(ss, fit((ss), *popt), color = 'g', linestyle = '--')
        elif (i == cospt1[1]):
             coss= cospt1[(cospt1 == i)&(pt21>j)]
             dsigdts = 2*dsigdt1[(cospt1 == i)&(pt21>j)]
             sigs = 2*sig1[(cospt1 == i)&(pt21>j)]
             ss = s1[(cospt1 == i)&(pt21>j)]
             pts = pt21[(cospt1 == i)&(pt21>j)]
             plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'b')
             popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
             plt.semilogy(ss, fit((ss), *popt), color = 'b', linestyle = '--')
        elif (i == cospt1[2]):
             coss= cospt1[(cospt1 == i)&(pt21>j)]
             dsigdts = 4*dsigdt1[(cospt1 == i)&(pt21>j)]
             sigs = 4*sig1[[(cospt1 == i)&(pt21>j)]]
             ss = s1[(cospt1 == i)&(pt21>j)]
             pts = pt21[[(cospt1 == i)&(pt21>j)]]
             plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'r')
             popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
             plt.semilogy(ss, fit((ss), *popt), color = 'r', linestyle = '--')
        elif (i == cospt1[3]):
             coss= cospt1[(cospt1 == i)&(pt21>j)]
             dsigdts = 8*dsigdt1[(cospt1 == i)&(pt21>j)]
             sigs = 8*sig1[(cospt1 == i)&(pt21>j)]
             ss = s1[(cospt1 == i)&(pt21>j)]
             pts = pt21[(cospt1 == i)&(pt21>j)]
             plt.errorbar(ss, dsigdts, yerr = sigs, fmt= 'o', marker = 'v', color = 'orange')
             popt, pcov = curve_fit(fit, (ss), dsigdts, sigma=sigs, maxfev= 5000)
             plt.semilogy(ss, fit((ss), *popt), color = 'orange', linestyle = '--')
             plt.legend(['$\cos \Theta = -0.15$','$\cos \Theta = -0.05$','$\cos \Theta = 0.05$' ,'$\cos \Theta = 0.15$' ],loc = 'lower left', fontsize = 19)
 
    # legend with results of the fit of all cosine bins
    plt.legend(title = r'$N$ = %2.2f $\pm$  %2.2f ' %(N, N_err), title_fontsize = 20)
    # title, includes cut [GeV^2] and reduced chi-squared    
    plt.title(r'$\gamma  p \rightarrow \eta p$: $\frac{d\sigma}{dt}$=$(A + B \cos \Theta)s^{N \pm \delta N}$, $p_{T _{min}} ^2 = %2.2f}$, $\chi ^2 /df = %2.2f$' %(j,redchi)  , size = 25)
    # axis labels
    plt.ylabel(r'$\frac{d\sigma}{dt}$ $[\mu$bGeV$^{-2}]$', size =35)
    plt.xlabel('s [$GeV^2$]', size = 25)
    # semi-log scale
    plt.yscale('log')
    # size of the numbers in the axis
    plt.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
   
  
    PDF.savefig()
    
PDF.close()
