# %%
#Limiting the number of threads before importing numpy/scipy for multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"

#importing all the packages
import scipy
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import time
import combined_mcmc_functions_final_ls as sf
import importlib
importlib.reload(sf) 

#constants
gev_to_cm2 = (2e-14)**2 #converts cross-section from GeV^-2 to cm^2
U_e1 = 0.8232455344549402  
U_e3 = 0.14842506526863986
U_m1 = 0.3282301021033085
U_m3 = 0.7479297025790592


#Degrees of freedom

#---old---
#df_dof = pd.read_table('gstar.txt', skiprows=0,names=['Temp', 'g_s', 'g_ss'])
#df_dof['Temp'] = 1e-9*df_dof['Temp'] #eV to GeV

#---new---
df_dof = pd.read_table('eos2020.dat', skiprows=0,sep=" ",names=['Temp', 'g_ss', 'g_s'])

#Linear temperature arrays
temperature = df_dof['Temp'].to_numpy()
g_s = df_dof['g_s'].to_numpy()
g_ss = df_dof['g_ss'].to_numpy()

#Log temperature arrays
temperature_ln = np.log(temperature)
g_s_ln = np.log(g_s)
g_ss_ln = np.log(g_ss)

g_s_new = interpolate.interp1d(temperature,g_s) #be careful of extrapolation! should be ok for high temp.
g_ss_new = interpolate.interp1d(temperature,g_ss)

g_s_ln_new = interpolate.CubicSpline(temperature_ln,g_s_ln)
g_ss_ln_new = interpolate.CubicSpline(temperature_ln,g_ss_ln)

# %%
#Neff constraints
df_neff = pd.read_table('Neff_chi_sq.txt', skiprows=0, names=['mx','chi_sq'])
df_neff['log10(mx)'] = np.log10(1e-3*df_neff['mx'])   #from MeV to GeV

#creating arrays for interpolation
neff_mx = df_neff['log10(mx)'].to_numpy()
neff_chi_sq = df_neff['chi_sq'].to_numpy()

#interpolation for chi^2
neff_chi_sq_int = interpolate.interp1d(neff_mx,neff_chi_sq)

# %%
#s-wave <sv> constraints
df_swave = pd.read_csv('SK-ve-limits.csv',skiprows=0,names=['mx','sigmav'])
df_swave['log10(mx)'] = np.log10(df_swave['mx'])

#creating arrays for interpolation
swave_mx = df_swave['log10(mx)'].to_numpy()
swave_sigmav = df_swave['sigmav'].to_numpy()

#interpolation for chi^2
swave_sigmav_int = interpolate.interp1d(swave_mx,swave_sigmav,kind='linear')

# %%
#NuFit delta_m31 constraints
df_deltam31sq_chi_2 = pd.read_table('deltam31_chi_2', skiprows=1,sep=" ",names=['delta_m31sq', 'delta_chi2'])
df_deltam31sq_chi_2['delta_m31sq'] = 1e-3*df_deltam31sq_chi_2['delta_m31sq'] #in eV^2
df_deltam31sq_chi_2

#creating arrays for interpolation
deltam31sq = df_deltam31sq_chi_2['delta_m31sq'].to_numpy()
deltam31sq_chi_sq = df_deltam31sq_chi_2['delta_chi2'].to_numpy()

#interpolation for chi^2
deltam31sq_chi_sq_int = interpolate.interp1d(deltam31sq,deltam31sq_chi_sq)

# %%
import emcee
import corner

#models
def scattering_model(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    #constrain on the opacity: cross-section/dm mass
    sigmav = sf.sigmav_sc(Tnu,10**mx,10**mphi1,10**del_mphi,10**mn,10**lam,10**Ynu) #Tnu temperature CNB today
    model = sigmav*gev_to_cm2/(10**mx) #cross-section in cm^2 and mass in GeV
    return model

def annihilation_model(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    model = sf.relic_density_swave(10**mx,10**mphi1,10**del_mphi,10**mn,10**lam,10**Ynu,g_s_new,g_ss_new,g_ss_ln_new,gx)
    return model

def nu_mass_model(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    model = sf.mnu(10**mphi1,10**del_mphi,10**mn,10**Ynu)
    return model

# def w_decay_model(theta):
#     mx, mphi1, del_mphi, mn, lam, Ynu = theta
#     model = sf.w_decay_width(10**mphi1,10**mn,10**Ynu)
#     return model

# def D_decay_model(theta):
#     mx, mphi1, del_mphi, mn, lam, Ynu = theta
#     #model = sf.m_decay_width(m_D,f_D,10**mn,10**Ynu)
#     model = 10**Ynu
#     return model

def nu_mass_model(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    model = 1e9*sf.mnu(10**mphi1,10**del_mphi,10**mn,10**Ynu) #convert from GeV to eV
    return model

def z_decay_model(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    model = sf.z_decay_width(10**mphi1,10**mn,10**Ynu)
    return model

def K_decay_model(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    #model = sf.m_decay_width(m_K,f_K,10**mn,10**Ynu)
    model = 10**Ynu
    return model

#log likelihoods to get full posterior
def ln_likelihood(theta, y1, y2, yerr1, yerr2, yerr3, yerr4, yerr5, yerr_ke, yerr_km, yerr_z, neff_chi_sq_int, swave_sigmav_int):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    
    yu_model = scattering_model(theta)
    yra_model,ysv_model = annihilation_model(theta)
    m3_model = nu_mass_model(theta) #in eV
    deltam31sq_model = m3_model**2-m1**2
    
    #for underabundant dark matter
    f_fraction = yra_model/0.12
    #ysv_model_rescaled = (1/f_fraction)**2*ysv_model
    
#     if yra_model >= y1:
#         lnlike_ra = -0.5*np.sum(((y1-yra_model)/yerr1)**2)
#     else:
#         lnlike_ra = 0
    lnlike_ra = -0.5*((y1-yra_model)/yerr1)**2
    
    if -4<=theta[0]<=np.log10(0.02):
            lnlike_neff = -0.5*neff_chi_sq_int(theta[0])
    else:
            lnlike_neff = 0
            
    if yu_model >= y2:
        lnlike_u = -0.5*np.sum(((y2-yu_model)/0.5*yerr2)**2) #95% CL
    else:
        lnlike_u = 0

#     if ynu_model >= y2:
#         lnlike_nu = -0.5*np.sum(((y2-ynu_model)/(0.5*yerr3))**2)   #95% CL = 2 sigma
#     else:
#         lnlike_nu = 0
    if 0.2e-3<=deltam31sq_model<=7e-3:#in eV
            lnlike_deltam31sq = -0.5*deltam31sq_chi_sq_int(deltam31sq_model)
            #print('inside', lnlike_deltam31sq, np.exp(lnlike_deltam31sq))
    else:
            lnlike_deltam31sq = -np.inf #outside of this range is totally constrained, not 0
            #print('outside', lnlike_deltam31sq, np.exp(lnlike_deltam31sq))
    
    if ysv_model >= y2:
        if -2.0292883347469104<=theta[0]<=-1.5023866744647538:
            lnlike_sv = -0.5*np.sum(((y2-3*U_e3**4*ysv_model)/(1/1.645*swave_sigmav_int(theta[0])))**2)  #90% CL = 1.645 sigma
            #print(r'sv constrained: ', np.exp(lnlike_sv))
        else:
            lnlike_sv = 0
    else: 
        lnlike_sv = 0
        
#     if 10**theta[1]+10**theta[3]<mw:
#         yw_model = w_decay_model(theta)
#         if yw_model >= y2:
#             lnlike_w = -0.5*np.sum(((y2-yw_model)/(0.5*yerr4))**2)  #95% CL
#         else:
#             lnlike_w = 0
#     else: 
#         yw_model = 0
#         lnlike_w = 0 
    
    if 10**theta[1]+10**theta[3]<mz:
        yz_model = z_decay_model(theta)
        if yz_model >= y2:
            lnlike_z = -0.5*np.sum(((y2-yz_model)/(0.5*yerr_z))**2)  #95% CL
        else:
            lnlike_z = 0
    else: 
        yz_model = 0
        lnlike_z = 0
    
#     if m_K<10**theta[1]+10**theta[3]<m_D:
#         yD_model = D_decay_model(theta)
#         if yD_model >= y2:
#             lnlike_D = -0.5*np.sum(((y2-yD_model)/(1/1.645*yerr5))**2)  #90% CL
#         else:
#             lnlike_D = 0
#     else: 
#         yD_model = 0
#         lnlike_D = 0 
        
    if 10**theta[1]+10**theta[3]<m_K:
        yK_model = K_decay_model(theta)
        if yK_model >= y2:
            lnlike_Ke = -0.5*np.sum(((y2-yK_model*U_e3)/(1/1.645*yerr_ke))**2) 
            lnlike_Km = -0.5*np.sum(((y2-yK_model*U_m3)/(1/1.645*yerr_km))**2) #90% CL
            lnlike_K = lnlike_Ke + lnlike_Km
        else:
            lnlike_K = 0
    else: 
        yK_model = 0
        lnlike_K = 0 
        
    ln_like_total = lnlike_ra + lnlike_u + lnlike_sv + lnlike_neff + lnlike_deltam31sq + lnlike_z + lnlike_K
    return ln_like_total, lnlike_ra, lnlike_u, lnlike_sv, lnlike_neff, lnlike_deltam31sq, lnlike_z, lnlike_K, yra_model, ysv_model, yu_model, m3_model, yz_model, yK_model

# %%
def ln_prior(theta):
    mx, mphi1, del_mphi, mn, lam, Ynu, m1 = theta
    mphi2_lin = 10**mphi1*(1-10**del_mphi)
    
    if 1.1>m1>=0 and 10**mn>10**mphi1>=mphi2_lin>=10**mx and -4<=mx<=8 and -4<=mphi1<=8 and -16<=del_mphi<=0 and -4<=mn<=8 and -7<=lam<=np.log10(4*np.pi) and -7<=Ynu<=np.log10(np.sqrt(4*np.pi)):
        lnlike_nu1 = -0.5*np.sum(((0-m1)/(0.5*0.037))**2)    #mnu1 posterior now used as prior 95% CL, in eV
        lnlike_all = 0 #log-flat prior
    else:
        lnlike_nu1 = -np.inf #no negative masses
        lnlike_all = -np.inf
    return lnlike_nu1 + lnlike_all

# %%
def ln_posterior(theta, y1, y2, yerr1, yerr2, yerr3, yerr4, yerr5, yerr_ke, yerr_km, yerr_z, neff_chi_sq_int, swave_sigmav_int):
    #print('posterior: ', theta)
    log_prior = ln_prior(theta) #using log-flat prior
    
    if np.isinf(log_prior):
        return log_prior, np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
    else:
        log_like = ln_likelihood(theta, y1, y2, yerr1, yerr2, yerr3, yerr4, yerr5, yerr_ke, yerr_km, yerr_z, neff_chi_sq_int, swave_sigmav_int)
        return log_prior + log_like[0], log_like


# **Parallelizing the code with Pool**

# now let's set up the MCMC chains
ndim = 7         #N-dimensinal parameter space
nwalkers = 1000  #even more walkers, even less steps
steps = 20000    #2x original steps


# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "scalar_dm_saturated_heavy_result.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

#define and initialize model parameters

#constants
gx = 1                  #Real scalar DM coupling
mw = 80.377             #W boson mass in GeV, PDG 2022 fit
mz = 91.1876            #Z boson mass in GeV, PDG 2023 fit
Tnu = 1.68e-13          #CNB temperature today in GeV
m_K = 0.493677          #Kaon mass in GeV, PDG 2022
m_D = 1.86966           #D meson mass in GeV, PDG 2022
al = 7.2973525693e-3    #fine structure constant, PDG 2022
sinw = np.sqrt(0.23121) #weak-mixing angle, PDG 2022
cosw = np.sqrt(1 - 0.23121)
h_bar = 6.582119569e-25 #GeV s
c = 2.99792458e+10 #cgs
cgs_to_gev = 1/(h_bar**2*c**3)

#scan parameter space
y1 = 0.12
y_dm = 0.12                  #DM relic abundance from Planck 2018
yerr1 = 0.0012         #upper limit 68% C.L. Planck TT, TE, EE + lowE + lensing
y2 = 0                       #assuming best fit centered at y=0
#yerr2 = 0.54e-13            #68% CL on 10^13 u_0 from Wilkinson et al 2014 for fixed Neff
#yerr2 = 1e-45               #68% CL on sigmav/mdm Lyman-alpha constraint from Wilkinson eq. 13 
yerr2 = 1e-46                #95% CL on sigmav/mdm dSph bound from Akita & Ando
yerr3 = 0.037e-9             #95% CL on the lightest neutrino mass in GeV for NO, GAMBIT paper
yerr4 = 0.042                #95% CL on the maximum contribution to the W boson decay width in GeV, PDG 2022
yerr5 = 0.4                  #90% CL upper limit on Ynu coupling to electron neutrinos from D meson decay, Alvey and Fairbairn 2019
yerr_ke = 3e-3               #90% CL upper limit on Ynu coupling to electron neutrinos from Kaon decay, Alvey and Fairbairn 2019
yerr_km = 1e-2               #90% CL upper limit on Ynu coupling to muon neutrinos from Kaon decay, Alvey and Fairbairn 2019
yerr_z = 0.0023              #95% CL on the maximum contribution to the Z boson decay width in GeV, PDG 2023
mx_i = -3
mphi1_i = -2
del_mphi_i = -9
mn_i = -1   
lam_i = np.log10(0.5)
Ynu_i = np.log10(5e-3)

#theta_i = [mx_i,mphi1_i,del_mphi_i,mn_i,lam_i,Ynu_i] #manually initializing values in low probability area
mle_manual = [-1,5.9,-12,6,-2,-1,0]

# import scipy.optimize as opt
# nll = lambda *args: -ln_likelihood(*args)[0]
# lb = [-3,1,-10,2,-7,-7]
# ub = [0,2,-2,3,1,1]
# result = opt.minimize(nll, theta_i,
#                       args=(y1, y2, yerr1, yerr2, yerr3, yerr4, yerr5, yerr6, neff_chi_sq_int, swave_sigmav_int),
#                       bounds=opt.Bounds(lb,ub))
# mle = result['x']
# print('Maximum likelihood estimation',mle, 10**mle)

# initialize the walkers to the vicinity of the parameters
#pos = [mle + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

#initializing in a larger vicinity
pos = [mle_manual + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

from multiprocessing import Pool

# initialze the sampler
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1)) #CPUs in 1 node
print('Number of CPUs ', ncpus)

with Pool(processes=ncpus) as pool:
    sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_posterior,blobs_dtype=float,args=(y1, y2, yerr1, yerr2, yerr3, yerr4, yerr5, yerr_ke, yerr_km, yerr_z, neff_chi_sq_int, swave_sigmav_int),pool=pool, backend=backend) 
    start = time.time()
    sampler.run_mcmc(pos, steps, progress=True)
    samples = sampler.chain
    end = time.time()
    multi_time = end-start