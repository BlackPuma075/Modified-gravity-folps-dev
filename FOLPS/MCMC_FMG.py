#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import FMG as FOLPS
import cosmo_class
import sys, os
import time
import emcee
import numpy as np
from random import random
from schwimmbad import MPIPool





fn = 'chains_MG/DESIY1_LRG1_fr0_ells02'
save_fn = 'LRG1_FMG_posteriors'
steps = 10000
tracer_params = {
    0: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_SGC_z0.4-0.6_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_SGC_z0.4-0.6_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_SGC_z0.4-0.6_default_FKP_lin_thetacut0.05.npy'
    },
    1: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_NGC_z0.4-0.6_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_NGC_z0.4-0.6_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_NGC_z0.4-0.6_default_FKP_lin_thetacut0.05.npy'
    }
}


# In[3]:


from pypower import PowerSpectrumMultipoles, BaseMatrix

power = PowerSpectrumMultipoles.load(tracer_params[0]['data_fn'])
k, data = power.select((0.02, 0.2))(ell=[0, 2], return_k=True, complex=False)

#Reshaping the data vector pk02=[p0,p2]
pk0 = data[0]
pk2 = data[1]
pk02 = np.concatenate((pk0,pk2))
pk02.shape


# In[4]:


from desilike.observables import ObservableCovariance

covariance = ObservableCovariance.load(tracer_params[0]['covariance_fn'])
covariance = covariance.view(xlim=(0.02, 0.2), projs=[0, 2])


# In[5]:


ells = [0, 2]
wmatrix = BaseMatrix.load(tracer_params[0]['wmatrix_fn'])
wmatrix.select_x(xinlim=(0.001, 0.35), xoutlim=(0.02, 0.2))
wmatrix.select_proj(projsout=[(ell, None) for ell in ells])
kin = wmatrix.xin[0]
kout = wmatrix.xout[0]
z = wmatrix.attrs['zeff']
wmatrix = wmatrix.value.T


k_ev = kin


path_fits = 'chains_MG'

os.makedirs(str(path_fits), exist_ok=True)


OmAP = 0.310            #Fiducial value



Matrices = FOLPS.Matrices()


def FOLPS_LRGz1z3(h, omega_cdm, omega_b, logA_s, fR0,
                  b1p_LRGz1_NGC, b2p_LRGz1_NGC,  k_ev):
    
    global s8, s8_scale, fz, fz_scale
    
    "Fixed values: CosmoParams"
    #Omega_i = w_i/h² , w_i: omega_i
    Mnu = 0.06                         #Total neutrino mass [eV]
    omega_ncdm = Mnu/93.14;            #massive neutrinos 
    A_s = np.exp(logA_s)/(10**10);     #A_s = 2.04e-09;  
    n_s = 0.9649;                      #spectral index (Planck)
    omega_ncdm = 0;
    z1_pk = 0.51;         #z evaluation
    fR0 = 1e-5
    CosmoParams = [z1_pk, omega_b, omega_cdm, omega_ncdm, h, fR0]
    #Generate the power spectrum for this input params
    ps = cosmo_class.run_class(h=h, ombh2=omega_b, omch2=omega_cdm, omnuh2=omega_ncdm, 
                           As=A_s, ns=n_s, z=z1_pk, z_scale=None,spectra='cb')
    inputpkT = np.array([ps['k'], ps['pk'] ])
    
    LoopCorrections = FOLPS.NonLinear(inputpkl = inputpkT, CosmoParams = CosmoParams, EdSkernels = False, fR0 = fR0, nk = 120)
    s8 = ps['s8']
    s8_scale = ps['s8_scale']
    
    fz = ps['fz']
    fz_scale = ps['fz_scale']               
    
    def Pk_convolved(WM, b1p, b2p, z_pk, sig8_zpk, f0, growth_D_zold = 1.0, growth_D_znew = 1.0, z_bin = '', cap = ''):
        
        #aca el sigma es para cada redshift
        b1 = b1p / sig8_zpk
        b2 = b2p / sig8_zpk**2
        
        "Fixed values: NuisanParams"
        bs2 = -4/7*(b1 - 1);            #coevolution   
        b3nl = 32/315*(b1 - 1);         #coevolution  
        alpha0 = 23;                    #only for reference - does not affect the final result
        alpha2 = -58;                   #only for reference - does not affect the final result 
        alpha4 = 0.0;                       
        ctilde = 0.0;                   #degenerate with alphashot2
        alphashot0 = -0.06;             #only for reference - does not affect the final result
        alphashot2 = -6.38;             #only for reference - does not affect the final result    
        pshotp = 10000;                 #degenerate with alphashot0,2 (value from Florian (e)BOSS)  
        NuisanParams = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, pshotp]
    
    
        ##Pℓ,const
        Pkl024_const = FOLPS.RSDmultipoles(k_ev, z_pk = z_pk, f0=f0,
                                          NuisanParams = NuisanParams,
                                          Omfid = OmAP, AP = True)
        
        ##Pℓ,i=∂Pℓ/∂α_i
        Pkl024_i = FOLPS.RSDmultipoles_marginalized_derivatives(k_ev, z_pk = z_pk, f0=f0,
                                                              NuisanParams = NuisanParams,
                                                              growth_D_zold= growth_D_zold,
                                                              growth_D_znew= growth_D_znew,
                                                              Omfid = OmAP, AP = True, Hexa = True)
        
        #P = [P0,P2,P4]
        #convolved: Pℓ,const
        Pkl024 = np.concatenate((Pkl024_const[1], Pkl024_const[2], Pkl024_const[3]))
        convolved_Pkl024_const = WM @(Pkl024) 
        
        
        #Selection of the pk points corresponding to the k vector
        indices = []
        for i in k:
            index = np.where(np.isclose(kout, i, atol=1e-3))[0]
            if len(index) > 0:
                indices.append(index[0])
        indices = np.array(indices)
        ind = np.concatenate((indices,180+indices))
        
        convolved_Pkl02_const = convolved_Pkl024_const[ind]
        
        #convolved: Pℓ,i
        Pkl024_i_joint = np.concatenate((Pkl024_i), axis = 1)
        convolved_Pkl024_i = np.zeros((len(Pkl024_i_joint), len(WM)))
        for ii in range(len(Pkl024_i_joint)):
            convolved_Pkl024_i[ii, :] = WM @ Pkl024_i_joint[ii]
            convolved_Pkl02_i = convolved_Pkl024_i[ii,:][ind]   
        
   
        
        return {'pl_const_'+str(z_bin)+'_'+str(cap)+'':convolved_Pkl02_const,
                'pl_i_'+str(z_bin)+'_'+str(cap)+'':convolved_Pkl02_i}
    
    Pkl024_WF_LRGz1_NGC = Pk_convolved(WM=wmatrix,
                                       b1p=b1p_LRGz1_NGC, b2p=b2p_LRGz1_NGC,
                                       z_pk=z1_pk, 
                                       sig8_zpk=ps['s8'],
                                       f0=ps['fz'], z_bin='z1', cap='NGC')    
    
    
    return Pkl024_WF_LRGz1_NGC#, Pkl024_WF_LRGz1_SGC, Pkl024_WF_LRGz3_NGC, Pkl024_WF_LRGz3_SGC


# In[22]:


#priors range
h_min = 0.5; h_max = 0.9   
ocdm_min = 0.05; ocdm_max = 0.2                           
##oncdm_min = 1e-5 ; oncdm_max = 0.0222   #M_\nu :(0.0009 - 2 eV)
#omega_b: Gaussian prior
ob_min = 0.0170684; ob_max = 0.0270684
oncdm_min = 1e-5 ; oncdm_max = 0.10736/2   #(0.0009 - 5 eV)
logAs_min = 2.0; logAs_max = 4.0 
fR0_min = 0.0; fR0_max = 0.1
b1_min = 1e-5; b1_max = 10 
b2_min = -50;  b2_max = 50
#alpha0_min = -400; alpha0_max = 400
#alpha2_min = -400; alpha2_max = 400
#alpha4_min = -400; alpha4_max = 400
#alphashot0_min = -200; alphashot0_max = 200
#alphashot2_min = -200; alphashot2_max = 200


def log_prior(theta):
    ''' The natural logarithm of the prior probability. '''

    lp = 0.
    
    # unpack the model parameters
    (h, omega_cdm, omega_b, logA_s, fR0,b1p_LRGz1_NGC, b2p_LRGz1_NGC) = theta
    
    """set prior to 1 (log prior to 0) if in the range+
       and zero (-inf) outside the range"""
    
    #uniform (flat) priors
    if (h_min < h < h_max and
        ocdm_min < omega_cdm < ocdm_max and
        logAs_min < logA_s < logAs_max and
        fR0_min < fR0 < fR0_max and
        #Nuicances: LRGz1 NGC
        b1_min < b1p_LRGz1_NGC < b1_max and
        b2_min < b2p_LRGz1_NGC < b2_max):
        
        lp = 0.
        
    
    else:
        lp = -np.inf    
        
    # Gaussian prior on omega_b
    omega_bmu = 0.02237              # mean of the Gaussian prior - Planck/Ivanov et al
    omega_bsigma = 0.00037           # standard deviation of the Gaussian prior - BBN
    lp -= 0.5*((omega_b - omega_bmu)/omega_bsigma)**2
    
    return lp


# In[20]:


def log_likelihood(theta, Hexa = False):
    '''The natural logarithm of the likelihood.'''
    
    # unpack the model parameters
    (h, omega_cdm, omega_b, logA_s, fR0,
     #LRGz1 NGC
     b1p_LRGz1_NGC, b2p_LRGz1_NGC) = theta
    
    # condition to evaluate the model 
    if (h_min < h < h_max and
        ocdm_min < omega_cdm < ocdm_max and
        ob_min < omega_b < ob_max and
        logAs_min < logA_s < logAs_max and
        fR0_min < fR0 < fR0_max and
        #LRGz1 NGC, SGC
        b1_min < b1p_LRGz1_NGC < b1_max): 
        
        #evaluate the model    
        md = FOLPS_LRGz1z3(h, omega_cdm, omega_b, logA_s, fR0,
                           b1p_LRGz1_NGC, b2p_LRGz1_NGC, k_ev)
        
        md_LRGz1_NGC = md
        
        
        md_LRGz1_const_NGC = md_LRGz1_NGC['pl_const_z1_NGC'];
        
        #if Hexa == False:
            #delete the array for alpha_4, no hexa (delete third row of model['pl02_i'])
        #    md_LRGz1_i_NGC = np.delete(md_LRGz1_NGC['pl_i_z1_NGC'], 2, 0);
        #else:
        md_LRGz1_i_NGC_ = md_LRGz1_NGC['pl_i_z1_NGC']    
        md_LRGz1_i_NGC = md_LRGz1_i_NGC_[:, np.newaxis].T
        #L0    
        
        L0_LRGz1_NGC = FOLPS.compute_L0(Pl_const=md_LRGz1_const_NGC, 
                                  Pl_data=pk02, 
                                  invCov=covariance)
        
        
        #L1i
        
        L1i_LRGz1_NGC = FOLPS.compute_L1i(Pl_i=md_LRGz1_i_NGC, Pl_const=md_LRGz1_const_NGC,
                                    Pl_data=pk02, invCov=covariance)
        
        #L2ij
        
        L2ij_LRGz1_NGC = FOLPS.compute_L2ij(Pl_i=md_LRGz1_i_NGC, invCov=covariance)
        
        #compute the inverse of L2ij
        invL2ij_LRGz1_NGC = np.linalg.inv(L2ij_LRGz1_NGC)
        
        #compute the determinat of L2ij
        detL2ij_LRGz1_NGC = np.linalg.det(L2ij_LRGz1_NGC)
                        
        
        #marginalized likelihood
        term1_LRGz1_NGC = FOLPS.startProduct(L1i_LRGz1_NGC, L1i_LRGz1_NGC, invL2ij_LRGz1_NGC)
        
        term2_LRGz1_NGC = np.log(abs(detL2ij_LRGz1_NGC))
        
        L_marginalized_LRGz1_NGC = (L0_LRGz1_NGC + 0.5 * term1_LRGz1_NGC - 0.5 * term2_LRGz1_NGC) 
        
        L_marginalized = (L_marginalized_LRGz1_NGC)              
        
    else:
        L_marginalized = 10e10
   
    # return the log likelihood
    return L_marginalized


# In[13]:


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# In[24]:


# #### Initial positions:  
start0 = np.array([0.667,    #h
                   0.1169,   #ocdm
                   0.02232,  #ob
                   3.1002,   #logAs
                   1e-5,     #fR0
                   ###### LRGz1 NGC ######
                   1.8608*0.6,   #b1_LRGz1_NGC
                   -0.7811*0.6**2,  #b2_LRGz1_NGC
                  ])


ndim = len(start0) # Number of parameters/dimensions
nwalkers = 2*ndim # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = steps # Number of steps/iterations.

#Advice from: arxiv: 1202.3665, pag 10
start = np.array([start0 + 1e-3*np.random.rand(ndim) for i in range(nwalkers)])


# In[ ]:


#Set up convergence
max_n = nsteps
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf



with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
backend = emcee.backends.HDFBackend(fn)
        
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,pool = pool, backend=backend)
    # Now we'll sample for up to max_n steps
for sample in sampler.sample(start, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue
            
            
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
        
        # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau


# In[ ]:


np.savetxt('autocorr_index'+str(index)+'.dat', np.transpose([autocorr[:index]]), 
           header = 'index ='+str(index)+',  mean_autocorr')

print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
)


print('All computations have been completed successfully')


chain = backend.get_chain(discard = 0)


from cosmoprimo.fiducial import DESI

cosmo = DESI()


plt.rcParams['figure.dpi'] = 200
import getdist
import IPython
from getdist import plots, MCSamples

labels = [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{10^{10}A_s}$', r'fR0', r'$b_1$', r'$b_2$']

samples = MCSamples(samples=chain, names = labels)# ranges={r'$fR0$':(1e-10, 0.1)})
#plt.rcParams['text.usetex'] = True
s = samples.copy(settings={'mult_bias_correction_order':1,
                       'smooth_scale_2D':0.55, 
                       'smooth_scale_1D':0.55})
g = plots.get_subplot_plotter()

g.triangle_plot([s], 
                [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{10^{10}A_s}$',r'fR0',r'$b_1$'], 
                filled=[True], alpha = [1],
               contour_colors=['darkgreen'],
                contour_lws = [1.5], contour_ls = ['-'],marker_args = ({'color':'k','lw' :  1.2}),
               markers = [cosmo['h'], cosmo['omega_cdm'], cosmo['omega_b'], cosmo['logA'], cosmo['n_s']],
               legend_labels = [r'LRGz0'], legend_loc = 'upper right')

plt.suptitle(r'LRG1_NGC FMG', fontsize = 16);
plt.savefig(save_fn)

