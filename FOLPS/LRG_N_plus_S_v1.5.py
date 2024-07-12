#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


from desilike.theories.galaxy_clustering import FOLPSAXTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate, FixedPowerSpectrumTemplate

template = DirectPowerSpectrumTemplate( fiducial = 'DESI', z = 0.5)
theory = FOLPSAXTracerPowerSpectrumMultipoles(template = template, prior_basis = 'physical')


# In[6]:


template.params['n_s'].update(fixed = False)
template.params['n_s'].update(prior={'dist':'norm','loc':0.9649, 'scale':0.048}) #Planck width x10
theory.params['b3p'].update(fixed = False)


# In[7]:


#Data from Y1 clustering products

import numpy as np
from pypower import PowerSpectrumMultipoles, BaseMatrix
from desilike.observables import ObservableCovariance

data_fn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_SGC_z0.4-0.6_thetacut0.05.npy'
wmatrix_fn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_SGC_z0.4-0.6_thetacut0.05.npy'
covariance_fn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_SGC_z0.4-0.6_default_FKP_lin_thetacut0.05.npy'


# In[8]:


covariance = ObservableCovariance.load(covariance_fn)
covariance = covariance.select(xlim = (0.02, 0.2), projs = [0,2])


# In[9]:


from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable

observable = TracerPowerSpectrumMultipolesObservable(data=data_fn, 
                                                     covariance=covariance,
                                                     klim={ell: [0.02, 0.2, 0.005] for ell in [0,2]},
                                                     theory=theory,
                                                     wmatrix = wmatrix_fn,
                                                     kin = np.arange(0.001, 0.8, 0.001),
                                                     )


# In[10]:


observable.plot_covariance_matrix();


# In[11]:


from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import setup_logging

setup_logging()

likelihood = ObservablesGaussianLikelihood(observables = [observable])


# In[12]:


template2 = DirectPowerSpectrumTemplate( fiducial = 'DESI')
theory2 = FOLPSAXTracerPowerSpectrumMultipoles(template = template,tracer = 'LRG', prior_basis = 'physical')


# In[13]:


data_fn_2 = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_NGC_z0.8-1.1_thetacut0.05.npy'
wmatrix_fn_2 = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_NGC_z0.8-1.1_thetacut0.05.npy'
covariance_fn_2 = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_NGC_z0.8-1.1_default_FKP_lin_thetacut0.05.npy'


# In[14]:


covariance_2 = ObservableCovariance.load(covariance_fn_2)
covariance_2 = covariance.select(xlim = (0.02, 0.2), projs = [0,2])


# In[15]:


observable2 = TracerPowerSpectrumMultipolesObservable(data=data_fn_2, 
                                                     covariance=covariance_2,
                                                     klim={ell: [0.02, 0.2, 0.005] for ell in [0,2]},
                                                     theory=theory2,
                                                     wmatrix = wmatrix_fn_2,
                                                     kin = np.arange(0.001, 0.8, 0.001),
                                                     )


# In[16]:


observable2.plot_covariance_matrix();


# In[17]:


likelihood2 = ObservablesGaussianLikelihood(observables = [observable2])


# In[18]:


from desilike.emulators import Emulator, EmulatedCalculator, TaylorEmulatorEngine

pt = EmulatedCalculator.load('Emulator/FOLPSAX_Taylor_order3_emulator.npy')


# In[19]:


observable.init.update(theory=pt)
observable2.init.update(theory=pt)


# In[20]:


from desilike.likelihoods import SumLikelihood

Likelihood = SumLikelihood(likelihoods = (likelihood, likelihood2))

Likelihood()


# In[21]:


from desilike.samplers import EmceeSampler

sampler = EmceeSampler(Likelihood,seed = 42,save_fn = 'Chains/LRG_N_plus_S_script')
sampler.run(check={'max_eigen_gr': 0.05})


# In[36]:


from desilike.samples import Chain, plotting

chain = Chain.load('Chains/LRG_N_plus_S_script.npy').remove_burnin(0.3)


# In[ ]:


a = np.array(chain)
a = a[0:14, ::, ::]


import getdist
import IPython
from getdist import plots, MCSamples



labels = [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{10^{10}A_s}$', r'$n_s$', r'$b_1$', r'$b_2$', r'$b_s$', r'$b_3$', r'$\alpha_0$', r'$\alpha_2$', r'$\alpha_4$', r'$sn_0$', r'$sn_2$']

samples = MCSamples(samples=a.T, names = labels)# ranges={r'$fR0$':(1e-10, 0.1)})
#plt.rcParams['text.usetex'] = True
s = samples.copy(settings={'mult_bias_correction_order':1,
                       'smooth_scale_2D':0.55, 
                       'smooth_scale_1D':0.55})

g = plots.get_subplot_plotter()

g.triangle_plot([s], 
                [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{10^{10}A_s}$', r'$n_s$'], 
                filled=True,
               contour_colors=['orange', 'green'],
               markers = [cosmo['h'], cosmo['omega_cdm'], cosmo['omega_b'], cosmo['logA'], cosmo['n_s']],
               legend_labels = [r'LRG_NGC+SGC'])
#plt.suptitle(r'LRG_NGC', fontsize = 20);


plt.savefig('Graphs/Fit_DESIY1_LRG_NGCplusSGC')



labels = [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{10^{10}A_s}$', r'$n_s$', r'$b_1$', r'$b_2$', r'$b_s$', r'$b_3$', r'$\alpha_0$', r'$\alpha_2$', r'$\alpha_4$', r'$sn_0$', r'$sn_2$']

samples = MCSamples(samples=a.T, names = labels)# ranges={r'$fR0$':(1e-10, 0.1)})
#plt.rcParams['text.usetex'] = True
s = samples.copy(settings={'mult_bias_correction_order':1,
                       'smooth_scale_2D':0.55, 
                       'smooth_scale_1D':0.55})

g = plots.get_subplot_plotter()
g.triangle_plot([s], 
                [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{10^{10}A_s}$', r'$n_s$', r'$b_1$', r'$b_2$', r'$b_s$', r'$b_3$', r'$\alpha_0$', r'$\alpha_2$', r'$\alpha_4$', r'$sn_0$', r'$sn_2$'], 
                filled=True,
               contour_colors=['orange'],
               markers = [cosmo['h'], cosmo['omega_cdm'], cosmo['omega_b'], cosmo['logA'], cosmo['n_s']],
               legend_labels = ['LRG_NGC+SGC'])
#plt.suptitle(r'LRG_NGC', fontsize = 20);


plt.savefig('Graphs/Fit_DESIY1_LRG_NGCplus_SGC')


# In[ ]:


labels = [r'$h$', r'$\omega_{cdm}$', r'$\omega_b$', r'$\log{A}$', r'$n_s$', r'$b_1$', r'$b_2$', r'$b_s$', r'$b_3$', r'$\alpha_0$', r'$\alpha_2$', r'$\alpha_4$', r'$sn_0$', r'$sn_2$']
fig, axes = plt.subplots(len(labels), figsize=(10, 40), sharex=True)
for i in range(14):
    ax = axes[i]
    ax.plot(a[i,:, :], "k", alpha=0.3)
    ax.set_xlim(0, len(a[0,:,0]))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig('Graphs/Walkers_LRG_NGCplusSGC')