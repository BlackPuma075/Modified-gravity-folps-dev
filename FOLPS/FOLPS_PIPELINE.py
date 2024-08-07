#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import FOLPSAXTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable 
from desilike.observables import ObservableCovariance
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging
from desilike.samplers import EmceeSampler


# In[ ]:


#List of tracers
tracers = ['LRG1_SGC', 'LRG1_NGC'] #Add tracers as needed
prior = 'physical' #Specify the prior
chain_name = 'Chains/LRG1_NGC_SGC_ell02_kmax0.2_sigm8_Omega_m_v3' #fn to save the chain

#File paths for each tracer (same order as tracers list)
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


# In[ ]:


#Defining a cosmology to get sigma_8 and Omega_m
cosmo = Cosmoprimo(engine='class')
#cosmo = template.cosmo
cosmo.init.params['sigma8_m'] = dict(derived=True) 
cosmo.init.params['Omega_m'] = dict(derived=True)


# In[ ]:


#Update cosmo priors
for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA', 'tau_reio']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.048}) 
    if param == 'omega_b':
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.1,1.0]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.01, 0.99]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})    


# In[ ]:


#Defining tracer types and their corresponding redshifts
tracer_redshifts = {
    'LRG1_NGC': 0.5, 'LRG1_SGC': 0.5,
    'LRG2_NGC': 0.7, 'LRG2_SGC': 0.7,
    'LRG3_NGC': 0.9, 'LRG3_SGC': 0.9,
    'QSO_NGC': 1.5, 'QSO_SGC': 1.5,
    'ELG_NGC': 1.3, 'ELG_SGC': 1.3,
    'BGS_NGC': 0.3, 'BGS_SGC': 0.3
}

#Initialize an empty list to store the theory objects
theories = []
fiducial = DESI()
#Iterate over each tracer and create the corresponding theory object
for tracer in tracers:
    if tracer in tracer_redshifts:
        z = tracer_redshifts[tracer]
    else:
        print(f'Invalid tracer: {tracer}')
        continue
    #Create the template and theory
    template = DirectPowerSpectrumTemplate(cosmo = cosmo, z=z)
    theory = FOLPSAXTracerPowerSpectrumMultipoles(template=template, prior_basis=prior)
    
    #Updating free parameters
    for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA']:
        template.params[param].update(fixed=False)
        if param == 'n_s':
            template.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.048})

    theory.params['b3p'].update(fixed=False)
    
    #Append the theory object to the list
    theories.append(theory)


# In[ ]:


for param in theory.all_params:
    print(param,':',theory.all_params[param].prior)


# In[ ]:


#Defining a function to create an observable
def create_observable(data_fn, wmatrix_fn, covariance_fn, theory, i):
    covariance = ObservableCovariance.load(covariance_fn)
    covariance = covariance.select(xlim=(0.02, 0.2), projs=[0, 2])
    
    #Create and return the observable
    return TracerPowerSpectrumMultipolesObservable(
        data=data_fn,
        covariance=covariance,
        klim={ell: [0.02, 0.2, 0.005] for ell in [0, 2]},
        theory=theories[i],
        wmatrix=wmatrix_fn,
        kin=np.arange(0.001, 0.35, 0.001),
    )

# Create observables for each tracer
observables = [create_observable(tracer_params[i]['data_fn'], tracer_params[i]['wmatrix_fn'], tracer_params[i]['covariance_fn'],theories, i) 
                for i in range(len(tracers))]


# In[ ]:


#Emulate and update each theory object
for i in range(len(theories)):
    theories[i] = observables[i].wmatrix.theory
    emulator = Emulator(theories[i].pt, engine=TaylorEmulatorEngine(method = 'finite', order = 2))
    #emulator.save('Emulator/FOLPSAX_mf_Taylor_o4_LRG1')
    emulator.set_samples()
    emulator.fit()
    
    theories[i].init.update(pt = emulator.to_calculator())


# In[ ]:


#Analytic marg.
for i in range(len(theories)): 
    for param in ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p','sn2p']:    
        theories[i].params[param].update(derived = '.marg')
#Update namespace of bias parameters (to have one parameter per tracer)
for i in range(len(theories)):    
    for param in theories[i].init.params:
        # Update names
        param.update(namespace='{}'.format(tracers[i]))   


# In[ ]:


setup_logging()
Likelihoods = [] #Empty list to store the likelihoods (per observable)
for i in range(len(theories)):
    Likelihoods.append(ObservablesGaussianLikelihood(observables = [observables[i]])) #Create a Gaussian likelihood and save it


# In[ ]:


likelihood = SumLikelihood(likelihoods = (Likelihoods)) #Sum of all likelihoods

likelihood() #Initialize


# In[ ]:


sampler = EmceeSampler(likelihood,save_fn = 'Chains/Omega_sigma_test_v5') #Sampling and saving the chain
sampler.run(check={'max_eigen_gr': 0.1}) #Convergence test 

