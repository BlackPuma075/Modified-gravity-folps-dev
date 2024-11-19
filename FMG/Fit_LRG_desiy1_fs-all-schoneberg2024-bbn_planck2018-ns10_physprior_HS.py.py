#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import FOLPSTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable 
from desilike.observables import ObservableCovariance
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging


# List of tracers
tracers = ['LRG1_SGC', 'LRG1_NGC', 'LRG2_NGC', 'LRG2_SGC', 'LRG3_NGC', 'LRG3_SGC']  # Add more tracers as needed
prior = 'physical' #Prior to be used
chain_name = 'Chains/LRG123_ell02_kmax0.17_fR0_corrected' #fn to save the chain


# Define file paths for each tracer (same order as tracers list)
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
    },
    2: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_NGC_z0.6-0.8.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_NGC_z0.6-0.8_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_NGC_z0.6-0.8_default_FKP_lin_thetacut0.05.npy'
    },
    3: {
       'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_SGC_z0.6-0.8.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_SGC_z0.6-0.8_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_SGC_z0.6-0.8_default_FKP_lin_thetacut0.05.npy'
    },
    4: {
        'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_NGC_z0.8-1.1_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_NGC_z0.8-1.1_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_NGC_z0.8-1.1_default_FKP_lin_thetacut0.05.npy'
    },
    5: {
       'data_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/corrected/pkpoles_corrected_LRG_SGC_z0.8-1.1_thetacut0.05.npy',
        'wmatrix_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/baseline_2pt/pk/wmatrix_smooth_LRG_SGC_z0.8-1.1_thetacut0.05.npy',
        'covariance_fn': '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/unblinded/desipipe/cov_2pt/ezmock/v1/covariance_power_LRG_SGC_z0.8-1.1_default_FKP_lin_thetacut0.05.npy' 
    }
}    


#Defining a cosmology to get sigma_8 and Omega_m
cosmo = Cosmoprimo(engine='class')
#cosmo.init.params['sigma8_m'] = dict(derived=True) 
#cosmo.init.params['Omega_m'] = dict(derived=True)
cosmo.init.params['fR0'] = dict(derived=False)
fiducial = DESI()

#Update cosmo priors
for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA', 'tau_reio', 'fR0']:
    cosmo.params[param].update(fixed = False)
    if param == 'tau_reio':
        cosmo.params[param].update(fixed = True)
    if param == 'n_s':
            cosmo.params[param].update(fixed = True)
            cosmo.params[param].update(value = 0.9649)
    if param == 'omega_b':
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
    if param == 'h':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.5,0.9]})
    if param == 'omega_cdm':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.05, 0.2]})
    if param == 'logA':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})
    if param == 'fR0':
        cosmo.params[param].update(prior = {'dist':'uniform','limits': [0, 9e-5]})
        cosmo.params[param].update(fixed = False) 
        cosmo.params[param].update(ref={'limits': [0, 9e-5]})       

# Define the tracer types and their corresponding redshifts
tracer_redshifts = {
    'LRG1_NGC': 0.510, 'LRG1_SGC': 0.510,
    'LRG2_NGC': 0.706, 'LRG2_SGC': 0.706,
    'LRG3_NGC': 0.930, 'LRG3_SGC': 0.930,
    'QSO_NGC': 1.491, 'QSO_SGC': 1.491,
    'ELG_NGC': 1.317, 'ELG_SGC': 1.317,
    'BGS_NGC': 0.295, 'BGS_SGC': 0.295
}

# Initialize an empty list to store the theory objects
theories = []

# Iterate over each tracer and create the corresponding theory object
for tracer in tracers:
    if tracer in tracer_redshifts:
        z = tracer_redshifts[tracer]
    else:
        print(f'Invalid tracer: {tracer}. Skipping.')
        continue

    # Create the template and theory objects
    template = DirectPowerSpectrumTemplate(fiducial = fiducial,cosmo = cosmo, z=z)
    theory = FOLPSTracerPowerSpectrumMultipoles(template=template, prior_basis=prior)
    
    # Update parameters
    for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA']:
        template.params[param].update(fixed=False)
        if param == 'n_s':
            template.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.048})

    theory.params['bsp'].update(fixed=True)
    theory.params['b1p'].update(prior = {'dist':'uniform','limits': [1e-5, 10]})
    theory.params['b2p'].update(prior = {'dist':'uniform','limits': [-50, 50]})
    theory.params['alpha0p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': 6.9})
    theory.params['alpha2p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': 6.9})
    theory.params['alpha4p'].update(prior={'dist': 'norm', 'loc': 0, 'scale': 6.9})
    
    # Append the theory object to the list
    theories.append(theory)

#Priors
for param in theory.all_params:
    print(param,':',theory.all_params[param].prior)


# In[12]:


#Define a function to create an observable
def create_observable(data_fn, wmatrix_fn, covariance_fn, theory, tracer_index):
    # Load and process covariance
    covariance = ObservableCovariance.load(covariance_fn)
    covariance = covariance.select(xlim=(0.02, 0.17), projs=[0, 2])
    
    # Create and return the observable
    return TracerPowerSpectrumMultipolesObservable(
        data=data_fn,
        covariance=covariance,
        klim={ell: [0.02, 0.17, 0.005] for ell in [0, 2]},
        theory=theories[tracer_index],
        wmatrix=wmatrix_fn,
        kin=np.arange(0.001, 0.35, 0.001),
    )

# Create observables for each tracer
observables = [create_observable(params['data_fn'], params['wmatrix_fn'], params['covariance_fn'],theories, i) 
                for i, params in tracer_params.items()]


# In[13]:


for i in range(len(theories)):
    theories[i] = observables[i].wmatrix.theory
    emulator = Emulator(theories[i].pt, engine=TaylorEmulatorEngine(method = 'finite', order = 4))
    #emulator.save('Emulator/FOLPSAX_mf_Taylor_o4_LRG1')
    emulator.set_samples()
    emulator.fit()
    
    theories[i].init.update(pt = emulator.to_calculator())

print('All theories have been emulated succesfully')
# In[14]:


for i in range(len(theories)): 
    for param in ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p','sn2p']:    
        theories[i].params[param].update(derived = '.marg')
# Update namespace of bias parameters (to have one parameter per tracer / z-bin)
for i in range(len(theories)):    
    for param in theories[i].init.params:
        # Update latex just to have better labels
        param.update(namespace='{}'.format(tracers[i]))
                     #latex=param.latex(namespace=#'\mathrm{{pre}},
                      #                 '\mathrm{{{}}}, {:d}'.format('LRG', 0), inline=False))    


# In[19]:


setup_logging()
Likelihoods = []
for i in range(len(theories)):
    Likelihoods.append(ObservablesGaussianLikelihood(observables = [observables[i]]))


# In[20]:


likelihood = SumLikelihood(likelihoods = (Likelihoods))

likelihood()



from desilike.samplers import MCMCSampler

sampler = MCMCSampler(likelihood ,save_fn = chain_name)
sampler.run(check={'max_eigen_gr': 0.03})

