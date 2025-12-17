import numpy as np
import pymc as pm

import pandas as pd



import pytensor.tensor as pt

import xarray as xr

import os,glob,sys
from pytensor.scan import scan
from utils import damip_utils

#Helper function to fit AR(p) to residuals (ie non-forced response)
def logp_arp(residual, rho, sigma):
    """
    Custom log-likelihood for AR(p) process with known rho and sigma.
    residual: symbolic tensor (T,)
    rho: vector of shape (p,)
    sigma: scalar
    """
    if isinstance(rho, pt.TensorVariable):
        p=rho.shape.eval()[0]
    else:
        p=len(rho)

    # First p values ~ stationary distribution approximation
    logp0 = pm.logp(
        pm.Normal.dist(mu=0, sigma=sigma / pt.sqrt(1 - pt.sum(rho**2))),
        residual[:p]
    )

    # Create matrix of lagged residuals: shape (T - p, p)
    # Instead of using Python range(), we use PyTensor indexing
    def lag_matrix(residual, p):
        """
        Construct a lag matrix of shape (T - p, p)
        Each row t contains: [residual[t-1], ..., residual[t-p]]
        """
        return pt.stack(
            [residual[i : -(p - i)] for i in range(p)],
            axis=1
        )


    eps_mat = lag_matrix(residual, p)     # shape (T - p, p)
    eps_pred = pt.sum(eps_mat * rho[::-1], axis=1)  # predicted epsilon_t
    eps_actual = residual[p:]             # actual epsilon_t

    logp_rest = pm.logp(pm.Normal.dist(mu=eps_pred, sigma=sigma), eps_actual)

    return logp0 + pt.sum(logp_rest)

def create_noise_submodel(C_conc,rho_prior,sigma_innov_prior):
    """
    Create sub-model for concatenated preindustrial noise
    
    Parameters:
    -----------
    C_conc : array-like, shape (n_years,)
        Concatenated piControl forcing
    rho_prior: TensorVariable, shape (p,)
        The prior for the AR(p) coefficients ρ_i. Number of lags will be inferred from rho shape
    sigma_innov_prior: TensorVariable, shape (1,)
        The prior for the innovation term σ
    Returns:
    --------
    dict with model parameters
    """
    
    likelihood = pm.AR("likelihood_piC",
                       rho=rho_prior,
                       sigma=sigma_innov_prior,
                       observed=C_conc)
    return {"likelihood_piC": likelihood}

def create_forcing_submodel(ensemble_data, model_name, forcing_name, F_true, sigma_shared):
    """
    Create sub-model for one GCM's ensemble members for a specific forcing.
    
    Parameters:
    -----------
    ensemble_data : array-like, shape (n_ensemble, n_years)
        Ensemble members for this model and forcing
    model_name : str
        Name of the GCM model
    forcing_name : str
        Name of the forcing (e.g., 'aer', 'CO2', 'nat', 'GHG')
    F_true : TensorVariable, shape (n_years,)
        The model's true forced response (shared parameter)
    sigma_shared : TensorVariable
        Shared within-model variability parameter
    
    Returns:
    --------
    dict with model parameters
    """
    # Likelihood: each ensemble member ~ N(F_model^true, sigma_model)
    # Use model-specific and forcing-specific ensemble dimension name
    likelihood = pm.Normal(
        f"obs_{forcing_name}_{model_name}",
        mu=F_true,
        sigma=sigma_shared,
        observed=ensemble_data,
        dims=(f"ensemble_{forcing_name}_{model_name}", "time")
    )
    
    return {"likelihood": likelihood}

def create_multi_forcing_model(GCM_F, 
                               priors,
                               forcings=['hist-aer', 'hist-CO2', 'hist-nat', 'hist-GHG'],
                               n_years=165, 
                               H_obs=None,
                               estimate_betas=True,
                               use_historical=False,
                               n_ar=3,
                               mu_E=None,
                               sigma_E=None):
    """
    Create joint hierarchical model for multiple DAMIP forcings with detection & attribution.
    
    Parameters:
    -----------
    GCM_F : dict
        GCM_F[forcing][model] = array of shape (n_ensemble, n_years)
        GCM_F["piControl"][model] = array of shape (n_control, n_years) for internal variability
    forcings : list of str
        List of forcing types to model jointly
    n_years : int
        Number of years (default 165)
    H_obs : array-like, shape (n_years,), optional
        Observed time series to attribute
    estimate_betas : bool
        Whether to estimate scaling factors for detection & attribution
    
    Returns:
    --------
    pm.Model object, dict of submodels
    """
    # Get all unique model names across forcings
    all_models = set()
    for forcing in forcings:
        if forcing in GCM_F:
            all_models.update(GCM_F[forcing].keys())
    model_names = sorted(list(all_models))
    n_models = len(model_names)
    
    # Map forcing names to clean variable names
    forcing_map = {
        'hist-aer': 'aer',
        'hist-CO2': 'CO2',
        'hist-nat': 'nat',
        'hist-GHG': 'GHG',
        'historical': 'all'
    }

    # Concatenate piControl
    C_conc=damip_utils.concatenate_piControl(GCM_F)
    
    # Set up coordinates
    coords = {
        "time": np.arange(n_years)+1850,
        "model": model_names,
    }
    
    # Add ensemble coordinates AND model-specific coordinates for each forcing
    for forcing in forcings + ["historical"]:
        if forcing not in GCM_F:
            continue
        forcing_clean = forcing_map[forcing]
        
        # Add model coordinate for this forcing (list of models with this forcing)
        forcing_models = list(GCM_F[forcing].keys())
        coords[f"model_{forcing_clean}"] = forcing_models
        
        # Add ensemble coordinates for each model
        for model_name in forcing_models:
            n_ensemble = GCM_F[forcing][model_name].shape[0]
            coords[f"ensemble_{forcing_clean}_{model_name}"] = np.arange(n_ensemble)
    
#     # Add piControl coordinates if available
#     if "piControl" in GCM_F:
#         for model_name in GCM_F["piControl"].keys():
#             n_control = GCM_F["piControl"][model_name].shape[0]
#             coords[f"ensemble_piControl_{model_name}"] = np.arange(n_control)
    
    with pm.Model(coords=coords) as model:
        # ===== Cumulative CO2 emissions (from OWID) =====
        if mu_E is not None and sigma_E is not None:
            E = pm.Normal(
                "E_cum",
                mu=mu_E,
                sigma=sigma_E,
                dims="time"
            )

        # ===== Shared within-model variability =====
        # Same sigma for all forcings (internal variability)
        sigma_internal={}
        for model_name in model_names:
            sigma_internal[model_name] = priors["sigma_internal"](f"sigma_internal_{model_name}")
            #sigma_internal[model_name] = pm.HalfNormal(f"sigma_internal_{model_name}", sigma=1.0)
        
        # ===== Store submodels and forcing-specific parameters =====
        submodels = {}
        F_dict = {}  # Store the latent true forced responses
        sigma_structural = {}
        # ===== Process each forcing =====
        for forcing in forcings:
            if forcing not in GCM_F:
                print(f"Warning: {forcing} not found in GCM_F, skipping...")
                continue
                
            forcing_clean = forcing_map[forcing]
            forcing_data = GCM_F[forcing]
            forcing_models = list(forcing_data.keys())
            n_forcing_models = len(forcing_models)
            
            # Between-model variability (structural uncertainty) - specific to each forcing
            sigma_structural[forcing_clean] = priors["sigma_structural"](f"sigma_structural_{forcing_clean}")
            
    
            
            # Latent "true" forced response for this forcing
            #Put a really broad prior on it
            F_mean = priors["F_mean"](f"F_{forcing_clean}")
           
            
            # Model-specific latent forced responses
            # F_model^true ~ N(F, tau) for each model
            # Only create for models that have data for this forcing
            F_models = pm.Normal(
                f"F_{forcing_clean}_model_true",
                mu=F_mean,
                sigma=sigma_structural[forcing_clean],
                shape=(n_forcing_models, n_years),
                dims=(f"model_{forcing_clean}", "time")
            )
            
            # Store the latent forcing for derived quantities
            F_dict[forcing_clean] = F_mean
            
            # Create sub-models for each GCM
            submodels[forcing_clean] = {}
            for i, model_name in enumerate(forcing_models):
                ensemble_data = forcing_data[model_name]
                
                submodels[forcing_clean][model_name] = create_forcing_submodel(
                    ensemble_data=ensemble_data,
                    model_name=model_name,
                    forcing_name=forcing_clean,
                    F_true=F_models[i, :],
                    sigma_shared=sigma_internal[model_name]
                )
            
            # Multi-model mean and spread for this forcing
            mm_mean = pm.Deterministic(
                f"mm_mean_{forcing_clean}", 
                F_models.mean(axis=0), 
                dims="time"
            )
            mm_std = pm.Deterministic(
                f"mm_std_{forcing_clean}", 
                F_models.std(axis=0), 
                dims="time"
            )
        
        # ===== Derived forcing: F_nonCO2 = F_GHG - F_CO2 =====
        if 'GHG' in F_dict and 'CO2' in F_dict:
            F_nonCO2 = pm.Deterministic(
                "F_nonCO2",
                F_dict['GHG'] - F_dict['CO2'],
                dims="time"
            )
            F_dict['nonCO2'] = F_nonCO2
        
        # ===== Optional: Total anthropogenic forcing =====
        if 'GHG' in F_dict and 'aer' in F_dict:
            F_anthro = pm.Deterministic(
                "F_anthro",
                F_dict['GHG'] + F_dict['aer'],  # GHG + aerosols
                dims="time"
            )
        
        # ===== Detection & Attribution Component =====
        if H_obs is not None and estimate_betas:
            
            # ===== Scaling factors (betas) =====
            # beta_GHG applies to both CO2 and nonCO2
#             beta_GHG = pm.Normal("beta_GHG", mu=1, sigma=0.5)
#             beta_aer = pm.Normal("beta_aer", mu=1, sigma=0.5)
#             beta_nat = pm.Normal("beta_nat", mu=1, sigma=0.5)
            
            beta_GHG = priors["beta"]("beta_GHG")
            beta_aer = priors["beta"]("beta_aer")
            beta_nat = priors["beta"]("beta_nat")
            
#             beta_GHG = pm.HalfNormal("beta_GHG",sigma=0.5)
#             beta_aer = pm.HalfNormal("beta_aer",  sigma=0.5)
#             beta_nat = pm.HalfNormal("beta_nat", sigma=0.5)
            
     
            # ===== Construct expected observations =====
            # H = beta_aer*F_aer + beta_GHG*F_CO2 + beta_GHG*F_nonCO2 + beta_nat*F_nat
           
            H_forced = pm.math.zeros_like(np.zeros(n_years))
            
            
            if 'aer' in F_dict:
                H_forced = H_forced + beta_aer * F_dict['aer']
            if 'CO2' in F_dict:
                H_forced = H_forced + beta_GHG * F_dict['CO2']
            if 'nonCO2' in F_dict:
                H_forced = H_forced + beta_GHG * F_dict['nonCO2']
            if 'nat' in F_dict:
                H_forced = H_forced + beta_nat * F_dict['nat']
            
            H_forced = pm.Deterministic("H_forced", H_forced, dims="time")
            
            
            
            internal_variability=H_obs.values-H_forced
            
            
            # ===== Sample the AR(p) parameters learned from piControl =====
            
#             rho=pm.Normal("rho",mu=0,sigma=1,shape=(n_ar,))

#             sigma_innovation=pm.HalfNormal("sigma_innov",1)
                
   

            rho = priors["rho"]("rho")
            #rho=pm.Normal("rho",mu=rho_mean,sigma=rho_std)
            sigma_innovation = priors["sigma_innovation"]("sigma_innovation")
#             sigma_innovation=pm.TruncatedNormal("sigma_innovation",
#                                        mu=sigma_innov_mean,
#                                                 sigma=sigma_innov_std,
#                                                lower=0)
            submodels["piControl"]=create_noise_submodel(C_conc,rho,sigma_innovation)
            pm.Potential("arp_likelihood", logp_arp(internal_variability, rho=rho, sigma=sigma_innovation))
            
            # ===== Process each forcing =====
            if use_historical:
                forcing = "historical"  
                forcing_clean = "all"
                forcing_data = GCM_F["historical"]
                forcing_models = list(forcing_data.keys())

                n_forcing_models = len(forcing_models)

                forcing_models = list(GCM_F[forcing].keys())


                # Between-model variability (structural uncertainty) - specific to each forcing
                #tau = pm.HalfNormal(f"tau_{forcing_clean}", sigma=0.5)
                quad_sum=pm.math.sum([sigma_structural[forcing_map[k]]**2 for k in forcings])
                sigma_structural["all"] = pm.Deterministic("sigma_structural_all",pm.math.sqrt(quad_sum))
                

                # Latent "true" forced response for this forcing
                F_mean = pm.Deterministic("F_historical",\
                                          F_dict["aer"]+F_dict["CO2"]+F_dict["nonCO2"]+F_dict["nat"])

                # Model-specific latent forced responses
                # F_model^true ~ N(F, tau) for each model
                # Only create for models that have data for this forcing
                F_models = pm.Normal(
                    f"F_{forcing_clean}_model_true",
                    mu=F_mean,
                    sigma=sigma_structural["all"],
                    shape=(n_forcing_models, n_years),
                    dims=(f"model_{forcing_clean}", "time")
                )

                # Store the latent forcing for derived quantities
                F_dict[forcing_clean] = F_mean

                # Create sub-models for each GCM
                submodels[forcing_clean] = {}
                for i, model_name in enumerate(forcing_models):
                    ensemble_data = forcing_data[model_name]

                    submodels[forcing_clean][model_name] = create_forcing_submodel(
                        ensemble_data=ensemble_data,
                        model_name=model_name,
                        forcing_name=forcing_clean,
                        F_true=F_models[i, :],
                        sigma_shared=sigma_internal[model_name]
                    )




        
            # ===== Attributable responses =====
            # attributable_forcing = beta_forcing * F_forcing
            if 'aer' in F_dict:
                pm.Deterministic("attributable_aer", beta_aer * F_dict['aer'], dims="time")
            if 'CO2' in F_dict:
          
            
                T_CO2 = pm.Deterministic("attributable_CO2", beta_GHG * F_dict['CO2'], dims="time")
                if mu_E is not None:
                    # ===== AR6-consistent TCRE (slope through origin) =====
                    TCRE = pm.Deterministic(
                        "TCRE",
                        pm.math.sum(E * T_CO2) /
                        pm.math.sum(E**2)
                        )
                    
            if 'nonCO2' in F_dict:
                pm.Deterministic("attributable_nonCO2", beta_GHG * F_dict['nonCO2'], dims="time")
            if 'nat' in F_dict:
                pm.Deterministic("attributable_nat", beta_nat * F_dict['nat'], dims="time")
           
            
            # Total attributable anthropogenic
            if 'GHG' in F_dict and 'aer' in F_dict:
                pm.Deterministic(
                    "attributable_anthro",
                    beta_GHG * F_dict['GHG'] + beta_aer * F_dict['aer'],
                    dims="time"
                )
    
    return model, submodels

def fit_multi_forcing_model(GCM_F, 
                            priors,
                            forcings=['hist-aer', 'hist-CO2', 'hist-nat', 'hist-GHG'],
                             n_years=165, H_obs=None, estimate_betas=True,
                             n_samples=2000, n_tune=1000, n_chains=4,\
                             use_historical=False,n_ar=3,\
                             mu_E=None,sigma_E=None):
    """
    Fit the multi-forcing hierarchical model.
    
    Parameters:
    -----------
    GCM_F : dict
        GCM_F[forcing][model] = array of shape (n_ensemble, n_years)
    forcings : list of str
        List of forcing types to model jointly
    H_obs : array-like, shape (n_years,), optional
        Observed time series for detection & attribution
    estimate_betas : bool
        Whether to estimate scaling factors
    n_samples, n_tune, n_chains : int
        MCMC sampling parameters
        
    Returns:
    --------
    model, trace, submodels
    """
    # Create model
    model, submodels = create_multi_forcing_model(GCM_F, priors,forcings, n_years, H_obs, estimate_betas,use_historical,mu_E=mu_E,sigma_E=sigma_E)
    
    # Sample
    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=0.95,
            return_inferencedata=True
        )
    
    return model, trace, submodels
