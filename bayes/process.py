#data
import numpy as np
import pandas as pd
import pdfplumber
import xarray as xr
from pathlib import Path
from typing import Dict, Any


# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
sns.set_style("whitegrid")

#Bayes
import pymc as pm
import arviz as az
import pytensor.tensor as pt

#Make sure it can see the system path
from pathlib import Path
import sys

# Add parent directory of the notebook (the project root) to sys.path
ROOT = Path().resolve().parent   # X/
sys.path.insert(0, str(ROOT))

#Other modules
from utils import data_utils
from bayes import DATA_DIR


coords=data_utils.get_coords()

def make_process_model(model,evidence,priors,hyperpriors):


    with model:
        for k,values in coords.items():
            if k not in model.coords:
                model.add_coord(k,values)
        etalist=[]
        for process in coords["process"]:
            etalist+=[priors[f"η_{process}"](name=f"η_{process}")]
        eta = pm.Deterministic("η",pm.math.stack(etalist),dims=("process",))


        #Assume model spread is the same for every process
        #This is the PRECISION: 1/sigma
        #The bigger it is the more representative a single process-based run will be
       
        tau_eta=hyperpriors["τ_eta"](name="τ_eta")
        tau_nu=hyperpriors["τ_nu"](name="τ_nu")
        #Now update the posteriors for eta_process with the process ensembles
        for k,process in enumerate(coords["process"]):
            pm.Normal("lik_eta_"+process,eta[k],tau=tau_eta,observed=evidence[f"η_{process}"])


        nulist=[]
        for process in coords["process"]:
            nulist+=[priors[f"ν_{process}"](name=f"ν_{process}")]
        nu = pm.Deterministic("ν",pm.math.stack(nulist),dims=("process",))

    # Assume common spread tau_process
    #Now update the posteriors for eta_process with the process ensembles
        for k,process in enumerate(coords["process"]):
            pm.Normal("lik_nu_"+process,nu[k],tau=tau_nu,observed=evidence[f"ν_{process}"])    
    return {"ν":nu,"η":eta, "τ_eta":tau_eta,"τ_nu":tau_nu}



def make_emergent_constraint_model(model,evidence,priors,hyperpriors):
    with model:
        ### EMERGENT CONSTRAINT ###
        # EMERGENT CONSTRAINT ON TROPICAL γLT
        n_models_EC=len(evidence['γ_LT'])
        γLT=priors["γLT"](name="γLT")
            #Assume linear relationships
        # Hyperpriors on slope and intercept
        m = hyperpriors["m"](name="m")
        b = hyperpriors["b"](name="b")

        # Errors-in-variables model for model gammaLT- use standard deviations reported by Zechlau
        x_true = pm.Normal("γLT_CMIP", mu=evidence['γ_LT'], \
                           sigma=evidence['σ_LT'], \
                           shape=n_models_EC)

        # Linear model for "true IAV" values
        y_true = m * x_true + b

        # Likelihood of observed IAV values
        y_likelihood = pm.Normal("IAV_CMIP", mu=y_true, \
                                 sigma=evidence['σ_IAV'],\
                                 shape=n_models_EC,\
                                 observed=evidence['γ_IAV'])
        #Emergent constraint relationship

        mu_obs = m * γLT + b
        # IAV observed ~ N(-4.3,0.67) #CHECK IF THIS IS ONE OR TWO SIGMA
        IAV_true=pm.Normal("IAV_true",mu=mu_obs,\
                           sigma=evidence["IAV_observed_std"],\
                           observed = [evidence["IAV_observed_mean"]])
        return {"γLT":γLT}



#D must have dimensions (Nmodels,2)


def make_covariance_model(model,evidence,priors,hyperpriors):
    
    D=np.stack([evidence["βL_cmip"],evidence["γL_cmip"]]).T


    #def covariance_model(model):
    with model:

        ########
        #get eta, nu, tau_eta, tau_nu from the process model
        #######
        process_model=make_process_model(model,
                                         evidence,
                                         priors,
                                         hyperpriors)
        eta=process_model["η"]
        nu=process_model["ν"]
        tau_eta=process_model["τ_eta"]
        tau_nu=process_model["τ_nu"]

        ## Add the CMIP data in the form of mutable data
        CMIP = pm.MutableData("CMIP",D)



        for k,values in coords.items():
                if k not in model.coords:
                    model.add_coord(k,values)

        # Prior on gamma_extratropics
        gamma_extratropics=priors["γLX"](name="γLX")
        # Get gamma tropics from the emergent constraint model
        emergent_constraint_model=make_emergent_constraint_model(model,\
                                                                 evidence,\
                                                                 priors,\
                                                                 hyperpriors)
        gamma_tropics=emergent_constraint_model["γLT"]

        # The total gamma is the sum of extratropics and tropics
        gamma = pm.Deterministic("γL",gamma_extratropics+gamma_tropics)
        # and the prior on beta
        beta=priors["βL"](name="βL")
        mu=pm.Deterministic("μ",pm.math.stack([beta,gamma]),dims=("parameter",))

        # Prior on Cholesky decomposition of covariance matrix (has to be LKJ)
        # sd_dist gives priors for standard deviations
        #Make these really wide if you like- reflects model structural spread
        chol, corr, sigma = hyperpriors["chol"](name="chol")


        # Construct covariance matrix from Cholesky
        Sigma = pm.Deterministic("Sigma", chol.dot(chol.T),dims=("parameter","cross_parameter"))

        #get eta for each model


        Eta_arr = pt.ones((len(coords["process"]), len(coords["model"])))
        Nu_arr = pt.ones((len(coords["process"]), len(coords["model"])))

        for j in range(len(coords["process"])):
            # Use pm.math.switch to multiply by eta[j] only where lookup_table[j, :] == 1
            Eta_arr = pt.set_subtensor(Eta_arr[j, :], 
                                              pm.math.switch(pt.eq(evidence["lookup table"][j, :], 1), eta[j], 1.0))
            Nu_arr = pt.set_subtensor(Nu_arr[j, :], 
                                             pm.math.switch(pt.eq(evidence["lookup table"][j, :], 1), nu[j], 1.0))

        # Unscaled B

        B = beta / pm.math.prod(eta)
        G = gamma / pm.math.prod(nu)


        # Do mu
        mu_model=[]
        for i,model in enumerate(coords["model"]):

            eta_model=pm.math.prod(Eta_arr[:,i])
            nu_model=pm.math.prod(Nu_arr[:,i])
            mu_model+=[pm.math.stack([eta_model*B,nu_model*G])]
        mu_totals = pm.Deterministic("all_mus",
                                     pm.math.stack([mu_model[i] for i,model in enumerate(coords["model"])]),
                                                  dims=("model",))

        pm.MvNormal(f'D', mu=mu_totals, chol=chol, observed=CMIP)
        return {"η":eta,\
                "ν":nu,\
                "βL":beta,\
                "γL": gamma,\
                "γLT":gamma_tropics,\
                "γLX": gamma_extratropics\
               }
