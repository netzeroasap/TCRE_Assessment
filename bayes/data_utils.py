# data_utils.py

import numpy as np
import pandas as pd
import pdfplumber
import xarray as xr
from pathlib import Path
from typing import Dict, Any
from . import DATA_DIR 


def load_CMIP_land_data(kind="2xCO2"):
    """
    Load and preprocess CMIP6 data from Norman.
    
    Parameters
    ----------
    kind : str
        One of 2xCO2 or 4xCO2: specifies when to calculate beta and gamma
        
    Returns
    -------
    dict
        Evidence dictionary with processed data
    """
    TCREsource_betagamma = pd.read_csv('DATA/TCREsource_betagamma.csv')
    beta = TCREsource_betagamma[f'beta_L_{kind}'].values
    gamma = TCREsource_betagamma[f'gamma_L_{kind}'].values

    beta_cmip6 = beta[TCREsource_betagamma['source'].isin(['CMIP6', 'CMIP6+'])]
    gamma_cmip6 = gamma[TCREsource_betagamma['source'].isin(['CMIP6', 'CMIP6+'])]

    return {"βL_cmip":beta_cmip6,
                "γL_cmip":gamma_cmip6}


def load_emergent_constraint_evidence():
    with pdfplumber.open("DATA/Zechlau2022.pdf") as pdf:
        page = pdf.pages[6]
        zechtable = page.extract_table()

    title=" ".join([x.split("\n") for x in zechtable[0]][0])
    rawcolumns=zechtable[1][0].split(" ")
    columns=["Model", "γ_LT", "σ_LT", 'γ_IAV', "σ_IAV"]
    zdata=zechtable[2:][0][0].split()
    ncol=len(columns)

    cmip5_list = [item for item in zdata[1:zdata.index("CMIP6")] if item != '±']
    L=len(cmip5_list)

    nrows=int(L/ncol)
    cmip5_gammas=pd.DataFrame(np.array(cmip5_list).reshape((nrows,ncol)),columns=columns)

    cmip6_list = [item for item in zdata[zdata.index("CMIP6")+1:zdata.index("OBS")] if item != '±']
    cmip6_list
    L6=len(cmip6_list)
    rows6=int(L6/ncol)
    cmip6_strings=np.array(cmip6_list).reshape((rows6,ncol))
    cmip6_gammas=pd.DataFrame(cmip6_strings,columns=columns)


    # now need to do some more data cleaning :(
    mystr=cmip6_gammas["γ_LT"].values[0]
    # For some reason it can't read the minus sign from the pdf- replace by hand
    badval=mystr[0]


    for c in columns[1:]:
        try:
            cmip6_gammas[c]=pd.to_numeric(cmip6_gammas[c])
        except:

            cmip6_gammas[c]=[float(x.replace(badval,"-")) for x in cmip6_gammas[c]]

    evidence_EC = {"γ_LT":cmip6_gammas.γ_LT.values,\
                "γ_IAV":cmip6_gammas.γ_IAV.values,\
                "σ_LT":cmip6_gammas.σ_LT.values,\
                "σ_IAV":cmip6_gammas.σ_IAV.values,\
                "IAV_observed_mean":-4.3,\
                "IAV_observed_std": 0.67 }
    evidence=evidence_EC
    return evidence

def load_process_evidence(kind="2xCO2"):
    # process grouping
   
    TCREsource_betagamma = pd.read_csv('TCREsource_betagamma.csv')
    beta = TCREsource_betagamma[f'beta_L_{kind}'].values
    gamma = TCREsource_betagamma[f'gamma_L_{kind}'].values

    beta_cmip = beta[TCREsource_betagamma['source'].isin(['CMIP6', 'CMIP6+'])]
    gamma_cmip = gamma[TCREsource_betagamma['source'].isin(['CMIP6', 'CMIP6+'])]

    cmip6_models=TCREsource_betagamma.model[TCREsource_betagamma['source'].isin(['CMIP6', 'CMIP6+'])].values

    cmip6_hasNitro = np.array([True, False, False, True, False, False, False, True, True, True, True, True, True])
    cmip6_hasPF = np.array([False, False, False, True, False, False, False, False, False, True, False, False, True])
    cmip6_hasFire = np.array([False, False, False, True, True, True, False, False, True, True, False, True, True])
    cmip6_hasDynveg = np.array([False, False, False, False, False, True, False, False, True, False, True, True, False])

    lookup_table=np.stack([cmip6_hasNitro,cmip6_hasPF,cmip6_hasFire,cmip6_hasDynveg]).astype(np.float32)

    eta_proc={}
    eta_nlim_access= beta[TCREsource_betagamma['model'].isin(['ACCESS-ESM_CN'])]/\
    beta[TCREsource_betagamma['model'].isin(['ACCESS-ESM_C'])] #0.69 

    eta_nlim_ukesm= beta[TCREsource_betagamma['model'].isin(['UKESM1-1_ctrl'])]/\
    beta[TCREsource_betagamma['model'].isin(['UKESM1-1_nonlim'])] #0.77

    eta_proc["η_nitrogen"] = [float(eta_nlim_access),float(eta_nlim_ukesm)]



    #FIRE
    eta_fire_ukesm= beta[TCREsource_betagamma['model'].isin(['UKESM1-1_fire'])]/\
    beta[TCREsource_betagamma['model'].isin(['UKESM1-1_ctrl'])]

    eta_proc["η_fire"] = [float(x) for x in eta_fire_ukesm]

    #Dynamic vegetation
    # from 4xCO2
    eta_dynveg_ukesm= beta[TCREsource_betagamma['model'].isin(['UKESM1-1_ctrl'])]/\
    beta[TCREsource_betagamma['model'].isin(['UKESM1-1_nodgvm'])]

    eta_proc["η_vegetation"] = [float(x) for x in eta_dynveg_ukesm]

    ###### nu #####
   
    nu_proc={}
    nu_nlim_access= gamma[TCREsource_betagamma['model'].isin(['ACCESS-ESM_CN'])]/\
    gamma[TCREsource_betagamma['model'].isin(['ACCESS-ESM_C'])] #0.69 

    nu_nlim_ukesm= gamma[TCREsource_betagamma['model'].isin(['UKESM1-1_ctrl'])]/\
    gamma[TCREsource_betagamma['model'].isin(['UKESM1-1_nonlim'])] #0.77

    nu_proc["ν_nitrogen"] = [float(nu_nlim_access),float(nu_nlim_ukesm)]



    #FIRE
    nu_fire_ukesm= gamma[TCREsource_betagamma['model'].isin(['UKESM1-1_fire'])]/\
    gamma[TCREsource_betagamma['model'].isin(['UKESM1-1_ctrl'])]

    nu_proc["ν_fire"] = [float(x) for x in nu_fire_ukesm]

    #Dynamic vegettion
   
    nu_dynveg_ukesm= gamma[TCREsource_betagamma['model'].isin(['UKESM1-1_ctrl'])]/\
    gamma[TCREsource_betagamma['model'].isin(['UKESM1-1_nodgvm'])]

    nu_proc["ν_vegetation"] = [float(x) for x in nu_dynveg_ukesm]
    
    return eta_proc | nu_proc | {"lookup table":lookup_table}
    




def validate_priors(priors: Dict,hyperpriors:Dict) -> None:
    """Validate prior specifications."""
    required_priors = ["βL",
                            "γLT",
                            "γLX",
                            'η_nitrogen',
                            'η_fire',
                            'δβ_permafrost',
                            'η_vegetation',
                            'ν_nitrogen',
                            'ν_fire',
                            'δγ_permafrost',
                            'ν_vegetation']
    required_hyperpriors = ["m",
                                "b",
                                "τ_eta",
                                "τ_nu",
                                "chol"]
    missing_priors=[]
    for key in required_priors:
        if key not in priors:
            missing_priors+=[key]
    missing_hyperpriors=[]
    for key in required_hyperpriors:
        if key not in hyperpriors:
            missing_hyperpriors+=[key]
    if (missing_priors!=[] or missing_hyperpriors !=[]):
        #return missing_priors
        raise ValueError(f"Missing required priors: {str(missing_priors)} \
         Missing required hyperpriors: {str(missing_hyperpriors)}")
                             
