import numpy as np
import pandas as pd
import xarray as xr
import glob
from pathlib import Path
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Base paths (relative to repository)
# -----------------------------------------------------------------------------

# bayes/damip_utils.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "DATA"
DAMIP_DIR = DATA_DIR / "DAMIP"
OBS_DIR = DATA_DIR / "Observations"
EMISSIONS_DIR = DATA_DIR / "Emissions"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def fopen(direc, fname):
    picfile = xr.open_dataset(Path(direc) / fname)
    annualmean = picfile.tas.groupby("time.year").mean(dim="time")
    return annualmean


# -----------------------------------------------------------------------------
# piControl
# -----------------------------------------------------------------------------

def get_piControl(model):
    piC_dir = DAMIP_DIR / model / "piControl"
    picfiles = sorted(piC_dir.glob("*.nc"))

    data_list = []
    for fname in picfiles:
        with xr.open_dataset(fname,use_cftime=True) as ds:
            vec = ds.tas.groupby("time.year").mean(dim="time")
            data_list.append(vec - vec.mean())

    return np.concatenate(data_list)



# -----------------------------------------------------------------------------
# DAMIP experiments
# -----------------------------------------------------------------------------

def get_experiment(model, experiment):
   
    ex_dir = DAMIP_DIR / model / experiment
    efiles = sorted(ex_dir.glob("*.nc"))

    edata_list = []
    for fname in efiles:
        with xr.open_dataset(fname) as ds:
            annualmean = ds.tas.groupby("time.year").mean(dim="time")
            if experiment != "piControl":
                vec = annualmean.sel(year=slice(1850, 2014))
            else:
                vec = annualmean
            edata_list.append(vec - vec[:50].mean())

    return np.vstack(edata_list)
def read_DAMIP_simulations():
    GCM_F = {}

    models = sorted(
        p.name for p in DAMIP_DIR.iterdir() if p.is_dir()
    )

    forcings = ["hist-GHG", "hist-CO2", "hist-aer", "hist-nat"]
    for forcing in forcings:
        GCM_F[forcing] = {}
        valid_models = [
            m for m in models if (DAMIP_DIR / m / forcing).is_dir()
        ]

        for model in tqdm(valid_models, desc=forcing):
            GCM_F[forcing][model] = get_experiment(model, forcing)

    GCM_F["historical"] = {}
    valid_models = [
        m for m in models if (DAMIP_DIR / m / "historical").is_dir()
    ]
    for model in tqdm(valid_models, desc="historical"):
        GCM_F["historical"][model] = get_experiment(model, "historical")

    GCM_F["piControl"] = {}
    valid_models = [
        m for m in models if (DAMIP_DIR / m / "piControl").is_dir()
    ]
    for model in tqdm(valid_models, desc="piControl"):
        GCM_F["piControl"][model] = get_piControl(model)

    return GCM_F




# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

def read_HadCRUT():
    hadcrut_name = "HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.nc"
    f_obs = xr.open_dataset(OBS_DIR / hadcrut_name)

    H_obs = (
        f_obs.tas.mean(dim="realization")
        .sel(time=slice("1850-01-01", "2014-12-31"))
    )

    H_obs = H_obs - H_obs.sel(
        time=slice("1850-01-01", "1899-12-31")
    ).mean().values

    return H_obs


# -----------------------------------------------------------------------------
# Emissions
# -----------------------------------------------------------------------------

def read_emissions():
    ## Emissions are given in units of MtCO2
    ## Convert to exagrams c
    ##  million tonnes CO2=12/44 x 10^{-6}
    emissions_df = pd.read_csv(EMISSIONS_DIR / "owid-co2-data.csv")

    owid_years = emissions_df[emissions_df.country == "World"].year.values

    cumu = xr.DataArray(
        emissions_df[emissions_df.country == "World"]
        .cumulative_co2_including_luc.values,
        coords={"time": owid_years},
    ).dropna(dim="time").sel(time=slice(1850, 2025))

    ann = xr.DataArray(
        emissions_df[emissions_df.country == "World"]
        .co2_including_luc.values,
        coords={"time": owid_years},
    ).dropna(dim="time").sel(time=slice(1850, 2025))

    ann_fossil = xr.DataArray(
        emissions_df[emissions_df.country == "World"].co2.values,
        coords={"time": owid_years},
    ).dropna(dim="time").sel(time=slice(1850, 2025))

    ann_lu = ann - ann_fossil
    years = ann_lu.time.values

    ann_lu_uncertainty = xr.DataArray(
        np.array([np.random.normal(ann_lu, 2570.0) for _ in range(1000)]),
        coords={"sample": np.arange(1000), "time": years},
    )

    cum_lu_uncertainty = ann_lu_uncertainty.cumsum(dim="time")

    ann_fossil_uncertainty = xr.DataArray(
        np.array([np.random.normal(ann_fossil, 0.05 * ann_fossil) for _ in range(1000)]),
        coords={"sample": np.arange(1000), "time": years},
    )

    cum_fossil_uncertainty = ann_fossil_uncertainty.cumsum(dim="time")

    cumulative_emissions = (
        0.272727e-6 * cum_fossil_uncertainty
        + 0.272727e-6 * cum_lu_uncertainty
    )

    return cumulative_emissions


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def concatenate_piControl(GCM_F, drift_L=500):
    return np.concatenate(
        [GCM_F["piControl"][model][-drift_L:] for model in GCM_F["piControl"]]
    )
