"""
Factory functions for the four TCRE component models.

Each function takes:
  - the relevant CMIP6 data as an xr.DataArray
  - a prior dict from config.priors 

and returns a PyMC model ready for pm.sample().

Additive/multiplicative process structure and 
partial pooling is done by CMIP_process.build_vectorized_process_models.
"""
from copy import deepcopy

import numpy as np
import pymc as pm
import xarray as xr

from bayes import CMIP_process


# ── Prior translators ────────────────────────────────────────────────────

def _noprocess_prior(spec, var_name):
    """Convert a baseline spec dict to the lambda form expected by CMIP_process."""
    dist, mu, sigma = spec["dist"], spec["mu"], spec["sigma"]
    if dist == "LogNormal":
        return {var_name: lambda name, _mu=mu, _s=sigma: pm.LogNormal(name, _mu, _s)}
    if dist == "Normal":
        return {var_name: lambda name, _mu=mu, _s=sigma: pm.Normal(name, _mu, _s)}
    raise ValueError(f"Unsupported distribution: {dist}")


def _delta_priors(delta_spec):
    """Convert a delta spec dict to the lambda form expected by CMIP_process."""
    priors = {}
    for p, spec in delta_spec.items():
        dist, mu, sigma = spec["dist"], spec["mu"], spec["sigma"]
        if dist == "Normal":
            priors[p] = lambda name, _mu=mu, _s=sigma: pm.Normal(name, _mu, _s)
        elif dist == "LogNormal":
            priors[p] = lambda name, _mu=mu, _s=sigma: pm.LogNormal(name, _mu, _s)
        elif dist == "NegLogNormal":
            # Permafrost: always negative, so we negate a LogNormal
            priors[p] = lambda name, _mu=mu, _s=sigma: pm.Deterministic(
                name, -1 * pm.LogNormal(f"negative_{name}", _mu, _s)
            )
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
    return priors


# ── Model builders ───────────────────────────────────────────────────────

def build_beta_land_model(cmip_beta_land, lookup_table, priors):
    """
    Multiplicative process model for land carbon-concentration feedback (beta_L).

    beta_L = baseline * prod_p(eta_p ^ L_pm)

    where L_pm flags whether CMIP6 model m includes process p, and
    eta_p = exp(log_eta_p) is the multiplicative factor for process p.

    An observed eta_nitrogen from paired model experiments is added as a
    pseudo-observation to further constrain the nitrogen effect.

    Parameters
    ----------
    cmip_beta_land : xr.DataArray  shape (model,)     CMIP6 beta_L values
    lookup_table   : xr.DataArray  shape (process, model)  process presence flags
    priors         : dict          config.priors.beta_land
    """
    mult_processes = list(priors["eta"].keys())  # ["nitrogen", "fire", "veg"]

    model = CMIP_process.build_vectorized_process_model(
        cmip_beta_land,
        lookup_table,
        additive_processes=None,
        multiplicative_processes=mult_processes,
        noprocess_prior=_noprocess_prior(priors["baseline"], "beta_land"),
        sigma_struct=priors["sigma_struct"],
        rho=None,
        sigma_process_struct=None,
        sigma_mult_struct=priors["sigma_mult_struct"],
        delta_priors=None,
        eta_priors=priors["eta"],
        likelihood_noise=priors["likelihood_noise"],
        var_name="beta_land",
    )
    with model:
        CMIP_process.add_process_information(
            model,
            "nitrogen",
            priors["nitrogen_obs"]["values"],
            process_sigma=priors["nitrogen_obs"]["sigma"],
            process_iteration=1,
            var_name="beta_land",
        )
    return model


def _model_dataarray(df, value_col, model_col=None):
    """Convert a per-model DataFrame column into an xr.DataArray with a "model" dim.

    Looks for a "model" or "Model" column to supply the coordinate; if
    neither is present, falls back to the DataFrame's index.
    """
    if model_col is None:
        for candidate in ("model", "Model"):
            if candidate in df.columns:
                model_col = candidate
                break

    models = df[model_col].values if model_col is not None else df.index.values
    return xr.DataArray(df[value_col].values, dims="model", coords={"model": models})


def build_gamma_land_model(
    cmip_gamma_land_tropics,
    cmip_gamma_land_extratropics,
    lookup_table,
    iav_df,
    iav_obs_mean,
    iav_obs_sigma,
    priors,
):
    """
    Additive process model for land carbon-climate feedback (gamma_L),
    split into tropical and extratropical components:

        gamma_land = gamma_land_tropics + gamma_land_extratropics

    In either case, the inclusion of a process is assumed to add to the gamma term.  
    Models with the process tell us about delta_p, the process contribution
    Models without still tell us SOMETHING
    
        gamma_land_{i} = baseline_{i} + sum_p(delta_p_{i} + eps_pm) * L_pm

    Both region use the same default priors, except permafrost is  forced to be 0 in
    the tropics. Extratropics/tropics gets its own baseline and its own
    delta_p per process (e.g. delta_nitrogen is estimated independently
    for tropics and extratropics).

    The tropics branch is additionally constrained by the emergent relationship between gamma_land_tropics and the
    interannual variability (IAV) of the atmospheric CO2 growth rate

    Parameters
    ----------
    cmip_gamma_land_tropics : xr.DataArray  shape (model,)
        CMIP6 gamma_L values in the tropics.
    cmip_gamma_land_extratropics : xr.DataArray  shape (model,)
        CMIP6 gamma_L values in the extratropics (
        gamma_land - gamma_land_tropics). Must cover the same models, in
        the same order, as cmip_gamma_land_tropics.
    lookup_table : xr.DataArray  shape (process, model)
        Process presence binary table, shared by tropics and extratropics.
    iav_df : pandas.DataFrame
        Per-model CMIP-simulated IAV emergent-constraint data, with
        columns "gamma_IAV" (simulated IAV) and "sigma_iav" (its
        uncertainty), and a "model"/"Model" column (or model-labeled
        index) identifying which CMIP6 model each row belongs to. May
        cover only a subset of the models in cmip_gamma_land_tropics
        I SHOULD MAKE THE DATA READ IN PRETTIER, THIS IS A MESS
        PUT EVERYTHING IN DATA UTILS
        
    iav_obs_mean, iav_obs_sigma : float
        Observed real-world IAV value and its uncertainty (e.g. Cox et
        al. 2013's IAV = -4.3 +/- 0.67 PgC/yr/K, from
        utils.data_utils.load_emergent_constraint_evidence()).
    priors : dict
        config.priors.gamma_land -- shared baseline/delta/sigma
        structure applied to both branches (permafrost is overridden to
        0 for the tropics branch internally; PER NORMAN pass a deepcopy so the
        caller's dict is untouched).

    Returns
    -------
    model : pm.Model
        Single combined model exposing gamma_land_tropics,
        gamma_land_extratropics, and gamma_land .
    """
    if not np.array_equal(
        np.asarray(cmip_gamma_land_extratropics.model.values),
        np.asarray(cmip_gamma_land_tropics.model.values),
    ):
        raise ValueError(
            "cmip_gamma_land_tropics and cmip_gamma_land_extratropics must "
            "share the same models, in the same order, since they are "
            "combined into one gamma_land model."
        )

    additive_processes = lookup_table.process.values.tolist()

    # Tropics: same priors as extratropics, but permafrost =0
    # In practice just set this to a normal dist with reallllly small sigma
    priors_tropics = deepcopy(priors)
    priors_tropics["delta"]["permafrost"] = {"dist": "Normal", "mu": 0.0, "sigma": 1e-6}

    with pm.Model() as model:
        # Process model for the tropics- use CMIP data, the lookup table, and our priors on CMIP structural bias
        CMIP_process.build_vectorized_process_model(
            cmip_gamma_land_tropics,
            lookup_table,
            additive_processes=additive_processes,
            multiplicative_processes=None,
            noprocess_prior=_noprocess_prior(priors_tropics["baseline"], "gamma_land_tropics"),
            sigma_struct=priors_tropics["sigma_struct"],
            rho=None,
            sigma_process_struct=priors_tropics["sigma_process_struct"],
            sigma_mult_struct=None,
            delta_priors=_delta_priors(priors_tropics["delta"]),
            eta_priors=None,
            likelihood_noise=priors_tropics["likelihood_noise"],
            var_name="gamma_land_tropics",
        )
        # Process model for the extratropics- use CMIP data, the lookup table, and our priors on CMIP structural bias
        CMIP_process.build_vectorized_process_model(
            cmip_gamma_land_extratropics,
            lookup_table,
            additive_processes=additive_processes,
            multiplicative_processes=None,
            noprocess_prior=_noprocess_prior(priors["baseline"], "gamma_land_extratropics"),
            sigma_struct=priors["sigma_struct"],
            rho=None,
            sigma_process_struct=priors["sigma_process_struct"],
            sigma_mult_struct=None,
            delta_priors=_delta_priors(priors["delta"]),
            eta_priors=None,
            likelihood_noise=priors["likelihood_noise"],
            var_name="gamma_land_extratropics",
        )

        # Emergent constraint on the tropics 
        add_emergent_constraint(
            model,
            latent_var_name="gamma_land_tropics",
            observable_var_name="IAV",
            observable_CMIP=_model_dataarray(iav_df, "gamma_IAV"),
            observable_CMIP_sigma=_model_dataarray(iav_df, "sigma_iav"),
            observable_obs=iav_obs_mean,
            observable_obs_sigma=iav_obs_sigma,
        )

        # gamma_land = tropics + extratropics
        pm.Deterministic(
            "gamma_land",
            model["gamma_land_tropics"] + model["gamma_land_extratropics"],
        )

    return model


def build_beta_ocean_model(cmip_beta_ocean, O_m, O_obs, O_obs_unc, priors):
    """
    Emergent-constraint model for ocean beta (beta_O).

    A multivariate linear regression links log(beta_O) to three standardized
    ocean observables (AMOC, SSS, CUC) across the CMIP6 ensemble. Conditioning
    on real-world observations then constrains the posterior on beta_O.

    We use log(beta_O) because we can be sure it's positive and taking the log of everything makes multiplicative -> additive

    Parameters
    ----------
    cmip_beta_ocean : xr.DataArray  shape (model,)  CMIP6 beta_O values at 2xCO2
    O_m             : np.ndarray    shape (M, 3)    standardized model observables
    O_obs           : np.ndarray    shape (3,)      standardized observed values
    O_obs_unc       : np.ndarray    shape (3,)      standardized obs uncertainties
    priors          : dict          config.priors.beta_ocean
    """
    theta_m = np.log(cmip_beta_ocean.values)

    with pm.Model() as model:
        a = pm.Normal("a", mu=priors["regression"]["mu"],
                      sigma=priors["regression"]["sigma"], shape=3)
        b = pm.Normal("b", mu=priors["regression"]["mu"],
                      sigma=priors["regression"]["sigma"], shape=3)

        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov", n=3,
            eta=priors["lkj"]["eta"],
            sd_dist=pm.HalfNormal.dist(priors["lkj"]["sd_sigma"]),
            compute_corr=True,
        )

        # Learn emergent relationship from models
        pm.MvNormal("model_obs", mu=a + b * theta_m[:, None], chol=chol, observed=O_m)

        # Prior on log(beta_O), updated through the emergent constraint
        theta = pm.Normal("theta", mu=priors["theta"]["mu"], sigma=priors["theta"]["sigma"])
        O_true = pm.MvNormal("O_true", mu=a + b * theta, chol=chol, shape=3)
        pm.Normal("obs", mu=O_true, sigma=O_obs_unc, observed=O_obs)

        pm.Deterministic("beta_ocean", pm.math.exp(theta))

    return model


def build_joint_ocean_model(cmip_beta_ocean, cmip_gamma_ocean, O_m, O_obs, O_obs_unc, priors):
    """
    Joint emergent-constraint model for beta_O and gamma_O.

    A multivariate linear regression links both log(beta_O) and standardized
    gamma_O to three standardized ocean observables (AMOC, SSS, CUC) across
    the CMIP6 ensemble.  The joint regression captures the posterior correlation
    between the two feedbacks that the observables induce.

    Parameters
    ----------
    cmip_beta_ocean  : xr.DataArray  shape (model,)  CMIP6 beta_O at 2xCO2
    cmip_gamma_ocean : xr.DataArray  shape (model,)  CMIP6 gamma_O at 2xCO2
    O_m              : np.ndarray    shape (M, 3)    standardized model observables
    O_obs            : np.ndarray    shape (3,)      standardized observed values
    O_obs_unc        : np.ndarray    shape (3,)      standardized obs uncertainties
    priors           : dict          config.priors.joint_ocean
    """
    theta_m     = np.log(cmip_beta_ocean.values)
    gamma_vals  = cmip_gamma_ocean.values
    gamma_mean  = float(gamma_vals.mean())
    gamma_std   = float(gamma_vals.std())
    gamma_s     = (gamma_vals - gamma_mean) / gamma_std

    X_m = np.column_stack([theta_m, gamma_s])  # (M, 2)

    with pm.Model() as model:
        a = pm.Normal("a", mu=priors["regression"]["mu"],
                      sigma=priors["regression"]["sigma"], shape=3)
        b = pm.Normal("b", mu=priors["regression"]["mu"],
                      sigma=priors["regression"]["sigma"], shape=(3, 2))

        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov", n=3,
            eta=priors["lkj"]["eta"],
            sd_dist=pm.HalfNormal.dist(priors["lkj"]["sd_sigma"]),
            compute_corr=True,
        )

        # Learn joint emergent relationship from models
        pm.MvNormal("model_obs", mu=a + X_m @ b.T, chol=chol, observed=O_m)

        # Priors on latent true predictors
        theta = pm.Normal("theta",
                          mu=priors["theta"]["mu"],
                          sigma=priors["theta"]["sigma"])
        gamma_s_true = pm.Normal("gamma_s_true",
                                 mu=priors["gamma_s_true"]["mu"],
                                 sigma=priors["gamma_s_true"]["sigma"])
        X_true = pm.math.stack([theta, gamma_s_true])

        # True latent climate observable
        O_true = pm.MvNormal("O_true", mu=a + b @ X_true, chol=chol, shape=3)
        pm.Normal("obs", mu=O_true, sigma=O_obs_unc, observed=O_obs)

        # Transform back to physical units
        pm.Deterministic("beta_ocean", pm.math.exp(theta))
        pm.Deterministic("gamma_ocean", gamma_std * gamma_s_true + gamma_mean)

    return model


def build_gamma_ocean_model(cmip_gamma_ocean, priors):
    """
    DEPRECATED: USE MULTIVARIATE EC INSTEAD
    Correlated random-effects model for ocean gamma feedback (gamma_O).

    sigma_struct is fixed to the observed CMIP6 intermodel spread rather than
    inferred, to stabilize sampling given the small ensemble size.

    Parameters
    ----------
    cmip_gamma_ocean : xr.DataArray  shape (model,)  CMIP6 gamma_O values at 2xCO2
    priors           : dict          config.priors.gamma_ocean
    """
    sigma_struct = float(np.std(cmip_gamma_ocean.values))
    return CMIP_process.build_vectorized_process_model(
        cmip_gamma_ocean,
        sigma_struct={"sigma_struct": sigma_struct},
        var_name="gamma_ocean",
    )


def add_emergent_constraint(
    model,
    latent_var_name,
    observable_var_name,
    observable_CMIP,
    observable_CMIP_sigma,
    observable_obs,
    observable_obs_sigma,
    priors={},
):
    """
    Add an emergent constraint submodel to an existing model:

        observable_CMIP ~ Normal(m * latent_var_CMIP + b, sigma_total)
        observable_obs  ~ Normal(m * X_true + b, observable_obs_sigma)

    where sigma_total combines the scatter of the emergent regression with
    each CMIP model's own uncertainty in its simulated observable:

        sigma_total = sqrt(regression_sigma**2 + observable_CMIP_sigma**2)

   Adapted from CMIP_BAYES/bayes

    Parameters
    ----------
    model : pm.Model
        Existing PyMC model exposing the latent variable `latent_var_name`
        and its per-model counterpart `{latent_var_name}_CMIP` (dims
        "model"), e.g. as built by a correlated random-effects model. 
    latent_var_name : str
        Name of the latent variable already present in `model`.
    observable_var_name : str
        Name used to build the new variables in this submodel
        (e.g. "AMOC").
    observable_CMIP : xr.DataArray, dims ("model",)
        CMIP-simulated values of the observable. May cover only a subset
        of `model`'s models -- it is reindexed against `model.coords["model"]`
        and missing models are masked out.
    observable_CMIP_sigma : float or xr.DataArray, dims ("model",)
        Per-model uncertainty on the simulated observable. Pass a scalar
        to share one value across all models, or an xr.DataArray aligned
        with `observable_CMIP` (same "model" coordinate) for per-model
        values.
    observable_obs : float
        Observed (real-world) value of the observable.
    observable_obs_sigma : float
        Observational uncertainty on `observable_obs`.
    priors : dict, optional
        Optional callable prior constructors keyed by "m", "b", and
        "regression_sigma". If a key is absent, default weakly
        informative priors are used.
    """

    with model:
        # Inherit X_true and X_m from the host model
        X_true = model[f"{latent_var_name}"]
        X_m = model[f"{latent_var_name}_CMIP"]

        # Emergent constraint may be calculated on a subset of models
        # Reindex to align
        observable_CMIP = observable_CMIP.reindex(
            model=model.coords["model"],
            fill_value=np.nan,
        )

        # Only grab the models that have simulated observables
        mask = ~np.isnan(observable_CMIP.values)

        # observable_CMIP_sigma can be shared across models (scalar) or
        # given per model (xr.DataArray aligned like observable_CMIP)
        if np.isscalar(observable_CMIP_sigma):
            cmip_sigma_masked = float(observable_CMIP_sigma)
        else:
            observable_CMIP_sigma = observable_CMIP_sigma.reindex(
                model=model.coords["model"],
                fill_value=np.nan,
            )
            cmip_sigma_masked = observable_CMIP_sigma[mask].values

        # ===== Regression parameters =====
        if "m" in priors:
            m = priors["m"](f"m_{observable_var_name}")
        else:
            m = pm.Normal(f"m_{observable_var_name}", 0, 5)

        if "b" in priors:
            b = priors["b"](f"b_{observable_var_name}")
        else:
            b = pm.Normal(f"b_{observable_var_name}", 0, 5)

        if "regression_sigma" in priors:
            sigma_Y = priors["regression_sigma"](f"regression_sigma_{observable_var_name}")
        else:
            sigma_Y = pm.HalfNormal(f"regression_sigma_{observable_var_name}", 1.0)

        # Assume regression scatter and per-model simulated-observable uncertainty are independent sources of error
      
        sigma_total = pm.Deterministic(
            f"sigma_{observable_var_name}_CMIP_total",
            pm.math.sqrt(sigma_Y ** 2 + cmip_sigma_masked ** 2),
        )

        # ===== Emergent relationship in the ensemble =====
        pm.Normal(
            f"{observable_var_name}_CMIP_lik",
            mu=m * X_m[mask] + b,
            sigma=sigma_total,
            observed=observable_CMIP[mask].values,
        )

        # ===== Real-world emergent constraint =====
        pm.Normal(
            f"{observable_var_name}_obs_lik",
            mu=m * X_true + b,
            sigma=observable_obs_sigma,
            observed=observable_obs,
        )
