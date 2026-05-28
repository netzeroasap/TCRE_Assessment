"""
Factory functions for the four TCRE component models.

Each function takes:
  - the relevant CMIP6 data as an xr.DataArray
  - a prior dict from config.priors (plain Python values)

and returns a PyMC model ready for pm.sample().

The heavy lifting (vectorised additive/multiplicative process structure,
partial pooling) is done by CMIP_process.build_vectorized_process_model;
these builders are thin adapters that translate plain-dict priors into the
callable form that function expects.
"""
import numpy as np
import pymc as pm

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
        )
    return model


def build_gamma_land_model(cmip_gamma_land, lookup_table, priors):
    """
    Additive process model for land carbon-climate feedback (gamma_L).

    gamma_L = baseline + sum_p(delta_p + eps_pm) * L_pm

    where delta_p is the mean process contribution and eps_pm ~ N(0, sigma_p)
    is a model-level random effect.

    Parameters
    ----------
    cmip_gamma_land : xr.DataArray  shape (model,)     CMIP6 gamma_L values
    lookup_table    : xr.DataArray  shape (process, model)  process presence flags
    priors          : dict          config.priors.gamma_land
    """
    additive_processes = lookup_table.process.values.tolist()

    return CMIP_process.build_vectorized_process_model(
        cmip_gamma_land,
        lookup_table,
        additive_processes=additive_processes,
        multiplicative_processes=None,
        noprocess_prior=_noprocess_prior(priors["baseline"], "gamma_land"),
        sigma_struct=priors["sigma_struct"],
        rho=None,
        sigma_process_struct=priors["sigma_process_struct"],
        sigma_mult_struct=None,
        delta_priors=_delta_priors(priors["delta"]),
        eta_priors=None,
        likelihood_noise=priors["likelihood_noise"],
        var_name="gamma_land",
    )


def build_beta_ocean_model(cmip_beta_ocean, O_m, O_obs, O_obs_unc, priors):
    """
    Emergent-constraint model for ocean carbon-concentration feedback (beta_O).

    A multivariate linear regression links log(beta_O) to three standardised
    ocean observables (AMOC, SSS, CUC) across the CMIP6 ensemble. Conditioning
    on real-world observations then constrains the posterior on beta_O.

    Parameters
    ----------
    cmip_beta_ocean : xr.DataArray  shape (model,)  CMIP6 beta_O values at 2xCO2
    O_m             : np.ndarray    shape (M, 3)    standardised model observables
    O_obs           : np.ndarray    shape (3,)      standardised observed values
    O_obs_unc       : np.ndarray    shape (3,)      standardised obs uncertainties
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

    A multivariate linear regression links both log(beta_O) and standardised
    gamma_O to three standardised ocean observables (AMOC, SSS, CUC) across
    the CMIP6 ensemble.  The joint regression captures the posterior correlation
    between the two feedbacks that the observables induce.

    Parameters
    ----------
    cmip_beta_ocean  : xr.DataArray  shape (model,)  CMIP6 beta_O at 2xCO2
    cmip_gamma_ocean : xr.DataArray  shape (model,)  CMIP6 gamma_O at 2xCO2
    O_m              : np.ndarray    shape (M, 3)    standardised model observables
    O_obs            : np.ndarray    shape (3,)      standardised observed values
    O_obs_unc        : np.ndarray    shape (3,)      standardised obs uncertainties
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
    Correlated random-effects model for ocean carbon-climate feedback (gamma_O).

    sigma_struct is fixed to the observed CMIP6 intermodel spread rather than
    inferred, to stabilise sampling given the small ensemble size.

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
