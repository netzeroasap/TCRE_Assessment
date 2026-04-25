"""
Default prior hyperparameters for the TCRE component models.

All values are plain Python numbers — no PyMC or numpy required here.
The model builders in bayes/model_builders.py translate these into PyMC
distributions at model-construction time.

Physical formula
----------------
TCRE = alpha * AF,   AF = 1 / (K + beta + alpha * gamma)

  alpha = TCR / pco2_ref          (K ppm^-1)
  beta  = beta_land + beta_ocean  (PgC ppm^-1)
  gamma = gamma_land + gamma_ocean (PgC K^-1)
  K     = 2.12 PgC ppm^-1        (CO2-to-carbon conversion)

All parameters evaluated at 2xCO2 (CO2 doubled from 280 to 560 ppm).
"""
import math

# Physical constant
K = 2.12  # PgC ppm^-1

# ── alpha (transient climate sensitivity) ────────────────────────────────
# Derived from a prior on TCR (Transient Climate Response):
#   TCR ~ Normal(TCR_mu, TCR_sigma)  K
#   alpha = TCR / pco2_ref           K ppm^-1
alpha = {
    "TCR_mu":    1.7,    # K — central estimate of TCR
    "TCR_sigma": 0.36,   # K — uncertainty on TCR
    "pco2_ref":  280.0,  # ppm — pre-industrial CO2 (= CO2 change at doubling)
}

# ── beta_land (land carbon-concentration feedback, PgC ppm^-1) ──────────
# Model: beta_land = baseline * prod_p(eta_p ^ L_pm)
#   where L_pm in {0,1} flags whether CMIP6 model m includes process p,
#   and eta_p = exp(log_eta_p) is the multiplicative factor for process p.
beta_land = {
    # Baseline beta_L before process adjustments
    # LogNormal enforces positivity; mu=0,sigma=0.5 spans ~0.6–1.6 PgC ppm^-1
    "baseline": {"dist": "LogNormal", "mu": 0.0, "sigma": 0.5},

    # log(eta_p) ~ Normal(mu, sigma) for each multiplicative process
    # eta < 1 reduces beta_L, eta > 1 enhances it
    "eta": {
        "nitrogen": {"mu": -0.2, "sigma": 0.3},  # N-limitation weakens land sink
        "fire":     {"mu":  0.0, "sigma": 0.3},
        "veg":      {"mu":  0.0, "sigma": 0.3},
    },

    # Model-level spread in log(eta) for each process (sd of eps_mult)
    "sigma_mult_struct": {"nitrogen": 0.01, "fire": 0.01, "veg": 0.01},

    # Structural variance: sigma_struct ~ HalfNormal(sigma)
    "sigma_struct": {"sigma": 0.5},

    # Likelihood noise (sd around CMIP6 values, PgC ppm^-1)
    "likelihood_noise": 0.1,

    # Direct observations of eta_nitrogen from model-comparison experiments:
    #   ACCESS-ESM CN/C ratio -> 0.69,  UKESM1 ctrl/nonlim ratio -> 0.77
    "nitrogen_obs": {"values": [0.69, 0.77], "sigma": 0.1},
}

# ── gamma_land (land carbon-climate feedback, PgC K^-1) ─────────────────
# Model: gamma_land = baseline + sum_p(delta_p + eps_pm) * L_pm
#   where delta_p is the process contribution and eps_pm ~ N(0, sigma_p)
#   is a model-level random effect.
gamma_land = {
    # Baseline gamma_L before process contributions — deliberately uninformative
    "baseline": {"dist": "Normal", "mu": 0.0, "sigma": 100.0},

    # Additive process contributions (delta_p)
    "delta": {
        # Permafrost thaw always releases carbon on warming -> constrained negative
        "permafrost": {"dist": "NegLogNormal", "mu": math.log(30), "sigma": 1.0},
        # Nitrogen availability increases with warming -> constrained positive
        "nitrogen":   {"dist": "LogNormal",    "mu": math.log(10), "sigma": 1.0},
        "fire":       {"dist": "Normal",       "mu": 0.0,          "sigma": 10.0},
        "veg":        {"dist": "Normal",       "mu": 0.0,          "sigma": 10.0},
    },

    # Model-level random-effect sd for each process (PgC K^-1)
    "sigma_process_struct": {
        "permafrost": 100.0,
        "fire":        30.0,
        "veg":         50.0,
        "nitrogen":     5.0,
    },

    # Structural variance: sigma_struct ~ HalfNormal(sigma)
    "sigma_struct": {"sigma": 40.0},

    # Likelihood noise (sd around CMIP6 values, PgC K^-1)
    "likelihood_noise": 5.0,
}

# ── beta_ocean (ocean carbon-concentration feedback, PgC ppm^-1) ─────────
# Emergent constraint: beta_O is linked to three observable ocean metrics
# (AMOC, SSS, CUC) via a multivariate linear regression across CMIP6 models.
# Conditioning on real-world observations then constrains the posterior.
beta_ocean = {
    # Prior on theta = log(beta_O)
    "theta": {"mu": 0.0, "sigma": 0.5},

    # Prior on regression intercepts (a) and slopes (b) — one per observable
    "regression": {"mu": 0.0, "sigma": 10.0},

    # LKJCholeskyCov hyperpriors for the cross-observable covariance matrix
    "lkj": {"eta": 2.0, "sd_sigma": 5.0},
}

# ── gamma_ocean (ocean carbon-climate feedback, PgC K^-1) ────────────────
# Simple correlated random-effects model; sigma_struct is set from the
# observed CMIP6 spread at model-build time rather than inferred.
gamma_ocean = {
    "baseline": {"dist": "Normal", "mu": 0.0, "sigma": 100.0},
}
