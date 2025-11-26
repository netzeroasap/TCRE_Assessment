import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bayes import data_utils, DATA_DIR
from bayes.priors_config import get_prior_config

# =============================================================================
# 1. REUSABLE MODEL BUILDER (Refactored from analysis.py)
# =============================================================================

def make_process_model_logic(model, evidence, priors, hyperpriors, coords):
    """
    Sub-component: Handles the Process runs (η and ν scaling).
    """
    # 1. Eta (β scaling)
    etalist = []
    for process in coords["process"]:
        # Support both new (flat dict) and old (potential nested) structures if needed
        # But here we assume standardized dictionary access
        etalist.append(priors[f"η_{process}"](name=f"η_{process}"))
    
    eta = pm.Deterministic("η", pm.math.stack(etalist), dims=("process",))

    # Hyperpriors for spread
    tau_eta = hyperpriors["τ_eta"](name="τ_eta")
    tau_nu = hyperpriors["τ_nu"](name="τ_nu")

    # Likelihood for Eta
    for k, process in enumerate(coords["process"]):
        pm.Normal(f"lik_eta_{process}", eta[k], tau=tau_eta, observed=evidence[f"η_{process}"])

    # 2. Nu (γ scaling)
    nulist = []
    for process in coords["process"]:
        nulist.append(priors[f"ν_{process}"](name=f"ν_{process}"))
        
    nu = pm.Deterministic("ν", pm.math.stack(nulist), dims=("process",))

    # Likelihood for Nu
    for k, process in enumerate(coords["process"]):
        pm.Normal(f"lik_nu_{process}", nu[k], tau=tau_nu, observed=evidence[f"ν_{process}"])

    return {"ν": nu, "η": eta, "τ_eta": tau_eta, "τ_nu": tau_nu}

def make_emergent_constraint_logic(model, evidence, priors, hyperpriors):
    """
    Sub-component: Handles Emergent Constraint (γ_LT).
    """
    n_models_EC = len(evidence['γ_LT'])
    
    # Prior on Tropical Gamma
    γLT = priors["γLT"](name="γLT")
    
    # Hyperpriors on slope/intercept
    m = hyperpriors["m"](name="m")
    b = hyperpriors["b"](name="b")

    # Errors-in-variables model
    x_true = pm.Normal("x_true", mu=evidence['γ_LT'], sigma=evidence['σ_LT'], shape=n_models_EC)
    y_true = m * x_true + b
    
    # Likelihood of observed IAV
    pm.Normal("y_obs", mu=y_true, sigma=evidence['σ_IAV'], shape=n_models_EC, observed=evidence['γ_IAV'])

    # Emergent Constraint Calculation
    mu_obs = m * γLT + b
    pm.Normal("IAV_true", mu=mu_obs, sigma=evidence["IAV_observed_std"], observed=[evidence["IAV_observed_mean"]])
    
    return {"γLT": γLT}

def build_tcre_model(priors, hyperpriors, evidence, coords):
    """
    The Full TCRE Model (The 'bigmodel' from analysis.py).
    """
    D = np.stack([evidence["βL_cmip"], evidence["γL_cmip"]]).T
    
    with pm.Model(coords=coords) as model:
        # 1. Process Model
        proc_out = make_process_model_logic(model, evidence, priors, hyperpriors, coords)
        eta, nu = proc_out["η"], proc_out["ν"]
        
        # 2. Emergent Constraint Model
        ec_out = make_emergent_constraint_logic(model, evidence, priors, hyperpriors)
        gamma_tropics = ec_out["γLT"]
        
        # 3. Data Containers
        CMIP = pm.Data("CMIP", D) # mutable=True if you plan to swap data later

        # 4. Global Parameters
        gamma_extratropics = priors["γLX"](name="γLX")
        gamma = pm.Deterministic("γL", gamma_extratropics + gamma_tropics)
        beta = priors["βL"](name="βL")
        
        # 5. Covariance Structure
        chol, corr, sigma = ol = hyperpriors["chol"](name="chol", n=len(coords["parameter"]))
        Sigma = pm.Deterministic("Sigma", chol.dot(chol.T), dims=("parameter", "cross_parameter"))
        
        # 6. Apply Lookup Table (Scaling Logic)
        # Using pytensor.tensor (pt) for set_subtensor operations
        Eta_arr = pt.ones((len(coords["process"]), len(coords["model"])))
        Nu_arr = pt.ones((len(coords["process"]), len(coords["model"])))
        
        lookup = evidence["lookup table"]
        
        for j in range(len(coords["process"])):
            # Set row j of Eta_arr: if lookup is 1, use eta[j], else 1.0
            Eta_arr = pt.set_subtensor(
                Eta_arr[j, :], 
                pm.math.switch(pt.eq(lookup[j, :], 1), eta[j], 1.0)
            )
            Nu_arr = pt.set_subtensor(
                Nu_arr[j, :], 
                pm.math.switch(pt.eq(lookup[j, :], 1), nu[j], 1.0)
            )
            
        # 7. Calculate Mus
        # Unscaled base values
        B_base = beta / pm.math.prod(eta)
        G_base = gamma / pm.math.prod(nu)
        
        mu_list = []
        for i, _ in enumerate(coords["model"]):
            # Scaling for specific model i
            eta_model = pm.math.prod(Eta_arr[:, i])
            nu_model = pm.math.prod(Nu_arr[:, i])
            
            # Re-scale
            mu_list.append(pm.math.stack([eta_model * B_base, nu_model * G_base]))
            
        mu_totals = pm.Deterministic(
            "all_mus", 
            pm.math.stack(mu_list), 
            dims=("model", "parameter")
        )
        
        # 8. Final Likelihood
        pm.MvNormal('D', mu=mu_totals, chol=chol, observed=CMIP)
        
    return model

# =============================================================================
# 2. SETUP AND EXECUTION
# =============================================================================

# Load Evidence
print("Loading Data...")
coords = data_utils.get_coords()
evidence = (
    data_utils.load_process_evidence(kind="2xCO2") | 
    data_utils.load_emergent_constraint_evidence() | 
    data_utils.load_CMIP_land_data(kind="2xCO2")
)

# ---------------------------------------------------------
# A. THE OLD WAY (Hardcoded in Script)
# ---------------------------------------------------------
# Note: Added **kwargs to lambdas to handle 'dims'/'n' arguments cleanly
old_priors = {
    "βL": lambda name, **kwargs: pm.Uniform(name, 0, 10, **kwargs),
    "γLX": lambda name, **kwargs: pm.Normal(name, -100, 100, **kwargs),
    "γLT": lambda name, **kwargs: pm.Normal(name, -100, 100, **kwargs),
    
    "η_nitrogen": lambda name, **kwargs: pm.Uniform(name, 0.001, 1, **kwargs),
    "η_fire": lambda name, **kwargs: pm.Uniform(name, 0.001, 2, **kwargs),
    "δβ_permafrost": lambda name, **kwargs: pm.Normal(name, 1.0, 0.00001, **kwargs), # Note: Not used in current model logic but present in analysis.py
    "η_vegetation": lambda name, **kwargs: pm.Uniform(name, 0.001, 2, **kwargs),
    
    "ν_nitrogen": lambda name, **kwargs: pm.Uniform(name, 0.001, 1, **kwargs),
    "ν_fire": lambda name, **kwargs: pm.Uniform(name, 0.001, 2, **kwargs),
    "δγ_permafrost": lambda name, **kwargs: pm.Normal(name, -100.0, 100, **kwargs),
    "ν_vegetation": lambda name, **kwargs: pm.Uniform(name, 0.001, 2, **kwargs)
}

old_hyperpriors = {
    "τ_eta": lambda name, **kwargs: pm.HalfNormal(name, 0.5, **kwargs),
    "τ_nu": lambda name, **kwargs: pm.HalfNormal(name, 0.5, **kwargs),
    "m": lambda name, **kwargs: pm.Normal(name, 0, 10, **kwargs),
    "b": lambda name, **kwargs: pm.Normal(name, 0, 10, **kwargs),
    "chol": lambda name, **kwargs: pm.LKJCholeskyCov(
        name, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0), **kwargs
    )
}


print("Running OLD WAY (Hardcoded) - PROCESS MODEL ONLY...")

# 1. Create a new, empty model context.
with pm.Model(coords=coords) as process_model_old:
    
    # 2. Call ONLY the process model logic, just like the fast analysis.py notebook.
    #    We use the logic function defined earlier in compare_analysis.py.
    make_process_model_logic(
        process_model_old, 
        evidence, 
        old_priors, 
        old_hyperpriors, 
        coords
    )
    
    # 3. Sample the simple model.
    #    This is now modeling only the simple Normal distributions, 
    #    which results in the fast sampling speed you saw (Step size ~0.5, Grad evals ~7).
    #    The draws=5000 is included to match the notebook cell exactly.
    trace_old = pm.sample(draws=5000)

# ---------------------------------------------------------
# B. THE NEW WAY (Config Object) - Suggestion
# ---------------------------------------------------------
# If you want the "New Way" section (B) to also run fast for comparison:
print("Running NEW WAY (Config Object) - PROCESS MODEL ONLY...")
config = get_prior_config("default")
new_priors, new_hyperpriors = config.to_pymc_dict()

with pm.Model(coords=coords) as process_model_new:
    make_process_model_logic(
        process_model_new, 
        evidence, 
        new_priors, 
        new_hyperpriors, 
        coords
    )
    trace_new = pm.sample(draws=5000)


##############

"""print("Running OLD WAY (Hardcoded)...")
model_old = build_tcre_model(old_priors, old_hyperpriors, evidence, coords)
with model_old:
    # Short run for comparison
    trace_old = pm.sample(draws=5000)

# ---------------------------------------------------------
# B. THE NEW WAY (Config Object)
# ---------------------------------------------------------
print("Running NEW WAY (Config Object)...")
config = get_prior_config("default")
new_priors, new_hyperpriors = config.to_pymc_dict()

model_new = build_tcre_model(new_priors, new_hyperpriors, evidence, coords)
with model_new:
    trace_new = pm.sample(draws=5000)"""

# =============================================================================
# 3. COMPARISON PLOT
# =============================================================================
print("Generating Comparison Plot...")

# Plot key variables
var_names = ["η_nitrogen", "η_fire"]

az.plot_density(
    [trace_old, trace_new],
    var_names=var_names,
    data_labels=["Old (Hardcoded)", "New (Config Object)"],
    shade=0.4,
    hdi_prob=0.95
)

plt.suptitle("Verification: Posteriors should be identical")
plt.tight_layout()

output_path = ROOT / "plots" / "old_vs_new_comparison.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path)
print(f"Comparison plot saved to {output_path}")
plt.show()

# =============================================================================
# 3. COMPARISON PLOT: POSTERIOR STATS
# =============================================================================
print("Generating Process Model Posterior Plot...")

var_names = ["η_nitrogen", "η_fire"]

# Create a figure with 2 rows (Old vs New)
fig, axes = plt.subplots(2, len(var_names), figsize=(12, 8))

# Row 1: Old Model
az.plot_posterior(
    trace_old, 
    var_names=var_names, 
    hdi_prob=0.95,       # 95% Probability Mass
    point_estimate='mean', # Shows the mean line and label
    ax=axes[0]           # Plot on top row
)

# Row 2: New Model
az.plot_posterior(
    trace_new, 
    var_names=var_names, 
    hdi_prob=0.95, 
    point_estimate='mean',
    ax=axes[1]           # Plot on bottom row
)

# Add row titles for clarity
axes[0, 0].set_ylabel("Old (Hardcoded)", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("New (Config)", fontsize=12, fontweight='bold')

plt.suptitle("Process Model: Mean & 95% HDI Comparison", fontsize=14)
plt.tight_layout()

output_path = ROOT / "plots" / "process_posterior_comparison.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path)
plt.show()




# =============================================================================
# 4. EMERGENT CONSTRAINT COMPARISON (Standalone)
# =============================================================================

def build_emergent_constraint_model(priors, hyperpriors, evidence):
    """Builds the standalone Emergent Constraint model."""
    with pm.Model() as model:
        # Note: We reuse the logic function defined at the top of the script
        make_emergent_constraint_logic(model, evidence, priors, hyperpriors)
    return model

print("\n--- Running EMERGENT CONSTRAINT Comparison ---")

# A. Run Old (Hardcoded)
# Filter priors/hyperpriors to only what's needed for EC to avoid unused var warnings
ec_keys = ["γLT", "m", "b"]
ec_priors_old = {k: v for k, v in old_priors.items() if k in ec_keys}
ec_hyperpriors_old = {k: v for k, v in old_hyperpriors.items() if k in ec_keys}

print("Sampling Old EC Model...")
with build_emergent_constraint_model(ec_priors_old, ec_hyperpriors_old, evidence):
    trace_ec_old = pm.sample(draws=2000, target_accept=0.9)

# B. Run New (Config)
print("Sampling New EC Model...")
# The config object handles key filtering automatically usually, but we pass all
with build_emergent_constraint_model(new_priors, new_hyperpriors, evidence):
    trace_ec_new = pm.sample(draws=2000, target_accept=0.9)

# C. Plot EC Comparison
print("Plotting EC Results...")
az.plot_density(
    [trace_ec_old, trace_ec_new],
    var_names=["γLT", "m"],
    data_labels=["Old (Hardcoded)", "New (Config)"],
    hdi_prob=0.95,
    shade=0.4
)
plt.suptitle("Emergent Constraint Model Verification")
plt.tight_layout()
plt.show()


# =============================================================================
# 5. FULL MODEL COMPARISON (The 'Big Model')
# =============================================================================
print("\n--- Running FULL TCRE MODEL Comparison ---")
# WARNING: This is the heavy model. We use target_accept=0.99 to match analysis.py
# and prevent the sampler from getting stuck (which caused the slowness before).

# A. Run Old
print("Sampling Old Full Model (This may take a minute)...")
model_full_old = build_tcre_model(old_priors, old_hyperpriors, evidence, coords)
with model_full_old:
    # We use 1000 draws to be efficient, matching default analysis.py behavior
    trace_full_old = pm.sample(draws=1000, target_accept=0.99)

# B. Run New
print("Sampling New Full Model...")
model_full_new = build_tcre_model(new_priors, new_hyperpriors, evidence, coords)
with model_full_new:
    trace_full_new = pm.sample(draws=1000, target_accept=0.99)

# C. Plot Full Results
print("Plotting Full Model Results...")
# We plot the high-level variables that depend on the integration of all parts
var_names_full = ["βL", "γLX", "γLT", "γL"]

az.plot_density(
    [trace_full_old, trace_full_new],
    var_names=var_names_full,
    hdi_prob=0.95,
    combined=True,
    data_labels=["Old (Hardcoded)", "New (Config)"],
    shade=0.4
)
plt.suptitle("Full TCRE Model Verification")
plt.tight_layout()
plt.savefig(ROOT / "plots" / "full_model_comparison.png")
plt.show()


