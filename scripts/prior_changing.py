import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from bayes.priors_config import get_prior_config
from bayes.data_utils import generate_config_id
# from bayes.model import build_tcre_model  # Assuming you have a model builder function
from bayes.io_manager import save_experiment, list_experiments

def run_analysis_pipeline(config_obj, label):
    """
    Runs the model for a specific config and returns the trace + id.
    """
    # 1. Generate Unique ID for this setup
    config_id = generate_config_id(config_obj)
    print(f"--- Running {label} [ID: {config_id}] ---")
    
    # 2. Get Pymc dictionaries
    priors, hyperpriors = config_obj.to_pymc_dict()
    
    # 3. Build and Sample Model (Mocking the model context here)
    # In reality: with build_tcre_model(priors, hyperpriors):
    with pm.Model() as model:
        # --- MOCK MODEL STRUCTURE FOR DEMONSTRATION ---
        # This represents your actual TCRE model structure
        
        # Example: Using the prior dictionary
        # Note: We call the lambda function with the variable name
        beta_L = priors['βL']('βL') 
        
        # If we are tweaking nitrogen, let's include it
        eta_nit = priors['η_nitrogen']('η_nitrogen')
        
        # Dummy likelihood for demonstration
        obs = pm.Normal('obs', mu=beta_L * eta_nit, sigma=1.0, observed=[2.5, 3.0, 2.8])
        
        # Sample
        trace = pm.sample(1000, tune=500, return_inferencedata=True, progressbar=False)
        
    return config_id, trace

# ==========================================
# 1. Run Baseline (The "Old" Way / Default)
# ==========================================
config_baseline = get_prior_config("default")
id_base, trace_base = run_analysis_pipeline(config_baseline, "Baseline")

# ==========================================
# 2. Run Tweaked (The "New" Way)
# ==========================================
config_new = get_prior_config("default")

# Apply tweaks (e.g., tighter constraints on Nitrogen based on new paper)
config_new.modify_prior("η_nitrogen", lower=0.4, upper=0.6)
config_new.modify_prior("βL", lower=2.0, upper=5.0)

id_new, trace_new = run_analysis_pipeline(config_new, "Tweaked_Nitrogen")

# ==========================================
# 3. Compare Posteriors
# ==========================================
print(f"\nComparing Baseline ({id_base}) vs Tweaked ({id_new})")

var_names = ["βL", "η_nitrogen"]

az.plot_density(
    [trace_base, trace_new],
    var_names=var_names,
    data_labels=[f"Base ({id_base})", f"New ({id_new})"],
    shade=0.3
)

plt.suptitle(f"Posterior Comparison: {id_base} vs {id_new}")
plt.tight_layout()
plt.show()

config_id = save_experiment(config_baseline, trace_base, run_name="baseline_example")
print("Saved config_id is", config_id)
config_id = save_experiment(config_new, trace_new, run_name="Nitrogen_Test_01")
print("Saved config_id is", config_id)


### LET'S LOAD SAVED EXAMPLE
from bayes.io_manager import load_experiment
import arviz as az
import matplotlib.pyplot as plt
# 1. Load by Name (Easy to remember)
hash_base, conf_base, trace_base = load_experiment("baseline_example")
hash_nitro, conf_nitro, trace_nitro = load_experiment("Nitrogen_Test_01")

# 2. Load by Hash (If you only have the ID from a log file)
# conf_old, trace_old = load_experiment("a1b2c3d4")

var_names = ["βL", "η_nitrogen"]

az.plot_density(
    [trace_base, trace_nitro],
    var_names=var_names,
    data_labels=[f"Base ({hash_base})", f"New ({hash_nitro})"],
    shade=0.3
)

plt.suptitle(f"Posterior Comparison: {hash_base} vs {hash_nitro}")
plt.tight_layout()
plt.show()
