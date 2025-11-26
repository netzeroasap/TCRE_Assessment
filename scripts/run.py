import argparse
import yaml
import sys
import pymc as pm
import arviz as az
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bayes import data_utils, DATA_DIR
from bayes.priors_config import get_prior_config
from bayes.io_manager import save_experiment

# =============================================================================
# 1. MODEL LOGIC (Extracted from compare_analysis.py)
# =============================================================================

def make_emergent_constraint_logic(model, evidence, priors, hyperpriors):
    """
    Defines the PyMC structure for the Emergent Constraint model.
    Ref: compare_analysis.py
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
    
    return {"γLT": γLT, "m": m, "b": b}

def run_ec_experiment(run_name: str, settings: dict, evidence: dict):
    """
    Configures and runs a single experiment based on YAML settings.
    """
    print(f"\n[{run_name}] Initializing...")

    # 1. Initialize Configuration
    base_scenario = settings.get("base_scenario", "default")
    config = get_prior_config(base_scenario)
    
    # 2. Apply Overrides from YAML
    overrides = settings.get("overrides", {})
    print(f"[{run_name}] Applying {len(overrides)} overrides...")
    
    for prior_name, params in overrides.items():
        # Check if the prior exists in the config before modifying
        try:
            config.modify_prior(prior_name, **params)
            print(f"  -> Modified {prior_name}: {params}")
        except KeyError:
            print(f"  -> Warning: '{prior_name}' not found in configuration. Skipping.")

    # 3. Convert to PyMC dictionaries
    priors_dict, hyperpriors_dict = config.to_pymc_dict()

    # 4. Build and Sample
    print(f"[{run_name}] Building Emergent Constraint Model...")
    with pm.Model() as model:
        make_emergent_constraint_logic(
            model, 
            evidence, 
            priors_dict, 
            hyperpriors_dict
        )
        
        # Sampling
        print(f"[{run_name}] Sampling...")
        trace = pm.sample(draws=2000, target_accept=0.95, progressbar=True)
        
    # 5. Save Results
    print(f"[{run_name}] Saving...")
    hash_id = save_experiment(config, trace, run_name=run_name, overwrite=True)
    print(f"[{run_name}] Complete. Hash: {hash_id}")

# =============================================================================
# 2. MAIN CLI RUNNER
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EC experiments from YAML")
    parser.add_argument("--file", type=str, default="experiments.yaml", help="Path to YAML config")
    parser.add_argument("--run", nargs="+", help="Specific experiment names to run (space separated)")
    
    args = parser.parse_args()

    # 1. Load Data (Once)
    print("Loading Evidence Data...")
    evidence = data_utils.load_emergent_constraint_evidence()

    # 2. Load YAML
    yaml_path = Path(args.file)
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)
        
    with open(yaml_path, "r") as f:
        all_experiments = yaml.safe_load(f)

    # 3. Determine which experiments to run
    if args.run:
        # Filter strictly for the names provided in CLI
        experiments_to_run = {k: v for k, v in all_experiments.items() if k in args.run}
        if len(experiments_to_run) < len(args.run):
            print("Warning: Some requested experiments were not found in the YAML file.")
    else:
        # Run everything in the file
        experiments_to_run = all_experiments

    # 4. Execute Loop
    print(f"Found {len(experiments_to_run)} experiments to run.")
    for name, settings in experiments_to_run.items():
        run_ec_experiment(name, settings, evidence)
