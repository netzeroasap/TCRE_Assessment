# bayes/io_manager.py
import json
import shutil
from pathlib import Path
from datetime import datetime
import arviz as az
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from .data_utils import generate_config_id
from .priors_config import PriorConfiguration

# Constants
RESULTS_DIR = Path(__file__).parent.parent / "results"
REGISTRY_PATH = RESULTS_DIR / "registry.json"

def _load_registry() -> Dict:
    """Load the experiment registry JSON."""
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)

def _save_registry(registry: Dict):
    """Save the experiment registry JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)

def save_experiment(
    config: PriorConfiguration, 
    trace: az.InferenceData, 
    run_name: Optional[str] = None,
    overwrite: bool = False
) -> str:
    """
    Saves the config and trace to results/{hash}/.
    Updates the registry with the run_name if provided.
    
    Returns
    -------
    str
        The hash ID of the saved experiment.
    """
    # 1. Generate Hash
    config_id = generate_config_id(config)
    
    # 2. Prepare Directory
    save_dir = RESULTS_DIR / config_id
    if save_dir.exists() and not overwrite:
        print(f"Note: Configuration {config_id} already exists.")
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Config (Metadata)
        with open(save_dir / "config.json", 'w') as f:
            # We use the state dict we created earlier
            json.dump(config.get_state_dict(), f, indent=4, sort_keys=True)
            
        # Save Trace (Data)
        # NetCDF is standard for ArviZ/PyMC
        trace_path = save_dir / "posterior.nc"
        trace.to_netcdf(str(trace_path))
        print(f"Saved data to {save_dir}")

    # 3. Update Registry (Mapping Name -> Hash)
    if run_name:
        registry = _load_registry()
        
        # Check if name exists and points to a different hash
        if run_name in registry and registry[run_name]['hash'] != config_id:
            print(f"Warning: Overwriting name '{run_name}' (was {registry[run_name]['hash']})")
            
        registry[run_name] = {
            "hash": config_id,
            "timestamp": datetime.now().isoformat(),
            "scenario": config.scenario
        }
        _save_registry(registry)
        print(f"Registered '{run_name}' -> {config_id}")
        
    return config_id

def load_experiment(name_or_hash: str) -> Tuple[Dict, az.InferenceData]:
    """
    Load an experiment by its human-readable name OR its hash.
    
    Returns
    -------
    (config_dict, trace)
    """
    registry = _load_registry()
    
    # Determine if input is a name in registry or a raw hash
    target_hash = None
    
    if name_or_hash in registry:
        target_hash = registry[name_or_hash]['hash']
        print(f"Loading '{name_or_hash}' (Hash: {target_hash})...")
    else:
        # Check if it looks like a hash folder that exists
        if (RESULTS_DIR / name_or_hash).exists():
            target_hash = name_or_hash
            print(f"Loading by Hash: {target_hash}...")
        else:
            raise ValueError(f"Could not find experiment: '{name_or_hash}'")
            
    # Load Data
    exp_dir = RESULTS_DIR / target_hash
    
    with open(exp_dir / "config.json", 'r') as f:
        config_dict = json.load(f)
        
    trace = az.from_netcdf(str(exp_dir / "posterior.nc"))
    
    return target_hash, config_dict, trace

def list_experiments() -> pd.DataFrame:
    """Returns a pandas DataFrame of all registered experiments."""
    registry = _load_registry()
    if not registry:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(registry, orient='index')
    df.index.name = "Run Name"
    return df.reset_index()