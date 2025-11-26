import argparse
import arviz as az
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bayes.io_manager import load_experiment, list_experiments

def compare_runs(run_identifiers, variables):
    """
    Loads experiments and plots their posteriors.
    """
    traces = []
    labels = []
    
    print(f"--- Comparing {len(run_identifiers)} Experiments ---")
    
    for identifier in run_identifiers:
        try:
            # load_experiment handles both Names and Hash IDs
            run_hash, config, trace = load_experiment(identifier)
            traces.append(trace)
            labels.append(f"{identifier}\n({run_hash})")
            print(f"Loaded: {identifier}")
        except Exception as e:
            print(f"Error loading '{identifier}': {e}")
            return

    if not traces:
        print("No data to plot.")
        return

    print(f"Plotting variables: {variables}")
    
    ax = az.plot_density(
        traces,
        var_names=variables,
        data_labels=labels,
        shade=0.4,
        hdi_prob=0.95
    )
    
    plt.suptitle("Experiment Comparison")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare TCRE Experiment Posteriors")
    
    # Required: List of experiment names or hashes
    parser.add_argument("experiments", nargs="+", help="List of experiment names (from YAML) or Hashes")
    
    # Optional: Variables to plot
    parser.add_argument("--var", nargs="+", default=["γLT", "m", "b"], help="Variables to plot (default: γLT m b)")
    
    # Optional: List available experiments
    parser.add_argument("--list", action="store_true", help="Show all registered experiments and exit")

    args = parser.parse_args()

    if args.list:
        print(list_experiments())
        sys.exit(0)

    compare_runs(args.experiments, args.var)
