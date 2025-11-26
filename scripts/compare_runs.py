import argparse
import arviz as az
import matplotlib.pyplot as plt
from bayes.io_manager import load_experiment

def compare_runs(id1, id2, variables):
    # Load using your existing io_manager logic
    hash1, conf1, trace1 = load_experiment(id1)
    hash2, conf2, trace2 = load_experiment(id2)
    
    print(f"Comparing {hash1} vs {hash2}")
    
    az.plot_density(
        [trace1, trace2],
        var_names=variables,
        data_labels=[id1, id2],
        shade=0.4
    )
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--var", nargs="+", default=["βL", "γL"])
    
    args = parser.parse_args()
    compare_runs(args.baseline, args.candidate, args.var)
