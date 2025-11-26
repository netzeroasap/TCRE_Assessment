"""
Prior specifications for Bayesian TCRE analysis.
"""
from bayes.priors import DistributionType, PriorSpec
from typing import Dict, Callable, List, Any
import pymc as pm
import json



class PriorConfiguration:
    """
    Manager for all prior specifications in the TCRE analysis.
    
    This class provides methods to:
    - Define priors based on physical constraints
    - Validate completeness
    - Export to PyMC-compatible format
    - Modify priors for sensitivity analysis
    """
    
    def __init__(self, scenario: str = "default"):
        """
        Initialize prior configuration.
        
        Parameters
        ----------
        scenario : str
            One of: "default", "conservative", "optimistic", "ipcc_informed"
        """
        self.scenario = scenario
        self.priors: Dict[str, PriorSpec] = {}
        self.hyperpriors: Dict[str, PriorSpec] = {}
        
        # Load defaults
        self._load_default_priors()
        self._load_default_hyperpriors()
        
        # Apply scenario modifications
        #if scenario != "default":
        #    self._apply_scenario(scenario)
    
    def _load_default_priors(self):
        """Load default prior specifications based on physical constraints.
        I am using the ones from the analysis.ipynb notebook from Kate.
        """
        
        # Baseline parameters
        self.add_prior(PriorSpec(
            name="βL",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0, 'upper': 10},
            description="Baseline land carbon uptake response to CO2",
            source="Default range"
        ))
        
        self.add_prior(PriorSpec(
            name="γLT",
            distribution=DistributionType.NORMAL,
            params={'mu': -100, 'sigma': 100},
            description="Gamma tropical long-term",
            source="Wide uninformative prior"
        ))
        
        self.add_prior(PriorSpec(
            name="γLX",
            distribution=DistributionType.NORMAL,
            params={'mu': -100, 'sigma': 100},
            description="Gamma extratropical",
            source="Wide uninformative prior"
        ))
        
        # NITROGEN CYCLE
        # Beta scaling: 0 to 1 (reduces uptake)
        # Gamma scaling: can be negative (could increase storage)
        self.add_prior(PriorSpec(
            name="η_nitrogen",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0.001, 'upper': 1.0},
            description="Beta scaling for nitrogen cycle (0 to 1, reduces CO2 fertilization)",
            source="Nov 14, 2025 discussion: Pierre noted gamma could be -ve"
        ))
        
        self.add_prior(PriorSpec(
            name="ν_nitrogen",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0.001, 'upper': 1.0},
            description="Gamma scaling for nitrogen (can be negative in theory, but bounded here)",
            source="Nov 14, 2025: Could turn negative gamma into positive"
        ))
        
        # FIRE
        # Beta: won't change sign, so positive, but uncertain if >1 or <1
        # Gamma: increases carbon loss, so >=1
        self.add_prior(PriorSpec(
            name="η_fire",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0.001, 'upper': 2.0},
            description="Beta scaling for fire (positive, uncertain direction)",
            source="Nov 14, 2025: Won't change sign, unclear if >1 or <1"
        ))
        
        self.add_prior(PriorSpec(
            name="ν_fire",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0.001, 'upper': 2.0},
            description="Gamma scaling for fire (>=1, increases carbon loss)",
            source="Nov 14, 2025: Fire increases carbon loss, so scaling >=1"
        ))
        
        # VEGETATION DYNAMICS
        # Beta: likely enables greater uptake, so >1
        # Gamma: could be either sign
        self.add_prior(PriorSpec(
            name="η_vegetation",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0.001, 'upper': 2.0},
            description="Beta scaling for veg dynamics (>1, enables greater uptake)",
            source="Nov 14, 2025: Likely enables greater uptake, so >1"
        ))
        
        self.add_prior(PriorSpec(
            name="ν_vegetation",
            distribution=DistributionType.UNIFORM,
            params={'lower': 0.001, 'upper': 2.0},
            description="Gamma scaling for veg dynamics (either sign possible)",
            source="Nov 14, 2025: Could increase or decrease, either sign"
        ))
        
        # PERMAFROST (additive terms)
        # Beta: could be positive (more/deeper soils increase CO2 response)
        # Gamma: should be negative (increases carbon loss)
        self.add_prior(PriorSpec(
            name="δβ_permafrost",
            distribution=DistributionType.NORMAL,
            params={'mu': 1.0, 'sigma': 0.00001},
            description="Additive beta term for permafrost (could be +ve)",
            source="Nov 14, 2025: Pierre suggested more/deeper soils could increase response"
        ))
        
        self.add_prior(PriorSpec(
            name="δγ_permafrost",
            distribution=DistributionType.NORMAL,
            params={'mu': -100.0, 'sigma': 100},
            description="Additive gamma term for permafrost (negative, releases carbon)",
            source="Nov 14, 2025: Should be negative, IPCC numbers for bounds"
        ))
    
    def _load_default_hyperpriors(self):
        """Load default hyperprior specifications."""
        
        # Model spread in process scaling
        self.add_hyperprior(PriorSpec(
            name="τ_eta",
            distribution=DistributionType.HALF_NORMAL,
            params={'sigma': 0.5},
            description="Precision for eta scaling factors (1/variance of model spread)",
            source="Assumes moderate spread in process-based runs"
        ))
        
        self.add_hyperprior(PriorSpec(
            name="τ_nu",
            distribution=DistributionType.HALF_NORMAL,
            params={'sigma': 0.5},
            description="Precision for nu scaling factors",
            source="Assumes moderate spread in process-based runs"
        ))
        
        # Emergent constraint parameters
        self.add_hyperprior(PriorSpec(
            name="m",
            distribution=DistributionType.NORMAL,
            params={'mu': 0.0, 'sigma': 10.0},
            description="Slope of emergent constraint relationship",
            source="Uninformative prior on slope"
        ))
        
        self.add_hyperprior(PriorSpec(
            name="b",
            distribution=DistributionType.NORMAL,
            params={'mu': 0.0, 'sigma': 10.0},
            description="Intercept of emergent constraint relationship",
            source="Uninformative prior on intercept"
        ))
        
        # Covariance structure
        # Note: This needs special handling in PyMC
        self.hyperpriors["chol"] = PriorSpec(
            name="chol",
            distribution=DistributionType.NORMAL,  # Placeholder, handled specially
            params={'eta': 2.0, 'n': 2, 'sd': 1.0},
            description="Cholesky factor for beta-gamma correlation",
            source="LKJ prior with eta=2 (mild preference for independence)"
        )
    
    """ def _apply_scenario(self, scenario: str):
        #Apply modifications for different scenarios.
        
        if scenario == "conservative":
            # Widen all uncertainties
            for prior in self.priors.values():
                if 'sigma' in prior.params:
                    prior.params['sigma'] *= 2.0
                    prior.source += " (conservative: 2x uncertainty)"
        
        elif scenario == "optimistic":
            # Narrow uncertainties, assume processes work as expected
            for prior in self.priors.values():
                if 'sigma' in prior.params:
                    prior.params['sigma'] *= 0.5
                    prior.source += " (optimistic: 0.5x uncertainty)"  """
            
    def add_prior(self, prior: PriorSpec):
        """Add or update a prior specification."""
        self.priors[prior.name] = prior
    
    def add_hyperprior(self, prior: PriorSpec):
        """Add or update a hyperprior specification."""
        self.hyperpriors[prior.name] = prior
    
    def get_prior(self, name: str) -> PriorSpec:
        """Get a prior specification by name."""
        if name in self.priors:
            return self.priors[name]
        elif name in self.hyperpriors:
            return self.hyperpriors[name]
        else:
            raise KeyError(f"Prior '{name}' not found")
    
    def modify_prior(self, name: str, **new_params):
        """
        Modify parameters of an existing prior.
        
        Example
        -------
        config.modify_prior("η_nitrogen", mu=0.8, sigma=0.1)
        """
        prior = self.get_prior(name)
        prior.params.update(new_params)
        prior.source += f" (modified: {new_params})"
    
    def to_pymc_dict(self) -> tuple[Dict[str, Callable], Dict[str, Callable]]:
        """
        Export to PyMC-compatible dictionary format.
        
        Returns
        -------
        tuple
            (priors_dict, hyperpriors_dict) where each is a dict of lambda functions
        """
        priors_dict = {name: spec.to_pymc() for name, spec in self.priors.items()}
        
        # Special handling for chol
        hyperpriors_dict = {}
        for name, spec in self.hyperpriors.items():
            if name == "chol":
                # Custom handler for LKJ Cholesky
                hyperpriors_dict[name] = lambda n: pm.LKJCholeskyCov(
                    n,
                    n=spec.params['n'],
                    eta=spec.params['eta'],
                    sd_dist=pm.HalfNormal.dist(spec.params['sd'])
                )
            else:
                hyperpriors_dict[name] = spec.to_pymc()
        
        return priors_dict, hyperpriors_dict
    
    def validate(self) -> tuple[List[str], List[str]]:
        """
        Validate that all required priors are specified.
        
        Returns
        -------
        tuple
            (missing_priors, missing_hyperpriors)
        """
        required_priors = [
            "βL", "γLT", "γLX",
            "η_nitrogen", "η_fire", "η_vegetation",
            "ν_nitrogen", "ν_fire", "ν_vegetation",
            "δβ_permafrost", "δγ_permafrost"
        ]
        
        required_hyperpriors = ["m", "b", "τ_eta", "τ_nu", "chol"]
        
        missing_priors = [p for p in required_priors if p not in self.priors]
        missing_hyperpriors = [h for h in required_hyperpriors if h not in self.hyperpriors]
        
        return missing_priors, missing_hyperpriors
    
    def summary(self) -> str:
        """Generate a human-readable summary of all priors."""
        lines = [f"\n{'='*80}"]
        lines.append(f"PRIOR CONFIGURATION: {self.scenario}")
        lines.append(f"{'='*80}\n")
        
        lines.append("PRIORS:")
        lines.append("-" * 80)
        for name, spec in self.priors.items():
            lines.append(f"{name:20} {spec.distribution.value:20} {str(spec.params):30}")
            lines.append(f"{'':20} {spec.description}")
            lines.append(f"{'':20} Source: {spec.source}\n")
        
        lines.append("\nHYPERPRIORS:")
        lines.append("-" * 80)
        for name, spec in self.hyperpriors.items():
            lines.append(f"{name:20} {spec.distribution.value:20} {str(spec.params):30}")
            lines.append(f"{'':20} {spec.description}")
            lines.append(f"{'':20} Source: {spec.source}\n")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"PriorConfiguration(scenario='{self.scenario}', n_priors={len(self.priors)}, n_hyperpriors={len(self.hyperpriors)})"


    def get_state_dict(self) -> Dict[str, Any]:
        """
        Extracts a dictionary representation of the configuration state 
        suitable for hashing.
        """
        state = {
            "scenario": self.scenario,
            "priors": {},
            "hyperpriors": {}
        }
        
        # Helper to serialize a spec
        def serialize_spec(spec):
            return {
                "dist": spec.distribution.value,
                # Sort keys to ensure consistent ordering for hashing
                "params": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                          for k, v in sorted(spec.params.items())} 
            }

        for name, spec in sorted(self.priors.items()):
            state["priors"][name] = serialize_spec(spec)
            
        for name, spec in sorted(self.hyperpriors.items()):
            state["hyperpriors"][name] = serialize_spec(spec)
            
        return state


# Convenience function for quick usage
def get_prior_config(scenario: str = "default", overrides: dict = None) -> PriorConfiguration:
    config = PriorConfiguration(scenario=scenario)
    
    # Apply specific overrides passed from CLI/YAML
    if overrides:
        for prior_name, params in overrides.items():
            # Check if prior exists to avoid typos
            if prior_name in config.priors:
                config.modify_prior(prior_name, **params)
            elif prior_name in config.hyperpriors:
                 # You might need a modify_hyperprior method or generic modify
                 config.modify_prior(prior_name, **params) 
            else:
                print(f"Warning: {prior_name} not found in config.")
                
    return config
