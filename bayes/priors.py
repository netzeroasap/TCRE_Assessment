"""
Dealing with general prior definitions.
"""

import pymc as pm
import numpy as np
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class DistributionType(Enum):
    """Supported prior distribution types."""
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"
    HALF_NORMAL = "half_normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"


@dataclass
class PriorSpec:
    """
    Specification for a single prior distribution.
    
    Attributes
    ----------
    name : str
        Parameter name (e.g., "η_nitrogen")
    distribution : DistributionType
        Type of distribution
    params : dict
        Distribution parameters (mu, sigma, lower, upper, etc.)
    description : str
        Physical interpretation and reasoning
    source : str
        Reference or reasoning source (e.g., "Chris/Pierre discussion", "IPCC AR6")
    """
    name: str
    distribution: DistributionType
    params: Dict[str, Any]
    description: str
    source: str = "Default"
    
    def to_pymc(self, var_name: Optional[str] = None) -> Callable:
        """
        Convert this prior specification to a PyMC distribution function.
        
        Parameters
        ----------
        var_name : str, optional
            Override the variable name
            
        Returns
        -------
        callable
            Lambda function that creates PyMC distribution
        """

        dist_type = self.distribution
        params = self.params
        
        if dist_type == DistributionType.NORMAL:
            return lambda name, **kwargs: pm.Normal(name, mu=params['mu'], sigma=params['sigma'], **kwargs)
        
        elif dist_type == DistributionType.TRUNCATED_NORMAL:
            return lambda name, **kwargs: pm.TruncatedNormal(
                name, 
                mu=params['mu'], 
                sigma=params['sigma'],
                lower=params.get('lower', -np.inf),
                upper=params.get('upper', np.inf), **kwargs
            )
        
        elif dist_type == DistributionType.UNIFORM:
            return lambda name, **kwargs: pm.Uniform(name, lower=params['lower'], upper=params['upper'], **kwargs)
        
        elif dist_type == DistributionType.HALF_NORMAL:
            return lambda name, **kwargs: pm.HalfNormal(name, sigma=params['sigma'], **kwargs)
        
        elif dist_type == DistributionType.LOGNORMAL:
            return lambda name, **kwargs: pm.LogNormal(name, mu=params['mu'], sigma=params['sigma'], **kwargs)
        
        elif dist_type == DistributionType.BETA:
            return lambda name, **kwargs: pm.Beta(name, alpha=params['alpha'], beta=params['beta'], **kwargs)
        
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    def __repr__(self) -> str:
        return f"PriorSpec({self.name}: {self.distribution.value}{self.params})"

