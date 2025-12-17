# TCRE_ASSESSMENT

**Bayesian analysis framework for TCRE Assessment**

## Overview

This repository provides a Bayesian analysis framework to support the assessment of the Transient Climate Response to cumulative carbon Emissions (TCRE). The TCRE is a key metric in climate science, used to estimate the relationship between cumulative CO₂ emissions and global temperature increase.

## Features

- Bayesian inference to estimate TCRE parameters
- Tools for processing climate model outputs and observational datasets
- Reproducible analysis workflows
- Multiple lines of evidence: processes and historical observations

## Quick start
```
git clone https://github.com/netzeroasap/TCRE_Assessment.git
cd TCRE_Assessment
conda env create -f pymc_environment.yml
conda activate tcre
```
### Prerequisites

- Python 3.8+
- Recommended: [conda](https://docs.conda.io/) or [venv](https://docs.python.org/3/library/venv.html) for environment management


## Usage

Example scripts and Jupyter notebooks are available in the `notebooks/` directory.
- TCRE_from_DAMIP: uses CMIP6 simulations from the Detection and Attribution Model Intercomparison Project (DAMIP) to calculate the historical warming attributable to CO2.  TCRE is calculated from CO2 emissions.
- TCRE_from_process: uses process-based simulations, CMIP earth system models, and observed emergent constraints to calculate process-informed posteriors for the CO2 fertilization effect and the climate effect (CURRENTLY LAND ONLY).

```
jupyter lab notebooks/analysis.ipynb
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes or enhancements.

## License

[MIT License](LICENSE)

## Acknowledgements

Go here
