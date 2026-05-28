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
conda env create -f tcre_environment.yml
conda activate tcre
```
### Prerequisites

- Python 3.8+
- Recommended: [conda](https://docs.conda.io/) 


## Usage

To use  process-based simulations, CMIP earth system models, and
observed emergent constraints to calculate process-informed posteriors
for the CO2 fertilization effect and the climate effect over land and ocean.
```
jupyter notebook notebooks/TCRE_total.ipynb
```
## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes

## License

[MIT License](LICENSE)

