# TCRE_ASSESSMENT

**Bayesian analysis framework for TCRE Assessment**

## Overview

This repository provides a Bayesian analysis framework to support the assessment of the Transient Climate Response to cumulative carbon Emissions (TCRE). The TCRE is a key metric in climate science, used to estimate the relationship between cumulative CO₂ emissions and global temperature increase.

## Features

- Bayesian inference to estimate TCRE parameters
- Tools for processing climate model outputs and observational datasets
- Reproducible analysis workflows
- Visualization utilities for assessment and reporting

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: [conda](https://docs.conda.io/) or [venv](https://docs.python.org/3/library/venv.html) for environment management

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/netzeroasap/TCRE_ASSESSMENT.git
    cd TCRE_ASSESSMENT
    ```
2. (Optional) Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Example scripts and Jupyter notebooks are available in the `examples/` directory.

```
python scripts/run_bayesian_analysis.py --input data/input_file.csv --output results/
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes or enhancements.

## License

[MIT License](LICENSE)

## Acknowledgements

Go here
