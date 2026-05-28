# bayes/__init__.py
from pathlib import Path
import importlib
import os

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"

# --- Automatically import all .py files in this package ---
__all__ = ['PROJECT_ROOT', 'DATA_DIR']  # always include constants

# # Get all Python files in this directory (except __init__.py)
# current_dir = Path(__file__).parent
# for file in current_dir.glob("*.py"):
#     if file.name == "__init__.py":
#         continue
#     module_name = file.stem
#     # Dynamically import and add to package namespace
#     module = importlib.import_module(f".{module_name}", package=__name__)
#     globals()[module_name] = module
#     __all__.append(module_name)