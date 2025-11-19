from pathlib import Path

# Get the project root (parent of bayes/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"

# Export for easy access
__all__ = ['PROJECT_ROOT', 'DATA_DIR']
