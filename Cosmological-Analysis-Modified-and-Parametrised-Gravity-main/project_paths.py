"""Shared path setup for the project’s folder-organized Python scripts."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent


def configure_import_paths():
    """Make modules in the named workflow folders importable from any entry point."""
    folders = (
        "01_run_configuration",
        "02_plotting",
        "03_data_and_likelihood",
        "04_diagnostics",
        "05_results_and_reconstruction",
    )
    for folder in reversed(folders):
        path = str(PROJECT_ROOT / folder)
        if path not in sys.path:
            sys.path.insert(0, path)
    return PROJECT_ROOT
