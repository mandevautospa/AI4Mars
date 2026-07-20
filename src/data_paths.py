"""
src/data_paths.py
=================
Central location for all project path constants.

Using pathlib.Path keeps paths cross-platform (works on Windows, macOS, Linux)
and avoids hard-coded separator characters like '\\' or '/'.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

# __file__ is the path to this script.
# .parent gives the src/ directory.
# .parent again gives the project root.
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()


# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"          # downloaded archives / extracted files
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # cleaned / resized versions
SAMPLES_DATA_DIR: Path = DATA_DIR / "samples"  # small subset for quick iteration


# ---------------------------------------------------------------------------
# Model and output directories
# ---------------------------------------------------------------------------

MODELS_DIR: Path = PROJECT_ROOT / "models"     # saved .pth checkpoint files
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"    # matplotlib plots saved to disk
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"  # predicted mask images
LOGS_DIR: Path = OUTPUTS_DIR / "logs"          # training CSV / TensorBoard logs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ensure_project_dirs() -> None:
    """Create all project directories if they do not already exist.

    Safe to call multiple times — it will never overwrite existing content.
    Call this at the top of any notebook or script before writing files.
    """
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLES_DATA_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("All project directories are ready.")
