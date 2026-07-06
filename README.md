# Mars Terrain Segmentation for Rover Navigation

A computer vision project using NASA's AI4Mars dataset to train a semantic segmentation model that classifies Martian terrain in rover images. The pipeline covers data preprocessing, PyTorch model training, IoU evaluation, visual error analysis, and lays the groundwork for uncertainty-aware navigation support.

---

## What is Semantic Segmentation?

Semantic segmentation is a computer vision task where every pixel in an image is assigned a class label. Unlike object detection (which draws bounding boxes), segmentation produces a pixel-level map — useful for understanding _where_ each terrain type is in a Mars rover image.

## About the AI4Mars Dataset

The [AI4Mars dataset](https://zenodo.org/record/4033453) was released by NASA/JPL and contains thousands of Mars rover images (Curiosity, Opportunity, Spirit) with crowd-sourced pixel-level terrain labels. Classes include **soil**, **bedrock**, **sand**, and **big rock**. The goal was to accelerate autonomous rover navigation research.

> **Note:** You must download the dataset manually. See setup instructions below.

---

## Repository Structure

```
mars-terrain-segmentation/
│
├── notebooks/
│   ├── 00_nasa_api_discovery.ipynb      # Discover dataset metadata via NASA/Zenodo APIs
│   ├── 01_dataset_inspection.ipynb      # Inspect extracted files and verify structure
│   ├── 02_dataset_viewer.ipynb          # Visualize image/mask pairs and overlays
│   ├── 03_baseline_training.ipynb       # Train a minimal segmentation model
│   └── 04_evaluation_error_analysis.ipynb  # Evaluate predictions and analyse errors
│
├── src/
│   ├── data_paths.py     # Path constants and directory helpers
│   ├── nasa_catalog.py   # NASA/Zenodo API functions
│   ├── dataset.py        # PyTorch Dataset class and file helpers
│   ├── transforms.py     # Image/mask preprocessing utilities
│   ├── visualize.py      # Matplotlib visualization helpers
│   ├── metrics.py        # Pixel accuracy and IoU metrics
│   └── train_utils.py    # Training loop, checkpointing, evaluation
│
├── data/
│   ├── raw/              # Raw downloads (git-ignored)
│   ├── processed/        # Processed data (git-ignored)
│   └── samples/          # Small samples for quick testing (git-ignored)
│
├── models/               # Saved checkpoints (git-ignored)
│
├── outputs/
│   ├── figures/          # Saved plots (committed)
│   ├── predictions/      # Model predictions (git-ignored)
│   └── logs/             # Training logs (git-ignored)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions (VS Code + Jupyter)

### 1. Clone the Repository

```bash
git clone https://github.com/mandevautospa/AI4Mars.git
cd AI4Mars
```

### 2. Create and Activate a Virtual Environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you get a script execution error, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register the Kernel with Jupyter

```powershell
python -m ipykernel install --user --name mars-seg --display-name "Python (mars-seg)"
```

### 5. Open in VS Code

Open the repository folder in VS Code, then open any notebook. Select the **Python (mars-seg)** kernel when prompted.

---

## Notebook Workflow

| # | Notebook | Goal |
|---|----------|------|
| 00 | `00_nasa_api_discovery.ipynb` | Use NASA CKAN and Zenodo APIs to locate dataset metadata and download links |
| 01 | `01_dataset_inspection.ipynb` | Inspect extracted files, count images/masks, verify pairing logic |
| 02 | `02_dataset_viewer.ipynb` | Visualise image/mask overlays and audit class label quality |
| 03 | `03_baseline_training.ipynb` | Train a minimal CNN segmentation baseline end-to-end |
| 04 | `04_evaluation_error_analysis.ipynb` | Evaluate predictions with pixel accuracy and per-class IoU |

---

## Current Milestone

> **Load Mars rover image/mask pairs, visualise overlays, and verify class labels before model training.**

Start with `00_nasa_api_discovery.ipynb` to find download links, then manually download and extract the AI4Mars dataset into `data/raw/`. Then run `01_dataset_inspection.ipynb` to verify the structure before proceeding.

---

## Future Work

- **Per-class IoU reporting** — detailed breakdown by terrain class
- **Uncertainty-aware segmentation** — predict confidence alongside class labels for safer navigation
- **Rover-to-rover generalisation** — transfer between Curiosity, Opportunity, and Spirit data
- **Hazard / traversability maps** — convert class predictions into actionable navigation masks
