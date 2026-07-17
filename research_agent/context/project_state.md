# AI4Mars Project State

Last updated: 2026-07-17

## Research question

How effectively can pretrained encoder-decoder segmentation models classify navigational terrain classes in AI4Mars under local hardware constraints, and which terrain classes remain failure modes?

This is a working question, not a finalized publication claim. Novelty and appropriate comparison literature still require a systematic review.

## Dataset and task

- Task: multiclass semantic segmentation of Martian rover imagery.
- Current evaluated classes: soil, bedrock, sand, big_rock.
- Dataset: AI4Mars per-pixel terrain segmentation release documented in the repository README.
- Important unresolved validity checks: exact split construction, rover/image-source grouping, leakage risk, ignored-label handling, per-class pixel and image support, and annotation consistency.

## Current preliminary baseline

The following values were reported from one early three-epoch training run. They are preliminary and must not be treated as a stable benchmark without verifying the notebook, split, seed, checkpoint selection, and metric implementation.

| Epoch | Train loss | Validation loss | Pixel accuracy | Mean IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.9714 | 0.7934 | 0.6848 | 0.4755 |
| 2 | 0.7653 | 0.7545 | 0.7049 | 0.4778 |
| 3 | 0.6749 | 0.6416 | 0.7477 | 0.5364 |

Epoch 3 per-class IoU:

- soil: 0.5905
- bedrock: 0.6964
- sand: 0.5810
- big_rock: 0.2776

Current observation: big_rock is the weakest class. This does not yet establish why. Candidate explanations include rarity, small-object scale, annotation ambiguity, class confusion, or training/loss behavior.

## Compute and methodology constraints

- Local GPU: NVIDIA GTX 1080 Ti with 11 GB VRAM.
- During evaluation/error analysis, VS Code reported an out-of-memory condition and the NVIDIA graphics driver crashed. The exact allocation failure has not been isolated.
- Low-compute methodology is part of the research setting. Record image size, batch size, precision, peak VRAM, wall-clock time, model parameters, preprocessing, and recovery steps for each controlled run.

## Current workflow

- `00_nasa_api_discovery.ipynb`: dataset discovery and metadata.
- `01_dataset_inspection.ipynb`: structure and pairing validation.
- `02_dataset_viewer.ipynb`: masks, overlays, and class distributions.
- `03_baseline_training.ipynb`: pretrained segmentation baseline.
- `04_evaluation_error_analysis.ipynb`: prediction metrics and qualitative errors.
- Reusable implementation lives under `src/`.

## Near-term research priorities

1. Freeze and document a trustworthy train/validation/test split with leakage checks.
2. Verify class mapping, ignored pixels, confusion-matrix orientation, and metric aggregation.
3. Establish one reproducible baseline configuration and repeat it across seeds when feasible.
4. Quantify big_rock support and error modes before selecting imbalance remedies.
5. Compare one intervention at a time against the frozen baseline.
6. Track accuracy, per-class IoU, macro metrics, qualitative failures, compute cost, and uncertainty.

## Claims that are not yet justified

- State-of-the-art performance.
- Publication novelty.
- Generalization to unseen rover missions or geographic regions.
- Safety or readiness for autonomous navigation.
- A causal explanation for big_rock underperformance.
