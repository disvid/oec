# MTS-CNN-LSTM SOC Estimation
**Paper:** Dong et al. (2025) — *"Adaptive SOC estimation of grid-level BESS for multiple operational scenarios"*, Journal of Energy Storage

---

## Project Structure
soc_estimation/
├── battery-data.csv          ← your dataset
├── config.py                 ← all hyperparameters and file paths
├── current_decoupler.py      ← dynamic threshold current decoupling algorithm
├── data_processor.py         ← CSV loading, normalisation, sliding windows, DataLoaders
├── model.py                  ← MTS-CNN-LSTM, MTS-TCN, MTS-CNN, SingleBranchModel
├── trainer.py                ← training loop, early stopping, progress bars
├── evaluator.py              ← MAE / MAX / RMSE metrics, ablation runner
├── visualizer.py             ← diagnostic plots (Fig 1, 2, 4, training loss)
├── paper_figures.py          ← paper-exact Fig 8–13
├── main.py                   ← entry point, runs everything in order
└── figures/                  ← all saved output figures (auto-created)
---

## Requirements

### `requirements.txt`
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
scipy>=1.9.0
tqdm>=4.65.0
### Install

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn matplotlib scipy tqdm

# OR install everything at once from file
pip install -r requirements.txt
```

> For GPU, get your torch install command from [pytorch.org](https://pytorch.org/get-started/locally/)

---

## Setup

Open `config.py` and set your dataset path:

```python
DATA_PATH = r"C:\Users\dishi\Desktop\oec\battery-data.csv"
```

If your CSV column names are different from `Voltage`, `Current`, `SOC`, also update:

```python
COL_VOLTAGE = "Voltage"
COL_CURRENT = "Current"
COL_SOC     = "SOC"
```

---

## Run

```bash
python main.py
```

---

## What `main.py` Does (in order)

| Step | What happens |
|---|---|
| **Step 1** | Loads CSV, runs current decoupling, normalises, builds sliding windows, creates DataLoaders |
| **Step 2** | Saves diagnostic figures: Fig 1 (working curves), Fig 2 (decoupling), Fig 4 (plateau region) |
| **Step 3** | Trains the proposed MTS-CNN-LSTM model with Adam + MSE loss + early stopping |
| **Step 4** | Trains baseline models: MTS-TCN, MTS-CNN, and 6 single-branch ablation models |
| **Step 5** | Generates paper-exact Fig 8–13 and prints MAE / MAX / RMSE for all models |

---

## Output Figures

Saved to the `figures/` folder:

| Figure | File | Description |
|---|---|---|
| Fig 1 | `fig1_working_curves.png` | Voltage, current, SOC over 2 days |
| Fig 2 | `fig2_decoupling.png` | Original vs peak-shaving vs freq-reg current |
| Fig 4 | `fig4_plateau_region.png` | Voltage plateau region zoom |
| Fig 8 | `fig8_lab_model_comparison.png` | Lab test: MTS-CNN-LSTM vs MTS-TCN vs MTS-CNN |
| Fig 9 | `fig9_decoupling_comparison.png` | 5 decoupling strategies compared |
| Fig 10 | `fig10_error_regions.png` | Voltage/current/SOC in high-error regions |
| Fig 11 | `fig11_kernel1_ablation.png` | Single-branch CNN-LSTM, kernel=1, 3 time scales |
| Fig 12 | `fig12_kernel7_ablation.png` | Single-branch CNN-LSTM, kernel=7, 3 time scales |
| Fig 13 | `fig13_realworld_comparison.png` | Real-world long-term model comparison |
| — | `training_loss.png` | Train vs validation MSE loss curve |

---

## Key Settings in `config.py`

| Parameter | Value | Description |
|---|---|---|
| `T1, T2, T3` | `600, 300, 100` | Time steps per branch (minutes) |
| `EPOCHS` | `100` | Max training epochs |
| `PATIENCE` | `20` | Early stopping patience |
| `LR` | `0.001` | Adam learning rate |
| `BATCH_SIZE` | `128` | Training batch size |
| `ABLATION_EPOCHS` | `20` | Epochs for baseline/ablation models |

---

## Estimated Runtime (CPU)

| Stage | Time |
|---|---|
| Data loading + decoupling | ~1 min |
| MTS-CNN-LSTM training | ~2–4 hours |
| Baseline + ablation (8 models) | ~1–2 hours |
| Figure generation | ~1 min |
| **Total** | **~3–6 hours** |

To speed up, reduce `T1/T2/T3` and `EPOCHS` in `config.py`:
```python
T1, T2, T3 = 300, 150, 50   # faster, slightly less accurate
EPOCHS = 30
ABLATION_EPOCHS = 10
```

---

## Common Errors

| Error | Fix |
|---|---|
| `FileNotFoundError` | Update `DATA_PATH` in `config.py` |
| `ValueError: Cannot find columns` | Set `COL_VOLTAGE/COL_CURRENT/COL_SOC` to your actual CSV column names |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Training very slow | Reduce `T1/T2/T3` and `EPOCHS` in `config.py` |

---

## Paper Reference

> Jiawei Dong et al. *"Adaptive SOC estimation of grid-level BESS for multiple operational scenarios."*
> Journal of Energy Storage, 2025. DOI: [10.1016/j.est.2025.117776](https://doi.org/10.1016/j.est.2025.117776)
