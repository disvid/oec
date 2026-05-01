# =============================================================================
# config.py  –  Balanced settings: faster than full paper, better than tiny.
# =============================================================================

import os
import multiprocessing

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = r"C:\Users\dishi\Desktop\oec\battery-data.csv"
DATA_PATH_LOCAL = os.path.join(BASE_DIR, "battery-data.csv")
CKPT_PATH       = os.path.join(BASE_DIR, "best_model.pt")
FIG_DIR         = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Data split ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

# ── Current decoupling ──────────────────────────────────────────────────────
WINDOW_N          = 15
K_THRESHOLD       = 0.5
LAMBDA1           = 0.7
LAMBDA2           = 0.3
FLUCTUATION_LIMIT = 3

# ── Multi-time-scale branches ───────────────────────────────────────────────
# Balanced: keeps 6:3:1 ratio from paper, runs in reasonable time on CPU.
# For full paper quality set T1=600, T2=300, T3=100.
T1 = 300   # 5 h
T2 = 150   # 2.5 h
T3 = 50    # 50 min

# ── CNN ─────────────────────────────────────────────────────────────────────
KERNEL1 = 1
KERNEL2 = 7
CNN_OUT = 32   # filters per conv path (64 after concat)

# ── LSTM ─────────────────────────────────────────────────────────────────────
HIDDEN_SIZE     = 64   # restored to paper value
NUM_LSTM_LAYERS = 2    # restored to paper value
DROPOUT         = 0.2

# ── Attention ────────────────────────────────────────────────────────────────
ATTN_HEADS = 1
ATTN_EMBED = 64

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 128   # larger batch → faster epochs on 171k samples
EPOCHS          = 50
LR              = 1e-3
PATIENCE        = 15

# ── Ablation: each baseline gets this many epochs ────────────────────────────
ABLATION_EPOCHS = 20

# ── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0   # 0 is safest on Windows

# ── Feature column hints ─────────────────────────────────────────────────────
COL_VOLTAGE = "Voltage"
COL_CURRENT = "Current"
COL_SOC     = "SOC"