# =============================================================================
# evaluator.py  –  Metrics + ablation + decoupling comparison.
# Fix: single-branch ablation uses get_single_branch_loaders(timestep)
#      so each model sees the correct window length.
# =============================================================================

import os
import numpy as np
import torch
import config
from model   import MTS_CNN_LSTM, SingleBranchModel, MTS_TCN, MTS_CNN
from trainer import Trainer
from data_processor import DataProcessor
from tqdm import tqdm


# ── Shape-safe metric helpers ────────────────────────────────────────────────
def _align(y, yhat):
    n = min(len(y.ravel()), len(yhat.ravel()))
    return y.ravel()[:n], yhat.ravel()[:n]

def mae(y, yhat):
    y, yhat = _align(y, yhat); return float(np.mean(np.abs(y-yhat)))

def max_error(y, yhat):
    y, yhat = _align(y, yhat); return float(np.max(np.abs(y-yhat)))

def rmse(y, yhat):
    y, yhat = _align(y, yhat)
    return float(np.sqrt(np.mean((y-yhat)**2)))

def print_metrics(y_true, y_pred, label=""):
    y_true, y_pred = _align(np.asarray(y_true), np.asarray(y_pred))
    m = dict(MAE=mae(y_true,y_pred),
             MAX=max_error(y_true,y_pred),
             RMSE=rmse(y_true,y_pred))
    pad = max(0, 40-len(label))
    print(f"\n  ┌─ {label} {'─'*pad}┐")
    for k,v in m.items():
        print(f"  │  {k:<6}: {v*100:7.4f} %                              │")
    print(f"  └{'─'*43}┘")
    return m


# ── Quick-train helper ───────────────────────────────────────────────────────
def quick_train(model, tr_loader, val_loader, te_loader,
                epochs=None, label=""):
    if epochs is None:
        epochs = config.ABLATION_EPOCHS

    import config as cfg
    orig_ep, orig_pat = cfg.EPOCHS, cfg.PATIENCE
    cfg.EPOCHS   = epochs
    cfg.PATIENCE = max(5, epochs // 3)

    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    ckpt = os.path.join(cfg.BASE_DIR, f"_tmp_{safe}.pt")

    t    = Trainer(model, ckpt=ckpt)
    t.fit(tr_loader, val_loader)
    pred = t.predict(te_loader)

    cfg.EPOCHS   = orig_ep
    cfg.PATIENCE = orig_pat
    if os.path.exists(ckpt): os.remove(ckpt)
    return pred


# ── Single-branch ablation  ──────────────────────────────────────────────────
def run_single_branch_ablation(dp: DataProcessor):
    """
    Train 6 single-branch CNN-LSTM models (kernel=1 and kernel=7, each at
    T1/T2/T3 time steps) using correctly-sized windows for each timestep.

    Returns dict:
      { "k=1 T=T3": pred_array, "k=1 T=T2": ..., ..., "k=7 T=T1": ... }
    """
    timesteps = {
        "T1": config.T1,
        "T2": config.T2,
        "T3": config.T3,
    }
    kernels = [1, 7]
    results = {}

    configs_list = [(k, ts_name, ts_val)
                    for k in kernels
                    for ts_name, ts_val in timesteps.items()]

    bar = tqdm(configs_list, desc="  Ablation", unit="model",
               dynamic_ncols=True, bar_format="{l_bar}{bar:30}{r_bar}")

    for k, ts_name, ts_val in bar:
        label = f"k={k} T={ts_name}"
        bar.set_description(f"  Ablation [{label}]")

        # Build loaders with the CORRECT window length for this branch
        tr_, val_, te_, y_te_ = dp.get_single_branch_loaders(ts_val)

        model = SingleBranchModel(kernel1=k, kernel2=k)
        pred  = quick_train(model, tr_, val_, te_,
                            epochs=config.ABLATION_EPOCHS,
                            label=label.replace(" ","_"))
        results[label] = (pred.ravel(), y_te_.ravel())

    bar.close()
    return results


# ── Decoupling strategy comparison ──────────────────────────────────────────
def run_decoupling_comparison(dp: DataProcessor):
    from current_decoupler import CurrentDecoupler
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import TensorDataset, DataLoader
    import torch

    strategies = {
        "Proposed method": "dynamic",
        "No decoupling":   "none",
        "Fourier dec.":    "fourier",
        "EMD dec.":        "emd",
        "DWT dec.":        "dwt",
    }
    predictions = {}

    bar = tqdm(strategies.items(), desc="  Decoupling strategies",
               unit="strategy", dynamic_ncols=True,
               bar_format="{l_bar}{bar:25}{r_bar}")

    for name, kind in bar:
        bar.set_description(f"  Decoupling: {name}")

        if kind == "dynamic":
            I_ps, I_fr = CurrentDecoupler(
                config.WINDOW_N, config.K_THRESHOLD,
                config.LAMBDA1, config.LAMBDA2, config.FLUCTUATION_LIMIT
            ).run(dp.current)
        elif kind == "none":
            I_ps, I_fr = CurrentDecoupler.no_decoupling(dp.current)
        elif kind == "fourier":
            I_ps, I_fr = _fourier_decouple(dp.current)
        elif kind == "emd":
            I_ps, I_fr = _emd_decouple(dp.current)
        elif kind == "dwt":
            I_ps, I_fr = _dwt_decouple(dp.current)

        n    = len(I_ps)
        n_tr = int(n * config.TRAIN_RATIO)
        n_val= int(n * config.VAL_RATIO)

        sc_v = MinMaxScaler().fit(dp.voltage[:n_tr].reshape(-1,1))
        sc_ps= MinMaxScaler().fit(I_ps[:n_tr].reshape(-1,1))
        sc_fr= MinMaxScaler().fit(I_fr[:n_tr].reshape(-1,1))

        v_n  = sc_v.transform(dp.voltage.reshape(-1,1)).ravel().astype(np.float32)
        ps_n = sc_ps.transform(I_ps.reshape(-1,1)).ravel().astype(np.float32)
        fr_n = sc_fr.transform(I_fr.reshape(-1,1)).ravel().astype(np.float32)

        X1,X2,X3,Y = DataProcessor._make_windows(
            v_n, ps_n, fr_n, dp.soc,
            config.T1, config.T2, config.T3)

        total  = len(Y)
        n_tr_w = int(total * config.TRAIN_RATIO)
        n_va_w = int(total * config.VAL_RATIO)

        def _ld(x1,x2,x3,y,sh):
            ds = TensorDataset(
                torch.from_numpy(x1), torch.from_numpy(x2),
                torch.from_numpy(x3), torch.from_numpy(y))
            return DataLoader(ds, batch_size=config.BATCH_SIZE,
                              shuffle=sh, num_workers=0)

        tr_  = _ld(X1[:n_tr_w],              X2[:n_tr_w],
                   X3[:n_tr_w],              Y[:n_tr_w],              True)
        val_ = _ld(X1[n_tr_w:n_tr_w+n_va_w], X2[n_tr_w:n_tr_w+n_va_w],
                   X3[n_tr_w:n_tr_w+n_va_w], Y[n_tr_w:n_tr_w+n_va_w], False)
        te_  = _ld(X1[n_tr_w+n_va_w:],       X2[n_tr_w+n_va_w:],
                   X3[n_tr_w+n_va_w:],       Y[n_tr_w+n_va_w:],       False)

        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        pred = quick_train(MTS_CNN_LSTM(), tr_, val_, te_,
                           epochs=config.ABLATION_EPOCHS,
                           label=f"dec_{safe}")
        predictions[name] = pred.ravel()

    bar.close()
    return predictions


def _fourier_decouple(current, cutoff=0.01):
    F    = np.fft.rfft(current)
    freq = np.fft.rfftfreq(len(current))
    Fl   = F.copy(); Fl[np.abs(freq)>cutoff]  = 0
    Fh   = F.copy(); Fh[np.abs(freq)<=cutoff] = 0
    return (np.fft.irfft(Fl, n=len(current)).astype(np.float32),
            np.fft.irfft(Fh, n=len(current)).astype(np.float32))

def _emd_decouple(current):
    from scipy.ndimage import uniform_filter1d
    ps = uniform_filter1d(current.astype(np.float64), size=30).astype(np.float32)
    return ps, (current-ps).astype(np.float32)

def _dwt_decouple(current):
    from scipy.signal import butter, filtfilt
    b,a = butter(4, 0.05, btype='low')
    ps  = filtfilt(b,a,current.astype(np.float64)).astype(np.float32)
    return ps, (current-ps).astype(np.float32)