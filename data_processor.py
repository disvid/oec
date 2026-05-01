# =============================================================================
# data_processor.py  –  Load → Decouple → Normalise → Window → Loaders
#
# Key addition: get_single_branch_loaders(timestep)
#   Returns loaders where x1 = window of given length, x2=x3=dummy.
#   Used so SingleBranchModel ablations get correctly-sized inputs.
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import config
from current_decoupler import CurrentDecoupler


class DataProcessor:

    def __init__(self, csv_path=None):
        if csv_path is None:
            if os.path.exists(config.DATA_PATH):
                csv_path = config.DATA_PATH
            elif os.path.exists(config.DATA_PATH_LOCAL):
                csv_path = config.DATA_PATH_LOCAL
            else:
                raise FileNotFoundError(
                    f"Dataset not found.\nTried:\n"
                    f"  {config.DATA_PATH}\n  {config.DATA_PATH_LOCAL}")
        self.csv_path = csv_path

        self.scaler_v  = MinMaxScaler()
        self.scaler_ps = MinMaxScaler()
        self.scaler_fr = MinMaxScaler()

        self.voltage = self.current = self.soc = None
        self.I_ps = self.I_fr = None
        self.voltage_norm = self.I_ps_norm = self.I_fr_norm = None
        self.df_raw = None
        self.Y_test = None
        self.n_train = self.n_val = None

    # ── Load ───────────────────────────────────────────────────────────
    def load(self):
        print(f"\n[DataProcessor] Loading '{self.csv_path}' …")
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()

        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ","").replace("_","")
            if any(k in cl for k in ("voltage","volt","v(")):
                col_map.setdefault(config.COL_VOLTAGE, c)
            elif any(k in cl for k in ("current","curr","i(")):
                col_map.setdefault(config.COL_CURRENT, c)
            elif "soc" in cl:
                col_map.setdefault(config.COL_SOC, c)

        missing = [k for k in (config.COL_VOLTAGE, config.COL_CURRENT, config.COL_SOC)
                   if k not in col_map]
        if missing:
            raise ValueError(
                f"Cannot find columns: {missing}\n"
                f"CSV has: {list(df.columns)}\n"
                "Set COL_* in config.py.")

        self.df_raw  = df.copy()
        self.voltage = df[col_map[config.COL_VOLTAGE]].values.astype(np.float32)
        self.current = df[col_map[config.COL_CURRENT]].values.astype(np.float32)
        self.soc     = df[col_map[config.COL_SOC]].values.astype(np.float32)
        if self.soc.max() > 1.5:
            self.soc /= 100.0

        print(f"  ✓ {len(df):,} rows  |  "
              f"V: {self.voltage.min():.3f}–{self.voltage.max():.3f}  "
              f"I: {self.current.min():.3f}–{self.current.max():.3f}  "
              f"SOC: {self.soc.min()*100:.1f}%–{self.soc.max()*100:.1f}%")
        return self

    # ── Decouple ───────────────────────────────────────────────────────
    def decouple(self):
        print("\n[DataProcessor] Decoupling current …")
        N    = len(self.current)
        I_ps = np.zeros(N, dtype=np.float32)
        I_fr = np.zeros(N, dtype=np.float32)

        I_stable = float(self.current[0])
        fluc_dur = 0

        bar = tqdm(range(N), desc="  Decoupling", unit="sample",
                   dynamic_ncols=True, miniters=N//100,
                   bar_format="{l_bar}{bar:30}{r_bar}")
        for t in bar:
            start = max(0, t - config.WINDOW_N)
            std   = float(np.std(self.current[start:t+1]))
            thr   = config.K_THRESHOLD * std
            delta = float(self.current[t]) - I_stable

            if abs(delta) <= thr:
                I_stable = config.LAMBDA1*I_stable + config.LAMBDA2*float(self.current[t])
                I_ps[t]  = I_stable
                I_fr[t]  = 0.0
                fluc_dur = 0
            else:
                fluc_dur += 1
                if fluc_dur <= config.FLUCTUATION_LIMIT:
                    I_ps[t] = I_stable
                    I_fr[t] = delta
                else:
                    end     = min(t + config.FLUCTUATION_LIMIT, N)
                    I_stable= float(np.mean(self.current[t:end]))
                    I_ps[t] = I_stable
                    I_fr[t] = 0.0
                    fluc_dur= 0
        bar.close()

        self.I_ps, self.I_fr = I_ps, I_fr
        print(f"  ✓ I_ps [{I_ps.min():.2f}, {I_ps.max():.2f}]  "
              f"I_fr [{I_fr.min():.2f}, {I_fr.max():.2f}]")
        return self

    # ── Normalise ──────────────────────────────────────────────────────
    def normalise(self):
        print("\n[DataProcessor] Normalising …")
        n_tr = int(len(self.voltage) * config.TRAIN_RATIO)
        self.scaler_v.fit(self.voltage[:n_tr].reshape(-1,1))
        self.scaler_ps.fit(self.I_ps[:n_tr].reshape(-1,1))
        self.scaler_fr.fit(self.I_fr[:n_tr].reshape(-1,1))

        self.voltage_norm = self.scaler_v.transform(
            self.voltage.reshape(-1,1)).ravel().astype(np.float32)
        self.I_ps_norm = self.scaler_ps.transform(
            self.I_ps.reshape(-1,1)).ravel().astype(np.float32)
        self.I_fr_norm = self.scaler_fr.transform(
            self.I_fr.reshape(-1,1)).ravel().astype(np.float32)
        print("  ✓ Done")
        return self

    # ── Build windows (static, reusable) ───────────────────────────────
    @staticmethod
    def _make_windows(v, ps, fr, soc, t1, t2, t3):
        """
        Slide window of length t1; branch-2 uses last t2, branch-3 uses last t3.
        Returns X1(N,t1,3), X2(N,t2,3), X3(N,t3,3), Y(N,1).
        """
        N  = len(v) - t1
        X1 = np.zeros((N, t1, 3), dtype=np.float32)
        X2 = np.zeros((N, t2, 3), dtype=np.float32)
        X3 = np.zeros((N, t3, 3), dtype=np.float32)
        Y  = np.zeros((N, 1),     dtype=np.float32)

        bar = tqdm(range(N), desc="  Building windows", unit="win",
                   dynamic_ncols=True, miniters=max(1,N//100),
                   bar_format="{l_bar}{bar:30}{r_bar}")
        for i in bar:
            e = i + t1
            X1[i,:,0]=v[i:e];      X1[i,:,1]=ps[i:e];      X1[i,:,2]=fr[i:e]
            X2[i,:,0]=v[e-t2:e];   X2[i,:,1]=ps[e-t2:e];   X2[i,:,2]=fr[e-t2:e]
            X3[i,:,0]=v[e-t3:e];   X3[i,:,1]=ps[e-t3:e];   X3[i,:,2]=fr[e-t3:e]
            Y[i,0]   =soc[e]
        bar.close()
        return X1, X2, X3, Y

    @staticmethod
    def _make_single_windows(v, ps, fr, soc, timestep):
        """
        Build windows of a single fixed length `timestep`.
        Returns X(N,timestep,3), Y(N,1).
        Used for single-branch ablation models.
        """
        N = len(v) - timestep
        X = np.zeros((N, timestep, 3), dtype=np.float32)
        Y = np.zeros((N, 1),           dtype=np.float32)
        bar = tqdm(range(N), desc=f"  Windows T={timestep}", unit="win",
                   dynamic_ncols=True, miniters=max(1,N//100),
                   bar_format="{l_bar}{bar:25}{r_bar}")
        for i in bar:
            e = i + timestep
            X[i,:,0]=v[i:e]; X[i,:,1]=ps[i:e]; X[i,:,2]=fr[i:e]
            Y[i,0]  =soc[e]
        bar.close()
        return X, Y

    # ── Standard loaders (3-branch MTS model) ─────────────────────────
    def get_loaders(self):
        self.load().decouple().normalise()

        print("\n[DataProcessor] Building MTS windows …")
        X1, X2, X3, Y = self._make_windows(
            self.voltage_norm, self.I_ps_norm, self.I_fr_norm, self.soc,
            config.T1, config.T2, config.T3)

        n     = len(Y)
        n_tr  = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)
        self.n_train = n_tr
        self.n_val   = n_val

        def _ld(x1,x2,x3,y,sh):
            ds = TensorDataset(torch.from_numpy(x1), torch.from_numpy(x2),
                               torch.from_numpy(x3), torch.from_numpy(y))
            return DataLoader(ds, batch_size=config.BATCH_SIZE,
                              shuffle=sh, num_workers=config.NUM_WORKERS)

        tr  = _ld(X1[:n_tr],           X2[:n_tr],           X3[:n_tr],           Y[:n_tr],           True)
        val = _ld(X1[n_tr:n_tr+n_val], X2[n_tr:n_tr+n_val], X3[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], False)
        te  = _ld(X1[n_tr+n_val:],     X2[n_tr+n_val:],     X3[n_tr+n_val:],     Y[n_tr+n_val:],     False)

        self.Y_test = Y[n_tr+n_val:]
        print(f"\n  ✓ train={n_tr:,}  val={n_val:,}  test={n-n_tr-n_val:,}")
        return tr, val, te

    # ── Single-branch loaders (ablation) ──────────────────────────────
    def get_single_branch_loaders(self, timestep: int):
        """
        Returns (tr_loader, val_loader, te_loader, Y_test) where every
        sample has shape (timestep, 3).  x1=window, x2=x3=zero dummies.
        The SingleBranchModel only reads x1, so dummies are never used.
        """
        # Normalised arrays must already be built (call get_loaders first)
        assert self.voltage_norm is not None, "Call get_loaders() first."

        X, Y = self._make_single_windows(
            self.voltage_norm, self.I_ps_norm, self.I_fr_norm, self.soc,
            timestep)

        n     = len(Y)
        n_tr  = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)

        # Dummy tensors for x2 / x3 (shape doesn't matter, never accessed)
        dummy = np.zeros((1,1,3), dtype=np.float32)

        def _ld(x, y, sh):
            # Pass X as x1; x2, x3 are dummies expanded to batch size
            class _DS(torch.utils.data.Dataset):
                def __init__(self, X, Y):
                    self.X = torch.from_numpy(X)
                    self.Y = torch.from_numpy(Y)
                    self.d = torch.zeros(1,1,3)
                def __len__(self): return len(self.Y)
                def __getitem__(self, i):
                    return self.X[i], self.d[0], self.d[0], self.Y[i]
            return DataLoader(_DS(x,y), batch_size=config.BATCH_SIZE,
                              shuffle=sh, num_workers=config.NUM_WORKERS)

        tr  = _ld(X[:n_tr],           Y[:n_tr],           True)
        val = _ld(X[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], False)
        te  = _ld(X[n_tr+n_val:],     Y[n_tr+n_val:],     False)
        y_te = Y[n_tr+n_val:]
        return tr, val, te, y_te