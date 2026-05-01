# =============================================================================
# trainer.py  –  Training loop with tqdm progress bars and live metrics.
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config


class EarlyStopping:
    """Stop training when val loss stops improving for `patience` epochs."""

    def __init__(self, patience=config.PATIENCE, delta=1e-7):
        self.patience = patience
        self.delta    = delta
        self.counter  = 0
        self.best     = np.inf
        self.stop     = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
            return True
        return False


class Trainer:
    """
    Generic training engine with:
      - per-batch tqdm progress bar
      - per-epoch summary line (train loss | val loss | best | ETA)
      - early stopping
      - gradient clipping
    """

    def __init__(self, model: nn.Module, device: str = None,
                 ckpt: str = config.CKPT_PATH):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model   = model.to(self.device)
        self.ckpt    = ckpt
        self.opt     = torch.optim.Adam(model.parameters(), lr=config.LR)
        self.loss_fn = nn.MSELoss()
        self.es      = EarlyStopping()
        self.history = {"train": [], "val": []}

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Trainer] Device : {self.device}")
        print(f"[Trainer] Params : {total:,}")

    # ── One epoch ──────────────────────────────────────────────────────
    def _epoch(self, loader, train: bool, desc: str = "") -> float:
        self.model.train(train)
        total_loss = 0.0
        n_samples  = 0

        bar = tqdm(
            loader,
            desc        = desc,
            leave       = False,        # overwrite bar each epoch
            unit        = "batch",
            dynamic_ncols = True,
            bar_format  = "{l_bar}{bar:25}{r_bar}",
        )

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x1, x2, x3, y in bar:
                x1, x2, x3, y = (t.to(self.device) for t in (x1, x2, x3, y))
                pred = self.model(x1, x2, x3)
                loss = self.loss_fn(pred, y)

                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()

                batch_loss  = loss.item()
                total_loss += batch_loss * len(y)
                n_samples  += len(y)

                # Live loss shown on the right of the bar
                bar.set_postfix(loss=f"{batch_loss:.5f}", refresh=False)

        bar.close()
        return total_loss / max(n_samples, 1)

    # ── Full training loop ─────────────────────────────────────────────
    def fit(self, tr_loader, val_loader) -> dict:
        best_val   = np.inf
        t0         = time.time()
        epoch_times = []

        print(f"\n{'─'*62}")
        print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  "
              f"{'Best':>10}  {'ETA':>8}  {'Status'}")
        print(f"{'─'*62}")

        for ep in range(1, config.EPOCHS + 1):
            t_ep = time.time()

            tr_l  = self._epoch(tr_loader,  True,
                                desc=f"  Ep {ep:3d}/{config.EPOCHS} [train]")
            val_l = self._epoch(val_loader, False,
                                desc=f"  Ep {ep:3d}/{config.EPOCHS} [val  ]")

            self.history["train"].append(tr_l)
            self.history["val"].append(val_l)

            epoch_times.append(time.time() - t_ep)
            avg_ep_time = np.mean(epoch_times[-5:])   # rolling avg of last 5
            remaining   = int(avg_ep_time * (config.EPOCHS - ep))
            eta_str     = f"{remaining//60}m{remaining%60:02d}s"

            saved = ""
            if val_l < best_val:
                best_val = val_l
                torch.save(self.model.state_dict(), self.ckpt)
                saved = "✓ saved"

            # Colour codes: green if improving, yellow if patience counting
            patience_info = (f"patience {self.es.counter}/{self.es.patience}"
                             if self.es.counter > 0 else "")
            print(f"  {ep:6d}  {tr_l:10.6f}  {val_l:10.6f}  "
                  f"{best_val:10.6f}  {eta_str:>8}  {saved}  {patience_info}")

            if self.es.step(val_l):
                print(f"\n  ⏹  Early stopping at epoch {ep}  "
                      f"(no improvement for {config.PATIENCE} epochs).")
                break

        elapsed = int(time.time() - t0)
        print(f"{'─'*62}")
        print(f"  Training complete in {elapsed//60}m{elapsed%60:02d}s  |  "
              f"Best val MSE = {best_val:.6f}")

        # Restore best weights
        self.model.load_state_dict(
            torch.load(self.ckpt, map_location=self.device))
        return self.history

    # ── Inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        self.model.eval()
        out = []
        bar = tqdm(loader, desc="  Predicting", leave=False,
                   unit="batch", dynamic_ncols=True,
                   bar_format="{l_bar}{bar:20}{r_bar}")
        for x1, x2, x3, _ in bar:
            x1, x2, x3 = (t.to(self.device) for t in (x1, x2, x3))
            out.append(self.model(x1, x2, x3).cpu().numpy())
        bar.close()
        return np.concatenate(out, axis=0)