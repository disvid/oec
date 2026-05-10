import os, time
import numpy as np
import torch
from tqdm import tqdm

import config
from data_processor import DataProcessor
from model          import MTS_CNN_LSTM, MTS_TCN, MTS_CNN
from trainer        import Trainer
from evaluator      import (quick_train, run_single_branch_ablation,
                             run_decoupling_comparison, print_metrics, _align)
import visualizer as VIS


def _banner(text):
    print(f"\n{'═'*62}")
    print(f"  {text}")
    print(f"{'═'*62}")


def main():
    t0 = time.time()
    _banner("MTS-CNN-LSTM  SOC Estimation  –  Dong et al. (2025)")
    print(f"  PyTorch {torch.__version__}  |  "
          f"{'GPU ✓' if torch.cuda.is_available() else 'CPU'}")

    _banner("STEP 1 / 6  –  Data Loading & Preprocessing")
    dp = DataProcessor()
    tr_loader, val_loader, te_loader = dp.get_loaders()
    Y_test = dp.Y_test.ravel()

    _banner("STEP 2 / 6  –  Diagnostic Figures (Fig 1, 2, 4)")
    VIS.plot_working_curves(dp.voltage, dp.current, dp.soc)
    VIS.plot_decoupling(dp.current, dp.I_ps, dp.I_fr)
    VIS.plot_plateau_region(dp.voltage, dp.soc)
    print("  ✓ Saved.")

    _banner("STEP 3 / 6  –  Training MTS-CNN-LSTM (proposed)")
    main_model = MTS_CNN_LSTM()
    trainer    = Trainer(main_model)
    history    = trainer.fit(tr_loader, val_loader)
    VIS.plot_training_history(history)

    pred_main_raw = trainer.predict(te_loader).ravel()
    pred_main, Y_te = _align(pred_main_raw, Y_test)
    print_metrics(Y_te, pred_main, "MTS-CNN-LSTM (proposed)")

    _banner("STEP 4 / 6  –  Baseline Models")

    print("\n  [4a] MTS-TCN …")
    pred_tcn_raw = quick_train(
        MTS_TCN(), tr_loader, val_loader, te_loader, label="tcn").ravel()
    pred_tcn, _ = _align(pred_tcn_raw, Y_te)
    print_metrics(Y_te, pred_tcn, "MTS-TCN")

    print("\n  [4b] MTS-CNN …")
    pred_cnn_raw = quick_train(
        MTS_CNN(), tr_loader, val_loader, te_loader, label="cnn_base").ravel()
    pred_cnn, _ = _align(pred_cnn_raw, Y_te)
    print_metrics(Y_te, pred_cnn, "MTS-CNN")

    _banner("STEP 5 / 6  –  Single-Branch Ablation + Decoupling Comparison")

    print("\n  [5a] Single-branch kernel ablation …")
    ab_results = run_single_branch_ablation(dp)
    for name, (pred, y_te_ab) in ab_results.items():
        print_metrics(y_te_ab, pred, name)

    print("\n  [5b] Decoupling strategy comparison …")
    dec_preds_raw = run_decoupling_comparison(dp)
    dec_preds = {}
    for name, raw in dec_preds_raw.items():
        p, _ = _align(raw, Y_te)
        dec_preds[name] = p
        print_metrics(Y_te, p, name)

    _banner("STEP 6 / 6  –  Generating Paper Figures")

    print("  Plotting Fig 8 …")
    VIS.plot_fig8_model_comparison(Y_te, pred_main, pred_tcn, pred_cnn)

    print("  Plotting Fig 9 …")
    VIS.plot_fig9_decoupling_strategy(Y_te, dec_preds)

    print("  Plotting Fig 10 …")
    offset  = config.T1                   
    N_full  = len(dp.soc)
    N_te    = len(Y_te)
    err_abs_start = offset + int(N_te * 0.30)
    err_abs_end   = offset + int(N_te * 0.75)
    err_abs_start = min(err_abs_start, N_full - 500)
    err_abs_end   = min(err_abs_end,   N_full)

    VIS.plot_fig10_error_regions(
        dp.voltage, dp.current, dp.soc,
        t_start_abs = err_abs_start,
        t_end_abs   = err_abs_end,
    )

    print("  Plotting Fig 11 …")
    VIS.plot_fig11_kernel1_ablation(Y_te, pred_main, ab_results)

    print("  Plotting Fig 12 …")
    VIS.plot_fig12_kernel7_ablation(Y_te, pred_main, ab_results)

    print("  Plotting Fig 13 …")
    VIS.plot_fig13_model_comparison(Y_te, pred_main, pred_tcn, pred_cnn)

    elapsed = int(time.time() - t0)
    _banner(f"DONE  –  {elapsed//60}m {elapsed%60:02d}s")
    print(f"  Figures  → {config.FIG_DIR}")
    print(f"  Weights  → {config.CKPT_PATH}\n")


if __name__ == "__main__":
    main()
