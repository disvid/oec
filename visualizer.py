import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import config

# ── Global style exactly matching the paper (Elsevier / serif) ──────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":     "dejavuserif",
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8.5,
    "ytick.labelsize":      8.5,
    "legend.fontsize":      8,
    "legend.frameon":       True,
    "legend.framealpha":    1.0,
    "legend.edgecolor":     "black",
    "legend.borderpad":     0.4,
    "lines.linewidth":      1.0,
    "axes.linewidth":       0.8,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "axes.grid":            True,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.6,
    "grid.color":           "#cccccc",
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     3,
    "ytick.major.size":     3,
})

COL_REAL    = "#8B1A1A"   # very dark red  – Real SOC
COL_MTS     = "#E07030"   # orange         – MTS-CNN-LSTM / Proposed
COL_TCN     = "#00BFBF"   # teal/cyan      – MTS-TCN
COL_CNN_MTS = "#1A237E"   # dark navy      – MTS-CNN

COL_PROPOSED = "#C0392B"  # red-ish        – Proposed method
COL_FOURIER  = "#1ABC9C"  # teal           – Fourier decoupling
COL_EMD      = "#2980B9"  # steel blue     – EMD decoupling
COL_DWT      = "#E91E8C"  # magenta-pink   – Discrete wavelet decoupling
COL_NODEC    = "#6A0DAD"  # purple         – No decoupling

COL_CURRENT = "#1565C0"   # blue   – current
COL_VOLTAGE = "#2E7D32"   # green  – voltage
COL_SOC_RAW = "#C62828"   # red    – SOC

COL_SB_100  = "#1A237E"   # dark blue  – T=100
COL_SB_300  = "#26A69A"   # teal       – T=300
COL_SB_600  = "#6A1B9A"   # purple     – T=600


def _save(fig, name):
    path = os.path.join(config.FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[Visualizer] Saved → {path}")

def plot_working_curves(voltage, current, soc, n_days=2):
    samples = min(n_days * 24 * 60, len(voltage))
    t = np.arange(samples) / 60      # hours
    soc_pct = soc[:samples] * 100 if soc.max() <= 1.0 else soc[:samples]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    axes[0].plot(t, voltage[:samples], color=COL_VOLTAGE, lw=0.8)
    axes[0].set_ylabel("Voltage (V)")

    axes[1].plot(t, current[:samples], color=COL_CURRENT, lw=0.7)
    axes[1].axhline(0, color='k', lw=0.5, ls='--')
    axes[1].set_ylabel("Current (A)")

    axes[2].plot(t, soc_pct, color=COL_SOC_RAW, lw=0.8)
    axes[2].set_ylabel("SOC (%)")
    axes[2].set_xlabel("Time (h)")

    axes[0].set_title("Voltage, Current, and SOC Working Curves (Two-Day Period)")
    _save(fig, "fig1_working_curves.png")

def plot_decoupling(current, I_ps, I_fr, n_minutes=1800):
    n = min(n_minutes, len(current))
    t = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(hspace=0.07)

    for ax, trace, color, label in [
        (axes[0], current[:n], COL_CURRENT,  "Original Current"),
        (axes[1], I_ps[:n],    COL_MTS,      "Peak Shaving Current"),
        (axes[2], I_fr[:n],    COL_PROPOSED, "Frequency Regulation Current"),
    ]:
        ax.plot(t, trace, color=color, lw=0.7, label=label)
        ax.axhline(0, color='k', lw=0.4, ls='--')
        ax.set_ylabel("Current (A)")
        ax.legend(loc="upper right", fontsize=8)

    axes[2].set_xlabel("Time (min)")
    axes[0].set_title("(a) Proposed Current Decoupling Method")
    _save(fig, "fig2_decoupling.png")

def plot_plateau_region(voltage, soc, start=500, length=300):
    end   = min(start + length, len(voltage))
    t     = np.arange(end - start)
    v_seg = voltage[start:end]
    s_seg = soc[start:end] * 100 if soc.max() <= 1.0 else soc[start:end]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    l1, = ax1.plot(t, v_seg, color=COL_VOLTAGE, lw=1.1, label="Voltage (V)")
    l2, = ax2.plot(t, s_seg, color=COL_SOC_RAW, lw=1.1, ls='--', label="SOC (%)")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Voltage (V)", color=COL_VOLTAGE)
    ax2.set_ylabel("SOC (%)",     color=COL_SOC_RAW)
    ax1.tick_params(axis='y', labelcolor=COL_VOLTAGE)
    ax2.tick_params(axis='y', labelcolor=COL_SOC_RAW)

    dv     = np.abs(np.gradient(v_seg))
    smooth = np.convolve(dv, np.ones(20) / 20, mode='same')
    pc     = int(np.argmin(smooth))
    ph     = 30
    ax1.axvspan(max(0, pc-ph), min(len(t)-1, pc+ph), alpha=0.12, color='gold')
    ax1.annotate("Plateau region", xy=(pc, v_seg[pc]),
                 xytext=(pc + 20, v_seg.mean()), fontsize=8, color='darkorange',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]], loc="lower right")
    ax1.set_title("Plateau Region of BESS Voltage–SOC Curve")
    _save(fig, "fig4_plateau_region.png")

def _two_panel(y_true_pct, pred_dict, fname,
               title_a, title_b,
               xlabel_a, xlabel_b,
               xlim_a, xlim_b,
               ylim=(10, 80),
               region_labels=None,   # list of (x_in_b_coords, label_str)
               legend_loc="upper left"):
 
    N      = len(y_true_pct)
    t_mins = np.arange(N)   # time axis in minutes (or seconds for fig8)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.30)

    for ax, xlim, xlabel, title in [
        (ax_a, xlim_a, xlabel_a, title_a),
        (ax_b, xlim_b, xlabel_b, title_b),
    ]:
        # Determine index slice from x-limits
        i0 = max(0, int(xlim[0]))
        i1 = min(N, int(xlim[1]) + 1)
        t_sl = t_mins[i0:i1]

        # Real SOC
        ax.plot(t_sl, y_true_pct[i0:i1],
                color=COL_REAL, lw=1.3, label="Real SOC", zorder=10)

        # Model predictions
        for label, (pred_pct, color, ls, lw) in pred_dict.items():
            p = np.asarray(pred_pct).ravel()
            n_use = min(len(t_sl), len(p) - i0)
            if n_use <= 0:
                continue
            ax.plot(t_sl[:n_use], p[i0:i0+n_use],
                    color=color, lw=lw, linestyle=ls,
                    label=label, zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("SOC (%)", fontsize=9)
        ax.set_title(title, fontsize=10)

        # Legend on panel (a) only at top-left inside box
        if ax is ax_a:
            ax.legend(loc=legend_loc, fontsize=8,
                      handlelength=1.8, labelspacing=0.3)

        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # Region labels (I II III IV) on panel (b)
    if region_labels:
        ylim_b = ax_b.get_ylim()
        y_text = ylim_b[0] + (ylim_b[1] - ylim_b[0]) * 0.88
        for x_pos, text in region_labels:
            if xlim_b[0] <= x_pos <= xlim_b[1]:
                ax_b.text(x_pos, y_text, text, fontsize=9,
                          ha='center', va='center', color='black',
                          fontweight='normal')

    _save(fig, fname)

def plot_fig8_model_comparison(y_test, pred_mts, pred_tcn, pred_cnn):
    # Convert to % if needed
    def _pct(a): return a * 100 if np.asarray(a).max() <= 1.5 else np.asarray(a)
    y_pct = _pct(y_test)
    N     = len(y_pct)

    # For fig8 the paper uses "Time (s)" – the lab dataset has 1s resolution
    # Our dataset is 1-min, so 1 step = 60 s  → multiply x by 60
    scale = 60   # convert minute-index to seconds
    t_s   = np.arange(N) * scale

    pred_dict = {
        "MTS-CNN-LSTM": (_pct(pred_mts), COL_MTS,     "--", 0.95),
        "MTS-TCN":      (_pct(pred_tcn), COL_TCN,     "--", 0.95),
        "MTS-CCN":      (_pct(pred_cnn), COL_CNN_MTS, "--", 0.95),
    }

    # Build in-seconds xlims
    xlim_a = (0, min(N * scale, 42000))
    xlim_b = (0, min(N * scale, 12000))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.30)

    for ax, xlim, title in [
        (ax_a, xlim_a, "(a) 96-hour equivalent range"),
        (ax_b, xlim_b, "(b) 24-hour equivalent range"),
    ]:
        i0 = max(0, int(xlim[0] / scale))
        i1 = min(N, int(xlim[1] / scale) + 1)
        t_sl = t_s[i0:i1]

        ax.plot(t_sl, y_pct[i0:i1], color=COL_REAL, lw=1.3,
                label="Real SOC", zorder=10)

        for label, (pred_pct, color, ls, lw) in pred_dict.items():
            p = np.asarray(pred_pct).ravel()
            n_u = min(len(t_sl), max(0, len(p) - i0))
            if n_u > 0:
                ax.plot(t_sl[:n_u], p[i0:i0+n_u],
                        color=color, lw=lw, ls=ls, label=label, zorder=5)

        ax.set_ylim(20, 80)
        ax.set_xlim(xlim)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("SOC (%)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        if ax is ax_a:
            ax.legend(loc="upper left", fontsize=8,
                      handlelength=1.8, labelspacing=0.3)

    _save(fig, "fig8_lab_model_comparison.png")

def plot_fig9_decoupling_strategy(y_test, dec_preds):
    def _pct(a): return a * 100 if np.asarray(a).max() <= 1.5 else np.asarray(a)
    y_pct = _pct(y_test)
    N     = len(y_pct)

    legend_order = [
        ("Proposed method",          COL_PROPOSED),
        ("Fourier decoupling",        COL_FOURIER),
        ("EMD decoupling",            COL_EMD),
        ("Discrete wavelet decoupling", COL_DWT),
        ("No decoupling",             COL_NODEC),
    ]
    key_to_display = {
        "Proposed method": "Proposed method",
        "Fourier dec.":    "Fourier decoupling",
        "EMD dec.":        "EMD decoupling",
        "DWT dec.":        "Discrete wavelet decoupling",
        "No decoupling":   "No decoupling",
    }

    pred_dict = {}
    for internal_key, (display_name, color) in zip(
            key_to_display.keys(),
            [(d, c) for d, c in legend_order]):
        if internal_key in dec_preds:
            p = _pct(dec_preds[internal_key])
            pred_dict[display_name] = (p, color, "--", 0.85)


    zoom_s, zoom_e = 1800, min(3400, N)
    seg = y_pct[zoom_s:zoom_e]

    region_x = _find_region_labels(y_pct, zoom_s, zoom_e)

    xlim_a = (0, N)
    xlim_b = (zoom_s, zoom_e)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.30)

    for ax, xlim, title, xlabel in [
        (ax_a, xlim_a, "(a) 144-hour range", "Time (min)"),
        (ax_b, xlim_b, "(b) 24-hour range",  "Time (min)"),
    ]:
        i0, i1 = max(0, xlim[0]), min(N, xlim[1])
        t_sl = np.arange(i0, i1)

        ax.plot(t_sl, y_pct[i0:i1], color=COL_REAL, lw=1.3,
                label="Real SOC", zorder=10)

        for display_name, (pred_pct, color, ls, lw) in pred_dict.items():
            p   = np.asarray(pred_pct).ravel()
            n_u = min(len(t_sl), max(0, len(p) - i0))
            if n_u > 0:
                ax.plot(t_sl[:n_u], p[i0:i0+n_u],
                        color=color, lw=lw, ls=ls,
                        label=display_name, zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(10, 70)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("SOC (%)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        if ax is ax_a:
            ax.legend(loc="upper left", fontsize=7.5,
                      handlelength=1.8, labelspacing=0.25)

    if region_x:
        labels = ["I", "II", "III", "IV"]
        ylim_top = ax_b.get_ylim()[1]
        for x_pos, lbl in zip(region_x, labels):
            if zoom_s <= x_pos <= zoom_e:
                # Place label at the local SOC value at that x
                y_at = float(y_pct[int(x_pos)]) if int(x_pos) < N else 60
                ax_b.text(x_pos, y_at + 3, lbl, fontsize=9,
                          ha='center', va='bottom', color='black')

    _save(fig, "fig9_decoupling_comparison.png")


def _find_region_labels(y_pct, zoom_s, zoom_e):
    seg   = y_pct[zoom_s:zoom_e]
    N_seg = len(seg)
    if N_seg < 4:
        return []
    # Split segment into 4 equal quarters and take the argmax of each
    q = N_seg // 4
    positions = []
    for i in range(4):
        chunk = seg[i*q:(i+1)*q]
        if i == 1:   # II is the trough (min)
            positions.append(zoom_s + i*q + int(np.argmin(chunk)))
        else:         # I, III, IV are at local flat tops (max in window)
            positions.append(zoom_s + i*q + int(np.argmax(chunk)))
    return positions

def plot_fig10_error_regions(voltage, current, soc,
                              t_start_abs=1200, t_end_abs=3600):
    N   = len(soc)
    s   = max(0, t_start_abs)
    e   = min(N, t_end_abs)
    t   = np.arange(s, e)          

    soc_pct = soc[s:e] * 100 if soc.max() <= 1.0 else soc[s:e]
    v_seg   = voltage[s:e]
    i_seg   = current[s:e]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True)
    fig.subplots_adjust(hspace=0.06, left=0.14, right=0.95,
                        top=0.97, bottom=0.07)

    axes[0].plot(t, i_seg, color=COL_CURRENT, lw=0.55)
    axes[0].set_ylabel("Current (A)", fontsize=9)
    # Match paper y-limits (symmetric ±50 A)
    i_max = max(abs(i_seg.max()), abs(i_seg.min()), 10)
    axes[0].set_ylim(-i_max * 1.15, i_max * 1.15)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(
        round(i_max / 2 / 5) * 5 or 10))

    axes[1].plot(t, v_seg, color=COL_VOLTAGE, lw=0.65)
    axes[1].set_ylabel("Voltage (V)", fontsize=9)
    # Give a bit of padding around the voltage range
    v_lo = v_seg.min();  v_hi = v_seg.max()
    pad  = (v_hi - v_lo) * 0.15 + 0.1
    axes[1].set_ylim(v_lo - pad, v_hi + pad)

    axes[2].plot(t, soc_pct, color=COL_SOC_RAW, lw=1.0)
    axes[2].set_ylabel("SOC (%)", fontsize=9)
    axes[2].set_xlabel("Time (min)", fontsize=9)
    soc_lo = soc_pct.min();  soc_hi = soc_pct.max()
    axes[2].set_ylim(soc_lo - 5, soc_hi + 5)

    seg_len = len(soc_pct)
    q       = seg_len // 4
    region_chars = ["I", "II", "III", "IV"]
    y_top   = soc_hi + 2
    for i, lbl in enumerate(region_chars):
        chunk_start = i * q
        chunk_end   = (i + 1) * q
        chunk       = soc_pct[chunk_start:chunk_end]
        if len(chunk) == 0:
            continue
        if lbl == "II":   # trough
            local_idx = int(np.argmin(chunk))
        else:             # flat top
            local_idx = int(np.argmax(chunk))
        abs_idx = s + chunk_start + local_idx
        axes[2].text(abs_idx, y_top, lbl, fontsize=9,
                     ha='center', va='bottom', color='black')

    axes[0].set_xlim(s, e)
    _save(fig, "fig10_error_regions.png")
def plot_fig11_kernel1_ablation(y_test, pred_mts, ab_results):
    _plot_single_branch_fig(
        y_test, pred_mts, ab_results,
        kernel=1, fname="fig11_kernel1_ablation.png",
        fig_num=11,
    )


def plot_fig12_kernel7_ablation(y_test, pred_mts, ab_results):
    _plot_single_branch_fig(
        y_test, pred_mts, ab_results,
        kernel=7, fname="fig12_kernel7_ablation.png",
        fig_num=12,
    )


def _plot_single_branch_fig(y_test, pred_mts, ab_results, kernel, fname, fig_num):
    def _pct(a): return a * 100 if np.asarray(a).max() <= 1.5 else np.asarray(a)
    y_pct = _pct(y_test)
    N     = len(y_pct)

    # Build prediction dict in paper legend order
    ts_map = {
        config.T3: f"CNN-LSTM(Time Steps={config.T3}min)",
        config.T2: f"CNN-LSTM(Time Steps={config.T2}min)",
        config.T1: f"CNN-LSTM(Time Steps={config.T1}min)",
    }
    ts_colors = {
        config.T3: COL_SB_100,
        config.T2: COL_SB_300,
        config.T1: COL_SB_600,
    }
    ts_names_in_config = {
        config.T3: "T3",
        config.T2: "T2",
        config.T1: "T1",
    }

    pred_dict = {
        "MTSCNN-LSTM": (_pct(pred_mts), COL_MTS, "--", 1.0),
    }
    for ts_val, display_name in ts_map.items():
        ab_key = f"k={kernel} T={ts_names_in_config[ts_val]}"
        if ab_key in ab_results:
            pred_ab, _ = ab_results[ab_key]
            pred_dict[display_name] = (_pct(pred_ab), ts_colors[ts_val], "--", 0.9)

    # Panel limits
    xlim_a = (0, N)
    # Panel (b): 3600–5000 min (as in paper)
    zoom_s = min(int(N * 0.45), 3600)
    zoom_e = min(zoom_s + 1400, N)
    xlim_b = (zoom_s, zoom_e)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.30)

    title_a = "(a) 144-hour range"
    title_b = "(b) 24-hour range"

    for ax, xlim, title in [(ax_a, xlim_a, title_a), (ax_b, xlim_b, title_b)]:
        i0, i1 = max(0, xlim[0]), min(N, xlim[1])
        t_sl = np.arange(i0, i1)

        ax.plot(t_sl, y_pct[i0:i1], color=COL_REAL, lw=1.2,
                label="Real SOC", zorder=10)

        for label, (pred_pct, color, ls, lw) in pred_dict.items():
            p   = np.asarray(pred_pct).ravel()
            n_u = min(len(t_sl), max(0, len(p) - i0))
            if n_u > 0:
                ax.plot(t_sl[:n_u], p[i0:i0+n_u],
                        color=color, lw=lw, ls=ls, label=label, zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(10, 70)
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel("SOC (%)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        if ax is ax_a:
            ax.legend(loc="upper left", fontsize=7.5,
                      handlelength=1.8, labelspacing=0.25)

    _save(fig, fname)

def plot_fig13_model_comparison(y_test, pred_mts, pred_tcn, pred_cnn):
    def _pct(a): return a * 100 if np.asarray(a).max() <= 1.5 else np.asarray(a)
    y_pct = _pct(y_test)
    N     = len(y_pct)

    pred_dict = {
        "MTS-CNN-LSTM": (_pct(pred_mts), COL_MTS,     "--", 1.0),
        "MTS-TCN":      (_pct(pred_tcn), COL_TCN,     "--", 0.95),
        "MTS-CNN":      (_pct(pred_cnn), COL_CNN_MTS, "--", 0.95),
    }

    xlim_a = (0, N)
    # Panel (b): 5000–6400 min as in paper
    zoom_s = min(int(N * 0.62), 5000)
    zoom_e = min(zoom_s + 1400, N)
    xlim_b = (zoom_s, zoom_e)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.30)

    for ax, xlim, title in [
        (ax_a, xlim_a, "(a) 144-hour range"),
        (ax_b, xlim_b, "(b) 24-hour range"),
    ]:
        i0, i1 = max(0, xlim[0]), min(N, xlim[1])
        t_sl = np.arange(i0, i1)

        ax.plot(t_sl, y_pct[i0:i1], color=COL_REAL, lw=1.2,
                label="Real SOC", zorder=10)

        for label, (pred_pct, color, ls, lw) in pred_dict.items():
            p   = np.asarray(pred_pct).ravel()
            n_u = min(len(t_sl), max(0, len(p) - i0))
            if n_u > 0:
                ax.plot(t_sl[:n_u], p[i0:i0+n_u],
                        color=color, lw=lw, ls=ls, label=label, zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(10, 70)
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel("SOC (%)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        if ax is ax_a:
            ax.legend(loc="upper left", fontsize=8,
                      handlelength=1.8, labelspacing=0.3)

    _save(fig, "fig13_realworld_comparison.png")

def plot_training_history(history: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], color=COL_MTS, label="Train MSE")
    ax.plot(history["val"],   color=COL_TCN, label="Val MSE", ls='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.set_yscale("log")
    _save(fig, "training_loss.png")
