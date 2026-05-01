# =============================================================================
# model.py  –  MTS-CNN-LSTM + ablation models
#
# Key fix: SingleBranchModel no longer ignores x2/x3.
# The ablation runner passes the RIGHT time-scale window as x1,
# and SingleBranchModel uses only x1.
# =============================================================================

import torch
import torch.nn as nn
import config


class DualKernelCNN(nn.Module):
    """
    Two parallel Conv1d paths (kernel k1 and k2) concatenated on channels.
    Input : (B, T, F)  →  Output : (B, T', 2*out_ch)
    """
    def __init__(self, in_ch, out_ch,
                 k1=config.KERNEL1, k2=config.KERNEL2):
        super().__init__()
        pad2 = k2 // 2
        self.path1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k1, padding=0),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
        )
        self.path2 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k2, padding=pad2),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
        )
        self.out_ch = 2 * out_ch

    def forward(self, x):
        x  = x.permute(0, 2, 1)          # (B,F,T)
        o1 = self.path1(x)
        o2 = self.path2(x)
        L  = min(o1.size(2), o2.size(2))
        return torch.cat([o1[:,:,:L], o2[:,:,:L]], dim=1).permute(0,2,1)


class SubCNNLSTM(nn.Module):
    """One branch: DualKernelCNN → 2-layer LSTM → last hidden state."""
    def __init__(self, in_feat=3, cnn_out=config.CNN_OUT,
                 hidden=config.HIDDEN_SIZE,
                 n_layers=config.NUM_LSTM_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()
        self.cnn  = DualKernelCNN(in_feat, cnn_out)
        self.lstm = nn.LSTM(
            input_size  = self.cnn.out_ch,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.cnn(x)
        _, (h, _) = self.lstm(z)
        return self.drop(h[-1])   # (B, hidden)


class MTS_CNN_LSTM(nn.Module):
    """
    Full Multi-Time-Scale CNN-LSTM.
    Inputs: x1(B,T1,3), x2(B,T2,3), x3(B,T3,3)  →  SOC (B,1)
    """
    def __init__(self, in_feat=3, cnn_out=config.CNN_OUT,
                 hidden=config.HIDDEN_SIZE,
                 n_layers=config.NUM_LSTM_LAYERS,
                 dropout=config.DROPOUT,
                 attn_heads=config.ATTN_HEADS,
                 attn_embed=config.ATTN_EMBED):
        super().__init__()
        self.b1 = SubCNNLSTM(in_feat, cnn_out, hidden, n_layers, dropout)
        self.b2 = SubCNNLSTM(in_feat, cnn_out, hidden, n_layers, dropout)
        self.b3 = SubCNNLSTM(in_feat, cnn_out, hidden, n_layers, dropout)

        self.proj = nn.Linear(hidden * 3, attn_embed)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_embed, num_heads=attn_heads,
            dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(attn_embed, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3):
        fused = torch.cat([self.b1(x1), self.b2(x2), self.b3(x3)], dim=1)
        proj  = self.proj(fused).unsqueeze(1)
        ao, _ = self.attn(proj, proj, proj)
        return self.head(ao.squeeze(1))


# ─────────────────────────────────────────────────────────────────────────────
# SingleBranchModel
# -----------------
# IMPORTANT: only uses x1. The ablation runner is responsible for passing
# the correct time-scale window as x1 for each branch experiment.
# ─────────────────────────────────────────────────────────────────────────────
class SingleBranchModel(nn.Module):
    def __init__(self, in_feat=3, cnn_out=config.CNN_OUT,
                 hidden=config.HIDDEN_SIZE,
                 n_layers=config.NUM_LSTM_LAYERS,
                 dropout=config.DROPOUT,
                 kernel1=config.KERNEL1, kernel2=config.KERNEL2):
        super().__init__()
        self.cnn  = DualKernelCNN(in_feat, cnn_out, k1=kernel1, k2=kernel2)
        self.lstm = nn.LSTM(
            input_size  = self.cnn.out_ch,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x1, *_):
        # Always uses x1 only — caller sets the right window length
        z = self.cnn(x1)
        _, (h, _) = self.lstm(z)
        return self.head(self.drop(h[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# TCN baseline
# ─────────────────────────────────────────────────────────────────────────────
class _TCNBlock(nn.Module):
    def __init__(self, ch, k=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)),
            nn.ReLU(), nn.Dropout(dropout),
            nn.utils.weight_norm(
                nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.trim = pad   # causal: remove right-side padding

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)] + x


class MTS_TCN(nn.Module):
    """Multi-time-scale TCN baseline (same 3-branch structure, TCN instead of LSTM)."""
    def __init__(self, in_feat=3, ch=64, dropout=config.DROPOUT):
        super().__init__()
        def _branch():
            return nn.Sequential(
                nn.Conv1d(in_feat, ch, 1),
                _TCNBlock(ch, dilation=1, dropout=dropout),
                _TCNBlock(ch, dilation=2, dropout=dropout),
                _TCNBlock(ch, dilation=4, dropout=dropout),
            )
        self.b1 = _branch(); self.b2 = _branch(); self.b3 = _branch()
        self.head = nn.Sequential(
            nn.Linear(ch*3, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def _enc(self, branch, x):
        return branch(x.permute(0,2,1)).mean(dim=2)

    def forward(self, x1, x2, x3):
        h = torch.cat([self._enc(self.b1,x1),
                       self._enc(self.b2,x2),
                       self._enc(self.b3,x3)], dim=1)
        return self.head(h)


class MTS_CNN(nn.Module):
    """Multi-time-scale pure CNN baseline (no LSTM)."""
    def __init__(self, in_feat=3, ch=64, dropout=config.DROPOUT):
        super().__init__()
        def _branch():
            return nn.Sequential(
                nn.Conv1d(in_feat, ch, kernel_size=1), nn.ReLU(),
                nn.Conv1d(ch, ch, kernel_size=7, padding=3), nn.ReLU(),
                nn.Conv1d(ch, ch, kernel_size=7, padding=3), nn.ReLU(),
            )
        self.b1 = _branch(); self.b2 = _branch(); self.b3 = _branch()
        self.head = nn.Sequential(
            nn.Linear(ch*3, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def _enc(self, branch, x):
        return branch(x.permute(0,2,1)).mean(dim=2)

    def forward(self, x1, x2, x3):
        h = torch.cat([self._enc(self.b1,x1),
                       self._enc(self.b2,x2),
                       self._enc(self.b3,x3)], dim=1)
        return self.head(h)