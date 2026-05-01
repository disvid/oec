# =============================================================================
# current_decoupler.py
#
# Implements the scenario-adaptive dynamic threshold current decoupling
# algorithm described in Section 3.1 of Dong et al. (2025).
#
# Algorithm
# ---------
# For each time step t:
#   1.  I_threshold = k * std(I[t-n : t])
#   2a. |I_t - I_stable| <= I_threshold
#         → peak-shaving region; update I_stable via weighted EMA
#   2b. |I_t - I_stable| >  I_threshold  AND  fluc_dur <= 3
#         → frequency-regulation spike;  ΔI = I_t - I_stable
#   2c. |I_t - I_stable| >  I_threshold  AND  fluc_dur >  3
#         → start of new peak-shaving operation;
#           I_stable = mean(I[t : t+3])
# =============================================================================

import numpy as np


class CurrentDecoupler:
    """
    Separates a raw battery current trace into:
        I_ps  – peak-shaving component  (slow, stable baseline)
        I_fr  – frequency-regulation component  (fast, high-frequency spikes)

    Parameters
    ----------
    window_n          : rolling std window length (samples = minutes)
    k                 : proportional coefficient for threshold
    lambda1           : EMA weight for previous stable value (λ1 = 0.7)
    lambda2           : EMA weight for current sample       (λ2 = 0.3)
    fluctuation_limit : max consecutive fluctuation samples before
                        reclassifying as new peak-shaving
    """

    def __init__(self, window_n=15, k=0.5,
                 lambda1=0.7, lambda2=0.3, fluctuation_limit=3):
        self.window_n          = window_n
        self.k                 = k
        self.lambda1           = lambda1
        self.lambda2           = lambda2
        self.fluctuation_limit = fluctuation_limit

    # ------------------------------------------------------------------
    def run(self, current: np.ndarray):
        """
        Parameters
        ----------
        current : 1-D np.ndarray, length N  (raw battery current in A)

        Returns
        -------
        I_ps : np.ndarray (N,)  – peak-shaving component
        I_fr : np.ndarray (N,)  – frequency-regulation component
        """
        N    = len(current)
        I_ps = np.zeros(N, dtype=np.float32)
        I_fr = np.zeros(N, dtype=np.float32)

        I_stable      = float(current[0])
        fluc_duration = 0          # consecutive steps in fluctuation mode

        for t in range(N):
            # ── Step 1: compute dynamic threshold ────────────────────────
            start      = max(0, t - self.window_n)
            window_std = float(np.std(current[start : t + 1]))
            threshold  = self.k * window_std

            delta = float(current[t]) - I_stable

            # ── Step 2: classify ─────────────────────────────────────────
            if abs(delta) <= threshold:
                # ─ Peak-shaving region ─
                # Update stable baseline with exponential moving average
                I_stable = self.lambda1 * I_stable + self.lambda2 * float(current[t])
                I_ps[t]  = I_stable
                I_fr[t]  = 0.0
                fluc_duration = 0

            else:
                fluc_duration += 1

                if fluc_duration <= self.fluctuation_limit:
                    # ─ Short fluctuation → frequency regulation ─
                    I_ps[t] = I_stable    # stable component unchanged
                    I_fr[t] = delta       # deviation is the freq-reg signal

                else:
                    # ─ Prolonged fluctuation → new peak-shaving operation ─
                    # Use mean of the next `fluctuation_limit` samples as new stable
                    end_idx  = min(t + self.fluctuation_limit, N)
                    I_stable = float(np.mean(current[t : end_idx]))
                    I_ps[t]  = I_stable
                    I_fr[t]  = 0.0
                    fluc_duration = 0

        return I_ps, I_fr

    # ------------------------------------------------------------------
    # Alternative "no decoupling" baseline: return raw current as PS, zeros as FR
    @staticmethod
    def no_decoupling(current: np.ndarray):
        return current.copy().astype(np.float32), np.zeros_like(current, dtype=np.float32)

    # Alternative "static threshold" baseline
    @staticmethod
    def static_threshold(current: np.ndarray, threshold: float = 5.0):
        I_ps = np.where(np.abs(current) <= threshold, current, 0.0).astype(np.float32)
        I_fr = (current - I_ps).astype(np.float32)
        return I_ps, I_fr