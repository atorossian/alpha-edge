# hmm_engine.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


_EPS = 1e-12


def _logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """Stable logsumexp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + _EPS)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    s = np.maximum(s, _EPS)
    return mat / s


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim != 2:
        raise ValueError("Observations X must be 1D or 2D array")
    return X


def _log_gaussian_diag(X: np.ndarray, means: np.ndarray, vars_diag: np.ndarray) -> np.ndarray:
    """
    Log N(x | mu_k, diag(var_k)) for each k.
    X: (T,D)
    means: (K,D)
    vars_diag: (K,D) positive
    returns logB: (T,K)
    """
    T, D = X.shape
    K = means.shape[0]
    vars_diag = np.maximum(vars_diag, 1e-8)

    # log det diag cov
    log_det = np.sum(np.log(vars_diag), axis=1)  # (K,)
    # quadratic term
    # (T,K,D) = (T,1,D) - (1,K,D)
    diff = X[:, None, :] - means[None, :, :]
    quad = np.sum((diff * diff) / vars_diag[None, :, :], axis=2)  # (T,K)

    log_norm = -0.5 * (D * np.log(2.0 * np.pi) + log_det)  # (K,)
    return log_norm[None, :] - 0.5 * quad


@dataclass(frozen=True)
class HMMParams:
    pi: np.ndarray        # (K,)
    A: np.ndarray         # (K,K)
    means: np.ndarray     # (K,D)
    vars: np.ndarray      # (K,D) diagonal variances


@dataclass
class HMMFitResult:
    params: HMMParams
    loglik: float
    n_iter: int
    converged: bool


class GaussianHMM:
    """
    Gaussian HMM with diagonal covariance emissions.
    - EM training (Baum-Welch)
    - log-space forward/backward for stability
    """

    def __init__(
        self,
        n_states: int,
        n_dim: int = 1,
        *,
        seed: Optional[int] = None,
        min_var: float = 1e-6,
    ):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if n_dim < 1:
            raise ValueError("n_dim must be >= 1")
        self.K = int(n_states)
        self.D = int(n_dim)
        self.rng = np.random.default_rng(seed)
        self.min_var = float(min_var)

        self.params: Optional[HMMParams] = None

    # ---------- initialization ----------

    def _init_params(self, X: np.ndarray) -> HMMParams:
        T, D = X.shape
        K = self.K

        # pi uniform
        pi = np.full(K, 1.0 / K, dtype=np.float64)

        # A with "sticky" diagonal by default (helps regimes persist)
        A = np.full((K, K), 1.0 / K, dtype=np.float64)
        A = 0.05 * A + 0.95 * np.eye(K)
        A = _normalize_rows(A)

        # means from random samples in X
        idx = self.rng.choice(T, size=K, replace=False) if T >= K else self.rng.integers(0, T, size=K)
        means = X[idx].copy()

        # variances from global var
        gvar = np.var(X, axis=0, ddof=1)
        gvar = np.maximum(gvar, self.min_var)
        vars_diag = np.tile(gvar[None, :], (K, 1))

        return HMMParams(pi=pi, A=A, means=means, vars=vars_diag)

    # ---------- core inference ----------

    def _forward_backward(
        self,
        X: np.ndarray,
        params: HMMParams,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns:
          log_alpha: (T,K)
          log_beta:  (T,K)
          gamma:     (T,K)  posterior state probs (smoothed)
          loglik:    float
        """
        pi, A, means, vars_diag = params.pi, params.A, params.means, params.vars
        K = self.K
        T = X.shape[0]

        log_pi = np.log(np.maximum(pi, _EPS))
        log_A = np.log(np.maximum(A, _EPS))
        log_B = _log_gaussian_diag(X, means, vars_diag)  # (T,K)

        log_alpha = np.empty((T, K), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            # log_alpha[t,k] = log_B[t,k] + logsumexp_j (log_alpha[t-1,j] + log_A[j,k])
            tmp = log_alpha[t - 1][:, None] + log_A  # (K,K)
            log_alpha[t] = log_B[t] + _logsumexp(tmp, axis=0)

        loglik = float(_logsumexp(log_alpha[-1], axis=0))

        log_beta = np.empty((T, K), dtype=np.float64)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            # log_beta[t,j] = logsumexp_k (log_A[j,k] + log_B[t+1,k] + log_beta[t+1,k])
            tmp = log_A + (log_B[t + 1] + log_beta[t + 1])[None, :]  # (K,K)
            log_beta[t] = _logsumexp(tmp, axis=1)

        # gamma[t,k] ∝ exp(log_alpha[t,k] + log_beta[t,k])
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        return log_alpha, log_beta, gamma, loglik

    def _expected_transitions(
        self,
        X: np.ndarray,
        params: HMMParams,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
    ) -> np.ndarray:
        """
        xi[t,i,j] = P(S_t=i, S_{t+1}=j | X) for t=0..T-2
        returns xi_sum: (K,K) sum_t xi[t]
        """
        pi, A, means, vars_diag = params.pi, params.A, params.means, params.vars
        K = self.K
        T = X.shape[0]

        log_A = np.log(np.maximum(A, _EPS))
        log_B = _log_gaussian_diag(X, means, vars_diag)  # (T,K)

        xi_sum = np.zeros((K, K), dtype=np.float64)

        for t in range(T - 1):
            # log_xi[i,j] = log_alpha[t,i] + log_A[i,j] + log_B[t+1,j] + log_beta[t+1,j]
            log_xi = (
                log_alpha[t][:, None]
                + log_A
                + log_B[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            log_xi -= _logsumexp(log_xi, axis=None, keepdims=True)
            xi = np.exp(log_xi)
            xi_sum += xi

        return xi_sum

    # ---------- training ----------

    def fit(
        self,
        X: np.ndarray,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        smoothing: float = 1e-3,
        verbose: bool = False,
    ) -> HMMFitResult:
        """
        EM / Baum-Welch:
        - X: (T,) or (T,D)
        - smoothing: Dirichlet-like pseudocount for transitions
        """
        X = _as_2d(X)
        T, D = X.shape
        if D != self.D:
            raise ValueError(f"Model n_dim={self.D} but X has D={D}")

        params = self._init_params(X)
        prev_ll = -np.inf
        converged = False

        for it in range(1, max_iter + 1):
            log_alpha, log_beta, gamma, ll = self._forward_backward(X, params)
            xi_sum = self._expected_transitions(X, params, log_alpha, log_beta)

            # M-step updates
            # pi
            pi_new = np.maximum(gamma[0], _EPS)
            pi_new = pi_new / pi_new.sum()

            # A
            A_new = xi_sum + smoothing
            A_new = _normalize_rows(A_new)

            # means
            gamma_sum = np.maximum(gamma.sum(axis=0), _EPS)  # (K,)
            means_new = (gamma.T @ X) / gamma_sum[:, None]   # (K,D)

            # vars (diag)
            vars_new = np.empty((self.K, self.D), dtype=np.float64)
            for k in range(self.K):
                diff = X - means_new[k]
                vars_k = (gamma[:, k][:, None] * (diff * diff)).sum(axis=0) / gamma_sum[k]
                vars_new[k] = np.maximum(vars_k, self.min_var)

            params = HMMParams(pi=pi_new, A=A_new, means=means_new, vars=vars_new)

            if verbose:
                print(f"[HMM] iter={it} loglik={ll:.6f} delta={ll - prev_ll:.6f}")

            if np.isfinite(prev_ll) and (ll - prev_ll) < tol:
                converged = True
                prev_ll = ll
                break

            prev_ll = ll

        self.params = params
        return HMMFitResult(params=params, loglik=float(prev_ll), n_iter=it, converged=converged)

    # ---------- outputs you’ll use in production ----------

    def smooth_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed posterior: P(S_t=k | X_1:T) for each t. Shape (T,K)."""
        if self.params is None:
            raise RuntimeError("Model not fit yet")
        X = _as_2d(X)
        _, _, gamma, _ = self._forward_backward(X, self.params)
        return gamma

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Filtered posterior: P(S_t=k | X_1:t) for each t.
        This is what you want for 'today' decisions.
        Shape (T,K).
        """
        if self.params is None:
            raise RuntimeError("Model not fit yet")
        X = _as_2d(X)

        pi, A, means, vars_diag = self.params.pi, self.params.A, self.params.means, self.params.vars
        K = self.K
        T = X.shape[0]

        log_pi = np.log(np.maximum(pi, _EPS))
        log_A = np.log(np.maximum(A, _EPS))
        log_B = _log_gaussian_diag(X, means, vars_diag)

        log_alpha = np.empty((T, K), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        # filtered probs at time t ∝ exp(log_alpha[t])
        filt = np.empty((T, K), dtype=np.float64)

        lg = log_alpha[0] - _logsumexp(log_alpha[0], axis=0)
        filt[0] = np.exp(lg)

        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = log_B[t] + _logsumexp(tmp, axis=0)
            lg = log_alpha[t] - _logsumexp(log_alpha[t], axis=0)
            filt[t] = np.exp(lg)

        return filt

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely state path (hard labels). Shape (T,)."""
        if self.params is None:
            raise RuntimeError("Model not fit yet")
        X = _as_2d(X)

        pi, A, means, vars_diag = self.params.pi, self.params.A, self.params.means, self.params.vars
        K = self.K
        T = X.shape[0]

        log_pi = np.log(np.maximum(pi, _EPS))
        log_A = np.log(np.maximum(A, _EPS))
        log_B = _log_gaussian_diag(X, means, vars_diag)

        dp = np.empty((T, K), dtype=np.float64)
        ptr = np.empty((T, K), dtype=np.int32)

        dp[0] = log_pi + log_B[0]
        ptr[0] = -1

        for t in range(1, T):
            scores = dp[t - 1][:, None] + log_A  # (K,K)
            ptr[t] = np.argmax(scores, axis=0)
            dp[t] = log_B[t] + np.max(scores, axis=0)

        path = np.empty(T, dtype=np.int32)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]
        return path

    def next_state_proba(self, last_filtered: np.ndarray) -> np.ndarray:
        """Given P(S_t|data), return predictive P(S_{t+1}|data) = p_t A."""
        if self.params is None:
            raise RuntimeError("Model not fit yet")
        p = np.asarray(last_filtered, dtype=np.float64)
        p = p / np.maximum(p.sum(), _EPS)
        return p @ self.params.A


def select_regime(
    p: np.ndarray,
    *,
    commit_threshold: float = 0.65,
) -> int | None:
    """
    Return regime index if max prob >= threshold, else None (mixed/neutral).
    """
    p = np.asarray(p, dtype=np.float64)
    k = int(np.argmax(p))
    if float(p[k]) >= commit_threshold:
        return k
    return None

# --- NEW: regime labels + mapping helpers ---



REGIME_LABELS_4 = [
    "CALM_BULL",    # +drift, low vol
    "CHOPPY_BULL",  # +drift, higher vol
    "CHOPPY_BEAR",  # -/flat drift, higher vol
    "STRESS_BEAR",  # -/flat drift, highest vol
]

@dataclass(frozen=True)
class StateDiagnostics:
    # per hidden-state stats computed from data + posteriors
    drift: float          # mean return in state
    vol: float            # std of return in state
    neg_rate: float       # fraction of negative return days in state
    weight: float         # average posterior mass for this state (how often it appears)


def _safe_weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    wsum = float(np.sum(w))
    if wsum <= _EPS:
        return 0.0
    return float(np.sum(w * x) / wsum)


def _safe_weighted_var(x: np.ndarray, w: np.ndarray, mean: float) -> float:
    wsum = float(np.sum(w))
    if wsum <= _EPS:
        return 0.0
    return float(np.sum(w * (x - mean) ** 2) / wsum)


def compute_state_diagnostics(
    r: np.ndarray,
    gamma: np.ndarray,
) -> List[StateDiagnostics]:
    """
    r: (T,) portfolio returns aligned with gamma
    gamma: (T,K) posterior probs (filtered or smoothed)
    Returns list length K.
    """
    r = np.asarray(r, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    T, K = gamma.shape
    if r.shape[0] != T:
        raise ValueError("r and gamma must be aligned in time")

    out: List[StateDiagnostics] = []
    for k in range(K):
        w = gamma[:, k]
        mu = _safe_weighted_mean(r, w)
        var = _safe_weighted_var(r, w, mu)
        vol = float(np.sqrt(max(var, 0.0)))
        neg_rate = _safe_weighted_mean((r < 0).astype(np.float64), w)
        weight = float(np.mean(w))
        out.append(StateDiagnostics(drift=mu, vol=vol, neg_rate=neg_rate, weight=weight))
    return out


def label_states_4(
    diags: List[StateDiagnostics],
) -> Dict[int, str]:
    """
    Map hidden state index -> human label (4 regimes).
    Robust to arbitrary HMM state ordering.
    """
    K = len(diags)
    if K != 4:
        raise ValueError("label_states_4 expects exactly 4 states")

    vols = np.array([d.vol for d in diags], dtype=np.float64)
    drifts = np.array([d.drift for d in diags], dtype=np.float64)

    # 1) STRESS_BEAR = highest vol (usually crisis)
    stress_state = int(np.argmax(vols))

    remaining = [i for i in range(4) if i != stress_state]

    # 2) split remaining into bull candidates (positive drift) vs others
    bull = [i for i in remaining if drifts[i] > 0.0]
    nonbull = [i for i in remaining if i not in bull]

    # If HMM didn't produce two positive-drift states, fallback:
    # pick top-2 drifts as "bull side"
    if len(bull) < 2:
        order = sorted(remaining, key=lambda i: drifts[i], reverse=True)
        bull = order[:2]
        nonbull = [i for i in remaining if i not in bull]

    # 3) CALM_BULL / CHOPPY_BULL by vol within bull
    bull_sorted = sorted(bull, key=lambda i: vols[i])  # low vol first
    calm_bull = bull_sorted[0]
    choppy_bull = bull_sorted[1] if len(bull_sorted) > 1 else bull_sorted[0]

    # 4) CHOPPY_BEAR is the highest vol among the remaining nonbull states
    # (the last remaining state becomes CHOPPY_BEAR)
    if len(nonbull) == 0:
        # pathological case: all remaining are bullish; pick lowest drift remaining as choppy_bear
        candidates = [i for i in remaining if i not in (calm_bull, choppy_bull)]
        choppy_bear = candidates[0] if candidates else calm_bull
    else:
        choppy_bear = int(max(nonbull, key=lambda i: vols[i]))

    mapping = {
        int(calm_bull): "CALM_BULL",
        int(choppy_bull): "CHOPPY_BULL",
        int(choppy_bear): "CHOPPY_BEAR",
        int(stress_state): "STRESS_BEAR",
    }
    return mapping


def regime_probs_from_state_probs(
    p_state: np.ndarray,
    mapping: Dict[int, str],
) -> Dict[str, float]:
    """
    Convert p(S_t=k|data) to p(label|data).
    """
    p_state = np.asarray(p_state, dtype=np.float64)
    out = {lab: 0.0 for lab in REGIME_LABELS_4}
    for k, lab in mapping.items():
        out[lab] += float(p_state[int(k)])
    return out


def select_regime_label(
    p_label: Dict[str, float],
    *,
    commit_threshold: float = 0.65,
) -> Optional[str]:
    """
    Commit to label if max prob >= threshold else None.
    """
    best = max(p_label.items(), key=lambda kv: kv[1])
    if float(best[1]) >= commit_threshold:
        return str(best[0])
    return None
