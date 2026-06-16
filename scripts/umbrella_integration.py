"""Umbrella Integration (UI) with block-averaging error estimation.

Implements the Kästner–Thiel Umbrella Integration method using a Gaussian
approximation of the biased distribution in each umbrella window, combined with
block-averaging for error estimation.

References:
    Kästner, J. & Thiel, W. (2006). Bridging the gap between thermodynamic
    integration and umbrella sampling provides a novel analysis method:
    "Umbrella integration". J. Chem. Phys. 123, 144104.

    Kästner, J. (2011). Umbrella sampling. WIREs Comput. Mol. Sci. 1, 932–942.

Units throughout:
    distances      : nm
    energies       : kJ/mol
    force constants: kJ/mol/nm²

Temperature convention: T = 303.15 K (matching the freeGP preprocessing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Physical constants (matching freeGP/preprocess.py) ─────────────────────
k_B: float = 8.3144621e-3   # kJ / (mol · K)
T: float    = 303.15         # K
kBT: float  = k_B * T       # kJ / mol


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class UIResult:
    """Result of an Umbrella Integration run on one set of windows.

    Attributes
    ----------
    x_grid : shape (M,)
        Evaluation grid [nm].
    pmf : shape (M,)
        PMF from the start of the grid (pmf[0] = 0) [kJ/mol].
    pmf_std : shape (M,)
        Block-average standard deviation of the PMF [kJ/mol].
    pmf_variance : shape (M,)
        Square of pmf_std [kJ/mol²].
    mean_force : shape (M,)
        Weighted mean force A'(q) [kJ/mol/nm].
    mean_force_std : shape (M,)
        Block-average standard deviation of the mean force [kJ/mol/nm].
    """

    x_grid: np.ndarray
    pmf: np.ndarray
    pmf_std: np.ndarray
    pmf_variance: np.ndarray
    mean_force: np.ndarray
    mean_force_std: np.ndarray


@dataclass
class UIAblationResult:
    """Aggregated result over multiple replicates for one ablation cell.

    Attributes mirror the freeGP HyperposteriorPredictiveSummary / ReferenceComparison
    decomposition so that equivalent metrics can be computed.
    """

    x_test: np.ndarray        # shape (M,)
    pmf_mean: np.ndarray      # mean PMF across replicates [kJ/mol], shape (M,)
    pmf_reps: np.ndarray      # individual replicate PMFs [kJ/mol], shape (n_reps, M)
    within_variance: np.ndarray   # mean block-avg variance across replicates [kJ/mol²]
    between_variance: np.ndarray  # variance of PMF means across replicates [kJ/mol²]
    total_variance: np.ndarray    # within + between [kJ/mol²]
    replicate_count: int


# ── Core mathematical primitives ─────────────────────────────────────────────

def _gaussian_pdf(x: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
    """Evaluate N(x; mu, sigma²) PDF."""
    return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2.0 * np.pi * sigma2)


def _mean_force_gaussian(
    x_grid: np.ndarray,
    mu: float,
    sigma2: float,
    x_eq: float,
    k: float,
) -> np.ndarray:
    """Per-window mean force A'_i(q) under a Gaussian approximation.

    Derivation:
        ρ_i^b(q)  ≈  N(q; μ_i, σ_i²)

        −(1/β) ∂ ln ρ_i^b / ∂q = kBT (q − μ_i) / σ_i²

        A'_i(q) = −(1/β) ∂ ln ρ_i^b / ∂q − k_i (q − x_i*)
                = kBT (q − μ_i) / σ_i²  −  k_i (q − x_i*)
    """
    return kBT * (x_grid - mu) / sigma2 - k * (x_grid - x_eq)


def _integrate_trapz(x_grid: np.ndarray, A_prime: np.ndarray) -> np.ndarray:
    """Integrate mean force A'(q) using the trapezoidal rule.

    Returns PMF with pmf[0] = 0.
    """
    pmf = np.zeros_like(A_prime)
    for j in range(1, len(x_grid)):
        dx = x_grid[j] - x_grid[j - 1]
        pmf[j] = pmf[j - 1] + 0.5 * dx * (A_prime[j - 1] + A_prime[j])
    return pmf


def _propagate_variance_trapz(
    x_grid: np.ndarray,
    A_prime_var: np.ndarray,
) -> np.ndarray:
    """Propagate mean-force variance through trapezoidal integration.

    For  A(x_j) = Σ_{k=0}^{j-1} (Δx/2)(A'_k + A'_{k+1})  the propagated
    variance (treating adjacent grid-point contributions as independent) is:

        Var[A(x_j)] ≈ Σ_{k=0}^{j-1} (Δx/2)² (Var[A'_k] + Var[A'_{k+1}])

    which accumulates as a running sum.
    """
    pmf_var = np.zeros_like(A_prime_var)
    for j in range(1, len(x_grid)):
        dx = x_grid[j] - x_grid[j - 1]
        coeff = (0.5 * dx) ** 2
        pmf_var[j] = pmf_var[j - 1] + coeff * (A_prime_var[j - 1] + A_prime_var[j])
    return pmf_var


# ── Block-averaging within a single window ───────────────────────────────────

def _block_average_window(
    positions: np.ndarray,
    x_grid: np.ndarray,
    x_eq: float,
    k: float,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Block-average the mean-force contribution of a single umbrella window.

    The trajectory is divided into ``n_blocks`` contiguous blocks of equal
    size.  For each block a Gaussian is fitted to the positions; the
    per-window mean force is evaluated from that Gaussian.  The block-average
    mean and squared standard error (SE²) are returned.

    Returns
    -------
    A_prime_mean : shape (M,)
        Block-average per-window mean force.
    A_prime_se2 : shape (M,)
        Squared block-average standard error of the per-window mean force.
    mu_full : float
        Sample mean of all positions (used for weighting).
    sigma2_full : float
        Sample variance of all positions (used for weighting).
    """
    n = len(positions)

    # Full-trajectory Gaussian fit (used for weight computation).
    mu_full = float(np.mean(positions))
    sigma2_full = max(float(np.var(positions, ddof=1)), 1e-10)

    block_size = n // n_blocks
    if block_size < 2:
        # Too few samples to form valid blocks: no error estimate.
        A_prime = _mean_force_gaussian(x_grid, mu_full, sigma2_full, x_eq, k)
        return A_prime, np.zeros_like(A_prime), mu_full, sigma2_full

    block_A_primes: list[np.ndarray] = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n
        block_pos = positions[start:end]
        if len(block_pos) < 2:
            continue
        mu_b = float(np.mean(block_pos))
        sigma2_b = max(float(np.var(block_pos, ddof=1)), 1e-10)
        block_A_primes.append(_mean_force_gaussian(x_grid, mu_b, sigma2_b, x_eq, k))

    if not block_A_primes:
        A_prime = _mean_force_gaussian(x_grid, mu_full, sigma2_full, x_eq, k)
        return A_prime, np.zeros_like(A_prime), mu_full, sigma2_full

    arr = np.array(block_A_primes)       # shape (n_valid_blocks, M)
    n_valid = arr.shape[0]
    A_prime_mean = arr.mean(axis=0)
    # SE² = sample_var / n_blocks
    A_prime_se2 = arr.var(axis=0, ddof=1) / n_valid

    return A_prime_mean, A_prime_se2, mu_full, sigma2_full


# ── Main UI routine ──────────────────────────────────────────────────────────

def umbrella_integration(
    windows_data: list[tuple[np.ndarray, float, float]],
    x_grid: np.ndarray,
    n_blocks: int = 10,
) -> UIResult:
    """Run Umbrella Integration with block-averaging error estimation.

    Parameters
    ----------
    windows_data : list of (positions, x_eq, k)
        Each element represents one umbrella window:
        - ``positions``: 1-D array of trajectory positions after equilibration
          and trajectory-fraction truncation [nm].
        - ``x_eq``: harmonic restraint equilibrium position [nm].
        - ``k``: harmonic force constant [kJ/mol/nm²].
    x_grid : 1-D array
        Grid at which to evaluate the PMF [nm].
    n_blocks : int
        Number of contiguous blocks for block averaging.

    Returns
    -------
    UIResult
        PMF, PMF standard deviation, mean force, and mean-force standard
        deviation at every grid point.
    """
    if not windows_data:
        raise ValueError("windows_data must not be empty.")

    # ── Step 1: per-window block-averaged mean force + full-traj Gaussian fit ──
    n_win = len(windows_data)
    A_prime_means = np.zeros((n_win, len(x_grid)))
    A_prime_se2s  = np.zeros((n_win, len(x_grid)))
    mu_fulls      = np.zeros(n_win)
    sigma2_fulls  = np.zeros(n_win)
    n_samples     = np.zeros(n_win, dtype=int)

    for i, (positions, x_eq, k) in enumerate(windows_data):
        positions = np.asarray(positions, dtype=float)
        A_mean, A_se2, mu_f, s2_f = _block_average_window(
            positions, x_grid, x_eq, k, n_blocks=n_blocks
        )
        A_prime_means[i] = A_mean
        A_prime_se2s[i]  = A_se2
        mu_fulls[i]      = mu_f
        sigma2_fulls[i]  = s2_f
        n_samples[i]     = len(positions)

    # ── Step 2: weights from full-trajectory Gaussian fits ─────────────────
    # w_i(q) = N_i · N(q; μ_i, σ_i²)
    weights = np.zeros((n_win, len(x_grid)))
    for i in range(n_win):
        pdf = _gaussian_pdf(x_grid, mu_fulls[i], sigma2_fulls[i])
        weights[i] = n_samples[i] * pdf

    total_weight = weights.sum(axis=0)
    valid = total_weight > 0.0
    norm_weights = np.zeros_like(weights)
    norm_weights[:, valid] = weights[:, valid] / total_weight[valid]

    # ── Step 3: weighted mean force and its variance ────────────────────────
    # A'(q) = Σ_i ñ_i(q) A'_i(q)
    # Var[A'(q)] ≈ Σ_i ñ_i(q)² SE²_i(q)   (independence across windows)
    A_prime     = (norm_weights * A_prime_means).sum(axis=0)
    A_prime_var = (norm_weights ** 2 * A_prime_se2s).sum(axis=0)
    A_prime_std = np.sqrt(np.clip(A_prime_var, 0.0, None))

    # ── Step 4: integrate and propagate errors ──────────────────────────────
    pmf     = _integrate_trapz(x_grid, A_prime)
    pmf_var = _propagate_variance_trapz(x_grid, A_prime_var)
    pmf_std = np.sqrt(np.clip(pmf_var, 0.0, None))

    return UIResult(
        x_grid=x_grid.copy(),
        pmf=pmf,
        pmf_std=pmf_std,
        pmf_variance=pmf_var,
        mean_force=A_prime,
        mean_force_std=A_prime_std,
    )


# ── Replicate-level aggregation ──────────────────────────────────────────────

def aggregate_ui_replicates(
    results: list[UIResult],
) -> UIAblationResult:
    """Aggregate UI results across replicates.

    Decomposes total uncertainty into:

    * **within variance** – average block-averaging variance across replicates
      (captures trajectory-sampling noise within a single selection).
    * **between variance** – variance of mean PMFs across replicates
      (captures selection noise: which windows / which trajectory segment).
    * **total variance** – within + between.

    This mirrors the GP code's ``_aggregate_predictive_summaries`` decomposition
    (within = average conditional covariance; between = sample covariance of
    conditional means).

    Parameters
    ----------
    results : list of UIResult
        One entry per replicate, all sharing the same ``x_grid``.

    Returns
    -------
    UIAblationResult
    """
    pmf_reps       = np.array([r.pmf for r in results])          # (n_reps, M)
    within_var_reps = np.array([r.pmf_variance for r in results]) # (n_reps, M)

    pmf_mean     = pmf_reps.mean(axis=0)                          # (M,)
    within_var   = within_var_reps.mean(axis=0)                   # (M,)
    centered     = pmf_reps - pmf_mean[np.newaxis, :]
    between_var  = (centered ** 2).mean(axis=0)                   # (M,)
    total_var    = within_var + between_var

    return UIAblationResult(
        x_test=results[0].x_grid.copy(),
        pmf_mean=pmf_mean,
        pmf_reps=pmf_reps,
        within_variance=within_var,
        between_variance=between_var,
        total_variance=total_var,
        replicate_count=len(results),
    )


# ── Metric helpers ───────────────────────────────────────────────────────────

def shift_pmf(pmf: np.ndarray, alignment: str = "max") -> np.ndarray:
    """Shift PMF so that its maximum (or minimum) is zero."""
    if alignment == "max":
        return pmf - pmf.max()
    if alignment == "min":
        return pmf - pmf.min()
    raise ValueError(f"alignment must be 'max' or 'min', got {alignment!r}")


def rmse_vs_reference(
    x_test: np.ndarray,
    pmf_pred: np.ndarray,
    x_ref: np.ndarray,
    pmf_ref: np.ndarray,
    alignment: str = "max",
) -> float:
    """RMSE of the shifted predicted PMF vs an interpolated reference curve.

    Both curves are independently aligned before comparison.

    Parameters
    ----------
    x_test : shape (M,) — test grid
    pmf_pred : shape (M,) — predicted PMF
    x_ref : shape (N,) — reference grid
    pmf_ref : shape (N,) — reference PMF
    alignment : 'max' or 'min'
    """
    ref_at_test = np.interp(x_test, x_ref, shift_pmf(pmf_ref, alignment))
    pred_shifted = shift_pmf(pmf_pred, alignment)
    return float(np.sqrt(np.mean((pred_shifted - ref_at_test) ** 2)))
