""" UTILITY FUNCTIONS """

from defaults import *
from constants import *


# ---- AUTOCORRELATION TIME FUNCTION ----
#
# Real autocorrelation functions can have negative lobes, especially for oscillatory signals or finite data.
# On real data the autocorrelation time estimate might be slightly negative at some lags also due to noise. This can 
# cause the integrated autocorrelation time 'tau' to be underestimated or even negative, leading to invalid noise 
# STDs or negative variances! These negative values are fed into the GP's kernel matrix K_dd, which must be 
# positive-definite for the Cholesky decomposition to succeed. In practise we need to treat these negative 
# autocorrelation values carefully to avoid systematic errors in the GP's noise estimate.
#
# ----- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -----
# Important note: autocorrelation time estimate has quite some implications for the performance of GPs! 
# We need to resolve which approach is the closest to the true underlying autocorrelation time function.
# ----- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -----
#
# TBA (possibly): more sophisticated methods like FFT-based approaches (SDA: Spectral Density Approach).
#


# ** Simple Autocorrelation Time Estimator ** (DEPRECATED - crude heuristics)
# Coded by: Brennon, Martin and GPT-4
# 
#
# Different truncation methods (treatment of negative autocorrelation):
#
# (1) Clamp: the most crude heuristics we can apply. Artificially clamping the values could poison the uncertainty estimate
# with systematic errors. The thing is that so far it seems to yield the most monotonous RMSE convergence.
#
# (2) IMS and (3) IPS: initial monotone sequence (IMS) and Geyer's initial positive sequence (IPS) rule. These are standard 
# in MCMC literature to truncate the autocorrelation sum when it turns non-positive. Both give very similar results 
# and faster, but result in a non-monotonous RMSE convergence.
# 
# TBA (possibly): Bayesian (model-based) truncation.

def compute_autocorrelation_time(x, max_lag=None, method='clamp'):
    
    """
    Estimate integrated autocorrelation time from a 1D torch tensor.

    Args:
        x (torch.Tensor): 1D input tensor (time series).
        max_lag (int, optional): Maximum lag to compute autocorrelation. Defaults to min(N//2, 1000).
        method (str): One of 'ips', 'ims', 'clamp' to select truncation method.
            'ips'   : Initial Positive Sequence (stop at first negative autocorr)
            'ims'   : Initial Monotone Sequence (Geyer's method)
            'clamp' : Clamp negative autocorrelation values to zero (crude heuristics)
            
            !!! ... we could add some Bayesian (model-based) truncation method !!!

    Returns:
        float: Estimated integrated autocorrelation time.
    """
    
    # Zero-mean the input tensor
    x = x - x.mean()
    N = x.numel()
    max_lag = min(N // 2, 1000) if max_lag is None else max_lag

    # Variance
    var = x.var(unbiased=True)
    if var <= 0:
        # Degenerate: no fluctuations => zero autocorrelation time
        return 0.0

    # Allocate autocorrelation tensor
    ac = torch.zeros(max_lag)
    # Compute autocorrelation for each lag
    for k in range(max_lag):
        # x[0]*x[k], ..., x[N-k-1]*x[N-1]
        ac[k] = (x[:N-k] * x[k:]).mean() / var

    # Enforce exact rho(0)=1
    ac[0] = 1.0

    if method == 'ims':
        # Initial Monotone Sequence (IMS) truncation (Geyer's method) - most robust out of these 3
        tau = 1.0  # start with ac[0]
        prev_Gk = float('inf')

        for k in range(1, max_lag // 2):
            Gk = ac[2 * k - 1] + ac[2 * k]
            if Gk < 0 or Gk > prev_Gk:
                break
            tau += 2.0 * Gk
            prev_Gk = Gk

    elif method == 'ips':
        # Initial Positive Sequence (IPS) truncation - less robust, but "faster"
        tau = 1.0 # start with ac[0]
        for k in range(1, max_lag):
            if ac[k] < 0:
                break
            tau += 2.0 * ac[k]

    elif method == 'clamp':
        # Clamp negative autocorrelation values to zero (introduces large systematic error/bias?)
        ac_clamped = torch.clamp(ac, min=0.0)
        # Integrated autocorrelation time
        tau = 1.0 + 2.0 * ac_clamped[1:].sum()

    else:
        raise ValueError(f"Unknown method '{method}', choose from ['ips', 'ims', 'clamp'].")

    return tau.item()


# ** Bayesian Autocorrelation Time Function ** (currently in use)
# Coded by: Martin
#
# This one is efficient AS FUCK!!!
#
# Should be more robust and rigorous - especially the AR(p) process, and is way more efficient than the previous ones. Data driven.
# The only method that should be more rigorous is the Fourier-based spectral density method (also model-based) - could utilize FFT to make it efficient.
#
# Consider AR(p) instead of AR(1) if the problem is more complex. We could also use the full posterior distribution 
# of phi instead of just the mean, compute expected tau as E[tau] = E[(1+phi)/(1-phi)]. This may 
# require sampling (perhaps MCMC?) and analytic integration.
#
# Seems to also give better mean estimate with initially faster convergence as compared to the previous method (no matter the treatment used).
#

#
# NOTE: IMPORTANT CAVEATS TO THINK ABOUT.
#
# - Implicit noise variance: we use unit variance. If the true noise variance (sigma^2) is unknown, fully Bayesian treatment is needed (model the noise var).
#   Put a prior on sigma^2 and integrate it out, or jointly infer phi and sigma^2.
#
# - Posterior mean only: we use just the posterior mean as a point estimate of phi. More Bayesian would be to compute the whole posterior distribution 
#   of tau = (1+phi)/(1-phi), which is non-linear in phi. Report posterior mean and credibility interval. This has no simple closed form.
#   It could instead be done by sampling from the posterior distribution of phi and computing tau for each sample, or by numerical integration.
#
# - When phi -> 1: tau -> infinity (long correlation time limit). Small estimation errors in phi near 1 can lead to large swings in tau!
#
# - Discussion with Brennon: these are equilibrium sims so the autocorrelation time should be fine like this,
#   we don't need to worry about non-stationarity etc. We should have a constant noise along the chain.
#   TODO: verify this assumption (print out the noise along the chain, make a plot).

# -----------------------------------------------------------------------
# TODO: IMPLEMENT NOISE VARIANCE ESTIMATION (FULLY BAYESIAN TREATMENT)!!!
# -----------------------------------------------------------------------

def bayes_autocorrelation_time(x, prior_mean=0.0, prior_precision=1e-2, eps=1e-8):
    
    """
    Estimate integrated autocorrelation time assuming AR(1) dynamics with
    Bayesian posterior over the AR coefficient phi.

    Args:
        x (torch.Tensor): 1D time series.
        prior_mean (float): Prior mean of phi (default: 0.0).
        prior_precision (float): Prior precision (1/variance) of phi (default: 1e-2).
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Estimated integrated autocorrelation time.
    """

    x = x.flatten()
    x_centered = x - x.mean()
    x_prev = x_centered[:-1]
    x_curr = x_centered[1:]

    # Sufficient statistics
    sum_xx = torch.sum(x_prev * x_prev)
    sum_xy = torch.sum(x_prev * x_curr)

    # Posterior precision and mean
    post_precision = prior_precision + sum_xx
    post_mean = (prior_precision * prior_mean + sum_xy) / (post_precision + eps)

    # Soft clamping posterior mean to a physically valid AR(1) range (-1, 1)
    phi_post = torch.tanh(post_mean)

    # Estimate integrated autocorrelation time
    tau = (1 + phi_post) / (1 - phi_post)

    return tau.item()


# ---- SAMPLE TRUNCATION TEST SUMMARY ---- (So far performed on GPR(d) only!)
#
# Dataset: 'QKKYRAHARAGDQIAESLLNMS_human'.
#
# Convergence to nRMSE < 0.005:
# - Bayes: around 70 % of the trajectory samples 
# - IPS: around 70 % of the trajectory samples (same as Bayes)
# - IMS: around 50 - 60 % of the trajectory samples (a bit faster than Bayes)
# - Clamp: around 30 - 45 % of the trajectory samples (fastest so far) - but how big is the systematic error?
#
# ----- - - - - - - - - - - - - - - - -----


# Random indices
# Made by: Adam
def random_indices(N_max,N):
     full_idx = np.arange(N_max)
     r = np.sort(np.random.choice(full_idx, N, replace=False))
     return torch.tensor(r)


# ----- LOAD DATA UTILITIES -----

# ** Load Katka's Data Utility Functions ** (DEPRECATED - old data structure)
# Coded by: GPT-4

def load_pullx(file_path):
    time, position = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                time.append(float(parts[0]))
                position.append(float(parts[1]))
    return torch.tensor(time), torch.tensor(position)

def load_last_mdp_value(file_path):
    last_key, last_value = None, None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(';') and '=' in line:
                last_key, last_value = map(str.strip, line.split('=', 1))
    return last_key, last_value

def natural_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]

def extract_number(folder_name):
    match = re.search(r'd_([0-9]+\.[0-9]+)', folder_name)
    return float(match.group(1)) if match else None
