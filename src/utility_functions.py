""" UTILITY FUNCTIONS """

from defaults import *
from constants import *


# ---- AUTOCORRELATION TIME FUNCTION ----

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


# Random indices
# Made by: Adam
def random_indices(N_max,N):
     full_idx = np.arange(N_max)
     r = np.sort(np.random.choice(full_idx, N, replace=False))
     return torch.tensor(r)


