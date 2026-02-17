""" ALL THINGS GAUSSIAN PROCESS (GP) """

from defaults import *
from constants import *


# ----- BUILD DISTANCE MATRICES -----
def pairwise_differences(x1, x2):
    # x1: (n,) or (n,1), x2: (m,) or (m,1)
    X1 = x1.unsqueeze(1)
    X2 = x2.unsqueeze(0)
    Xd = X1 - X2
    Xdd = Xd**2
    return Xd, Xdd


# ----- STATIONARY KERNEL FUNCTIONS -----

# Squared Exponential (SE) Kernel (so-called RBF)
def se_kernel(Xdd, ell, w):
    return w**2 * torch.exp(- Xdd / (2 * ell**2))

# First Derivative of SE Kernel
def fdse_kernel(Xd, Xdd, ell, w):
    return w**2 / ell**2 * torch.exp(-Xdd / (2 * ell**2)) * (Xd)

# Second Derivative of SE Kernel
def ddse_kernel(Xd, Xdd, ell, w):
    exp_term = torch.exp(-Xdd / (2 * ell**2))
    return w**2 / ell**4 * exp_term * (ell**2 - Xdd)


# Matern Kernel (ν=5/2) - WORKS HORRIBLY HERE (too noisy)
def matern_kernel(Xdd, ell, w):
     eps = 1e-12
     r = torch.sqrt(torch.clamp(Xdd, min=0.0) + eps)
     sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=r.dtype, device=r.device))
     z = sqrt5 * r / ell
     return w**2 * (1.0 + z + z**2 / 3.0) * torch.exp(-z)

def fdmatern_kernel(Xd, Xdd, ell, w):
     eps = 1e-12
     r = torch.sqrt(torch.clamp(Xdd, min=0.0) + eps)
     sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=r.dtype, device=r.device))
     z = sqrt5 * r / ell
     dk_dr = (-w**2 * (5.0 / (3.0 * ell**2)) * r * (1.0 + z) * torch.exp(-z))
     # Safe Xd/r: define it as 0 on the diagonal (r ~ 0)
     Xd_over_r = torch.where(r > 0, Xd / r, torch.zeros_like(Xd))
     return Xd_over_r * dk_dr

def ddmatern_kernel(Xd, Xdd, ell, w):
     eps = 1e-12
     r = torch.sqrt(torch.clamp(Xdd, min=0.0) + eps)
     sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=r.dtype, device=r.device))
     z = sqrt5 * r / ell
     return -(w**2 * (5.0 / (3.0 * ell**2)) * (z**2 - z - 1.0) * torch.exp(-z))


# Rational Quadratic Kernel - DOESN'T WORK WELL EITHER
def rq_kernel(Xdd, ell, w):
     # Rational Quadratic kernel
     alpha = 5.0  # you can change this, but keep fixed unless you re-tune everything
     A = 1.0 + Xdd / (2.0 * alpha * ell**2)
     return w**2 * A**(-alpha)

def fdrq_kernel(Xd, Xdd, ell, w):
     # ∂k/∂x
     alpha = 5.0
     A = 1.0 + Xdd / (2.0 * alpha * ell**2)
     # ∂k/∂x = -w^2/ell^2 * (x-x') * A^(-alpha-1)
     return -(w**2 / ell**2) * Xd * A**(-alpha - 1.0)

def ddrq_kernel(Xd, Xdd, ell, w):
     # ∂²k/∂x∂x'  (force–force mixed derivative, PD-safe)
     alpha = 5.0
     A = 1.0 + Xdd / (2.0 * alpha * ell**2)

     term1 = (w**2 / ell**2) * A**(-alpha - 1.0)
     term2 = (w**2 * (alpha + 1.0) / (alpha * ell**4)) * Xdd * A**(-alpha - 2.0)

     return term1 - term2


# ----- NON-STATIONARY KERNEL FUNCTIONS -----

# Gibbs kernel with error function w decay
# ------------------------------------------------------------
# width function definition (complimentary Gaussian cumulative distribution function)
# ------------------------------------------------------------
def width(r, s, u, w):
    #return 0.5 * s * (1.0 + torch.erf(-(r - u) / w)) # Error function decay
    return s * (1 + torch.tanh(-(r - u) / w))

def width_deriv(r, s, u, w):
    # d/dr [0.5*s*(1+erf(-(r-u)/w))]
    #return -(s / (torch.sqrt(torch.tensor(torch.pi)) * w)) * torch.exp(-((r - u) / w) ** 2)
    return - (s / w) * (1 - torch.tanh(-(r - u) / w)**2)

# ------------------------------------------------------------
# Kernel: K(r, r') = sigma(r) sigma(r') * exp(-(r-r')^2 / (2 ell^2))
# Inputs:
#   Xd  = r - r'
#   Xdd = (r - r')^2
#   r, rp are the two argument locations (same broadcastable shape as Xd)
# ------------------------------------------------------------
def gibbs_kernel(Xdd, ell, s, u, w, r, rp):
    sig  = width(r,  s, u, w)
    sigp = width(rp, s, u, w)
    return sig * sigp * torch.exp(-Xdd / (2.0 * ell**2))

# ------------------------------------------------------------
# First derivative wrt FIRST argument r: ∂K/∂r
# ------------------------------------------------------------
def fdgibbs_kernel(Xd, Xdd, ell, s, u, w, r, rp):
    sig   = width(r,  s, u, w)
    sigp  = width(rp, s, u, w)
    sig_d = width_deriv(r, s, u, w)
    exp_term = torch.exp(-Xdd / (2.0 * ell**2))
    return exp_term * sigp * (sig_d - sig * (Xd / (ell**2)))

# ------------------------------------------------------------
# First derivative wrt SECOND argument r': ∂K/∂r'
# ------------------------------------------------------------
def sdgibbs_kernel(Xd, Xdd, ell, s, u, w, r, rp):
    sig   = width(r,  s, u, w)
    sigp  = width(rp, s, u, w)
    sigp_d = width_deriv(rp, s, u, w)
    exp_term = torch.exp(-Xdd / (2.0 * ell**2))
    return exp_term * sig * (sigp_d + sigp * (Xd / (ell**2)))

# ------------------------------------------------------------
# Mixed second derivative: ∂²K/(∂r ∂r')
# (this is the force–force block)
# ------------------------------------------------------------
def ddgibbs_kernel(Xd, Xdd, ell, s, u, w, r, rp):
    sig    = width(r,  s, u, w)
    sigp   = width(rp, s, u, w)
    sig_d  = width_deriv(r,  s, u, w)
    sigp_d = width_deriv(rp, s, u, w)
    exp_term = torch.exp(-Xdd / (2.0 * ell**2))
    term1 = sig_d * sigp_d
    term2 = (Xd / (ell**2)) * (sig_d * sigp - sig * sigp_d)
    term3 = sig * sigp * (1.0 / (ell**2) - Xdd / (ell**4))
    return exp_term * (term1 + term2 + term3)



# ----- GPR(D): DERIVATIVE GP PREDICTION FUNCTIONS -----
# Coded by: Brennon and Martin (cf. Csányi 2014)

# ** Basic Derivative GP with Constant Noise ** (DEPRECATED - c'mon we need to model the noise)
# ..very first attempt for toy problems
def derivative_gp_predict(x_obs, dy_obs_noisy, x_test, ell, w, noise_std, jitter=1e-6):
    # Covariances
    Xd_obs, Xdd_obs = pairwise_differences(x_obs, x_obs)
    K_dd = ddse_kernel(Xd_obs, Xdd_obs, ell, w) + noise_std**2 * torch.eye(len(x_obs))

    Xd_test_obs, Xdd_test_obs = pairwise_differences(x_test, x_obs)
    K_fd = fdse_kernel(Xd_test_obs, Xdd_test_obs, ell, w)

    Xd_test, Xdd_test = pairwise_differences(x_test, x_test)
    K_ff = se_kernel(Xdd_test, ell, w)

    # Cholesky solve
    L = torch.linalg.cholesky(K_dd + jitter * torch.eye(K_dd.shape[0]))
    alpha = torch.cholesky_solve(dy_obs_noisy.unsqueeze(-1), L)

    # Predictions
    pred_mean = (K_fd @ alpha).squeeze()
    v = torch.cholesky_solve(K_fd.T, L)
    pred_cov = K_ff - K_fd @ v

    return pred_mean, pred_cov


# ** GP with Heteroscedastic Noise ** (currently in use)
def gpr_d(x_obs, dy_obs, x_test, ell, w, noise_std_vec, jitter=1e-6):
    
    """
    Derivative-only GP regression with stationary SE kernel. (MODIFY!!!)
    
    :param x_obs: (n_d,) observed input locations for derivatives
    :param dy_obs: (n_d,) observed derivative values
    :param x_test: (m,) test input locations where predictions are made
    :param ell: lengthscale hyperparameter
    :param w: weight hyperparameter
    :param noise_std_vec: (n_d,) vector of noise standard deviations at derivative observations
    :param jitter: small diagonal jitter for numerical stability (by default set to 1e-6)

    :return pred_mean: (m,)
    :return pred_cov: (m, m)
    :return K_dd: (n_d, n_d)  -- derivative-derivative covariance matrix at observations
    :return L: (n_d, n_d)     -- Cholesky factor of K_dd
    :return alpha: (n_d,)     -- flattened alpha vector for later use
    """

    # Build kernels
    Xd_obs, Xdd_obs = pairwise_differences(x_obs, x_obs)
    K_dd = ddse_kernel(Xd_obs, Xdd_obs, ell, w)
    K_dd += torch.diag(noise_std_vec**2) + jitter * torch.eye(len(x_obs))

    Xd_test_obs, Xdd_test_obs = pairwise_differences(x_test, x_obs)
    K_fd = fdse_kernel(Xd_test_obs, Xdd_test_obs, ell, w)

    Xd_test, Xdd_test = pairwise_differences(x_test, x_test)
    K_ff = se_kernel(Xdd_test, ell, w)

    # GP equations
    L = torch.linalg.cholesky(K_dd)
    alpha = torch.cholesky_solve(dy_obs.unsqueeze(-1), L).squeeze()
    v = torch.cholesky_solve(K_fd.T, L)

    pred_mean = K_fd @ alpha
    pred_cov = K_ff - K_fd @ v

    return pred_mean, pred_cov, K_dd, L, alpha


# CAUTION WITH THIS ONE - DEBUGGING IN PROGRESS!!!
def gpr_d_gibbs(x_obs, dy_obs, x_test, ell, s, u, w, noise_std_vec, jitter=1e-6):
    
    """
    Derivative-only GP regression with non-stationary Gibbs kernel. (MODIFY!!!)
    
    :param x_obs: (n_d,) observed input locations for derivatives
    :param dy_obs: (n_d,) observed derivative values
    :param x_test: (m,) test input locations where predictions are made
    :param ell: lengthscale hyperparameter
    :param s: scale hyperparameter for Gibbs kernel
    :param u: location hyperparameter for Gibbs kernel
    :param w: weight hyperparameter
    :param noise_std_vec: (n_d,) vector of noise standard deviations at derivative observations
    :param jitter: small diagonal jitter for numerical stability (by default set to 1e-6)

    :return pred_mean: (m,)
    :return pred_cov: (m, m)
    :return K_dd: (n_d, n_d)  -- derivative-derivative covariance matrix at observations
    :return L: (n_d, n_d)     -- Cholesky factor of K_dd
    :return alpha: (n_d,)     -- flattened alpha vector for later use
    """

    dtype  = x_obs.dtype
    device = x_obs.device

    # ---- (1) derivative-derivative block: K_dd (n_d, n_d)
    Xd_obs, Xdd_obs = pairwise_differences(x_obs, x_obs)
    r_dd  = x_obs[:, None]
    rp_dd = x_obs[None, :]
    K_dd = ddgibbs_kernel(Xd_obs, Xdd_obs, ell, s, u, w, r_dd, rp_dd)
    K_dd = K_dd + torch.diag(noise_std_vec**2) + jitter * torch.eye(len(x_obs), dtype=dtype, device=device)

    #### THIS PART IS FUCKED???

    # ---- (2) test-derivative cross block: K_fd (m, n_d) = Cov[f(x_test), df/dx(x_obs)]
    Xd_test_obs, Xdd_test_obs = pairwise_differences(x_test, x_obs)
    r_fd  = x_test[:, None]
    rp_fd = x_obs[None, :]
    K_fd = sdgibbs_kernel(Xd_test_obs, Xdd_test_obs, ell, s, u, w, r_fd, rp_fd)

    # ADDED
    Xd_df, Xdd_df = pairwise_differences(x_obs, x_test)    # (n_d, n_f) : Xd = x_der - x_test
    r_df  = x_obs[:, None]
    rp_df = x_test[None, :]
    K_df = fdgibbs_kernel(Xd_df, Xdd_df, ell, s, u, w, r_df, rp_df)  # (n_d, n_f)

    # TEST
    #K_df = fdse_kernel(Xd_test_obs, Xdd_test_obs, ell, w)

    # ---- (3) function-function test block: K_ff (m, m)
    Xd_test, Xdd_test = pairwise_differences(x_test, x_test)
    r_ff  = x_test[:, None]
    rp_ff = x_test[None, :]
    K_ff = gibbs_kernel(Xdd_test, ell, s, u, w, r_ff, rp_ff)
    K_ff = K_ff + jitter * torch.eye(len(x_test), dtype=dtype, device=device)

    # DEBUG
    #Xd_dd, Xdd_dd = pairwise_differences(x_obs, x_obs)
    #r_dd  = x_obs[:, None]
    #rp_dd = x_obs[None, :]

    #sig    = width(r_dd,  s, u, w)
    #sig_d  = width_deriv(r_dd, s, u, w)

    #diag_formula = sig_d.squeeze()**2 + sig.squeeze()**2 / ell**2

    # DEBUG PRINTS
    #print("max K_dd diag from kernel :", torch.diag(ddgibbs_kernel(Xd_dd, Xdd_dd, ell, s, u, w, r_dd, rp_dd)).max())
    #print("max sig_d^2             :", (sig_d.squeeze()**2).max())
    #print("max sig^2 / ell^2       :", (sig.squeeze()**2 / ell**2).max())
    #print("argmax index            :", torch.argmax(sig_d.squeeze()**2))
    #print("x at argmax             :", x_obs[torch.argmax(sig_d.squeeze()**2)])

    # ---- GP equations
    L = torch.linalg.cholesky(K_dd)
    alpha = torch.cholesky_solve(dy_obs.unsqueeze(-1), L).squeeze(-1)   # (n_d,)
    #v = torch.cholesky_solve(K_fd.T, L)                                 # (n_d, m)
    v = torch.cholesky_solve(K_df, L)                                 # (n_d, m)

    pred_mean = K_fd @ alpha                                            # (m,)
    pred_cov  = K_ff - K_fd @ v                                         # (m, m)

    return pred_mean, pred_cov, K_dd, L, alpha


# ----- GPR(H): HISTOGRAM GP PREDICTION FUNCTION -----
# (i.e. GPR with linear basis, cf. Rasmussen §2.7)
# Coded by: Martin

def gpr_h(x_obs, y_obs, x_test,
                          ell, w,                # hyperparameters (so far fixed, i.e. part of the prior)
                          noise_cov,             # full covariance matrix from pos. observations (n_obs, n_obs)
                          H_obs,                 # (n_obs, p) design matrix for additive params
                          H_test=None,           # (n_test, p) design matrix at test points (often zeros)
                          jitter=1e-8):
    
    """
    GP regression with linear basis H (unknown additive coefficients marginalized out) with stationary SE kernel. (MODIFY!!!)
    
    :param x_obs: (n,) observed input locations
    :param y_obs: (n,) observed functional values
    :param x_test: (m,) test input locations where predictions are made
    :param ell: lengthscale hyperparameter
    :param w: weight hyperparameter
    :param H_obs: (n, p) design matrix for additive params at observations
    :param H_test: (m, p) torch or None, design matrix at test points (often zeros, default None)
    :param noise_cov: (n, n) full covariance matrix from functional observations
    :param jitter: small diagonal jitter for numerical stability (by default set to 1e-8)

    :return pred_mean: (m,)
    :return pred_cov: (m, m)
    :return K_yy: (n, n)  -- covariance matrix at observations
    :return L: (n, n)     -- Cholesky factor of K_yy
    :return y_col: (n,)   -- reshaped y_obs for later use
    :return m_off: (n,)   -- mean offset at observations for later use
    :return alpha: (n,)     -- flattened alpha vector for later use
    """

    n = x_obs.shape[0]
    m = x_test.shape[0]
    p = H_obs.shape[1]

    if H_test is None:
        H_test = torch.zeros((m, p))

    # (0)
    # Precompute distance matrices from data
    Xd_yy, Xdd_yy = pairwise_differences(x_obs, x_obs)
    Xd_yx, Xdd_yx = pairwise_differences(x_obs, x_test)
    Xd_xx, Xdd_xx = pairwise_differences(x_test, x_test)

    # (1)
    # Define and precompute covariance matrices

    # K_yy (n,n)
    K_yy = se_kernel(Xdd_yy, ell, w)
    K_yy = K_yy + noise_cov #+ jitter * torch.eye(n)

    K_yy = 0.5 * (K_yy + K_yy.T) #symmetrize - DOESN'T HELP WITH CHOLESKY??!
    K_yy += jitter * torch.eye(n) # jitter here??

    # Cross covariances
    K_yx = se_kernel(Xdd_yx, ell, w)            # (n, m)
    K_xy = K_yx.T                               # (m, n)
    K_xx = se_kernel(Xdd_xx, ell, w)
    K_xx = K_xx + jitter * torch.eye(m)

    # (2)
    # Cholesky factor of K_yy
    L = torch.linalg.cholesky(K_yy, upper=False)  # lower-triangular: L L^T = K_yy

    # (3)
    # Compute K_yy^{-1} y and K_yy^{-1} H (stable) using L
    # Use cholesky_solve: note it expects (b, L) with L from cholesky
    y_col = y_obs.reshape(-1,1)
    # u
    Kinv_y = torch.cholesky_solve(y_col, L)         # (n, 1)
    # A
    Kinv_H = torch.cholesky_solve(H_obs, L)         # (n, p)

    # (4)
    # Small p x p matrix S = H^T K_yy^{-1} H
    S = H_obs.T @ Kinv_H                            # (p, p)
    # Make S symmetric numerically
    S = 0.5 * (S + S.T)

    # (5)
    # Solve for beta_hat: S beta_hat = H^T K_yy^{-1} y
    rhs = H_obs.T @ Kinv_y                          # (p, 1)
    # Solve linear system for beta_hat (use solve, p is small)
    beta_hat = torch.linalg.solve(S, rhs)           # (p, 1)

    # (6)
    # alpha = K_yy^{-1} y - K_yy^{-1} H beta_hat = K_tilde y (.. subtracting the mean)
    alpha = Kinv_y - Kinv_H @ beta_hat              # (n, 1)

    # (7)
    # Posterior (predictive) mean: K_*y alpha + H_* beta_hat
    pred_mean = (K_xy @ alpha).squeeze(-1) + (H_test @ beta_hat).squeeze(-1)  # (m,)

    # Posterior (predictive) covariance:
    # term1 = K_xx - K_xy K_yy^{-1} K_yx  (efficient via triangular solves)
    # compute v = L^{-1} K_yx  => then K_xy K_yy^{-1} K_yx = v.T @ v
    v = torch.linalg.solve_triangular(L, K_yx, upper=False)   # (n, m)
    term1 = K_xx - v.T @ v                                    # (m, m)

    # term2 = (H_* - K_*y K_yy^{-1} H) S^{-1} (H_* - K_*y K_yy^{-1} H)^T
    # Kinv_H                                            # (n, p) ..just to see it here for clarity
    M = K_xy @ Kinv_H                                   # (m, p)
    D = H_test - M                                      # (m, p)
    S_inv = torch.linalg.inv(S)                         # (p, p)   (p small)
    term2 = D @ (S_inv @ D.T)                           # (m, m)

    pred_cov = term1 + term2

    #flatten alpha and beta for returning
    alpha = alpha.squeeze(-1)                  # (n,)
    #beta_vec = beta_hat.squeeze(-1)           # (p,)

    m_off = (H_obs @ beta_hat).squeeze(-1)     # (n,)
    y_col = y_col.squeeze(-1)                  # (n,)

    return pred_mean, pred_cov, K_yy, L, y_col, m_off, alpha


def gpr_h_gibbs(x_obs, y_obs, x_test,
                     ell, s, u, w,          # Gibbs hyperparameters
                     noise_cov,             # (n_obs, n_obs)
                     H_obs,                 # (n_obs, p)
                     H_test=None,           # (n_test, p)
                     jitter=1e-8
                     ):
    """
    GP regression with linear basis H (unknown additive coefficients marginalized out) with non-stationary Gibbs kernel. (MODIFY!!!)
    
    :param x_obs: (n,) observed input locations
    :param y_obs: (n,) observed functional values
    :param x_test: (m,) test input locations where predictions are made
    :param ell: lengthscale hyperparameter
    :param s: signal variance (scaling) hyperparameter of Gibbs kernel
    :param u: shift hyperparameter of Gibbs kernel
    :param w: weight hyperparameter
    :param H_obs: (n, p) design matrix for additive params at observations
    :param H_test: (m, p) torch or None, design matrix at test points (often zeros, default None)
    :param noise_cov: (n, n) full covariance matrix from functional observations
    :param jitter: small diagonal jitter for numerical stability (by default set to 1e-8)

    :return pred_mean: (m,)
    :return pred_cov: (m, m)
    :return K_yy: (n, n)  -- covariance matrix at observations
    :return L: (n, n)     -- Cholesky factor of K_yy
    :return y_col: (n,)   -- reshaped y_obs for later use
    :return m_off: (n,)   -- mean offset at observations for later use
    :return alpha: (n,)     -- flattened alpha vector for later use
    """

    n = x_obs.shape[0]
    m = x_test.shape[0]
    p = H_obs.shape[1]

    if H_test is None:
        H_test = torch.zeros((m, p), device=x_obs.device, dtype=x_obs.dtype)

    # ---------------------------------------------------
    # (0) pairwise diffs
    # ---------------------------------------------------
    Xd_yy, Xdd_yy = pairwise_differences(x_obs, x_obs)
    Xd_yx, Xdd_yx = pairwise_differences(x_obs, x_test)
    Xd_xx, Xdd_xx = pairwise_differences(x_test, x_test)

    r_yy, rp_yy = x_obs[:, None], x_obs[None, :]
    r_yx, rp_yx = x_obs[:, None], x_test[None, :]
    r_xx, rp_xx = x_test[:, None], x_test[None, :]

    # ---------------------------------------------------
    # (1) Gibbs covariance matrices
    # ---------------------------------------------------
    K_yy = gibbs_kernel(Xdd_yy, ell, s, u, w, r_yy, rp_yy)
    K_yy = K_yy + noise_cov + jitter * torch.eye(n, device=x_obs.device)

    K_yx = gibbs_kernel(Xdd_yx, ell, s, u, w, r_yx, rp_yx)
    K_xy = K_yx.T

    K_xx = gibbs_kernel(Xdd_xx, ell, s, u, w, r_xx, rp_xx)
    K_xx = K_xx + jitter * torch.eye(m, device=x_obs.device)

    K_yy = 0.5 * (K_yy + K_yy.T)

    # ---------------------------------------------------
    # (2) Cholesky
    # ---------------------------------------------------
    L = torch.linalg.cholesky(K_yy, upper=False)

    # ---------------------------------------------------
    # (3) Solve systems
    # ---------------------------------------------------
    y_col = y_obs.reshape(-1, 1)

    Kinv_y = torch.cholesky_solve(y_col, L)
    Kinv_H = torch.cholesky_solve(H_obs, L)

    # ---------------------------------------------------
    # (4) S matrix
    # ---------------------------------------------------
    S = H_obs.T @ Kinv_H
    S = 0.5 * (S + S.T)

    # ---------------------------------------------------
    # (5) beta_hat
    # ---------------------------------------------------
    rhs = H_obs.T @ Kinv_y
    beta_hat = torch.linalg.solve(S, rhs)

    # ---------------------------------------------------
    # (6) alpha
    # ---------------------------------------------------
    alpha = Kinv_y - Kinv_H @ beta_hat

    # ---------------------------------------------------
    # (7) predictive mean
    # ---------------------------------------------------
    pred_mean = (K_xy @ alpha).squeeze(-1) + (H_test @ beta_hat).squeeze(-1)

    # ---------------------------------------------------
    # (8) predictive covariance
    # ---------------------------------------------------
    v = torch.linalg.solve_triangular(L, K_yx, upper=False)
    term1 = K_xx - v.T @ v

    M = K_xy @ Kinv_H
    D = H_test - M
    S_inv = torch.linalg.inv(S)
    term2 = D @ (S_inv @ D.T)

    pred_cov = term1 + term2

    #flatten alpha and beta for returning
    alpha_vec = alpha.squeeze(-1)      # (n,)
    #beta_vec = beta_hat.squeeze(-1)    # (p,)

    m_off = (H_obs @ beta_hat).squeeze(-1)     # (n,)
    y_col = y_col.squeeze(-1)             # (n,)

    return pred_mean, pred_cov, K_yy, L, y_col, m_off, alpha_vec



# ----- GPR(H+D): JOINED INFERENCE GP PREDICTION FUNCTION -----
# Coded by: Martin

def gpr_hd(x_func, y_func,                    # function (histogram) observations
           x_der, dy_der,                    # derivative observations (df/dx)
           x_test,
           ell, w,                           # kernel hyperparameters
           noise_func_cov,                  # (n_f,n_f) covariances for function obs => we need full covariance matrix here!
           noise_deriv_diag,                 # (n_d,) variances for derivative obs
           H_func,                           # (n_f, p) design matrix for function obs
           H_test=None,                      # (m, p) design matrix at test points (or None)
           jitter=1e-8
           ):
    
    """
    Joint GP using BOTH function (histogram) observations and derivative observations, based on the stationary SE kernel.

    Returns:
      pred_mean: (m,)
      pred_cov : (m, m)
      K_joint: 
      L:
      y_joint:
      m_joint:
      alpha    : (n_f + n_d,)  -- weights for the joint observations (useful for diagnostics)

    Notes:
      - Function observations are at x_func with values y_func and noise covariances noise_func_cov.
      - Derivative observations are at x_der with values dy_der and noise variances noise_deriv_diag.
      - H_func is the (n_f, p) basis for the function observations only. Derivative rows receive zeros for the basis.
      - If there are no function observations (n_f==0) or no derivative observations (n_d==0),
        this function delegates to your existing gpr_d / gpr_h routines for simplicity.
    """

    # quick guards: delegate to specialized routines if one type of observation is missing
    n_f = 0 if x_func is None else x_func.shape[0]
    n_d = 0 if x_der is None else x_der.shape[0]
    m = x_test.shape[0]

    # If no function observations -> use derivative-only GP (existing)
    #if n_f == 0:
        # expects (x_obs, dy_obs, x_test, ell, w, noise_std_vec)
    #    return gpr_d(x_der, dy_der, x_test, ell, w, noise_deriv_diag, jitter=jitter)

    # If no derivative observations -> use histogram-only GP (existing)
    #if n_d == 0:
    #    return gpr_h_full(x_func, y_func, x_test, ell, w,
    #                 noise_func_cov, H_func, H_test=H_test, jitter=jitter)

    # --- Build covariance blocks ------------------------------------------------
    # (1) function-function block (n_f, n_f)
    Xd_ff, Xdd_ff = pairwise_differences(x_func, x_func)
    K_ff = se_kernel(Xdd_ff, ell, w)
    K_ff = K_ff + noise_func_cov + jitter * torch.eye(n_f)

    # (2) derivative-derivative block (n_d, n_d)
    Xd_dd, Xdd_dd = pairwise_differences(x_der, x_der)
    K_dd = ddse_kernel(Xd_dd, Xdd_dd, ell, w)
    K_dd = K_dd + torch.diag(noise_deriv_diag) + jitter * torch.eye(n_d)

    # (3) function-derivative cross blocks. (NOTE: K_df = K_fd.T)
    #    K_fd = Cov[f(x_func), df/dx(x_der)]  -> use fdse_kernel with Xd = x_func - x_der
    Xd_fd, Xdd_fd = pairwise_differences(x_func, x_der)   # (n_f, n_d)
    K_fd = fdse_kernel(Xd_fd, Xdd_fd, ell, w)              # (n_f, n_d)

    #    K_df = Cov[df/dx(x_der), f(x_func)]  -> use fdse_kernel with Xd = x_der - x_func (gives correct sign)
    #Xd_df, Xdd_df = pairwise_differences(x_der, x_func)   # (n_d, n_f)
    #K_df = fdse_kernel(Xd_df, Xdd_df, ell, w)              # (n_d, n_f)
    K_df = K_fd.T                                       # (n_d, n_f)

    # (4) assemble joint K (n = n_f + n_d)
    top = torch.cat([K_ff, K_fd], dim=1)                   # (n_f, n_f + n_d)
    bot = torch.cat([K_df, K_dd], dim=1)                   # (n_d, n_f + n_d)
    K_joint = torch.cat([top, bot], dim=0)                 # (n, n)

    # small extra jitter to the whole joint matrix (numerical safety)
    n_joint = n_f + n_d
    K_joint = 0.5 * (K_joint + K_joint.T)
    K_joint = K_joint + jitter * torch.eye(n_joint)
    # print(K_joint.size())

    # --- Build joint observation vector y_joint and H_full ---------------------
    # y_joint stacked: [y_func; dy_der]
    y_f_col = y_func.reshape(-1, 1)                        # (n_f, 1)
    y_d_col = dy_der.reshape(-1, 1)                        # (n_d, 1)
    y_joint_col = torch.cat([y_f_col, y_d_col], dim=0)     # (n, 1)

    # H_full: function rows have H_func, derivative rows are zeros (derivatives do not share the linear basis)
    p = H_func.shape[1]
    H_zeros_for_der = torch.zeros((n_d, p), dtype=H_func.dtype, device=H_func.device)
    H_full = torch.cat([H_func, H_zeros_for_der], dim=0)   # (n, p)

    if H_test is None:
        H_test = torch.zeros((m, p), dtype=H_func.dtype, device=H_func.device)

    # --- Cholesky solves (stable linear algebra) --------------------------------
    L = torch.linalg.cholesky(K_joint)                     # L L^T = K_joint

    # K_joint^{-1} y and K_joint^{-1} H (via cholesky_solve)
    Kinv_y = torch.cholesky_solve(y_joint_col, L)          # (n, 1)
    Kinv_H = torch.cholesky_solve(H_full, L)              # (n, p)

    # S = H^T K^{-1} H  (small p x p)
    S = H_full.T @ Kinv_H
    # numerical symmetrize
    S = 0.5 * (S + S.T)

    # Solve for beta_hat: S beta = H^T K^{-1} y
    rhs = H_full.T @ Kinv_y                                # (p, 1)
    beta_hat = torch.linalg.solve(S, rhs)                  # (p, 1)

    # alpha = K^{-1} y - K^{-1} H beta_hat   (n,1)
    alpha = Kinv_y - Kinv_H @ beta_hat                    # (n, 1)

    # --- Predictive mean for function values at x_test -------------------------
    # Build covariances between test function points and training blocks:
    # K_xf: Cov[f(x_test), f(x_func)]    -> se_kernel with pairwise(x_test, x_func)
    Xd_xf, Xdd_xf = pairwise_differences(x_test, x_func)  # (m, n_f)
    K_xf = se_kernel(Xdd_xf, ell, w)                      # (m, n_f)

    # K_xd: Cov[f(x_test), df/dx(x_der)] -> fdse_kernel with pairwise(x_test, x_der)
    Xd_xd, Xdd_xd = pairwise_differences(x_test, x_der)   # (m, n_d)
    K_xd = fdse_kernel(Xd_xd, Xdd_xd, ell, w)             # (m, n_d)

    # stack horizontally to get K_xY (m, n)
    K_xY = torch.cat([K_xf, K_xd], dim=1)                 # (m, n)

    # predictive mean = K_xY @ alpha + H_test @ beta_hat
    pred_mean = (K_xY @ alpha).squeeze(-1) + (H_test @ beta_hat).squeeze(-1)  # (m,)

    # --- Predictive covariance -------------------------------------------------
    # term1 = K_xx - K_xY K^{-1} K_Yx  (efficient via triangular solves)
    # compute K_Yx = K_xY.T (n, m), then v = L^{-1} K_Yx  -> term1 = K_xx - v.T @ v
    K_Yx = K_xY.T                                          # (n, m)
    v = torch.linalg.solve_triangular(L, K_Yx, upper=False)  # (n, m)

    # K_xx
    Xd_xx, Xdd_xx = pairwise_differences(x_test, x_test)
    K_xx = se_kernel(Xdd_xx, ell, w)
    K_xx = K_xx + jitter * torch.eye(m)

    term1 = K_xx - v.T @ v                                 # (m, m)

    # term2 = (H_test - K_xY K^{-1} H_full) S^{-1} (H_test - K_xY K^{-1} H_full)^T
    M = K_xY @ Kinv_H                                      # (m, p)
    D = H_test - M                                         # (m, p)
    S_inv = torch.linalg.inv(S)                            # small p x p
    term2 = D @ (S_inv @ D.T)                              # (m, m)

    pred_cov = term1 + term2

    # flatten alpha and beta for returning
    alpha_vec = alpha.squeeze(-1)      # (n,)
    #beta_vec = beta_hat.squeeze(-1)    # (p,)

    m_joint = (H_full @ beta_hat).squeeze(-1)     # (n,)
    y_joint = y_joint_col.squeeze(-1)             # (n,)

    return pred_mean, pred_cov, K_joint, L, y_joint, m_joint, alpha_vec


def gpr_hd_gibbs(x_func, y_func,                   # function (histogram) observations
                 x_der, dy_der,                    # derivative observations (df/dx)
                 x_test,
                 ell, s, u, w,                     # kernel hyperparameters (ell fixed; width params)
                 noise_func_cov,                   # (n_f,n_f) full cov for function obs
                 noise_deriv_diag,                 # (n_d,) variances for derivative obs
                 H_func,                           # (n_f, p) design matrix for function obs
                 H_test=None,                      # (m, p) design matrix at test points (or None)
                 jitter=1e-6
                ):
    
    """
    Joint GP using BOTH function (histogram) observations and derivative observations, built with the non-stationary Gibbs kernel.

    Returns:
      pred_mean: (m,)
      pred_cov : (m, m)
      K_joint: (n_f + n_d, n_f + n_d)
      L:
      y_joint:
      m_joint:
      alpha    : (n_f + n_d,)  -- weights for the joint observations (useful for diagnostics)

    Uses Gibbs-like kernel with erf width envelope:
      K(r,r') = width(r) width(r') exp(-(r-r')^2/(2 ell^2))
    and its derivatives:
      fdgibbs_kernel = ∂K/∂r
      ddgibbs_kernel = ∂²K/(∂r ∂r')
    """

    n_f = 0 if x_func is None else x_func.shape[0]
    n_d = 0 if x_der  is None else x_der.shape[0]
    m   = x_test.shape[0]

    # Delegate if one type of observation is missing
    #if n_f == 0:
    #    return gpr_d(x_der, dy_der, x_test, ell, w, noise_deriv_diag, jitter=jitter)  # unchanged fallback
    #if n_d == 0:
    #    return gpr_h_full(x_func, y_func, x_test, ell, w, noise_func_cov, H_func, H_test=H_test, jitter=jitter)

    # --- Build covariance blocks ------------------------------------------------
    # (1) function-function block (n_f, n_f)
    Xd_ff, Xdd_ff = pairwise_differences(x_func, x_func)   # (n_f, n_f)
    # need r, rp grids for width terms
    r_ff  = x_func[:, None]   # (n_f, 1)
    rp_ff = x_func[None, :]   # (1, n_f)
    K_ff = gibbs_kernel(Xdd_ff, ell, s, u, w, r_ff, rp_ff)
    K_ff = K_ff + noise_func_cov + jitter * torch.eye(n_f, dtype=K_ff.dtype, device=K_ff.device)

    # (2) derivative-derivative block (n_d, n_d): mixed derivative ∂²/∂r∂r'
    Xd_dd, Xdd_dd = pairwise_differences(x_der, x_der)     # (n_d, n_d)
    r_dd  = x_der[:, None]
    rp_dd = x_der[None, :]
    K_dd = ddgibbs_kernel(Xd_dd, Xdd_dd, ell, s, u, w, r_dd, rp_dd)
    K_dd = K_dd + torch.diag(noise_deriv_diag) + jitter * torch.eye(n_d, dtype=K_dd.dtype, device=K_dd.device)

    # (3) function-derivative cross blocks
    # K_fd = Cov[f(x_func), df/dx(x_der)]
    Xd_fd, Xdd_fd = pairwise_differences(x_func, x_der)    # (n_f, n_d) : Xd = x_func - x_der
    r_fd  = x_func[:, None]  # first arg grid
    rp_fd = x_der[None, :]   # second arg grid
    K_fd = sdgibbs_kernel(Xd_fd, Xdd_fd, ell, s, u, w, r_fd, rp_fd)  # (n_f, n_d)

    # K_df = Cov[df/dx(x_der), f(x_func)] should be ∂/∂(first arg) evaluated at (x_der, x_func)
    # Instead of K_fd.T (which would miss the sign when width depends on the first arg),
    # compute explicitly to be safe:
    Xd_df, Xdd_df = pairwise_differences(x_der, x_func)    # (n_d, n_f) : Xd = x_der - x_func
    r_df  = x_der[:, None]
    rp_df = x_func[None, :]
    K_df = fdgibbs_kernel(Xd_df, Xdd_df, ell, s, u, w, r_df, rp_df)  # (n_d, n_f)

    # (4) assemble joint K (n = n_f + n_d)
    top = torch.cat([K_ff, K_fd], dim=1)                   # (n_f, n_f + n_d)
    bot = torch.cat([K_df, K_dd], dim=1)                   # (n_d, n_f + n_d)
    K_joint = torch.cat([top, bot], dim=0)                 # (n, n)

    # numerical safety
    n_joint = n_f + n_d
    K_joint = 0.5 * (K_joint + K_joint.T) #symmetrize
    K_joint = K_joint + jitter * torch.eye(n_joint, dtype=K_joint.dtype, device=K_joint.device)

    # --- Build y_joint and H_full ----------------------------------------------
    y_joint_col = torch.cat([y_func.reshape(-1, 1), dy_der.reshape(-1, 1)], dim=0)  # (n,1)

    p = H_func.shape[1]
    H_zeros_for_der = torch.zeros((n_d, p), dtype=H_func.dtype, device=H_func.device)
    H_full = torch.cat([H_func, H_zeros_for_der], dim=0)   # (n, p)

    if H_test is None:
        H_test = torch.zeros((m, p), dtype=H_func.dtype, device=H_func.device)

    # --- Cholesky solves --------------------------------------------------------
    L = torch.linalg.cholesky(K_joint, upper=False)

    Kinv_y = torch.cholesky_solve(y_joint_col, L)          # (n, 1)
    Kinv_H = torch.cholesky_solve(H_full, L)               # (n, p)

    S = H_full.T @ Kinv_H
    S = 0.5 * (S + S.T)
    rhs = H_full.T @ Kinv_y
    beta_hat = torch.linalg.solve(S, rhs)                  # (p,1)

    alpha = Kinv_y - Kinv_H @ beta_hat                     # (n,1)

    # --- Predictive mean at x_test ---------------------------------------------
    # K_xf: Cov[f(x_test), f(x_func)]
    Xd_xf, Xdd_xf = pairwise_differences(x_test, x_func)   # (m, n_f)
    r_xf  = x_test[:, None]
    rp_xf = x_func[None, :]
    K_xf = gibbs_kernel(Xdd_xf, ell, s, u, w, r_xf, rp_xf)  # (m, n_f)

    # K_xd: Cov[f(x_test), df/dx(x_der)]
    Xd_xd, Xdd_xd = pairwise_differences(x_test, x_der)    # (m, n_d)
    r_xd  = x_test[:, None]
    rp_xd = x_der[None, :]
    K_xd = sdgibbs_kernel(Xd_xd, Xdd_xd, ell, s, u, w, r_xd, rp_xd)  # (m, n_d)

    K_xY = torch.cat([K_xf, K_xd], dim=1)                  # (m, n)

    # TEST: Should we calculate K_Yx like this or just use K_xY.T?
    # Using K_Yx = K_xY.T yields slightly bigger uncertainty..
    # Does this even matter?
    Xd_dx, Xdd_dx = pairwise_differences(x_der, x_test)    # (m, n_d)
    r_dx  = x_der[:, None]
    rp_dx = x_test[None, :]
    K_dx = fdgibbs_kernel(Xd_dx, Xdd_dx, ell, s, u, w, r_dx, rp_dx)  # (n_d, m)
    K_Yx = torch.cat([K_xf.T, K_dx], dim=0)                          # (n, m)  

    pred_mean = (K_xY @ alpha).squeeze(-1) + (H_test @ beta_hat).squeeze(-1)

    # --- Predictive covariance -------------------------------------------------
    #K_Yx = K_xY.T                                          # (n, m)
    v = torch.linalg.solve_triangular(L, K_Yx, upper=False) # (n, m)

    # K_xx
    Xd_xx, Xdd_xx = pairwise_differences(x_test, x_test)    # (m, m)
    r_xx  = x_test[:, None]
    rp_xx = x_test[None, :]
    K_xx = gibbs_kernel(Xdd_xx, ell, s, u, w, r_xx, rp_xx)
    K_xx = K_xx + jitter * torch.eye(m, dtype=K_xx.dtype, device=K_xx.device)

    term1 = K_xx - v.T @ v

    M = K_xY @ Kinv_H
    D = H_test - M
    S_inv = torch.linalg.inv(S)
    term2 = D @ (S_inv @ D.T)

    pred_cov = term1 + term2

    # flatten alpha and beta for returning
    alpha_vec = alpha.squeeze(-1)      # (n,)
    #beta_vec = beta_hat.squeeze(-1)    # (p,)

    m_joint = (H_full @ beta_hat).squeeze(-1)     # (n,)
    y_joint = y_joint_col.squeeze(-1)             # (n,)

    return pred_mean, pred_cov, K_joint, L, y_joint, m_joint, alpha_vec



# ----- TESTING ONLY ------
# This is getting buggy and I'm getting tired of this..

# MERGED GPR CODE FOR STATIONARY KERNELS ONLY
def gpr(x_func, y_func,                    # function (histogram) observations
        x_der, dy_der,                    # derivative observations (df/dx)
        x_test,
        ell, w,                           # kernel hyperparameters
        noise_func_cov=None,              # (n_f,n_f) covariances for function obs
        noise_deriv_diag=None,            # (n_d,) variances for derivative obs
        H_func=None,                      # (n_f, p) design matrix for function obs
        H_test=None,                      # (m, p) design matrix at test points
        jitter=1e-8):
    """
    Unified GP using function observations, derivative observations, or both.
    Stationary SE kernel with derivative support.

    Works for:
      - histogram only        (x_der is None)
      - derivative only       (x_func is None)
      - joint histogram+grad  (both provided)
    """

    device = x_test.device
    dtype  = x_test.dtype

    n_f = 0 if x_func is None else x_func.shape[0]
    n_d = 0 if x_der  is None else x_der.shape[0]
    m   = x_test.shape[0]

    use_f = n_f > 0
    use_d = n_d > 0

    if not use_f and not use_d:
        raise ValueError("At least one of x_func or x_der must be provided.")

    # -----------------------------
    # Defaults
    # -----------------------------
    if use_f:
        if H_func is None:
            H_func = torch.zeros((n_f, 1), dtype=dtype, device=device)
        p = H_func.shape[1]
        if noise_func_cov is None:
            noise_func_cov = torch.zeros((n_f, n_f), dtype=dtype, device=device)
    else:
        p = 0

    if use_d and noise_deriv_diag is None:
        noise_deriv_diag = torch.zeros(n_d, dtype=dtype, device=device)

    if H_test is None:
        H_test = torch.zeros((m, p), dtype=dtype, device=device)

    # -----------------------------
    # Build covariance blocks
    # -----------------------------
    blocks = []

    if use_f:
        Xd_ff, Xdd_ff = pairwise_differences(x_func, x_func)
        K_ff = se_kernel(Xdd_ff, ell, w)
        K_ff = K_ff + noise_func_cov + jitter * torch.eye(n_f, device=device)
        blocks.append(K_ff)

    if use_f and use_d:
        Xd_fd, Xdd_fd = pairwise_differences(x_func, x_der)
        K_fd = fdse_kernel(Xd_fd, Xdd_fd, ell, w)
        K_df = K_fd.T

    if use_d:
        Xd_dd, Xdd_dd = pairwise_differences(x_der, x_der)
        K_dd = ddse_kernel(Xd_dd, Xdd_dd, ell, w)
        K_dd = K_dd + torch.diag(noise_deriv_diag) + jitter * torch.eye(n_d, device=device)

    if use_f and use_d:
        top = torch.cat([K_ff, K_fd], dim=1)
        bot = torch.cat([K_df, K_dd], dim=1)
        K_joint = torch.cat([top, bot], dim=0)
    elif use_f:
        K_joint = K_ff
    else:
        K_joint = K_dd

    n_joint = K_joint.shape[0]
    K_joint = 0.5 * (K_joint + K_joint.T) + jitter * torch.eye(n_joint, device=device)

    # -----------------------------
    # Build y_joint and H_full
    # -----------------------------
    ys = []
    Hs = []

    if use_f:
        ys.append(y_func.reshape(-1, 1))
        Hs.append(H_func)

    if use_d:
        ys.append(dy_der.reshape(-1, 1))
        Hs.append(torch.zeros((n_d, p), dtype=dtype, device=device))

    y_joint = torch.cat(ys, dim=0)
    H_full  = torch.cat(Hs, dim=0) if p > 0 else torch.zeros((n_joint, 0), device=device)

    # -----------------------------
    # Cholesky solves
    # -----------------------------
    L = torch.linalg.cholesky(K_joint)

    Kinv_y = torch.cholesky_solve(y_joint, L)
    Kinv_H = torch.cholesky_solve(H_full, L) if p > 0 else None

    if p > 0:
        S = H_full.T @ Kinv_H
        S = 0.5 * (S + S.T)
        rhs = H_full.T @ Kinv_y
        beta_hat = torch.linalg.solve(S, rhs)
        alpha = Kinv_y - Kinv_H @ beta_hat
    else:
        beta_hat = None
        alpha = Kinv_y

    # -----------------------------
    # Prediction mean
    # -----------------------------
    Kx_parts = []

    if use_f:
        Xd_xf, Xdd_xf = pairwise_differences(x_test, x_func)
        K_xf = se_kernel(Xdd_xf, ell, w)
        Kx_parts.append(K_xf)

    if use_d:
        Xd_xd, Xdd_xd = pairwise_differences(x_test, x_der)
        K_xd = fdse_kernel(Xd_xd, Xdd_xd, ell, w)
        Kx_parts.append(K_xd)

    K_xY = torch.cat(Kx_parts, dim=1)

    pred_mean = (K_xY @ alpha).squeeze(-1)
    if p > 0:
        pred_mean += (H_test @ beta_hat).squeeze(-1)

    # -----------------------------
    # Prediction covariance
    # -----------------------------
    K_Yx = K_xY.T
    v = torch.linalg.solve_triangular(L, K_Yx, upper=False)

    Xd_xx, Xdd_xx = pairwise_differences(x_test, x_test)
    K_xx = se_kernel(Xdd_xx, ell, w) + jitter * torch.eye(m, device=device)

    term1 = K_xx - v.T @ v

    if p > 0:
        M = K_xY @ Kinv_H
        D = H_test - M
        S_inv = torch.linalg.inv(S)
        term2 = D @ (S_inv @ D.T)
        pred_cov = term1 + term2
    else:
        pred_cov = term1

    alpha_vec = alpha.squeeze(-1)
    m_joint = (H_full @ beta_hat).squeeze(-1) if p > 0 else torch.zeros(n_joint, device=device)

    return pred_mean, pred_cov, K_joint, L, y_joint.squeeze(-1), m_joint, alpha_vec