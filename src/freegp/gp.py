"""Stationary GP kernels and the joint histogram+derivative GP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def pairwise_differences(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(1, -1)
    xd = x1 - x2
    return xd, xd**2


def se_kernel(xdd: torch.Tensor, ell: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return w**2 * torch.exp(-xdd / (2 * ell**2))


def fdse_kernel(xd: torch.Tensor, xdd: torch.Tensor, ell: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return w**2 / ell**2 * torch.exp(-xdd / (2 * ell**2)) * xd


def ddse_kernel(xd: torch.Tensor, xdd: torch.Tensor, ell: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    exp_term = torch.exp(-xdd / (2 * ell**2))
    return w**2 / ell**4 * exp_term * (ell**2 - xdd)


@dataclass(frozen=True)
class JointGPPosterior:
    x_func: torch.Tensor
    y_func: torch.Tensor
    x_der: torch.Tensor
    dy_der: torch.Tensor
    ell: torch.Tensor
    w: torch.Tensor
    noise_func_cov: torch.Tensor
    noise_deriv_diag: torch.Tensor
    H_func: torch.Tensor
    H_full: torch.Tensor
    K_joint: torch.Tensor
    L: torch.Tensor
    y_joint: torch.Tensor
    m_joint: torch.Tensor
    alpha: torch.Tensor
    beta_hat: torch.Tensor
    Kinv_H: torch.Tensor
    jitter: float


def _derivative_noise_matrix(noise_deriv_diag: torch.Tensor) -> torch.Tensor:
    if noise_deriv_diag.ndim == 2:
        return noise_deriv_diag
    return torch.diag(noise_deriv_diag)


def build_joint_gp(
    *,
    x_func: torch.Tensor,
    y_func: torch.Tensor,
    x_der: torch.Tensor,
    dy_der: torch.Tensor,
    ell: torch.Tensor,
    w: torch.Tensor,
    noise_func_cov: torch.Tensor,
    noise_deriv_diag: torch.Tensor,
    H_func: torch.Tensor,
    jitter: float = 1e-8,
) -> JointGPPosterior:
    """Build the profiled joint GP posterior for function + derivative observations."""
    n_f = x_func.shape[0]
    n_d = x_der.shape[0]

    xd_ff, xdd_ff = pairwise_differences(x_func, x_func)
    K_ff = se_kernel(xdd_ff, ell, w) + noise_func_cov + jitter * torch.eye(
        n_f, dtype=x_func.dtype, device=x_func.device
    )

    xd_dd, xdd_dd = pairwise_differences(x_der, x_der)
    K_dd = ddse_kernel(xd_dd, xdd_dd, ell, w) + _derivative_noise_matrix(noise_deriv_diag)
    K_dd = K_dd + jitter * torch.eye(n_d, dtype=x_der.dtype, device=x_der.device)

    xd_fd, xdd_fd = pairwise_differences(x_func, x_der)
    K_fd = fdse_kernel(xd_fd, xdd_fd, ell, w)
    K_df = K_fd.T

    top = torch.cat([K_ff, K_fd], dim=1)
    bot = torch.cat([K_df, K_dd], dim=1)
    K_joint = torch.cat([top, bot], dim=0)
    K_joint = 0.5 * (K_joint + K_joint.T)
    K_joint = K_joint + jitter * torch.eye(n_f + n_d, dtype=K_joint.dtype, device=K_joint.device)

    y_joint_col = torch.cat([y_func.reshape(-1, 1), dy_der.reshape(-1, 1)], dim=0)

    p = H_func.shape[1]
    H_zeros_for_der = torch.zeros((n_d, p), dtype=H_func.dtype, device=H_func.device)
    H_full = torch.cat([H_func, H_zeros_for_der], dim=0)

    L = torch.linalg.cholesky(K_joint)
    Kinv_y = torch.cholesky_solve(y_joint_col, L)
    Kinv_H = torch.cholesky_solve(H_full, L)

    S = 0.5 * ((H_full.T @ Kinv_H) + (H_full.T @ Kinv_H).T)
    rhs = H_full.T @ Kinv_y
    beta_hat = torch.linalg.solve(S, rhs)
    alpha = Kinv_y - Kinv_H @ beta_hat

    y_joint = y_joint_col.squeeze(-1)
    m_joint = (H_full @ beta_hat).squeeze(-1)

    return JointGPPosterior(
        x_func=x_func,
        y_func=y_func,
        x_der=x_der,
        dy_der=dy_der,
        ell=ell,
        w=w,
        noise_func_cov=noise_func_cov,
        noise_deriv_diag=noise_deriv_diag,
        H_func=H_func,
        H_full=H_full,
        K_joint=K_joint,
        L=L,
        y_joint=y_joint,
        m_joint=m_joint,
        alpha=alpha.squeeze(-1),
        beta_hat=beta_hat.squeeze(-1),
        Kinv_H=Kinv_H,
        jitter=jitter,
    )


def predict_function(
    posterior: JointGPPosterior,
    x_test: torch.Tensor,
    *,
    H_test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict latent function values at x_test from the joint posterior."""
    m = x_test.shape[0]
    p = posterior.H_func.shape[1]
    if H_test is None:
        H_test = torch.zeros((m, p), dtype=posterior.H_func.dtype, device=posterior.H_func.device)

    xd_xf, xdd_xf = pairwise_differences(x_test, posterior.x_func)
    K_xf = se_kernel(xdd_xf, posterior.ell, posterior.w)

    xd_xd, xdd_xd = pairwise_differences(x_test, posterior.x_der)
    K_xd = fdse_kernel(xd_xd, xdd_xd, posterior.ell, posterior.w)

    K_xY = torch.cat([K_xf, K_xd], dim=1)
    pred_mean = (K_xY @ posterior.alpha.reshape(-1, 1)).squeeze(-1)
    pred_mean = pred_mean + (H_test @ posterior.beta_hat.reshape(-1, 1)).squeeze(-1)

    K_Yx = K_xY.T
    v = torch.linalg.solve_triangular(posterior.L, K_Yx, upper=False)
    xd_xx, xdd_xx = pairwise_differences(x_test, x_test)
    K_xx = se_kernel(xdd_xx, posterior.ell, posterior.w)
    K_xx = K_xx + posterior.jitter * torch.eye(m, dtype=K_xx.dtype, device=K_xx.device)

    term1 = K_xx - v.T @ v
    M = K_xY @ posterior.Kinv_H
    D = H_test - M
    S = 0.5 * ((posterior.H_full.T @ posterior.Kinv_H) + (posterior.H_full.T @ posterior.Kinv_H).T)
    S_inv = torch.linalg.inv(S)
    term2 = D @ (S_inv @ D.T)
    pred_cov = term1 + term2
    return pred_mean, pred_cov


def gpr_hd(
    x_func: torch.Tensor,
    y_func: torch.Tensor,
    x_der: torch.Tensor,
    dy_der: torch.Tensor,
    x_test: torch.Tensor,
    ell: torch.Tensor,
    w: torch.Tensor,
    noise_func_cov: torch.Tensor,
    noise_deriv_diag: torch.Tensor,
    H_func: torch.Tensor,
    H_test: torch.Tensor | None = None,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    posterior = build_joint_gp(
        x_func=x_func,
        y_func=y_func,
        x_der=x_der,
        dy_der=dy_der,
        ell=ell,
        w=w,
        noise_func_cov=noise_func_cov,
        noise_deriv_diag=noise_deriv_diag,
        H_func=H_func,
        jitter=jitter,
    )
    pred_mean, pred_cov = predict_function(posterior, x_test, H_test=H_test)
    return (
        pred_mean,
        pred_cov,
        posterior.K_joint,
        posterior.L,
        posterior.y_joint,
        posterior.m_joint,
        posterior.alpha,
    )


def precision_diag_from_cholesky(L: torch.Tensor) -> torch.Tensor:
    I = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
    W = torch.linalg.solve_triangular(L, I, upper=False)
    return (W**2).sum(dim=0)


def loo_loglik_from_alpha_and_Qii(
    y: torch.Tensor,
    m: torch.Tensor,
    alpha: torch.Tensor,
    Qii: torch.Tensor,
) -> torch.Tensor:
    sigma2 = 1.0 / Qii
    err = alpha / Qii
    logp_i = -0.5 * (torch.log(2 * np.pi * sigma2) + (err**2) / sigma2)
    return logp_i.sum()


def joint_log_marginal_likelihood(posterior: JointGPPosterior, *, include_constant: bool = True) -> torch.Tensor:
    r = posterior.y_joint - posterior.m_joint
    logdetK = 2.0 * torch.sum(torch.log(torch.diagonal(posterior.L)))
    quad = torch.dot(r, posterior.alpha)
    value = -0.5 * (quad + logdetK)
    if include_constant:
        value = value - 0.5 * r.numel() * np.log(2.0 * np.pi)
    return value


def joint_loo_loglik(posterior: JointGPPosterior) -> torch.Tensor:
    Qii = precision_diag_from_cholesky(posterior.L)
    return loo_loglik_from_alpha_and_Qii(
        posterior.y_joint,
        posterior.m_joint,
        posterior.alpha,
        Qii,
    )
