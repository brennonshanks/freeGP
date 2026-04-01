"""Stationary and nonstationary GP kernels for joint histogram+derivative inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GibbsKernelConfig:
    length_model: str = "exp_linear_bump"
    width_model: str = "tanh_decay"


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


def width(
    r: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    *,
    model: str = "tanh_decay",
) -> torch.Tensor:
    if model == "constant":
        return s * torch.ones_like(r)
    if model == "tanh_decay":
        return s * (1.0 + torch.tanh(-(r - u) / width_w))
    raise ValueError(f"Unsupported width model: {model}")


def width_deriv(
    r: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    *,
    model: str = "tanh_decay",
) -> torch.Tensor:
    if model == "constant":
        return torch.zeros_like(r)
    if model == "tanh_decay":
        return -(s / width_w) * (1.0 - torch.tanh(-(r - u) / width_w) ** 2)
    raise ValueError(f"Unsupported width model: {model}")


def length(
    r: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    *,
    model: str = "exp_linear_bump",
) -> torch.Tensor:
    if model == "constant":
        return torch.exp(a0) * torch.ones_like(r)
    if model == "exp_linear_bump":
        bump = torch.exp(-0.5 * ((r - c) / length_w) ** 2)
        g = a0 + a1 * r + b * bump
        return torch.exp(g)
    raise ValueError(f"Unsupported length model: {model}")


def length_deriv(
    r: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    *,
    model: str = "exp_linear_bump",
) -> torch.Tensor:
    if model == "constant":
        return torch.zeros_like(r)
    if model == "exp_linear_bump":
        bump = torch.exp(-0.5 * ((r - c) / length_w) ** 2)
        ell_val = torch.exp(a0 + a1 * r + b * bump)
        dbump_dr = -((r - c) / (length_w**2)) * bump
        dg_dr = a1 + b * dbump_dr
        return ell_val * dg_dr
    raise ValueError(f"Unsupported length model: {model}")


def gibbs_kernel(
    xdd: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    r: torch.Tensor,
    rp: torch.Tensor,
    *,
    config: GibbsKernelConfig | None = None,
) -> torch.Tensor:
    config = config or GibbsKernelConfig()
    ell = length(r, a0, a1, b, c, length_w, model=config.length_model)
    ellp = length(rp, a0, a1, b, c, length_w, model=config.length_model)
    sig = width(r, s, u, width_w, model=config.width_model)
    sigp = width(rp, s, u, width_w, model=config.width_model)
    return sig * sigp * torch.sqrt((2.0 * ell * ellp) / (ell**2 + ellp**2)) * torch.exp(
        -xdd / (ell**2 + ellp**2)
    )


def fdgibbs_kernel(
    xd: torch.Tensor,
    xdd: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    r: torch.Tensor,
    rp: torch.Tensor,
    *,
    config: GibbsKernelConfig | None = None,
) -> torch.Tensor:
    config = config or GibbsKernelConfig()
    ell = length(r, a0, a1, b, c, length_w, model=config.length_model)
    ellp = length(rp, a0, a1, b, c, length_w, model=config.length_model)
    ell_d = length_deriv(r, a0, a1, b, c, length_w, model=config.length_model)
    sig = width(r, s, u, width_w, model=config.width_model)
    sigp = width(rp, s, u, width_w, model=config.width_model)
    sig_d = width_deriv(r, s, u, width_w, model=config.width_model)
    B = ell**2 + ellp**2
    exp_term = torch.exp(-xdd / B)
    L_term = (2.0 * ell * ellp) / B
    Omg_term = (ellp * ell_d) / B * (1.0 - 2.0 * ell**2 / B)
    Gam_term = -2.0 * xd / B + (2.0 * xdd * ell * ell_d) / (B**2)
    return exp_term * sigp * (
        sig_d * torch.sqrt(L_term)
        + sig * torch.rsqrt(L_term) * Omg_term
        + sig * torch.sqrt(L_term) * Gam_term
    )


def sdgibbs_kernel(
    xd: torch.Tensor,
    xdd: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    r: torch.Tensor,
    rp: torch.Tensor,
    *,
    config: GibbsKernelConfig | None = None,
) -> torch.Tensor:
    config = config or GibbsKernelConfig()
    ell = length(r, a0, a1, b, c, length_w, model=config.length_model)
    ellp = length(rp, a0, a1, b, c, length_w, model=config.length_model)
    ellp_d = length_deriv(rp, a0, a1, b, c, length_w, model=config.length_model)
    sig = width(r, s, u, width_w, model=config.width_model)
    sigp = width(rp, s, u, width_w, model=config.width_model)
    sigp_d = width_deriv(rp, s, u, width_w, model=config.width_model)
    B = ell**2 + ellp**2
    exp_term = torch.exp(-xdd / B)
    L_term = (2.0 * ell * ellp) / B
    Omg_term = (ell * ellp_d) / B * (1.0 - 2.0 * ellp**2 / B)
    Gam_term = 2.0 * xd / B + (2.0 * xdd * ellp * ellp_d) / (B**2)
    return exp_term * sig * (
        sigp_d * torch.sqrt(L_term)
        + sigp * torch.rsqrt(L_term) * Omg_term
        + sigp * torch.sqrt(L_term) * Gam_term
    )


def ddgibbs_kernel(
    xd: torch.Tensor,
    xdd: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    r: torch.Tensor,
    rp: torch.Tensor,
    *,
    config: GibbsKernelConfig | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mixed derivative d^2 K / (dr dr') for the full 1D Gibbs kernel."""
    config = config or GibbsKernelConfig()
    ell = length(r, a0, a1, b, c, length_w, model=config.length_model)
    ellp = length(rp, a0, a1, b, c, length_w, model=config.length_model)
    ell_d = length_deriv(r, a0, a1, b, c, length_w, model=config.length_model)
    ellp_d = length_deriv(rp, a0, a1, b, c, length_w, model=config.length_model)
    sig = width(r, s, u, width_w, model=config.width_model)
    sigp = width(rp, s, u, width_w, model=config.width_model)
    sig_d = width_deriv(r, s, u, width_w, model=config.width_model)
    sigp_d = width_deriv(rp, s, u, width_w, model=config.width_model)

    B = ell**2 + ellp**2 + eps
    L_term = (2.0 * ell * ellp) / B
    L = torch.sqrt(L_term + eps)
    exp_term = torch.exp(-xdd / B)
    k = sig * sigp * L * exp_term

    inv_sig = 1.0 / (sig + eps)
    inv_sigp = 1.0 / (sigp + eps)
    inv_ell = 1.0 / (ell + eps)
    inv_ellp = 1.0 / (ellp + eps)

    A_sig = sig_d * inv_sig
    A_L = 0.5 * ell_d * inv_ell - (ell * ell_d) / B
    A_E = -(2.0 * xd) / B + (2.0 * xdd * ell * ell_d) / (B**2)
    A = A_sig + A_L + A_E

    Ap_sig = sigp_d * inv_sigp
    Ap_L = 0.5 * ellp_d * inv_ellp - (ellp * ellp_d) / B
    Ap_E = (2.0 * xd) / B + (2.0 * xdd * ellp * ellp_d) / (B**2)
    Ap = Ap_sig + Ap_L + Ap_E

    dB_drp = 2.0 * ellp * ellp_d
    dA_L_drp = (ell * ell_d) * dB_drp / (B**2)
    dA_E_drp = (
        2.0 / B
        + (2.0 * xd * dB_drp) / (B**2)
        - (4.0 * ell * ell_d * xd) / (B**2)
        - (4.0 * ell * ell_d * xdd * dB_drp) / (B**3)
    )
    dA_drp = dA_L_drp + dA_E_drp
    return k * (A * Ap + dA_drp)


@dataclass(frozen=True)
class JointGPPosterior:
    x_func: torch.Tensor
    y_func: torch.Tensor
    x_der: torch.Tensor
    dy_der: torch.Tensor
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
    kernel_name: str
    kernel_params: dict[str, torch.Tensor]


def _derivative_noise_matrix(noise_deriv_diag: torch.Tensor) -> torch.Tensor:
    if noise_deriv_diag.ndim == 2:
        return noise_deriv_diag
    return torch.diag(noise_deriv_diag)


def _finalize_joint_posterior(
    *,
    x_func: torch.Tensor,
    y_func: torch.Tensor,
    x_der: torch.Tensor,
    dy_der: torch.Tensor,
    noise_func_cov: torch.Tensor,
    noise_deriv_diag: torch.Tensor,
    H_func: torch.Tensor,
    K_joint: torch.Tensor,
    y_joint_col: torch.Tensor,
    jitter: float,
    kernel_name: str,
    kernel_params: dict[str, torch.Tensor],
) -> JointGPPosterior:
    n_d = x_der.shape[0]
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
        kernel_name=kernel_name,
        kernel_params=kernel_params,
    )


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
    """Build the profiled joint stationary GP posterior for function + derivative observations."""
    n_f = x_func.shape[0]
    n_d = x_der.shape[0]

    xd_ff, xdd_ff = pairwise_differences(x_func, x_func)
    K_ff = se_kernel(xdd_ff, ell, w) + noise_func_cov

    xd_dd, xdd_dd = pairwise_differences(x_der, x_der)
    K_dd = ddse_kernel(xd_dd, xdd_dd, ell, w) + _derivative_noise_matrix(noise_deriv_diag)

    xd_fd, xdd_fd = pairwise_differences(x_func, x_der)
    K_fd = fdse_kernel(xd_fd, xdd_fd, ell, w)
    K_df = K_fd.T

    top = torch.cat([K_ff, K_fd], dim=1)
    bot = torch.cat([K_df, K_dd], dim=1)
    K_joint = torch.cat([top, bot], dim=0)
    K_joint = 0.5 * (K_joint + K_joint.T)
    K_joint = K_joint + jitter * torch.eye(n_f + n_d, dtype=K_joint.dtype, device=K_joint.device)

    y_joint_col = torch.cat([y_func.reshape(-1, 1), dy_der.reshape(-1, 1)], dim=0)
    return _finalize_joint_posterior(
        x_func=x_func,
        y_func=y_func,
        x_der=x_der,
        dy_der=dy_der,
        noise_func_cov=noise_func_cov,
        noise_deriv_diag=noise_deriv_diag,
        H_func=H_func,
        K_joint=K_joint,
        y_joint_col=y_joint_col,
        jitter=jitter,
        kernel_name="stationary_se",
        kernel_params={"ell": ell, "w": w},
    )


def build_joint_gp_gibbs(
    *,
    x_func: torch.Tensor,
    y_func: torch.Tensor,
    x_der: torch.Tensor,
    dy_der: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    noise_func_cov: torch.Tensor,
    noise_deriv_diag: torch.Tensor,
    H_func: torch.Tensor,
    config: GibbsKernelConfig | None = None,
    jitter: float = 1e-8,
) -> JointGPPosterior:
    """Build the profiled joint nonstationary Gibbs GP posterior."""
    config = config or GibbsKernelConfig()
    n_f = x_func.shape[0]
    n_d = x_der.shape[0]

    xd_ff, xdd_ff = pairwise_differences(x_func, x_func)
    r_ff = x_func[:, None]
    rp_ff = x_func[None, :]
    K_ff = gibbs_kernel(
        xdd_ff, a0, a1, b, c, length_w, s, u, width_w, r_ff, rp_ff, config=config
    )
    K_ff = K_ff + noise_func_cov

    xd_dd, xdd_dd = pairwise_differences(x_der, x_der)
    r_dd = x_der[:, None]
    rp_dd = x_der[None, :]
    K_dd = ddgibbs_kernel(
        xd_dd, xdd_dd, a0, a1, b, c, length_w, s, u, width_w, r_dd, rp_dd, config=config
    )
    K_dd = K_dd + _derivative_noise_matrix(noise_deriv_diag)

    xd_fd, xdd_fd = pairwise_differences(x_func, x_der)
    r_fd = x_func[:, None]
    rp_fd = x_der[None, :]
    K_fd = sdgibbs_kernel(
        xd_fd, xdd_fd, a0, a1, b, c, length_w, s, u, width_w, r_fd, rp_fd, config=config
    )

    xd_df, xdd_df = pairwise_differences(x_der, x_func)
    r_df = x_der[:, None]
    rp_df = x_func[None, :]
    K_df = fdgibbs_kernel(
        xd_df, xdd_df, a0, a1, b, c, length_w, s, u, width_w, r_df, rp_df, config=config
    )

    top = torch.cat([K_ff, K_fd], dim=1)
    bot = torch.cat([K_df, K_dd], dim=1)
    K_joint = torch.cat([top, bot], dim=0)
    K_joint = 0.5 * (K_joint + K_joint.T)
    K_joint = K_joint + jitter * torch.eye(n_f + n_d, dtype=K_joint.dtype, device=K_joint.device)

    y_joint_col = torch.cat([y_func.reshape(-1, 1), dy_der.reshape(-1, 1)], dim=0)
    return _finalize_joint_posterior(
        x_func=x_func,
        y_func=y_func,
        x_der=x_der,
        dy_der=dy_der,
        noise_func_cov=noise_func_cov,
        noise_deriv_diag=noise_deriv_diag,
        H_func=H_func,
        K_joint=K_joint,
        y_joint_col=y_joint_col,
        jitter=jitter,
        kernel_name="gibbs",
        kernel_params={
            "a0": a0,
            "a1": a1,
            "b": b,
            "c": c,
            "length_w": length_w,
            "s": s,
            "u": u,
            "width_w": width_w,
            "length_model": config.length_model,
            "width_model": config.width_model,
        },
    )


def _default_H_test(posterior: JointGPPosterior, x_test: torch.Tensor, H_test: torch.Tensor | None) -> torch.Tensor:
    if H_test is not None:
        return H_test
    m = x_test.shape[0]
    p = posterior.H_func.shape[1]
    return torch.zeros((m, p), dtype=posterior.H_func.dtype, device=posterior.H_func.device)


def _predict_stationary(
    posterior: JointGPPosterior,
    x_test: torch.Tensor,
    *,
    H_test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    H_test = _default_H_test(posterior, x_test, H_test)
    ell = posterior.kernel_params["ell"]
    w = posterior.kernel_params["w"]

    xd_xf, xdd_xf = pairwise_differences(x_test, posterior.x_func)
    K_xf = se_kernel(xdd_xf, ell, w)

    xd_xd, xdd_xd = pairwise_differences(x_test, posterior.x_der)
    K_xd = fdse_kernel(xd_xd, xdd_xd, ell, w)

    K_xY = torch.cat([K_xf, K_xd], dim=1)
    pred_mean = (K_xY @ posterior.alpha.reshape(-1, 1)).squeeze(-1)
    pred_mean = pred_mean + (H_test @ posterior.beta_hat.reshape(-1, 1)).squeeze(-1)

    v = torch.linalg.solve_triangular(posterior.L, K_xY.T, upper=False)
    xd_xx, xdd_xx = pairwise_differences(x_test, x_test)
    K_xx = se_kernel(xdd_xx, ell, w)

    term1 = K_xx - v.T @ v
    M = K_xY @ posterior.Kinv_H
    D = H_test - M
    S = 0.5 * ((posterior.H_full.T @ posterior.Kinv_H) + (posterior.H_full.T @ posterior.Kinv_H).T)
    S_inv = torch.linalg.inv(S)
    term2 = D @ (S_inv @ D.T)
    pred_cov = term1 + term2
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    return pred_mean, pred_cov


def _predict_gibbs(
    posterior: JointGPPosterior,
    x_test: torch.Tensor,
    *,
    H_test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    H_test = _default_H_test(posterior, x_test, H_test)
    params = posterior.kernel_params
    config = GibbsKernelConfig(
        length_model=params["length_model"],
        width_model=params["width_model"],
    )
    a0 = params["a0"]
    a1 = params["a1"]
    b = params["b"]
    c = params["c"]
    length_w = params["length_w"]
    s = params["s"]
    u = params["u"]
    width_w = params["width_w"]

    xd_xf, xdd_xf = pairwise_differences(x_test, posterior.x_func)
    r_xf = x_test[:, None]
    rp_xf = posterior.x_func[None, :]
    K_xf = gibbs_kernel(
        xdd_xf, a0, a1, b, c, length_w, s, u, width_w, r_xf, rp_xf, config=config
    )

    xd_xd, xdd_xd = pairwise_differences(x_test, posterior.x_der)
    r_xd = x_test[:, None]
    rp_xd = posterior.x_der[None, :]
    K_xd = sdgibbs_kernel(
        xd_xd, xdd_xd, a0, a1, b, c, length_w, s, u, width_w, r_xd, rp_xd, config=config
    )
    K_xY = torch.cat([K_xf, K_xd], dim=1)

    xd_dx, xdd_dx = pairwise_differences(posterior.x_der, x_test)
    r_dx = posterior.x_der[:, None]
    rp_dx = x_test[None, :]
    K_dx = fdgibbs_kernel(
        xd_dx, xdd_dx, a0, a1, b, c, length_w, s, u, width_w, r_dx, rp_dx, config=config
    )
    K_Yx = torch.cat([K_xf.T, K_dx], dim=0)

    pred_mean = (K_xY @ posterior.alpha.reshape(-1, 1)).squeeze(-1)
    pred_mean = pred_mean + (H_test @ posterior.beta_hat.reshape(-1, 1)).squeeze(-1)

    v = torch.linalg.solve_triangular(posterior.L, K_Yx, upper=False)
    xd_xx, xdd_xx = pairwise_differences(x_test, x_test)
    r_xx = x_test[:, None]
    rp_xx = x_test[None, :]
    K_xx = gibbs_kernel(
        xdd_xx, a0, a1, b, c, length_w, s, u, width_w, r_xx, rp_xx, config=config
    )

    term1 = K_xx - v.T @ v
    M = K_xY @ posterior.Kinv_H
    D = H_test - M
    S = 0.5 * ((posterior.H_full.T @ posterior.Kinv_H) + (posterior.H_full.T @ posterior.Kinv_H).T)
    S_inv = torch.linalg.inv(S)
    term2 = D @ (S_inv @ D.T)
    pred_cov = term1 + term2
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    return pred_mean, pred_cov


def predict_function(
    posterior: JointGPPosterior,
    x_test: torch.Tensor,
    *,
    H_test: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict latent function values at x_test from the joint posterior."""
    if posterior.kernel_name == "stationary_se":
        return _predict_stationary(posterior, x_test, H_test=H_test)
    if posterior.kernel_name == "gibbs":
        return _predict_gibbs(posterior, x_test, H_test=H_test)
    raise ValueError(f"Unsupported posterior kernel: {posterior.kernel_name}")


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


def gpr_hd_gibbs(
    x_func: torch.Tensor,
    y_func: torch.Tensor,
    x_der: torch.Tensor,
    dy_der: torch.Tensor,
    x_test: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_w: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    width_w: torch.Tensor,
    noise_func_cov: torch.Tensor,
    noise_deriv_diag: torch.Tensor,
    H_func: torch.Tensor,
    H_test: torch.Tensor | None = None,
    config: GibbsKernelConfig | None = None,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    posterior = build_joint_gp_gibbs(
        x_func=x_func,
        y_func=y_func,
        x_der=x_der,
        dy_der=dy_der,
        a0=a0,
        a1=a1,
        b=b,
        c=c,
        length_w=length_w,
        s=s,
        u=u,
        width_w=width_w,
        noise_func_cov=noise_func_cov,
        noise_deriv_diag=noise_deriv_diag,
        H_func=H_func,
        config=config,
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


def projected_precision_diag(posterior: JointGPPosterior) -> torch.Tensor:
    """Diagonal of the profiled precision P = K^-1 - K^-1 H (H^T K^-1 H)^-1 H^T K^-1."""
    Kinv_diag = precision_diag_from_cholesky(posterior.L)
    S = 0.5 * ((posterior.H_full.T @ posterior.Kinv_H) + (posterior.H_full.T @ posterior.Kinv_H).T)
    S_inv = torch.linalg.inv(S)
    low_rank_diag = torch.sum((posterior.Kinv_H @ S_inv) * posterior.Kinv_H, dim=1)
    return Kinv_diag - low_rank_diag


def loo_loglik_from_alpha_and_precision_diag(alpha: torch.Tensor, precision_diag: torch.Tensor) -> torch.Tensor:
    sigma2 = 1.0 / precision_diag
    err = alpha / precision_diag
    logp_i = -0.5 * (torch.log(2 * np.pi * sigma2) + (err**2) / sigma2)
    return logp_i.sum()


def joint_log_marginal_likelihood(posterior: JointGPPosterior, *, include_constant: bool = True) -> torch.Tensor:
    """Marginal log likelihood with linear mean coefficients integrated out under a flat prior."""
    r = posterior.y_joint - posterior.m_joint
    logdetK = 2.0 * torch.sum(torch.log(torch.diagonal(posterior.L)))
    quad = torch.dot(r, posterior.alpha)
    p = posterior.H_full.shape[1]
    if p > 0:
        S = 0.5 * ((posterior.H_full.T @ posterior.Kinv_H) + (posterior.H_full.T @ posterior.Kinv_H).T)
        Ls = torch.linalg.cholesky(S)
        logdetS = 2.0 * torch.sum(torch.log(torch.diagonal(Ls)))
    else:
        logdetS = torch.tensor(0.0, dtype=r.dtype, device=r.device)

    value = -0.5 * (quad + logdetK + logdetS)
    if include_constant:
        value = value - 0.5 * (r.numel() - p) * np.log(2.0 * np.pi)
    return value


def joint_loo_loglik(posterior: JointGPPosterior) -> torch.Tensor:
    Pii = projected_precision_diag(posterior)
    return loo_loglik_from_alpha_and_precision_diag(posterior.alpha, Pii)
