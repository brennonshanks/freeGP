#!/usr/bin/env python3
"""CLI runner for the extracted GPR(H+D) HMC-NUTS workflow."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import pickle
import sys
import time
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.config import resolve_device
    from freegp.gp import GibbsKernelConfig, gpr_hd, gpr_hd_gibbs
    from freegp.hmc import (
        NUTSConfig,
        display_samples_for_diagnostics,
        maximum_a_posteriori_prediction,
        run_hmc_nuts,
        sample_posterior_functions,
        summarize_chain_diagnostics,
    )
    from freegp.posterior import summarize_hyperposterior_predictive
    from freegp.workflow import move_workflow_bundle, prepare_gprhd_hmc_inputs
else:
    from .config import resolve_device
    from .gp import GibbsKernelConfig, gpr_hd, gpr_hd_gibbs
    from .hmc import (
        NUTSConfig,
        display_samples_for_diagnostics,
        maximum_a_posteriori_prediction,
        run_hmc_nuts,
        sample_posterior_functions,
        summarize_chain_diagnostics,
    )
    from .posterior import summarize_hyperposterior_predictive
    from .workflow import move_workflow_bundle, prepare_gprhd_hmc_inputs


def _parse_device_list(value: str) -> list[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _normalize_config_value(key: str, value):
    if key == "compare_devices" and isinstance(value, list):
        return ",".join(str(piece) for piece in value)
    return value


def _load_config_defaults(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        loaded = tomllib.load(handle)

    if "single_run" in loaded and isinstance(loaded["single_run"], dict):
        loaded = loaded["single_run"]

    key_aliases = {
        "results_dir": "figure_dir",
        "figure_dir": "figure_dir",
        "method": "mode",
        "random_seed": "seed",
    }
    defaults = {
        key_aliases.get(key.replace("-", "_"), key.replace("-", "_")):
        _normalize_config_value(key_aliases.get(key.replace("-", "_"), key.replace("-", "_")), value)
        for key, value in loaded.items()
    }
    defaults["config"] = str(config_path)
    return defaults


def build_parser(*, defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the extracted GPR(H+D) workflow from the old HMC-NUTS notebook."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to freeGP-datasets or directly to GPs_umbrellas_Katka. "
        "If omitted, FREEGP_DATASETS is used.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to the freeGP-dev project root. Defaults to this installed package location.",
    )
    parser.add_argument(
        "--reference-wham-path",
        type=str,
        default=None,
        help="Path to a reference WHAM PMF file (2-column: x, F). Plotted alongside GP output.",
    )
    parser.add_argument(
        "--reference-wham-x-units",
        type=str,
        default="nm",
        help="X-axis units of the WHAM PMF file: 'nm' or 'angstrom'.",
    )
    parser.add_argument(
        "--reference-ui-path",
        type=str,
        default=None,
        help="Path to a reference UI PMF file (2-column: x, F). Plotted alongside GP output.",
    )
    parser.add_argument(
        "--reference-ui-x-units",
        type=str,
        default="nm",
        help="X-axis units of the UI PMF file: 'nm' or 'angstrom'.",
    )
    parser.add_argument(
        "--pmf-alignment",
        choices=("max", "min"),
        default="max",
        help="Align shifted PMF curves at their maximum ('max', default) or minimum ('min').",
    )
    parser.add_argument("--n-equilibration", type=int, default=40_000)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--num-test-points", type=int, default=400)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument(
        "--test-grid-source",
        choices=("umbrella_centers", "histogram_support"),
        default="umbrella_centers",
        help="Default x-grid range when x-min/x-max are not provided.",
    )
    parser.add_argument(
        "--mode",
        choices=("gp", "nuts"),
        default="gp",
        help="Run a deterministic GP prediction or the HMC-NUTS hyperparameter sampler.",
    )
    parser.add_argument(
        "--kernel",
        choices=("stationary", "gibbs"),
        default="stationary",
        help="Kernel family used in GP mode or inside NUTS.",
    )
    parser.add_argument(
        "--length-model",
        choices=("exp_linear_bump", "constant"),
        default="exp_linear_bump",
        help="Length-scale function used by the Gibbs kernel.",
    )
    parser.add_argument(
        "--width-model",
        choices=("tanh_decay", "constant"),
        default="tanh_decay",
        help="Amplitude/width envelope used by the Gibbs kernel.",
    )
    parser.add_argument(
        "--ell",
        type=float,
        default=float(np.pi / 2.0),
        help="Initial/test GP length scale. Defaults to the fixed Csanyi-style baseline.",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=float(4.184 * np.sqrt(10.0)),
        help="Initial/test GP amplitude. Defaults to the fixed Csanyi-style baseline in kJ/mol.",
    )
    parser.add_argument("--a0", type=float, default=np.log(4.0), help="Gibbs log-length baseline.")
    parser.add_argument("--a1", type=float, default=0.0, help="Gibbs linear trend in log length.")
    parser.add_argument("--b", type=float, default=0.0, help="Gibbs bump amplitude in log length.")
    parser.add_argument("--c", type=float, default=None, help="Gibbs bump center. Defaults to data midpoint.")
    parser.add_argument(
        "--length-w",
        type=float,
        default=0.5,
        help="Gibbs bump width parameter for the length function.",
    )
    parser.add_argument("--s", type=float, default=1.65, help="Gibbs width/amplitude scale.")
    parser.add_argument("--u", type=float, default=None, help="Gibbs width-envelope center. Defaults to data midpoint.")
    parser.add_argument(
        "--w2",
        type=float,
        default=0.5,
        help="Gibbs width-envelope width parameter.",
    )
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for torch/Pyro sampling.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for GP and NUTS tensor computations.",
    )
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument(
        "--objective",
        choices=("lml", "loo", "both"),
        default="lml",
        help="Objective used inside NUTS. 'both' runs lml and loo sequentially into subdirectories.",
    )
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=50,
        help="If > 0 in nuts mode, draw this many posterior functions on the test grid.",
    )
    parser.add_argument(
        "--predictive-samples",
        type=int,
        default=100,
        help="Maximum retained hyperposterior samples used in the total-variance predictive summary.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional pickle output path for results.",
    )
    parser.add_argument(
        "--results-dir",
        "--figure-dir",
        dest="figure_dir",
        type=str,
        default=None,
        help="Directory for diagnostic outputs. Defaults to ./results/<timestamped-run>/",
    )
    parser.add_argument(
        "--compare-devices",
        type=str,
        default=None,
        help="Comma-separated devices to benchmark with the same run configuration, e.g. cpu,cuda.",
    )
    parser.add_argument(
        "--benchmark-cpu-num-chains",
        type=int,
        default=None,
        help="If set during --compare-devices, override CPU runs to use this many Pyro chains.",
    )
    parser.add_argument(
        "--benchmark-gpu-runs",
        type=int,
        default=1,
        help="If set during --compare-devices, run this many sequential single-chain GPU runs.",
    )
    parser.add_argument(
        "--no-corner",
        action="store_true",
        help="Disable the corner plot in NUTS mode.",
    )
    if defaults:
        known_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known_dests)
        if unknown:
            raise ValueError(f"Unsupported config keys: {unknown}")
        parser.set_defaults(**defaults)
    return parser


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prepare_figure_dir(path: str | None, mode: str, *, project_root: str | None = None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if project_root is None:
            repo_root = Path(__file__).resolve().parents[2]
        else:
            repo_root = Path(project_root).expanduser().resolve()
        root = repo_root / "results" / f"gprhd-{mode}-{stamp}"
    else:
        root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def plot_histograms(bundle, figure_dir: Path) -> None:
    processed = bundle.processed
    plt.figure(figsize=(10, 6))
    for center, density, folder_number in zip(
        processed.bin_centers_list,
        processed.histogram_densities,
        processed.folder_numbers,
    ):
        plt.plot(_to_numpy(center), _to_numpy(density), label=f"d_{folder_number.item():.2f}")
    plt.xlabel("Position [nm]")
    plt.ylabel("Probability density")
    plt.title("Umbrella Histograms After Equilibration Cut")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(figure_dir / "histograms.png", dpi=200)
    plt.close()


def plot_unbiased_windows(bundle, figure_dir: Path) -> None:
    processed = bundle.processed
    F_list = bundle.observations.F_list
    plt.figure(figsize=(10, 6))
    for i, (x, free_energy) in enumerate(zip(processed.bin_centers_list, F_list)):
        plt.plot(_to_numpy(x), _to_numpy(free_energy), label=f"window {i}")
    plt.xlabel("Position [nm]")
    plt.ylabel("Free energy [kJ/mol]")
    plt.title("Unbiased Per-Window Free Energy Curves")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(figure_dir / "unbiased_windows.png", dpi=200)
    plt.close()


def plot_gp_posterior(
    bundle,
    pred_mean,
    pred_cov,
    figure_dir: Path,
    *,
    filename: str,
    pmf_alignment: str = "max",
) -> None:
    x_test = _to_numpy(bundle.x_test).ravel()
    pred_mean_np = _to_numpy(pred_mean).ravel()
    pred_std_np = np.sqrt(np.clip(np.diag(_to_numpy(pred_cov)), a_min=0.0, a_max=None))
    anchor = np.max(pred_mean_np) if pmf_alignment == "max" else np.min(pred_mean_np)
    shifted_mean = pred_mean_np - anchor

    refs = bundle.references
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, shifted_mean, lw=2, color="royalblue", label="Posterior mean")
    plt.fill_between(
        x_test,
        shifted_mean - 2 * pred_std_np,
        shifted_mean + 2 * pred_std_np,
        alpha=0.25,
        color="royalblue",
        label="±2σ",
    )
    if refs.has_wham:
        ref_anchor = np.max(refs.wham_f) if pmf_alignment == "max" else np.min(refs.wham_f)
        wham_shift = refs.wham_f - ref_anchor
        plt.errorbar(refs.wham_x, wham_shift, yerr=refs.wham_e, capsize=3, color="crimson", alpha=0.5, label="WHAM")
    if refs.has_ui:
        ref_anchor = np.max(refs.umbrella_f) if pmf_alignment == "max" else np.min(refs.umbrella_f)
        umbrella_shift = refs.umbrella_f - ref_anchor
        plt.errorbar(
            refs.umbrella_x,
            umbrella_shift,
            yerr=refs.umbrella_e,
            capsize=3,
            color="steelblue",
            alpha=0.5,
            label="UI (Semen)",
        )
    plt.xlabel("Position [nm]")
    plt.ylabel("Free Energy [kJ/mol]")
    plt.title("GPR(H+D) Posterior")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=200)
    plt.close()


def plot_matrix(matrix, figure_dir: Path, *, filename: str, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(_to_numpy(matrix), aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=200)
    plt.close()


def plot_nuts_traces(samples: dict[str, torch.Tensor], figure_dir: Path) -> None:
    names = list(samples.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 3 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        values = _to_numpy(samples[name])
        if values.ndim == 1:
            chains = values.reshape(1, -1)
        else:
            chains = values.reshape(values.shape[0], values.shape[1], -1)[:, :, 0]
        for chain_idx, chain_values in enumerate(chains):
            label = f"chain {chain_idx}" if chains.shape[0] > 1 else None
            axes[row, 0].plot(chain_values, lw=1.0, alpha=0.9, label=label)
        axes[row, 0].set_title(f"{name} trace")
        for chain_idx, chain_values in enumerate(chains):
            label = f"chain {chain_idx}" if chains.shape[0] > 1 else None
            axes[row, 1].hist(chain_values, bins=30, alpha=0.45, label=label)
        axes[row, 1].set_title(f"{name} histogram")
        if chains.shape[0] > 1:
            axes[row, 0].legend(fontsize=8)
            axes[row, 1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "nuts_traces.png", dpi=200)
    plt.close(fig)


def plot_corner(samples: dict[str, torch.Tensor], figure_dir: Path, *, config: NUTSConfig) -> bool:
    try:
        import corner
    except ImportError:
        print("corner is not installed; skipping corner plot.")
        return False

    chain, labels = display_samples_for_diagnostics(samples, config=config)
    chain = chain.detach().cpu().numpy()
    if chain.ndim != 2 or chain.shape[0] <= chain.shape[1]:
        print(
            "Skipping corner plot because there are too few retained samples "
            f"({chain.shape[0]}) for the number of plotted parameters ({chain.shape[1]})."
        )
        return False
    figure = corner.corner(
        chain,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    figure.savefig(figure_dir / "corner.png", dpi=200)
    plt.close(figure)
    return True


def plot_posterior_draws(bundle, pred_means, function_draws, figure_dir: Path) -> None:
    x_test = _to_numpy(bundle.x_test).ravel()
    pred_means_np = _to_numpy(pred_means)
    function_draws_np = _to_numpy(function_draws)
    mean_marg = pred_means_np.mean(axis=0)
    var_marg = pred_means_np.var(axis=0) + function_draws_np.var(axis=0)
    std_marg = np.sqrt(np.clip(var_marg, a_min=0.0, a_max=None))

    plt.figure(figsize=(10, 6))
    for draw in function_draws_np[: min(100, len(function_draws_np))]:
        plt.plot(x_test, draw, color="black", alpha=0.15, lw=1)
    plt.plot(x_test, mean_marg, color="crimson", lw=2, label="Hyperposterior mean")
    plt.fill_between(
        x_test,
        mean_marg - 2 * std_marg,
        mean_marg + 2 * std_marg,
        color="gray",
        alpha=0.3,
        label="±2σ",
    )
    plt.xlabel("Position [nm]")
    plt.ylabel("Free Energy [kJ/mol]")
    plt.title("Posterior Function Draws")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "posterior_draws.png", dpi=200)
    plt.close()


def plot_variance_decomposition(bundle, summary, figure_dir: Path) -> None:
    x_test = _to_numpy(bundle.x_test).ravel()
    total_var = _to_numpy(summary.total_variance).ravel()
    within_var = _to_numpy(summary.within_variance).ravel()
    between_var = _to_numpy(summary.between_variance).ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, total_var, lw=2, color="black", label="Total variance")
    plt.plot(x_test, within_var, lw=2, color="royalblue", label="E_theta[Var(f|D,theta)]")
    plt.plot(x_test, between_var, lw=2, color="crimson", label="Var_theta(E[f|D,theta])")
    plt.xlabel("Position [nm]")
    plt.ylabel("Predictive variance")
    plt.title("Hyperposterior Variance Decomposition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "nuts_variance_decomposition.png", dpi=200)
    plt.close()


def write_run_summary(figure_dir: Path, lines: list[str]) -> None:
    (figure_dir / "run_summary.txt").write_text("\n".join(lines) + "\n")


def _diagnostic_value_to_python(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            scalar = float(value.detach().cpu().item())
            return None if not math.isfinite(scalar) else scalar
        return _diagnostic_value_to_python(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _diagnostic_value_to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): _diagnostic_value_to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value_to_python(item) for item in value]
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return None if not math.isfinite(scalar) else scalar
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _flatten_numeric_values(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: list[float] = []
        for item in value.values():
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    if isinstance(value, (int, float)):
        scalar = float(value)
        return [scalar] if math.isfinite(scalar) else []
    return []


def _divergence_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_divergence_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return int(value)


def extract_multi_chain_diagnostics(mcmc) -> dict[str, object]:
    diagnostics = _diagnostic_value_to_python(mcmc.diagnostics())
    parameter_diags = {
        key: value for key, value in diagnostics.items()
        if isinstance(value, dict) and "r_hat" in value and "n_eff" in value
    }
    r_hats = []
    n_effs = []
    for value in parameter_diags.values():
        r_hats.extend(_flatten_numeric_values(value.get("r_hat")))
        n_effs.extend(_flatten_numeric_values(value.get("n_eff")))

    divergence_map = diagnostics.get("divergences", {})
    divergence_total = _divergence_count(divergence_map)

    summary = {
        "max_r_hat": max(r_hats) if r_hats else None,
        "min_r_hat": min(r_hats) if r_hats else None,
        "min_n_eff": min(n_effs) if n_effs else None,
        "max_n_eff": max(n_effs) if n_effs else None,
        "divergence_total": divergence_total,
        "num_diagnostic_parameters": len(parameter_diags),
    }
    return {
        "raw": diagnostics,
        "summary": summary,
    }


def write_chain_diagnostics(figure_dir: Path, diagnostics: dict[str, object]) -> None:
    (figure_dir / "chain_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def _data_midpoint(bundle) -> float:
    x_all = torch.cat([bundle.observations.x_obs.reshape(-1), bundle.observations.x_der.reshape(-1)])
    return float(0.5 * (x_all.min().item() + x_all.max().item()))


def _format_param_mapping(params: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in params.items()}


def _kernel_summary_lines(args: argparse.Namespace) -> list[str]:
    lines = [f"kernel: {args.kernel}"]
    if args.kernel == "stationary":
        lines.extend(
            [
                f"ell: {args.ell}",
                f"w: {args.w}",
            ]
        )
    else:
        lines.extend(
            [
                f"length_model: {args.length_model}",
                f"width_model: {args.width_model}",
                f"a0: {args.a0}",
                f"a1: {args.a1}",
                f"b: {args.b}",
                f"c: {args.c}",
                f"length_w: {args.length_w}",
                f"s: {args.s}",
                f"u: {args.u}",
                f"w2: {args.w2}",
            ]
        )
    return lines


def run_gp_mode(args: argparse.Namespace, bundle, figure_dir: Path):
    obs = bundle.observations
    if args.kernel == "stationary":
        pred_mean, pred_cov, K_joint, L, y_joint, m_joint, alpha = gpr_hd(
            x_func=obs.x_obs,
            y_func=obs.y_obs,
            x_der=obs.x_der,
            dy_der=obs.dy_der,
            x_test=bundle.x_test,
            ell=torch.tensor(args.ell, dtype=torch.float64),
            w=torch.tensor(args.w, dtype=torch.float64),
            noise_func_cov=obs.noise_func_cov,
            noise_deriv_diag=obs.noise_deriv_diag,
            H_func=obs.H_obs,
            H_test=None,
            jitter=args.jitter,
        )
    else:
        midpoint = _data_midpoint(bundle)
        pred_mean, pred_cov, K_joint, L, y_joint, m_joint, alpha = gpr_hd_gibbs(
            x_func=obs.x_obs,
            y_func=obs.y_obs,
            x_der=obs.x_der,
            dy_der=obs.dy_der,
            x_test=bundle.x_test,
            a0=torch.tensor(args.a0, dtype=torch.float64),
            a1=torch.tensor(args.a1, dtype=torch.float64),
            b=torch.tensor(args.b, dtype=torch.float64),
            c=torch.tensor(args.c if args.c is not None else midpoint, dtype=torch.float64),
            length_w=torch.tensor(args.length_w, dtype=torch.float64),
            s=torch.tensor(args.s, dtype=torch.float64),
            u=torch.tensor(args.u if args.u is not None else midpoint, dtype=torch.float64),
            width_w=torch.tensor(args.w2, dtype=torch.float64),
            noise_func_cov=obs.noise_func_cov,
            noise_deriv_diag=obs.noise_deriv_diag,
            H_func=obs.H_obs,
            H_test=None,
            config=GibbsKernelConfig(
                length_model=args.length_model,
                width_model=args.width_model,
            ),
            jitter=args.jitter,
        )
    result = {
        "mode": "gp",
        "kernel": args.kernel,
        "dataset_root": str(bundle.dataset_root),
        "x_test": bundle.x_test,
        "pred_mean": pred_mean,
        "pred_cov": pred_cov,
        "K_joint": K_joint,
        "L": L,
        "y_joint": y_joint,
        "m_joint": m_joint,
        "alpha": alpha,
        "figure_dir": str(figure_dir),
    }
    plot_histograms(bundle, figure_dir)
    plot_unbiased_windows(bundle, figure_dir)
    plot_gp_posterior(bundle, pred_mean, pred_cov, figure_dir, filename="gp_posterior.png", pmf_alignment=args.pmf_alignment)
    plot_matrix(K_joint, figure_dir, filename="joint_covariance.png", title="Joint Covariance Matrix")
    plot_matrix(pred_cov, figure_dir, filename="predictive_covariance.png", title="Predictive Covariance Matrix")
    write_run_summary(
        figure_dir,
        [
            "mode: gp",
            f"device: {args.device}",
            f"seed: {args.seed}",
            f"num_chains: {args.num_chains}",
            *_kernel_summary_lines(args),
            f"dataset_root: {bundle.dataset_root}",
            f"x_obs shape: {tuple(obs.x_obs.shape)}",
            f"x_der shape: {tuple(obs.x_der.shape)}",
            f"x_test shape: {tuple(bundle.x_test.shape)}",
            f"x_test range: ({bundle.x_test.min().item():.6g}, {bundle.x_test.max().item():.6g})",
            f"pred_mean shape: {tuple(pred_mean.shape)}",
            f"pred_cov shape: {tuple(pred_cov.shape)}",
            f"predictive variance min: {torch.diagonal(pred_cov).min().item():.6g}",
        ],
    )
    print("Prepared workflow bundle")
    print(f"dataset_root: {bundle.dataset_root}")
    print(f"figure_dir: {figure_dir}")
    print(f"x_obs shape: {tuple(obs.x_obs.shape)}")
    print(f"x_der shape: {tuple(obs.x_der.shape)}")
    print(f"x_test shape: {tuple(bundle.x_test.shape)}")
    print(f"pred_mean shape: {tuple(pred_mean.shape)}")
    print(f"pred_cov shape: {tuple(pred_cov.shape)}")
    print(f"predictive variance min: {torch.diagonal(pred_cov).min().item():.6g}")
    return result


def run_nuts_mode(args: argparse.Namespace, bundle, figure_dir: Path):
    config = NUTSConfig(
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        num_chains=args.num_chains,
        target_accept_prob=args.target_accept_prob,
        jitter=args.jitter,
        objective=args.objective,
        kernel=args.kernel,
        length_model=args.length_model,
        width_model=args.width_model,
    )
    mcmc, samples = run_hmc_nuts(bundle.observations, config=config)
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    try:
        summary = mcmc.summary(prob=0.9)
        summary_note = "pyro summary computed"
    except AssertionError as exc:
        summary = None
        summary_note = f"pyro summary skipped: {exc}"
    chain_diagnostics = extract_multi_chain_diagnostics(mcmc)
    single_chain_diagnostics = summarize_chain_diagnostics(mcmc, samples, config=config)
    result = {
        "mode": "nuts",
        "kernel": args.kernel,
        "dataset_root": str(bundle.dataset_root),
        "samples": samples,
        "grouped_samples": grouped_samples,
        "summary": summary,
        "chain_diagnostics": chain_diagnostics,
        "single_chain_diagnostics": {
            "step_size": single_chain_diagnostics.step_size,
            "mean_accept_prob": single_chain_diagnostics.mean_accept_prob,
            "accept_count": single_chain_diagnostics.accept_count,
            "divergence_count": single_chain_diagnostics.divergence_count,
            "sample_std_by_name": single_chain_diagnostics.sample_std_by_name,
            "mean_sample_std": single_chain_diagnostics.mean_sample_std,
            "max_sample_std": single_chain_diagnostics.max_sample_std,
            "min_sample_std": single_chain_diagnostics.min_sample_std,
            "poor_acceptance": single_chain_diagnostics.poor_acceptance,
            "looks_stuck": single_chain_diagnostics.looks_stuck,
        },
        "figure_dir": str(figure_dir),
    }
    plot_histograms(bundle, figure_dir)
    plot_unbiased_windows(bundle, figure_dir)
    plot_nuts_traces(grouped_samples, figure_dir)
    corner_written = False
    if not args.no_corner:
        corner_written = plot_corner(samples, figure_dir, config=config)
    write_chain_diagnostics(figure_dir, chain_diagnostics)
    summary_lines = [
        "mode: nuts",
        f"device: {args.device}",
        f"seed: {args.seed}",
        f"dataset_root: {bundle.dataset_root}",
        f"figure_dir: {figure_dir}",
        f"objective: {args.objective}",
        *_kernel_summary_lines(args),
        f"num_samples: {args.num_samples}",
        f"warmup_steps: {args.warmup_steps}",
        f"num_chains: {args.num_chains}",
        f"x_test range: ({bundle.x_test.min().item():.6g}, {bundle.x_test.max().item():.6g})",
        summary_note,
        f"chain diagnostics file: {figure_dir / 'chain_diagnostics.json'}",
        f"max r_hat: {chain_diagnostics['summary']['max_r_hat']}",
        f"min n_eff: {chain_diagnostics['summary']['min_n_eff']}",
        f"total divergences: {chain_diagnostics['summary']['divergence_total']}",
        f"step_size: {single_chain_diagnostics.step_size}",
        f"mean_accept_prob: {single_chain_diagnostics.mean_accept_prob}",
        f"mean_sample_std: {single_chain_diagnostics.mean_sample_std}",
        f"looks_stuck: {single_chain_diagnostics.looks_stuck}",
    ]
    print("Finished NUTS run")
    print(f"dataset_root: {bundle.dataset_root}")
    print(f"figure_dir: {figure_dir}")
    for key, value in samples.items():
        print(f"{key}: {tuple(value.shape)}")
        summary_lines.append(f"{key}: {tuple(value.shape)}")
    for key, value in grouped_samples.items():
        summary_lines.append(f"{key} grouped: {tuple(value.shape)}")
    summary_lines.append(f"corner plot written: {corner_written}")

    predictive_summary = summarize_hyperposterior_predictive(
        bundle.observations,
        samples,
        bundle.x_test,
        config=config,
        max_samples=args.predictive_samples,
    )
    plot_gp_posterior(
        bundle,
        predictive_summary.mean,
        predictive_summary.total_cov,
        figure_dir,
        filename="nuts_hyperposterior_predictive.png",
        pmf_alignment=args.pmf_alignment,
    )
    plot_variance_decomposition(bundle, predictive_summary, figure_dir)
    result["hyperposterior_predictive"] = predictive_summary
    summary_lines.append(
        f"predictive samples used: {int(predictive_summary.selected_indices.numel())}"
    )
    summary_lines.append(
        f"average total predictive variance: {float(predictive_summary.total_variance.mean().item()):.6g}"
    )
    summary_lines.append(
        f"average within predictive variance: {float(predictive_summary.within_variance.mean().item()):.6g}"
    )
    summary_lines.append(
        f"average between predictive variance: {float(predictive_summary.between_variance.mean().item()):.6g}"
    )

    if args.posterior_draws > 0:
        pred_means, function_draws = sample_posterior_functions(
            bundle.observations,
            samples,
            bundle.x_test,
            n_draws=args.posterior_draws,
            config=config,
        )
        result["x_test"] = bundle.x_test
        result["pred_means"] = pred_means
        result["function_draws"] = function_draws
        map_idx, map_theta, map_score, map_pred_mean, map_pred_cov = maximum_a_posteriori_prediction(
            bundle.observations,
            samples,
            bundle.x_test,
            config=config,
        )
        plot_posterior_draws(bundle, pred_means, function_draws, figure_dir)
        plot_gp_posterior(
            bundle,
            map_pred_mean,
            map_pred_cov,
            figure_dir,
            filename="nuts_map_posterior.png",
            pmf_alignment=args.pmf_alignment,
        )
        result["map_sample_index"] = map_idx
        result["map_theta"] = map_theta
        result["map_log_posterior"] = map_score
        print(f"posterior function draws: {tuple(function_draws.shape)}")
        summary_lines.append(f"posterior function draws: {tuple(function_draws.shape)}")
        summary_lines.append(f"map sample index: {map_idx}")
        summary_lines.append(f"map parameters: {_format_param_mapping(map_theta)}")
        summary_lines.append(f"map log posterior: {float(map_score.detach().cpu().item()):.6g}")

    write_run_summary(figure_dir, summary_lines)

    return result


def maybe_save_output(output_path: str | None, payload) -> None:
    if not output_path:
        return
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"saved results to: {path}")


def _set_run_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import pyro
    except ImportError:
        return
    pyro.set_rng_seed(seed)


def _result_summary(payload) -> dict[str, object]:
    summary = {
        "mode": payload.get("mode"),
        "kernel": payload.get("kernel"),
        "dataset_root": payload.get("dataset_root"),
        "figure_dir": payload.get("figure_dir"),
    }
    if payload.get("mode") == "gp":
        pred_cov = payload["pred_cov"]
        summary["avg_total_variance"] = float(torch.diagonal(pred_cov).mean().detach().cpu().item())
    elif payload.get("mode") == "nuts":
        predictive = payload.get("hyperposterior_predictive")
        if predictive is not None:
            summary["avg_total_variance"] = float(predictive.total_variance.mean().detach().cpu().item())
            summary["avg_within_variance"] = float(predictive.within_variance.mean().detach().cpu().item())
            summary["avg_between_variance"] = float(predictive.between_variance.mean().detach().cpu().item())
        chain_diagnostics = payload.get("chain_diagnostics")
        if chain_diagnostics is not None:
            diag_summary = chain_diagnostics.get("summary", {})
            if diag_summary.get("max_r_hat") is not None:
                summary["max_r_hat"] = float(diag_summary["max_r_hat"])
            if diag_summary.get("min_n_eff") is not None:
                summary["min_n_eff"] = float(diag_summary["min_n_eff"])
            summary["divergence_total"] = int(diag_summary.get("divergence_total", 0))
        if "map_log_posterior" in payload:
            summary["map_log_posterior"] = float(payload["map_log_posterior"].detach().cpu().item())
    return summary


def _aggregate_summaries(summaries: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, float]]:
    if not summaries:
        return {}, {}
    mean_summary: dict[str, object] = {}
    std_summary: dict[str, float] = {}
    keys = summaries[0].keys()
    for key in keys:
        values = [summary[key] for summary in summaries]
        first = values[0]
        if isinstance(first, (int, float)):
            arr = np.asarray(values, dtype=float)
            mean_summary[key] = float(arr.mean())
            std_summary[key] = float(arr.std(ddof=0))
        else:
            mean_summary[key] = first
    return mean_summary, std_summary


def _write_device_comparison(
    root_dir: Path,
    *,
    comparisons: list[dict[str, object]],
) -> None:
    text_lines = ["device benchmark comparison"]
    for item in comparisons:
        text_lines.append("")
        text_lines.append(f"label: {item['label']}")
        text_lines.append(f"device: {item['device']}")
        text_lines.append(f"num_runs: {item['num_runs']}")
        text_lines.append(f"num_chains_per_run: {item['num_chains_per_run']}")
        text_lines.append(f"elapsed_seconds_total: {item['elapsed_seconds_total']:.6f}")
        text_lines.append(f"elapsed_seconds_mean: {item['elapsed_seconds_mean']:.6f}")
        for key, value in item["summary_mean"].items():
            text_lines.append(f"{key}: {value}")
        if item["summary_std"]:
            text_lines.append("summary_std:")
            for key, value in item["summary_std"].items():
                text_lines.append(f"  {key}: {value}")
    (root_dir / "device_comparison.txt").write_text("\n".join(text_lines) + "\n")
    (root_dir / "device_comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")


def _run_single(
    args: argparse.Namespace,
    base_bundle,
    *,
    device: str,
    figure_dir: Path,
    num_chains: int | None = None,
    seed: int | None = None,
):
    seed_to_use = args.seed if seed is None else seed
    _set_run_seed(seed_to_use)
    run_args = argparse.Namespace(
        **{
            **vars(args),
            "device": device,
            "num_chains": args.num_chains if num_chains is None else num_chains,
            "seed": seed_to_use,
        }
    )
    bundle = move_workflow_bundle(base_bundle, device=device)
    start = time.perf_counter()
    if run_args.mode == "gp":
        payload = run_gp_mode(run_args, bundle, figure_dir)
    else:
        payload = run_nuts_mode(run_args, bundle, figure_dir)
    elapsed_seconds = time.perf_counter() - start
    payload["device"] = device
    payload["elapsed_seconds"] = elapsed_seconds
    payload["seed"] = seed_to_use
    payload["num_chains"] = run_args.num_chains
    return payload


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = _load_config_defaults(pre_args.config)

    parser = build_parser(defaults=config_defaults)
    args = parser.parse_args()
    figure_dir = prepare_figure_dir(args.figure_dir, args.mode, project_root=args.project_root)

    base_bundle = prepare_gprhd_hmc_inputs(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        reference_wham_path=args.reference_wham_path,
        reference_wham_x_units=args.reference_wham_x_units,
        reference_ui_path=args.reference_ui_path,
        reference_ui_x_units=args.reference_ui_x_units,
        n_equilibration=args.n_equilibration,
        num_bins=args.num_bins,
        num_test_points=args.num_test_points,
        x_min=args.x_min,
        x_max=args.x_max,
        test_grid_source=args.test_grid_source,
    )

    compare_devices = _parse_device_list(args.compare_devices) if args.compare_devices else None
    if compare_devices:
        comparisons = []
        for device in compare_devices:
            resolved_device = resolve_device(device)
            if resolved_device == "cpu":
                num_runs = 1
                num_chains = args.benchmark_cpu_num_chains or args.num_chains
                label = (
                    f"cpu_parallel_{num_chains}chains"
                    if num_chains > 1 else "cpu"
                )
            else:
                num_runs = max(1, int(args.benchmark_gpu_runs))
                num_chains = 1 if num_runs > 1 else args.num_chains
                label = (
                    f"cuda_serial_{num_runs}runs"
                    if num_runs > 1 else "cuda"
                )

            device_dir = figure_dir / label
            device_dir.mkdir(parents=True, exist_ok=True)
            payloads = []
            for run_idx in range(num_runs):
                run_dir = device_dir / f"run_{run_idx:02d}" if num_runs > 1 else device_dir
                run_dir.mkdir(parents=True, exist_ok=True)
                payload = _run_single(
                    args,
                    base_bundle,
                    device=resolved_device,
                    figure_dir=run_dir,
                    num_chains=num_chains,
                    seed=args.seed + run_idx,
                )
                payloads.append(payload)

            summaries = [_result_summary(payload) for payload in payloads]
            summary_mean, summary_std = _aggregate_summaries(summaries)
            comparisons.append(
                {
                    "label": label,
                    "device": resolved_device,
                    "num_runs": num_runs,
                    "num_chains_per_run": num_chains,
                    "elapsed_seconds_total": float(sum(payload["elapsed_seconds"] for payload in payloads)),
                    "elapsed_seconds_mean": float(np.mean([payload["elapsed_seconds"] for payload in payloads])),
                    "summary_mean": summary_mean,
                    "summary_std": summary_std,
                    "run_dirs": [str((device_dir / f"run_{idx:02d}").resolve()) for idx in range(num_runs)]
                    if num_runs > 1 else [str(device_dir.resolve())],
                }
            )
        _write_device_comparison(figure_dir, comparisons=comparisons)
        return

    device = resolve_device(args.device)
    objectives = ["lml", "loo"] if args.objective == "both" else [args.objective]
    for objective in objectives:
        obj_figure_dir = figure_dir / objective if len(objectives) > 1 else figure_dir
        if len(objectives) > 1:
            obj_figure_dir.mkdir(parents=True, exist_ok=True)
        obj_args = argparse.Namespace(**{**vars(args), "objective": objective})
        payload = _run_single(obj_args, base_bundle, device=device, figure_dir=obj_figure_dir)
        output_path = args.output
        if output_path and len(objectives) > 1:
            p = Path(output_path)
            output_path = str(p.with_name(f"{p.stem}_{objective}{p.suffix}"))
        maybe_save_output(output_path, payload)


if __name__ == "__main__":
    main()
