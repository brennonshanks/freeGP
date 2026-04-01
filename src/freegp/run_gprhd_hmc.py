#!/usr/bin/env python3
"""CLI runner for the extracted GPR(H+D) HMC-NUTS workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.gp import GibbsKernelConfig, gpr_hd, gpr_hd_gibbs
    from freegp.hmc import (
        NUTSConfig,
        display_samples_for_diagnostics,
        maximum_a_posteriori_prediction,
        run_hmc_nuts,
        sample_posterior_functions,
    )
    from freegp.posterior import summarize_hyperposterior_predictive
    from freegp.workflow import prepare_gprhd_hmc_inputs
else:
    from .gp import GibbsKernelConfig, gpr_hd, gpr_hd_gibbs
    from .hmc import (
        NUTSConfig,
        display_samples_for_diagnostics,
        maximum_a_posteriori_prediction,
        run_hmc_nuts,
        sample_posterior_functions,
    )
    from .posterior import summarize_hyperposterior_predictive
    from .workflow import prepare_gprhd_hmc_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the extracted GPR(H+D) workflow from the old HMC-NUTS notebook."
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
    parser.add_argument("--ell", type=float, default=4.0, help="Initial/test GP length scale.")
    parser.add_argument("--w", type=float, default=3.3, help="Initial/test GP amplitude.")
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
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument(
        "--objective",
        choices=("lml", "loo"),
        default="lml",
        help="Objective used inside NUTS.",
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
        "--figure-dir",
        type=str,
        default=None,
        help="Directory for diagnostic plots. Defaults to ./figures/<timestamped-run>/",
    )
    parser.add_argument(
        "--no-corner",
        action="store_true",
        help="Disable the corner plot in NUTS mode.",
    )
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
        root = repo_root / "figures" / f"gprhd-{mode}-{stamp}"
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


def plot_gp_posterior(bundle, pred_mean, pred_cov, figure_dir: Path, *, filename: str) -> None:
    x_test = _to_numpy(bundle.x_test).ravel()
    pred_mean_np = _to_numpy(pred_mean).ravel()
    pred_std_np = np.sqrt(np.clip(np.diag(_to_numpy(pred_cov)), a_min=0.0, a_max=None))
    shifted_mean = pred_mean_np - np.max(pred_mean_np)

    refs = bundle.references
    wham_shift = refs.wham_f - np.max(refs.wham_f)
    umbrella_shift = refs.umbrella_f - np.max(refs.umbrella_f)

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
    plt.errorbar(refs.wham_x, wham_shift, yerr=refs.wham_e, capsize=3, color="crimson", alpha=0.5, label="WHAM")
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
        values = _to_numpy(samples[name]).ravel()
        axes[row, 0].plot(values, lw=1.0)
        axes[row, 0].set_title(f"{name} trace")
        axes[row, 1].hist(values, bins=30, color="gray", alpha=0.8)
        axes[row, 1].set_title(f"{name} histogram")
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
    plot_gp_posterior(bundle, pred_mean, pred_cov, figure_dir, filename="gp_posterior.png")
    plot_matrix(K_joint, figure_dir, filename="joint_covariance.png", title="Joint Covariance Matrix")
    plot_matrix(pred_cov, figure_dir, filename="predictive_covariance.png", title="Predictive Covariance Matrix")
    write_run_summary(
        figure_dir,
        [
            "mode: gp",
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
    try:
        summary = mcmc.summary(prob=0.9)
        summary_note = "pyro summary computed"
    except AssertionError as exc:
        summary = None
        summary_note = f"pyro summary skipped: {exc}"
    result = {
        "mode": "nuts",
        "kernel": args.kernel,
        "dataset_root": str(bundle.dataset_root),
        "samples": samples,
        "summary": summary,
        "figure_dir": str(figure_dir),
    }
    plot_histograms(bundle, figure_dir)
    plot_unbiased_windows(bundle, figure_dir)
    plot_nuts_traces(samples, figure_dir)
    corner_written = False
    if not args.no_corner:
        corner_written = plot_corner(samples, figure_dir, config=config)
    summary_lines = [
        "mode: nuts",
        f"dataset_root: {bundle.dataset_root}",
        f"figure_dir: {figure_dir}",
        f"objective: {args.objective}",
        *_kernel_summary_lines(args),
        f"num_samples: {args.num_samples}",
        f"warmup_steps: {args.warmup_steps}",
        f"num_chains: {args.num_chains}",
        f"x_test range: ({bundle.x_test.min().item():.6g}, {bundle.x_test.max().item():.6g})",
        summary_note,
    ]
    print("Finished NUTS run")
    print(f"dataset_root: {bundle.dataset_root}")
    print(f"figure_dir: {figure_dir}")
    for key, value in samples.items():
        print(f"{key}: {tuple(value.shape)}")
        summary_lines.append(f"{key}: {tuple(value.shape)}")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    figure_dir = prepare_figure_dir(args.figure_dir, args.mode, project_root=args.project_root)

    bundle = prepare_gprhd_hmc_inputs(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        n_equilibration=args.n_equilibration,
        num_bins=args.num_bins,
        num_test_points=args.num_test_points,
        x_min=args.x_min,
        x_max=args.x_max,
        test_grid_source=args.test_grid_source,
    )

    if args.mode == "gp":
        payload = run_gp_mode(args, bundle, figure_dir)
    else:
        payload = run_nuts_mode(args, bundle, figure_dir)

    maybe_save_output(args.output, payload)


if __name__ == "__main__":
    main()
