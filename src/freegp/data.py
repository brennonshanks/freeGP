"""Dataset and reference-data loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import numpy as np
import torch

DEFAULT_DATASET_ENV = "FREEGP_DATASETS"
DEFAULT_KATKA_SUBDIR = "GPs_umbrellas_Katka"


@dataclass(frozen=True)
class UmbrellaWindow:
    folder: str
    folder_number: float
    time: torch.Tensor
    position: torch.Tensor
    mdp_last_key: str | None
    mdp_last_value: str | None


@dataclass(frozen=True)
class ReferenceCurves:
    umbrella_x: np.ndarray
    umbrella_f: np.ndarray
    umbrella_e: np.ndarray
    wham_x: np.ndarray
    wham_f: np.ndarray
    wham_e: np.ndarray


def resolve_dataset_root(
    base_path: str | os.PathLike[str] | None = None,
    *,
    env_var: str = DEFAULT_DATASET_ENV,
    default_subdir: str = DEFAULT_KATKA_SUBDIR,
) -> Path:
    """Resolve the umbrella-sampling dataset root.

    If ``base_path`` is not provided, the function looks for ``FREEGP_DATASETS``.
    When that variable points at a parent dataset directory, the
    ``GPs_umbrellas_Katka`` subdirectory is used automatically if present.
    """
    if base_path is not None:
        candidate = Path(base_path).expanduser().resolve()
    else:
        env_value = os.environ.get(env_var)
        if not env_value:
            raise FileNotFoundError(
                f"Dataset root not provided. Pass base_path or set {env_var}."
            )
        candidate = Path(env_value).expanduser().resolve()

    with_subdir = candidate / default_subdir
    root = with_subdir if with_subdir.exists() else candidate

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root


def load_pullx(file_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    time, position = [], []
    with file_path.open("r") as handle:
        for line in handle:
            if line.startswith(("#", "@")):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                time.append(float(parts[0]))
                position.append(float(parts[1]))
    return torch.tensor(time), torch.tensor(position)


def load_last_mdp_value(file_path: Path) -> tuple[str | None, str | None]:
    last_key, last_value = None, None
    with file_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(";") and "=" in line:
                last_key, last_value = map(str.strip, line.split("=", 1))
    return last_key, last_value


def natural_key(text: str) -> list[int | str]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", text)]


def extract_number(folder_name: str) -> float | None:
    match = re.search(r"d_([0-9]+\.[0-9]+)", folder_name)
    return float(match.group(1)) if match else None


def load_umbrella_windows(
    base_path: str | os.PathLike[str] | None = None,
) -> list[UmbrellaWindow]:
    """Load umbrella trajectories and force constants from the Katka dataset tree."""
    root = resolve_dataset_root(base_path)
    windows: list[UmbrellaWindow] = []

    for folder_name in sorted(os.listdir(root), key=natural_key):
        folder_path = root / folder_name
        pullx_file = folder_path / "step7_production_pullx.xvg"
        mdp_file = folder_path / "step7_production.mdp"
        if not folder_path.is_dir() or not pullx_file.is_file() or not mdp_file.is_file():
            continue

        folder_number = extract_number(folder_name)
        if folder_number is None:
            continue

        time, position = load_pullx(pullx_file)
        last_key, last_value = load_last_mdp_value(mdp_file)
        windows.append(
            UmbrellaWindow(
                folder=folder_name,
                folder_number=folder_number,
                time=time,
                position=position,
                mdp_last_key=last_key,
                mdp_last_value=last_value,
            )
        )

    if not windows:
        raise FileNotFoundError(f"No umbrella windows were found under {root}")
    return windows


def load_reference_curves(
    project_root: str | os.PathLike[str] | None = None,
) -> ReferenceCurves:
    """Load the reference WHAM and UI curves from the tracked project files."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    root = Path(project_root).resolve() / "freeGP_updated"

    umbrella_x, umbrella_f, umbrella_e = np.loadtxt(root / "UI-Semen" / "pmf.dat", unpack=True)
    wham_x, wham_f, wham_e = np.loadtxt(root / "wham.dat", unpack=True)
    return ReferenceCurves(
        umbrella_x=umbrella_x,
        umbrella_f=umbrella_f,
        umbrella_e=umbrella_e,
        wham_x=wham_x,
        wham_f=wham_f,
        wham_e=wham_e,
    )
