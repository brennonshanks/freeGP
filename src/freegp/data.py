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
    umbrella_x: np.ndarray | None = None
    umbrella_f: np.ndarray | None = None
    umbrella_e: np.ndarray | None = None
    wham_x: np.ndarray | None = None
    wham_f: np.ndarray | None = None
    wham_e: np.ndarray | None = None

    @property
    def has_wham(self) -> bool:
        return self.wham_x is not None

    @property
    def has_ui(self) -> bool:
        return self.umbrella_x is not None


_ANGSTROM_TO_NM = 0.1


def _load_pmf_file(
    path: str | os.PathLike[str],
    x_units: str = "nm",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column PMF file (x, free_energy kJ/mol). Converts x to nm if needed."""
    p = Path(path).expanduser().resolve()
    delimiter = "," if p.suffix.lower() == ".csv" else None
    data = np.loadtxt(p, delimiter=delimiter, comments=["#", "@", ";"])
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"PMF file {p} must have at least 2 columns.")
    x, f = data[:, 0], data[:, 1]
    if x_units.lower() in ("angstrom", "angstroms", "a", "å"):
        x = x * _ANGSTROM_TO_NM
    return x, f


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


def _is_denis_format(root: Path) -> bool:
    """Detect Denis-format dataset: a README file plus flat *-pullx.xvg files."""
    return (root / "README").is_file() and any(root.glob("*-pullx.xvg"))


def _parse_denis_readme(root: Path) -> dict[str, tuple[float, float]]:
    """Parse Denis README into {window_id: (r_eq_nm, force_constant_kJ_mol_nm2)}."""
    result: dict[str, tuple[float, float]] = {}
    with (root / "README").open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                result[parts[0]] = (float(parts[1]), float(parts[2]))
    return result


def _load_umbrella_windows_denis(root: Path) -> list[UmbrellaWindow]:
    """Load Denis-format ion dataset (flat pullx files + README metadata)."""
    params = _parse_denis_readme(root)
    windows: list[UmbrellaWindow] = []
    for pullx_file in sorted(root.glob("*-pullx.xvg"), key=lambda p: natural_key(p.name)):
        window_id = pullx_file.name[: -len("-pullx.xvg")]
        if window_id not in params:
            continue
        r_eq, k = params[window_id]
        time, position = load_pullx(pullx_file)
        windows.append(
            UmbrellaWindow(
                folder=window_id,
                folder_number=r_eq,
                time=time,
                position=position,
                mdp_last_key="pull-coord1-k",
                mdp_last_value=str(k),
            )
        )
    return windows


def _load_umbrella_windows_katka(root: Path) -> list[UmbrellaWindow]:
    """Load Katka-format membrane dataset (per-window subdirs with MDP files)."""
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
    return windows


def load_umbrella_windows(
    base_path: str | os.PathLike[str] | None = None,
) -> list[UmbrellaWindow]:
    """Load umbrella trajectories and force constants.

    Supports two dataset layouts automatically:
    - **Katka format**: per-window subdirectories containing
      ``step7_production_pullx.xvg`` and ``step7_production.mdp``.
    - **Denis format**: flat directory with ``*-pullx.xvg`` files and a
      ``README`` table (columns: window_id, r_eq [nm], k [kJ/mol/nm²]).
    """
    root = resolve_dataset_root(base_path)
    windows = _load_umbrella_windows_denis(root) if _is_denis_format(root) else _load_umbrella_windows_katka(root)
    if not windows:
        raise FileNotFoundError(f"No umbrella windows were found under {root}")
    return windows


def load_reference_curves(
    project_root: str | os.PathLike[str] | None = None,
    *,
    wham_path: str | os.PathLike[str] | None = None,
    wham_x_units: str = "nm",
    ui_path: str | os.PathLike[str] | None = None,
    ui_x_units: str = "nm",
) -> ReferenceCurves:
    """Load reference WHAM and/or UI PMF curves.

    Explicit paths (*wham_path*, *ui_path*) take priority over the project-relative
    defaults.  Either can be omitted independently — the corresponding
    ``ReferenceCurves`` fields will be ``None`` and that reference is skipped
    in all analyses and plots.  X-axis units can be ``"nm"`` or ``"angstrom"``.

    Falls back to the project layout (``reference_data/wham.dat`` and
    ``reference_data/UI-Semen/pmf.dat``) when no explicit paths are given.
    """
    wham_x = wham_f = wham_e = None
    ui_x = ui_f = ui_e = None

    if wham_path is not None:
        wham_x, wham_f = _load_pmf_file(wham_path, wham_x_units)
    if ui_path is not None:
        ui_x, ui_f = _load_pmf_file(ui_path, ui_x_units)

    if wham_path is None and ui_path is None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        project_path = Path(project_root).resolve()
        candidate_roots = [project_path / "reference_data", project_path / "freeGP_updated"]
        root = next((p for p in candidate_roots if p.exists()), None)
        if root is not None:
            wham_file = root / "wham.dat"
            ui_file = root / "UI-Semen" / "pmf.dat"
            if wham_file.is_file():
                wham_arr = np.loadtxt(wham_file, unpack=True)
                wham_x, wham_f = wham_arr[0], wham_arr[1]
                wham_e = wham_arr[2] if wham_arr.shape[0] > 2 else None
            if ui_file.is_file():
                ui_arr = np.loadtxt(ui_file, unpack=True)
                ui_x, ui_f = ui_arr[0], ui_arr[1]
                ui_e = ui_arr[2] if ui_arr.shape[0] > 2 else None

    return ReferenceCurves(
        umbrella_x=ui_x,
        umbrella_f=ui_f,
        umbrella_e=ui_e,
        wham_x=wham_x,
        wham_f=wham_f,
        wham_e=wham_e,
    )
