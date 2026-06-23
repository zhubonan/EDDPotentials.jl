#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from airsspy.restools import read_res_atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mattersim_ase import EnvMatterSimCalculator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = SCRIPT_DIR / "surface-stability-analysis"
SLAB_GA = 24
SLAB_SB = 24
CLEAN_SLAB_Z_MAX = 0.49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MatterSim Ga/Sb/GaSb reference energies and plot adatom "
            "surface formation-energy curves over the GaSb chemical-potential window."
        )
    )
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--overwrite-references", action="store_true")
    parser.add_argument("--no-relax-references", action="store_true")
    parser.add_argument("--overwrite-surface-energies", action="store_true")
    parser.add_argument(
        "--no-compute-surface-energies",
        action="store_true",
        help="Use only cached MatterSim surface energies and fail if any are missing.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def zincblende_gasb() -> Structure:
    lattice = Lattice.cubic(6.0959)
    return Structure.from_spacegroup(
        216,
        lattice,
        ["Ga", "Sb"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def alpha_ga() -> Structure:
    # Orthorhombic alpha-Ga, conventional Cmca setting, 8 atoms/cell.
    lattice = Lattice.orthorhombic(4.5193, 7.6602, 4.5260)
    return Structure.from_spacegroup(
        64,
        lattice,
        ["Ga"],
        [[0.0785, 0.0, 0.1525]],
    )


def rhombohedral_sb() -> Structure:
    # A7/rhombohedral Sb represented in the common hexagonal setting.
    lattice = Lattice.hexagonal(4.307, 11.273)
    return Structure.from_spacegroup(
        166,
        lattice,
        ["Sb"],
        [[0.0, 0.0, 0.2330]],
    )


def reference_structures() -> dict[str, Structure]:
    return {
        "Ga": alpha_ga(),
        "Sb": rhombohedral_sb(),
        "GaSb": zincblende_gasb(),
    }


def relax_reference(name: str, structure: Structure, max_steps: int, fmax: float) -> dict[str, object]:
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = EnvMatterSimCalculator()
    filt = FrechetCellFilter(atoms)
    started = time.time()
    opt = FIRE(filt, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    finished = time.time()
    energy = float(atoms.get_potential_energy())
    counts = {symbol: atoms.get_chemical_symbols().count(symbol) for symbol in sorted(set(atoms.get_chemical_symbols()))}
    if name == "GaSb":
        formula_units = min(counts.get("Ga", 0), counts.get("Sb", 0))
        energy_unit = energy / formula_units
        unit = "eV/GaSb_fu"
    else:
        formula_units = counts[name]
        energy_unit = energy / formula_units
        unit = f"eV/{name}_atom"
    return {
        "name": name,
        "energy_eV": energy,
        "energy_per_reference_unit_eV": energy_unit,
        "reference_unit": unit,
        "natoms": len(atoms),
        "counts": counts,
        "volume_A3": float(atoms.get_volume()),
        "converged": converged,
        "steps": int(opt.nsteps),
        "fmax_eV_A": fmax,
        "runtime_seconds": finished - started,
    }


def reference_cache_path(analysis_dir: Path) -> Path:
    return analysis_dir / "mattersim-reference-energies.json"


def surface_cache_path(analysis_dir: Path) -> Path:
    return analysis_dir / "mattersim-surface-energies.json"


def clean_slab_cache_path(analysis_dir: Path) -> Path:
    return analysis_dir / "mattersim-clean-slab-energy.json"


def load_or_compute_references(args: argparse.Namespace, analysis_dir: Path) -> dict[str, dict[str, object]]:
    cache = reference_cache_path(analysis_dir)
    if cache.is_file() and not args.overwrite_references:
        return json.loads(cache.read_text())
    if args.no_relax_references:
        raise FileNotFoundError(f"Reference cache not found: {cache}")
    results = {
        name: relax_reference(name, structure, args.max_steps, args.fmax)
        for name, structure in reference_structures().items()
    }
    cache.write_text(json.dumps(results, indent=2, sort_keys=True))
    rows = [
        {
            "name": name,
            "energy_eV": data["energy_eV"],
            "energy_per_reference_unit_eV": data["energy_per_reference_unit_eV"],
            "reference_unit": data["reference_unit"],
            "natoms": data["natoms"],
            "volume_A3": data["volume_A3"],
            "converged": data["converged"],
            "steps": data["steps"],
            "runtime_seconds": data["runtime_seconds"],
        }
        for name, data in results.items()
    ]
    write_csv(
        analysis_dir / "mattersim-reference-energies.csv",
        rows,
        [
            "name",
            "energy_eV",
            "energy_per_reference_unit_eV",
            "reference_unit",
            "natoms",
            "volume_A3",
            "converged",
            "steps",
            "runtime_seconds",
        ],
    )
    return results


def load_adatom_minima(analysis_dir: Path) -> list[dict[str, object]]:
    path = analysis_dir / "lowest-by-adatom-formula.csv"
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "adatom_formula": row["adatom_formula"],
                    "adatom_ga_count": int(float(row["adatom_ga_count"])),
                    "adatom_sb_count": int(float(row["adatom_sb_count"])),
                    "adatom_excess_ga_minus_sb": int(float(row["adatom_excess_ga_minus_sb"])),
                    "total_ga_count": SLAB_GA + int(float(row["adatom_ga_count"])),
                    "total_sb_count": SLAB_SB + int(float(row["adatom_sb_count"])),
                    "total_formula": row["total_formula"],
                    "energy_per_atom_eV": float(row["energy_per_atom_eV"]),
                    "enthalpy_eV": float(row["enthalpy_eV"]),
                    "label": row["label"],
                    "path": row["path"],
                }
            )
    return rows


def area_ab(atoms) -> float:
    cell = np.asarray(atoms.get_cell(), dtype=float)
    return float(np.linalg.norm(np.cross(cell[0], cell[1])))


def atom_counts(atoms) -> dict[str, int]:
    symbols = atoms.get_chemical_symbols()
    return {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}


def matter_surface_energy(row: dict[str, object]) -> dict[str, object]:
    path = Path(str(row["path"]))
    _, atoms = read_res_atoms(path.read_text().splitlines())
    atoms.calc = EnvMatterSimCalculator()
    started = time.time()
    energy = float(atoms.get_potential_energy())
    finished = time.time()
    counts = atom_counts(atoms)
    return {
        "adatom_formula": row["adatom_formula"],
        "total_formula": row["total_formula"],
        "label": row["label"],
        "path": str(path.resolve()),
        "matter_total_energy_eV": energy,
        "matter_energy_per_atom_eV": energy / len(atoms),
        "natoms": len(atoms),
        "total_ga_count": counts.get("Ga", 0),
        "total_sb_count": counts.get("Sb", 0),
        "area_ab_A2": area_ab(atoms),
        "energy_model": "MatterSim",
        "runtime_seconds": finished - started,
    }


def extract_clean_slab_atoms(source_path: Path):
    _, atoms = read_res_atoms(source_path.read_text().splitlines())
    scaled = atoms.get_scaled_positions(wrap=True)
    keep = [idx for idx, pos in enumerate(scaled) if float(pos[2]) < CLEAN_SLAB_Z_MAX]
    slab = atoms[keep]
    slab.set_cell(atoms.get_cell())
    slab.set_pbc(atoms.get_pbc())
    return slab


def matter_clean_slab_energy(reference_row: dict[str, object]) -> dict[str, object]:
    source_path = Path(str(reference_row["path"]))
    atoms = extract_clean_slab_atoms(source_path)
    counts = atom_counts(atoms)
    if counts.get("Ga", 0) != SLAB_GA or counts.get("Sb", 0) != SLAB_SB:
        raise ValueError(
            f"Clean slab extraction from {source_path} gave {counts}, "
            f"expected Ga{SLAB_GA}Sb{SLAB_SB}"
        )
    atoms.calc = EnvMatterSimCalculator()
    started = time.time()
    energy = float(atoms.get_potential_energy())
    finished = time.time()
    return {
        "energy_model": "MatterSim",
        "source_label": reference_row["label"],
        "source_path": str(source_path.resolve()),
        "matter_clean_slab_energy_eV": energy,
        "matter_clean_slab_energy_per_atom_eV": energy / len(atoms),
        "natoms": len(atoms),
        "ga_count": counts.get("Ga", 0),
        "sb_count": counts.get("Sb", 0),
        "area_ab_A2": area_ab(atoms),
        "z_cutoff_fractional": CLEAN_SLAB_Z_MAX,
        "runtime_seconds": finished - started,
    }


def load_or_compute_clean_slab_energy(
    args: argparse.Namespace,
    analysis_dir: Path,
    minima_rows: list[dict[str, object]],
) -> dict[str, object]:
    cache = clean_slab_cache_path(analysis_dir)
    if cache.is_file() and not args.overwrite_surface_energies:
        data = json.loads(cache.read_text())
    else:
        reference_row = minima_rows[0]
        data = matter_clean_slab_energy(reference_row)
        cache.write_text(json.dumps(data, indent=2, sort_keys=True))
    write_csv(
        analysis_dir / "mattersim-clean-slab-energy.csv",
        [data],
        [
            "energy_model",
            "source_label",
            "matter_clean_slab_energy_eV",
            "matter_clean_slab_energy_per_atom_eV",
            "natoms",
            "ga_count",
            "sb_count",
            "area_ab_A2",
            "z_cutoff_fractional",
            "runtime_seconds",
            "source_path",
        ],
    )
    return data


def load_or_compute_surface_energies(
    args: argparse.Namespace,
    analysis_dir: Path,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    cache = surface_cache_path(analysis_dir)
    cached_rows: list[dict[str, object]] = []
    if cache.is_file() and not args.overwrite_surface_energies:
        cached_rows = json.loads(cache.read_text())
    cached_by_label = {str(row["label"]): row for row in cached_rows}

    results: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for row in rows:
        cached = cached_by_label.get(str(row["label"]))
        if cached is None:
            missing.append(row)
        else:
            results.append(cached)

    if missing and args.no_compute_surface_energies:
        labels = ", ".join(str(row["label"]) for row in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(f"Missing cached MatterSim surface energies for: {labels}{suffix}")

    if missing:
        for index, row in enumerate(missing, start=1):
            label = row["label"]
            print(f"MatterSim surface energy {index}/{len(missing)}: {label}", flush=True)
            results.append(matter_surface_energy(row))

    by_label = {str(row["label"]): row for row in results}
    ordered = [by_label[str(row["label"])] for row in rows]
    cache.write_text(json.dumps(ordered, indent=2, sort_keys=True))
    write_csv(
        analysis_dir / "mattersim-surface-energies.csv",
        ordered,
        [
            "adatom_formula",
            "total_formula",
            "label",
            "energy_model",
            "matter_total_energy_eV",
            "matter_energy_per_atom_eV",
            "natoms",
            "total_ga_count",
            "total_sb_count",
            "area_ab_A2",
            "runtime_seconds",
            "path",
        ],
    )
    return ordered


def attach_surface_energies(
    minima_rows: list[dict[str, object]],
    surface_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_label = {str(row["label"]): row for row in surface_rows}
    attached: list[dict[str, object]] = []
    for row in minima_rows:
        surface = by_label[str(row["label"])]
        total_ga = int(surface["total_ga_count"])
        total_sb = int(surface["total_sb_count"])
        expected_ga = int(row["total_ga_count"])
        expected_sb = int(row["total_sb_count"])
        if (total_ga, total_sb) != (expected_ga, expected_sb):
            raise ValueError(
                f"MatterSim count mismatch for {row['label']}: "
                f"got Ga{total_ga}Sb{total_sb}, expected Ga{expected_ga}Sb{expected_sb}"
            )
        updated = dict(row)
        updated.update(surface)
        updated["eddp_enthalpy_eV"] = row["enthalpy_eV"]
        updated["enthalpy_eV"] = float(surface["matter_total_energy_eV"])
        attached.append(updated)
    return attached


def chemical_potential_window(refs: dict[str, dict[str, object]]) -> dict[str, float]:
    mu_ga = float(refs["Ga"]["energy_per_reference_unit_eV"])
    mu_sb = float(refs["Sb"]["energy_per_reference_unit_eV"])
    mu_gasb = float(refs["GaSb"]["energy_per_reference_unit_eV"])
    delta_hf = mu_gasb - mu_ga - mu_sb
    return {
        "mu_Ga_element_eV": mu_ga,
        "mu_Sb_element_eV": mu_sb,
        "mu_GaSb_bulk_eV_fu": mu_gasb,
        "delta_Hf_GaSb_eV_fu": delta_hf,
        "delta_mu_Sb_min_eV": delta_hf,
        "delta_mu_Sb_max_eV": 0.0,
        "delta_mu_Ga_min_eV": delta_hf,
        "delta_mu_Ga_max_eV": 0.0,
    }


def chemical_potentials_at_delta_mu_sb(
    limits: dict[str, float], delta_mu_sb: float
) -> tuple[float, float, float]:
    delta_hf = limits["delta_Hf_GaSb_eV_fu"]
    delta_mu_ga = delta_hf - delta_mu_sb
    mu_ga = limits["mu_Ga_element_eV"] + delta_mu_ga
    mu_sb = limits["mu_Sb_element_eV"] + delta_mu_sb
    return mu_ga, mu_sb, delta_mu_ga


def grand_potential(row: dict[str, object], limits: dict[str, float], delta_mu_sb: float) -> float:
    n_ga = int(row["total_ga_count"])
    n_sb = int(row["total_sb_count"])
    mu_ga, mu_sb, _ = chemical_potentials_at_delta_mu_sb(limits, delta_mu_sb)
    return (
        float(row["enthalpy_eV"])
        - n_ga * mu_ga
        - n_sb * mu_sb
    )


def reconstruction_formation_energy(
    row: dict[str, object],
    clean_slab: dict[str, object],
    limits: dict[str, float],
    delta_mu_sb: float,
) -> float:
    n_ga = int(row["adatom_ga_count"])
    n_sb = int(row["adatom_sb_count"])
    mu_ga, mu_sb, _ = chemical_potentials_at_delta_mu_sb(limits, delta_mu_sb)
    return (
        float(row["matter_total_energy_eV"])
        - float(clean_slab["matter_clean_slab_energy_eV"])
        - n_ga * mu_ga
        - n_sb * mu_sb
    )


def plot_curves(
    analysis_dir: Path,
    rows: list[dict[str, object]],
    limits: dict[str, float],
    clean_slab: dict[str, object],
) -> None:
    delta_hf = limits["delta_Hf_GaSb_eV_fu"]
    if delta_hf >= 0.0:
        raise ValueError(f"MatterSim reference gives non-negative GaSb formation enthalpy: {delta_hf}")
    dmu_sb = np.linspace(delta_hf, 0.0, 401)
    dmu_ga = delta_hf - dmu_sb
    grand_energies = np.array([[grand_potential(row, limits, float(mu)) for mu in dmu_sb] for row in rows])
    recon_energies = np.array(
        [[reconstruction_formation_energy(row, clean_slab, limits, float(mu)) for mu in dmu_sb] for row in rows]
    )
    areas = np.array([float(row["area_ab_A2"]) for row in rows])
    grand_energies_area = grand_energies / areas[:, None]
    recon_energies_area = recon_energies / areas[:, None]
    envelope_area = np.min(recon_energies_area, axis=0)
    rel = recon_energies_area - envelope_area[None, :]
    env_idx = np.argmin(recon_energies_area, axis=0)

    curve_rows: list[dict[str, object]] = []
    for row, grand_values, grand_values_area, recon_values, recon_values_area in zip(
        rows, grand_energies, grand_energies_area, recon_energies, recon_energies_area
    ):
        for mu_sb, mu_ga, grand_value, grand_value_area, recon_value, recon_value_area in zip(
            dmu_sb, dmu_ga, grand_values, grand_values_area, recon_values, recon_values_area
        ):
            chem_mu_ga, chem_mu_sb, _ = chemical_potentials_at_delta_mu_sb(limits, float(mu_sb))
            curve_rows.append(
                {
                    "energy_model": row["energy_model"],
                    "adatom_formula": row["adatom_formula"],
                    "total_formula": row["total_formula"],
                    "label": row["label"],
                    "total_ga_count": row["total_ga_count"],
                    "total_sb_count": row["total_sb_count"],
                    "adatom_ga_count": row["adatom_ga_count"],
                    "adatom_sb_count": row["adatom_sb_count"],
                    "matter_total_energy_eV": row["matter_total_energy_eV"],
                    "matter_energy_per_atom_eV": row["matter_energy_per_atom_eV"],
                    "matter_clean_slab_energy_eV": clean_slab["matter_clean_slab_energy_eV"],
                    "eddp_enthalpy_eV": row["eddp_enthalpy_eV"],
                    "area_ab_A2": row["area_ab_A2"],
                    "delta_mu_Sb_eV": float(mu_sb),
                    "delta_mu_Ga_eV": float(mu_ga),
                    "mu_Ga_eV": float(chem_mu_ga),
                    "mu_Sb_eV": float(chem_mu_sb),
                    "grand_potential_eV": float(grand_value),
                    "grand_potential_eV_A2": float(grand_value_area),
                    "reconstruction_formation_energy_eV": float(recon_value),
                    "reconstruction_formation_energy_eV_A2": float(recon_value_area),
                }
            )
    write_csv(
        analysis_dir / "adatom-formation-energy-curves.csv",
        curve_rows,
        [
            "energy_model",
            "adatom_formula",
            "total_formula",
            "label",
            "total_ga_count",
            "total_sb_count",
            "adatom_ga_count",
            "adatom_sb_count",
            "matter_total_energy_eV",
            "matter_energy_per_atom_eV",
            "matter_clean_slab_energy_eV",
            "eddp_enthalpy_eV",
            "area_ab_A2",
            "delta_mu_Sb_eV",
            "delta_mu_Ga_eV",
            "mu_Ga_eV",
            "mu_Sb_eV",
            "grand_potential_eV",
            "grand_potential_eV_A2",
            "reconstruction_formation_energy_eV",
            "reconstruction_formation_energy_eV_A2",
        ],
    )

    segments: list[dict[str, object]] = []
    start = 0
    for idx in range(1, len(dmu_sb) + 1):
        if idx == len(dmu_sb) or env_idx[idx] != env_idx[start]:
            row = rows[int(env_idx[start])]
            start_mu_sb = dmu_sb[start]
            start_mu_ga = dmu_ga[start]
            if segments:
                start_mu_sb = float(segments[-1]["delta_mu_Sb_end_eV"])
                start_mu_ga = float(segments[-1]["delta_mu_Ga_end_eV"])
            segments.append(
                {
                    "delta_mu_Sb_start_eV": float(start_mu_sb),
                    "delta_mu_Sb_end_eV": float(dmu_sb[idx - 1]),
                    "delta_mu_Ga_start_eV": float(start_mu_ga),
                    "delta_mu_Ga_end_eV": float(dmu_ga[idx - 1]),
                    "adatom_formula": row["adatom_formula"],
                    "total_formula": row["total_formula"],
                    "label": row["label"],
                    "path": row["path"],
                }
            )
            start = idx
    write_csv(
        analysis_dir / "adatom-formation-energy-envelope.csv",
        segments,
        [
            "delta_mu_Sb_start_eV",
            "delta_mu_Sb_end_eV",
            "delta_mu_Ga_start_eV",
            "delta_mu_Ga_end_eV",
            "adatom_formula",
            "total_formula",
            "label",
            "path",
        ],
    )

    labels = [str(row["adatom_formula"]) for row in rows]
    formula_cmap = plt.get_cmap("turbo")
    denom = max(len(rows) - 1, 1)
    colors = [formula_cmap(idx / denom) for idx in range(len(rows))]

    def add_rich_labels(ax: plt.Axes, left_label: str, right_label: str) -> None:
        ax.text(0.015, 0.025, left_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
        ax.text(0.985, 0.025, right_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    def add_formula_legend(fig: plt.Figure, handles: list[object]) -> None:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.835, 0.5),
            ncol=2,
            fontsize=5.6,
            frameon=False,
            handlelength=1.8,
            columnspacing=0.9,
        )

    def plot_all_curves(
        x_values: np.ndarray,
        y_values: np.ndarray,
        envelope_values: np.ndarray,
        output_name: str,
        xlabel: str,
        ylabel: str,
        title: str,
        xlim: tuple[float, float],
        rich_left_label: str,
        rich_right_label: str,
        relative: bool,
    ) -> None:
        fig, ax = plt.subplots(figsize=(15.5, 7.2))
        handles: list[object] = []
        for idx, row in enumerate(rows):
            handle = ax.plot(
                x_values,
                y_values[idx],
                lw=1.0,
                alpha=0.82,
                color=colors[idx],
                label=str(row["adatom_formula"]),
            )[0]
            handles.append(handle)
        if relative:
            ax.axhline(0.0, color="black", lw=1.5, label="envelope")
            y_min = -0.02
            y_max = max(0.2, float(np.nanmax(y_values)))
            ax.set_ylim(y_min, y_max + 0.04 * max(y_max, 1.0))
        else:
            ax.plot(x_values, envelope_values, color="black", lw=2.0, label="envelope")
            y_min = float(np.nanmin(y_values))
            y_max = float(np.nanmax(y_values))
            y_span = max(y_max - y_min, 1.0)
            ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.05 * y_span)
        ax.axvline(delta_hf, color="#666666", lw=0.8, ls="--")
        ax.axvline(0.0, color="#666666", lw=0.8, ls="--")
        ax.set_xlim(*xlim)
        add_rich_labels(ax, rich_left_label, rich_right_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        add_formula_legend(fig, handles)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        fig.savefig(analysis_dir / output_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

    plot_all_curves(
        dmu_sb,
        recon_energies_area,
        envelope_area,
        "adatom-formation-energy-all-vs-dmu-sb.png",
        "Delta mu_Sb (eV), Ga-rich -> Sb-rich",
        "reconstruction formation energy density (eV/A^2)",
        "MatterSim-only reconstruction formation energy vs Sb chemical potential",
        (delta_hf, 0.0),
        "Ga-rich",
        "Sb-rich",
        relative=False,
    )
    plot_all_curves(
        dmu_ga,
        recon_energies_area,
        envelope_area,
        "adatom-formation-energy-all-vs-dmu-ga.png",
        "Delta mu_Ga (eV), Ga-rich -> Sb-rich",
        "reconstruction formation energy density (eV/A^2)",
        "MatterSim-only reconstruction formation energy vs Ga chemical potential",
        (0.0, delta_hf),
        "Ga-rich",
        "Sb-rich",
        relative=False,
    )
    plot_all_curves(
        dmu_sb,
        rel,
        np.zeros_like(envelope_area),
        "adatom-formation-energy-relative-all-vs-dmu-sb.png",
        "Delta mu_Sb (eV), Ga-rich -> Sb-rich",
        "relative reconstruction formation energy density (eV/A^2)",
        "MatterSim-only reconstruction formation energy relative to envelope vs Sb chemical potential",
        (delta_hf, 0.0),
        "Ga-rich",
        "Sb-rich",
        relative=True,
    )
    plot_all_curves(
        dmu_ga,
        rel,
        np.zeros_like(envelope_area),
        "adatom-formation-energy-relative-all-vs-dmu-ga.png",
        "Delta mu_Ga (eV), Ga-rich -> Sb-rich",
        "relative reconstruction formation energy density (eV/A^2)",
        "MatterSim-only reconstruction formation energy relative to envelope vs Ga chemical potential",
        (0.0, delta_hf),
        "Ga-rich",
        "Sb-rich",
        relative=True,
    )

    near = np.where(np.min(rel, axis=1) <= 0.01)[0]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for idx in near:
        row = rows[int(idx)]
        ax.plot(dmu_sb, rel[idx], lw=0.9, alpha=0.45)
    ax.axhline(0.0, color="black", lw=1.1)
    ax.axvline(delta_hf, color="#666666", lw=0.8, ls="--")
    ax.axvline(0.0, color="#666666", lw=0.8, ls="--")
    ax.set_xlim(delta_hf, 0.0)
    add_rich_labels(ax, "Ga-rich", "Sb-rich")
    for seg in segments:
        mid = 0.5 * (seg["delta_mu_Sb_start_eV"] + seg["delta_mu_Sb_end_eV"])
        ax.text(mid, 0.00035, str(seg["adatom_formula"]), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Delta mu_Sb (eV), Ga-rich -> Sb-rich")
    ax.set_ylabel("relative reconstruction formation energy density (eV/A^2)")
    ax.set_title("MatterSim-only near-envelope reconstruction energy vs Sb chemical potential", fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.0002, max(0.002, min(0.08, float(np.nanpercentile(rel[near], 95)) if len(near) else 0.01)))
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(analysis_dir / "adatom-formation-energy-vs-dmu-sb.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for idx in near:
        ax.plot(dmu_ga, rel[idx], lw=0.9, alpha=0.45)
    ax.axhline(0.0, color="black", lw=1.1)
    ax.axvline(0.0, color="#666666", lw=0.8, ls="--")
    ax.axvline(delta_hf, color="#666666", lw=0.8, ls="--")
    ax.set_xlim(0.0, delta_hf)
    add_rich_labels(ax, "Ga-rich", "Sb-rich")
    ax.set_xlabel("Delta mu_Ga (eV), Ga-rich -> Sb-rich")
    ax.set_ylabel("relative reconstruction formation energy density (eV/A^2)")
    ax.set_title("MatterSim-only near-envelope reconstruction energy vs Ga chemical potential", fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.0002, max(0.002, min(0.08, float(np.nanpercentile(rel[near], 95)) if len(near) else 0.01)))
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(analysis_dir / "adatom-formation-energy-vs-dmu-ga.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/eddp-mattersim-mpl")
    analysis_dir = Path(args.analysis_dir).resolve()
    refs = load_or_compute_references(args, analysis_dir)
    limits = chemical_potential_window(refs)
    write_csv(
        analysis_dir / "mattersim-chemical-potential-limits.csv",
        [{"quantity": key, "value": value, "unit": "eV"} for key, value in limits.items()],
        ["quantity", "value", "unit"],
    )
    minima_rows = load_adatom_minima(analysis_dir)
    clean_slab = load_or_compute_clean_slab_energy(args, analysis_dir, minima_rows)
    surface_rows = load_or_compute_surface_energies(args, analysis_dir, minima_rows)
    rows = attach_surface_energies(minima_rows, surface_rows)
    plot_curves(analysis_dir, rows, limits, clean_slab)
    print(json.dumps(limits, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR))
    main()
