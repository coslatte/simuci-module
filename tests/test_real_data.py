"""Integration tests against real SimUci data files."""

from __future__ import annotations

from pathlib import Path

import pytest

from simuci import Experiment, single_run
from simuci.io.loaders import CentroidLoader
from simuci.io.process_data import get_fecha_ingreso, get_time_simulation, load_file


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing real data file: {path}")


def test_real_centroids_load(real_centroids_csv: Path) -> None:
    _skip_if_missing(real_centroids_csv)
    centroids = CentroidLoader().load(real_centroids_csv)
    assert centroids.shape[1] == 11
    assert centroids.shape[0] >= 1


def test_real_clustering_run(real_centroids_csv: Path, valid_params: dict) -> None:
    _skip_if_missing(real_centroids_csv)
    exp = Experiment(**valid_params)
    result = single_run(exp, centroids_path=real_centroids_csv)
    assert "uci" in result
    assert result["uci"] >= 0


def test_real_patient_csv_load(real_patients_csv: Path) -> None:
    _skip_if_missing(real_patients_csv)
    fechas = load_file(real_patients_csv, "fecha_ingreso")
    assert len(fechas) > 0


def test_real_patient_generators(real_patients_csv: Path) -> None:
    _skip_if_missing(real_patients_csv)
    iterator = get_fecha_ingreso(real_patients_csv)
    first = next(iterator)
    assert isinstance(first, tuple)
    assert len(first) == 2


def test_real_time_simulation(real_patients_csv: Path) -> None:
    _skip_if_missing(real_patients_csv)
    hours = get_time_simulation(real_patients_csv)
    assert hours > 0
