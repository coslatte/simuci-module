"""Shared fixtures for simuci tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simuci import Experiment

# ---------------------------------------------------------------------------
# Valid default kwargs — reuse across tests via `valid_params` fixture
# ---------------------------------------------------------------------------

VALID_EXPERIMENT_KWARGS: dict = {
    "age": 55,
    "diagnosis_admission1": 11,
    "diagnosis_admission2": 0,
    "diagnosis_admission3": 0,
    "diagnosis_admission4": 0,
    "apache": 20,
    "respiratory_insufficiency": 5,
    "artificial_ventilation": 1,
    "uti_stay": 100,
    "vam_time": 50,
    "preuti_stay_time": 10,
    "percent": 3,
}
"""A set of valid experiment parameters within all limits."""


@pytest.fixture()
def valid_params() -> dict:
    """Return a copy of valid experiment kwargs (safe to mutate)."""

    return VALID_EXPERIMENT_KWARGS.copy()


@pytest.fixture()
def experiment(valid_params: dict) -> Experiment:
    """Return a validated Experiment instance with default valid params."""

    return Experiment(**valid_params)


@pytest.fixture()
def centroids_csv(tmp_path) -> str:
    """Write a valid 3×18 centroids CSV and return its path."""

    rng = np.random.default_rng(0)
    data = rng.random((3, 18))
    df = pd.DataFrame(data, columns=[str(i) for i in range(18)])
    path = tmp_path / "centroids.csv"
    df.to_csv(path, index=False)

    return str(path)


@pytest.fixture()
def real_data_dir() -> Path:
    """Return the SimUci data directory when present."""

    root = Path(__file__).resolve().parents[2]

    return root / "SimUci" / "data"


@pytest.fixture()
def real_centroids_csv(real_data_dir: Path) -> Path:
    """Return the path to the real centroids CSV (if present)."""

    return real_data_dir / "df_centroides.csv"


@pytest.fixture()
def real_patients_csv(real_data_dir: Path) -> Path:
    """Return the path to the real patient CSV (if present)."""

    return real_data_dir / "datos_pacientes.csv"
