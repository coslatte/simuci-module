"""Tests for simuci.loaders — CentroidLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simuci.io.loaders import CentroidLoader
from simuci.validation.schemas import CentroidSchema


class TestCentroidLoader:
    """CentroidLoader validation and loading."""

    @pytest.fixture()
    def valid_csv(self, tmp_path: Path) -> str:
        """Write a valid 3x18 centroids CSV and return its path."""

        rng = np.random.default_rng(0)
        data = rng.random((3, 18))
        df = pd.DataFrame(data, columns=[str(i) for i in range(18)])
        path = tmp_path / "centroids.csv"
        df.to_csv(path, index=False)

        return str(path)

    def test_load_valid_csv(self, valid_csv: str) -> None:
        loader = CentroidLoader()
        centroids = loader.load(valid_csv)

        assert centroids.shape == (3, 11)
        assert centroids.dtype == np.float64

    def test_load_nonexistent_raises(self) -> None:
        loader = CentroidLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/no/such/file.csv")

    def test_too_few_columns_raises(self, tmp_path: Path) -> None:
        """CSV with fewer than 11 numeric columns should fail."""

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "small.csv"
        df.to_csv(path, index=False)

        loader = CentroidLoader()
        with pytest.raises(ValueError, match="numeric columns"):
            loader.load(path)

    def test_wrong_row_count_warns(self, tmp_path: Path) -> None:
        """CSV with ≠ 3 rows should log a warning but still load."""

        rng = np.random.default_rng(0)
        data = rng.random((5, 18))
        df = pd.DataFrame(data, columns=[str(i) for i in range(18)])
        path = tmp_path / "five_rows.csv"
        df.to_csv(path, index=False)

        loader = CentroidLoader()
        centroids = loader.load(str(path))

        assert centroids.shape == (5, 11)


class TestCentroidSchema:
    """Smoke test for CentroidSchema defaults."""

    def test_defaults(self) -> None:
        schema = CentroidSchema()
        assert schema.n_clusters == 3
        assert schema.n_total_columns == 18
        assert schema.n_used_columns == 11
        assert len(schema.feature_columns) == 18
