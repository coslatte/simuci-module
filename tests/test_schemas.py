"""Tests for simuci.schemas — CentroidSchema."""

from __future__ import annotations

from simuci.validation.schemas import CentroidSchema


class TestCentroidSchema:
    """Verify CentroidSchema defaults."""

    def test_defaults(self) -> None:
        schema = CentroidSchema()
        assert schema.n_clusters == 3
        assert schema.n_total_columns == 18
        assert schema.n_used_columns == 11
        assert len(schema.feature_columns) == 18
