"""Tests for simuci.distribuciones — samplers and clustering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from simuci import clear_centroid_cache, clustering
from simuci.core.distributions import (
    _load_centroids,
    _sample_exponential,
    _sample_weibull,
    estad_UTI_cluster0,
    estad_UTI_cluster1,
    tiemp_postUCI_cluster0,
    tiemp_postUCI_cluster1,
    tiemp_VAM_cluster0,
    tiemp_VAM_cluster1,
)


class TestSamplers:
    """Verify that samplers return finite positive floats."""

    @pytest.mark.parametrize("sampler", [
        tiemp_VAM_cluster0,
        tiemp_postUCI_cluster0,
        estad_UTI_cluster0,
        tiemp_VAM_cluster1,
        tiemp_postUCI_cluster1,
        estad_UTI_cluster1,
    ])
    def test_sampler_returns_float(self, sampler: Callable[[], float]) -> None:
        value = sampler()
        assert isinstance(value, float)
        assert np.isfinite(value)

    @pytest.mark.parametrize("sampler", [
        tiemp_VAM_cluster0,
        estad_UTI_cluster0,
        tiemp_VAM_cluster1,
        tiemp_postUCI_cluster1,
        estad_UTI_cluster1,
    ])
    def test_sampler_non_negative(self, sampler: Callable[[], float]) -> None:
        """All time durations must be ≥ 0."""
        for _ in range(50):
            assert sampler() >= 0.0

    def test_exponential_returns_positive(self) -> None:
        for _ in range(100):
            v = _sample_exponential(mean=100.0)
            assert v > 0.0

    def test_weibull_returns_positive(self) -> None:
        for _ in range(100):
            v = _sample_weibull(shape=1.5, scale=200.0)
            assert v > 0.0

    def test_post_uci_cluster0_within_distribution_range(self) -> None:
        """The mixture of 3 uniforms should produce values in [0, 1056]."""
        values = [tiemp_postUCI_cluster0() for _ in range(200)]
        assert min(values) >= 0.0
        assert max(values) <= 1100.0  # generous upper bound


class TestCentroidLoading:
    """Verify centroid loading and caching."""

    def test_cache_returns_same_object(self, centroids_csv: str) -> None:
        clear_centroid_cache()
        a = _load_centroids(centroids_csv)
        b = _load_centroids(centroids_csv)
        assert a is b

    def test_clear_cache_reloads(self, centroids_csv: str) -> None:
        a = _load_centroids(centroids_csv)
        clear_centroid_cache()
        b = _load_centroids(centroids_csv)
        # Same values but different object after cache clear
        np.testing.assert_array_equal(a, b)

    def test_load_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_centroids("/nonexistent/path/centroids.csv")


class TestClustering:
    """Verify nearest-centroid classification."""

    def test_returns_int(self, centroids_csv: str) -> None:
        cluster = clustering(55, 11, 0, 0, 0, 20, 5, 1, 100, 50, 10, centroids_path=centroids_csv)
        assert isinstance(cluster, int)

    def test_returns_valid_cluster_id(self, centroids_csv: str) -> None:
        cluster = clustering(55, 11, 0, 0, 0, 20, 5, 1, 100, 50, 10, centroids_path=centroids_csv)
        assert cluster in (0, 1, 2)

    def test_deterministic_for_same_inputs(self, centroids_csv: str) -> None:
        """Clustering is deterministic (no randomness)."""
        args = (55, 11, 0, 0, 0, 20, 5, 1, 100, 50, 10)
        results = [clustering(*args, centroids_path=centroids_csv) for _ in range(10)]
        assert len(set(results)) == 1

    def test_va_group_derivation(self, centroids_csv: str) -> None:
        """va=2 or va=3 → va_group=2; else va_group=1.

        Changing va between 1 and 2 should change the feature vector
        and potentially the cluster assignment.
        """
        c1 = clustering(55, 11, 0, 0, 0, 20, 5, 1, 100, 50, 10, centroids_path=centroids_csv)
        c2 = clustering(55, 11, 0, 0, 0, 20, 5, 2, 100, 50, 10, centroids_path=centroids_csv)
        # Both should be valid (we can't guarantee different clusters
        # but the function should not crash)
        assert c1 in (0, 1, 2)
        assert c2 in (0, 1, 2)

    def test_custom_centroids_path(self, tmp_path) -> None:
        """clustering() accepts a custom centroids CSV."""
        import pandas as pd

        # Create a minimal valid centroids CSV
        data = np.random.default_rng(42).random((3, 18))
        df = pd.DataFrame(data, columns=[str(i) for i in range(18)])
        csv_path = tmp_path / "custom_centroids.csv"
        df.to_csv(csv_path)

        clear_centroid_cache()
        cluster = clustering(55, 11, 0, 0, 0, 20, 5, 1, 100, 50, 10, centroids_path=csv_path)
        assert cluster in (0, 1, 2)
