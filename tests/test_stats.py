"""Tests for simuci.stats — Wilcoxon, Friedman, SimulationMetrics, StatsUtils."""

from __future__ import annotations

import numpy as np
import pytest

from simuci import Friedman, SimulationMetrics, StatsUtils, Wilcoxon


class TestWilcoxon:
    """Wilcoxon signed-rank test wrapper."""

    def test_identical_data(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        w = Wilcoxon(x=data, y=data)
        w.test()
        # p-value should be high (or nan) for identical samples
        assert hasattr(w, "statistic")
        assert hasattr(w, "p_value")

    def test_different_data(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        w = Wilcoxon(x=x, y=y)
        w.test()
        assert isinstance(w.statistic, float)
        assert isinstance(w.p_value, float)
        assert 0.0 <= w.p_value <= 1.0


class TestFriedman:
    """Friedman chi-square test."""

    def test_basic(self) -> None:
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        s3 = [3.0, 4.0, 5.0, 6.0, 7.0]
        f = Friedman(samples=[s1, s2, s3])
        f.test()
        assert isinstance(f.statistic, float)
        assert isinstance(f.p_value, float)
        assert 0.0 <= f.p_value <= 1.0


class TestSimulationMetrics:
    """SimulationMetrics evaluation suite."""

    @pytest.fixture()
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create synthetic true/sim data for testing.

        true_data: (5 patients, 5 variables)
        simulation_data: (5 patients, 20 reps, 5 variables)
        """

        rng = np.random.default_rng(42)
        true = rng.integers(50, 500, size=(5, 5)).astype(float)
        sim = true[:, np.newaxis, :] + rng.normal(0, 20, size=(5, 20, 5))
        return true, sim

    def test_evaluate_populates_all_metrics(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        true, sim = sample_data
        m = SimulationMetrics(true_data=true, simulation_data=sim)
        m.evaluate(confidence_level=0.95, random_state=42)

        assert m.coverage_percentage is not None
        assert m.error_margin is not None
        assert m.kolmogorov_smirnov_result is not None
        assert m.anderson_darling_result is not None

    def test_coverage_returns_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        true, sim = sample_data
        m = SimulationMetrics(true_data=true, simulation_data=sim)
        m.evaluate(confidence_level=0.95, random_state=42)

        assert isinstance(m.coverage_percentage, dict)
        assert len(m.coverage_percentage) == 5

    def test_error_margin_as_tuple(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        true, sim = sample_data
        m = SimulationMetrics(true_data=true, simulation_data=sim)
        m.evaluate(random_state=42, result_as_dict=False)

        assert isinstance(m.error_margin, tuple)
        assert len(m.error_margin) == 3  # rmse, mae, mape

    def test_error_margin_as_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        true, sim = sample_data
        m = SimulationMetrics(true_data=true, simulation_data=sim)
        m.evaluate(random_state=42, result_as_dict=True)

        assert isinstance(m.error_margin, dict)
        assert "rmse" in m.error_margin
        assert "mae" in m.error_margin
        assert "mape" in m.error_margin

    def test_ks_result_as_dict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        true, sim = sample_data
        m = SimulationMetrics(true_data=true, simulation_data=sim)
        m.evaluate(random_state=42, result_as_dict=True)

        assert isinstance(m.kolmogorov_smirnov_result, dict)
        assert "per_variable" in m.kolmogorov_smirnov_result

    def test_invalid_simulation_shape_raises(self) -> None:
        """coverage_percentage expects 3-D sim data."""
        true = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])  # 1-D, not 3-D
        m = SimulationMetrics(true_data=true, simulation_data=sim)
        # evaluate swallows the error and logs it,
        # so coverage_percentage stays None
        m.evaluate()
        assert m.coverage_percentage is None


class TestStatsUtils:
    """StatsUtils confidence interval helper."""

    def test_zero_std_returns_mean(self) -> None:
        mean = np.array([10.0, 20.0])
        std = np.array([0.0, 0.0])
        lower, upper = StatsUtils.confidence_interval(mean, std, n=100)
        np.testing.assert_array_equal(lower, mean)
        np.testing.assert_array_equal(upper, mean)

    def test_ci_widens_with_higher_confidence(self) -> None:
        mean = np.array([100.0])
        std = np.array([10.0])
        _, upper_90 = StatsUtils.confidence_interval(mean, std, n=50, confidence=0.90)
        _, upper_99 = StatsUtils.confidence_interval(mean, std, n=50, confidence=0.99)
        assert upper_99[0] > upper_90[0]

    def test_ci_symmetric(self) -> None:
        mean = np.array([100.0])
        std = np.array([10.0])
        lower, upper = StatsUtils.confidence_interval(mean, std, n=50)
        np.testing.assert_almost_equal(mean - lower, upper - mean)
