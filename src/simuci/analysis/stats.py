"""Statistical tests and metrics for ICU simulation validation.

Classes
-------
Wilcoxon
    Paired Wilcoxon signed-rank test.
Friedman
    Friedman chi-square test for repeated measures.
SimulationMetrics
    Evaluation suite comparing real patient data against simulation output
    (coverage percentage, error margin, Kolmogorov-Smirnov, Anderson-Darling).
StatsUtils
    Static helpers (confidence intervals).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import (
    anderson_ksamp,
    friedmanchisquare,
    ks_2samp,
    wilcoxon,
)
from scipy.stats import (
    t as t_dist,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from simuci.internals._constants import EXPERIMENT_VARIABLES_LABELS
from simuci.internals._types import ArrayLike1D, Metric

try:
    # Available in newer SciPy versions; used to request permutation-based
    # p-values for `anderson_ksamp` when supported.
    from scipy.stats import PermutationMethod as _PermutationMethod
except ImportError:
    _PermutationMethod: Any = None

logger = logging.getLogger(__name__)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Simple hypothesis tests
# ---------------------------------------------------------------------------


@dataclass
class Wilcoxon:
    """Paired Wilcoxon signed-rank test wrapper."""

    x: ArrayLike1D
    y: ArrayLike1D

    statistic: float = 0.0
    p_value: float = 0.0

    def test(self) -> None:
        """Run the test and populate :attr:`statistic` and :attr:`p_value`."""

        x = np.asarray(self.x)
        y = np.asarray(self.y)

        samples_are_identical = x.shape == y.shape and np.allclose(x, y)
        if samples_are_identical:
            self.statistic = 0.0
            self.p_value = 1.0
            return

        result = wilcoxon(x, y)
        self.statistic = float(getattr(result, "statistic", 0.0))
        self.p_value = float(getattr(result, "pvalue", 1.0))


@dataclass
class Friedman:
    """Friedman chi-square test for *k* related samples."""

    samples: Sequence[Sequence[float]]

    statistic: float = 0.0
    p_value: float = 0.0

    def test(self) -> None:
        """Run the test and populate :attr:`statistic` and :attr:`p_value`."""

        result = friedmanchisquare(*self.samples)
        self.statistic = float(getattr(result, "statistic", 0.0))
        self.p_value = float(getattr(result, "pvalue", 1.0))


# ---------------------------------------------------------------------------
# Simulation evaluation metrics
# ---------------------------------------------------------------------------


@dataclass
class SimulationMetrics:
    """Compare real (true) patient data with simulation output.

    After calling :meth:`evaluate`, the result attributes
    (:attr:`coverage_percentage`, :attr:`error_margin`, etc.) are populated.
    """

    true_data: np.ndarray
    simulation_data: np.ndarray

    coverage_percentage: Metric | None = None
    error_margin: Metric | None = None
    kolmogorov_smirnov_result: Metric | None = None
    anderson_darling_result: Metric | None = None

    # ---- public API -------------------------------------------------------

    def evaluate(
        self,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        result_as_dict: bool = False,
    ) -> None:
        """Run all evaluation metrics.

        Args:
            confidence_level: Confidence level for coverage percentage
                (0.80–0.95 recommended).
            random_state: Seed for the RNG used in sampling-based tests.
                ``None`` for non-reproducible.
            result_as_dict: When ``True`` individual metrics return dicts
                instead of tuples.
        """

        rng = np.random.default_rng(random_state)

        confidence_is_out_of_range = not (0.80 <= confidence_level <= 0.95)
        if confidence_is_out_of_range:
            logger.warning(
                "Confidence level %.2f is outside the recommended 0.80–0.95 range",
                confidence_level,
            )

        try:
            self.coverage_percentage = self._calculate_coverage_percentage(
                confidence_level=confidence_level,
            )
            self.error_margin = self._calculate_error_margin(
                as_dict=result_as_dict,
            )
            self.kolmogorov_smirnov_result = self._ks_test(
                as_dict=result_as_dict,
            )
            self.anderson_darling_result = self._ad_test(
                as_dict=result_as_dict,
                rng=rng,
            )
        except Exception:
            logger.exception("Evaluation failed")

    # ---- coverage percentage ----------------------------------------------

    def _calculate_coverage_percentage(
        self,
        confidence_level: float = 0.95,
    ) -> dict[str, float]:
        """Fraction of patients whose true value falls inside the simulation CI."""

        simulation_data = np.asarray(self.simulation_data)

        if simulation_data.ndim != 3:
            raise ValueError(
                "simulation_data must be 3-D " "(n_patients, n_replicates, n_variables)"
            )

        n_patients, n_replicates, n_variables = simulation_data.shape
        true_data = self._coerce_true_data(n_patients, n_variables)

        lower, upper = self._confidence_bounds(
            simulation_data,
            n_replicates,
            confidence_level,
        )

        value_inside_ci = (true_data >= lower) & (true_data <= upper)
        coverage_per_variable = value_inside_ci.mean(axis=0) * 100

        return {
            self._variable_label(i): float(coverage_per_variable[i])
            for i in range(n_variables)
        }

    @staticmethod
    def _confidence_bounds(
        simulation_data: np.ndarray,
        n_replicates: int,
        confidence_level: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute lower and upper confidence bounds from simulation replicates."""

        means = np.mean(simulation_data, axis=1)

        if n_replicates < 2:
            logger.warning(
                "Fewer than 2 replicates — CI degenerates to the point estimate"
            )

            return means, means

        alpha = 1 - confidence_level
        degrees_of_freedom = n_replicates - 1

        t_value = t_dist.ppf(1 - alpha / 2, degrees_of_freedom)
        standard_errors = np.std(simulation_data, axis=1, ddof=1) / np.sqrt(
            n_replicates
        )
        margin = t_value * standard_errors

        lower = means - margin
        upper = means + margin
        return lower, upper

    # ---- error margin (RMSE / MAE / MAPE) ---------------------------------

    def _aligned_true_and_simulated(self) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned ``(true_data, simulation_mean)`` for error metrics."""

        true_data = np.asarray(self.true_data)
        simulation_mean = np.mean(np.asarray(self.simulation_data), axis=1)

        aligned_true_data = self._align_shape(true_data, simulation_mean.shape)
        return aligned_true_data, simulation_mean

    def _calculate_rmse(self) -> float:
        """Compute Root Mean Squared Error between true data and simulation means."""

        true_data, simulation_mean = self._aligned_true_and_simulated()
        mse = mean_squared_error(true_data, simulation_mean)
        return float(np.sqrt(mse))

    def _calculate_mae(self) -> float:
        """Compute Mean Absolute Error between true data and simulation means."""

        true_data, simulation_mean = self._aligned_true_and_simulated()
        return float(mean_absolute_error(true_data, simulation_mean))

    def _calculate_mape(self) -> float:
        """Compute Mean Absolute Percentage Error between true data and simulation means.

        Returns ``nan`` when all true values are zero (division by zero).
        """

        true_data, simulation_mean = self._aligned_true_and_simulated()

        is_zero = true_data == 0

        if np.all(is_zero):
            logger.warning("MAPE undefined — all true values are zero")

            return _NAN

        with np.errstate(divide="ignore", invalid="ignore"):
            absolute_percentage_errors = np.abs(
                (true_data - simulation_mean) / true_data.astype(float)
            )

            # Mask out zero-valued entries to avoid inf/nan contamination
            masked_errors = np.where(is_zero, np.nan, absolute_percentage_errors)

            return float(np.nanmean(masked_errors) * 100)

    def _calculate_error_margin(self, as_dict: bool = False) -> Metric:
        """Compute RMSE, MAE and MAPE between true data and simulation means."""

        rmse = self._calculate_rmse()
        mae = self._calculate_mae()
        mape = self._calculate_mape()

        logger.info("RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%", rmse, mae, mape)

        if as_dict:
            return {"rmse": rmse, "mae": mae, "mape": mape}
        return (rmse, mae, mape)

    # ---- Kolmogorov–Smirnov -----------------------------------------------

    @staticmethod
    def _ks_single(
        sample_a: np.ndarray,
        sample_b: np.ndarray,
    ) -> tuple[float, float]:
        """Run a single two-sample KS test, returning ``(statistic, p_value)``."""
        try:
            result = ks_2samp(sample_a, sample_b)

            statistic = float(getattr(result, "statistic", _NAN))
            p_value = float(getattr(result, "pvalue", _NAN))

            return statistic, p_value
        except Exception as e:
            print(f"KS test failed: {e}")

            return _NAN, _NAN

    def _ks_test_flat(self, as_dict: bool) -> Metric:
        """KS test for flat (non-3-D) simulation data."""
        true_data = np.asarray(self.true_data)
        simulation_flat = np.asarray(self.simulation_data).flatten()

        statistic, p_value = self._ks_single(true_data, simulation_flat)

        logger.info("KS — stat=%.4f  p=%.4f", statistic, p_value)

        if as_dict:
            return {"statistic": statistic, "p_value": p_value}
        return (statistic, p_value)

    def _ks_test_per_variable(self, as_dict: bool) -> Metric:
        """Per-variable KS tests for 3-D simulation data."""
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)
        n_variables = simulation_data.shape[2]

        # Run one KS test per output variable
        per_variable_results: list[tuple[float, float]] = []

        for v in range(n_variables):
            true_column = (
                true_data[:, v].ravel() if true_data.ndim > 1 else true_data.ravel()
            )
            simulated_column = simulation_data[:, :, v].ravel()

            stat, p = self._ks_single(true_column, simulated_column)
            per_variable_results.append((stat, p))

        all_statistics = np.array([s for s, _ in per_variable_results])
        all_p_values = np.array([p for _, p in per_variable_results])
        overall_statistic = float(np.nanmean(all_statistics))
        overall_p_value = float(np.nanmean(all_p_values))

        logger.info("KS per-variable: %s", per_variable_results)
        logger.info(
            "KS overall — stat=%.4f  p=%.4f",
            overall_statistic,
            overall_p_value,
        )

        if as_dict:
            per_variable_dict = {
                self._variable_label(i): {"statistic": s, "p_value": p}
                for i, (s, p) in enumerate(per_variable_results)
            }
            return {
                "per_variable": per_variable_dict,
                "overall": {
                    "statistic": overall_statistic,
                    "p_value": overall_p_value,
                },
            }
        return (overall_statistic, overall_p_value)

    def _ks_test(self, as_dict: bool = False) -> Metric:
        """Two-sample Kolmogorov-Smirnov test.

        Dispatches to :meth:`_ks_test_flat` or :meth:`_ks_test_per_variable`
        depending on the shape of the simulation data.
        """
        simulation_data = np.asarray(self.simulation_data)

        if simulation_data.ndim == 3:
            return self._ks_test_per_variable(as_dict)
        return self._ks_test_flat(as_dict)

    # ---- Anderson–Darling -------------------------------------------------

    @staticmethod
    def _run_anderson_ksamp(
        real_sample: np.ndarray,
        simulated_sample: np.ndarray,
    ) -> tuple[float, float]:
        """Execute the Anderson–Darling k-sample test with best available options.

        Returns ``(statistic, significance_level)``.
        Tries ``variant="midrank"`` first, falls back if the scipy version
        does not support it.
        """

        kwargs: dict[str, object] = {}
        if _PermutationMethod is not None:
            kwargs["method"] = _PermutationMethod()

        try:
            result = anderson_ksamp(
                [real_sample, simulated_sample],
                variant="midrank",
                **kwargs,
            )
        except TypeError:
            # Older scipy versions may not accept 'variant'
            result = anderson_ksamp(
                [real_sample, simulated_sample],
                **kwargs,
            )

        statistic = float(getattr(result, "statistic", _NAN))
        significance = float(getattr(result, "significance_level", _NAN))
        return statistic, significance

    def _flatten_for_ad_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Flatten true data and simulation data for the AD test.

        For 3-D simulation data the simulation means are used.
        """

        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        if simulation_data.ndim == 3:
            simulation_means = np.mean(simulation_data, axis=1)
            return true_data.flatten(), simulation_means.flatten()

        return true_data, simulation_data

    def _ad_test(
        self,
        as_dict: bool = False,
        *,
        rng: np.random.Generator | None = None,
    ) -> Metric:
        """Anderson-Darling *k*-sample test comparing true vs. simulated data."""
        if rng is None:
            rng = np.random.default_rng()

        true_flat, sim_flat = self._flatten_for_ad_test()

        sample_size = min(len(true_flat), len(sim_flat))
        real_sample = rng.choice(true_flat, sample_size, replace=False)
        simulated_sample = rng.choice(sim_flat, sample_size, replace=False)

        try:
            statistic, significance = self._run_anderson_ksamp(
                real_sample,
                simulated_sample,
            )
        except Exception:
            logger.exception("Anderson-Darling test failed")
            statistic = _NAN
            significance = _NAN

        logger.info(
            "Anderson-Darling - stat=%.4f p≈%.3f",
            statistic,
            significance,
        )

        if as_dict:
            return {"statistic": statistic, "significance_level": significance}
        return (statistic, significance)

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _variable_label(index: int) -> str:
        """Return a human-readable variable name for *index*."""
        labels = list(EXPERIMENT_VARIABLES_LABELS.values())

        if index < len(labels):
            return labels[index]
        return f"variable_{index}"

    @staticmethod
    def _align_shape(
        arr: np.ndarray,
        target_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Best-effort reshape / resize of *arr* to *target_shape*."""

        if arr.shape == target_shape:
            return arr
        if arr.size == np.prod(target_shape):
            return arr.reshape(target_shape)
        return np.resize(arr, target_shape)

    def _coerce_true_data(
        self,
        n_patients: int,
        n_variables: int,
    ) -> np.ndarray:
        """Coerce :attr:`true_data` into shape ``(n_patients, n_variables)``.

        Handles scalar, 1-D, 2-D, and higher-dimensional inputs by
        reshaping, tiling, or resizing as needed.
        """

        td = np.asarray(self.true_data)
        target = (n_patients, n_variables)

        # Scalar → fill the entire target
        if td.ndim == 0:
            return np.full(target, float(td))

        if td.ndim == 1:
            return self._coerce_1d(td, n_patients, n_variables)

        if td.ndim == 2:
            return self._coerce_2d(td, n_patients, n_variables)

        # 3-D+ → last-resort flatten & resize
        logger.warning(
            "true_data has %d dimensions; flattening and resizing",
            td.ndim,
        )

        return np.resize(td.ravel(), target)

    @staticmethod
    def _coerce_1d(
        td: np.ndarray,
        n_patients: int,
        n_variables: int,
    ) -> np.ndarray:
        """Coerce a 1-D array into ``(n_patients, n_variables)``."""
        target = (n_patients, n_variables)

        # Exact total size → just reshape
        if td.size == n_patients * n_variables:
            return td.reshape(target)

        # One value per variable → repeat for each patient
        if td.size == n_variables:
            single_row = td.reshape(1, n_variables)
            return np.tile(single_row, (n_patients, 1))

        # One value per patient → broadcast across variables
        if td.size == n_patients:
            if n_variables == 1:
                return td.reshape(n_patients, 1)

            logger.warning("true_data has length n_patients; tiling across variables")
            single_column = td.reshape(n_patients, 1)
            return np.tile(single_column, (1, n_variables))

        # No match → resize with warning
        logger.warning(
            "true_data length mismatch; resizing to (%d, %d)",
            *target,
        )
        return np.resize(td, target)

    @staticmethod
    def _coerce_2d(
        td: np.ndarray,
        n_patients: int,
        n_variables: int,
    ) -> np.ndarray:
        """Coerce a 2-D array into ``(n_patients, n_variables)``."""
        target = (n_patients, n_variables)
        rows, cols = td.shape

        # Large enough → slice
        if rows >= n_patients and cols >= n_variables:
            return td[:n_patients, :n_variables]

        # Too small → resize with warning
        logger.warning(
            "true_data smaller than expected; resizing to (%d, %d)",
            *target,
        )
        return np.resize(td, target)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class StatsUtils:
    """Static statistical helper methods."""

    @staticmethod
    def confidence_interval(
        mean: ArrayLike1D,
        std: ArrayLike1D,
        n: int,
        confidence: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a confidence interval for the given summary statistics.

        Args:
            mean: Sample means.
            std: Sample standard deviations.
            n: Sample size.
            confidence: Confidence level (default 0.95).

        Returns:
            ``(lower_bound, upper_bound)`` arrays.
        """
        mean = np.asarray(mean)
        std = np.asarray(std)

        if np.all(std == 0):
            return mean, mean

        alpha = 1 - confidence
        degrees_of_freedom = n - 1

        t_value = t_dist.ppf(1 - alpha / 2, degrees_of_freedom)
        standard_error = std / np.sqrt(n)

        lower = mean - t_value * standard_error
        upper = mean + t_value * standard_error

        return lower, upper
