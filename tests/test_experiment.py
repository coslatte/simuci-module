"""Tests for simuci.experiment — Experiment, single_run, multiple_replication."""

from __future__ import annotations

import pandas as pd
import pytest

from simuci import Experiment, multiple_replication, single_run
from simuci.internals._constants import EXPERIMENT_VARIABLES_LABELS


class TestExperiment:
    """Experiment construction and init_results_variables."""

    def test_valid_construction(self, valid_params: dict) -> None:
        exp = Experiment(**valid_params)
        assert exp.edad == valid_params["age"]
        assert exp.apache == valid_params["apache"]

    def test_invalid_age_raises(self, valid_params: dict) -> None:
        valid_params["age"] = 999
        with pytest.raises(ValueError, match="age"):
            Experiment(**valid_params)

    def test_validate_false_skips_checks(self, valid_params: dict) -> None:
        valid_params["age"] = 999
        exp = Experiment(**valid_params, validate=False)
        assert exp.edad == 999

    def test_init_results_variables(self, experiment: Experiment) -> None:
        experiment.init_results_variables()
        assert set(experiment.result.keys()) == set(EXPERIMENT_VARIABLES_LABELS.keys())
        assert all(v == 0 for v in experiment.result.values())

    def test_result_starts_empty(self, experiment: Experiment) -> None:
        assert experiment.result == {}


class TestSingleRun:
    """single_run produces a valid result dict."""

    def test_returns_dict(self, experiment: Experiment, centroids_csv: str) -> None:
        result = single_run(experiment, centroids_path=centroids_csv)
        assert isinstance(result, dict)

    def test_result_keys_match_labels(self, experiment: Experiment, centroids_csv: str) -> None:
        result = single_run(experiment, centroids_path=centroids_csv)
        assert set(result.keys()) == set(EXPERIMENT_VARIABLES_LABELS.keys())

    def test_result_values_are_int(self, experiment: Experiment, centroids_csv: str) -> None:
        result = single_run(experiment, centroids_path=centroids_csv)
        for v in result.values():
            assert isinstance(v, int)

    def test_uci_stay_is_non_negative(self, experiment: Experiment, centroids_csv: str) -> None:
        result = single_run(experiment, centroids_path=centroids_csv)
        assert result["uci"] >= 0

    def test_vam_leq_uci(self, experiment: Experiment, centroids_csv: str) -> None:
        """VAM time should never exceed UCI stay."""
        for _ in range(20):
            result = single_run(experiment, centroids_path=centroids_csv)
            assert result["vam"] <= result["uci"]

    def test_time_components_sum(self, experiment: Experiment, centroids_csv: str) -> None:
        """pre_vam + vam + post_vam == uci stay."""
        for _ in range(20):
            r = single_run(experiment, centroids_path=centroids_csv)
            assert r["pre_vam"] + r["vam"] + r["post_vam"] == r["uci"]


class TestMultipleReplication:
    """multiple_replication produces a proper DataFrame."""

    def test_returns_dataframe(self, experiment: Experiment, centroids_csv: str) -> None:
        df = multiple_replication(experiment, n_reps=50, centroids_path=centroids_csv)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self, experiment: Experiment, centroids_csv: str) -> None:
        n = 50
        df = multiple_replication(experiment, n_reps=n, centroids_path=centroids_csv)
        assert df.shape == (n, len(EXPERIMENT_VARIABLES_LABELS))

    def test_columns_match_labels(self, experiment: Experiment, centroids_csv: str) -> None:
        df = multiple_replication(experiment, n_reps=50, centroids_path=centroids_csv)
        assert list(df.columns) == list(EXPERIMENT_VARIABLES_LABELS.keys())

    def test_as_int_dtype(self, experiment: Experiment, centroids_csv: str) -> None:
        df = multiple_replication(experiment, n_reps=50, as_int=True, centroids_path=centroids_csv)
        for col in df.columns:
            assert df[col].dtype == "int64"

    def test_as_float_dtype(self, experiment: Experiment, centroids_csv: str) -> None:
        df = multiple_replication(experiment, n_reps=50, as_int=False, centroids_path=centroids_csv)
        for col in df.columns:
            assert df[col].dtype == "float64"

    def test_no_nans(self, experiment: Experiment, centroids_csv: str) -> None:
        df = multiple_replication(experiment, n_reps=50, centroids_path=centroids_csv)
        assert not df.isna().any().any()
