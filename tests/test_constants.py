"""Tests for simuci._constants — verifies constant integrity."""

from __future__ import annotations

from simuci.internals._constants import (
    AGE_MAX,
    AGE_MIN,
    APACHE_MAX,
    APACHE_MIN,
    EXPERIMENT_VARIABLES_FROM_CSV,
    EXPERIMENT_VARIABLES_LABELS,
    N_CLUSTERING_FEATURES,
    PREUCI_DIAG,
    PREUTI_STAY_MAX,
    PREUTI_STAY_MIN,
    RESP_INSUF,
    SIM_RUNS_MAX,
    SIM_RUNS_MIN,
    UTI_STAY_MAX,
    UTI_STAY_MIN,
    VAM_T_MAX,
    VAM_T_MIN,
    VENTILATION_TYPE,
)


class TestExperimentVariables:
    """Ensure variable lists haven't been accidentally altered."""

    def test_csv_variables_count(self) -> None:
        assert len(EXPERIMENT_VARIABLES_FROM_CSV) == 11

    def test_labels_count(self) -> None:
        assert len(EXPERIMENT_VARIABLES_LABELS) == 5

    def test_labels_are_strings(self) -> None:
        for key, label in EXPERIMENT_VARIABLES_LABELS.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(label, str) and len(label) > 0

    def test_n_clustering_features(self) -> None:
        """N_CLUSTERING_FEATURES matches the CSV feature count used by UCI."""
        assert len(EXPERIMENT_VARIABLES_FROM_CSV) == N_CLUSTERING_FEATURES


class TestLimits:
    """Ensure min ≤ max for every limit pair."""

    def test_age_range(self) -> None:
        assert AGE_MIN < AGE_MAX

    def test_apache_range(self) -> None:
        assert APACHE_MIN <= APACHE_MAX

    def test_vam_range(self) -> None:
        assert VAM_T_MIN < VAM_T_MAX

    def test_uti_stay_range(self) -> None:
        assert UTI_STAY_MIN <= UTI_STAY_MAX

    def test_preuti_stay_range(self) -> None:
        assert PREUTI_STAY_MIN <= PREUTI_STAY_MAX

    def test_sim_runs_range(self) -> None:
        assert SIM_RUNS_MIN < SIM_RUNS_MAX


class TestMappings:
    """Ensure category mappings are non-empty and well-formed."""

    def test_ventilation_type_keys(self) -> None:
        assert set(VENTILATION_TYPE.keys()) == {0, 1, 2}

    def test_resp_insuf_keys(self) -> None:
        assert set(RESP_INSUF.keys()) == {0, 1, 2, 3, 4, 5}

    def test_preuci_diag_keys(self) -> None:
        assert len(PREUCI_DIAG) == 41
        assert min(PREUCI_DIAG.keys()) == 0
        assert max(PREUCI_DIAG.keys()) == 40

    def test_all_mapping_values_are_strings(self) -> None:
        for mapping in (VENTILATION_TYPE, RESP_INSUF, PREUCI_DIAG):
            for val in mapping.values():
                assert isinstance(val, str) and len(val) > 0
