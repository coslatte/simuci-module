"""Tests for simuci.validators — input validation."""

from __future__ import annotations

import pytest

from simuci.internals._constants import (
    AGE_MAX,
    AGE_MIN,
    APACHE_MAX,
    APACHE_MIN,
    PREUTI_STAY_MAX,
    SIM_RUNS_MAX,
    SIM_RUNS_MIN,
    UTI_STAY_MAX,
    VAM_T_MAX,
    VAM_T_MIN,
)
from simuci.validation.validators import validate_experiment_inputs, validate_simulation_runs


class TestValidateExperimentInputs:
    """Test each validation rule in validate_experiment_inputs."""

    def test_valid_inputs_pass(self, valid_params: dict) -> None:
        validate_experiment_inputs(**valid_params)  # should not raise

    # ---- age ----
    def test_age_too_low(self, valid_params: dict) -> None:
        valid_params["age"] = AGE_MIN - 1
        with pytest.raises(ValueError, match="age"):
            validate_experiment_inputs(**valid_params)

    def test_age_too_high(self, valid_params: dict) -> None:
        valid_params["age"] = AGE_MAX + 1
        with pytest.raises(ValueError, match="age"):
            validate_experiment_inputs(**valid_params)

    def test_age_boundary_min(self, valid_params: dict) -> None:
        valid_params["age"] = AGE_MIN
        validate_experiment_inputs(**valid_params)

    def test_age_boundary_max(self, valid_params: dict) -> None:
        valid_params["age"] = AGE_MAX
        validate_experiment_inputs(**valid_params)

    # ---- apache ----
    def test_apache_too_high(self, valid_params: dict) -> None:
        valid_params["apache"] = APACHE_MAX + 1
        with pytest.raises(ValueError, match="apache"):
            validate_experiment_inputs(**valid_params)

    def test_apache_too_low(self, valid_params: dict) -> None:
        valid_params["apache"] = APACHE_MIN - 1
        with pytest.raises(ValueError, match="apache"):
            validate_experiment_inputs(**valid_params)

    # ---- vam_time ----
    def test_vam_time_too_low(self, valid_params: dict) -> None:
        valid_params["vam_time"] = VAM_T_MIN - 1
        with pytest.raises(ValueError, match="vam_time"):
            validate_experiment_inputs(**valid_params)

    def test_vam_time_too_high(self, valid_params: dict) -> None:
        valid_params["vam_time"] = VAM_T_MAX + 1
        with pytest.raises(ValueError, match="vam_time"):
            validate_experiment_inputs(**valid_params)

    # ---- uti_stay ----
    def test_uti_stay_too_high(self, valid_params: dict) -> None:
        valid_params["uti_stay"] = UTI_STAY_MAX + 1
        with pytest.raises(ValueError, match="uti_stay"):
            validate_experiment_inputs(**valid_params)

    # ---- preuti_stay_time ----
    def test_preuti_stay_too_high(self, valid_params: dict) -> None:
        valid_params["preuti_stay_time"] = PREUTI_STAY_MAX + 1
        with pytest.raises(ValueError, match="preuti_stay_time"):
            validate_experiment_inputs(**valid_params)

    # ---- category keys ----
    def test_invalid_ventilation_type(self, valid_params: dict) -> None:
        valid_params["artificial_ventilation"] = 99
        with pytest.raises(ValueError, match="artificial_ventilation"):
            validate_experiment_inputs(**valid_params)

    def test_invalid_resp_insuf(self, valid_params: dict) -> None:
        valid_params["respiratory_insufficiency"] = 99
        with pytest.raises(ValueError, match="respiratory_insufficiency"):
            validate_experiment_inputs(**valid_params)

    def test_invalid_diagnosis(self, valid_params: dict) -> None:
        valid_params["diagnosis_admission1"] = 99
        with pytest.raises(ValueError, match="diagnosis_admission1"):
            validate_experiment_inputs(**valid_params)


class TestValidateSimulationRuns:
    """Test simulation run count validation."""

    def test_valid_runs(self) -> None:
        validate_simulation_runs(200)

    def test_too_few_runs(self) -> None:
        with pytest.raises(ValueError, match="n_reps"):
            validate_simulation_runs(SIM_RUNS_MIN - 1)

    def test_too_many_runs(self) -> None:
        with pytest.raises(ValueError, match="n_reps"):
            validate_simulation_runs(SIM_RUNS_MAX + 1)

    def test_boundary_min(self) -> None:
        validate_simulation_runs(SIM_RUNS_MIN)

    def test_boundary_max(self) -> None:
        validate_simulation_runs(SIM_RUNS_MAX)
