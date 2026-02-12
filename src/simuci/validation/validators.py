"""Input validation for experiment parameters.

All validators raise :class:`ValueError` with descriptive messages when
a value falls outside the accepted range.  The ranges are defined in
:mod:`simuci._constants`.
"""

from __future__ import annotations

from simuci.internals._constants import (
    AGE_MAX,
    AGE_MIN,
    APACHE_MAX,
    APACHE_MIN,
    PREUCI_DIAG,
    PREUTI_STAY_MAX,
    PREUTI_STAY_MIN,
    RESP_INSUF,
    SIM_PERCENT_MAX,
    SIM_PERCENT_MIN,
    SIM_RUNS_MAX,
    SIM_RUNS_MIN,
    UTI_STAY_MAX,
    UTI_STAY_MIN,
    VAM_T_MAX,
    VAM_T_MIN,
    VENTILATION_TYPE,
)


def _check_range(name: str, value: int, lo: int, hi: int) -> None:
    """Raise ``ValueError`` if *value* is outside [*lo*, *hi*]."""

    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")


def _check_key(name: str, value: int, mapping: dict[int, str]) -> None:
    """Raise ``ValueError`` if *value* is not a valid key in *mapping*."""

    if value not in mapping:
        valid = sorted(mapping)
        raise ValueError(f"{name} must be one of {valid}, got {value}")


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------


def validate_experiment_inputs(
    *,
    age: int,
    apache: int,
    respiratory_insufficiency: int,
    artificial_ventilation: int,
    uti_stay: int,
    vam_time: int,
    preuti_stay_time: int,
    percent: int = 3,
    diagnosis_admission1: int = 0,
    diagnosis_admission2: int = 0,
    diagnosis_admission3: int = 0,
    diagnosis_admission4: int = 0,
) -> None:
    """Validate all experiment inputs against known limits.

    Raises:
        ValueError: If any parameter is outside its valid range / set.
    """

    _check_range("age", age, AGE_MIN, AGE_MAX)
    _check_range("apache", apache, APACHE_MIN, APACHE_MAX)
    _check_range("vam_time", vam_time, VAM_T_MIN, VAM_T_MAX)
    _check_range("uti_stay", uti_stay, UTI_STAY_MIN, UTI_STAY_MAX)
    _check_range("preuti_stay_time", preuti_stay_time, PREUTI_STAY_MIN, PREUTI_STAY_MAX)
    _check_range("percent", percent, SIM_PERCENT_MIN, SIM_PERCENT_MAX)

    _check_key("respiratory_insufficiency", respiratory_insufficiency, RESP_INSUF)
    _check_key("artificial_ventilation", artificial_ventilation, VENTILATION_TYPE)

    for i, diag in enumerate(
        [diagnosis_admission1, diagnosis_admission2, diagnosis_admission3, diagnosis_admission4],
        start=1,
    ):
        _check_key(f"diagnosis_admission{i}", diag, PREUCI_DIAG)


def validate_simulation_runs(n_reps: int) -> None:
    """Validate the number of simulation replications.

    Raises:
        ValueError: If *n_reps* is outside the allowed range.
    """

    _check_range("n_reps", n_reps, SIM_RUNS_MIN, SIM_RUNS_MAX)
