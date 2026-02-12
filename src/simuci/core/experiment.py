"""Experiment definition and execution for ICU simulation.

Provides the :class:`Experiment` data holder and the runner functions
:func:`single_run` and :func:`multiple_replication` that drive the
discrete-event simulation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import simpy

from simuci.core import distributions
from simuci.core.simulation import Simulation
from simuci.internals._constants import EXPERIMENT_VARIABLES_LABELS
from simuci.validation.validators import validate_experiment_inputs

logger = logging.getLogger(__name__)


class Experiment:
    """Patient-level parameters and mutable result container for one simulation run.

    Input values are validated on construction.  Pass ``validate=False``
    to skip validation (e.g. when you have already validated externally).
    """

    def __init__(
        self,
        age: int,
        diagnosis_admission1: int,
        diagnosis_admission2: int,
        diagnosis_admission3: int,
        diagnosis_admission4: int,
        apache: int,
        respiratory_insufficiency: int,
        artificial_ventilation: int,
        uti_stay: int,
        vam_time: int,
        preuti_stay_time: int,
        percent: int = 10,
        *,
        validate: bool = True,
    ) -> None:
        if validate:
            validate_experiment_inputs(
                age=age,
                apache=apache,
                respiratory_insufficiency=respiratory_insufficiency,
                artificial_ventilation=artificial_ventilation,
                uti_stay=uti_stay,
                vam_time=vam_time,
                preuti_stay_time=preuti_stay_time,
                percent=percent,
                diagnosis_admission1=diagnosis_admission1,
                diagnosis_admission2=diagnosis_admission2,
                diagnosis_admission3=diagnosis_admission3,
                diagnosis_admission4=diagnosis_admission4,
            )

        self.edad: int = age
        self.diagn1: int = diagnosis_admission1
        self.diagn2: int = diagnosis_admission2
        self.diagn3: int = diagnosis_admission3
        self.diagn4: int = diagnosis_admission4
        self.apache: int = apache
        self.insuf_resp: int = respiratory_insufficiency
        self.va: int = artificial_ventilation
        self.estadia_uti: int = uti_stay
        self.tiempo_vam: int = vam_time
        self.tiempo_pre_uti: int = preuti_stay_time
        self.porcentaje: int = percent

        self.result: dict[str, int] = {}

    def init_results_variables(self) -> None:
        """Reset result dict with zeroes for every experiment variable."""

        self.result = {var: 0 for var in EXPERIMENT_VARIABLES_LABELS}


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def single_run(
    experiment: Experiment,
    *,
    centroids_path: str | Path,
) -> dict[str, int]:
    """Execute one simulation replication and return the result dict.

    Args:
        experiment: Configured :class:`Experiment` instance.
        centroids_path: Path to a custom centroids CSV.
    """

    env = simpy.Environment()
    experiment.init_results_variables()

    cluster = distributions.clustering(
        experiment.edad,
        experiment.diagn1,
        experiment.diagn2,
        experiment.diagn3,
        experiment.diagn4,
        experiment.apache,
        experiment.insuf_resp,
        experiment.va,
        experiment.estadia_uti,
        experiment.tiempo_vam,
        experiment.tiempo_pre_uti,
        centroids_path=centroids_path,
    )

    simulation = Simulation(experiment, cluster)
    env.process(simulation.uci(env))
    env.run()

    return experiment.result


def multiple_replication(
    experiment: Experiment,
    n_reps: int = 100,
    as_int: bool = True,
    *,
    centroids_path: str | Path,
) -> pd.DataFrame:
    """Run *n_reps* independent replications and return results as a DataFrame.

    Args:
        experiment: Configured :class:`Experiment` instance.
        n_reps: Number of independent replications.
        as_int: If ``True`` cast every value to ``int64``; otherwise keep ``float64``.
        centroids_path: Path to a custom centroids CSV.

    Returns:
        A :class:`~pandas.DataFrame` with one row per replication.
    """

    results: list[dict[str, int | float]] = []

    for _ in range(n_reps):
        raw = single_run(experiment, centroids_path=centroids_path)

        row: dict[str, int | float] = {}
        for key, value in raw.items():
            try:
                val = float(value)
            except (ValueError, TypeError):
                val = 0.0
            row[key] = int(val) if as_int else val
        results.append(row)

    df = pd.DataFrame(results)

    # Fill any unexpected NaN and enforce column types
    df = df.fillna(0 if as_int else 0.0)
    target_dtype = "int64" if as_int else "float64"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(target_dtype)

    return df
