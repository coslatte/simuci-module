"""simuci — ICU discrete-event simulation engine.

Public API
----------
Core simulation:
    Experiment, single_run, multiple_replication, Simulation

Clustering & distributions:
    clustering, clear_centroid_cache

Statistical validation:
    Wilcoxon, Friedman, SimulationMetrics, StatsUtils

Input validation:
    validate_experiment_inputs, validate_simulation_runs

Data loading:
    CentroidLoader

CSV utilities:
    process_data  (sub-module)
"""

from __future__ import annotations

from simuci.core.distributions import clear_centroid_cache, clustering
from simuci.core.experiment import Experiment, multiple_replication, single_run
from simuci.core.simulation import Simulation
from simuci.analysis.stats import Friedman, SimulationMetrics, StatsUtils, Wilcoxon
from simuci.validation.validators import validate_experiment_inputs, validate_simulation_runs
from simuci.io.loaders import CentroidLoader

__version__ = "0.1.0"

__all__ = [
    # Core
    "Experiment",
    "Simulation",
    "single_run",
    "multiple_replication",

    # Clustering
    "clustering",
    "clear_centroid_cache",

    # Statistics
    "Wilcoxon",
    "Friedman",
    "SimulationMetrics",
    "StatsUtils",

    # Validation
    "validate_experiment_inputs",
    "validate_simulation_runs",

    # Loaders
    "CentroidLoader",
]
