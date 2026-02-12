# simuci

> 🇪🇸 **Español**: [Léeme en español](README.es.md)

ICU discrete-event simulation engine — distribution sampling, patient clustering, and statistical validation.

## Installation

```bash
pip install simuci
```

For development:

```bash
git clone https://github.com/coslatte/simuci.git
cd simuci
pip install -e ".[dev]"
```

## Quick Start

```python
from simuci import Experiment, single_run, multiple_replication

# Create an experiment with patient parameters
exp = Experiment(
    age=55,
    diagnosis_admission1=11,
    diagnosis_admission2=0,
    diagnosis_admission3=0,
    diagnosis_admission4=0,
    apache=20,
    respiratory_insufficiency=5,
    artificial_ventilation=1,
    uti_stay=100,
    vam_time=50,
    preuti_stay_time=10,
    percent=3,
)

# Single replication (requires centroids CSV)
result = single_run(exp, centroids_path="path/to/centroids.csv")
print(result)
# {'Tiempo Pre VAM': 5, 'Tiempo VAM': 89, 'Tiempo Post VAM': 168, 'Estadia UCI': 262, 'Estadia Post UCI': 45}

# Multiple replications → DataFrame
df = multiple_replication(exp, n_reps=200, centroids_path="path/to/centroids.csv")
print(df.describe())
```

## Using Your Own Centroid Data

You must pass the path to your centroid CSV explicitly:

```python
from simuci import single_run, Experiment

exp = Experiment(age=55, ..., validate=False)

# Point to your centroids CSV
result = single_run(exp, centroids_path="path/to/real_centroids.csv")
```

The centroids CSV must have:

- An index column (cluster IDs: 0, 1, 2)
- At least 11 numeric columns (features used for nearest-centroid classification)

You can also use the loader directly:

```python
from simuci.loaders import CentroidLoader

loader = CentroidLoader()
centroids = loader.load("path/to/centroids.csv")  # returns numpy array
```

## Statistical Validation

```python
import numpy as np
from simuci import SimulationMetrics, Wilcoxon, Friedman

# Compare simulation output to real data
metrics = SimulationMetrics(
    true_data=np.array(...),       # (n_patients, n_variables)
    simulation_data=np.array(...), # (n_patients, n_replicates, n_variables)
)
metrics.evaluate(confidence_level=0.95, result_as_dict=True)

print(metrics.coverage_percentage)
print(metrics.error_margin)
print(metrics.kolmogorov_smirnov_result)
print(metrics.anderson_darling_result)
```

## Input Validation

All `Experiment` inputs are validated on construction by default:

```python
from simuci import Experiment

# This raises ValueError: age must be between 14 and 100
Experiment(age=200, ...)
```

Skip validation with `validate=False` if you've already validated externally.

## API Reference

| Symbol | Description |
|--------|-------------|
| `Experiment` | Patient parameters + result container |
| `single_run(exp)` | One simulation replication |
| `multiple_replication(exp, n_reps)` | N replications → DataFrame |
| `clustering(edad, ...)` | Nearest-centroid patient classifier |
| `Wilcoxon` | Paired Wilcoxon signed-rank test |
| `Friedman` | Friedman chi-square test |
| `SimulationMetrics` | Full evaluation suite (coverage, RMSE, KS, AD) |
| `StatsUtils` | Static CI helper |
| `CentroidLoader` | CSV loader with schema validation |
| `validate_experiment_inputs()` | Parameter range checking |

## Architecture

Project map (core vs. validation, I/O, statistics, tooling):
[docs/architecture.md](docs/architecture.md)

## License

MIT
