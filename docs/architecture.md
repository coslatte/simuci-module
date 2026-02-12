# Architecture (Quick Map)

This project separates the **simulation engine** (core) from **input/output**, **validation**, **statistics**, and **development tooling**.

## Core (runtime / essentials)

- [src/simuci/core/experiment.py](src/simuci/core/experiment.py)
  - `Experiment`: Container for patient/experiment parameters.
  - Orchestrates input validation (when `validate=True`).

- [src/simuci/core/simulation.py](src/simuci/core/simulation.py)
  - `Simulation`: Engine logic and simulation execution.

- [src/simuci/core/distributions.py](src/simuci/core/distributions.py)
  - Distribution sampling, patient clustering/centroid selection, and associated utilities.

## Input/Output (I/O) and Data Loading

- [src/simuci/io/loaders/base.py](src/simuci/io/loaders/base.py)
  - Interfaces/bases for loaders.

- [src/simuci/io/loaders/csv_loader.py](src/simuci/io/loaders/csv_loader.py)
  - `CentroidLoader`: Reading/normalization of centroids from CSV.

- [src/simuci/io/process_data.py](src/simuci/io/process_data.py)
  - Utilities for preparing/processing data (not part of the engine).

## Validation and Contracts (schemas / rules)

- [src/simuci/validation/validators.py](src/simuci/validation/validators.py)
  - Range/type rules and user input validation.

- [src/simuci/validation/schemas.py](src/simuci/validation/schemas.py)
  - "Contract" structures (shape/expected fields) for data/csv.

## Statistics (scientific evaluation/validation)

- [src/simuci/analysis/stats.py](src/simuci/analysis/stats.py)
  - Tests: `Wilcoxon`, `Friedman`.
  - Metrics: `SimulationMetrics` (coverage, margin of error, KS, AD).
  - Helpers: `StatsUtils`.

## Internals (Non-API)

- [src/simuci/internals/_types.py](src/simuci/internals/_types.py)
  - Types, aliases, `Metric`, etc.

- [src/simuci/internals/_constants.py](src/simuci/internals/_constants.py)
  - Shared constants (labels, etc.).

## Opt-in Tooling (Non-runtime)

- [src/simuci/tooling/envcheck.py](src/simuci/tooling/envcheck.py)
  - Verification of environment/dependencies/imports and optional audit.
  - Executed on demand: `python -m simuci.envcheck`.

- [src/simuci/envcheck.py](src/simuci/envcheck.py)
  - Shim for backward compatibility with `python -m simuci.envcheck`.

## Public API (Exposed Surface)

- [src/simuci/__init__.py](src/simuci/__init__.py)
  - Re-exports "stable" symbols so users can do `from simuci import ...`.
  - Rule of thumb: what is not re-exported here is considered internal/advanced.
