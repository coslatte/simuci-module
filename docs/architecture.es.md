# Arquitectura (mapa rápido)

Este proyecto separa **motor de simulación** (núcleo) de **entrada/salida**, **validación**, **estadística** y **herramientas de desarrollo**.

## Núcleo (runtime / lo esencial)

- [src/simuci/core/experiment.py](src/simuci/core/experiment.py)
  - `Experiment`: contenedor de parámetros del paciente/experimento.
  - Orquesta validación de inputs (cuando `validate=True`).

- [src/simuci/core/simulation.py](src/simuci/core/simulation.py)
  - `Simulation`: lógica del motor y ejecución de la simulación.

- [src/simuci/core/distributions.py](src/simuci/core/distributions.py)
  - Muestreo de distribuciones, clustering/selección de centroides y utilidades asociadas.

## Entrada/Salida (I/O) y carga de datos

- [src/simuci/io/loaders/base.py](src/simuci/io/loaders/base.py)
  - Interfaces/bases para loaders.

- [src/simuci/io/loaders/csv_loader.py](src/simuci/io/loaders/csv_loader.py)
  - `CentroidLoader`: lectura/normalización de centroides desde CSV.

- [src/simuci/io/process_data.py](src/simuci/io/process_data.py)
  - Utilidades para preparar/procesar datos (no es parte del motor).

## Validación y contratos (schemas / reglas)

- [src/simuci/validation/validators.py](src/simuci/validation/validators.py)
  - Reglas de rango/tipo y validación de inputs de usuario.

- [src/simuci/validation/schemas.py](src/simuci/validation/schemas.py)
  - Estructuras de “contrato” (shape/campos esperados) para datos/csv.

## Estadística (evaluación/validación científica)

- [src/simuci/analysis/stats.py](src/simuci/analysis/stats.py)
  - Tests: `Wilcoxon`, `Friedman`.
  - Métricas: `SimulationMetrics` (cobertura, error, KS, AD).
  - Helpers: `StatsUtils`.

## Internos (no-API)

- [src/simuci/internals/_types.py](src/simuci/internals/_types.py)
  - Tipos, aliases, `Metric`, etc.

- [src/simuci/internals/_constants.py](src/simuci/internals/_constants.py)
  - Constantes compartidas (labels, etc.).

## Herramientas (fuera del runtime)

- [src/simuci/tooling/envcheck.py](src/simuci/tooling/envcheck.py)
  - Verificación de entorno/dependencias/imports y auditoría opcional.
  - Se ejecuta bajo demanda: `python -m simuci.envcheck`.

- [src/simuci/envcheck.py](src/simuci/envcheck.py)
  - Shim para compatibilidad con `python -m simuci.envcheck`.

## API pública (cómo se expone hacia afuera)

- [src/simuci/__init__.py](src/simuci/__init__.py)
  - Re-exporta símbolos “estables” para que el usuario haga `from simuci import ...`.
  - Regla práctica: lo que no está re-exportado aquí se considera interno/avanzado.
