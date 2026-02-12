# simuci

Motor de simulación de eventos discretos para UCI — muestreo de distribuciones, clustering de pacientes y validación estadística.

## Instalación

```bash
pip install simuci
```

Para desarrollo:

```bash
git clone https://github.com/coslatte/simuci.git
cd simuci
pip install -e ".[dev]"
```

## Inicio Rápido

```python
from simuci import Experiment, single_run, multiple_replication

# Crear un experimento con parámetros del paciente
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

# Ejecución única (requiere CSV de centroides)
result = single_run(exp, centroids_path="ruta/a/centroides.csv")
print(result)
# {'Tiempo Pre VAM': 5, 'Tiempo VAM': 89, 'Tiempo Post VAM': 168, 'Estadia UCI': 262, 'Estadia Post UCI': 45}

# Múltiples réplicas → DataFrame
df = multiple_replication(exp, n_reps=200, centroids_path="ruta/a/centroides.csv")
print(df.describe())
```

## Usando tus propios Datos de Centroides

Debes pasar explícitamente la ruta a tu CSV de centroides:

```python
from simuci import single_run, Experiment

exp = Experiment(age=55, ..., validate=False)

# Apunta a tu CSV de centroides
result = single_run(exp, centroids_path="ruta/a/centroides_reales.csv")
```

El CSV de centroides debe tener:

- Una columna índice (IDs de cluster: 0, 1, 2)
- Al menos 11 columnas numéricas (características usadas para la clasificación nearest-centroid)

También puedes usar el cargador directamente:

```python
from simuci.loaders import CentroidLoader

loader = CentroidLoader()
centroids = loader.load("ruta/a/centroides.csv")  # devuelve numpy array
```

## Validación Estadística

```python
import numpy as np
from simuci import SimulationMetrics, Wilcoxon, Friedman

# Comparar salida de simulación con datos reales
metrics = SimulationMetrics(
    true_data=np.array(...),       # (n_pacientes, n_variables)
    simulation_data=np.array(...), # (n_pacientes, n_replicates, n_variables)
)
metrics.evaluate(confidence_level=0.95, result_as_dict=True)

print(metrics.coverage_percentage)
print(metrics.error_margin)
print(metrics.kolmogorov_smirnov_result)
print(metrics.anderson_darling_result)
```

## Validación de Entrada

Todos los inputs de `Experiment` se validan en la construcción por defecto:

```python
from simuci import Experiment

# Esto lanza ValueError: age debe estar entre 14 y 100
Experiment(age=200, ...)
```

Salta la validación con `validate=False` si ya has validado externamente.

## Referencia de API

| Símbolo | Descripción |
|--------|-------------|
| `Experiment` | Parámetros del paciente + contenedor de resultados |
| `single_run(exp)` | Una réplica de simulación |
| `multiple_replication(exp, n_reps)` | N réplicas → DataFrame |
| `clustering(edad, ...)` | Clasificador de pacientes nearest-centroid |
| `Wilcoxon` | Test de rangos con signo de Wilcoxon pareado |
| `Friedman` | Test chi-cuadrado de Friedman |
| `SimulationMetrics` | Suite completa de evaluación (cobertura, RMSE, KS, AD) |
| `StatsUtils` | Helper estático de intervalo de confianza |
| `CentroidLoader` | Cargador CSV con validación de esquema |
| `validate_experiment_inputs()` | Verificación de rango de parámetros |

## Arquitectura

Mapa del proyecto (núcleo vs. validación, I/O, estadística, herramientas):
[docs/architecture.es.md](docs/architecture.es.md)

## Licencia

MIT
