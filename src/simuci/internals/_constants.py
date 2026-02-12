"""Internal constants for the simuci simulation engine.

This module centralises domain constants that were previously scattered
across ``utils.constants.experiment``, ``utils.constants.limits`` and
``utils.constants.mappings`` in the application layer.  Only values
required by the **simulation engine** are kept here - UI-specific strings,
paths and theme colours stay in the application.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment variable names
# ---------------------------------------------------------------------------

EXPERIMENT_VARIABLES_FROM_CSV: list[str] = [
    "Edad",
    "Diag.Ing1",
    "Diag.Ing2",
    "Diag.Ing3",
    "Diag.Ing4",
    "APACHE",
    "InsufResp",
    "VA",
    "Est. UCI",
    "TiempoVAM",
    "Est. PreUCI",
]
"""Column names consumed from the research dataset (11 features)."""

EXPERIMENT_VARIABLES_LABELS: dict[str, str] = {
    "pre_vam": "Tiempo Pre VAM",
    "vam": "Tiempo VAM",
    "post_vam": "Tiempo Post VAM",
    "uci": "Estadia UCI",
    "post_uci": "Estadia Post UCI",
}
"""Keys are machine-readable identifiers, values are display labels."""

N_CLUSTERING_FEATURES: int = 11
"""Matches the 11 input variables from the CSV data used by the UCI app."""

# ---------------------------------------------------------------------------
# Input limits (validation ranges)
# ---------------------------------------------------------------------------

AGE_MIN: int = 14
AGE_MAX: int = 100
AGE_DEFAULT: int = 22

APACHE_MIN: int = 0
APACHE_MAX: int = 36
APACHE_DEFAULT: int = 12

VAM_T_MIN: int = 24
VAM_T_MAX: int = 700
VAM_T_DEFAULT: int = VAM_T_MIN

UTI_STAY_MIN: int = 0
UTI_STAY_MAX: int = 200
UTI_STAY_DEFAULT: int = 24

PREUTI_STAY_MIN: int = 0
PREUTI_STAY_MAX: int = 34
PREUTI_STAY_DEFAULT: int = 10

SIM_RUNS_MIN: int = 50
SIM_RUNS_MAX: int = 100_000
SIM_RUNS_DEFAULT: int = 200

SIM_PERCENT_MIN: int = 0
SIM_PERCENT_MAX: int = 10
SIM_PERCENT_DEFAULT: int = 3

# ---------------------------------------------------------------------------
# Category mappings
# ---------------------------------------------------------------------------

VENTILATION_TYPE: dict[int, str] = {
    0: "Tubo endotraqueal",
    1: "Traqueostomía",
    2: "Ambas",
}

PREUCI_DIAG: dict[int, str] = {
    0: "Vacío",
    1: "Intoxicación exógena",
    2: "Coma",
    3: "Trauma craneoencefálico severo",
    4: "SPO de toracotomía",
    5: "SPO de laparotomía",
    6: "SPO de amputación",
    7: "SPO de neurología",
    8: "PCR recuperado",
    9: "Encefalopatía metabólica",
    10: "Encefalopatía hipóxica",
    11: "Ahorcamiento incompleto",
    12: "Insuficiencia cardiaca descompensada",
    13: "Obstétrica grave",
    14: "EPOC descompensada",
    15: "ARDS",
    16: "BNB-EH",
    17: "BNB-IH",
    18: "BNV",
    19: "Miocarditis",
    20: "Leptospirosis",
    21: "Sepsis grave",
    22: "DMO",
    23: "Shock séptico",
    24: "Shock hipovolémico",
    25: "Shock cardiogénico",
    26: "IMA",
    27: "Politraumatizado",
    28: "Crisis miasténica",
    29: "Emergencia hipertensiva",
    30: "Status asmático",
    31: "Status epiléptico",
    32: "Pancreatitis",
    33: "Embolismo graso",
    34: "Accidente cerebrovascular",
    35: "Síndrome de apnea del sueño",
    36: "Sangramiento digestivo",
    37: "Insuficiencia renal crónica",
    38: "Insuficiencia renal aguda",
    39: "Trasplante renal",
    40: "Guillain Barré",
}

RESP_INSUF: dict[int, str] = {
    0: "Vacío",
    1: "Respiratorias",
    2: "TCE",
    3: "Estatus posoperatorio",
    4: "Afecciones no traumáticas del SNC",
    5: "Causas extrapulmonares",
}
