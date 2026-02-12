"""Type aliases for the simuci simulation engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Core aliases
# ---------------------------------------------------------------------------

type ClusterId = int
"""Cluster index returned by :func:`simuci.distribuciones.clustering` (0, 1, or 2)."""

type SimulationResult = dict[str, int]
"""Single-replication result mapping variable labels to hour values."""

type ArrayLike1D = Sequence[float] | np.ndarray | Any
"""Anything that can act as a 1-D array of floats."""

type Metric = tuple[float, ...] | dict[str, Any]
"""Return type of individual metric calculations in :class:`SimulationMetrics`."""
