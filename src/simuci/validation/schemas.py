"""Data schemas and structured types for CSV inputs.

These schemas document the expected column structure so that loaders
can validate data without exposing proprietary datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

# ---------------------------------------------------------------------------
# Centroid CSV schema
# ---------------------------------------------------------------------------


class CentroidRow(TypedDict):
    """Schema for a single centroid row (3 clusters x 18 numeric columns)."""

    cluster_id: int
    features: list[float]


@dataclass(frozen=True)
class CentroidSchema:
    """Describes the expected shape of the centroids CSV.

    The CSV has an index column (cluster id) and 18 numeric columns
    named ``"0"`` through ``"17"``.  Only the first
    :data:`~simuci._constants.N_CLUSTERING_FEATURES` columns are used
    by the clustering function.
    """

    n_clusters: int = 3
    n_total_columns: int = 18
    n_used_columns: int = 11
    index_column: str = ""
    feature_columns: list[str] = field(default_factory=lambda: [str(i) for i in range(18)])
