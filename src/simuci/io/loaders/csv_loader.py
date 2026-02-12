"""CSV-based loader for centroid data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from simuci.internals._constants import N_CLUSTERING_FEATURES
from simuci.io.loaders.base import BaseLoader
from simuci.validation.schemas import CentroidSchema

logger = logging.getLogger(__name__)

_SCHEMA = CentroidSchema()


class CentroidLoader(BaseLoader):
    """Load and validate centroid data from a CSV file.

    The CSV is expected to have an index column and at least
    :data:`~simuci._constants.N_CLUSTERING_FEATURES` numeric columns.

    Example::

        loader = CentroidLoader()
        centroids = loader.load("path/to/centroids.csv")
    """

    def load(self, path: str | Path) -> np.ndarray:
        """Load centroids and return an ``(n_clusters, n_features)`` array.

        Args:
            path: Path to the centroids CSV file.

        Returns:
            Numpy array of shape ``(n_clusters, N_CLUSTERING_FEATURES)``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the CSV does not match the expected schema.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Centroid file not found: {path}")

        df = pd.read_csv(path)
        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

        if len(numeric_cols) < N_CLUSTERING_FEATURES:
            raise ValueError(
                f"Centroid CSV must have at least {N_CLUSTERING_FEATURES} numeric columns, "
                f"got {len(numeric_cols)}"
            )

        if len(df) != _SCHEMA.n_clusters:
            logger.warning(
                "Expected %d cluster rows, got %d — using all rows",
                _SCHEMA.n_clusters,
                len(df),
            )

        centroids = df[numeric_cols[:N_CLUSTERING_FEATURES]].to_numpy(dtype=float)
        logger.debug("Loaded centroids with shape %s from %s", centroids.shape, path)

        return centroids
