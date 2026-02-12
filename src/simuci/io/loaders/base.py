"""Abstract base for data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLoader(ABC):
    """Interface for loading data required by the simulation engine.

    Concrete loaders validate the data against known schemas before
    returning it, ensuring early failure with clear error messages.
    """

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Load and validate data from *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If data does not match the expected schema.
        """
        ...
