from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ResultService.result import Result


class VisualizationObject(ABC):
    """Abstract class for visualizations."""

    @abstractmethod
    def __init__(self, result: Result):
        """Initialize the visualization."""
        self.result = result

    @abstractmethod
    def default_output(self):
        """Plot the visualization."""
        pass