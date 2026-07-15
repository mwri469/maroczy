"""Characteristic registry backed by the function decorator system.

Provides a simple query interface over the implemented characteristics
without requiring external CSV files at runtime.
"""

from __future__ import annotations

from maroczy.characteristics.functions import _REGISTRY, get_function, list_implemented

__all__ = ["CharacteristicRegistry"]


class CharacteristicRegistry:
    """Registry over implemented characteristics.

    Discovers characteristics from the ``@characteristic`` decorator registry
    (no external files needed at runtime).
    """

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __repr__(self) -> str:
        return f"CharacteristicRegistry({len(self)} implemented characteristics)"

    def names(self) -> list[str]:
        """List all implemented characteristic names."""
        return list_implemented()

    def get(self, name: str):
        """Return the implementation function for a given characteristic name."""
        return get_function(name)

    def coverage(self) -> dict:
        """Summary of implemented characteristics."""
        return {"implemented": len(self)}
