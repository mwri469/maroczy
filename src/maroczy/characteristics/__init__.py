"""Characteristic (factor) computation engine.

Quickstart::

    from maroczy.characteristics import CharacteristicEngine, CharacteristicRegistry

    registry = CharacteristicRegistry()
    print(registry.names())

    engine = CharacteristicEngine(registry)
    chars = engine.compute(bars, names=["ret_12_1", "ami_126d", "retvol"])
"""

from maroczy.characteristics.engine import CharacteristicEngine
from maroczy.characteristics.registry import CharacteristicRegistry

__all__ = ["CharacteristicEngine", "CharacteristicRegistry"]
