"""CSV-driven characteristic (factor) generation engine.

Quickstart::

    from maroczy.characteristics import CharacteristicEngine, CharacteristicRegistry

    registry = CharacteristicRegistry()
    print(registry.coverage())            # how many of your 300+ characteristics are implemented
    print(registry.names(data_class="crspd"))

    engine = CharacteristicEngine(registry)
    chars = engine.compute(bars, names=["ret_12_1", "ami_126d", "retvol"])
"""

from maroczy.characteristics.engine import CharacteristicEngine
from maroczy.characteristics.registry import CharacteristicRegistry

__all__ = ["CharacteristicEngine", "CharacteristicRegistry"]
