"""Characteristic function implementations registered via the
:func:`characteristic` decorator.

Each implemented characteristic is a plain function; submodules are grouped
by input data type:

- :mod:`price_daily`  -- daily price/volume based (operates on an OHLCV
  DataFrame indexed by date).
- :mod:`price_monthly`  -- monthly-frequency price based (also operates on
  an OHLCV DataFrame; period lengths are expressed in trading days so they
  work on daily bars too).
- :mod:`fundamentals_annual`  -- annual financial-statement ratios.
- :mod:`fundamentals_quarterly`  -- quarterly financial-statement ratios.
- :mod:`composite`  -- characteristics that combine price and fundamental data.
"""

from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def characteristic(name: str) -> Callable:
    """Decorator registering ``func`` as the implementation of characteristic ``name``.

    ``name`` should match the ``function`` column of ``characteristics
    mapping.csv`` so the registry/engine can discover it automatically.
    """

    def decorator(func: Callable) -> Callable:
        _REGISTRY[name] = func
        return func

    return decorator


def get_function(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"No implementation registered for characteristic {name!r}.")
    return _REGISTRY[name]


def list_implemented() -> list[str]:
    return sorted(_REGISTRY.keys())


# Import submodules so their @characteristic-decorated functions self-register.
from maroczy.characteristics.functions import (  # noqa: E402,F401
    alternative,
    composite,
    fundamentals_annual,
    fundamentals_quarterly,
    macro_derived,
    options_derived,
    price_daily,
    price_monthly,
)

__all__ = [
    "characteristic",
    "get_function",
    "list_implemented",
    "price_daily",
    "price_monthly",
    "fundamentals_annual",
    "fundamentals_quarterly",
    "composite",
    "options_derived",
    "macro_derived",
    "alternative",
]
