"""Characteristic function implementations, keyed by the ``function`` column
of ``characteristics mapping.csv`` via the :func:`characteristic` decorator.

Each implemented characteristic is a plain function; submodules are grouped
by the CSV's ``class`` column:

- :mod:`crspd`  -- daily price/volume based (operates on an OHLCV DataFrame,
  the same shape returned by :meth:`maroczy.broker.data.MarketData.history`).
- :mod:`crspm`  -- monthly-frequency price based (also operates on an OHLCV
  DataFrame; period lengths are expressed in trading days so they work on
  daily bars too).
- :mod:`funda`  -- annual fundamentals (operates on a user-supplied
  Compustat-style fundamentals DataFrame).
- :mod:`fundq`  -- quarterly fundamentals.
- :mod:`merge`  -- characteristics that combine price and fundamental data.
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
from maroczy.characteristics.functions import crspd, crspm, funda, fundq, merge  # noqa: E402,F401

__all__ = ["characteristic", "get_function", "list_implemented", "crspd", "crspm", "funda", "fundq", "merge"]
