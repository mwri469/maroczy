"""Batch characteristic computation engine.

Designed for one-liner notebook usage::

    engine = CharacteristicEngine()
    chars = engine.compute(bars, names=["ret_12_1", "ami_126d", "retvol"])
    panel = engine.compute_universe({"AAPL": aapl_bars, "MSFT": msft_bars})
"""

from __future__ import annotations

import inspect
import logging

import pandas as pd

from maroczy.characteristics.functions import get_function
from maroczy.characteristics.registry import CharacteristicRegistry

logger = logging.getLogger("maroczy.characteristics.engine")

__all__ = ["CharacteristicEngine"]


class CharacteristicEngine:
    def __init__(self, registry: CharacteristicRegistry | None = None):
        self.registry = registry or CharacteristicRegistry()

    def compute(
        self,
        data,
        names: list[str] | None = None,
        data_class: str | None = None,
        **extra_kwargs,
    ) -> pd.DataFrame:
        """Compute a batch of characteristics for one security/fundamentals frame.

        Parameters
        ----------
        data: the primary input each characteristic function expects as its
            first positional argument (e.g. an OHLCV DataFrame for
            ``crspd``/``crspm`` characteristics, or a fundamentals DataFrame
            for ``funda``/``fundq``).
        names: characteristic names to compute; defaults to every
            *implemented* characteristic (optionally filtered by
            ``data_class``).
        data_class: restrict the default ``names`` to a CSV ``class``
            (``"crspd"``, ``"crspm"``, ``"funda"``, ``"fundq"``, ``"merge"``).
        extra_kwargs: forwarded to whichever characteristic functions accept
            a matching keyword (e.g. ``mkt_ret=...``, ``shares_out=...``,
            ``me=...``); functions that don't declare a given keyword simply
            don't receive it.

        Returns
        -------
        DataFrame aligned to ``data``'s index with one column per requested
        characteristic (characteristics that raised an error are skipped
        with a warning, not included in the output).
        """
        names = names or self.registry.names(data_class=data_class, implemented_only=True)
        results: dict[str, pd.Series] = {}
        for name in names:
            try:
                func = get_function(name)
            except KeyError:
                logger.warning("Skipping %r: no implementation registered.", name)
                continue
            sig = inspect.signature(func)
            kwargs = {k: v for k, v in extra_kwargs.items() if k in sig.parameters}
            try:
                results[name] = func(data, **kwargs)
            except Exception as exc:  # noqa: BLE001 - report and continue for the whole batch
                logger.warning("Failed computing %r: %s", name, exc)
        if not results:
            return pd.DataFrame(index=getattr(data, "index", None))
        return pd.DataFrame(results)

    def compute_universe(
        self,
        panel: dict[str, pd.DataFrame],
        names: list[str] | None = None,
        data_class: str | None = None,
        **extra_kwargs,
    ) -> pd.DataFrame:
        """Compute characteristics across a universe of symbols.

        Parameters
        ----------
        panel: mapping ``symbol -> data`` (same ``data`` shape as
            :meth:`compute` expects).

        Returns
        -------
        DataFrame with a ``(date, symbol)`` MultiIndex.
        """
        frames = []
        for symbol, data in panel.items():
            char_df = self.compute(data, names=names, data_class=data_class, **extra_kwargs)
            char_df = char_df.copy()
            char_df["symbol"] = symbol
            frames.append(char_df.set_index("symbol", append=True))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()
