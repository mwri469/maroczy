"""Protocol for choosing between the live IBKR session and the London
Strategic Edge (LSE) databank when fetching market data.

Rule of thumb
-------------
**IBKR** (:class:`maroczy.broker.data.MarketData`) is your *execution-grade,
live* source: prices you can actually trade against right now, the only
source for your live positions/orders/account state, and the right choice
for anything not in the LSE catalog (a niche contract, an option/future you
already hold, a symbol quoted only on your exchange).

**LSE** (:class:`maroczy.datafeed.lse.LSEData`) is your *deep-history,
cross-asset research* source: years of candles down to 1-second resolution
without IBKR's historical-data pacing limits, plus fundamentals, macro
economics, bond yields, options chains/flow with greeks, and reference data
(insider trades, dividends, splits) that IBKR does not expose cleanly. It's
the natural home for characteristic generation and backtesting over a large
universe/long history.

:class:`UnifiedMarketData` implements this as a concrete, overridable
decision procedure (see :meth:`UnifiedMarketData.resolve_source`):

1. An explicit ``source="ibkr"``/``source="lse"`` always wins.
2. Real-time / streaming requests (``realtime=True``) require a connected
   IBKR session -- LSE candles are historical-only in this client (use
   :meth:`~maroczy.datafeed.lse.LSEData.stream` directly for LSE's own
   live WebSocket feed if you want LSE ticks instead).
3. Intraday bar sizes (sub-daily) requested over a lookback window longer
   than ``policy.ibkr_intraday_limit_days`` (IBKR paces/limits how much
   fine-grained intraday history you can pull per session) route to LSE
   if configured.
4. Otherwise, plain daily-or-coarser history routes to LSE by default when
   configured (``policy.prefer_lse_for_history``, deeper history, no
   pacing limits, one flat API), falling back to IBKR if LSE isn't set up.
5. If the chosen source raises and ``policy.fallback`` is True, the other
   configured source is tried once before the error propagates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from maroczy.broker.data import MarketData
from maroczy.datafeed.lse import IBKR_TO_LSE_TIMEFRAME, LSEData

logger = logging.getLogger("maroczy.datafeed.router")

__all__ = ["DataSourcePolicy", "UnifiedMarketData"]

_INTRADAY_IBKR_BAR_SIZES = frozenset(k for k in IBKR_TO_LSE_TIMEFRAME if k.endswith(("secs", "min", "mins")))

_DURATION_RE = re.compile(r"(\d+)\s*([SDWMY])", re.IGNORECASE)
_DURATION_UNIT_DAYS = {"S": 1 / 86400, "D": 1, "W": 7, "M": 30, "Y": 365}


def _duration_to_days(duration: str) -> float | None:
    """Parse an IBKR duration string (e.g. ``"2 Y"``, ``"30 D"``, ``"3600 S"``) into approximate days."""
    match = _DURATION_RE.search(duration.strip())
    if not match:
        return None
    value, unit = match.groups()
    return float(value) * _DURATION_UNIT_DAYS[unit.upper()]


@dataclass
class DataSourcePolicy:
    """Tunable knobs for :meth:`UnifiedMarketData.resolve_source`.

    Parameters
    ----------
    ibkr_intraday_limit_days: beyond this look-back window, sub-daily bar
        requests prefer LSE over IBKR (IBKR historical pacing gets painful
        for large intraday pulls; see IBKR's historical data limitations).
    prefer_lse_for_history: default to LSE for plain daily-or-coarser
        history when an LSE client is configured (deeper history, no
        pacing limits).
    fallback: if the chosen source raises, try the other configured source
        once before propagating the error.
    """

    ibkr_intraday_limit_days: int = 30
    prefer_lse_for_history: bool = True
    fallback: bool = True


class UnifiedMarketData:
    """Single ``.history()`` entrypoint that routes to IBKR or LSE per :class:`DataSourcePolicy`.

    Parameters
    ----------
    ibkr: a :class:`~maroczy.broker.data.MarketData` instance (optional).
    lse: a :class:`~maroczy.datafeed.lse.LSEData` instance (optional).
    policy: routing policy; defaults to :class:`DataSourcePolicy`'s defaults.

    At least one of ``ibkr``/``lse`` must be provided.
    """

    def __init__(
        self,
        ibkr: MarketData | None = None,
        lse: LSEData | None = None,
        policy: DataSourcePolicy | None = None,
    ):
        if ibkr is None and lse is None:
            raise ValueError("UnifiedMarketData needs at least one of `ibkr` or `lse` configured.")
        self.ibkr = ibkr
        self.lse = lse
        self.policy = policy or DataSourcePolicy()

    def resolve_source(
        self,
        bar_size: str = "1 day",
        duration: str = "2 Y",
        realtime: bool = False,
        source: str | None = None,
    ) -> str:
        """Return ``"ibkr"`` or ``"lse"`` per the module-level protocol, without fetching data."""
        if source is not None:
            if source not in ("ibkr", "lse"):
                raise ValueError(f"source must be 'ibkr', 'lse' or None, got {source!r}")
            if getattr(self, source) is None:
                raise ValueError(f"source={source!r} requested but no {source} client is configured.")
            return source

        if realtime:
            if self.ibkr is None:
                raise ValueError("Real-time data requires a connected IBKR MarketData; none configured.")
            return "ibkr"

        intraday = bar_size in _INTRADAY_IBKR_BAR_SIZES
        lookback_days = _duration_to_days(duration)

        if intraday and self.lse is not None:
            if lookback_days is None or lookback_days > self.policy.ibkr_intraday_limit_days or self.ibkr is None:
                return "lse"
            return "ibkr"

        if self.lse is not None and self.policy.prefer_lse_for_history:
            return "lse"
        if self.ibkr is not None:
            return "ibkr"
        return "lse"

    def history(
        self,
        symbol: str,
        bar_size: str = "1 day",
        duration: str = "2 Y",
        start: str | None = None,
        end: str | None = None,
        realtime: bool = False,
        source: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch OHLCV history, routed to IBKR or LSE per :meth:`resolve_source`.

        Returns the common shape both sources produce: a DataFrame indexed
        by timestamp with columns ``open, high, low, close, volume``.
        """
        chosen = self.resolve_source(bar_size=bar_size, duration=duration, realtime=realtime, source=source)
        try:
            return self._fetch(chosen, symbol, bar_size, duration, start, end, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if self.policy.fallback:
                other = "lse" if chosen == "ibkr" else "ibkr"
                if getattr(self, other, None) is not None:
                    logger.warning("Primary source %r failed (%s); falling back to %r.", chosen, exc, other)
                    return self._fetch(other, symbol, bar_size, duration, start, end, **kwargs)
            raise

    def _fetch(
        self,
        source: str,
        symbol: str,
        bar_size: str,
        duration: str,
        start: str | None,
        end: str | None,
        **kwargs,
    ) -> pd.DataFrame:
        if source == "ibkr":
            return self.ibkr.history(symbol, duration=duration, bar_size=bar_size, end=end or "", **kwargs)
        timeframe = IBKR_TO_LSE_TIMEFRAME.get(bar_size, bar_size)
        return self.lse.candles(symbol, timeframe=timeframe, start=start, end=end, **kwargs)
