"""London Strategic Edge (LSE) market data integration.

Thin, notebook-friendly wrapper around the official ``lse-data`` client
(``pip install "maroczy[lse]"`` or ``pip install lse-data``), mirroring the
shape of :class:`maroczy.broker.data.MarketData` (``candles`` returns the
same ``open, high, low, close, volume`` OHLCV frame indexed by timestamp)
so the two sources are interchangeable in :mod:`maroczy.characteristics`
and :mod:`maroczy.strategy`.

Deep vault coverage this wrapper exposes beyond plain candles: macro
economics series (14,000+), government bond yields, the economic calendar,
insider trades, dividends/splits, COT positioning, financial statements,
company profiles/fundamentals, and options chains/flow/candles with
implied vol & greeks. See
https://www.londonstrategicedge.com/api-documentation for the full schema.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger("maroczy.datafeed.lse")

DEFAULT_CACHE_PATH = Path("data/cache/lse_data.duckdb")

_CANDLES_TABLE = "lse_candles"
_CREATE_CANDLES_SQL = f"""
CREATE TABLE IF NOT EXISTS {_CANDLES_TABLE} (
    symbol    VARCHAR,
    timeframe VARCHAR,
    ts        TIMESTAMP,
    open      DOUBLE,
    high      DOUBLE,
    low       DOUBLE,
    close     DOUBLE,
    volume    DOUBLE,
    PRIMARY KEY (symbol, timeframe, ts)
);
"""

# IBKR-style bar sizes -> LSE timeframe codes (the 14 resolutions the vault stores).
IBKR_TO_LSE_TIMEFRAME = {
    "1 secs": "1s",
    "5 secs": "5s",
    "15 secs": "15s",
    "30 secs": "30s",
    "1 min": "1m",
    "3 mins": "3m",
    "5 mins": "5m",
    "15 mins": "15m",
    "30 mins": "30m",
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d",
    "1 week": "1w",
    "1 month": "1mo",
}
LSE_TIMEFRAMES = tuple(IBKR_TO_LSE_TIMEFRAME.values())
INTRADAY_LSE_TIMEFRAMES = frozenset(IBKR_TO_LSE_TIMEFRAME[k] for k in IBKR_TO_LSE_TIMEFRAME if k.endswith(("secs", "min", "mins")))

__all__ = ["LSEData", "IBKR_TO_LSE_TIMEFRAME", "LSE_TIMEFRAMES", "INTRADAY_LSE_TIMEFRAMES"]


def _import_client():
    try:
        from lse import LSE  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised when extra isn't installed
        raise ImportError(
            "The 'lse-data' package is required for London Strategic Edge integration. "
            'Install it with: pip install "maroczy[lse]"  (or: pip install lse-data)'
        ) from exc
    return LSE


class LSEData:
    """Wraps the ``lse-data`` client with local DuckDB caching of candles.

    Parameters
    ----------
    api_key: your ``lse_live_...`` key (see
        https://www.londonstrategicedge.com/data#api). Defaults to the
        ``LSE_API_KEY`` environment variable, same as the underlying client.
    cache_path: local DuckDB cache for candle pulls, or ``None`` to disable.
    """

    def __init__(self, api_key: str | None = None, cache_path: Path | str | None = DEFAULT_CACHE_PATH):
        LSE = _import_client()
        api_key = api_key or os.environ.get("LSE_API_KEY")
        if not api_key:
            raise ValueError(
                "No LSE API key found. Pass api_key=..., or set the LSE_API_KEY environment "
                "variable (get a key at https://www.londonstrategicedge.com/data)."
            )
        self._client = LSE(api_key=api_key)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.cache_path))
            self._con.execute(_CREATE_CANDLES_SQL)
        else:
            self._con = None

    def __repr__(self) -> str:
        return f"LSEData(cache={self.cache_path})"

    # -- candles -----------------------------------------------------------
    def candles(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        order: str = "asc",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """OHLCV candles for ``symbol`` at one of the 14 stored resolutions.

        Returns a DataFrame indexed by UTC timestamp with columns
        ``open, high, low, close, volume`` -- the same shape as
        :meth:`maroczy.broker.data.MarketData.history`. FX symbols carry no
        volume (no consolidated tape); the column is filled with ``NaN``.
        """
        if not force_refresh and self._con is not None:
            cached = self._read_cache(symbol, timeframe, start, end)
            if cached is not None and not cached.empty:
                return cached

        kwargs = {"start": start, "end": end, "limit": limit, "order": order}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        rows = self._client.candles(symbol, timeframe, **kwargs)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        ts_col = "ts" if "ts" in df.columns else "timestamp"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.rename(columns={ts_col: "ts"}).set_index("ts").sort_index()
        if "volume" not in df.columns:
            df["volume"] = float("nan")

        if self._con is not None:
            self._write_cache(df, symbol, timeframe)
        return df

    def history(self, symbol: str, timeframe: str = "1m", start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Bulk deep-history pull (runs as a vault export job; requires ``lse-data[frames]``)."""
        return self._client.history(symbol, timeframe=timeframe, start=start, end=end)

    def _read_cache(self, symbol: str, timeframe: str, start, end) -> pd.DataFrame | None:
        query = f"SELECT ts, open, high, low, close, volume FROM {_CANDLES_TABLE} WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]
        if start is not None:
            query += " AND ts >= ?"
            params.append(start)
        if end is not None:
            query += " AND ts <= ?"
            params.append(end)
        query += " ORDER BY ts"
        result = self._con.execute(query, params).fetchdf()
        if result.empty:
            return None
        result["ts"] = pd.to_datetime(result["ts"], utc=True)
        return result.set_index("ts")

    def _write_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        to_insert = df.reset_index()[["ts", "open", "high", "low", "close", "volume"]].copy()
        to_insert.insert(0, "timeframe", timeframe)
        to_insert.insert(0, "symbol", symbol)
        self._con.execute(f"DELETE FROM {_CANDLES_TABLE} WHERE symbol = ? AND timeframe = ?", [symbol, timeframe])
        self._con.execute(f"INSERT INTO {_CANDLES_TABLE} SELECT * FROM to_insert")

    # -- macro / reference data --------------------------------------------
    def economics(self, series: str) -> pd.DataFrame:
        """One macro economic series (date, value rows), full depth in one call."""
        return pd.DataFrame(self._client.economics(series))

    def bond_yields(self, symbol: str, start: str | None = None) -> pd.DataFrame:
        """Government bond yield tenor (e.g. ``US10Y``, ``DE10Y``)."""
        return pd.DataFrame(self._client.bond_yields(symbol, start=start) if start else self._client.bond_yields(symbol))

    def economic_calendar(self, region: str | None = None, event: str | None = None, **kwargs) -> pd.DataFrame:
        params = {"region": region, "event": event, **kwargs}
        return pd.DataFrame(self._client.economic_calendar(**{k: v for k, v in params.items() if v is not None}))

    def insider_trades(self, symbol: str, type: str | None = None) -> pd.DataFrame:  # noqa: A002
        kwargs = {"type": type} if type else {}
        return pd.DataFrame(self._client.insider_trades(symbol, **kwargs))

    def dividends(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(self._client.dividends(symbol))

    def splits(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(self._client.splits(symbol))

    def cot(self, code: str) -> pd.DataFrame:
        """CFTC Commitment of Traders positioning (futures market code, e.g. ``ZW``, ``GC``)."""
        return pd.DataFrame(self._client.cot(code))

    def financial_reports(self, symbol: str, report_type: str = "income", period: str = "FY") -> pd.DataFrame:
        """Income/balance/cashflow statements -- feed straight into `maroczy.characteristics.functions.funda`."""
        return pd.DataFrame(self._client.financial_reports(symbol, report_type=report_type, period=period))

    def company_profiles(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(self._client.company_profiles(symbol))

    def fundamentals(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(self._client.fundamentals(symbol))

    # -- options -------------------------------------------------------------
    def options_chain(self, underlying: str, type: str | None = None, **kwargs) -> pd.DataFrame:  # noqa: A002
        params = {"type": type, **kwargs}
        return pd.DataFrame(self._client.options(underlying, **{k: v for k, v in params.items() if v is not None}))

    def options_flow(self, underlying: str | None = None, min_premium: float | None = None, **kwargs) -> pd.DataFrame:
        params = {"underlying": underlying, "min_premium": min_premium, **kwargs}
        params = {k: v for k, v in params.items() if v is not None}
        return pd.DataFrame(self._client.options_flow(**params))

    def option_candles(self, contract: str, strike: float | None = None, expiry: str | None = None, type: str | None = None) -> pd.DataFrame:  # noqa: A002
        """1-minute premium OHLC bars for one contract (OSI ticker, or ``strike``/``expiry``/``type`` parts)."""
        kwargs = {"strike": strike, "expiry": expiry, "type": type}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return pd.DataFrame(self._client.option_candles(contract, **kwargs))

    # -- discovery / streaming ------------------------------------------------
    def catalog(self, category: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(self._client.catalog(category) if category else self._client.catalog())

    def vault_meta(self) -> dict:
        """The vault's shape: datasets, candle classes, the 14 timeframes, reference list."""
        return self._client.vault_meta()

    def stream(self, symbols: list[str], start: str | None = None):
        """Live (optionally replay-then-live) tick generator over the LSE WebSocket."""
        return self._client.stream(symbols, start=start) if start else self._client.stream(symbols)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
