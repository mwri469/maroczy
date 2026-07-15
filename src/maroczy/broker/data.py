"""Historical & streaming market data access with local DuckDB caching.

Every historical bar request is cached in a local DuckDB file
(``data/cache/market_data.duckdb`` by default) keyed by symbol, bar size and
"what to show", so repeated notebook runs don't re-hit the IBKR pacing
limits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
from ib_insync import Contract, Stock, util

from maroczy.broker.connection import Broker

logger = logging.getLogger("maroczy.broker.data")

DEFAULT_CACHE_PATH = Path("data/cache/market_data.duckdb")

_BARS_TABLE = "bars"

_CREATE_BARS_SQL = f"""
CREATE TABLE IF NOT EXISTS {_BARS_TABLE} (
    symbol      VARCHAR,
    bar_size    VARCHAR,
    what_to_show VARCHAR,
    date        TIMESTAMP,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    average     DOUBLE,
    bar_count   BIGINT,
    PRIMARY KEY (symbol, bar_size, what_to_show, date)
);
"""


class MarketData:
    """Historical/streaming data access backed by an IBKR :class:`Broker`.

    Parameters
    ----------
    broker:
        A connected :class:`~maroczy.broker.connection.Broker` instance.
    cache_path:
        Path to a local DuckDB file used to cache historical bars. Pass
        ``None`` to disable caching entirely.
    """

    def __init__(self, broker: Broker, cache_path: Path | str | None = DEFAULT_CACHE_PATH) -> None:
        self.broker = broker
        self.cache_path = Path(cache_path) if cache_path is not None else None
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.cache_path))
            self._con.execute(_CREATE_BARS_SQL)
        else:
            self._con = None

    # -- contract resolution ------------------------------------------
    @staticmethod
    def resolve_contract(symbol: str | Contract, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> Contract:
        if isinstance(symbol, Contract):
            return symbol
        if sec_type == "STK":
            return Stock(symbol, exchange, currency)
        raise ValueError(f"Unsupported sec_type {sec_type!r}; pass an explicit Contract instead.")

    # -- historical bars -----------------------------------------------
    def history(
        self,
        symbol: str | Contract,
        duration: str = "2 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        end: str = "",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars, using and populating the local cache.

        Returns a DataFrame indexed by ``date`` with columns
        ``open, high, low, close, volume, average, bar_count``.
        """
        sym_key = symbol if isinstance(symbol, str) else symbol.symbol

        if not force_refresh and self._con is not None:
            cached = self._read_cache(sym_key, bar_size, what_to_show)
            if cached is not None and not cached.empty:
                logger.info("Loaded %d cached bars for %s (%s)", len(cached), sym_key, bar_size)
                return cached

        contract = self.resolve_contract(symbol)
        self.broker.ib.qualifyContracts(contract)
        bars = self.broker.ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "average", "barCount"])
        df = df.rename(columns={"barCount": "bar_count"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if self._con is not None:
            self._write_cache(df, sym_key, bar_size, what_to_show)
        return df

    def _read_cache(self, symbol: str, bar_size: str, what_to_show: str) -> pd.DataFrame | None:
        query = f"""
            SELECT date, open, high, low, close, volume, average, bar_count
            FROM {_BARS_TABLE}
            WHERE symbol = ? AND bar_size = ? AND what_to_show = ?
            ORDER BY date
        """
        result = self._con.execute(query, [symbol, bar_size, what_to_show]).fetchdf()
        if result.empty:
            return None
        return result.set_index("date")

    def _write_cache(self, df: pd.DataFrame, symbol: str, bar_size: str, what_to_show: str) -> None:
        to_insert = df.reset_index()[["date", "open", "high", "low", "close", "volume", "average", "bar_count"]].copy()
        to_insert.insert(0, "what_to_show", what_to_show)
        to_insert.insert(0, "bar_size", bar_size)
        to_insert.insert(0, "symbol", symbol)
        self._con.execute(f"DELETE FROM {_BARS_TABLE} WHERE symbol = ? AND bar_size = ? AND what_to_show = ?", [symbol, bar_size, what_to_show])
        self._con.execute(f"INSERT INTO {_BARS_TABLE} SELECT * FROM to_insert")

    # -- streaming -------------------------------------------------------
    def stream_quotes(self, symbol: str | Contract):
        """Subscribe to streaming top-of-book quotes; returns an ib_insync Ticker."""
        contract = self.resolve_contract(symbol)
        self.broker.ib.qualifyContracts(contract)
        return self.broker.ib.reqMktData(contract, "", False, False)

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
