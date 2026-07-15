"""Multi-source market data: London Strategic Edge (LSE) client and the
IBKR-vs-LSE routing protocol.

- :class:`~maroczy.datafeed.lse.LSEData` -- wraps the ``lse-data`` client
  (candles, fundamentals, macro/bond series, options, reference data).
- :class:`~maroczy.datafeed.router.UnifiedMarketData` /
  :class:`~maroczy.datafeed.router.DataSourcePolicy` -- a single
  ``.history()`` entrypoint that decides whether to serve a request from
  the live IBKR session or the LSE vault. See
  :mod:`maroczy.datafeed.router` for the full protocol.
"""

from maroczy.datafeed.lse import LSEData
from maroczy.datafeed.router import DataSourcePolicy, UnifiedMarketData

__all__ = ["LSEData", "UnifiedMarketData", "DataSourcePolicy"]
