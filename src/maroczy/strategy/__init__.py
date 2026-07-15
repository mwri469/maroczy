"""Strategy tooling: signal construction, backtesting, and live execution.

Live order execution (:class:`~maroczy.broker.orders.LiveExecutor`) lives in
:mod:`maroczy.broker` since it needs a connected broker session; import it
from there.
"""

from maroczy.strategy.backtest import BacktestResult, run_backtest
from maroczy.strategy.signals import combine_signals, long_short_weights, rank_signal, zscore

__all__ = [
    "zscore",
    "rank_signal",
    "combine_signals",
    "long_short_weights",
    "run_backtest",
    "BacktestResult",
]
