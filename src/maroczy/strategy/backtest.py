"""Vectorized backtesting of a weights/returns panel.

Given a time series of target portfolio weights (``date x symbol``) and a
matching returns panel, computes realized PnL, turnover-adjusted returns
(with a simple linear transaction-cost model), and standard performance
statistics (Sharpe, max drawdown, hit rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["BacktestResult", "run_backtest"]


@dataclass
class BacktestResult:
    returns: pd.Series
    equity_curve: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    stats: dict

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"BacktestResult(sharpe={s['sharpe']:.2f}, ann_return={s['ann_return']:.2%}, "
            f"ann_vol={s['ann_vol']:.2%}, max_drawdown={s['max_drawdown']:.2%}, "
            f"avg_turnover={s['avg_turnover']:.2%})"
        )


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    cost_bps: float = 0.0,
    lag: int = 1,
) -> BacktestResult:
    """Backtest target weights against realized returns.

    Parameters
    ----------
    weights: ``date x symbol`` target portfolio weights (e.g. from
        :func:`maroczy.strategy.signals.long_short_weights` applied to each
        date's cross-section).
    returns: ``date x symbol`` realized simple returns, same columns as ``weights``.
    periods_per_year: for annualization (252 for daily bars, 12 for monthly).
    cost_bps: linear transaction cost in basis points, applied to turnover.
    lag: number of periods to lag the weights before applying them to returns
        (avoids look-ahead bias; default 1 = trade on the close, realize next
        period's return).
    """
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    lagged_weights = weights.shift(lag).fillna(0.0)

    gross_returns = (lagged_weights * returns).sum(axis=1)
    turnover = lagged_weights.diff().abs().sum(axis=1).fillna(lagged_weights.abs().sum(axis=1))
    costs = turnover * (cost_bps / 1e4)
    net_returns = gross_returns - costs

    equity_curve = (1 + net_returns).cumprod()
    ann_return = float(net_returns.mean() * periods_per_year)
    ann_vol = float(net_returns.std() * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = _max_drawdown(equity_curve)
    hit_rate = float((net_returns > 0).mean())

    stats = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": float(turnover.mean()),
        "hit_rate": hit_rate,
    }
    return BacktestResult(
        returns=net_returns,
        equity_curve=equity_curve,
        turnover=turnover,
        weights=lagged_weights,
        stats=stats,
    )
