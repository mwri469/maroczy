"""Order execution helpers and a live signal-to-order loop.

Designed so that live trading is opt-in and safe by default: the
:class:`LiveExecutor` requires ``auto=True`` to actually transmit orders,
otherwise it only logs / returns the orders it *would* place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from ib_insync import LimitOrder, MarketOrder, Order, Trade

from maroczy.broker.connection import Broker
from maroczy.broker.data import MarketData

logger = logging.getLogger("maroczy.broker.orders")


class OrderManager:
    """Simple wrapper for placing/canceling orders and inspecting positions."""

    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    def market_order(self, symbol: str, quantity: float, action: str | None = None) -> Order:
        action = action or ("BUY" if quantity > 0 else "SELL")
        return MarketOrder(action, abs(quantity))

    def limit_order(self, symbol: str, quantity: float, limit_price: float, action: str | None = None) -> Order:
        action = action or ("BUY" if quantity > 0 else "SELL")
        return LimitOrder(action, abs(quantity), limit_price)

    def place(self, symbol: str, order: Order) -> Trade:
        contract = MarketData.resolve_contract(symbol)
        self.broker.ib.qualifyContracts(contract)
        trade = self.broker.ib.placeOrder(contract, order)
        logger.info("Placed order: %s %s %s", order.action, order.totalQuantity, symbol)
        return trade

    def cancel_all(self) -> None:
        for trade in self.broker.ib.openTrades():
            self.broker.ib.cancelOrder(trade.order)

    def positions(self) -> pd.DataFrame:
        rows = [
            {
                "symbol": p.contract.symbol,
                "position": p.position,
                "avg_cost": p.avgCost,
                "account": p.account,
            }
            for p in self.broker.ib.positions()
        ]
        return pd.DataFrame(rows)

    def account_summary(self) -> pd.DataFrame:
        rows = [
            {"tag": row.tag, "value": row.value, "currency": row.currency}
            for row in self.broker.ib.accountSummary()
        ]
        return pd.DataFrame(rows)


@dataclass
class LiveExecutor:
    """Turns a signal function into (optionally automatic) live orders.

    Parameters
    ----------
    broker: Connected :class:`Broker`.
    signal_fn: Callable that takes a symbol and returns a target position
        weight in ``[-1, 1]`` (or ``None`` for "no opinion").
    capital: Capital allocated per symbol used to size orders.
    auto: If False (default), orders are computed and logged but never
        transmitted — call :meth:`run` and inspect ``.pending_orders``.
    """

    broker: Broker
    signal_fn: Callable[[str], float | None]
    capital: float = 10_000.0
    auto: bool = False
    order_manager: OrderManager = field(init=False)
    pending_orders: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.order_manager = OrderManager(self.broker)

    def run(self, symbols: list[str]) -> list[dict]:
        self.pending_orders.clear()
        positions = self.order_manager.positions().set_index("symbol")["position"] if True else None
        for symbol in symbols:
            target_weight = self.signal_fn(symbol)
            if target_weight is None:
                continue
            current_qty = float(positions.get(symbol, 0.0)) if positions is not None and symbol in positions.index else 0.0
            # naive sizing: weight * capital / last price, delta vs current
            ticker = self.broker.ib.reqMktData(MarketData.resolve_contract(symbol), "", False, False)
            self.broker.ib.sleep(0.5)
            price = ticker.last or ticker.close or ticker.marketPrice()
            if not price or price != price:  # NaN guard
                logger.warning("No price for %s, skipping", symbol)
                continue
            target_qty = round((target_weight * self.capital) / price)
            delta = target_qty - current_qty
            if abs(delta) < 1:
                continue
            order_record = {"symbol": symbol, "delta": delta, "target_qty": target_qty, "price": price}
            self.pending_orders.append(order_record)
            if self.auto:
                order = self.order_manager.market_order(symbol, delta)
                self.order_manager.place(symbol, order)
            else:
                logger.info("[DRY RUN] would trade %s delta=%s at ~%.2f", symbol, delta, price)
        return self.pending_orders
