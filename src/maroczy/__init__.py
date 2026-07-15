"""Maroczy: quant research toolkit for IBKR-integrated alpha research.

Sub-packages
------------
broker           Live/paper IBKR session management, historical data, orders.
features         Research-paper-derived alpha features (bubble detection,
                 covariance cleaning, CPCV, rough volatility).
characteristics  CSV-driven characteristic (factor) generation engine.
strategy         Signal construction, backtesting, live execution loop.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maroczy")
except PackageNotFoundError:  # pragma: no cover - editable/local dev
    __version__ = "0.0.0"

__all__ = ["__version__"]
