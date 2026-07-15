"""Maroczy: quant research toolkit for IBKR-integrated alpha research.

Sub-packages
------------
broker           Live/paper IBKR session management, historical data, orders.
features         Research-paper-derived alpha features (bubble detection,
                 covariance cleaning, CPCV, rough volatility).
characteristics  CSV-driven characteristic (factor) generation engine.
strategy         Signal construction, backtesting, live execution loop.
datafeed         London Strategic Edge integration + IBKR/LSE data routing.

Importing this package loads per-user secrets (e.g. ``LSE_API_KEY``) from a
local ``.env`` file; see :mod:`maroczy.config` and ``.env.example``.
"""

from importlib.metadata import PackageNotFoundError, version

from maroczy import config  # noqa: F401 - imported for its .env-loading side effect

try:
    __version__ = version("maroczy")
except PackageNotFoundError:  # pragma: no cover - editable/local dev
    __version__ = "0.0.0"

__all__ = ["__version__", "config"]
