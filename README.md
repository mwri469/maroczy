# Maroczy

Quant research toolkit that plugs into a running Interactive Brokers (IBKR) session and provides:

- **Live/paper broker integration** via `ib_insync` with local DuckDB caching of historical data.
- **London Strategic Edge (LSE) integration** via `maroczy.datafeed` — deep-history candles (14 resolutions), macro economics, bond yields, options chains/flow with greeks, and reference data (insider trades, dividends, splits, financial statements), plus a routing protocol that automatically picks IBKR vs. LSE for a given request (see [Data sources: IBKR vs. LSE](#data-sources-ibkr-vs-lse) below).
- **Alpha research features**:
  - `maroczy.features.bubble` — Phillips, Shi & Yu (2015) GSADF/BSADF explosive-bubble detection & date-stamping.
  - `maroczy.features.covariance` — Bongiorno & Challet (2022) style covariance matrix cleaning (eigenvalue clipping, RIE, OAS).
  - `maroczy.features.cpcv` — Combinatorial Purged Cross-Validation (de Prado) with purging/embargo and PBO.
  - `maroczy.features.rough_vol` — Hager, Horst, Wagenhofer & Xu (2026) rough log-normal volatility: Hurst estimation, fBM/rough Bergomi simulation, Poisson microstructure model.
- **Characteristic generation engine** driven by `characteristics mapping.csv`, with a growing library of implemented characteristics under `maroczy.characteristics.functions`.
- **Strategy tooling**: signal construction, vectorized backtesting, and a live signal-to-order loop.

## Install

```powershell
pip install -e ".[notebook]"

# optional: London Strategic Edge integration
pip install -e ".[notebook,lse]"
$env:LSE_API_KEY = "lse_live_..."   # or pass api_key=... explicitly
```

## Quickstart

See `notebooks/quickstart.ipynb`. In short:

```python
from maroczy.broker import Broker
from maroczy.features.bubble import gsadf, bsadf_dating
from maroczy.characteristics import CharacteristicEngine

broker = Broker()          # connects to a running TWS/Gateway session
broker.connect()
bars = broker.history("AAPL", duration="2 Y", bar_size="1 day")

stat, crit = gsadf(bars["close"])
dates = bsadf_dating(bars["close"])

engine = CharacteristicEngine()
chars = engine.compute(bars, names=["ret_12_1", "vol_252d", "ami_126d"])
```

Requires a running TWS or IB Gateway instance (paper or live) with API access enabled.

## Data sources: IBKR vs. LSE

`maroczy.datafeed.UnifiedMarketData` gives you one `.history()` call that
decides, per request, whether to serve it from your live IBKR session or
from the London Strategic Edge (LSE) databank. The full decision procedure
lives in `maroczy.datafeed.router` (read the module docstring for the
exact rules); in short:

| Situation | Source | Why |
|---|---|---|
| Real-time/streaming quotes, or anything you're about to trade against | **IBKR** | Only IBKR is a live, execution-grade connection to your account. |
| A symbol/contract not in the LSE catalog (an option/future you already hold, an odd listing) | **IBKR** | LSE only covers what's in its vault. |
| Intraday bars (sub-daily) over a long look-back (`> ibkr_intraday_limit_days`, default 30 days) | **LSE** | IBKR paces/limits large intraday historical pulls; LSE has no such limit. |
| Plain daily/weekly/monthly history for research, backtesting, or characteristic generation | **LSE** (falls back to IBKR if no LSE key is configured) | Deeper history (US stocks since 2003, FX since 2009, crypto since 2017), no pacing limits, plus fundamentals/economics/options LSE-only. |
| The chosen source errors out | **the other one**, once | `DataSourcePolicy.fallback=True` (default). |

```python
from maroczy.broker import Broker, MarketData
from maroczy.datafeed import LSEData, UnifiedMarketData, DataSourcePolicy

broker = Broker().connect()
router = UnifiedMarketData(
    ibkr=MarketData(broker),
    lse=LSEData(),                          # reads LSE_API_KEY from the environment
    policy=DataSourcePolicy(ibkr_intraday_limit_days=30),
)

daily = router.history("AAPL", bar_size="1 day", duration="5 Y")        # -> LSE
intraday = router.history("AAPL", bar_size="1 min", duration="5 D")     # -> IBKR (short look-back)
deep_intraday = router.history("AAPL", bar_size="1 min", duration="1 Y")  # -> LSE (long look-back)
live = router.history("AAPL", realtime=True)                            # -> IBKR (forced)
forced = router.history("AAPL", source="lse")                           # explicit override
```

LSE also exposes fundamentals/reference data with no IBKR equivalent used
by `maroczy.characteristics.functions.funda`/`fundq`:

```python
lse = LSEData()
income_stmt = lse.financial_reports("AAPL", report_type="income", period="FY")
fundamentals = lse.fundamentals("AAPL")
cpi = lse.economics("cpi_yoy")
us10y = lse.bond_yields("US10Y", start="2000-01-01")
```
