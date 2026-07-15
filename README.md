# Maroczy

Quant research toolkit that plugs into a running Interactive Brokers (IBKR) session and provides:

- **Live/paper broker integration** via `ib_insync` with local DuckDB caching of historical data.
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
