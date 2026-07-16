"""Daily price/volume characteristics.

All functions take an OHLCV DataFrame (columns ``open, high, low, close,
volume``, indexed by date). Functions requiring the market portfolio's
returns accept an aligned ``mkt_ret: pd.Series`` keyword; functions
requiring shares outstanding accept ``shares_out``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


def _ret(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()


def _rolling_multi(y: pd.Series, x: pd.Series, window: int, func) -> pd.Series:
    """Apply a two-series rolling function; used for beta/coskew/idiovol style metrics."""
    aligned = pd.concat([y, x], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    yv, xv = aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy()
    out = np.full(len(aligned), np.nan)
    for i in range(window - 1, len(aligned)):
        out[i] = func(yv[i - window + 1 : i + 1], xv[i - window + 1 : i + 1])
    return pd.Series(out, index=aligned.index)


# ---------------------------------------------------------------------------
# Volatility / tail-risk
# ---------------------------------------------------------------------------

@characteristic("retvol")
def retvol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Ang et al. (2006): rolling return volatility."""
    return _ret(df).rolling(window).std()


@characteristic("rmax1_21d")
def rmax1_21d(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Bali, Cakici & Whitelaw (2011): maximum daily return over the window."""
    return _ret(df).rolling(window).max()


@characteristic("rmax5_21d")
def rmax5_21d(df: pd.DataFrame, window: int = 21, top: int = 5) -> pd.Series:
    """Bali, Brown & Tang (2017): average of the top-``top`` daily returns."""
    r = _ret(df)
    return r.rolling(window).apply(lambda x: np.sort(x)[-top:].mean(), raw=True)


@characteristic("rmax5_rvol_21d")
def rmax5_rvol_21d(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Assness et al. (2020): highest-5-days return scaled by return volatility."""
    return rmax5_21d(df, window) / retvol(df, window)


@characteristic("rskew_21d")
def rskew_21d(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Bali, Engle & Murray (2016): rolling return skewness."""
    return _ret(df).rolling(window).skew()


@characteristic("prc_highprc_252d")
def prc_highprc_252d(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """George & Hwang (2004): price relative to its 52-week high."""
    return df["close"] / df["high"].rolling(window).max()


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------

@characteristic("ami_126d")
def ami_126d(df: pd.DataFrame, window: int = 126) -> pd.Series:
    """Amihud (2002) illiquidity: mean(|ret|/dollar_volume), scaled by 1e6."""
    ret = _ret(df).abs()
    dollar_vol = (df["close"] * df["volume"]).replace(0, np.nan)
    illiq = ret / dollar_vol
    return illiq.rolling(window).mean() * 1e6


@characteristic("dolvol_126d")
def dolvol_126d(df: pd.DataFrame, window: int = 126) -> pd.Series:
    """Brennan, Chordia & Subrahmanyam (1998): log average dollar trading volume."""
    dollar_vol = df["close"] * df["volume"]
    return np.log(dollar_vol.rolling(window).mean().replace(0, np.nan))


@characteristic("dolvol_var_126d")
def dolvol_var_126d(df: pd.DataFrame, window: int = 126) -> pd.Series:
    """Chordia, Subrahmanyam & Anshuman (2001): volatility of dollar trading volume."""
    dollar_vol = df["close"] * df["volume"]
    mean = dollar_vol.rolling(window).mean()
    return dollar_vol.rolling(window).std() / mean.replace(0, np.nan)


@characteristic("turnover_126d")
def turnover_126d(df: pd.DataFrame, shares_out: pd.Series | float, window: int = 126) -> pd.Series:
    """Datar, Naik & Radcliffe (1998): average share turnover (volume / shares outstanding)."""
    turn = df["volume"] / shares_out
    return turn.rolling(window).mean()


@characteristic("turnover_var_126d")
def turnover_var_126d(df: pd.DataFrame, shares_out: pd.Series | float, window: int = 126) -> pd.Series:
    """Chordia, Subrahmanyam & Anshuman (2001): volatility of share turnover."""
    turn = df["volume"] / shares_out
    return turn.rolling(window).std()


@characteristic("bidaskhl_21d")
def bidaskhl_21d(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Corwin & Schultz (2012) high-low bid-ask spread estimator."""
    high, low = df["high"], df["low"]
    beta = (np.log(high / low)) ** 2
    beta = beta + beta.shift(1)
    hl2_high = high.rolling(2).max()
    hl2_low = low.rolling(2).min()
    gamma = (np.log(hl2_high / hl2_low)) ** 2
    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)
    return spread.rolling(window).mean()


# ---------------------------------------------------------------------------
# Market-model betas / co-moments (need the market portfolio's return series)
# ---------------------------------------------------------------------------

@characteristic("beta_60m")
def beta_60m(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    """Fama & MacBeth (1973)-style market beta (CAPM slope)."""
    ret = _ret(df)

    def f(y, x):
        xd = x - x.mean()
        yd = y - y.mean()
        return np.sum(xd * yd) / np.sum(xd * xd)

    return _rolling_multi(ret, mkt_ret, window, f)


@characteristic("betadown_252d")
def betadown_252d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 252) -> pd.Series:
    """Ang, Chen & Xing (2006): downside beta (conditional on negative market returns)."""
    ret = _ret(df)

    def f(y, x):
        mask = x < 0
        if mask.sum() < 5:
            return np.nan
        xm, ym = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
        denom = np.sum(xm * xm)
        return np.sum(xm * ym) / denom if denom else np.nan

    return _rolling_multi(ret, mkt_ret, window, f)


@characteristic("corr_1260d")
def corr_1260d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 1260) -> pd.Series:
    """Assness et al. (2020): rolling correlation with the market."""
    ret = _ret(df)

    def f(y, x):
        return np.corrcoef(y, x)[0, 1]

    return _rolling_multi(ret, mkt_ret, window, f)


@characteristic("coskew_21d")
def coskew_21d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 21) -> pd.Series:
    """Harvey & Siddique (2000): coskewness with the market."""
    ret = _ret(df)

    def f(y, x):
        yd, xd = y - y.mean(), x - x.mean()
        denom = np.sqrt(np.mean(yd**2)) * np.mean(xd**2)
        return np.mean(yd * xd**2) / denom if denom else np.nan

    return _rolling_multi(ret, mkt_ret, window, f)


def _capm_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    xd, yd = x - x.mean(), y - y.mean()
    denom = np.sum(xd * xd)
    beta = np.sum(xd * yd) / denom if denom else 0.0
    alpha = y.mean() - beta * x.mean()
    return y - (alpha + beta * x)


@characteristic("ivol_capm_21d")
def ivol_capm_21d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 21) -> pd.Series:
    """Ang et al. (2006): idiosyncratic volatility relative to the CAPM."""
    ret = _ret(df)

    def f(y, x):
        resid = _capm_residuals(y, x)
        return np.std(resid, ddof=1)

    return _rolling_multi(ret, mkt_ret, window, f)


@characteristic("idiovol")
def idiovol(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 21) -> pd.Series:
    """Ali, Hwang & Trombley (2003): idiosyncratic volatility (CAPM residual std dev)."""
    return ivol_capm_21d(df, mkt_ret, window)


@characteristic("iskew_capm_21d")
def iskew_capm_21d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 21) -> pd.Series:
    """Bali, Engle & Murray (2016): idiosyncratic skewness relative to the CAPM."""
    ret = _ret(df)

    def f(y, x):
        resid = _capm_residuals(y, x)
        s = resid.std()
        if s == 0:
            return np.nan
        m3 = np.mean((resid - resid.mean()) ** 3)
        return m3 / s**3

    return _rolling_multi(ret, mkt_ret, window, f)


# ---------------------------------------------------------------------------
# Extended idiosyncratic vol/skewness (multi-factor models)
# ---------------------------------------------------------------------------

def _multi_factor_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of y on X (X should include intercept)."""
    try:
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return y - y.mean()
    return y - X @ theta


def _rolling_multi_factor(ret: pd.Series, factors: list[pd.Series], window: int, stat_fn) -> pd.Series:
    """Rolling window multi-factor regression, apply stat_fn to residuals."""
    frame = pd.concat([ret] + factors, axis=1).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    y = frame.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame.iloc[:, j].to_numpy() for j in range(1, frame.shape[1])])
    out = np.full(len(frame), np.nan)
    for i in range(window - 1, len(frame)):
        sl = slice(i - window + 1, i + 1)
        resid = _multi_factor_residuals(y[sl], X[sl])
        out[i] = stat_fn(resid)
    return pd.Series(out, index=frame.index)


@characteristic("ivol_capm_252d")
def ivol_capm_252d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 252) -> pd.Series:
    """Ali, Hwang & Trombley (2003): idiosyncratic volatility (CAPM, 252d)."""
    ret = _ret(df)

    def f(y, x):
        resid = _capm_residuals(y, x)
        return np.std(resid, ddof=1)

    return _rolling_multi(ret, mkt_ret, window, f)


@characteristic("ivol_ff3_21d")
def ivol_ff3_21d(df: pd.DataFrame, mkt_ret: pd.Series, smb: pd.Series = None, hml: pd.Series = None, window: int = 21) -> pd.Series:
    """Ang et al. (2006): idiosyncratic volatility relative to the FF3 model."""
    if smb is None or hml is None:
        return ivol_capm_21d(df, mkt_ret, window)
    ret = _ret(df)
    return _rolling_multi_factor(ret, [mkt_ret, smb, hml], window, lambda r: np.std(r, ddof=1))


@characteristic("ivol_hxz4_21d")
def ivol_hxz4_21d(df: pd.DataFrame, mkt_ret: pd.Series, me_factor: pd.Series = None, ia_factor: pd.Series = None, roe_factor: pd.Series = None, window: int = 21) -> pd.Series:
    """Ang et al. (2006): idiosyncratic volatility relative to the q-factor model."""
    if me_factor is None or ia_factor is None or roe_factor is None:
        return ivol_capm_21d(df, mkt_ret, window)
    ret = _ret(df)
    return _rolling_multi_factor(ret, [mkt_ret, me_factor, ia_factor, roe_factor], window, lambda r: np.std(r, ddof=1))


@characteristic("iskew_ff3_21d")
def iskew_ff3_21d(df: pd.DataFrame, mkt_ret: pd.Series, smb: pd.Series = None, hml: pd.Series = None, window: int = 21) -> pd.Series:
    """Bali, Engle & Murray (2016): idiosyncratic skewness (FF3)."""
    if smb is None or hml is None:
        return iskew_capm_21d(df, mkt_ret, window)
    ret = _ret(df)

    def _skew(resid):
        s = resid.std()
        if s == 0:
            return np.nan
        return np.mean((resid - resid.mean()) ** 3) / s**3

    return _rolling_multi_factor(ret, [mkt_ret, smb, hml], window, _skew)


@characteristic("iskew_hxz4_21d")
def iskew_hxz4_21d(df: pd.DataFrame, mkt_ret: pd.Series, me_factor: pd.Series = None, ia_factor: pd.Series = None, roe_factor: pd.Series = None, window: int = 21) -> pd.Series:
    """Bali, Engle & Murray (2016): idiosyncratic skewness (q-factor)."""
    if me_factor is None or ia_factor is None or roe_factor is None:
        return iskew_capm_21d(df, mkt_ret, window)
    ret = _ret(df)

    def _skew(resid):
        s = resid.std()
        if s == 0:
            return np.nan
        return np.mean((resid - resid.mean()) ** 3) / s**3

    return _rolling_multi_factor(ret, [mkt_ret, me_factor, ia_factor, roe_factor], window, _skew)


# ---------------------------------------------------------------------------
# Additional betas
# ---------------------------------------------------------------------------

@characteristic("beta")
def beta_ghz(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 156) -> pd.Series:
    """Fama & MacBeth (1973) GHZ: market beta using weekly returns (~3yr)."""
    ret = _ret(df)
    ret_w = ret.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)
    mkt_w = mkt_ret.reindex(ret.index).resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)

    def f(y, x):
        xd = x - x.mean()
        denom = np.sum(xd * xd)
        return np.sum(xd * (y - y.mean())) / denom if denom else np.nan

    result = _rolling_multi(ret_w, mkt_w, window, f)
    return result.reindex(df.index, method="ffill")


@characteristic("betasq")
def betasq(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 156) -> pd.Series:
    """Fama & MacBeth (1973) GHZ: beta squared."""
    return beta_ghz(df, mkt_ret, window) ** 2


@characteristic("betabab_1260d")
def betabab_1260d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 1260) -> pd.Series:
    """Frazzini & Pedersen (2014): betting-against-beta (shrunk beta)."""
    ret = _ret(df)
    vol_s = ret.rolling(window, min_periods=window // 2).std()
    vol_m = mkt_ret.rolling(window, min_periods=window // 2).std()
    rho = ret.rolling(window, min_periods=window // 2).corr(mkt_ret)
    beta_ts = rho * vol_s / vol_m.replace(0, np.nan)
    shrink = 0.6
    return shrink * beta_ts + (1 - shrink) * 1.0


# ---------------------------------------------------------------------------
# Additional liquidity / volume measures
# ---------------------------------------------------------------------------

@characteristic("baspread")
def baspread(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Amihud & Mendelson (1986): bid-ask spread (high/low proxy if no bid/ask)."""
    bid = df["bid"] if "bid" in df.columns else df["low"]
    ask = df["ask"] if "ask" in df.columns else df["high"]
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid.replace(0, np.nan)
    return spread.rolling(window).mean()


@characteristic("std_dolvol")
def std_dolvol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Chordia, Subrahmanyam & Anshuman (2001) GHZ: 1-month vol of log dollar volume."""
    dollar_vol = np.log((df["close"] * df["volume"]).replace(0, np.nan))
    return dollar_vol.rolling(window).std()


@characteristic("std_turn")
def std_turn(df: pd.DataFrame, shares_out: pd.Series | float, window: int = 21) -> pd.Series:
    """Chordia, Subrahmanyam & Anshuman (2001) GHZ: 1-month vol of turnover."""
    turn = df["volume"] / shares_out
    return turn.rolling(window).std()


@characteristic("zero_trades_21d")
def zero_trades_21d(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Liu (2006): zero-trading-days measure (1 month)."""
    zero = (df["volume"] == 0).astype(float)
    turnover = df["volume"].rolling(window).mean().replace(0, np.nan)
    return zero.rolling(window).sum() + 1.0 / (window * turnover)


@characteristic("zero_trades_126d")
def zero_trades_126d(df: pd.DataFrame, window: int = 126) -> pd.Series:
    """Liu (2006): zero-trading-days measure (6 months)."""
    zero = (df["volume"] == 0).astype(float)
    turnover = df["volume"].rolling(window).mean().replace(0, np.nan)
    return zero.rolling(window).sum() + 1.0 / (window * turnover)


@characteristic("zero_trades_252d")
def zero_trades_252d(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Liu (2006): zero-trading-days measure (12 months)."""
    zero = (df["volume"] == 0).astype(float)
    turnover = df["volume"].rolling(window).mean().replace(0, np.nan)
    return zero.rolling(window).sum() + 1.0 / (window * turnover)


# ---------------------------------------------------------------------------
# Price delay
# ---------------------------------------------------------------------------

@characteristic("pricedelay")
def pricedelay(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 252) -> pd.Series:
    """Hou & Moskowitz (2005): price delay (R-squared based)."""
    ret = _ret(df)
    n_lags = 5
    cols = [ret, mkt_ret] + [mkt_ret.shift(k) for k in range(1, n_lags + 1)]
    frame = pd.concat(cols, axis=1).dropna()
    y = frame.iloc[:, 0].to_numpy()
    x_r = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1].to_numpy()])
    x_u = np.column_stack([np.ones(len(frame))] + [frame.iloc[:, j].to_numpy() for j in range(1, n_lags + 2)])
    out = np.full(len(frame), np.nan)
    for i in range(window - 1, len(frame)):
        sl = slice(i - window + 1, i + 1)
        yi = y[sl]
        ss_tot = np.sum((yi - yi.mean()) ** 2)
        if ss_tot == 0:
            continue
        theta_r, *_ = np.linalg.lstsq(x_r[sl], yi, rcond=None)
        r2_r = 1 - np.sum((yi - x_r[sl] @ theta_r) ** 2) / ss_tot
        theta_u, *_ = np.linalg.lstsq(x_u[sl], yi, rcond=None)
        r2_u = 1 - np.sum((yi - x_u[sl] @ theta_u) ** 2) / ss_tot
        out[i] = 1 - r2_r / r2_u if r2_u > 0 else 0.0
    return pd.Series(out, index=frame.index)


@characteristic("pricedelay_slope")
def pricedelay_slope(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 252) -> pd.Series:
    """Hou & Moskowitz (2005): price delay (slope-based)."""
    ret = _ret(df)
    n_lags = 5
    cols = [ret, mkt_ret] + [mkt_ret.shift(k) for k in range(1, n_lags + 1)]
    frame = pd.concat(cols, axis=1).dropna()
    y = frame.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame.iloc[:, j].to_numpy() for j in range(1, n_lags + 2)])
    out = np.full(len(frame), np.nan)
    for i in range(window - 1, len(frame)):
        sl = slice(i - window + 1, i + 1)
        try:
            theta, *_ = np.linalg.lstsq(X[sl], y[sl], rcond=None)
        except np.linalg.LinAlgError:
            continue
        betas = theta[1:]
        denom = np.sum(np.abs(betas))
        out[i] = np.sum(np.abs(betas[1:])) / denom if denom > 0 else 0.0
    return pd.Series(out, index=frame.index)


# ---------------------------------------------------------------------------
# Trend factor
# ---------------------------------------------------------------------------

@characteristic("trend_factor")
def trend_factor(df: pd.DataFrame) -> pd.Series:
    """Han, Zhou & Zhu (2016): trend factor from multiple MA signals."""
    px = df["close"]
    windows = [3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
    signals = pd.DataFrame(index=df.index)
    for w in windows:
        ma = px.rolling(w, min_periods=w).mean()
        signals[f"ma{w}"] = np.sign(px - ma)
    return signals.mean(axis=1)


@characteristic("pricedelay_tstat")
def pricedelay_tstat(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 252) -> pd.Series:
    """Hou & Moskowitz (2005): price delay D3 (SE-adjusted lagged slopes)."""
    ret = _ret(df)
    n_lags = 5
    cols = [ret, mkt_ret] + [mkt_ret.shift(k) for k in range(1, n_lags + 1)]
    frame = pd.concat(cols, axis=1).dropna()
    y = frame.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame.iloc[:, j].to_numpy() for j in range(1, n_lags + 2)])
    out = np.full(len(frame), np.nan)
    for i in range(window - 1, len(frame)):
        sl = slice(i - window + 1, i + 1)
        yi, Xi = y[sl], X[sl]
        try:
            theta, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yi - Xi @ theta
        s2 = np.sum(resid ** 2) / max(len(yi) - Xi.shape[1], 1)
        xtx_inv = np.linalg.pinv(Xi.T @ Xi)
        se = np.sqrt(np.diag(s2 * xtx_inv))
        betas = theta[1:]
        se_betas = se[1:]
        se_betas = np.where(se_betas == 0, np.nan, se_betas)
        tstats = np.abs(betas / se_betas)
        denom = np.nansum(tstats)
        out[i] = np.nansum(tstats[1:]) / denom if denom > 0 else 0.0
    return pd.Series(out, index=frame.index)
