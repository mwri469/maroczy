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
