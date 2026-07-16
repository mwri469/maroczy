"""Quarterly fundamentals characteristics.

Functions operate on a quarterly fundamentals DataFrame indexed by
fiscal-quarter-end date, using standard short-form column names (``atq`` =
total assets, ``ibq`` = quarterly income before extraordinary items, ``ceqq``
= common equity, ``saleq`` = quarterly sales, ``epspxq`` = quarterly EPS,
``cshprq`` = shares used for EPS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


def _get(fundq: pd.DataFrame, col: str) -> pd.Series:
    if col not in fundq.columns:
        raise KeyError(f"Quarterly fundamentals frame is missing required column {col!r}.")
    return fundq[col]


def _col(fundq: pd.DataFrame, col: str, default: float = 0) -> pd.Series:
    """Get column or return a default-valued Series (safe for chaining .fillna etc)."""
    if col in fundq.columns:
        return fundq[col].fillna(default)
    return pd.Series(default, index=fundq.index)


@characteristic("niq_at")
def niq_at(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): quarterly return on assets."""
    return _get(fundq, "ibq") / _get(fundq, "atq")


@characteristic("niq_at_chg1")
def niq_at_chg1(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): change in quarterly ROA."""
    return niq_at(fundq).diff()


@characteristic("niq_be")
def niq_be(fundq: pd.DataFrame) -> pd.Series:
    """Hou, Xue & Zhang (2015): quarterly return on equity."""
    return _get(fundq, "ibq") / _get(fundq, "ceqq")


@characteristic("niq_be_chg1")
def niq_be_chg1(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): change in quarterly ROE."""
    return niq_be(fundq).diff()


@characteristic("niq_su")
def niq_su(fundq: pd.DataFrame, window: int = 8) -> pd.Series:
    """Foster, Olsen & Shevlin (1984): standardized unexpected earnings (SUE).

    Seasonal random-walk surprise: ``(eps_t - eps_{t-4}) / std(eps_t - eps_{t-4})``
    over the trailing ``window`` quarters.
    """
    eps = _get(fundq, "epspxq")
    surprise = eps - eps.shift(4)
    return surprise / surprise.rolling(window).std()


@characteristic("saleq_su")
def saleq_su(fundq: pd.DataFrame, window: int = 8) -> pd.Series:
    """Jegadeesh & Livnat (2006): revenue surprise (seasonal random walk, scaled by price)."""
    sale_per_share = _get(fundq, "saleq") / _get(fundq, "cshprq")
    surprise = sale_per_share - sale_per_share.shift(4)
    return surprise / surprise.rolling(window).std()


@characteristic("roaq")
def roaq(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): quarterly ROA (alias of ``niq_at``)."""
    return niq_at(fundq)


@characteristic("roavol")
def roavol(fundq: pd.DataFrame, window: int = 16) -> pd.Series:
    """Francis et al. (2004): volatility of quarterly ROA."""
    return niq_at(fundq).rolling(window).std()


@characteristic("ni_inc8q")
def ni_inc8q(fundq: pd.DataFrame) -> pd.Series:
    """Barth, Elliott & Finn (1999): number of consecutive quarters with earnings increases."""
    ibq = _get(fundq, "ibq")
    increase = (ibq > ibq.shift(1)).astype(float)
    # Count consecutive increases (reset on any decrease)
    groups = (~increase.astype(bool)).cumsum()
    return increase.groupby(groups).cumsum()


@characteristic("ocfq_saleq_std")
def ocfq_saleq_std(fundq: pd.DataFrame, window: int = 16) -> pd.Series:
    """Huang (2009): cash flow volatility (std of quarterly OCF/sales)."""
    ocf = fundq.get("oancfq", fundq.get("ibq", _get(fundq, "ibq")))
    saleq = _get(fundq, "saleq")
    ratio = ocf / saleq.replace(0, np.nan)
    return ratio.rolling(window).std()


@characteristic("stdacc")
def stdacc(fundq: pd.DataFrame, window: int = 16) -> pd.Series:
    """Bandyopadhyay, Huang & Wirjanto (2010): accrual volatility."""
    ibq = _get(fundq, "ibq")
    oancfq = fundq.get("oancfq", pd.Series(0, index=fundq.index)).fillna(0)
    atq = _get(fundq, "atq")
    accrual = (ibq - oancfq) / atq
    return accrual.rolling(window).std()


@characteristic("chtx")
def chtx(fundq: pd.DataFrame) -> pd.Series:
    """Thomas & Zhang (2011): tax expense surprise (change in quarterly tax / lagged assets)."""
    txtq = fundq.get("txtq", pd.Series(0, index=fundq.index)).fillna(0)
    atq = _get(fundq, "atq")
    return (txtq - txtq.shift(4)) / atq.shift(4).replace(0, np.nan)


@characteristic("rsup")
def rsup(fundq: pd.DataFrame) -> pd.Series:
    """Karma (2009): revenue surprise (seasonal change in quarterly sales / market cap proxy)."""
    saleq = _get(fundq, "saleq")
    atq = _get(fundq, "atq")
    return (saleq - saleq.shift(4)) / atq.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Quarterly analogs of annual characteristics
# ---------------------------------------------------------------------------

def _avg_atq(fundq: pd.DataFrame) -> pd.Series:
    atq = _get(fundq, "atq")
    return ((atq + atq.shift(1)) / 2).replace(0, np.nan)


@characteristic("at_gr1q")
def at_gr1q(fundq: pd.DataFrame) -> pd.Series:
    """Cooper, Gulen & Schill (2008) quarterly: asset growth."""
    return _get(fundq, "atq").pct_change(4)


@characteristic("sale_gr1q")
def sale_gr1q(fundq: pd.DataFrame) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994) quarterly: sales growth."""
    return _get(fundq, "saleq").pct_change(4)


@characteristic("be_meq")
def be_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Rosenberg, Reid & Lanstein (1985) quarterly: book-to-market."""
    return _get(fundq, "ceqq") / me


@characteristic("sale_meq")
def sale_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Barbee, Mukherji & Raines (1996) quarterly: sales to price."""
    return _get(fundq, "saleq") * 4 / me


@characteristic("ni_meq")
def ni_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Basu (1977) quarterly: earnings to price."""
    return _get(fundq, "ibq") * 4 / me


@characteristic("debt_meq")
def debt_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Bhandari (1988) quarterly: debt to market."""
    dlttq = _col(fundq, "dlttq")
    dlcq = _col(fundq, "dlcq")
    return (dlttq + dlcq) / me


@characteristic("at_meq")
def at_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Fama & French (1992) quarterly: assets to market."""
    return _get(fundq, "atq") / me


@characteristic("at_beq")
def at_beq(fundq: pd.DataFrame) -> pd.Series:
    """Fama & French (1992) quarterly: book leverage."""
    return _get(fundq, "atq") / _get(fundq, "ceqq").replace(0, np.nan)


@characteristic("ocf_meq")
def ocf_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Desai, Rajgopal & Venkatachalam (2004) quarterly: operating CF to price."""
    oancfq = fundq.get("oancfq", _get(fundq, "ibq"))
    return oancfq * 4 / me


@characteristic("fcf_meq")
def fcf_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994) quarterly: cash flow to price."""
    ibq = _get(fundq, "ibq")
    dpq = _col(fundq, "dpq")
    return (ibq + dpq) * 4 / me


@characteristic("eqnpo_meq")
def eqnpo_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Boudoukh et al. (2007) quarterly: net payout yield."""
    sstkq = _col(fundq, "sstkq")
    prstkcy = _col(fundq, "prstkcy")
    dvq = _col(fundq, "dvq")
    return (dvq + prstkcy - sstkq) / me


@characteristic("eqpo_meq")
def eqpo_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Boudoukh et al. (2007) quarterly: payout yield."""
    prstkcy = _col(fundq, "prstkcy")
    dvq = _col(fundq, "dvq")
    return (dvq + prstkcy) / me


@characteristic("dy_q")
def dy_q(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Litzenberger & Ramaswamy (1979) quarterly: dividend yield."""
    dvq = _col(fundq, "dvq")
    return dvq * 4 / me


@characteristic("rd_meq")
def rd_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Chan, Lakonishok & Sougiannis (2001) quarterly: R&D to market."""
    xrdq = _col(fundq, "xrdq")
    return xrdq * 4 / me


@characteristic("rd_saleq")
def rd_saleq(fundq: pd.DataFrame) -> pd.Series:
    """Chan, Lakonishok & Sougiannis (2001) quarterly: R&D to sales."""
    xrdq = _col(fundq, "xrdq")
    return xrdq / _get(fundq, "saleq").replace(0, np.nan)


@characteristic("cop_atl1q")
def cop_atl1q(fundq: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016) quarterly: cash-based operating profitability to lagged assets."""
    saleq = _get(fundq, "saleq")
    cogsq = _col(fundq, "cogsq")
    xsgaq = _col(fundq, "xsgaq")
    oancfq = _col(fundq, "oancfq")
    ibq = _get(fundq, "ibq")
    accruals = ibq - oancfq
    return (saleq - cogsq - xsgaq - accruals) / _get(fundq, "atq").shift(1)


@characteristic("op_atl1q")
def op_atl1q(fundq: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016) quarterly: operating profits to lagged assets."""
    saleq = _get(fundq, "saleq")
    cogsq = _col(fundq, "cogsq")
    xsgaq = _col(fundq, "xsgaq")
    return (saleq - cogsq - xsgaq) / _get(fundq, "atq").shift(1)


@characteristic("ope_bel1q")
def ope_bel1q(fundq: pd.DataFrame) -> pd.Series:
    """Fama & French (2006) quarterly: operating profits to lagged book equity."""
    saleq = _get(fundq, "saleq")
    cogsq = _col(fundq, "cogsq")
    xsgaq = _col(fundq, "xsgaq")
    return (saleq - cogsq - xsgaq) / _get(fundq, "ceqq").shift(1).replace(0, np.nan)


@characteristic("gp_atl1q")
def gp_atl1q(fundq: pd.DataFrame) -> pd.Series:
    """Novy-Marx (2013) quarterly: gross profits to lagged assets."""
    gpq = fundq.get("gpq", _get(fundq, "saleq") - _col(fundq, "cogsq"))
    return gpq / _get(fundq, "atq").shift(1)


@characteristic("tangibilityq")
def tangibilityq(fundq: pd.DataFrame) -> pd.Series:
    """Hahn & Lee (2009) quarterly: asset tangibility."""
    cheq = _col(fundq, "cheq")
    rectq = _col(fundq, "rectq")
    invtq = _col(fundq, "invtq")
    ppentq = _col(fundq, "ppentq")
    return (cheq + 0.715 * rectq + 0.547 * invtq + 0.535 * ppentq) / _get(fundq, "atq")


@characteristic("at_turnoverq")
def at_turnoverq(fundq: pd.DataFrame) -> pd.Series:
    """Haugen & Baker (1996) quarterly: capital turnover."""
    saleq = _get(fundq, "saleq")
    atq = _get(fundq, "atq")
    return saleq / ((atq + atq.shift(1)) / 2).replace(0, np.nan)


@characteristic("ebit_saleq")
def ebit_saleq(fundq: pd.DataFrame) -> pd.Series:
    """Soliman (2008) quarterly: profit margin."""
    saleq = _get(fundq, "saleq")
    cogsq = _col(fundq, "cogsq")
    xsgaq = _col(fundq, "xsgaq")
    opq = saleq - cogsq - xsgaq
    return opq / saleq.replace(0, np.nan)


@characteristic("ebit_bevq")
def ebit_bevq(fundq: pd.DataFrame) -> pd.Series:
    """Soliman (2008) quarterly: return on net operating assets."""
    saleq = _get(fundq, "saleq")
    cogsq = _col(fundq, "cogsq")
    xsgaq = _col(fundq, "xsgaq")
    opq = saleq - cogsq - xsgaq
    atq = _get(fundq, "atq")
    cheq = _col(fundq, "cheq")
    dlcq = _col(fundq, "dlcq")
    dlttq = _col(fundq, "dlttq")
    ceqq = _get(fundq, "ceqq")
    noa = (atq - cheq) - (atq - dlcq - dlttq - ceqq)
    return opq / noa.replace(0, np.nan)


@characteristic("sale_bevq")
def sale_bevq(fundq: pd.DataFrame) -> pd.Series:
    """Soliman (2008) quarterly: asset turnover (sale / NOA)."""
    saleq = _get(fundq, "saleq")
    atq = _get(fundq, "atq")
    cheq = _col(fundq, "cheq")
    dlcq = _col(fundq, "dlcq")
    dlttq = _col(fundq, "dlttq")
    ceqq = _get(fundq, "ceqq")
    noa = (atq - cheq) - (atq - dlcq - dlttq - ceqq)
    return saleq / noa.replace(0, np.nan)


@characteristic("opex_atq")
def opex_atq(fundq: pd.DataFrame) -> pd.Series:
    """Novy-Marx (2011) quarterly: operating leverage."""
    xoprq = fundq.get("xoprq", _col(fundq, "cogsq") + _col(fundq, "xsgaq"))
    return xoprq / _get(fundq, "atq")


@characteristic("z_scoreq")
def z_scoreq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Dichev (1998) quarterly: Altman Z-score."""
    atq = _get(fundq, "atq")
    actq = fundq.get("actq", pd.Series(0, index=fundq.index)).fillna(0)
    lctq = fundq.get("lctq", pd.Series(0, index=fundq.index)).fillna(0)
    ltq = fundq.get("ltq", atq - _get(fundq, "ceqq"))
    saleq = _get(fundq, "saleq")
    ibq = _get(fundq, "ibq")
    req = fundq.get("req", pd.Series(0, index=fundq.index)).fillna(0)
    wc = actq - lctq
    return (1.2 * wc / atq + 1.4 * req / atq + 3.3 * ibq * 4 / atq
            + 0.6 * me / ltq.replace(0, np.nan) + 1.0 * saleq * 4 / atq)


@characteristic("o_scoreq")
def o_scoreq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Dichev (1998) quarterly: Ohlson O-score."""
    atq = _get(fundq, "atq")
    ltq = fundq.get("ltq", atq - _get(fundq, "ceqq"))
    ibq = _get(fundq, "ibq")
    oancfq = fundq.get("oancfq", pd.Series(0, index=fundq.index)).fillna(0)
    lctq = _col(fundq, "lctq")
    actq = _col(fundq, "actq")
    size = np.log(atq)
    tlta = ltq / atq
    wcta = (actq - lctq) / atq
    clca = lctq / actq.replace(0, np.nan)
    oeneg = (ltq > atq).astype(float)
    nita = ibq * 4 / atq
    ffota = oancfq * 4 / atq
    intwo = ((ibq.shift(1) < 0) & (ibq < 0)).astype(float)
    chin = (ibq - ibq.shift(1)) / (ibq.abs() + ibq.shift(1).abs()).replace(0, np.nan)
    return (-1.32 - 0.407 * size + 6.03 * tlta - 1.43 * wcta + 0.076 * clca
            - 1.72 * oeneg - 2.37 * nita - 1.83 * ffota + 0.285 * intwo - 0.521 * chin)


@characteristic("kz_indexq")
def kz_indexq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Lamont, Polk & Saa-Requejo (2001) quarterly: Kaplan-Zingales index."""
    ibq = _get(fundq, "ibq")
    dpq = _col(fundq, "dpq")
    atq = _get(fundq, "atq")
    dlttq = _col(fundq, "dlttq")
    dlcq = _col(fundq, "dlcq")
    ceqq = _get(fundq, "ceqq")
    dvq = _col(fundq, "dvq")
    cheq = _col(fundq, "cheq")
    cf = (ibq + dpq) / atq.shift(1).replace(0, np.nan)
    q = (atq + me - ceqq) / atq
    debt = (dlttq + dlcq) / (dlttq + dlcq + ceqq).replace(0, np.nan)
    div = dvq / atq.shift(1).replace(0, np.nan)
    cash = cheq / atq.shift(1).replace(0, np.nan)
    return -1.002 * cf - 39.368 * div - 1.315 * cash + 3.139 * debt + 0.283 * q


@characteristic("f_scoreq")
def f_scoreq(fundq: pd.DataFrame) -> pd.Series:
    """Piotroski (2000) quarterly: F-score."""
    atq = _get(fundq, "atq")
    ibq = _get(fundq, "ibq")
    oancfq = fundq.get("oancfq", pd.Series(0, index=fundq.index)).fillna(0)
    dlttq = _col(fundq, "dlttq")
    dlcq = _col(fundq, "dlcq")
    actq = _col(fundq, "actq")
    lctq = _col(fundq, "lctq")
    saleq = _get(fundq, "saleq")
    cogsq = _col(fundq, "cogsq")

    roa = ibq / atq
    cfo = oancfq / atq
    d_roa = roa - roa.shift(1)
    accrual = cfo - roa
    d_lever = (dlttq + dlcq) / atq - ((dlttq + dlcq) / atq).shift(1)
    d_liquid = (actq / lctq.replace(0, np.nan)) - (actq / lctq.replace(0, np.nan)).shift(1)
    gm = (saleq - cogsq) / saleq.replace(0, np.nan)
    d_margin = gm - gm.shift(1)
    d_turn = saleq / atq - (saleq / atq).shift(1)

    return (
        (roa > 0).astype(float) +
        (cfo > 0).astype(float) +
        (d_roa > 0).astype(float) +
        (accrual > 0).astype(float) +
        (d_lever < 0).astype(float) +
        (d_liquid > 0).astype(float) +
        (d_margin > 0).astype(float) +
        (d_turn > 0).astype(float)
    )


@characteristic("ebitda_mevq")
def ebitda_mevq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Loughran & Wellman (2011) quarterly: enterprise multiple."""
    oibdpq = fundq.get("oibdpq", _get(fundq, "saleq") - _col(fundq, "cogsq") - _col(fundq, "xsgaq"))
    dlttq = _col(fundq, "dlttq")
    dlcq = _col(fundq, "dlcq")
    mev = me + dlttq + dlcq
    return oibdpq * 4 / mev.replace(0, np.nan)


@characteristic("bev_mevq")
def bev_mevq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Penman, Richardson & Tuna (2007) quarterly: book-to-market enterprise value."""
    ceqq = _get(fundq, "ceqq")
    dlttq = _col(fundq, "dlttq")
    dlcq = _col(fundq, "dlcq")
    bev = ceqq + dlttq + dlcq
    mev = me + dlttq + dlcq
    return bev / mev.replace(0, np.nan)


@characteristic("netdebt_meq")
def netdebt_meq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Penman, Richardson & Tuna (2007) quarterly: net debt to price."""
    dlttq = _col(fundq, "dlttq")
    dlcq = _col(fundq, "dlcq")
    cheq = _col(fundq, "cheq")
    return (dlttq + dlcq - cheq) / me


@characteristic("aliq_atq")
def aliq_atq(fundq: pd.DataFrame) -> pd.Series:
    """Ortiz-Molina & Phillips (2014) quarterly: asset liquidity to book assets."""
    cheq = _col(fundq, "cheq")
    rectq = _col(fundq, "rectq")
    invtq = _col(fundq, "invtq")
    ppentq = _col(fundq, "ppentq")
    return (cheq + 0.75 * rectq + 0.5 * invtq + 0.25 * ppentq) / _get(fundq, "atq")


@characteristic("aliq_matq")
def aliq_matq(fundq: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Ortiz-Molina & Phillips (2014) quarterly: asset liquidity to market assets."""
    cheq = _col(fundq, "cheq")
    rectq = _col(fundq, "rectq")
    invtq = _col(fundq, "invtq")
    ppentq = _col(fundq, "ppentq")
    ltq = fundq.get("ltq", _get(fundq, "atq") - _get(fundq, "ceqq"))
    mat = me + ltq
    return (cheq + 0.75 * rectq + 0.5 * invtq + 0.25 * ppentq) / mat.replace(0, np.nan)


@characteristic("pi_nixq")
def pi_nixq(fundq: pd.DataFrame) -> pd.Series:
    """Lev & Nissim (2004) quarterly: taxable income to income."""
    piq = fundq.get("piq", _get(fundq, "ibq"))
    return piq / _get(fundq, "ibq").abs().replace(0, np.nan)
