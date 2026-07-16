"""Annual fundamentals characteristics.

Functions operate on a fundamentals DataFrame indexed by fiscal-period-end
date, using standard short-form column names (``at`` = total assets, ``ceq``
= common equity, ``sale`` = sales/revenue, ``ni`` = net income,
``dltt``/``dlc`` = long/short-term debt, ``ebit``, ``che`` = cash &
equivalents, ``act``/``lct`` = current assets/liabilities, ``lt`` = total
liabilities, ``re`` = retained earnings, ``capx`` = capex). Functions
needing market equity accept ``me: pd.Series``.  

This is a deliberately small, high-value core subset -- extend by adding
more ``@characteristic``-decorated functions following the same pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


def _get(funda: pd.DataFrame, col: str) -> pd.Series:
    if col not in funda.columns:
        raise KeyError(f"Fundamentals frame is missing required column {col!r}.")
    return funda[col]


@characteristic("at_gr1")
def at_gr1(funda: pd.DataFrame) -> pd.Series:
    """Cooper, Gulen & Schill (2008): asset growth."""
    return _get(funda, "at").pct_change()


@characteristic("sale_gr1")
def sale_gr1(funda: pd.DataFrame) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994): annual sales growth."""
    return _get(funda, "sale").pct_change()


@characteristic("sale_gr3")
def sale_gr3(funda: pd.DataFrame) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994): three-year sales growth."""
    sale = _get(funda, "sale")
    return sale / sale.shift(3) - 1


@characteristic("capx_gr2")
def capx_gr2(funda: pd.DataFrame) -> pd.Series:
    """Anderson & Garcia-Feijoo (2006): two-year capex growth."""
    capx = _get(funda, "capx")
    return capx / capx.shift(2) - 1


@characteristic("capx_gr3")
def capx_gr3(funda: pd.DataFrame) -> pd.Series:
    """Anderson & Garcia-Feijoo (2006): three-year capex growth."""
    capx = _get(funda, "capx")
    return capx / capx.shift(3) - 1


@characteristic("ni_me")
def ni_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Basu (1977): earnings to price (``ni / market equity``)."""
    return _get(funda, "ni") / me


@characteristic("debt_me")
def debt_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Bhandari (1988): debt to market (``(dltt + dlc) / market equity``)."""
    debt = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0)
    return debt / me


@characteristic("at_me")
def at_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Fama & French (1992): assets to market."""
    return _get(funda, "at") / me


@characteristic("fcf_me")
def fcf_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994): free cash flow to price."""
    ocf = funda.get("oancf")
    capx = funda.get("capx", 0)
    if ocf is None:
        raise KeyError("Fundamentals frame is missing required column 'oancf'.")
    return (ocf - capx.fillna(0)) / me


@characteristic("op_at")
def op_at(funda: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016): operating profitability to assets, ``(sale - cogs - xsga) / at``."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    return (sale - cogs - xsga) / _get(funda, "at")


@characteristic("roic")
def roic(funda: pd.DataFrame) -> pd.Series:
    """Brown & Rowe (2007): return on invested capital, ``ebit / (debt + equity - cash)``."""
    debt = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0)
    invested = debt + _get(funda, "ceq") - funda.get("che", 0).fillna(0)
    return _get(funda, "ebit") / invested.replace(0, np.nan)


@characteristic("z_score")
def z_score(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Dichev (1998): Altman Z-score (simplified 5-factor formulation)."""
    at_ = _get(funda, "at")
    wc = funda.get("act", np.nan) - funda.get("lct", np.nan)
    re = funda.get("re", np.nan)
    ebit = _get(funda, "ebit")
    sale = _get(funda, "sale")
    lt = _get(funda, "lt")
    return 1.2 * wc / at_ + 1.4 * re / at_ + 3.3 * ebit / at_ + 0.6 * me / lt + 1.0 * sale / at_


@characteristic("tangibility")
def tangibility(funda: pd.DataFrame) -> pd.Series:
    """Hahn & Lee (2009): asset tangibility, ``(che + 0.715*rect + 0.547*invt + 0.535*ppent) / at``."""
    che = funda.get("che", 0).fillna(0)
    rect = funda.get("rect", 0).fillna(0)
    invt = funda.get("invt", 0).fillna(0)
    ppent = funda.get("ppent", 0).fillna(0)
    return (che + 0.715 * rect + 0.547 * invt + 0.535 * ppent) / _get(funda, "at")


@characteristic("noa_at")
def noa_at(funda: pd.DataFrame) -> pd.Series:
    """Hirshleifer et al. (2004): net operating assets to total assets."""
    oa = funda.get("act", 0).fillna(0) + _get(funda, "at") - funda.get("che", 0).fillna(0) - funda.get("ivao", 0).fillna(0)
    ol = _get(funda, "at") - funda.get("dlc", 0).fillna(0) - funda.get("dltt", 0).fillna(0) - funda.get("mib", 0).fillna(0) - funda.get("pstk", 0).fillna(0) - _get(funda, "ceq")
    return (oa - ol) / _get(funda, "at").shift(1)


# ---------------------------------------------------------------------------
# Growth / investment
# ---------------------------------------------------------------------------

@characteristic("capx_gr1")
def capx_gr1(funda: pd.DataFrame) -> pd.Series:
    """Xie (2001): one-year capex growth."""
    return _get(funda, "capx").pct_change()


@characteristic("inv_gr1")
def inv_gr1(funda: pd.DataFrame) -> pd.Series:
    """Belo & Lin (2012): inventory growth."""
    invt = _get(funda, "invt")
    return invt.pct_change()


@characteristic("inv_gr1a")
def inv_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Thomas & Zhang (2002): inventory change scaled by average assets."""
    invt = _get(funda, "invt")
    at_ = _get(funda, "at")
    return (invt - invt.shift(1)) / ((at_ + at_.shift(1)) / 2)


@characteristic("emp_gr1")
def emp_gr1(funda: pd.DataFrame) -> pd.Series:
    """Belo, Lin & Bazdresch (2014): employment growth."""
    emp = _get(funda, "emp")
    avg = (emp + emp.shift(1)) / 2
    return (emp - emp.shift(1)) / avg.replace(0, np.nan)


@characteristic("ppeinv_gr1a")
def ppeinv_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Lyandres, Sun & Zhang (2008): change in PPE and inventory / assets."""
    ppe = funda.get("ppegt", funda.get("ppent", pd.Series(0, index=funda.index))).fillna(0)
    invt = funda.get("invt", 0).fillna(0)
    at_ = _get(funda, "at")
    numerator = (ppe + invt) - (ppe.shift(1) + invt.shift(1))
    return numerator / at_.shift(1).replace(0, np.nan)


@characteristic("debt_gr3")
def debt_gr3(funda: pd.DataFrame) -> pd.Series:
    """Lyandres, Sun & Zhang (2008): composite debt issuance (3-year growth in total debt)."""
    debt = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0)
    return debt / debt.shift(3).replace(0, np.nan) - 1


@characteristic("capex_abn")
def capex_abn(funda: pd.DataFrame) -> pd.Series:
    """Titman, Wei & Xie (2004): abnormal capital investment."""
    capx = _get(funda, "capx")
    sale = _get(funda, "sale")
    ce_ratio = capx / sale.replace(0, np.nan)
    avg_3yr = (ce_ratio.shift(1) + ce_ratio.shift(2) + ce_ratio.shift(3)) / 3
    return ce_ratio / avg_3yr.replace(0, np.nan) - 1


@characteristic("invest")
def invest(funda: pd.DataFrame) -> pd.Series:
    """Chen & Zhang (2010): CAPEX + inventory scaled by assets."""
    capx = _get(funda, "capx")
    invt_chg = funda.get("invt", 0).fillna(0).diff()
    at_ = _get(funda, "at")
    return (capx + invt_chg) / at_.shift(1).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Balance sheet decomposition (Richardson et al. 2005)
# ---------------------------------------------------------------------------

def _avg_at(funda: pd.DataFrame) -> pd.Series:
    at_ = _get(funda, "at")
    return ((at_ + at_.shift(1)) / 2).replace(0, np.nan)


@characteristic("coa_gr1a")
def coa_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in current operating assets / avg assets."""
    coa = funda.get("act", 0).fillna(0) - funda.get("che", 0).fillna(0)
    return coa.diff() / _avg_at(funda)


@characteristic("col_gr1a")
def col_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in current operating liabilities / avg assets."""
    col = funda.get("lct", 0).fillna(0) - funda.get("dlc", 0).fillna(0)
    return col.diff() / _avg_at(funda)


@characteristic("cowc_gr1a")
def cowc_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in net non-cash working capital / avg assets."""
    coa = funda.get("act", 0).fillna(0) - funda.get("che", 0).fillna(0)
    col = funda.get("lct", 0).fillna(0) - funda.get("dlc", 0).fillna(0)
    cowc = coa - col
    return cowc.diff() / _avg_at(funda)


@characteristic("ncoa_gr1a")
def ncoa_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in non-current operating assets / avg assets."""
    at_ = _get(funda, "at")
    ncoa = at_ - funda.get("act", 0).fillna(0) - funda.get("ivao", 0).fillna(0)
    return ncoa.diff() / _avg_at(funda)


@characteristic("ncol_gr1a")
def ncol_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in non-current operating liabilities / avg assets."""
    lt_ = _get(funda, "lt")
    ncol = lt_ - funda.get("lct", 0).fillna(0) - funda.get("dltt", 0).fillna(0)
    return ncol.diff() / _avg_at(funda)


@characteristic("nncoa_gr1a")
def nncoa_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in net non-current operating assets / avg assets."""
    at_ = _get(funda, "at")
    lt_ = _get(funda, "lt")
    ncoa = at_ - funda.get("act", 0).fillna(0) - funda.get("ivao", 0).fillna(0)
    ncol = lt_ - funda.get("lct", 0).fillna(0) - funda.get("dltt", 0).fillna(0)
    nncoa = ncoa - ncol
    return nncoa.diff() / _avg_at(funda)


@characteristic("fnl_gr1a")
def fnl_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in financial liabilities / avg assets."""
    fnl = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0) + funda.get("pstk", 0).fillna(0)
    return fnl.diff() / _avg_at(funda)


@characteristic("nfna_gr1a")
def nfna_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in net financial assets / avg assets."""
    fna = funda.get("ivst", 0).fillna(0) + funda.get("ivao", 0).fillna(0)
    fnl = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0) + funda.get("pstk", 0).fillna(0)
    nfna = fna - fnl
    return nfna.diff() / _avg_at(funda)


@characteristic("be_gr1a")
def be_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in common equity / avg assets."""
    ceq = _get(funda, "ceq")
    return ceq.diff() / _avg_at(funda)


@characteristic("lti_gr1a")
def lti_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in long-term investments / avg assets."""
    lti = funda.get("ivao", 0).fillna(0)
    return lti.diff() / _avg_at(funda)


@characteristic("sti_gr1a")
def sti_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): change in short-term investments / avg assets."""
    sti = funda.get("ivst", 0).fillna(0)
    return sti.diff() / _avg_at(funda)


@characteristic("taccruals_at")
def taccruals_at(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005): total accruals / avg assets."""
    at_ = _get(funda, "at")
    coa = funda.get("act", 0).fillna(0) - funda.get("che", 0).fillna(0)
    col = funda.get("lct", 0).fillna(0) - funda.get("dlc", 0).fillna(0)
    ncoa = at_ - funda.get("act", 0).fillna(0) - funda.get("ivao", 0).fillna(0)
    lt_ = _get(funda, "lt")
    ncol = lt_ - funda.get("lct", 0).fillna(0) - funda.get("dltt", 0).fillna(0)
    fna = funda.get("ivst", 0).fillna(0) + funda.get("ivao", 0).fillna(0)
    fnl = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0) + funda.get("pstk", 0).fillna(0)
    tacc = (coa - col + ncoa - ncol + fna - fnl).diff()
    return tacc / _avg_at(funda)


@characteristic("noa_gr1a")
def noa_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Hirshleifer et al. (2004): change in net operating assets / avg assets."""
    at_ = _get(funda, "at")
    lt_ = _get(funda, "lt")
    oa = funda.get("act", 0).fillna(0) + at_ - funda.get("che", 0).fillna(0) - funda.get("ivao", 0).fillna(0)
    ol = at_ - funda.get("dlc", 0).fillna(0) - funda.get("dltt", 0).fillna(0) - funda.get("mib", 0).fillna(0) - funda.get("pstk", 0).fillna(0) - _get(funda, "ceq")
    noa = oa - ol
    return noa.diff() / _avg_at(funda)


@characteristic("lnoa_gr1a")
def lnoa_gr1a(funda: pd.DataFrame) -> pd.Series:
    """Fairfield, Whisenant & Yohn (2003): change in long-term net operating assets / avg assets."""
    at_ = _get(funda, "at")
    lt_ = _get(funda, "lt")
    ppent = funda.get("ppent", 0).fillna(0)
    ivao = funda.get("ivao", 0).fillna(0)
    dltt = funda.get("dltt", 0).fillna(0)
    mib = funda.get("mib", 0).fillna(0)
    dp = funda.get("dp", 0).fillna(0)
    lnoa = ppent + ivao - dltt - mib
    return lnoa.diff() / _avg_at(funda)


@characteristic("lgr")
def lgr(funda: pd.DataFrame) -> pd.Series:
    """Richardson et al. (2005) GHZ: change in long-term debt."""
    lt_ = _get(funda, "lt")
    return lt_.pct_change()


# ---------------------------------------------------------------------------
# Accruals
# ---------------------------------------------------------------------------

@characteristic("acc")
def acc(funda: pd.DataFrame) -> pd.Series:
    """Sloan (1996) GHZ: operating accruals (original balance-sheet definition)."""
    act = funda.get("act", 0).fillna(0)
    lct = funda.get("lct", 0).fillna(0)
    che = funda.get("che", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    txp = funda.get("txp", 0).fillna(0)
    dp = funda.get("dp", 0).fillna(0)
    d_ca = act.diff() - che.diff()
    d_cl = lct.diff() - dlc.diff() - txp.diff()
    at_ = _get(funda, "at")
    return (d_ca - d_cl - dp) / ((at_ + at_.shift(1)) / 2)


@characteristic("oaccruals_at")
def oaccruals_at(funda: pd.DataFrame) -> pd.Series:
    """Sloan (1996) JKP: operating accruals / assets."""
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(np.nan, index=funda.index))
    at_ = _get(funda, "at")
    return (ni - oancf) / at_


@characteristic("oaccruals_ni")
def oaccruals_ni(funda: pd.DataFrame) -> pd.Series:
    """Hafzalla, Lundholm & Van Winkle (2011) JKP: percent operating accruals (scaled by |ni|)."""
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(np.nan, index=funda.index))
    return (ni - oancf) / ni.abs().replace(0, np.nan)


@characteristic("taccruals_ni")
def taccruals_ni(funda: pd.DataFrame) -> pd.Series:
    """Hafzalla, Lundholm & Van Winkle (2011): percent total accruals."""
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(np.nan, index=funda.index))
    # Total accruals = NI - total cash flow (operating + investing)
    ivncf = funda.get("ivncf", 0).fillna(0)
    return (ni - oancf - ivncf) / ni.abs().replace(0, np.nan)


@characteristic("pctacc")
def pctacc(funda: pd.DataFrame) -> pd.Series:
    """Hafzalla, Lundholm & Van Winkle (2011) GHZ: percent operating accruals (original)."""
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(np.nan, index=funda.index))
    return (ni - oancf) / ni.abs().replace(0, np.nan)


@characteristic("absacc")
def absacc(funda: pd.DataFrame) -> pd.Series:
    """Bandyopadhyay, Huang & Wirjanto (2010): absolute accruals."""
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(np.nan, index=funda.index))
    at_ = _get(funda, "at")
    return ((ni - oancf) / at_).abs()


# ---------------------------------------------------------------------------
# Profitability
# ---------------------------------------------------------------------------

@characteristic("gp_at")
def gp_at(funda: pd.DataFrame) -> pd.Series:
    """Novy-Marx (2013): gross profits to assets."""
    gp = funda.get("gp", _get(funda, "sale") - funda.get("cogs", 0).fillna(0))
    return gp / _get(funda, "at")


@characteristic("gp_atl1")
def gp_atl1(funda: pd.DataFrame) -> pd.Series:
    """Novy-Marx (2013): gross profits to lagged assets."""
    gp = funda.get("gp", _get(funda, "sale") - funda.get("cogs", 0).fillna(0))
    return gp / _get(funda, "at").shift(1)


@characteristic("op_atl1")
def op_atl1(funda: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016): operating profits to lagged assets."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    return (sale - cogs - xsga) / _get(funda, "at").shift(1)


@characteristic("cop_at")
def cop_at(funda: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016): cash-based operating profitability to assets."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    oancf = funda.get("oancf", 0).fillna(0)
    ni = _get(funda, "ni")
    # COP = operating profits - accruals = oancf + (sale - cogs - xsga) - ni ... simplified:
    accruals = ni - oancf
    return (sale - cogs - xsga - accruals) / _get(funda, "at")


@characteristic("cop_atl1")
def cop_atl1(funda: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016): cash-based operating profitability to lagged assets."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    oancf = funda.get("oancf", 0).fillna(0)
    ni = _get(funda, "ni")
    accruals = ni - oancf
    return (sale - cogs - xsga - accruals) / _get(funda, "at").shift(1)


@characteristic("ni_be")
def ni_be(funda: pd.DataFrame) -> pd.Series:
    """Haugen & Baker (1996): return on equity (ni / book equity)."""
    return _get(funda, "ni") / _get(funda, "ceq").replace(0, np.nan)


@characteristic("operprof")
def operprof(funda: pd.DataFrame) -> pd.Series:
    """Fama & French (2006) GHZ: operating profits to book equity."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    xint = funda.get("xint", 0).fillna(0)
    return (sale - cogs - xsga - xint) / _get(funda, "ceq").replace(0, np.nan)


@characteristic("ope_be")
def ope_be(funda: pd.DataFrame) -> pd.Series:
    """Fama & French (2006) JKP: operating profits to book equity."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    return (sale - cogs - xsga) / _get(funda, "ceq").replace(0, np.nan)


@characteristic("ope_bel1")
def ope_bel1(funda: pd.DataFrame) -> pd.Series:
    """Fama & French (2006): operating profits to lagged book equity."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    return (sale - cogs - xsga) / _get(funda, "ceq").shift(1).replace(0, np.nan)


@characteristic("ocf_at")
def ocf_at(funda: pd.DataFrame) -> pd.Series:
    """Bouchard et al. (2019): operating cash flow to assets."""
    oancf = funda.get("oancf")
    if oancf is None:
        raise KeyError("Fundamentals frame is missing required column 'oancf'.")
    return oancf / _get(funda, "at")


@characteristic("ocf_at_chg1")
def ocf_at_chg1(funda: pd.DataFrame) -> pd.Series:
    """Bouchard et al. (2019): change in operating cash flow to assets."""
    return ocf_at(funda).diff()


@characteristic("ocf_me")
def ocf_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Desai, Rajgopal & Venkatachalam (2004): operating cash flows to price."""
    oancf = funda.get("oancf")
    if oancf is None:
        raise KeyError("Fundamentals frame is missing required column 'oancf'.")
    return oancf / me


@characteristic("cfp")
def cfp(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Desai, Rajgopal & Venkatachalam (2004) GHZ: cash flow to price."""
    ib = funda.get("ib", _get(funda, "ni"))
    dp = funda.get("dp", 0).fillna(0)
    return (ib + dp) / me


@characteristic("ni_ar1")
def ni_ar1(funda: pd.DataFrame) -> pd.Series:
    """Francis et al. (2004): earnings persistence (AR(1) coefficient of NI/AT)."""
    roa = _get(funda, "ni") / _get(funda, "at")
    # Rolling 10-year AR(1) coefficient
    out = roa.rolling(10).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan, raw=True
    )
    return out


@characteristic("ni_ivol")
def ni_ivol(funda: pd.DataFrame) -> pd.Series:
    """Francis et al. (2004): earnings predictability (volatility of NI/AT residuals)."""
    roa = _get(funda, "ni") / _get(funda, "at")
    return roa.rolling(10).std()


@characteristic("earnings_variability")
def earnings_variability(funda: pd.DataFrame) -> pd.Series:
    """Francis et al. (2004): earnings smoothness = std(NI/AT) / std(OCF/AT)."""
    at_ = _get(funda, "at")
    roa = _get(funda, "ni") / at_
    oancf = funda.get("oancf", pd.Series(np.nan, index=funda.index))
    cfo = oancf / at_
    return roa.rolling(10).std() / cfo.rolling(10).std().replace(0, np.nan)


# ---------------------------------------------------------------------------
# Valuation ratios
# ---------------------------------------------------------------------------

@characteristic("be_me")
def be_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Rosenberg, Reid & Lanstein (1985): book-to-market."""
    return _get(funda, "ceq") / me


@characteristic("sale_me")
def sale_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Barbee, Mukherji & Raines (1996): sales to price."""
    return _get(funda, "sale") / me


@characteristic("ebitda_mev")
def ebitda_mev(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Loughran & Wellman (2011) JKP: EBITDA / market enterprise value."""
    ebitda = funda.get("ebitda", funda.get("oibdp", _get(funda, "ebit")))
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    mev = me + dltt + dlc
    return ebitda / mev.replace(0, np.nan)


@characteristic("enterprise_multiple")
def enterprise_multiple(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Loughran & Wellman (2011): enterprise multiple = EV / EBITDA."""
    oibdp = funda.get("oibdp", funda.get("ebitda", _get(funda, "ebit")))
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    pstk = funda.get("pstkrv", funda.get("pstk", 0)).fillna(0)
    ev = me + dltt + dlc + pstk - funda.get("che", 0).fillna(0)
    return ev / oibdp.replace(0, np.nan)


@characteristic("bev_mev")
def bev_mev(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Penman, Richardson & Tuna (2007): book-to-market enterprise value."""
    ceq = _get(funda, "ceq")
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    bev = ceq + dltt + dlc
    mev = me + dltt + dlc
    return bev / mev.replace(0, np.nan)


@characteristic("netdebt_me")
def netdebt_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Penman, Richardson & Tuna (2007): net debt to price."""
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    che = funda.get("che", 0).fillna(0)
    return (dltt + dlc - che) / me


@characteristic("at_be")
def at_be(funda: pd.DataFrame) -> pd.Series:
    """Fama & French (1992): book leverage (assets / book equity)."""
    return _get(funda, "at") / _get(funda, "ceq").replace(0, np.nan)


@characteristic("rd_me")
def rd_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Chan, Lakonishok & Sougiannis (2001): R&D to market."""
    xrd = funda.get("xrd", 0).fillna(0)
    return xrd / me


@characteristic("rd_sale")
def rd_sale(funda: pd.DataFrame) -> pd.Series:
    """Chan, Lakonishok & Sougiannis (2001): R&D to sales."""
    xrd = funda.get("xrd", 0).fillna(0)
    return xrd / _get(funda, "sale").replace(0, np.nan)


@characteristic("rd5_at")
def rd5_at(funda: pd.DataFrame) -> pd.Series:
    """Li (2011): R&D capital to assets (5-year cumulative with 20% depreciation)."""
    xrd = funda.get("xrd", 0).fillna(0)
    rd_cap = xrd + 0.8 * xrd.shift(1) + 0.6 * xrd.shift(2) + 0.4 * xrd.shift(3) + 0.2 * xrd.shift(4)
    return rd_cap / _get(funda, "at")


# ---------------------------------------------------------------------------
# DuPont decomposition / turnover
# ---------------------------------------------------------------------------

@characteristic("at_turnover")
def at_turnover(funda: pd.DataFrame) -> pd.Series:
    """Haugen & Baker (1996): capital turnover (sale / avg assets)."""
    sale = _get(funda, "sale")
    at_ = _get(funda, "at")
    return sale / ((at_ + at_.shift(1)) / 2).replace(0, np.nan)


@characteristic("sale_bev")
def sale_bev(funda: pd.DataFrame) -> pd.Series:
    """Soliman (2008): asset turnover (sale / net operating assets)."""
    sale = _get(funda, "sale")
    noa = _get(funda, "at") - funda.get("che", 0).fillna(0) - (
        _get(funda, "at") - funda.get("dlc", 0).fillna(0) - funda.get("dltt", 0).fillna(0) - funda.get("mib", 0).fillna(0) - funda.get("pstk", 0).fillna(0) - _get(funda, "ceq")
    )
    return sale / noa.replace(0, np.nan)


@characteristic("ebit_sale")
def ebit_sale(funda: pd.DataFrame) -> pd.Series:
    """Soliman (2008): profit margin (EBIT / sale)."""
    return _get(funda, "ebit") / _get(funda, "sale").replace(0, np.nan)


@characteristic("ebit_bev")
def ebit_bev(funda: pd.DataFrame) -> pd.Series:
    """Soliman (2008): return on net operating assets (EBIT / NOA)."""
    ebit = _get(funda, "ebit")
    noa = _get(funda, "at") - funda.get("che", 0).fillna(0) - (
        _get(funda, "at") - funda.get("dlc", 0).fillna(0) - funda.get("dltt", 0).fillna(0) - funda.get("mib", 0).fillna(0) - funda.get("pstk", 0).fillna(0) - _get(funda, "ceq")
    )
    return ebit / noa.replace(0, np.nan)


@characteristic("chatoia")
def chatoia(funda: pd.DataFrame) -> pd.Series:
    """Soliman (2008): change in asset turnover."""
    return at_turnover(funda).diff()


@characteristic("chpmia")
def chpmia(funda: pd.DataFrame) -> pd.Series:
    """Soliman (2008): change in profit margin."""
    return ebit_sale(funda).diff()


@characteristic("opex_at")
def opex_at(funda: pd.DataFrame) -> pd.Series:
    """Novy-Marx (2011): operating leverage (operating expenses / assets)."""
    xopr = funda.get("xopr", funda.get("cogs", 0).fillna(0) + funda.get("xsga", 0).fillna(0))
    return xopr / _get(funda, "at")


# ---------------------------------------------------------------------------
# Issuance / payout
# ---------------------------------------------------------------------------

@characteristic("eqnpo_me")
def eqnpo_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Boudoukh et al. (2007): net payout yield."""
    sstk = funda.get("sstk", 0).fillna(0)
    prstkc = funda.get("prstkc", 0).fillna(0)
    dv = funda.get("dv", funda.get("dvt", 0)).fillna(0)
    net_payout = dv + prstkc - sstk
    return net_payout / me


@characteristic("eqpo_me")
def eqpo_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Boudoukh et al. (2007): payout yield."""
    prstkc = funda.get("prstkc", 0).fillna(0)
    dv = funda.get("dv", funda.get("dvt", 0)).fillna(0)
    return (dv + prstkc) / me


@characteristic("chcsho")
def chcsho(funda: pd.DataFrame) -> pd.Series:
    """Pontiff & Woodgate (2008) GHZ: net stock issues (using funda)."""
    csho = _get(funda, "csho")
    return csho.pct_change()


@characteristic("dbnetis_at")
def dbnetis_at(funda: pd.DataFrame) -> pd.Series:
    """Bradshaw, Richardson & Sloan (2006): net debt finance."""
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    debt = dltt + dlc
    return debt.diff() / _get(funda, "at")


@characteristic("eqnetis_at")
def eqnetis_at(funda: pd.DataFrame) -> pd.Series:
    """Bradshaw, Richardson & Sloan (2006): net equity finance."""
    sstk = funda.get("sstk", 0).fillna(0)
    prstkc = funda.get("prstkc", 0).fillna(0)
    dv = funda.get("dv", funda.get("dvt", 0)).fillna(0)
    return (sstk - prstkc - dv) / _get(funda, "at")


@characteristic("netis_at")
def netis_at(funda: pd.DataFrame) -> pd.Series:
    """Bradshaw, Richardson & Sloan (2006): net external finance."""
    return dbnetis_at(funda) + eqnetis_at(funda)


@characteristic("dy")
def dy(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Litzenberger & Ramaswamy (1979) GHZ: dividend yield (from funda)."""
    dv = funda.get("dv", funda.get("dvt", 0)).fillna(0)
    return dv / me


# ---------------------------------------------------------------------------
# Financial health / distress
# ---------------------------------------------------------------------------

@characteristic("o_score")
def o_score(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Dichev (1998): Ohlson O-score (probability of bankruptcy)."""
    at_ = _get(funda, "at")
    lt_ = _get(funda, "lt")
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(0, index=funda.index)).fillna(0)
    lct = funda.get("lct", 0).fillna(0)
    act = funda.get("act", 0).fillna(0)
    size = np.log(at_)
    tlta = lt_ / at_
    wcta = (act - lct) / at_
    clca = lct / act.replace(0, np.nan)
    oeneg = (lt_ > at_).astype(float)
    nita = ni / at_
    ffota = oancf / at_
    intwo = (ni.shift(1) < 0).astype(float) & (ni < 0).astype(float)
    chin = (ni - ni.shift(1)) / (ni.abs() + ni.shift(1).abs()).replace(0, np.nan)
    return -1.32 - 0.407 * size + 6.03 * tlta - 1.43 * wcta + 0.076 * clca - 1.72 * oeneg - 2.37 * nita - 1.83 * ffota + 0.285 * intwo - 0.521 * chin


@characteristic("kz_index")
def kz_index(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Lamont, Polk & Saa-Requejo (2001): Kaplan-Zingales index."""
    ib = funda.get("ib", _get(funda, "ni"))
    dp = funda.get("dp", 0).fillna(0)
    at_ = _get(funda, "at")
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    ceq = _get(funda, "ceq")
    dv = funda.get("dv", funda.get("dvt", 0)).fillna(0)
    che = funda.get("che", 0).fillna(0)
    cf = (ib + dp) / at_.shift(1).replace(0, np.nan)
    q = (at_ + me - ceq) / at_
    debt = (dltt + dlc) / (dltt + dlc + ceq).replace(0, np.nan)
    div = dv / at_.shift(1).replace(0, np.nan)
    cash = che / at_.shift(1).replace(0, np.nan)
    return -1.002 * cf - 39.368 * div - 1.315 * cash + 3.139 * debt + 0.283 * q


@characteristic("f_score")
def f_score(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Piotroski (2000) JKP: Piotroski F-score (9 binary signals)."""
    at_ = _get(funda, "at")
    ni = _get(funda, "ni")
    oancf = funda.get("oancf", pd.Series(0, index=funda.index)).fillna(0)
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    act = funda.get("act", 0).fillna(0)
    lct = funda.get("lct", 0).fillna(0)
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)

    roa = ni / at_
    cfo = oancf / at_
    d_roa = roa - roa.shift(1)
    accrual = cfo - roa
    d_lever = (dltt + dlc) / at_ - ((dltt + dlc) / at_).shift(1)
    d_liquid = (act / lct.replace(0, np.nan)) - (act / lct.replace(0, np.nan)).shift(1)
    d_margin = (sale - cogs) / sale.replace(0, np.nan) - ((sale - cogs) / sale.replace(0, np.nan)).shift(1)
    d_turn = sale / at_ - (sale / at_).shift(1)
    eq_offer = _get(funda, "csho").diff()

    score = (
        (roa > 0).astype(float) +
        (cfo > 0).astype(float) +
        (d_roa > 0).astype(float) +
        (accrual > 0).astype(float) +
        (d_lever < 0).astype(float) +
        (d_liquid > 0).astype(float) +
        (eq_offer <= 0).astype(float) +
        (d_margin > 0).astype(float) +
        (d_turn > 0).astype(float)
    )
    return score


@characteristic("ps")
def ps(funda: pd.DataFrame) -> pd.Series:
    """Piotroski (2000) GHZ: Piotroski F-score (original definition)."""
    return f_score(funda, me=_get(funda, "at"))  # uses at as placeholder for me


# ---------------------------------------------------------------------------
# Industry-adjusted characteristics
# ---------------------------------------------------------------------------

@characteristic("bm_ia")
def bm_ia(funda: pd.DataFrame, me: pd.Series, industry: pd.Series = None) -> pd.Series:
    """Asness, Porter & Stevens (2000): industry-adjusted book-to-market."""
    bm = _get(funda, "ceq") / me
    if industry is None:
        return bm
    ind_median = bm.groupby(industry).transform("median")
    return bm - ind_median


@characteristic("cfp_ia")
def cfp_ia(funda: pd.DataFrame, me: pd.Series, industry: pd.Series = None) -> pd.Series:
    """Asness, Porter & Stevens (2000): industry-adjusted cash flow to price."""
    ib = funda.get("ib", _get(funda, "ni"))
    dp = funda.get("dp", 0).fillna(0)
    cf = (ib + dp) / me
    if industry is None:
        return cf
    ind_median = cf.groupby(industry).transform("median")
    return cf - ind_median


@characteristic("mve_ia")
def mve_ia(funda: pd.DataFrame, me: pd.Series, industry: pd.Series = None) -> pd.Series:
    """Asness, Porter & Stevens (2000): industry-adjusted firm size."""
    log_me = np.log(me.replace(0, np.nan))
    if industry is None:
        return log_me
    ind_median = log_me.groupby(industry).transform("median")
    return log_me - ind_median


@characteristic("chempia")
def chempia(funda: pd.DataFrame, industry: pd.Series = None) -> pd.Series:
    """Asness, Porter & Stevens (2000): industry-adjusted change in employees."""
    emp = _get(funda, "emp")
    emp_gr = emp.pct_change()
    if industry is None:
        return emp_gr
    ind_median = emp_gr.groupby(industry).transform("median")
    return emp_gr - ind_median


@characteristic("pchcapx_ia")
def pchcapx_ia(funda: pd.DataFrame, industry: pd.Series = None) -> pd.Series:
    """Abarbanell & Bushee (1998): industry-adjusted change in capital investment."""
    capx = _get(funda, "capx")
    capx_gr = capx.pct_change()
    if industry is None:
        return capx_gr
    ind_median = capx_gr.groupby(industry).transform("median")
    return capx_gr - ind_median


# ---------------------------------------------------------------------------
# Growth in fundamentals (Abarbanell & Bushee 1998)
# ---------------------------------------------------------------------------

@characteristic("dgp_dsale")
def dgp_dsale(funda: pd.DataFrame) -> pd.Series:
    """Abarbanell & Bushee (1998): gross margin growth relative to sales growth."""
    gp = funda.get("gp", _get(funda, "sale") - funda.get("cogs", 0).fillna(0))
    sale = _get(funda, "sale")
    gp_avg = (gp.shift(1) + gp.shift(2)) / 2
    sale_avg = (sale.shift(1) + sale.shift(2)) / 2
    gp_gr = (gp - gp.shift(1)) / gp_avg.replace(0, np.nan)
    sale_gr = (sale - sale.shift(1)) / sale_avg.replace(0, np.nan)
    return gp_gr - sale_gr


@characteristic("sale_emp_gr1")
def sale_emp_gr1(funda: pd.DataFrame) -> pd.Series:
    """Abarbanell & Bushee (1998): labor force efficiency (sales per employee growth)."""
    sale_emp = _get(funda, "sale") / _get(funda, "emp").replace(0, np.nan)
    return sale_emp.pct_change()


@characteristic("dsale_dinv")
def dsale_dinv(funda: pd.DataFrame) -> pd.Series:
    """Abarbanell & Bushee (1998): sales growth relative to inventory growth."""
    sale = _get(funda, "sale")
    invt = _get(funda, "invt")
    sale_avg = (sale.shift(1) + sale.shift(2)) / 2
    invt_avg = (invt.shift(1) + invt.shift(2)) / 2
    sale_gr = (sale - sale.shift(1)) / sale_avg.replace(0, np.nan)
    invt_gr = (invt - invt.shift(1)) / invt_avg.replace(0, np.nan)
    return sale_gr - invt_gr


@characteristic("dsale_drec")
def dsale_drec(funda: pd.DataFrame) -> pd.Series:
    """Abarbanell & Bushee (1998): sales growth relative to receivables growth."""
    sale = _get(funda, "sale")
    rect = _get(funda, "rect")
    sale_avg = (sale.shift(1) + sale.shift(2)) / 2
    rect_avg = (rect.shift(1) + rect.shift(2)) / 2
    sale_gr = (sale - sale.shift(1)) / sale_avg.replace(0, np.nan)
    rect_gr = (rect - rect.shift(1)) / rect_avg.replace(0, np.nan)
    return sale_gr - rect_gr


@characteristic("dsale_dsga")
def dsale_dsga(funda: pd.DataFrame) -> pd.Series:
    """Abarbanell & Bushee (1998): sales growth relative to SG&A growth."""
    sale = _get(funda, "sale")
    xsga = _get(funda, "xsga")
    sale_avg = (sale.shift(1) + sale.shift(2)) / 2
    xsga_avg = (xsga.shift(1) + xsga.shift(2)) / 2
    sale_gr = (sale - sale.shift(1)) / sale_avg.replace(0, np.nan)
    xsga_gr = (xsga - xsga.shift(1)) / xsga_avg.replace(0, np.nan)
    return sale_gr - xsga_gr


# ---------------------------------------------------------------------------
# Miscellaneous ratios
# ---------------------------------------------------------------------------

@characteristic("cash_at")
def cash_at(funda: pd.DataFrame) -> pd.Series:
    """Palazzo (2012): cash to assets."""
    return funda.get("che", 0).fillna(0) / _get(funda, "at")


@characteristic("cashpr")
def cashpr(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Chandrashekar & Rao (2009): cash productivity."""
    return (me + funda.get("dltt", 0).fillna(0) - _get(funda, "at")) / funda.get("che", 0).fillna(0).replace(0, np.nan)


@characteristic("cashdebt")
def cashdebt(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): cash flow to debt."""
    ib = funda.get("ib", _get(funda, "ni"))
    dp = funda.get("dp", 0).fillna(0)
    lt_ = _get(funda, "lt")
    return (ib + dp) / ((lt_ + lt_.shift(1)) / 2).replace(0, np.nan)


@characteristic("currat")
def currat(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): current ratio."""
    return funda.get("act", 0).fillna(0) / funda.get("lct", 0).fillna(0).replace(0, np.nan)


@characteristic("pchcurrat")
def pchcurrat(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): change in current ratio."""
    cr = currat(funda)
    return cr.pct_change()


@characteristic("quick")
def quick(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): quick ratio."""
    act = funda.get("act", 0).fillna(0)
    invt = funda.get("invt", 0).fillna(0)
    lct = funda.get("lct", 0).fillna(0)
    return (act - invt) / lct.replace(0, np.nan)


@characteristic("pchquick")
def pchquick(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): change in quick ratio."""
    qr = quick(funda)
    return qr.pct_change()


@characteristic("salecash")
def salecash(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): sales to cash."""
    return _get(funda, "sale") / funda.get("che", 0).fillna(0).replace(0, np.nan)


@characteristic("saleinv")
def saleinv(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): sales to inventory."""
    return _get(funda, "sale") / funda.get("invt", 0).fillna(0).replace(0, np.nan)


@characteristic("pchsaleinv")
def pchsaleinv(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): change in sales to inventory."""
    si = saleinv(funda)
    return si.pct_change()


@characteristic("salerec")
def salerec(funda: pd.DataFrame) -> pd.Series:
    """Ou & Penman (1989): sales to receivables."""
    return _get(funda, "sale") / funda.get("rect", 0).fillna(0).replace(0, np.nan)


@characteristic("depr")
def depr(funda: pd.DataFrame) -> pd.Series:
    """Holthausen & Larcker (1992): depreciation to PP&E."""
    dp = funda.get("dp", 0).fillna(0)
    ppent = funda.get("ppent", 0).fillna(0)
    return dp / ppent.replace(0, np.nan)


@characteristic("pchdepr")
def pchdepr(funda: pd.DataFrame) -> pd.Series:
    """Holthausen & Larcker (1992): change in depreciation to PP&E."""
    return depr(funda).pct_change()


@characteristic("realestate")
def realestate(funda: pd.DataFrame) -> pd.Series:
    """Tuzel (2010): real estate holdings (buildings + land) / PP&E."""
    fatb = funda.get("fatb", 0).fillna(0)
    fatl = funda.get("fatl", 0).fillna(0)
    ppegt = funda.get("ppegt", funda.get("ppent", 0)).fillna(0)
    return (fatb + fatl) / ppegt.replace(0, np.nan)


@characteristic("aliq_at")
def aliq_at(funda: pd.DataFrame) -> pd.Series:
    """Ortiz-Molina & Phillips (2014): asset liquidity to book assets."""
    che = funda.get("che", 0).fillna(0)
    rect = funda.get("rect", 0).fillna(0)
    invt = funda.get("invt", 0).fillna(0)
    ppent = funda.get("ppent", 0).fillna(0)
    intan = funda.get("intan", 0).fillna(0)
    at_ = _get(funda, "at")
    return (che + 0.75 * rect + 0.5 * invt + 0.25 * ppent) / at_


@characteristic("aliq_mat")
def aliq_mat(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Ortiz-Molina & Phillips (2014): asset liquidity to market assets."""
    che = funda.get("che", 0).fillna(0)
    rect = funda.get("rect", 0).fillna(0)
    invt = funda.get("invt", 0).fillna(0)
    ppent = funda.get("ppent", 0).fillna(0)
    lt_ = _get(funda, "lt")
    mat = me + lt_
    return (che + 0.75 * rect + 0.5 * invt + 0.25 * ppent) / mat.replace(0, np.nan)


@characteristic("eq_dur")
def eq_dur(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Dechow, Sloan & Soliman (2004): equity duration (simplified)."""
    ni = _get(funda, "ni")
    be = _get(funda, "ceq")
    roe = ni / be.replace(0, np.nan)
    bm = be / me
    r = 0.12  # assumed discount rate
    # Simplified duration formula
    duration = (1 + r) / r - (1 + r + roe * (bm - 1)) / (r + roe * bm)
    return duration


@characteristic("intrinsic_value")
def intrinsic_value(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Frankel & Lee (1998): intrinsic value to market (residual income model)."""
    be = _get(funda, "ceq")
    ni = _get(funda, "ni")
    roe = ni / be.replace(0, np.nan)
    r = 0.12
    # V = BE * (1 + (ROE - r) / r) simplified
    v = be + (roe - r) * be / r
    return v / me


# ---------------------------------------------------------------------------
# Tax / governance indicators
# ---------------------------------------------------------------------------

@characteristic("pi_nix")
def pi_nix(funda: pd.DataFrame) -> pd.Series:
    """Lev & Nissim (2004) JKP: taxable income to income (pre-tax income / ni)."""
    pi = funda.get("pi", funda.get("oibdp", _get(funda, "ebit")))
    return pi / _get(funda, "ni").abs().replace(0, np.nan)


@characteristic("tb")
def tb(funda: pd.DataFrame) -> pd.Series:
    """Lev & Nissim (2004) GHZ: taxable income to income."""
    pi = funda.get("pi", funda.get("oibdp", _get(funda, "ebit")))
    ib = funda.get("ib", _get(funda, "ni"))
    return pi / ib.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Indicator variables
# ---------------------------------------------------------------------------

@characteristic("rd")
def rd(funda: pd.DataFrame) -> pd.Series:
    """Eberhart, Maxwell & Siddique (2004): unexpected R&D increase indicator."""
    xrd = funda.get("xrd", 0).fillna(0)
    at_ = _get(funda, "at")
    rd_at = xrd / at_
    rd_at_prev = rd_at.shift(1)
    # indicator: R&D/assets increased by >= 5% and R&D > 0
    return ((rd_at - rd_at_prev >= 0.05) & (xrd > 0)).astype(float)


@characteristic("sin")
def sin(funda: pd.DataFrame, sic: pd.Series = None) -> pd.Series:
    """Hong & Kacperczyk (2009): sin stock indicator (alcohol, tobacco, gaming by SIC)."""
    if sic is None:
        return pd.Series(0.0, index=funda.index)
    sic_str = sic.astype(str).str.zfill(4)
    is_sin = (
        sic_str.str[:2].isin(["20", "21"]) |  # tobacco/food (alcohol)
        sic_str.str[:3].isin(["208", "213", "279"]) |  # beverages, tobacco, gaming
        sic_str.str[:4].isin(["2080", "2082", "2083", "2084", "2085", "7993", "7999"])
    )
    return is_sin.astype(float)


@characteristic("convind")
def convind(funda: pd.DataFrame) -> pd.Series:
    """Valta (2016): convertible debt indicator."""
    dc = funda.get("dc", 0).fillna(0)
    return (dc > 0).astype(float)


@characteristic("securedind")
def securedind(funda: pd.DataFrame) -> pd.Series:
    """Valta (2016): secured debt indicator."""
    dm = funda.get("dm", 0).fillna(0)
    return (dm > 0).astype(float)


@characteristic("secured")
def secured(funda: pd.DataFrame) -> pd.Series:
    """Valta (2016): secured debt to total debt."""
    dm = funda.get("dm", 0).fillna(0)
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    return dm / (dltt + dlc).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Industry concentration
# ---------------------------------------------------------------------------

@characteristic("herf_sale")
def herf_sale(funda: pd.DataFrame, industry: pd.Series = None) -> pd.Series:
    """Hou & Robinson (2006): industry concentration by sales (HHI)."""
    if industry is None:
        return pd.Series(np.nan, index=funda.index)
    sale = _get(funda, "sale")
    ind_total = sale.groupby(industry).transform("sum")
    share = sale / ind_total.replace(0, np.nan)
    return (share ** 2).groupby(industry).transform("sum")


@characteristic("herf_at")
def herf_at(funda: pd.DataFrame, industry: pd.Series = None) -> pd.Series:
    """Hou & Robinson (2006): industry concentration by total assets (HHI)."""
    if industry is None:
        return pd.Series(np.nan, index=funda.index)
    at_ = _get(funda, "at")
    ind_total = at_.groupby(industry).transform("sum")
    share = at_ / ind_total.replace(0, np.nan)
    return (share ** 2).groupby(industry).transform("sum")


@characteristic("herf_be")
def herf_be(funda: pd.DataFrame, industry: pd.Series = None) -> pd.Series:
    """Hou & Robinson (2006): industry concentration by book equity (HHI)."""
    if industry is None:
        return pd.Series(np.nan, index=funda.index)
    be = _get(funda, "ceq")
    ind_total = be.groupby(industry).transform("sum")
    share = be / ind_total.replace(0, np.nan)
    return (share ** 2).groupby(industry).transform("sum")


# ---------------------------------------------------------------------------
# Additional implementable characteristics (Group 8 from coverage analysis)
# ---------------------------------------------------------------------------

@characteristic("etr")
def etr(funda: pd.DataFrame) -> pd.Series:
    """Abarbanell & Bushee (1998): effective tax rate."""
    txt = funda.get("txt", 0).fillna(0)
    pi = funda.get("pi", _get(funda, "ebit"))
    return txt / pi.replace(0, np.nan)


@characteristic("cfp_vol")
def cfp_vol(funda: pd.DataFrame, me: pd.Series, window: int = 5) -> pd.Series:
    """Haugen & Baker (1996): cash flow to price volatility."""
    ib = funda.get("ib", _get(funda, "ni"))
    dp = funda.get("dp", 0).fillna(0)
    ratio = (ib + dp) / me
    return ratio.rolling(window).std()


@characteristic("ob")
def ob(funda: pd.DataFrame) -> pd.Series:
    """Rajgopal, Shevlin & Venkatachalam (2003): order backlog / assets."""
    backlog = funda.get("ob", 0).fillna(0)
    return backlog / _get(funda, "at")


@characteristic("dc_debt")
def dc_debt(funda: pd.DataFrame) -> pd.Series:
    """Valta (2016): convertible debt to total debt."""
    dc = funda.get("dc", 0).fillna(0)
    dltt = funda.get("dltt", 0).fillna(0)
    dlc = funda.get("dlc", 0).fillna(0)
    return dc / (dltt + dlc).replace(0, np.nan)


@characteristic("orgcap")
def orgcap(funda: pd.DataFrame) -> pd.Series:
    """Eisfeldt & Papanikolaou (2013): organizational capital / assets (simplified, no CPI).

    Capitalizes SG&A with a 15% depreciation rate over 5 years.
    """
    xsga = funda.get("xsga", 0).fillna(0)
    at_ = _get(funda, "at")
    oc = xsga + 0.85 * xsga.shift(1).fillna(0) + 0.72 * xsga.shift(2).fillna(0) + 0.61 * xsga.shift(3).fillna(0) + 0.52 * xsga.shift(4).fillna(0)
    return oc / at_


@characteristic("intangible_return")
def intangible_return(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Daniel & Titman (2006): intangible return (component of BM not explained by tangible assets)."""
    bm = _get(funda, "ceq") / me
    log_bm = np.log(bm.replace(0, np.nan))
    # Tangible BM growth approximation: log(BM_t) - log(BM_{t-5}) - cumulative ret component
    log_bm_5 = log_bm.shift(5)
    return log_bm - log_bm_5
