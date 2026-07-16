"""Research-paper-derived alpha features.

- :mod:`maroczy.features.bubble` -- Phillips, Shi & Yu (2015) GSADF/BSADF bubble detection.
- :mod:`maroczy.features.covariance` -- covariance matrix cleaning (Bongiorno & Challet style).
- :mod:`maroczy.features.cpcv` -- Combinatorial Purged Cross-Validation & PBO.
- :mod:`maroczy.features.rough_vol` -- rough log-normal volatility (Hager, Horst, Wagenhofer & Xu, 2026).
- :mod:`maroczy.features.options` -- IV surface fitting, risk-neutral densities, term structure.
- :mod:`maroczy.features.macro` -- yield curve PCA, economic surprise, recession probability.
- :mod:`maroczy.features.positioning` -- COT analysis, futures basis/carry.
"""

from maroczy.features import bubble, covariance, cpcv, macro, options, positioning, rough_vol

__all__ = ["bubble", "covariance", "cpcv", "macro", "options", "positioning", "rough_vol"]
