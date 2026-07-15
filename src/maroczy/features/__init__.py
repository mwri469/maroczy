"""Research-paper-derived alpha features.

- :mod:`maroczy.features.bubble` -- Phillips, Shi & Yu (2015) GSADF/BSADF bubble detection.
- :mod:`maroczy.features.covariance` -- covariance matrix cleaning (Bongiorno & Challet style).
- :mod:`maroczy.features.cpcv` -- Combinatorial Purged Cross-Validation & PBO.
- :mod:`maroczy.features.rough_vol` -- rough log-normal volatility (Hager, Horst, Wagenhofer & Xu, 2026).
"""

from maroczy.features import bubble, covariance, cpcv, rough_vol

__all__ = ["bubble", "covariance", "cpcv", "rough_vol"]
