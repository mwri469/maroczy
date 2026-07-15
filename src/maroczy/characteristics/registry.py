"""CSV-driven characteristic registry.

Loads ``characteristics mapping.csv`` (a bundled copy lives at
``maroczy/data/characteristics_mapping.csv``) and exposes metadata lookup,
filtering by data class, and cross-referencing against the functions
actually implemented in :mod:`maroczy.characteristics.functions`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from maroczy.characteristics.functions import list_implemented

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "characteristics_mapping.csv"

__all__ = ["CharacteristicRegistry", "DEFAULT_CSV_PATH"]


class CharacteristicRegistry:
    """Metadata registry over the full characteristics mapping CSV.

    Parameters
    ----------
    path: path to a tab-separated characteristics mapping CSV with the same
        schema as ``characteristics mapping.csv`` (columns: description,
        author, year, journal, function, class, vtype, ghz, jkp, hxz,
        hxz_ref, cz, note, sample_start, sample_end). Defaults to the
        bundled copy.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CSV_PATH
        self.table = pd.read_csv(self.path, sep="\t", index_col=0, encoding="cp1252")
        self.table.index.name = "id"

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"CharacteristicRegistry({len(self)} characteristics, {self.table['function'].notna().sum()} named, {len(self.implemented())} implemented)"

    def names(self, data_class: str | None = None, implemented_only: bool = False) -> list[str]:
        """List distinct ``function`` names, optionally filtered by ``class`` and/or implementation status."""
        df = self.table
        if data_class is not None:
            df = df[df["class"] == data_class]
        names = sorted(df["function"].dropna().unique().tolist())
        if implemented_only:
            impl = set(list_implemented())
            names = [n for n in names if n in impl]
        return names

    def get(self, name: str) -> pd.Series:
        """Metadata row for a given characteristic (``function``) name."""
        rows = self.table[self.table["function"] == name]
        if rows.empty:
            raise KeyError(f"{name!r} not found in the characteristics mapping.")
        return rows.iloc[0]

    def implemented(self) -> pd.DataFrame:
        """Rows of the mapping table that have a registered Python implementation."""
        impl = set(list_implemented())
        return self.table[self.table["function"].isin(impl)]

    def missing(self) -> pd.DataFrame:
        """Rows with a named ``function`` that is *not* yet implemented -- your TODO list."""
        impl = set(list_implemented())
        named = self.table[self.table["function"].notna()]
        return named[~named["function"].isin(impl)]

    def search(self, query: str) -> pd.DataFrame:
        """Case-insensitive substring search over description/author/function."""
        q = query.lower()
        cols = ["description", "author", "function"]
        mask = False
        for col in cols:
            mask = mask | self.table[col].astype(str).str.lower().str.contains(q, na=False)
        return self.table[mask]

    def coverage(self) -> dict:
        """Summary of how many named characteristics currently have an implementation."""
        total_named = int(self.table["function"].notna().sum())
        n_impl = len(self.implemented())
        return {
            "total_characteristics": len(self),
            "total_named": total_named,
            "implemented": n_impl,
            "pct_implemented": (n_impl / total_named) if total_named else 0.0,
        }
