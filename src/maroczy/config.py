"""Per-user configuration via a local ``.env`` file (``python-dotenv``).

Importing :mod:`maroczy` (which every submodule triggers, per Python's
import system) loads a ``.env`` file -- found by searching upward from the
current working directory -- into ``os.environ`` exactly once per process.
This is how per-user secrets like ``LSE_API_KEY`` (and optional IBKR
connection overrides) are supplied without hardcoding them or exporting
them by hand every session; each teammate keeps their own ``.env`` (listed
in ``.gitignore``, never committed) based on ``.env.example``.

Precedence matches ``python-dotenv``'s default: real environment variables
that are already set win over ``.env`` file values (``override=False``),
so CI/production environments always take priority over a stray local file.
"""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

#: Path to the ``.env`` file that was loaded, or ``""`` if none was found.
ENV_PATH = find_dotenv(usecwd=True)

load_dotenv(ENV_PATH, override=False)

__all__ = ["ENV_PATH"]
