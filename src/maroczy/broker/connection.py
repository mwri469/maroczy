"""IBKR session management built on top of ``ib_insync``.

The :class:`Broker` class wraps a single :class:`ib_insync.IB` connection and
is designed to be created once per notebook/process (it behaves like a
singleton keyed by host/port/client_id) so repeated ``Broker()`` calls in a
notebook reuse the same live session instead of erroring on a duplicate
client id.
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from ib_insync import IB, util

logger = logging.getLogger("maroczy.broker")

DEFAULT_PAPER_PORT = 7497
DEFAULT_LIVE_PORT = 7496
DEFAULT_GATEWAY_PAPER_PORT = 4002
DEFAULT_GATEWAY_LIVE_PORT = 4001


def _resolve_connection_args(host: str | None, port: int | None, client_id: int | None) -> tuple[str, int, int]:
    """Fill in unset connection args from ``IBKR_HOST``/``IBKR_PORT``/``IBKR_CLIENT_ID``
    (loaded from a per-user ``.env`` via :mod:`maroczy.config`), falling back to the
    usual paper-trading defaults. Used identically by ``__new__`` and ``__init__`` so
    the singleton cache key and the instance state always agree.
    """
    if host is None:
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
    if port is None:
        port = int(os.environ.get("IBKR_PORT", DEFAULT_PAPER_PORT))
    if client_id is None:
        client_id = int(os.environ.get("IBKR_CLIENT_ID", 1))
    return host, port, client_id


class Broker:
    """Thin, notebook-friendly wrapper around a live IBKR connection.

    Parameters
    ----------
    host:
        Host running TWS / IB Gateway. Defaults to the ``IBKR_HOST`` env var
        (see :mod:`maroczy.config` re: per-user ``.env`` files) or localhost.
    port:
        API port. Defaults to ``IBKR_PORT`` or the paper-trading TWS port
        (7497). Use ``DEFAULT_LIVE_PORT`` / ``DEFAULT_GATEWAY_*`` for other setups.
    client_id:
        Arbitrary integer identifying this API client. Each concurrent
        connection needs a distinct id; defaults to ``IBKR_CLIENT_ID`` or
        ``1`` -- set a per-user ``.env`` value when several people share one
        machine/gateway to avoid client-id collisions.
    readonly:
        If True, the API connection rejects order placement (safety net).

    Examples
    --------
    >>> broker = Broker()                # doctest: +SKIP
    >>> broker.connect()                 # doctest: +SKIP
    >>> bars = broker.history("AAPL")    # doctest: +SKIP
    """

    _instances: ClassVar[dict[tuple[str, int, int], "Broker"]] = {}

    def __new__(
        cls,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        readonly: bool = False,
    ) -> "Broker":
        host, port, client_id = _resolve_connection_args(host, port, client_id)
        key = (host, port, client_id)
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        readonly: bool = False,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        host, port, client_id = _resolve_connection_args(host, port, client_id)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.readonly = readonly
        self.ib = IB()
        self._initialized = True

    # -- lifecycle ---------------------------------------------------
    def connect(self, timeout: float = 10.0) -> "Broker":
        """Connect to a running TWS/IB Gateway session (idempotent)."""
        if self.ib.isConnected():
            logger.info("Already connected to %s:%s (clientId=%s)", self.host, self.port, self.client_id)
            return self
        util.startLoop()  # allows ib_insync to run inside Jupyter's event loop
        self.ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            readonly=self.readonly,
            timeout=timeout,
        )
        logger.info("Connected to IBKR at %s:%s (clientId=%s)", self.host, self.port, self.client_id)
        return self

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    def __enter__(self) -> "Broker":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        state = "connected" if self.connected else "disconnected"
        return f"Broker(host={self.host!r}, port={self.port}, client_id={self.client_id}, {state})"
