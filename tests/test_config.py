import maroczy
from maroczy.broker.connection import _resolve_connection_args


def test_env_path_is_a_string():
    assert isinstance(maroczy.config.ENV_PATH, str)


def test_resolve_connection_args_defaults_without_env(monkeypatch):
    monkeypatch.delenv("IBKR_HOST", raising=False)
    monkeypatch.delenv("IBKR_PORT", raising=False)
    monkeypatch.delenv("IBKR_CLIENT_ID", raising=False)
    assert _resolve_connection_args(None, None, None) == ("127.0.0.1", 7497, 1)


def test_resolve_connection_args_reads_env(monkeypatch):
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "42")
    assert _resolve_connection_args(None, None, None) == ("10.0.0.5", 4002, 42)


def test_resolve_connection_args_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "42")
    assert _resolve_connection_args("192.168.1.1", 7496, 7) == ("192.168.1.1", 7496, 7)
