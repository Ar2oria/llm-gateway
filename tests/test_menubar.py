import json
import os
from pathlib import Path

from gateway.menubar import (
    admin_url_for_config,
    gateway_alive,
    gateway_pid,
    restart_command,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_admin_url_default_port(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k"}, "endpoints": []}))
    assert admin_url_for_config(p) == "http://127.0.0.1:8080/admin"


def test_admin_url_custom_port(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"server": {"api_key": "k", "port": 8088}}))
    assert admin_url_for_config(p) == "http://127.0.0.1:8088/admin"


def test_admin_url_missing_file_falls_back(tmp_path):
    assert admin_url_for_config(tmp_path / "nope.json") == "http://127.0.0.1:8080/admin"


def test_gateway_alive_false_when_pid_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.menubar.PID_FILE", tmp_path / "missing.pid")
    assert gateway_alive() is False


def test_gateway_pid_and_alive_with_live_process(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr("gateway.menubar.PID_FILE", pid_file)
    assert gateway_pid() == os.getpid()
    assert gateway_alive() is True


def test_gateway_alive_false_for_dead_pid(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    pid_file.write_text("99999999")
    monkeypatch.setattr("gateway.menubar.PID_FILE", pid_file)
    assert gateway_alive() is False


def test_gateway_pid_ignores_garbage(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    pid_file.write_text("not-a-pid")
    monkeypatch.setattr("gateway.menubar.PID_FILE", pid_file)
    assert gateway_pid() is None
    assert gateway_alive() is False


# ---------------------------------------------------------------------------
# restart_command / restart_gateway
# ---------------------------------------------------------------------------

def test_restart_command_shape():
    """restart_command builds a gateway.sh restart argv with an absolute
    config path (gateway.sh resolves relative configs against its caller's
    cwd, which is not the menubar's cwd)."""
    argv = restart_command("config.json")
    assert argv[0] == "bash"
    assert argv[1] == str(REPO_ROOT / "gateway.sh")
    assert Path(argv[1]).is_file()
    assert argv[2] == "restart"
    assert Path(argv[3]).is_absolute()
    assert argv[3].endswith("config.json")


def test_restart_command_passes_absolute_config_through(tmp_path):
    config = tmp_path / "some-config.json"
    config.write_text("{}", encoding="utf-8")
    argv = restart_command(config)
    assert argv[3] == str(config)
