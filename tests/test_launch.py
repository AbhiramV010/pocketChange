import socket

import pytest

import launch


def test_free_port_skips_busy_ports():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert launch.free_port(port) != port


def test_env_file_created_from_example(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("DRY_RUN=true\n")
    launch.ensure_env_file()
    assert (tmp_path / ".env").read_text() == "DRY_RUN=true\n"
    (tmp_path / ".env").write_text("DRY_RUN=false\n")
    launch.ensure_env_file()  # never overwrites an existing .env
    assert (tmp_path / ".env").read_text() == "DRY_RUN=false\n"


def test_restart_budget_is_limited():
    svc = launch.Service("bot", ["x"])
    assert all(svc.may_restart() for _ in range(launch.MAX_RESTARTS))
    assert not svc.may_restart()


class FakeService(launch.Service):
    def __init__(self, name, codes):
        super().__init__(name, ["x"])
        self.codes, self.starts = list(codes), 0

    def exit_code(self):
        return self.codes[0] if self.codes else None

    def start(self):
        self.starts += 1
        self.codes.pop(0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)


def test_supervisor_restarts_a_crash():
    bot = FakeService("bot", [1])
    calls = {"n": 0}

    def ui_exit():
        calls["n"] += 1
        return 0 if calls["n"] > 3 else None  # stop the loop after the restart happened

    ui = FakeService("dashboard", [])
    ui.exit_code = ui_exit
    assert launch.supervise([bot, ui]) == 0
    assert bot.starts == 1


def test_supervisor_never_restarts_a_second_bot():
    bot = FakeService("bot", [launch.ALREADY_RUNNING_EXIT])
    assert launch.supervise([bot]) == 1 and bot.starts == 0


def test_supervisor_treats_clean_exit_as_shutdown():
    bot = FakeService("bot", [0])
    assert launch.supervise([bot]) == 0 and bot.starts == 0


def test_supervisor_gives_up_on_crash_loop():
    bot = FakeService("bot", [1] * (launch.MAX_RESTARTS + 1))
    assert launch.supervise([bot]) == 1 and bot.starts == launch.MAX_RESTARTS


def test_exit_code_constant_matches_bot():
    from bot.run import ALREADY_RUNNING_EXIT

    assert launch.ALREADY_RUNNING_EXIT == ALREADY_RUNNING_EXIT


def test_bot_lock_is_exclusive(tmp_path):
    from bot.run import acquire_instance_lock
    import subprocess, sys

    lock = tmp_path / "x.lock"
    held = acquire_instance_lock(lock)
    code = (
        "import sys; sys.path.insert(0, '.'); from pathlib import Path; from bot.run import acquire_instance_lock; "
        f"sys.exit(0 if acquire_instance_lock(Path(r'{lock}')) is None else 1)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=launch.ROOT).returncode == 0
    held.close()
