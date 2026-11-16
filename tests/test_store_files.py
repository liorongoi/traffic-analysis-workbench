"""Open files: the server raises its open-file limit, and a database that cannot be opened for want of free files is
tried again once the connections no longer in use are freed."""
import sqlite3

from trafficwb import store


def test_a_database_is_opened_again_after_freeing_unused_connections(tmp_path, monkeypatch):
    real, calls = store._open, []

    def flaky(path):
        calls.append(path)
        if len(calls) == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return real(path)
    monkeypatch.setattr(store, "_open", flaky)
    con = store.connect(tmp_path / "x.db")
    assert len(calls) == 2 and con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_the_server_raises_its_open_file_limit():
    resource = __import__("pytest").importorskip("resource")
    from trafficwb.web.app import _more_open_files
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, soft), hard))
        _more_open_files(1024)
        want = 1024 if hard == resource.RLIM_INFINITY else min(1024, hard)
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == want
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
