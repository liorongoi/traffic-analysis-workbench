"""Security checks: the Host header (DNS rebinding), the network access key, model names, worker checksums."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trafficwb import models
from trafficwb.web.app import _host_allowlist, access_key_for, create_app

H = {"X-TWB": "1"}


def _app(tmp_path, **listening):
    return create_app(tmp_path / "data", start_monitoring=False, listening=listening)


def test_host_check_on_this_computer_only():
    ok = _host_allowlist({"port": 8765})
    assert ok("127.0.0.1:8765") and ok("localhost:8765") and ok("[::1]:8765") and ok("localhost")
    assert not ok("evil.example:8765") and not ok("192.168.1.20:8765") and not ok("")


def test_host_check_on_the_network(monkeypatch):
    monkeypatch.setenv("TWB_ALLOWED_HOSTS", "lanepulse.corp, other")
    ok = _host_allowlist({"port": 8765, "lan": True})
    assert ok("192.168.1.20:8765") and ok("[fe80::1]:8765") and ok("lanepulse.corp:8765") and ok("OTHER")
    assert not ok("rebind.evil.example:8765")


def test_host_check_off_without_serve():
    assert _host_allowlist({}) is None


def test_rebinding_request_is_refused(tmp_path):
    c = TestClient(_app(tmp_path, port=8765), base_url="http://127.0.0.1:8765")
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/health", headers={"host": "rebind.evil.example:8765"}).status_code == 421


def test_no_key_on_this_computer(tmp_path):
    c = TestClient(_app(tmp_path, port=8765), base_url="http://127.0.0.1:8765")
    assert c.get("/api/access").json() == {"required": False, "ok": True, "generated": False}
    assert c.get("/api/missions").status_code == 200
    assert not (tmp_path / "data" / "access.json").exists()


def test_network_needs_the_access_key(tmp_path):
    c = TestClient(_app(tmp_path, port=8765, lan=True, discovery=False), base_url="http://10.0.0.5:8765")
    key = access_key_for(tmp_path / "data")
    if os.name != "nt":                                            # Windows has no such mode bits (ACLs instead)
        assert (tmp_path / "data" / "access.json").stat().st_mode & 0o077 == 0
    import json
    assert key not in json.loads((tmp_path / "data" / "access.json").read_text())["hash"]   # stored hashed
    assert c.get("/api/health").status_code == 200                 # installers and supervisors
    assert c.get("/").status_code in (200, 404)                    # the screens load, to ask for the key
    assert c.get("/api/missions").status_code == 401
    assert c.post("/api/workers/codes", headers=H).status_code == 401
    assert c.get("/api/folders", params={"path": "/"}).status_code == 401
    assert c.get("/api/access").json() == {"required": True, "ok": False, "generated": False}
    assert c.post("/api/access", json={"key": "wrong"}, headers=H).status_code == 403
    assert c.get("/api/missions").status_code == 401
    r = c.post("/api/access", json={"key": key}, headers=H)
    assert r.status_code == 200 and key not in r.headers["set-cookie"]
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=strict" in r.headers["set-cookie"].lower()
    assert c.get("/api/missions").status_code == 200
    # Scripts can send it as a bearer token.
    c2 = TestClient(c.app, base_url="http://10.0.0.5:8765")
    assert c2.get("/api/missions", headers={"Authorization": f"Bearer {key}"}).status_code == 200


def test_new_key_replaces_the_old(tmp_path):
    a = access_key_for(tmp_path)
    assert access_key_for(tmp_path) == a
    b = access_key_for(tmp_path, new=True)
    assert b != a and access_key_for(tmp_path) == b


def test_choose_change_and_reset_a_forgotten_password(tmp_path, monkeypatch, capsys):
    import io
    from trafficwb.web.app import main
    app = _app(tmp_path, port=8765, lan=True, discovery=False)
    c = TestClient(app, base_url="http://10.0.0.5:8765")
    first = access_key_for(tmp_path / "data")
    c.post("/api/access", json={"key": first}, headers=H)
    assert c.get("/api/access").json()["generated"] is True              # the app asks to choose one
    # Too short, or a wrong current one: nothing changes.
    assert c.post("/api/access/password", json={"current": first, "new": "short"}, headers=H).status_code == 400
    assert c.post("/api/access/password", json={"current": "nope", "new": "long enough"}, headers=H).status_code == 403
    r = c.post("/api/access/password", json={"current": first, "new": "my own password"}, headers=H)
    assert r.status_code == 200 and c.get("/api/missions").status_code == 200    # this browser stays signed in
    assert access_key_for(tmp_path / "data") is None                       # a chosen one is never shown
    other = TestClient(app, base_url="http://10.0.0.5:8765")
    assert other.post("/api/access", json={"key": first}, headers=H).status_code == 403
    assert other.post("/api/access", json={"key": "my own password"}, headers=H).status_code == 200
    # Forgotten: set a new one on the server; every browser must enter it.
    monkeypatch.setattr("sys.stdin", io.StringIO("brand new secret\n"))
    assert main(["password", "--data", str(tmp_path / "data"), "--stdin"]) == 0
    assert c.get("/api/missions").status_code == 401 and other.get("/api/missions").status_code == 401
    assert c.post("/api/access", json={"key": "brand new secret"}, headers=H).status_code == 200
    assert main(["password", "--data", str(tmp_path / "data"), "--show"]) == 0
    assert "cannot be shown" in capsys.readouterr().out


def test_an_older_access_key_still_opens(tmp_path):
    (tmp_path / "access-key").write_text("3f9a1c-07be42-aaaaaa-bbbbbb-cccccc\n")
    assert access_key_for(tmp_path) == "3f9a1c-07be42-aaaaaa-bbbbbb-cccccc"
    assert not (tmp_path / "access-key").exists()


@pytest.mark.parametrize("name", ["../x", "a/b", "a\\b", "\\\\host\\share\\pack", "C:pack", "/etc", "..", ""])
def test_model_names_are_never_paths(tmp_path, name):
    with pytest.raises(models.ModelPackError):
        models.find(name, root=tmp_path)


def test_worker_refuses_a_checksum_that_is_a_path(tmp_path):
    from trafficwb.workers import Worker
    w = Worker.__new__(Worker)
    w.cache = tmp_path / "cache"
    w.server, w.code = "http://127.0.0.1:9", "x"
    w.log = lambda *_: None
    with pytest.raises(RuntimeError, match="SHA-256"):
        w._pack({"name": "x", "version": "1", "sha256": "../../Library/LaunchAgents"})
    assert not (tmp_path / "Library").exists()


def test_lookup_indexes_are_used(tmp_path):
    """Results and the pipeline look up each image's run item and detections; those must not scan whole tables."""
    from trafficwb.store import DataRoot, MissionConfig
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir(), out.mkdir()
    root = DataRoot(tmp_path / "data")
    m = root.create_mission(MissionConfig(name="m", input_type="images", mode="general_traffic", country="IL",
                                          iana_zone="Asia/Jerusalem", source_dir=str(src), include_subdirs=True,
                                          processing="one_time", output_dir=str(out), export_formats=["csv"]), "t")
    m = root.open_mission(m.id)
    for q in ("SELECT * FROM run_item WHERE source_version_id=1 ORDER BY id DESC LIMIT 1",
              "SELECT * FROM detection WHERE run_item_id=1"):
        plan = " ".join(str(r[3]) for r in m.con.execute("EXPLAIN QUERY PLAN " + q))
        assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan
