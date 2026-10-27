"""Administration: backups, restore, verified updates and explicit mission deletion (P9, Q22, Q28).

- A backup is a folder with a consistent copy of control.db and every mission
  database (SQLite online backup, safe while the server runs), a manifest with
  checksums and schema versions, and optionally the evidence folders. It is
  written as ``.partial`` and renamed when complete; an existing backup is never
  overwritten.
- Restore only writes into an empty or new data folder, after verifying the
  backup. This is also the rollback after an update: install the previous
  application version and restore the backup taken before the update into a new
  data folder.
- Deleting a mission is explicit: the operator types the mission's name. The
  mission's databases and evidence are removed; source images and exported files
  are never touched. The deletion is written to the audit log with what was removed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .store import DataRoot, connect, utcnow


class AdminError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_db(src: Path, dest: Path) -> None:
    s = sqlite3.connect(src)
    d = sqlite3.connect(dest)
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")
    finally:
        d.close()
        s.close()


def _schema(path: Path) -> int | None:
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    except sqlite3.Error:
        return None
    finally:
        con.close()


def audit(root: DataRoot, author: str, action: str, **detail) -> None:
    root.con.execute("INSERT INTO audit_log(at,author,action,detail) VALUES (?,?,?,?)",
                     (utcnow(), author, action, json.dumps(detail, ensure_ascii=False)))


def audit_log(root: DataRoot, limit: int = 100) -> list[dict]:
    return [dict(r) | {"detail": json.loads(r["detail"])}
            for r in root.con.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]


def backup(data_root: Path | str, dest_dir: Path | str, author: str = "system", with_evidence: bool = False) -> Path:
    root = DataRoot(data_root)
    dest_dir = Path(dest_dir)
    if dest_dir.resolve().is_relative_to(root.path.resolve()):
        raise AdminError("Put backups outside the data folder, ideally on another drive.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final, n = dest_dir / f"twb-backup-{stamp}", 2
    while final.exists() or final.with_name(final.name + ".partial").exists():
        final, n = dest_dir / f"twb-backup-{stamp}-{n}", n + 1
    tmp = final.with_name(final.name + ".partial")
    try:
        (tmp / "missions").mkdir(parents=True)
        files = {}
        _copy_db(root.path / "control.db", tmp / "control.db")
        files["control.db"] = {"sha256": _sha(tmp / "control.db"), "schema": _schema(tmp / "control.db")}
        missions = []
        for m in root.missions():
            src = root.path / m["db_path"]
            if not src.is_file():
                missions.append({"id": m["id"], "name": m["name"], "missing": True})
                continue
            rel = Path("missions") / m["id"] / "mission.db"
            (tmp / rel).parent.mkdir(parents=True)
            _copy_db(src, tmp / rel)
            files[rel.as_posix()] = {"sha256": _sha(tmp / rel), "schema": _schema(tmp / rel)}
            if with_evidence and (src.parent / "evidence").is_dir():
                shutil.copytree(src.parent / "evidence", tmp / rel.parent / "evidence")
            missions.append({"id": m["id"], "name": m["name"]})
        manifest = {"app_version": __version__, "created_at": utcnow(), "source": str(root.path.resolve()),
                    "with_evidence": with_evidence, "missions": missions, "files": files}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(final)
    except Exception as exc:
        audit(root, author, "backup_failed", dest=str(dest_dir), error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    audit(root, author, "backup", folder=str(final), missions=len(missions), with_evidence=with_evidence)
    return final


def verify(backup_dir: Path | str) -> list[str]:
    """Problems found in a backup; an empty list means it is complete and readable."""
    b = Path(backup_dir)
    mf = b / "manifest.json"
    if not mf.is_file():
        return [f"{b} has no manifest.json; it is not a complete backup."]
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    problems = []
    for rel, meta in manifest["files"].items():
        p = b / rel
        if not p.is_file():
            problems.append(f"{rel} is missing")
            continue
        if _sha(p) != meta["sha256"]:
            problems.append(f"{rel} does not match its checksum")
            continue
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            con.close()
        if ok != "ok":
            problems.append(f"{rel} failed the SQLite integrity check: {ok}")
    return problems


def restore(backup_dir: Path | str, data_root: Path | str, author: str = "system") -> dict:
    b, target = Path(backup_dir), Path(data_root)
    problems = verify(b)
    if problems:
        raise AdminError("The backup did not verify; nothing was restored: " + "; ".join(problems))
    if target.exists() and any(target.iterdir()):
        raise AdminError(f"{target} is not empty. Restore into a new or empty folder; nothing was overwritten.")
    manifest = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(b / "control.db", target / "control.db")
    for rel in manifest["files"]:
        if rel == "control.db":
            continue
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(b / rel, target / rel)
        ev = b / Path(rel).parent / "evidence"
        if ev.is_dir():
            shutil.copytree(ev, target / Path(rel).parent / "evidence")
    root = DataRoot(target)
    audit(root, author, "restore", backup=str(b), backup_created_at=manifest["created_at"],
          backup_app_version=manifest["app_version"])
    return manifest


def delete_mission(data_root: Path | str, mission_id: str, author: str, confirm_name: str) -> dict:
    root = DataRoot(data_root)
    row = root.con.execute("SELECT * FROM mission WHERE id=?", (mission_id,)).fetchone()
    if row is None:
        raise AdminError("That mission does not exist.")
    if confirm_name.strip() != row["name"]:
        raise AdminError("Type the mission's name exactly to delete it.")
    mdir = root.path / "missions" / mission_id
    db = root.path / row["db_path"]
    counts = {}
    if db.is_file():
        con = connect(db)
        try:
            for t in ("source_version", "detection", "annotation_revision", "plate_read", "video_event", "training_dataset"):
                try:
                    counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except sqlite3.Error:
                    pass
            cfg = con.execute("SELECT payload FROM config_revision WHERE kind='mission' ORDER BY seq DESC LIMIT 1").fetchone()
        finally:
            con.close()
    else:
        cfg = None
    payload = json.loads(cfg[0]) if cfg else {}
    detail = {"mission_id": mission_id, "name": row["name"], "input_type": row["input_type"], "mode": row["mode"],
              "created_at": row["created_at"], "source_dir": payload.get("source_dir"), "output_dir": payload.get("output_dir"),
              "rows": counts, "kept": "source images and exported files were not touched"}
    root.con.execute("BEGIN")
    try:
        root.con.execute("INSERT INTO audit_log(at,author,action,detail) VALUES (?,?,?,?)",
                         (utcnow(), author, "mission_deleted", json.dumps(detail, ensure_ascii=False)))
        root.con.execute("DELETE FROM mission WHERE id=?", (mission_id,))
        root.con.execute("COMMIT")
    except Exception:
        root.con.execute("ROLLBACK")
        raise
    if mdir.is_dir():
        import gc
        gc.collect()                  # close database handles left by earlier requests (Windows locks open files)
        try:
            shutil.rmtree(mdir)
        except OSError as exc:
            detail["folder_left"] = f"{mdir} could not be removed ({exc}); delete it by hand after closing the server"
            audit(root, author, "mission_folder_left", mission_id=mission_id, folder=str(mdir), error=str(exc)[:300])
    return detail


def _accounts_cmd(args, who: str) -> int:
    """List the accounts, and for reset-password ask for the account and a new password (typed twice)."""
    import getpass

    from .store import connect
    from .web.auth import Auth, AuthError
    db = Path(args.data) / "control.db"
    if not db.exists():
        print(f"No Lanemark data in {args.data} (control.db not found).")
        return 2
    con = connect(db)
    try:
        auth = Auth(con)
        accts = auth.accounts()
        if not accts:
            print("There are no accounts yet. Open Lanemark in the browser to create the first one.")
            return 0
        print("Accounts:")
        for a in accts:
            print(f"  {a['username']:<20} {a['display_name']:<24} {a['role']}{'  (disabled)' if a['disabled'] else ''}")
        if args.cmd == "accounts":
            return 0
        user = args.user or input("\nUser name to reset: ").strip()
        if not any(a["username"].lower() == user.lower() for a in accts):
            print(f"There is no account named {user}.")
            return 2
        for _ in range(3):
            p1 = getpass.getpass("New password (at least 8 characters): ")
            if p1 != getpass.getpass("Type it again: "):
                print("The two passwords differ; try again.")
                continue
            try:
                u = auth.reset_password(user, p1, by=who)
            except AuthError as exc:
                print(exc.message)
                if exc.code == "no_account":
                    return 2
                continue
            audit(DataRoot(args.data), who, "password_reset", username=u.username)
            print(f"Password changed for {u.username}. Sign in with it in the browser.")
            return 0
        return 2
    finally:
        con.close()


def main(argv=None) -> int:
    from trafficwb import safe_console
    safe_console()
    import argparse
    import getpass

    ap = argparse.ArgumentParser(prog="trafficwb-admin", description="Backups, restore and mission deletion.")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("backup", help="back up every database to a folder")
    b.add_argument("--data", required=True)
    b.add_argument("--to", required=True)
    b.add_argument("--with-evidence", action="store_true", help="also copy crops, clips and other evidence")
    v = sub.add_parser("verify", help="check a backup's checksums and database integrity")
    v.add_argument("backup")
    r = sub.add_parser("restore", help="restore a backup into a new or empty data folder")
    r.add_argument("backup")
    r.add_argument("--data", required=True)
    d = sub.add_parser("delete-mission", help="delete a mission's databases and evidence (not its source images)")
    d.add_argument("--data", required=True)
    d.add_argument("--mission", required=True)
    a = sub.add_parser("audit", help="show the audit log")
    a.add_argument("--data", required=True)
    ac = sub.add_parser("accounts", help="list the sign-in accounts")
    ac.add_argument("--data", required=True)
    rp = sub.add_parser("reset-password", help="set a new password for an account (lists the accounts first)")
    rp.add_argument("--data", required=True)
    rp.add_argument("--user", help="the account's user name; asked for when left out")
    args = ap.parse_args(argv)
    who = getpass.getuser()
    try:
        if args.cmd == "backup":
            print(backup(args.data, args.to, who, args.with_evidence))
        elif args.cmd == "verify":
            problems = verify(args.backup)
            print("\n".join(problems) or "backup verified")
            return 1 if problems else 0
        elif args.cmd == "restore":
            m = restore(args.backup, args.data, who)
            print(f"restored {len(m['missions'])} mission(s) from a backup made {m['created_at']} "
                  f"by version {m['app_version']}")
        elif args.cmd == "delete-mission":
            name = input("Type the mission's name to delete it: ")
            print(json.dumps(delete_mission(args.data, args.mission, who, name), indent=2, ensure_ascii=False))
        elif args.cmd in ("accounts", "reset-password"):
            return _accounts_cmd(args, who)
        elif args.cmd == "audit":
            for e in audit_log(DataRoot(args.data)):
                print(e["at"], e["author"], e["action"], json.dumps(e["detail"], ensure_ascii=False))
        else:
            ap.print_help()
            return 1
    except AdminError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------------ nightly backup in the server

def backup_due(now_local: datetime, last_local: datetime | None, at: str) -> bool:
    """True once a day, at or after the HH:MM time, when no backup was made since that time today."""
    hh, mm = (int(x) for x in at.split(":"))
    slot = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return now_local >= slot and (last_local is None or last_local < slot)


class BackupScheduler:
    def __init__(self, data_root: Path | str, dest_dir: Path | str, at: str = "02:00", check_seconds: float = 60):
        import threading
        backup_due(datetime.now(), None, at)          # rejects a malformed time at start
        self.data_root, self.dest, self.at = Path(data_root), Path(dest_dir), at
        self.check = check_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="nightly-backup", daemon=True)
        self.last_error: str | None = None

    def last(self) -> datetime | None:
        r = DataRoot(self.data_root).con.execute(
            "SELECT at FROM audit_log WHERE action='backup' AND json_extract(detail,'$.scheduled')=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return datetime.fromisoformat(r[0].replace("Z", "+00:00")).astimezone().replace(tzinfo=None) if r else None

    def run_if_due(self, now_local: datetime | None = None) -> Path | None:
        if not backup_due(now_local or datetime.now(), self.last(), self.at):
            return None
        try:
            p = backup(self.data_root, self.dest, "scheduled")
            root = DataRoot(self.data_root)
            root.con.execute("UPDATE audit_log SET detail=json_set(detail,'$.scheduled',1) WHERE id="
                             "(SELECT MAX(id) FROM audit_log WHERE action='backup')")
            self.last_error = None
            return p
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _loop(self):
        while not self._stop.wait(self.check):
            self.run_if_due()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
