"""The password a server shared on the network (--lan) asks for.

Kept in <data>/access.json, readable by the server's account only:
  - hash, salt, iterations: PBKDF2-SHA256 of the password (never the password itself);
  - secret: random, signs the browser's cookie; replaced with the password, so changing it signs every browser out;
  - generated: the password made at install, kept only until someone chooses their own, so `lanepulse password
    --show` can print it for the first sign-in.
Changing it: in the app (knowing the current one), or on the server with `sudo lanepulse password` (forgotten).
An older install's <data>/access-key (a generated key in plain text) becomes the generated password.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from pathlib import Path

ITERATIONS = 310_000
MIN_LENGTH = 8


def generate() -> str:
    """Easy to read out and type: 3f9a1c-07be42-…"""
    return "-".join(secrets.token_hex(3) for _ in range(5))


def _hash(password: str, salt: bytes, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations).hex()


class AccessStore:
    def __init__(self, data_root: Path | str):
        self.path = Path(data_root) / "access.json"
        self.legacy = Path(data_root) / "access-key"
        self._lock = threading.Lock()
        self._cache: tuple[float, dict] | None = None

    # ---------- file ----------
    def _write(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name("access.json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)
        self._cache = None

    def _read(self) -> dict:
        """The stored settings, made at first use (a generated password), re-read when the file changes."""
        with self._lock:
            try:
                m = self.path.stat().st_mtime
                if self._cache and self._cache[0] == m:
                    return self._cache[1]
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self._cache = (m, d)
                return d
            except FileNotFoundError:
                pass
            old = self.legacy.read_text(encoding="utf-8").strip() if self.legacy.is_file() else ""
            d = self._record(old or generate(), generated=True)
            self._write(d)
            if old:
                self.legacy.unlink(missing_ok=True)
            return d

    @staticmethod
    def _record(password: str, generated: bool) -> dict:
        salt = secrets.token_bytes(16)
        return {"hash": _hash(password, salt, ITERATIONS), "salt": salt.hex(), "iterations": ITERATIONS,
                "secret": secrets.token_hex(32), "generated": password if generated else None}

    # ---------- use ----------
    def check(self, password: str) -> bool:
        d = self._read()
        return hmac.compare_digest(_hash(password.strip(), bytes.fromhex(d["salt"]), d["iterations"]), d["hash"])

    def token(self) -> str:
        """The browser's cookie: signed with the secret, so it is never the password and ends when that changes."""
        d = self._read()
        return hmac.new(bytes.fromhex(d["secret"]), b"lanepulse-access", hashlib.sha256).hexdigest()

    def token_ok(self, value: str) -> bool:
        return bool(value) and hmac.compare_digest(value, self.token())

    def generated(self) -> str | None:
        return self._read().get("generated")

    def set(self, password: str) -> None:
        password = password.strip()
        if len(password) < MIN_LENGTH:
            raise ValueError(f"The password needs at least {MIN_LENGTH} characters.")
        self._write(self._record(password, generated=False))

    def reset_generated(self) -> str:
        p = generate()
        self._write(self._record(p, generated=True))
        return p


def main(argv=None) -> int:
    """trafficwb-web password: set (asks twice), --show the generated one, or --generate a new one."""
    import argparse
    import getpass
    import sys
    ap = argparse.ArgumentParser(prog="trafficwb-web password")
    ap.add_argument("--data", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--show", action="store_true", help="print the password made at install (until one is chosen)")
    g.add_argument("--generate", action="store_true", help="replace the password with a new generated one")
    g.add_argument("--stdin", action="store_true", help="read the new password from standard input")
    a = ap.parse_args(argv)
    st = AccessStore(a.data)
    if a.show:
        p = st.generated()
        print(p if p else "A password was chosen in the app or with `lanepulse password`, so it cannot be shown. "
                          "If it is forgotten, set a new one: sudo lanepulse password")
        return 0
    if a.generate:
        print(st.reset_generated())
        return 0
    if a.stdin:
        new = sys.stdin.readline().rstrip("\n")
    else:
        new = getpass.getpass("New LanePulse password: ")
        if getpass.getpass("Type it again: ") != new:
            print("The two do not match; nothing was changed.")
            return 1
    try:
        st.set(new)
    except ValueError as exc:
        print(exc)
        return 1
    print("Password changed. Browsers that were signed in must enter the new one.")
    return 0
