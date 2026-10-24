"""Named local accounts and sessions (ARC-03, DER-01, ENV-03).

- Every operator has an individual account, because corrections record who made them.
- Passwords are stored as scrypt hashes with a per-account salt.
- Sessions are random tokens in an HttpOnly, SameSite=Strict cookie; only a hash is stored.
- One active operator at a time: signing in while another account has an active session is
  refused unless the operator chooses to take over, which ends the other session and is recorded.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..store import utcnow

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
  id INTEGER PRIMARY KEY,
  username TEXT NOT NULL UNIQUE COLLATE NOCASE,
  display_name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'operator',      -- admin | operator
  disabled INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session (
  token_hash TEXT PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES account(id),
  created_at TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  client TEXT,
  ended_at TEXT,
  end_reason TEXT
);
CREATE TABLE IF NOT EXISTS auth_event (
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  username TEXT,
  kind TEXT NOT NULL,                         -- login | login_failed | logout | takeover | expired | account_created
  client TEXT,
  detail TEXT
);
"""

IDLE_MINUTES = 30
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


class AuthError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class User:
    id: int
    username: str
    display_name: str
    role: str


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)
    return f"scrypt${salt.hex()}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, dk = stored.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, bytes.fromhex(salt)).split("$")[2], dk)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class Auth:
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        con.executescript(AUTH_SCHEMA)

    def _event(self, kind: str, username: str | None, client: str | None = None, detail: str | None = None) -> None:
        self.con.execute("INSERT INTO auth_event(at,username,kind,client,detail) VALUES (?,?,?,?,?)",
                         (utcnow(), username, kind, client, detail))

    def has_accounts(self) -> bool:
        return self.con.execute("SELECT COUNT(*) FROM account").fetchone()[0] > 0

    def create_account(self, username: str, display_name: str, password: str, role: str = "operator") -> User:
        username = username.strip()
        if not username or not username.replace(".", "").replace("-", "").replace("_", "").isalnum():
            raise AuthError("bad_username", "Use letters, digits, dot, dash or underscore for the user name.")
        if len(password) < 8:
            raise AuthError("weak_password", "The password needs at least 8 characters.")
        if role not in ("admin", "operator"):
            raise AuthError("bad_role", "Role must be admin or operator.")
        try:
            cur = self.con.execute(
                "INSERT INTO account(username,display_name,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                (username, display_name.strip() or username, hash_password(password), role, utcnow()))
        except sqlite3.IntegrityError:
            raise AuthError("exists", f"An account named {username} already exists.") from None
        self._event("account_created", username)
        return User(cur.lastrowid, username, display_name.strip() or username, role)

    def _active_sessions(self):
        cutoff = (_now() - timedelta(minutes=IDLE_MINUTES)).isoformat(timespec="seconds").replace("+00:00", "Z")
        return self.con.execute(
            "SELECT s.*, a.username FROM session s JOIN account a ON a.id=s.account_id "
            "WHERE s.ended_at IS NULL AND s.last_seen >= ?", (cutoff,)).fetchall()

    def login(self, username: str, password: str, client: str | None = None, take_over: bool = False) -> tuple[str, User]:
        row = self.con.execute("SELECT * FROM account WHERE username=?", (username.strip(),)).fetchone()
        if row is None or row["disabled"] or not check_password(password, row["password_hash"]):
            self._event("login_failed", username, client)
            raise AuthError("bad_credentials", "User name or password is wrong.")
        others = [s for s in self._active_sessions() if s["account_id"] != row["id"]]
        if others and not take_over:
            raise AuthError("operator_active",
                            f"{others[0]['username']} is signed in. Only one operator can work at a time. "
                            "Sign in again with 'take over' to end their session.")
        for s in others:
            self.con.execute("UPDATE session SET ended_at=?, end_reason='taken_over' WHERE token_hash=?",
                             (utcnow(), s["token_hash"]))
            self._event("takeover", row["username"], client, f"ended session of {s['username']}")
        token = secrets.token_urlsafe(32)
        self.con.execute("INSERT INTO session(token_hash,account_id,created_at,last_seen,client) VALUES (?,?,?,?,?)",
                         (hashlib.sha256(token.encode()).hexdigest(), row["id"], utcnow(), utcnow(), client))
        self._event("login", row["username"], client)
        return token, User(row["id"], row["username"], row["display_name"], row["role"])

    def user_for(self, token: str | None) -> User:
        if not token:
            raise AuthError("not_signed_in", "Sign in to continue.")
        th = hashlib.sha256(token.encode()).hexdigest()
        row = self.con.execute(
            "SELECT s.*, a.username, a.display_name, a.role, a.disabled FROM session s JOIN account a ON a.id=s.account_id "
            "WHERE s.token_hash=?", (th,)).fetchone()
        if row is None or row["ended_at"] is not None or row["disabled"]:
            reason = row["end_reason"] if row is not None else None
            msg = "Another operator took over this workstation." if reason == "taken_over" else "Sign in to continue."
            raise AuthError("not_signed_in", msg)
        if _now() - _parse(row["last_seen"]) > timedelta(minutes=IDLE_MINUTES):
            self.con.execute("UPDATE session SET ended_at=?, end_reason='expired' WHERE token_hash=?", (utcnow(), th))
            self._event("expired", row["username"])
            raise AuthError("not_signed_in", f"Signed out after {IDLE_MINUTES} minutes without activity.")
        self.con.execute("UPDATE session SET last_seen=? WHERE token_hash=?", (utcnow(), th))
        return User(row["account_id"], row["username"], row["display_name"], row["role"])

    def logout(self, token: str) -> None:
        th = hashlib.sha256(token.encode()).hexdigest()
        row = self.con.execute("SELECT a.username FROM session s JOIN account a ON a.id=s.account_id WHERE token_hash=?",
                               (th,)).fetchone()
        self.con.execute("UPDATE session SET ended_at=?, end_reason='logout' WHERE token_hash=? AND ended_at IS NULL",
                         (utcnow(), th))
        if row:
            self._event("logout", row["username"])
