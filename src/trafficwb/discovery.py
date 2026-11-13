"""Finding a LanePulse server on the local network, so a helping computer needs no address typed (phase 5).

A server started with network access (``--lan``) answers on UDP port 8766. A worker with no address sends one
small question to the whole network segment and lists the servers that answer: their id, name, address and port.
The answer says only how to reach the server; helping still needs the operator to allow the computer (or a pairing
code). Across separate networks or a VPN, broadcasts do not arrive: the address is typed there instead.
"""
from __future__ import annotations

import json
import socket
import threading
import time

PORT = 8766
ASK = b"LANEPULSE-FIND-1"


class Responder:
    """Answers workers looking for a server. Runs in a background thread; a port in use only means no answers."""

    def __init__(self, http_port: int, server_id: str, version: str, port: int = PORT):
        self.http_port, self.server_id, self.version, self.port = http_port, server_id, version, port
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self.error: str | None = None

    def start(self) -> "Responder":
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sk.bind(("", self.port))
            sk.settimeout(0.5)
            self._sock = sk
        except OSError as exc:
            self.error = f"cannot answer on UDP port {self.port}: {exc}"
            return self
        threading.Thread(target=self._serve, name="lanepulse-discovery", daemon=True).start()
        return self

    def _serve(self) -> None:
        answer = json.dumps({"lanepulse": 1, "id": self.server_id, "name": socket.gethostname(),
                             "port": self.http_port, "version": self.version}).encode()
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            if data.strip() == ASK:
                try:
                    self._sock.sendto(answer, addr)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            self._sock.close()


def find(timeout: float = 2.0, port: int = PORT, targets: list[str] | None = None) -> list[dict]:
    """The servers that answer within ``timeout``: [{"id", "name", "url", "version"}], each once."""
    found: dict[str, dict] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sk.settimeout(0.2)
        for t in targets or ["255.255.255.255", "<broadcast>"]:
            try:
                sk.sendto(ASK, (t, port))
            except OSError:
                pass
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, addr = sk.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                d = json.loads(data)
                if d.get("lanepulse") != 1:
                    continue
                url = f"http://{addr[0]}:{int(d['port'])}"
            except (ValueError, KeyError, TypeError):
                continue
            found.setdefault(str(d.get("id") or url), {"id": d.get("id"), "name": str(d.get("name", ""))[:80],
                                                       "url": url, "version": d.get("version")})
    return list(found.values())
