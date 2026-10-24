import io
import sys

from trafficwb import safe_console


def test_non_latin_names_do_not_crash_a_cp1252_console(monkeypatch):
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    safe_console()
    print("Offline check – בדיקה.csv")
    out.flush()
    assert b"\\u05d1" in buf.getvalue()
