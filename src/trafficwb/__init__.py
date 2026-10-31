"""LanePulse (formerly Traffic Analysis Workbench) foundations."""
__version__ = "0.2.4"


def safe_console() -> None:
    """Never crash when printing Hebrew, Persian or Arabic names to a console that cannot show them (e.g. cp1252)."""
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
