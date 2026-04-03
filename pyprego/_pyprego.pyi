"""Type stubs for the _pyprego C extension module."""

def version() -> str:
    """Return the _pyprego extension version string."""

class error(Exception):
    """Custom exception raised by _pyprego C extension functions."""
