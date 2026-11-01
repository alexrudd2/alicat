"""Test configuraton parameters."""

def pytest_addoption(parser):
    """Hardware testing option for pytest."""
    parser.addoption("--hardware", action="store_true", help="Run hardware tests.")
