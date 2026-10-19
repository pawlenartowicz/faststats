from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[3] / "NeuroStats" / "tests" / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    assert FIXTURES.is_dir(), f"crate fixtures not found at {FIXTURES}"
    return FIXTURES
