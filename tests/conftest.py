from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOTS = Path(__file__).parent / "snapshots"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Rewrite snapshot files instead of comparing against them",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "docling: needs Docling models (slow, skipped when absent)")
    config.addinivalue_line("markers", "integration: needs Docker (testcontainers)")


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def snapshot(request: pytest.FixtureRequest):
    """Compare a JSON-serialisable value with `tests/snapshots/<name>.json`.

    Run `pytest --update-snapshots` to (re)generate; review the diff before committing.
    """
    update = request.config.getoption("--update-snapshots")

    def check(name: str, value) -> None:  # noqa: ANN001
        path = SNAPSHOTS / f"{name}.json"
        rendered = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        if update or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            if not update:
                pytest.fail(f"snapshot {path.name} did not exist; written, re-run to verify")
            return
        expected = path.read_text(encoding="utf-8")
        assert rendered == expected, (
            f"snapshot mismatch for {path.name}; run with --update-snapshots if intended"
        )

    return check
