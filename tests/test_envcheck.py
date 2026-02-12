from __future__ import annotations

from pathlib import Path

from simuci.tooling.envcheck import run_environment_check


def test_environment_check_runs_without_audit() -> None:
    # Best-effort: point at repo root so declared dependencies are loaded.
    root = Path(__file__).resolve().parents[1]

    report = run_environment_check(project_root=root, audit_vulnerabilities=False)

    # The check should run and produce a report.
    assert report.python

    # In the test environment, core deps should be present.
    missing = [r for r in report.requirements if r.installed_version is None]
    assert missing == []

    # Public imports should succeed.
    assert report.public_import_errors == ()
