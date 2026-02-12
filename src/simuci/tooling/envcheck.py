"""Environment/dependency checks (opt-in).

This module is intentionally separate from the simulation runtime.
It can be invoked manually (or from CI) to verify:
- declared dependencies are installed and satisfy version constraints
- core public API modules import correctly
- (optional) dependency vulnerability audit via pip-audit

Run:
    python -m simuci.envcheck
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import tomllib  # for >= python 3.11
except ModuleNotFoundError:
    tomllib = None


@dataclass(frozen=True)
class RequirementStatus:
    """Status for a single declared dependency."""

    name: str
    spec: str
    installed_version: str | None
    satisfied: bool
    error: str | None = None


@dataclass(frozen=True)
class VulnerabilityFinding:
    """Single vulnerability entry (best-effort schema from pip-audit JSON)."""

    package: str
    installed_version: str | None
    id: str | None
    description: str | None
    fix_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentCheckReport:
    """Aggregate report."""

    ok: bool
    python: str
    platform: str
    project_root: str | None
    requirements: tuple[RequirementStatus, ...]
    public_import_errors: tuple[str, ...]
    vulnerability_audit_ran: bool
    vulnerabilities: tuple[VulnerabilityFinding, ...]
    notes: tuple[str, ...] = ()


def run_environment_check(
    *,
    project_root: str | os.PathLike[str] | None = None,
    include_extras: Iterable[str] = (),
    audit_vulnerabilities: bool = False,
) -> EnvironmentCheckReport:
    """Run an opt-in environment verification.

    Args:
        project_root: Root directory containing `pyproject.toml`.
            Defaults to "best effort" (walk up from this file).
        include_extras: Optional extras from `[project.optional-dependencies]` to include.
        audit_vulnerabilities: When True, attempts to run pip-audit (if available).

    Returns:
        EnvironmentCheckReport
    """

    root_path = _resolve_project_root(project_root)

    declared = _load_declared_requirements(root_path, include_extras=include_extras)
    req_statuses, req_notes = _check_requirements(declared)

    import_errors = tuple(_check_public_imports())

    vulnerabilities: tuple[VulnerabilityFinding, ...] = ()
    audit_ran = False
    audit_notes: list[str] = []

    if audit_vulnerabilities:
        audit_ran = True
        vulnerabilities, audit_notes = _run_pip_audit_best_effort()

    ok = (
        all(r.satisfied for r in req_statuses)
        and len(import_errors) == 0
        and (len(vulnerabilities) == 0)
    )

    return EnvironmentCheckReport(
        ok=ok,
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        project_root=str(root_path) if root_path is not None else None,
        requirements=tuple(req_statuses),
        public_import_errors=import_errors,
        vulnerability_audit_ran=audit_ran,
        vulnerabilities=vulnerabilities,
        notes=tuple(req_notes + audit_notes),
    )


def _resolve_project_root(project_root: str | os.PathLike[str] | None) -> Path | None:
    """Resolve the project root directory containing pyproject.toml."""
    if project_root is not None:
        p = Path(project_root).resolve()
        return p

    # Best-effort: walk up from this file until we find a pyproject.toml
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _load_declared_requirements(
    project_root: Path | None,
    *,
    include_extras: Iterable[str],
) -> list[str]:
    """Return requirement strings declared in pyproject.

    If pyproject cannot be read, returns an empty list (with a note elsewhere).
    """

    if project_root is None:
        return []

    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file() or tomllib is None:
        return []

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return []

    project = data.get("project", {})
    deps: list[str] = list(project.get("dependencies", []) or [])

    extras_table = project.get("optional-dependencies", {}) or {}
    for extra in include_extras:
        extra_deps = extras_table.get(extra, []) or []
        deps.extend(list(extra_deps))

    return deps


def _check_requirements(requirements: list[str]) -> tuple[list[RequirementStatus], list[str]]:
    """Check installed packages + best-effort version satisfaction."""

    notes: list[str] = []

    if not requirements:
        notes.append("No declared requirements loaded (pyproject.toml missing/unreadable or empty).")
        return [], notes

    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        def parse_req(req: str) -> tuple[str, str, Any]:
            r = Requirement(req)
            return r.name, str(r.specifier), r.specifier

        def is_satisfied(installed: str, spec: Any) -> bool:
            if not str(spec):
                return True
            return Version(installed) in SpecifierSet(str(spec))

    except Exception:  # pragma: no cover
        notes.append("`packaging` not available; skipping version constraint checks.")

        def parse_req(req: str) -> tuple[str, str, Any]:
            name = req.strip().split()[0]
            # crude split at first version/operator char
            for i, ch in enumerate(name):
                if ch in "<>=!~":
                    name = name[:i]
                    break
            return name, req, None

        def is_satisfied(installed: str, spec: Any) -> bool:
            return True

    results: list[RequirementStatus] = []
    for req_str in requirements:
        name, spec_str, spec_obj = parse_req(req_str)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            results.append(
                RequirementStatus(
                    name=name,
                    spec=spec_str,
                    installed_version=None,
                    satisfied=False,
                    error="Package not installed",
                )
            )
            continue
        except Exception as e:
            results.append(
                RequirementStatus(
                    name=name,
                    spec=spec_str,
                    installed_version=None,
                    satisfied=False,
                    error=f"Failed to read installed version: {e}",
                )
            )
            continue

        try:
            satisfied = is_satisfied(installed, spec_obj)
        except Exception as e:
            satisfied = True
            notes.append(f"Could not evaluate specifier for {name!r}: {e}")

        results.append(
            RequirementStatus(
                name=name,
                spec=spec_str,
                installed_version=installed,
                satisfied=bool(satisfied),
                error=None if satisfied else "Installed version does not satisfy specifier",
            )
        )

    return results, notes


def _check_public_imports() -> Iterable[str]:
    """Ensure the core public modules import without side-effects failures."""

    # Keep these imports light: only module import, no execution.
    modules = (
        "simuci.core.distributions",
        "simuci.core.experiment",
        "simuci.io.loaders",
        "simuci.core.simulation",
        "simuci.analysis.stats",
        "simuci.validation.validators",
        "simuci.validation.schemas",
        "simuci.io.process_data",
    )

    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            yield f"{mod}: {type(e).__name__}: {e}"


def _run_pip_audit_best_effort() -> tuple[tuple[VulnerabilityFinding, ...], list[str]]:
    """Run pip-audit if available and parse JSON output.

    This is optional by design; when pip-audit isn't installed, the check is
    skipped and reported via notes.
    """

    notes: list[str] = []

    cmd = [sys.executable, "-m", "pip_audit", "-f", "json"]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        notes.append(f"pip-audit execution failed: {e}")
        return (), notes

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        combined = "\n".join([s for s in (stdout, stderr) if s])
        if "No module named" in combined and "pip_audit" in combined:
            notes.append("pip-audit not installed; vulnerability audit skipped.")
        else:
            notes.append(f"pip-audit returned non-zero ({proc.returncode}); audit skipped.")
            if combined:
                # Keep it short; avoid dumping large text.
                notes.append(combined.splitlines()[0][:200])
        return (), notes

    raw = (proc.stdout or "").strip()
    if not raw:
        notes.append("pip-audit produced no output; audit skipped.")
        return (), notes

    try:
        payload = json.loads(raw)
    except Exception:
        notes.append("pip-audit output was not valid JSON; audit skipped.")
        return (), notes

    findings: list[VulnerabilityFinding] = []

    # pip-audit JSON is a list of package entries with vulnerabilities.
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            pkg = str(entry.get("name") or "")
            ver = entry.get("version")
            vulns = entry.get("vulns") or entry.get("vulnerabilities") or []
            if not isinstance(vulns, list):
                continue
            for v in vulns:
                if not isinstance(v, dict):
                    continue
                findings.append(
                    VulnerabilityFinding(
                        package=pkg,
                        installed_version=str(ver) if ver is not None else None,
                        id=(v.get("id") or v.get("cve") or v.get("ghsa")) and str(
                            v.get("id") or v.get("cve") or v.get("ghsa")
                        ),
                        description=(v.get("description") and str(v.get("description")))
                        or (v.get("details") and str(v.get("details")))
                        or None,
                        fix_versions=tuple(map(str, v.get("fix_versions") or v.get("fix_versions") or ())),
                    )
                )

    return tuple(findings), notes


def _format_report(report: EnvironmentCheckReport) -> str:
    """Format the environment check report as a human-readable string."""

    lines: list[str] = []
    lines.append("SimUCI environment check")
    lines.append(f"Python: {report.python}")
    lines.append(f"Platform: {report.platform}")
    if report.project_root:
        lines.append(f"Project root: {report.project_root}")

    if report.requirements:
        lines.append("\nDependencies:")
        for r in report.requirements:
            status = "OK" if r.satisfied else "FAIL"
            inst = r.installed_version or "<missing>"
            spec = r.spec or ""
            lines.append(f"  - {r.name} {spec}  (installed: {inst})  [{status}]")
            if r.error and not r.satisfied:
                lines.append(f"      {r.error}")
    else:
        lines.append("\nDependencies: <not checked>")

    if report.public_import_errors:
        lines.append("\nPublic import checks: FAIL")
        for e in report.public_import_errors:
            lines.append(f"  - {e}")
    else:
        lines.append("\nPublic import checks: OK")

    if report.vulnerability_audit_ran:
        if report.vulnerabilities:
            lines.append("\nVulnerability audit: FAIL")
            lines.append(f"  Findings: {len(report.vulnerabilities)}")
            # Print at most a few findings to keep output readable.
            for f in report.vulnerabilities[:10]:
                vid = f.id or "<unknown-id>"
                pv = f.installed_version or "?"
                lines.append(f"  - {f.package}=={pv}: {vid}")
        else:
            lines.append("\nVulnerability audit: OK (or skipped)")

    if report.notes:
        lines.append("\nNotes:")
        for n in report.notes:
            lines.append(f"  - {n}")

    lines.append("\nOverall: " + ("OK" if report.ok else "FAIL"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the environment check."""

    argv = list(sys.argv[1:] if argv is None else argv)

    audit = False
    extras: list[str] = []

    for a in list(argv):
        if a == "--audit":
            audit = True
            argv.remove(a)
        elif a.startswith("--extra="):
            extras.append(a.split("=", 1)[1].strip())
            argv.remove(a)

    report = run_environment_check(include_extras=extras, audit_vulnerabilities=audit)
    sys.stdout.write(_format_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
