"""Artifact path management and run ID generation for chemometrics MCP tools.

All tools must write artifacts under approved run directories only. Tools must
never accept caller-supplied absolute paths that escape the configured runs root.
"""
from __future__ import annotations

import datetime
import os
import re
import uuid
from pathlib import Path

from chemometrics_contracts import DEFAULT_ARTIFACTS_DIR, DEFAULT_RUNS_DIR, ArtifactReference

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]")


def make_run_id(slug: str = "") -> str:
    """Return a unique run ID following the project convention.

    Format: ``run-YYYYMMDD-HHMMSS-<slug>-<short-uuid>``
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_slug = _SLUG_RE.sub("-", slug.strip())[:32].strip("-") if slug else "run"
    short_uid = uuid.uuid4().hex[:6]
    return f"run-{ts}-{safe_slug}-{short_uid}"


def run_artifacts_dir(run_id: str, runs_root: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Return the artifacts directory path for a run ID.

    The returned path is always a subdirectory of *runs_root*. Raises
    ``ValueError`` if *run_id* contains path-separator characters.
    """
    if os.sep in run_id or "/" in run_id:
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")
    return Path(runs_root) / run_id / DEFAULT_ARTIFACTS_DIR


def ensure_run_dir(run_id: str, runs_root: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Create and return the artifacts directory for *run_id*."""
    path = run_artifacts_dir(run_id, runs_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_ref(
    run_id: str,
    filename: str,
    kind: str,
    *,
    label: str | None = None,
    mime_type: str | None = None,
    runs_root: str | Path = DEFAULT_RUNS_DIR,
) -> ArtifactReference:
    """Build an :class:`ArtifactReference` for a file inside a run directory.

    The URI uses forward slashes for portability.
    """
    rel = run_artifacts_dir(run_id, runs_root) / filename
    return ArtifactReference(
        kind=kind,
        uri=rel.as_posix(),
        label=label,
        mime_type=mime_type,
    )
