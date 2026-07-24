"""Small, filesystem-only storage primitives for reproducible analyses.

The module deliberately has no dependency on the application contracts.  It is
therefore safe to use while those contracts are evolving, and is also useful to
tools which only need to persist plain dictionaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ALLOWED_ROOTS_ENV = "CHEMOMETRICS_ALLOWED_ROOTS"


def enforce_allowed_path(path: str | Path) -> Path:
    """Resolve a path and enforce an optional deployment allowlist."""
    resolved = Path(path).expanduser().resolve()
    configured = os.environ.get(_ALLOWED_ROOTS_ENV, "").strip()
    if not configured:
        return resolved
    roots = [
        Path(item).expanduser().resolve()
        for item in configured.split(os.pathsep)
        if item.strip()
    ]
    if not any(
        resolved == root or root in resolved.parents for root in roots
    ):
        raise ValueError(
            f"path is outside configured {_ALLOWED_ROOTS_ENV} roots"
        )
    return resolved


def slugify_project_id(value: str) -> str:
    """Return a filesystem-safe, lowercase project identifier.

    Empty or punctuation-only values are rejected rather than silently turning
    into a surprising filename.
    """
    slug = _SLUG_RE.sub("-", str(value).strip().lower()).strip("-.")
    if not slug:
        raise ValueError("project id must contain at least one letter or number")
    return slug


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert the common model protocols to plain JSON-compatible data."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_ready(value.model_dump(mode="json"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_ready(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(data: Any) -> bytes:
    """Serialize data deterministically for evidence hashes and JSON writes."""
    return json.dumps(
        _json_ready(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def data_hash(data: Any) -> str:
    """Return the SHA-256 hash of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def fingerprint_source(source_root: str | Path, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Fingerprint regular source files, excluding the output directory itself."""
    source = Path(source_root).resolve()
    excluded = (Path(output_root).resolve() if output_root is not None else source / "chemometrics-output")
    records: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            continue
        # resolve also makes an output symlink unambiguously excluded.
        resolved = path.resolve()
        if resolved == excluded or excluded in resolved.parents or not path.is_file():
            continue
        stat = path.stat()
        records.append({
            "relative_path": path.relative_to(source).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": stat.st_size,
            "format": path.suffix.lower().lstrip("."),
        })
    return records


# A more explicit spelling is convenient to callers and future contracts.
source_fingerprint = fingerprint_source


class ProjectStore:
    """Safely persist JSON evidence under one already-created output root."""

    def __init__(self, output_root: str | Path):
        self.output_root = enforce_allowed_path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        for directory in ("manifests", "plans", "runs"):
            (self.output_root / directory).mkdir(exist_ok=True)

    def _path(self, artifact_name: str | Path) -> Path:
        raw = Path(artifact_name)
        if raw.is_absolute() or ".." in raw.parts or not raw.parts:
            raise ValueError("artifact name must be a relative path without '..'")
        candidate = self.output_root.joinpath(raw)
        # Existing symlinks (including a final symlink) must never redirect I/O.
        resolved = candidate.resolve()
        if resolved != self.output_root and self.output_root not in resolved.parents:
            raise ValueError("artifact path escapes the project output root")
        return candidate

    def write_json(self, artifact_name: str | Path, data: Any) -> Path:
        """Atomically write canonical JSON below the project root."""
        return self.write_bytes(artifact_name, canonical_json_bytes(data) + b"\n")

    def write_bytes(self, artifact_name: str | Path, payload: bytes) -> Path:
        """Atomically write bytes below the project root.

        This is used for compact array payloads as well as JSON.  It has the
        same containment checks as :meth:`write_json`.
        """
        destination = self._path(artifact_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        parent = destination.parent.resolve()
        if parent != self.output_root and self.output_root not in parent.parents:
            raise ValueError("artifact path escapes the project output root")
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def path_for(self, artifact_name: str | Path) -> Path:
        """Return a verified output-root-relative path without writing it."""
        return self._path(artifact_name)

    def read_json(self, artifact_name: str | Path) -> Any:
        with self._path(artifact_name).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _versioned_name(self, directory: str, name: str, version: str | int | None) -> str:
        stem = slugify_project_id(name)
        if version is None:
            existing = sorted((self.output_root / directory).glob(f"{stem}-v*.json"))
            version = len(existing) + 1
        token = slugify_project_id(str(version))
        return f"{directory}/{stem}-v{token}.json"

    def save_manifest(self, data: Any, name: str = "manifest", version: str | int | None = None) -> Path:
        return self.write_json(self._versioned_name("manifests", name, version), data)

    def save_plan(self, data: Any, name: str = "plan", version: str | int | None = None) -> Path:
        return self.write_json(self._versioned_name("plans", name, version), data)

    def save_run(self, data: Any, run_id: str | None = None) -> Path:
        run_id = run_id or f"run-{uuid.uuid4().hex}"
        return self.write_json(f"runs/{slugify_project_id(run_id)}.json", data)

    def list_runs(self) -> list[Path]:
        return sorted(path for path in (self.output_root / "runs").glob("*.json") if path.is_file())


def create_project_layout(
    source_root: str | Path,
    output_root: str | Path | None = None,
    project_id: str | None = None,
) -> ProjectStore:
    """Create an isolated evidence layout, without changing source inputs."""
    source = enforce_allowed_path(source_root)
    if not source.is_dir():
        raise ValueError(f"source root is not a directory: {source}")
    output = (
        enforce_allowed_path(output_root)
        if output_root is not None
        else enforce_allowed_path(source / "chemometrics-output")
    )
    if output == source or output in source.parents:
        raise ValueError("output root must not be the source root or one of its parents")
    requested_project_id = slugify_project_id(project_id) if project_id else None
    project_id = requested_project_id or f"{slugify_project_id(source.name)}-{uuid.uuid4().hex[:12]}"
    store = ProjectStore(output)
    project_file = store.output_root / "project.json"
    if project_file.exists():
        existing = store.read_json("project.json")
        if requested_project_id is not None and existing.get("project_id") != requested_project_id:
            raise ValueError(
                f"output root already belongs to project {existing.get('project_id')!r}"
            )
    else:
        store.write_json("project.json", {
            "project_id": project_id,
            "source_root": str(source),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return store
