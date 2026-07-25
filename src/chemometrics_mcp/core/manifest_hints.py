"""Manifest hints: declared-mapping loader and filename-convention heuristic.

Two tiers
---------
1. **Declared mapping** (``manifest_hints.json`` in the project source_root):
   authoritative, user-supplied truth.  Loaded automatically when present;
   applied with full confidence as ``provenance="declared"``.
   JSON only — no YAML dependency is introduced (PyYAML is not listed in
   pyproject.toml).  The file schema is::

       {
         "<filename-stem-or-glob>": {
           "role":           "reference|sample|calibration|…",
           "reference_name": "compound-A",          # optional
           "composition":    {"A": 0.3, "B": 0.7}  # optional
         }
       }

   Keys are matched against measurement filename stems in order.  Exact stem
   matches take priority over ``fnmatch``-style glob patterns.

2. **Filename-convention heuristic** (opt-in, advisory):
   derives role/composition guesses from common naming conventions such as
   ``pure_A``, ``ref_B``, ``mix_30_70``, or ``mix_A_30_B_70``.  Each inferred
   hint is tagged ``provenance="inferred"`` plus a human-readable ``note`` so the
   scientist knows to confirm before treating the result as ground truth.

When no hints file exists and the heuristic flag is off, no hints are produced
and the downstream draft manifest is byte-identical to its pre-hints state.
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SampleHint:
    """Hint for a single sample keyed by filename stem."""
    role: str | None = None
    reference_name: str | None = None
    composition: dict[str, float] | None = None
    # "declared" = from manifest_hints.json; "inferred" = from filename heuristic
    provenance: str = "declared"
    note: str | None = None


@dataclass
class ManifestHints:
    """Collection of hints keyed by filename stem (or glob pattern)."""
    entries: dict[str, SampleHint] = field(default_factory=dict)
    # human-readable provenance tag for the collection as a whole
    source: str = ""

    def lookup(self, stem: str) -> SampleHint | None:
        """Return the hint for *stem*, preferring exact match over glob match."""
        if stem in self.entries:
            return self.entries[stem]
        for pattern, hint in self.entries.items():
            if fnmatch.fnmatch(stem, pattern):
                return hint
        return None


# ---------------------------------------------------------------------------
# Declared-mapping loader
# ---------------------------------------------------------------------------

_HINTS_FILENAMES = ("manifest_hints.json",)


def load_hints_file(source_root: str | Path) -> ManifestHints | None:
    """Return hints loaded from ``manifest_hints.json`` in *source_root*, or
    ``None`` if no hints file is present.

    Raises ``ValueError`` when a file exists but cannot be parsed.
    YAML (``.yaml``/``.yml``) is not supported because PyYAML is not a project
    dependency; use JSON.
    """
    root = Path(source_root)
    for name in _HINTS_FILENAMES:
        path = root / name
        if path.exists():
            return _parse_hints_json(path)
    return None


def _parse_hints_json(path: Path) -> ManifestHints:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level value must be a JSON object, got {type(data).__name__}"
        )
    entries: dict[str, SampleHint] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"{path}: entry {key!r} must be a JSON object, got {type(value).__name__}"
            )
        composition = value.get("composition")
        if composition is not None and not isinstance(composition, dict):
            raise ValueError(
                f"{path}: entry {key!r} 'composition' must be a JSON object"
            )
        entries[key] = SampleHint(
            role=value.get("role"),
            reference_name=value.get("reference_name"),
            composition={str(k): float(v) for k, v in composition.items()} if composition else None,
            provenance="declared",
        )
    return ManifestHints(entries=entries, source=str(path))


# ---------------------------------------------------------------------------
# Filename-convention heuristic
# ---------------------------------------------------------------------------

# Patterns that unambiguously signal a pure reference spectrum.
# Matches: pure_A, ref_B, reference_C, A_pure, B_ref, B_reference (any case).
_PURE_RE = re.compile(
    r"(?:^(?:pure|ref|reference)[_\-])|(?:[_\-](?:pure|ref|reference)$)",
    re.IGNORECASE,
)

# mix_30_70 — two numbers (percentages or fractions), no component labels.
_MIX_POSITIONAL_RE = re.compile(
    r"^mix(?:ture)?[_\-](\d+(?:\.\d+)?)[_\-](\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)

# mix_A_30_B_70 — named components with numeric fractions.
_MIX_NAMED_RE = re.compile(
    r"^mix(?:ture)?[_\-]([a-z][a-z0-9]*)[_\-](\d+(?:\.\d+)?)[_\-]([a-z][a-z0-9]*)[_\-](\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def infer_hints_from_filenames(stems: list[str]) -> ManifestHints:
    """Return advisory hints inferred from filename stems.

    Each returned hint has ``provenance="inferred"`` and a ``note`` explaining
    the inference so the scientist can confirm before treating it as truth.
    Stems that match no recognized pattern are absent from the result.

    Recognized conventions
    ~~~~~~~~~~~~~~~~~~~~~~
    - ``pure_<name>`` / ``ref_<name>`` / ``<name>_pure`` / ``<name>_ref`` ->
      ``role="reference"``
    - ``mix_30_70`` / ``mixture_30_70`` -> ``role="sample"``,
      ``composition={"A": 0.30, "B": 0.70}`` (component labels A/B are
      provisional; confirm identities)
    - ``mix_A_30_B_70`` / ``mixture_A_30_B_70`` -> ``role="sample"``,
      ``composition={"A": 0.30, "B": 0.70}``
    """
    entries: dict[str, SampleHint] = {}
    for stem in stems:
        hint = _infer_one(stem)
        if hint is not None:
            entries[stem] = hint
    return ManifestHints(entries=entries, source="filename_heuristic")


def _infer_one(stem: str) -> SampleHint | None:
    if _PURE_RE.search(stem):
        return SampleHint(
            role="reference",
            provenance="inferred",
            note=f"role=reference inferred from filename stem {stem!r}; confirm this is a pure-component reference spectrum",
        )

    m = _MIX_POSITIONAL_RE.match(stem)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        total = a + b
        if total > 0:
            comp = {"A": round(a / total, 6), "B": round(b / total, 6)}
            return SampleHint(
                role="sample",
                composition=comp,
                provenance="inferred",
                note=(
                    f"composition inferred from filename stem {stem!r}; "
                    "component labels A/B are provisional -- confirm actual component identities"
                ),
            )

    m2 = _MIX_NAMED_RE.match(stem)
    if m2:
        name_a, frac_a = m2.group(1), float(m2.group(2))
        name_b, frac_b = m2.group(3), float(m2.group(4))
        total = frac_a + frac_b
        if total > 0:
            comp = {name_a: round(frac_a / total, 6), name_b: round(frac_b / total, 6)}
            return SampleHint(
                role="sample",
                composition=comp,
                provenance="inferred",
                note=f"composition inferred from filename stem {stem!r}; confirm component identities and fractions",
            )

    return None
