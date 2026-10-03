#!/usr/bin/env python3
"""Generate CHANGELOG.md from the Git history for this project."""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
)
PROJECT_PATH = ROOT.relative_to(REPO_ROOT).as_posix()
OUTPUT = ROOT / "CHANGELOG.md"

TYPE_LABELS = {
    "feat": "Features",
    "fix": "Fixes",
    "docs": "Documentacion",
    "build": "Build",
    "chore": "Mantenimiento",
    "test": "Tests",
    "refactor": "Refactors",
    "perf": "Performance",
    "ci": "CI",
}

CONVENTIONAL_RE = re.compile(r"^(?P<kind>[a-z]+)(?:\((?P<scope>[^)]+)\))?: (?P<message>.+)$")


def git_log() -> list[tuple[str, str, str]]:
    raw = subprocess.check_output(
        [
            "git",
            "log",
            "--date=short",
            "--pretty=format:%h%x1f%ad%x1f%s",
            "--",
            PROJECT_PATH,
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    if not raw:
        return []

    rows = []
    for line in raw.splitlines():
        short_hash, date, subject = line.split("\x1f", 2)
        rows.append((short_hash, date, subject))
    return rows


def normalize(subject: str) -> tuple[str, str]:
    match = CONVENTIONAL_RE.match(subject)
    if not match:
        return "Otros cambios", subject

    kind = match.group("kind")
    scope = match.group("scope")
    message = match.group("message")
    label = TYPE_LABELS.get(kind, kind)
    if scope:
        message = f"{scope}: {message}"
    return label, message


def render(rows: list[tuple[str, str, str]]) -> str:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for short_hash, date, subject in rows:
        label, message = normalize(subject)
        grouped[label].append((short_hash, date, message))

    ordered_labels = list(TYPE_LABELS.values()) + ["Otros cambios"]
    labels = [label for label in ordered_labels if label in grouped]
    labels.extend(sorted(set(grouped) - set(labels)))

    lines = [
        "# Changelog",
        "",
        "Este archivo se genera automaticamente desde el historial de Git.",
        "Para actualizarlo, ejecutar:",
        "",
        "```bash",
        "python3 scripts/generate_changelog.py",
        "```",
        "",
    ]

    for label in labels:
        lines.append(f"## {label}")
        lines.append("")
        for short_hash, date, message in grouped[label]:
            lines.append(f"- {date} `{short_hash}` {message}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUTPUT.write_text(render(git_log()), encoding="utf-8")


if __name__ == "__main__":
    main()
