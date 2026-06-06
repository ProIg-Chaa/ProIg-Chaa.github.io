#!/usr/bin/env python3
"""Prepare formatted Markdown notes for publishing.

The script never edits source notes. It reads Markdown files, normalizes common
math delimiters, refreshes frontmatter timestamps, and writes formatted copies
to a separate output directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DISPLAY_OPEN_RE = re.compile(r"^\s*\\\[\s*$")
DISPLAY_CLOSE_RE = re.compile(r"^\s*\\\]\s*$")
MATH_FENCE_RE = re.compile(r"^\s*\$\$\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_PAREN_RE = re.compile(r"\\\((.+?)\\\)")
INLINE_BRACKET_ONE_LINE_RE = re.compile(r"\\\[(.+?)\\\]")
DEFAULT_INPUT = r"D:\GS_LearningAndWork\ai infra\CS336\typora"


@dataclass
class Finding:
    path: Path
    line: int
    message: str
    severity: str = "fix"


@dataclass
class PreparedFile:
    source: Path
    output: Path
    findings: list[Finding]
    changed: bool


def split_frontmatter(lines: list[str]) -> tuple[list[str], list[str]]:
    if not lines or lines[0].strip() != "---":
        return [], lines

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return lines[: index + 1], lines[index + 1 :]

    return [], lines


def normalize_math_boundaries(lines: list[str], path: Path) -> tuple[list[str], list[Finding]]:
    findings: list[Finding] = []
    out: list[str] = []
    in_code = False
    in_math = False

    for index, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        if FENCE_RE.match(line):
            in_code = not in_code
            out.append(line)
            continue

        if in_code:
            out.append(line)
            continue

        if DISPLAY_OPEN_RE.match(line):
            findings.append(Finding(path, index, r"converted display delimiter \[ to $$"))
            line = "$$"
        elif DISPLAY_CLOSE_RE.match(line):
            findings.append(Finding(path, index, r"converted display delimiter \] to $$"))
            line = "$$"

        def replace_one_line_display(match: re.Match[str]) -> str:
            findings.append(Finding(path, index, r"converted one-line \[...\] display math to $$ block"))
            return f"\n$$\n{match.group(1).strip()}\n$$\n"

        line = INLINE_BRACKET_ONE_LINE_RE.sub(replace_one_line_display, line)

        if MATH_FENCE_RE.match(line):
            if out and out[-1].strip() and not in_math:
                out.append("")
            out.append("$$")
            in_math = not in_math
            continue

        if not in_math:
            converted = INLINE_PAREN_RE.sub(lambda match: f"${match.group(1)}$", line)
            if converted != line:
                findings.append(Finding(path, index, r"converted inline math \( ... \) to $...$"))
            line = converted

        out.append(line)

    if in_math:
        findings.append(Finding(path, len(lines), "unclosed $$ math block", severity="error"))
    if in_code:
        findings.append(Finding(path, len(lines), "unclosed fenced code block", severity="error"))

    return out, findings


def collapse_extra_blank_lines_around_math(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line == "$$" and out and out[-1] == "" and len(out) >= 2 and out[-2] == "":
            out.pop()
        out.append(line)
    return out


def upsert_frontmatter(frontmatter: list[str], timestamp: str) -> list[str]:
    if frontmatter:
        body = frontmatter[1:-1]
    else:
        body = []

    seen_date = False
    seen_updated = False
    next_body: list[str] = []

    for line in body:
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if key == "date":
            seen_date = True
            next_body.append(line)
        elif key == "updated":
            seen_updated = True
            next_body.append(f"updated: {timestamp}")
        else:
            next_body.append(line)

    if not seen_date:
        next_body.insert(0, f"date: {timestamp}")
    if not seen_updated:
        insert_at = 1 if next_body and next_body[0].startswith("date:") else len(next_body)
        next_body.insert(insert_at, f"updated: {timestamp}")

    return ["---", *next_body, "---", ""]


def prepare_text(path: Path, timestamp: str) -> tuple[str, list[Finding], bool]:
    original = path.read_text(encoding="utf-8-sig")
    normalized_original = original.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
    lines = normalized_original.split("\n")
    frontmatter, body = split_frontmatter(lines)

    fixed_body, findings = normalize_math_boundaries(body, path)
    fixed_body = collapse_extra_blank_lines_around_math(fixed_body)
    fixed_frontmatter = upsert_frontmatter(frontmatter, timestamp)
    fixed = "\n".join(fixed_frontmatter + fixed_body).rstrip() + "\n"

    return fixed, findings, fixed != normalized_original


def iter_markdown_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def relative_output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / path.name
    return output_root / path.relative_to(input_root)


def safe_reset_output_dir(output_root: Path, repo: Path) -> Path:
    resolved_output = output_root.resolve()
    resolved_repo = repo.resolve()

    if resolved_output == resolved_repo:
        raise ValueError("Refusing to use repository root as output directory.")
    if not str(resolved_output).startswith(str(resolved_repo)):
        raise ValueError("Output directory must stay inside this repository.")
    def remove_readonly_and_retry(function, path, exc_info) -> None:
        try:
            os.chmod(path, 0o700)
            function(path)
        except PermissionError as error:
            raise error

    if resolved_output.exists():
        try:
            shutil.rmtree(resolved_output, onexc=remove_readonly_and_retry)
        except PermissionError:
            fallback = resolved_output.with_name(
                f"{resolved_output.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            fallback.mkdir(parents=True, exist_ok=False)
            print(
                f"Could not reset {resolved_output}; writing to fallback output {fallback}."
            )
            return fallback

    resolved_output.mkdir(parents=True, exist_ok=True)
    return resolved_output


def prepare_files(input_root: Path, output_root: Path, timestamp: str) -> list[PreparedFile]:
    prepared: list[PreparedFile] = []
    for source in iter_markdown_files(input_root):
        output = relative_output_path(source, input_root, output_root)
        text, findings, changed = prepare_text(source, timestamp)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
        prepared.append(PreparedFile(source, output, findings, changed))
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="Create formatted Markdown copies for publishing.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Markdown file or directory to prepare. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        default="formatted-notes",
        help="Output directory for formatted copies. Defaults to formatted-notes/.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Frontmatter timestamp to write. Defaults to current local time.",
    )
    args = parser.parse_args()

    repo = Path.cwd()
    input_root = Path(args.input)
    output_root = Path(args.output)
    if not input_root.is_absolute():
        input_root = repo / input_root
    if not output_root.is_absolute():
        output_root = repo / output_root

    timestamp = args.timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
    output_root = safe_reset_output_dir(output_root, repo)
    prepared = prepare_files(input_root, output_root, timestamp)

    all_findings = [finding for item in prepared for finding in item.findings]
    errors = [finding for finding in all_findings if finding.severity == "error"]
    fixes = [finding for finding in all_findings if finding.severity != "error"]

    print(f"Prepared {len(prepared)} Markdown file(s).")
    print(f"Source: {input_root}")
    print(f"Output: {output_root}")
    print(f"Timestamp: {timestamp}")

    if fixes:
        print("\nMath formatting fixes applied in generated copies:")
        for item in fixes:
            rel = item.path.relative_to(repo) if item.path.is_relative_to(repo) else item.path
            print(f"- {rel}:{item.line}: {item.message}")
    else:
        print("\nNo math delimiter fixes were needed.")

    changed = [item for item in prepared if item.changed]
    if changed:
        print("\nGenerated files with source-to-output changes:")
        for item in changed:
            src = item.source.relative_to(repo) if item.source.is_relative_to(repo) else item.source
            dst = item.output.relative_to(repo) if item.output.is_relative_to(repo) else item.output
            print(f"- {src} -> {dst}")
    else:
        print("\nGenerated copies match normalized source content except for output location.")

    if errors:
        print("\nBlocking math/code formatting errors:")
        for item in errors:
            rel = item.path.relative_to(repo) if item.path.is_relative_to(repo) else item.path
            print(f"- {rel}:{item.line}: {item.message}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
