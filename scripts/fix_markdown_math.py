#!/usr/bin/env python3
"""Normalize Markdown math syntax for the Astro/remark-math note pipeline.

Default mode is a dry run. Use --write to update files in place.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DISPLAY_OPEN_RE = re.compile(r"^\s*\\\[\s*$")
DISPLAY_CLOSE_RE = re.compile(r"^\s*\\\]\s*$")
MATH_FENCE_RE = re.compile(r"^\s*\$\$\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_PAREN_RE = re.compile(r"\\\((.+?)\\\)")
INLINE_BRACKET_ONE_LINE_RE = re.compile(r"\\\[(.+?)\\\]")


@dataclass
class Finding:
    path: Path
    line: int
    message: str


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
            converted = INLINE_PAREN_RE.sub(lambda m: f"${m.group(1)}$", line)
            if converted != line:
                findings.append(Finding(path, index, r"converted inline math \( ... \) to $...$"))
            line = converted

        out.append(line)

    if in_math:
        findings.append(Finding(path, len(lines), "unclosed $$ math block"))

    return out, findings


def collapse_extra_blank_lines_around_math(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line == "$$" and out and out[-1] == "" and len(out) >= 2 and out[-2] == "":
            out.pop()
        out.append(line)
    return out


def fix_file(path: Path) -> tuple[str, list[Finding], bool]:
    original = path.read_text(encoding="utf-8-sig")
    lines = original.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    frontmatter, body = split_frontmatter(lines)

    fixed_body, findings = normalize_math_boundaries(body, path)
    fixed_body = collapse_extra_blank_lines_around_math(fixed_body)
    fixed = "\n".join(frontmatter + fixed_body).rstrip() + "\n"
    original_normalized = original.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"

    return fixed, findings, bool(findings) and fixed != original_normalized


def iter_markdown_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Markdown math delimiters in notes.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["notes"],
        help="Markdown files or directories to scan. Defaults to notes/",
    )
    parser.add_argument("--write", action="store_true", help="Write fixes to disk.")
    args = parser.parse_args()

    repo = Path.cwd()
    targets: list[Path] = []
    for value in args.paths:
        target = Path(value)
        if not target.is_absolute():
            target = repo / target
        targets.extend(iter_markdown_files(target))

    total_findings: list[Finding] = []
    changed: list[Path] = []

    for path in targets:
        fixed, findings, did_change = fix_file(path)
        total_findings.extend(findings)
        if did_change:
            changed.append(path)
            if args.write:
                path.write_text(fixed, encoding="utf-8", newline="\n")

    if total_findings:
        print("Potential math formatting fixes:")
        for item in total_findings:
            rel = item.path.relative_to(repo) if item.path.is_relative_to(repo) else item.path
            print(f"- {rel}:{item.line}: {item.message}")
    else:
        print("No math delimiter issues found.")

    if changed:
        mode = "Updated" if args.write else "Would update"
        print(f"\n{mode} {len(changed)} file(s):")
        for path in changed:
            rel = path.relative_to(repo) if path.is_relative_to(repo) else path
            print(f"- {rel}")
    else:
        print("\nNo files need changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
