

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

CREATE_RE = re.compile(r'^CREATE\s+TABLE\s+(?:ONLY\s+)?(?:"([^"]+)"|(\w+))\s*\($', re.I)
COPY_RE = re.compile(
    r'^COPY\s+(?:(?:"[^"]+"|\w+)\.)?(?:"([^"]+)"|(\w+))\s*\(([^)]+)\)\s+FROM\s+stdin;\s*$', re.I
)
UNIQUE_RE = re.compile(
    r'ALTER\s+TABLE\s+(?:ONLY\s+)?(?:"([^"]+)"|(\w+))[^;]*?UNIQUE\s*\(([^)]+)\)', re.I | re.S
)


# ---------------------------------------------------------------------------
# Dump parsing and rendering
# ---------------------------------------------------------------------------

@dataclass
class CopyBlock:
    table: str
    header: list[str]
    rows: list[list[Optional[str]]] = field(default_factory=list)


def clean_name(raw: str) -> str:
    return raw.strip().strip('"')


def parse_row(line: str, table: str, width: int) -> list[Optional[str]]:
    fields = line.split("\t")
    if len(fields) != width:
        raise ValueError(f"{table}: COPY row has {len(fields)} fields, expected {width}")
    return [None if value == r"\N" else value for value in fields]


def format_row(row: list[Optional[str]]) -> str:
    return "\t".join(r"\N" if value is None else value for value in row)


def scan_copies(lines: list[str]) -> dict[str, CopyBlock]:
    blocks: dict[str, CopyBlock] = {}
    i = 0
    while i < len(lines):
        match = COPY_RE.match(lines[i].rstrip("\r"))
        if not match:
            i += 1
            continue
        table = clean_name(match.group(1) or match.group(2))
        header = [clean_name(c) for c in match.group(3).split(",")]
        block = CopyBlock(table=table, header=header)
        j = i + 1
        while j < len(lines) and lines[j].rstrip("\r") != "\\.":
            block.rows.append(parse_row(lines[j].rstrip("\r"), table, len(header)))
            j += 1
        if j >= len(lines):
            raise ValueError(f"COPY block for {table} is not terminated by \\.")
        blocks[table] = block
        i = j + 1
    return blocks


def unique_columns(text: str) -> dict[str, set[str]]:
    unique: dict[str, set[str]] = {}
    for match in UNIQUE_RE.finditer(text):
        table = clean_name(match.group(1) or match.group(2))
        unique.setdefault(table, set()).update(clean_name(c) for c in match.group(3).split(","))
    return unique


def render_dump(
    lines: list[str],
    copies: dict[str, CopyBlock],
    extra_columns: dict[str, list[str]],
) -> str:
    out: list[str] = []
    created: set[str] = set()
    i = 0
    while i < len(lines):
        bare = lines[i].rstrip("\r")
        create = CREATE_RE.match(bare)
        if create:
            name = clean_name(create.group(1) or create.group(2))
            if name in extra_columns:
                j = i + 1
                body: list[str] = []
                while j < len(lines) and not lines[j].rstrip("\r").startswith(")"):
                    body.append(lines[j].rstrip("\r"))
                    j += 1
                if j >= len(lines):
                    raise ValueError(f"CREATE TABLE {name} is not terminated by ');'")
                non_empty = [k for k, text in enumerate(body) if text.strip()]
                last = non_empty[-1]
                if not body[last].rstrip().endswith(","):
                    body[last] = body[last].rstrip() + ","
                out.append(bare)
                out.extend(body)
                new_columns = extra_columns[name]
                for k, column in enumerate(new_columns):
                    out.append(f"    {column} text" + ("," if k < len(new_columns) - 1 else ""))
                out.append(lines[j].rstrip("\r"))
                created.add(name)
                i = j + 1
                continue
        copy = COPY_RE.match(bare)
        if copy:
            name = clean_name(copy.group(1) or copy.group(2))
            if name in copies:
                block = copies[name]
                prefix = bare[: bare.index("(")]
                out.append(f"{prefix}({', '.join(block.header)}) FROM stdin;")
                out.extend(format_row(row) for row in block.rows)
                out.append("\\.")
                j = i + 1
                while j < len(lines) and lines[j].rstrip("\r") != "\\.":
                    j += 1
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    missing = set(extra_columns) - created
    if missing:
        raise ValueError(f"could not rewrite CREATE TABLE for: {sorted(missing)}")
    return "\n".join(out)


def shared_column_names(tables) -> set:
    """Column names that occur in two or more tables. NATURAL JOIN and USING join on these."""
    seen: dict = {}
    for table in tables.values():
        for column in table.column_names():
            seen[column] = seen.get(column, 0) + 1
    return {name for name, count in seen.items() if count >= 2}
