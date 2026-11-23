
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

from .evaluate import QPAIR_FIELDS, clean_qpair_value
from .dump_edit import render_dump, scan_copies
from .robustness import copy_base_scenario, scenario_source
from .schema import Table, parse_sql_dump

# Token-level dictionaries. Keys are lowercase words; identifiers are split on "_".
# Targets were chosen to avoid words already present in the RODI cmt schemas.
SCHEMA_SYNONYMS: dict[str, str] = {
    "author": "writer", "authors": "writers",
    "paper": "article", "papers": "articles",
    "conference": "meeting", "conferences": "meetings",
    "person": "individual", "persons": "individuals",
    "reviewer": "critic", "reviewers": "critics",
    "review": "assessment", "reviews": "assessments",
    "decision": "verdict", "decisions": "verdicts",
    "email": "mail_address",
    "date": "day",
    "user": "account", "users": "accounts",
    "document": "file", "documents": "files",
    "chair": "presider", "chairs": "presiders", "chairmans": "presidents",
    "member": "participant", "members": "participants",
    "committee": "board", "committees": "boards",
    "program": "plan",
    "subject": "topic", "area": "field",
    "bid": "offer", "bids": "offers",
    "administrator": "manager", "administrators": "managers",
    "accepted": "approved", "rejected": "declined",
    "comment": "remark",
    "site": "homepage", "url": "link",
    "title": "heading",
    "abstract": "summary",
    "logo": "emblem",
    "label": "tag",
}

SCHEMA_ABBREVIATIONS: dict[str, str] = {
    "administrator": "admin", "administrators": "admins",
    "author": "auth", "authors": "auths",
    "conference": "cnf", "conferences": "cnfs",  # "conf" already appears in the RODI schema
    "committee": "cmte", "committees": "cmtes",
    "member": "mbr", "members": "mbrs",
    "program": "prog",
    "paper": "pap", "papers": "paps",
    "review": "rvw", "reviews": "rvws",
    "reviewer": "rvwr", "reviewers": "rvwrs",
    "person": "prsn", "persons": "prsns",
    "document": "doc", "documents": "docs",
    "decision": "dec", "decisions": "decs",
    "subject": "subj",
    "chair": "chr", "chairs": "chrs", "chairmans": "chms",
    "preference": "pref", "preferences": "prefs",
    "accepted": "acc", "rejected": "rej",
    "assigned": "asgn",
    "conflict": "cnfl",
    "email": "eml",
    "description": "desc",
    "number": "num",
}

PERTURBATIONS: dict[str, dict[str, str]] = {
    "schema_synonym": SCHEMA_SYNONYMS,
    "schema_abbreviation": SCHEMA_ABBREVIATIONS,
}

# Words that must not become identifiers in the generated schema.
RESERVED_WORDS = {
    "all", "and", "asc", "check", "column", "default", "desc", "end", "from", "group",
    "limit", "offset", "order", "select", "table", "user", "where", "year", "time",
}
_VALID_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_SQL_TOKEN = re.compile(r"'(?:[^']|'')*'|\"[^\"]*\"|\b[A-Za-z_][A-Za-z0-9_]*\b")


def _rename_identifier(name: str, dictionary: dict[str, str]) -> str:
    if name != name.lower():
        return name
    return "_".join(dictionary.get(token, token) for token in name.split("_"))


def schema_identifiers(dump: Path) -> set[str]:
    tables = parse_sql_dump(dump)
    names = set(tables)
    for table in tables.values():
        names.update(column.name for column in table.columns)
    return names


def build_identifier_map(identifiers: Iterable[str], dictionary: dict[str, str]) -> dict[str, str]:
    """Return original -> new name for identifiers that change.

    Raises ValueError if the mapping is not injective, produces an invalid or
    reserved identifier, or changes nothing.
    """
    originals = sorted(set(identifiers))
    mapping = {name: _rename_identifier(name, dictionary) for name in originals}
    owners: dict[str, list[str]] = {}
    for old, new in mapping.items():
        owners.setdefault(new, []).append(old)
    collisions = {new: olds for new, olds in owners.items() if len(olds) > 1}
    if collisions:
        raise ValueError(f"identifier rename collides: {collisions}")
    for new in owners:
        if not _VALID_IDENT.match(new) or new in RESERVED_WORDS:
            raise ValueError(f"invalid perturbed identifier: {new!r}")
    changed = {old: new for old, new in mapping.items() if old != new}
    if not changed:
        raise ValueError("perturbation changes no identifiers in this scenario")
    return changed


def rewrite_sql(sql: str, mapping: dict[str, str]) -> str:
    """Rename schema identifiers in SQL, leaving string literals untouched."""

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("'"):
            return token
        if token.startswith('"'):
            inner = token[1:-1]
            return '"' + mapping.get(inner, inner) + '"'
        if token in mapping:
            return mapping[token]
        # Unquoted SQL identifiers are case-insensitive in PostgreSQL.
        return mapping.get(token.lower(), token)

    return _SQL_TOKEN.sub(repl, sql)


_COLUMN_DEF = re.compile(r'^(\s+)("[^"]+"|\w+)(\s.*)?$')
_TABLE_CONSTRAINT = re.compile(r"\s*(CONSTRAINT|PRIMARY|FOREIGN|UNIQUE|CHECK|EXCLUDE)\b", re.I)


def rewrite_dump(text: str, mapping: dict[str, str]) -> str:
    """Rename identifiers in a pg_dump text, skipping COPY data rows.

    In CREATE TABLE column definitions only the column name is renamed. The type
    is left alone, so a column called ``date`` keeps the type ``date``.
    """
    out: list[str] = []
    in_copy = False
    in_create = False
    for line in text.split("\n"):
        if in_copy:
            if line == r"\.":
                in_copy = False
            out.append(line)
            continue
        if re.match(r"COPY\s", line, re.I) and line.rstrip().endswith("FROM stdin;"):
            in_copy = True
        if re.match(r"CREATE\s+TABLE\b", line, re.I):
            in_create = True
        elif in_create and line.startswith(")"):
            in_create = False
        match = _COLUMN_DEF.match(line) if in_create else None
        if match and not _TABLE_CONSTRAINT.match(line):
            name = match.group(2)
            if name.startswith('"'):
                new = '"' + mapping.get(name[1:-1], name[1:-1]) + '"'
            else:
                new = mapping.get(name, name)
            out.append(match.group(1) + new + (match.group(3) or ""))
            continue
        out.append(rewrite_sql(line, mapping))
    return "\n".join(out)


_SQL_SECTION = re.compile(
    r"(?ms)^(\s*sql\s*=)(.*?)(?=^\s*(?:" + "|".join(f for f in QPAIR_FIELDS if f != "sql") + r")\s*=|\Z)"
)


def rewrite_qpair_text(text: str, mapping: dict[str, str]) -> str:
    match = _SQL_SECTION.search(text)
    if not match:
        return text
    sql = rewrite_sql(clean_qpair_value(match.group(2)), mapping)
    return text[: match.start()] + f"{match.group(1)}{sql}\n\n" + text[match.end():]


def create_perturbed_scenario(
    rodi_root: Path,
    scenario: str,
    out_root: Path,
    perturbation: str,
) -> dict[str, Any]:
    """Write ``out_root/data/<scenario>__<perturbation>`` and return its manifest."""
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturbation!r}; expected one of {sorted(PERTURBATIONS)}")
    src = scenario_source(rodi_root, scenario)
    dump = src / "dump.sql"
    mapping = build_identifier_map(schema_identifiers(dump), PERTURBATIONS[perturbation])

    name = f"{scenario}__{perturbation}"
    dst = Path(out_root) / "data" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("dump.sql", "queries", ".DS_Store"))

    dump_text = dump.read_text(encoding="utf-8", errors="replace")
    # Write LF explicitly to match the RODI source dumps (Windows text mode would write CRLF).
    with open(dst / "dump.sql", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(rewrite_dump(dump_text, mapping))

    qdir = src / "queries"
    if qdir.exists():
        (dst / "queries").mkdir()
        for qpair in sorted(qdir.glob("*.qpair")):
            text = qpair.read_text(encoding="utf-8", errors="replace")
            with open(dst / "queries" / qpair.name, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(rewrite_qpair_text(text, mapping))

    manifest = {
        "scenario": scenario,
        "perturbation": perturbation,
        "name": name,
        "renamed_identifiers": len(mapping),
        "mapping": mapping,
    }
    (dst / "dr_spider_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest

# Report Section 3.1.1 proposes "a few irrelevant elements added". This is not one of Dr.Spider's own
# perturbations. Added columns are NULL-valued, and their names carry the table name, e.g. papers_remarks.
# A name shared by two tables would become a join key for NATURAL JOIN and USING queries and change the
# reference answers, so the names must be unique per table.
DISTRACTOR_SUFFIXES: tuple[str, ...] = ("remarks", "status_code", "internal_note", "legacy_flag")
DISTRACTORS_PER_TABLE = 2
DISTRACTOR_PERTURBATION = "distractor_columns"


def distractor_plan(tables: dict[str, Table]) -> dict[str, list[str]]:
    """Choose ``DISTRACTORS_PER_TABLE`` free, table-qualified names per table."""
    plan: dict[str, list[str]] = {}
    for name in sorted(tables):
        taken = set(tables[name].column_names())
        free = [f"{name}_{s}" for s in DISTRACTOR_SUFFIXES if f"{name}_{s}" not in taken]
        if len(free) < DISTRACTORS_PER_TABLE:
            raise ValueError(f"table {name} has too few free distractor names: {free}")
        plan[name] = free[:DISTRACTORS_PER_TABLE]
    return plan


def create_distractor_scenario(rodi_root: Path, scenario: str, out_root: Path) -> dict[str, Any]:
    """Write ``out_root/data/<scenario>__distractor_columns`` with NULL-valued distractor columns."""
    src = scenario_source(rodi_root, scenario)
    dump_path = src / "dump.sql"
    text = dump_path.read_text(encoding="utf-8", errors="replace")
    tables = parse_sql_dump(dump_path)
    lines = text.split("\n")
    copies = scan_copies(lines)
    plan = distractor_plan(tables)
    for name, extra in plan.items():
        block = copies.get(name)
        if block is not None:
            for row in block.rows:
                row.extend([None] * len(extra))
            block.header = block.header + extra
    rendered = render_dump(lines, copies, plan)

    name = f"{scenario}__{DISTRACTOR_PERTURBATION}"
    dst = Path(out_root) / "data" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("dump.sql", ".DS_Store"))
    with open(dst / "dump.sql", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(rendered)
    manifest = {
        "scenario": scenario,
        "perturbation": DISTRACTOR_PERTURBATION,
        "name": name,
        "added_columns": plan,
    }
    (dst / "dr_spider_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest



def load_scenario_f1(metrics_details: Path) -> float:
    """Mean query F1 from an evaluate_graph ``metrics_details.json`` file."""
    rows = json.loads(Path(metrics_details).read_text(encoding="utf-8"))
    scored = [float(row["f1"]) for row in rows if row.get("f1") is not None]  # N/A queries have f1 = None
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def score_robustness(base_f1: float, perturbed_f1: dict[str, float]) -> dict[str, Any]:
    per: dict[str, dict[str, Optional[float]]] = {}
    retentions: list[float] = []
    for name, f1 in sorted(perturbed_f1.items()):
        retention = f1 / base_f1 if base_f1 > 0 else None
        per[name] = {"f1": f1, "drop": base_f1 - f1, "retention": retention}
        if retention is not None:
            retentions.append(retention)
    return {
        "base_f1": base_f1,
        "perturbations": per,
        "mean_perturbed_f1": sum(perturbed_f1.values()) / len(perturbed_f1) if perturbed_f1 else None,
        "mean_retention": sum(retentions) / len(retentions) if retentions else None,
        "min_retention": min(retentions) if retentions else None,
    }


def _cmd_create(args: argparse.Namespace) -> None:
    names = args.perturbations or sorted(PERTURBATIONS)
    base = copy_base_scenario(Path(args.rodi_root), args.scenario, Path(args.out_root))
    print(f"{args.scenario}: copied unperturbed scenario to {base}")
    for perturbation in names:
        manifest = create_perturbed_scenario(Path(args.rodi_root), args.scenario, Path(args.out_root), perturbation)
        print(f"{manifest['name']}: renamed {manifest['renamed_identifiers']} identifiers")
    if args.with_distractors:
        manifest = create_distractor_scenario(Path(args.rodi_root), args.scenario, Path(args.out_root))
        print(f"{manifest['name']}: added {DISTRACTORS_PER_TABLE} NULL columns per table")


def _cmd_score(args: argparse.Namespace) -> None:
    base = load_scenario_f1(Path(args.base))
    runs: dict[str, float] = {}
    for item in args.run:
        key, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects NAME=metrics_details.json, got {item!r}")
        runs[key] = load_scenario_f1(Path(path))
    report = score_robustness(base, runs)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Dr.Spider-style schema robustness for FGF (RODI scenarios).")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Write perturbed RODI scenario copies.")
    create.add_argument("--rodi-root", required=True, help="Directory containing data/<scenario>.")
    create.add_argument("--scenario", required=True)
    create.add_argument("--out-root", required=True, help="Root that will be passed as --rodi-root to the pipeline.")
    create.add_argument("--perturbations", nargs="*", choices=sorted(PERTURBATIONS))
    create.add_argument(
        "--with-distractors",
        action="store_true",
        help="Also write the distractor_columns scenario (report Section 3.1.1 idea; not a Dr.Spider perturbation).",
    )
    create.set_defaults(func=_cmd_create)

    score = sub.add_parser("score", help="Compute robustness from evaluation metrics_details.json files.")
    score.add_argument("--base", required=True, help="metrics_details.json of the unperturbed scenario.")
    score.add_argument("--run", action="append", default=[], help="NAME=metrics_details.json; repeatable.")
    score.add_argument("--output", help="Optional JSON output path.")
    score.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
