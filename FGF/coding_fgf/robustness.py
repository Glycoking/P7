"""Shared scoring for the optional robustness extensions (Dr.Spider and KROWN).

This module is not imported by the main FGF pipeline. It is used by:

* ``coding_fgf.dr_spider`` -- schema-level Dr.Spider perturbations (synonym, abbreviation)
  and the distractor-column variant.
* ``coding_fgf.krown``     -- KROWN-style data-shape perturbations (rows, nulls, duplicates, columns).

Both families write perturbed RODI scenarios with the same layout, so the pipeline runs
them unchanged. This module then compares each perturbed run with base runs, query by query.

Scoring rules
-------------
* **Vacuous query.** A query whose reference result and prediction are both empty gets
  F1 = 1.0 from ``evaluate.calculate_precision_recall_f1``. That is not evidence of
  correctness. Such a query is marked vacuous and excluded from every paired metric
  (R_rel, accuracies, McNemar, bootstrap, F1 deltas). Excluded query ids are listed, and
  if no query is scoreable the metric is ``null`` with a ``status`` explaining why (N/A).
  The main pipeline's own F1 summary is unchanged.
* **Correctness.** A scored query is correct when F1 >= threshold (default 1.0).
* **R_rel (report Eq. 3.1).** #{correct before and after} / #{correct before}, over scored queries.
* **Reference-row retention.** Sum of reference answer rows after perturbation divided by
  the sum before, over queries where both sizes are known. This shows how much of the answer
  set survives the perturbation. It is not data retention.
* **Repeats and noise.** With several base runs and perturbed runs, every base/perturbed pair
  is compared. Base-vs-base comparisons are a null perturbation, so their spread is the noise
  floor. An effect is "within noise" when its median F1 delta is below that floor. Without at
  least two base runs the noise is N/A.

Run-level failure stages (report RQ3), first matching stage wins:

* ``no_program``      -- no generated FGF code was produced.
* ``crash``           -- materialization did not finish. ``sub_cause`` says why:
                         ``rejected`` (sandbox policy), ``runtime`` (exception in generated code),
                         ``timeout``, ``memory``, ``platform``, ``no_log``, or ``other``.
* ``invalid_rdf``     -- materialization finished but emitted invalid RDF triples.
* ``valid_but_wrong`` -- valid graph, at least one scored query below threshold.
* ``success``         -- valid graph, every scored query at threshold.
* ``unevaluated``     -- valid graph but no evaluation output (incomplete run).
* ``unscored``        -- evaluated, but every query is vacuous, so nothing can be judged (N/A).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

STAGES = ("no_program", "crash", "invalid_rdf", "valid_but_wrong", "success")
DEFAULT_THRESHOLD = 1.0
GENERATED_CODE = "generated_fgf.py"
MATERIALIZATION_LOG = "import.materialization_log.json"
METRICS_RELPATH = Path("eval") / "metrics_details.json"
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Scenario copying (shared by Dr.Spider and KROWN generators)
# ---------------------------------------------------------------------------

def scenario_source(rodi_root: Path, scenario: str) -> Path:
    for candidate in (rodi_root / "data" / scenario, rodi_root / scenario):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find scenario {scenario} below {rodi_root}")


def copy_base_scenario(rodi_root: Path, scenario: str, out_root: Path) -> Path:
    """Copy the unperturbed scenario under out_root/data so one mount serves all runs."""
    src = scenario_source(rodi_root, scenario)
    dst = Path(out_root) / "data" / scenario
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Per-query records
# ---------------------------------------------------------------------------

def is_vacuous(row: Mapping[str, Any]) -> bool:
    """N/A query: flagged by the evaluator, or F1 is null, or both result lists are empty.

    Missing result fields alone are not vacuous, so older files without them are still scored.
    """
    if row.get("vacuous") is True or row.get("f1") is None:
        return True
    if "sql_results" not in row or "sparql_results" not in row:
        return False
    return not row["sql_results"] and not row["sparql_results"]


def reference_size(row: Mapping[str, Any]) -> Optional[int]:
    value = row.get("sql_results")
    return len(value) if isinstance(value, list) else None


def load_query_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``metrics_details.json`` keyed by qpair id. Accepts a run dir or the file itself."""
    file_path = path / METRICS_RELPATH if path.is_dir() else path
    rows = json.loads(file_path.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["id"])
        if qid in by_id:
            raise ValueError(f"duplicate query id {qid} in {file_path}")
        by_id[qid] = row
    return by_id


def mean_f1(rows: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    """Mean F1 over scored (non-vacuous) queries, matching the evaluator's own average."""
    values = [float(r["f1"]) for r in rows.values() if not is_vacuous(r)]
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class PairedQuery:
    qid: str
    name: str
    base_f1: float  # 0.0 placeholder when vacuous; never used, because vacuous queries are not scored
    pert_f1: float
    base_vacuous: bool
    pert_vacuous: bool
    base_ref: Optional[int]
    pert_ref: Optional[int]
    threshold: float = DEFAULT_THRESHOLD

    @property
    def scored(self) -> bool:
        return not self.base_vacuous and not self.pert_vacuous

    @property
    def pre(self) -> bool:
        return self.base_f1 >= self.threshold - _EPS

    @property
    def post(self) -> bool:
        return self.pert_f1 >= self.threshold - _EPS


def pair_queries(
    base: Mapping[str, Mapping[str, Any]],
    pert: Mapping[str, Mapping[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[PairedQuery]:
    """Pair base and perturbed results by qpair id. Both runs must cover the same queries."""
    if set(base) != set(pert):
        mismatch = sorted(set(base) ^ set(pert))
        raise ValueError(f"base and perturbed runs evaluate different queries: {mismatch}")
    return [
        PairedQuery(
            qid=qid,
            name=str(base[qid].get("name", qid)),
            base_f1=float(base[qid]["f1"]) if base[qid].get("f1") is not None else 0.0,
            pert_f1=float(pert[qid]["f1"]) if pert[qid].get("f1") is not None else 0.0,
            base_vacuous=is_vacuous(base[qid]),
            pert_vacuous=is_vacuous(pert[qid]),
            base_ref=reference_size(base[qid]),
            pert_ref=reference_size(pert[qid]),
            threshold=threshold,
        )
        for qid in sorted(base)
    ]


# ---------------------------------------------------------------------------
# Paired metrics (scored queries only)
# ---------------------------------------------------------------------------

def relative_robustness(scored: list[PairedQuery]) -> dict[str, Any]:
    """Eq. 3.1 of the report, plus the pre/post accuracies it is built from. N/A when undefined."""
    n = len(scored)
    pre = [p for p in scored if p.pre]
    both = sum(1 for p in pre if p.post)
    if n == 0:
        return {
            "n_scored": 0,
            "n_pre_correct": 0,
            "n_pre_and_post_correct": 0,
            "pre_accuracy": None,
            "post_accuracy": None,
            "relative_robustness": None,
            "status": "N/A: no scored queries",
        }
    status = "ok" if pre else "N/A: no query is correct before perturbation"
    return {
        "n_scored": n,
        "n_pre_correct": len(pre),
        "n_pre_and_post_correct": both,
        "pre_accuracy": len(pre) / n,
        "post_accuracy": sum(1 for p in scored if p.post) / n,
        "relative_robustness": both / len(pre) if pre else None,
        "status": status,
    }


def mcnemar_exact_p(scored: list[PairedQuery]) -> Optional[float]:
    """Exact two-sided McNemar test on discordant pairs. None when there are no scored queries."""
    if not scored:
        return None
    b = sum(1 for p in scored if p.pre and not p.post)
    c = sum(1 for p in scored if not p.pre and p.post)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def bootstrap_mean_diff_ci(
    scored: list[PairedQuery], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> tuple[Optional[float], Optional[float]]:
    """Paired bootstrap percentile CI for mean(pert_f1 - base_f1) over scored queries."""
    if not scored:
        return None, None
    diffs = [p.pert_f1 - p.base_f1 for p in scored]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


# ---------------------------------------------------------------------------
# Run-level failure stage
# ---------------------------------------------------------------------------

def crash_subcause(log: Mapping[str, Any]) -> str:
    status = log.get("status")
    error = str(log.get("error") or "")
    if status == "timeout":
        return "timeout"
    if status == "memory_exhausted":
        return "memory"
    if status == "worker_error":
        return "runtime"
    if status == "unsupported_platform":
        return "platform"
    if status == "error" and error.startswith("SandboxViolation"):
        return "rejected"
    return "other"


def classify_run(run_dir: Path, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Assign one report stage to a pipeline run directory (``runs/<scenario>``)."""
    code_path = run_dir / GENERATED_CODE
    if not code_path.exists() or not code_path.read_text(encoding="utf-8").strip():
        return {"stage": "no_program", "reason": "no generated FGF code"}

    log_path = run_dir / MATERIALIZATION_LOG
    if not log_path.exists():
        return {"stage": "crash", "sub_cause": "no_log", "reason": "generated code exists but no materialization log"}
    log = json.loads(log_path.read_text(encoding="utf-8"))
    status = log.get("status")
    invalid = int(log.get("invalid_triple_count") or 0)
    if status == "invalid_triples" or invalid > 0:
        return {"stage": "invalid_rdf", "reason": f"{invalid} invalid RDF triple(s)"}
    if status != "success":
        return {
            "stage": "crash",
            "sub_cause": crash_subcause(log),
            "reason": f"{status}: {log.get('error')}",
        }

    metrics = run_dir / METRICS_RELPATH
    if not metrics.exists():
        return {"stage": "unevaluated", "reason": "graph materialized but no evaluation output"}
    rows = load_query_rows(metrics)
    scored = {qid: row for qid, row in rows.items() if not is_vacuous(row)}
    vacuous = sorted(qid for qid in rows if qid not in scored)
    if not scored:
        return {
            "stage": "unscored",
            "reason": "N/A: every query has an empty reference and an empty prediction",
            "vacuous_queries": vacuous,
        }
    failed = sorted(qid for qid, row in scored.items() if float(row["f1"]) < threshold - _EPS)
    if failed:
        return {
            "stage": "valid_but_wrong",
            "reason": f"{len(failed)}/{len(scored)} scored queries below threshold",
            "failed_queries": failed,
            "vacuous_queries": vacuous,
        }
    return {
        "stage": "success",
        "reason": f"all {len(scored)} scored queries at threshold",
        "vacuous_queries": vacuous,
    }


# ---------------------------------------------------------------------------
# Comparisons and aggregation
# ---------------------------------------------------------------------------

def compare_run(base_dir: Path, pert_dir: Path, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    base_rows = load_query_rows(base_dir)
    pert_rows = load_query_rows(pert_dir)
    pairs = pair_queries(base_rows, pert_rows, threshold)
    scored = [p for p in pairs if p.scored]
    excluded = [p.qid for p in pairs if not p.scored]

    base_scored = mean(p.base_f1 for p in scored)
    pert_scored = mean(p.pert_f1 for p in scored)
    f1_delta = pert_scored - base_scored if scored else None
    retention = pert_scored / base_scored if scored and base_scored > 0 else None
    lo, hi = bootstrap_mean_diff_ci(scored)

    ref_pairs = [p for p in pairs if p.base_ref is not None and p.pert_ref is not None]
    base_ref_total = sum(p.base_ref for p in ref_pairs)
    pert_ref_total = sum(p.pert_ref for p in ref_pairs)

    rr = relative_robustness(scored)
    return {
        "run_dir": str(pert_dir),
        "stage": classify_run(pert_dir, threshold),
        "n_queries": len(pairs),
        "n_scored": len(scored),
        "excluded_vacuous_qids": excluded,
        "mean_f1_scored_base": base_scored,
        "mean_f1_scored_pert": pert_scored,
        "f1_delta_scored": f1_delta,
        "f1_delta_ci95": [lo, hi] if scored else None,
        "f1_retention_scored": retention,
        "reference_row_retention": pert_ref_total / base_ref_total if base_ref_total > 0 else None,
        "mcnemar_p": mcnemar_exact_p(scored),
        **rr,
        "pass_to_fail": [p.qid for p in scored if p.pre and not p.post],
        "fail_to_pass": [p.qid for p in scored if not p.pre and p.post],
    }


def mean(values) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(values: list[Optional[float]]) -> dict[str, Any]:
    defined = [v for v in values if v is not None]
    if not defined:
        return {"n": 0, "median": None, "min": None, "max": None, "status": "N/A: no defined values"}
    return {
        "n": len(defined),
        "median": statistics.median(defined),
        "min": min(defined),
        "max": max(defined),
        "status": "ok",
    }


def noise_floor(base_dirs: list[Path], threshold: float) -> dict[str, Any]:
    """Compare base runs with each other. Every such comparison is a null perturbation."""
    if len(base_dirs) < 2:
        return {"status": "N/A: needs at least two base runs to estimate noise"}
    null = [compare_run(bi, bj, threshold) for i, bi in enumerate(base_dirs) for j, bj in enumerate(base_dirs) if i != j]
    deltas = [abs(r["f1_delta_scored"]) for r in null if r["f1_delta_scored"] is not None]
    rr_values = [r["relative_robustness"] for r in null if r["relative_robustness"] is not None]
    return {
        "status": "ok" if deltas else "N/A: no scored queries in base comparisons",
        "n_null_comparisons": len(null),
        "max_abs_f1_delta": max(deltas) if deltas else None,
        "relative_robustness_range": [min(rr_values), max(rr_values)] if rr_values else None,
    }


def build_report(
    base_dirs: list[Path], runs: Mapping[str, list[Path]], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    if not base_dirs:
        raise ValueError("at least one base run is required")
    base_summaries = []
    for base in base_dirs:
        rows = load_query_rows(base)
        scored = [r for r in rows.values() if not is_vacuous(r)]
        base_summaries.append(
            {
                "run_dir": str(base),
                "stage": classify_run(base, threshold),
                "n_queries": len(rows),
                "n_scored": len(scored),
                "n_pass_scored": sum(1 for r in scored if float(r["f1"]) >= threshold - _EPS),
                "mean_f1_scored": mean_f1(rows),
            }
        )

    noise = noise_floor(base_dirs, threshold)
    perturbations: dict[str, Any] = {}
    for name, dirs in runs.items():
        comparisons = [compare_run(base, pert, threshold) for base in base_dirs for pert in dirs]
        deltas = [c["f1_delta_scored"] for c in comparisons]
        median_delta = summarize(deltas)["median"]
        if "max_abs_f1_delta" not in noise or noise.get("max_abs_f1_delta") is None:
            verdict = "N/A: noise floor not estimated"
        elif median_delta is None:
            verdict = "N/A: no scored queries"
        else:
            verdict = "exceeds noise" if abs(median_delta) > noise["max_abs_f1_delta"] else "within noise"
        perturbations[name] = {
            "summary": {
                "n_comparisons": len(comparisons),
                "relative_robustness": summarize([c["relative_robustness"] for c in comparisons]),
                "f1_delta_scored": summarize(deltas),
                "reference_row_retention": summarize([c["reference_row_retention"] for c in comparisons]),
                "stage_counts": dict(Counter(c["stage"]["stage"] for c in comparisons)),
                "effect_vs_noise": verdict,
            },
            "comparisons": comparisons,
        }
    return {
        "threshold": threshold,
        "base": base_summaries,
        "noise_floor": noise,
        "perturbations": perturbations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_score(args: argparse.Namespace) -> None:
    runs: dict[str, list[Path]] = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects NAME=run_dir, got {item!r}")
        runs.setdefault(name, []).append(Path(path))
    report = build_report([Path(b) for b in args.base], runs, args.threshold)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


def _cmd_stage(args: argparse.Namespace) -> None:
    print(json.dumps(classify_run(Path(args.run_dir), args.threshold), indent=2))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Robustness scoring for Dr.Spider and KROWN perturbations.")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Compare perturbed pipeline runs with base runs.")
    score.add_argument("--base", action="append", required=True,
                       help="Pipeline run dir (runs/<scenario>) of the unperturbed scenario; repeat for repeated runs.")
    score.add_argument("--run", action="append", default=[],
                       help="NAME=run_dir; repeat with the same NAME for repeated runs of one perturbation.")
    score.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="F1 needed for a scored query to count as correct.")
    score.add_argument("--output", help="Optional JSON output path.")
    score.set_defaults(func=_cmd_score)

    stage = sub.add_parser("stage", help="Print the failure stage of one pipeline run dir.")
    stage.add_argument("--run-dir", required=True)
    stage.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    stage.set_defaults(func=_cmd_stage)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
