# Robustness extensions: Dr.Spider, KROWN and scoring

This covers the optional robustness extensions (report research questions 2–4). They are kept outside the main pipeline: nothing in `coding_fgf` imports them.

## Modules

| Module | Role | Source of the idea |
|---|---|---|
| `coding_fgf/dr_spider.py` | Schema-identifier perturbations `schema_synonym`, `schema_abbreviation`; plus `distractor_columns` (opt-in) | Dr.Spider (Chang et al., ICLR 2023); distractors from report §3.1.1 |
| `coding_fgf/krown.py` | Data-shape perturbations, one knob at a time: `rows`, `nulls`, `duplicates`, `columns` | KROWN (Van Assche et al., ISWC 2024) |
| `coding_fgf/robustness.py` | Scoring for both families | Report Eq. 3.1 (R_rel), RQ3 failure stages |
| `coding_fgf/dump_edit.py` | Shared pg_dump editing used by the two generators | — |

Every generator writes `<OUT>/data/<name>/`, so the pipeline runs the scenarios unchanged.

## Workflow

```bash
# 1. Generate perturbed scenarios
python -m coding_fgf.dr_spider create --rodi-root ../RODI --scenario cmt_renamed --out-root <OUT> --with-distractors
python -m coding_fgf.krown    create --rodi-root ../RODI --scenario cmt_renamed --out-root <OUT>
python -m coding_fgf.krown    create --rodi-root ../RODI --scenario cmt_renamed --out-root <OUT> --spec rows:0.5 --spec columns:8

# 2. Run the pipeline on the base and each perturbed scenario. configs/krown_cheap.yaml lists the default KROWN scenarios.
DR_SPIDER_DIR=<OUT> docker compose -f docker-compose.yml -f docker-compose.dr_spider.yml \
  run --rm fgf-pipeline python scripts/run_pipeline.py --config configs/krown_cheap.yaml

# 3. Score. Repeat --base and --run NAME=... for repeated runs.
python -m coding_fgf.robustness score \
  --base <work>/runs/cmt_renamed --base <work2>/runs/cmt_renamed \
  --run cmt_renamed__krown_nulls_p20=<work>/runs/cmt_renamed__krown_nulls_p20 \
  --output robustness.json

python -m coding_fgf.robustness stage --run-dir <work>/runs/cmt_renamed__krown_rows_p50
```

## Scoring rules

**Vacuous queries are N/A, not passes.** `evaluate.calculate_precision_recall_f1` returns F1 = 1.0 when the reference and the prediction are both empty. That is no evidence of correctness. In robustness scoring such a query is marked vacuous and excluded from every paired metric. The excluded ids are listed per run. If no query is scoreable, the metric is `null` and its `status` says `N/A: ...`. The same rule now applies in the main evaluator (`coding_fgf/evaluate.py`): a both-empty query gets `precision`, `recall` and `f1` = `null`, `vacuous: true` and a `score_status` starting `N/A`. The run-level average covers scored queries only, and the summary reports `scored_count` and `vacuous_count`. Consumers handle `null`: the benchmark failure list skips N/A queries, the failure-analysis script exempts them from numeric checks, and the CLI prints `N/A`.

**Correctness** of a scored query is F1 ≥ `--threshold` (default 1.0).

**R_rel (Eq. 3.1)** is the fraction of queries correct before perturbation that are still correct afterwards, over scored queries.

**Reference-row retention** is the total number of reference answer rows after perturbation divided by the total before, over queries where both are known. It shows how much of the answer set survives. It is not data retention. The per-table data retention is in each KROWN manifest.

**Repeats and noise.** With several base runs, each base/perturbed pair is compared. Base-vs-base comparisons are a null perturbation, so their spread is the noise floor. Each arm gets `effect_vs_noise`: "exceeds noise" or "within noise" when the median F1 delta is beyond the noise floor. With a single base run the noise is `N/A`.

**Failure stages (RQ3).** The first matching stage wins:

| Stage | Condition |
|---|---|
| `no_program` | `generated_fgf.py` missing or blank |
| `crash` | Materialization log status `timeout`, `memory_exhausted`, `worker_error`, `unsupported_platform`, `error`; or code present but no log |
| `invalid_rdf` | Log status `invalid_triples`, or invalid triple count > 0 |
| `valid_but_wrong` | Valid graph, at least one scored query below threshold |
| `success` | Valid graph, every scored query at threshold |
| `unevaluated` | Valid graph, no evaluation output (incomplete run; outside the five stages) |
| `unscored` | Evaluated, but every query is vacuous (N/A; outside the five stages) |

`crash` carries a `sub_cause` so the headline count stays actionable: `rejected` (sandbox policy violation), `runtime` (exception in generated code), `timeout`, `memory`, `platform`, `no_log`, or `other`.

## Decisions taken

| Question | Choice | Why |
|---|---|---|
| Empty-on-both-sides queries | Marked vacuous, excluded from paired metrics, shown as N/A | A both-empty query scores 1.0 by construction, so counting it as a pass would fake robustness |
| `rows` knob | Keep uniform sampling plus FK pruning; report realised retention; flag vacuous queries | Anchor sampling is ill-defined on RODI's FK cycles, where one anchor's closure can cover most of the database |
| `columns` filler values | NULL, with table-qualified names | NULL keeps content unchanged, so only width varies. Table-qualified names stop them becoming join keys (see below) |
| `duplicates` | Value-level, following KROWN's "duplicate values" | Keys are untouched, so instance identity does not change |
| Distractor columns | Separate opt-in perturbation in the Dr.Spider module, NULL-valued, table-qualified names (`papers_remarks`) | Report §3.1.1's "irrelevant elements" idea. It is not one of Dr.Spider's own perturbations, so it is labelled as such |
| Sandbox-rejected code | `crash` with `sub_cause=rejected` | Keeps the five report stages while separating a policy failure from a runtime bug |
| Repeats | Supported in scoring (repeated `--base` and `--run`). Not automated: you choose the count | Running the pipeline costs API money, so the runs are yours to start |
| Local Dr.Spider data | Regenerated with LF line endings | Matches the RODI source format. Content identical except for line endings |

## Finding: join keys in the RODI schema

Seven `cmt_renamed` queries (Q32, Q34, Q35, Q36, Q41, Q42, Q43) use `NATURAL JOIN` or `USING`. They join on every shared column name. A column name shared by two tables is therefore a join key.

A first version of both the distractor and the filler perturbations used shared names such as `remarks` and `krown_filler_1`. That silently changed those seven queries: their reference answers became empty. The fix is table-qualified names, which can never be shared. The invariant is tested on the real scenario: the set of shared column names is identical before and after each perturbation (`test_real_columns_keep_shared_names`, `test_real_distractors_do_not_add_join_keys`).

The renames (`schema_synonym`, `schema_abbreviation`) keep join keys intact because each old name maps to one new name everywhere.

## KROWN knobs

Each scenario changes one knob at one level. `DATA_SEED` is fixed (0), so a given spec always gives the same database.

- **`rows` (fraction in (0, 1]).** Each table is sampled to that fraction. Then rows whose non-NULL FK values have no surviving parent are removed, repeating until nothing changes. This also handles FK cycles. Removals cascade. On `cmt_renamed`, `rows:0.5` keeps about 20% of rows (1835 of 9123), and `rows:0.25` about 8%. At `rows:0.25`, three reference answers are empty on the data side (Q32, Q34, Q38). The manifest records before, after-sampling and after counts per table.
- **`nulls` (fraction in (0, 1)).** Cells in nullable, non-key columns become NULL. PK, FK and UNIQUE columns are never touched. A reference row with a NULL is not an answer, so it is dropped (see the pilot result).
- **`duplicates` (fraction in (0, 1)).** Cells in non-key, non-unique columns are replaced by a value from the same column. Rows are not duplicated, because that would break primary keys.
- **`columns` (integer 1–50).** Adds `k` NULL-valued text columns to every table, named `<table>_krown_filler_<i>`. This is KROWN's "number of columns" knob. It changes width only.

Not ported: KROWN's join-count knob (RODI schemas are fixed; join count would need restructuring) and cell-size knob.

Reference SQL is executed live: `evaluate_graph` runs each `.qpair` SQL against the loaded database. The `queries/` directory is copied unchanged, so references follow the perturbed data.

## Verification

- **Tests:** `tests/test_robustness.py`, `tests/test_krown.py`, `tests/test_dr_spider.py` — 81 passing in the last run. They cover integrity, determinism, NULL and duplicate constraints, filler width, LF output, the R_rel arithmetic, McNemar and bootstrap values, vacuous exclusion, N/A cases, every crash sub-cause, repeat aggregation and noise floors, and the join-key invariant on the real scenario.
- **PostgreSQL 15** loads every dump of the default grid (base, 8 KROWN, 2 Dr.Spider incl. distractors) with `ON_ERROR_STOP=1`, enforcing the dump's own PK, FK and UNIQUE constraints. All exit 0 with 48 tables.
- **Reference SQL:** all 29 queries run without error on every database. A negative control (original SQL on the renamed database) gives 29 errors.
- **Saved runs:** `coding_fgf.robustness score` on `outputs/dr_spider_cheap`: base 24/29 pass, synonym run 16/29, R_rel 0.667, McNemar p = 0.0078, all 8 changes pass→fail. None of the saved runs contains a vacuous query, so the vacuous rule does not change their numbers.
- **Full FGF suite:** 349 passed, 31 failed. The failures come from this host: `openai` and `google.auth` are not installed in the Python used here, and the sandbox needs Linux. None of the failing tests touch these modules. The Docker image is the intended runner and was not used.

## Pilot result (one arm, live run)

Base `cmt_renamed` and arm `cmt_renamed__krown_nulls_p20`, same cheap config (`outputs/krown_pilot`).

- Base: F1 0.8276 (24/29 pass). Arm as originally scored: F1 0.7763, R_rel 0.625, McNemar p = 0.0039, nine pass-to-fail queries.
- Cause: the reference SQL returned the string `None` for every NULL cell. RDF has no triple for an unknown value, so the mapping correctly emitted nothing, and the scorer counted the missing `None` as a miss. Every drop came from this.
- Fix: `execute_sql_psycopg2` now drops any reference row containing a NULL. Base reference answers contain no NULL, so the baseline is unchanged.
- Offline re-score of the same arm (stored SPARQL and SQL results, no new API calls): F1 0.8276, identical to base.

The pilot did not record token usage or cost. Run metadata contains no LLM events, so no spend figure is available from the outputs. Read the cost from the OpenAI usage dashboard for the pilot window, or add usage persistence before the full grid.

## Known limitations

- **No LLM runs yet.** The pipeline has not been run on any perturbed scenario, so there are no robustness results from this code yet.
- **Small N.** 29 queries; one query is 3.4 percentage points. Use the noise floor before reading an effect.
- **Vacuous rule applies to new runs only.** The saved outputs contain no vacuous queries, so their numbers are unchanged. The Docker image must be rebuilt (`docker compose build fgf-pipeline`) before any new run, or the container keeps the old evaluator. This was checked: the old image had zero matches for the N/A status string.
- **Report inconsistency.** Table 3.4's base 0.828 (24/29) matches `dr_spider_cheap`. The text in §3.2 (23 passes, six failures including Q38) matches `openai_best_fol_default` (0.793). The report should name the run it describes.
- **Git.** `.gitignore` contains `FGF/**`, so nothing under `FGF/` is tracked by git. None of this work is in version control.
