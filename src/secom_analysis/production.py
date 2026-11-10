"""Load the imputed training datasets into the ``secom_production`` Postgres schema.

Layering:

* ``secom_raw_staging`` - the untouched source files (``staging.py``); never modified here.
* ``secom_production``  - modelling-ready data derived from the training split:

  * ``train_prepared``              - the common input to every imputer: 450 features after
                                      rough reduction and 3σ capping, missing = ``NULL``;
  * ``train_imputed_<strategy>``    - one complete table per imputation candidate;
  * ``imputation_strategy``         - registry: description, parameters, comparison
                                      metrics, dataset hash, and ``status``
                                      (``candidate`` → ``selected`` / ``rejected``);
  * ``train_selected`` (view)       - created by ``--select``; points at the one chosen
                                      strategy so downstream code never names a candidate.

Every table carries ``source_line``, so any row joins back to ``secom_raw_staging``.
Loads run in one transaction and replace the previous candidate tables.

Run with::

    python -m secom_analysis.production --select mice_extratrees   # after the comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from .impute import ID_COLUMN, STRATEGY_BY_NAME, TARGET_COLUMN
from .staging import database_url

SCHEMA = "secom_production"


def _create_table(cur: psycopg.Cursor, table: str, frame: pd.DataFrame) -> None:
    schema = sql.Identifier(SCHEMA)
    definitions = [
        sql.SQL("{} integer PRIMARY KEY").format(sql.Identifier(ID_COLUMN)),
        sql.SQL("{} smallint NOT NULL CHECK ({} IN (0, 1))").format(
            sql.Identifier(TARGET_COLUMN), sql.Identifier(TARGET_COLUMN)
        ),
    ]
    for column in frame.columns:
        if column in (ID_COLUMN, TARGET_COLUMN):
            continue
        kind = "smallint" if column.startswith("missing_flag_") else "double precision"
        definitions.append(sql.SQL("{} " + kind).format(sql.Identifier(column)))
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(schema, sql.Identifier(table)))
    cur.execute(
        sql.SQL("CREATE TABLE {}.{} ({})").format(
            schema, sql.Identifier(table), sql.SQL(", ").join(definitions)
        )
    )


def _copy(cur: psycopg.Cursor, table: str, frame: pd.DataFrame) -> None:
    ordered = frame[
        [
            ID_COLUMN,
            TARGET_COLUMN,
            *[c for c in frame.columns if c not in (ID_COLUMN, TARGET_COLUMN)],
        ]
    ]
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in ordered.columns)
    values = ordered.astype(object).where(ordered.notna(), None).to_numpy()
    with cur.copy(
        sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
            sql.Identifier(SCHEMA), sql.Identifier(table), columns
        )
    ) as copy:
        for row in values:
            copy.write_row(
                [
                    None if v is None else (int(v) if isinstance(v, (np.integer,)) else v)
                    for v in row
                ]
            )


def load_production(
    datasets: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    prepared_features: pd.DataFrame,
    prepared_path: Path,
    url: str | None = None,
) -> dict[str, int]:
    """Replace ``train_prepared`` and every candidate table; upsert the registry."""
    url = url or database_url()
    prepared = pd.read_csv(prepared_path)
    input_sha = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    schema = sql.Identifier(SCHEMA)
    counts: dict[str, int] = {}
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        cur.execute(
            sql.SQL(
                "COMMENT ON SCHEMA {} IS 'Modelling-ready SECOM training data: prepared input, "
                "imputed candidates, strategy registry'"
            ).format(schema)
        )
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.imputation_strategy ("
                "strategy text PRIMARY KEY, label text, description text, params jsonb, "
                "metrics jsonb, table_name text, row_count integer, column_count integer, "
                "dataset_sha256 text, input_sha256 text, "
                "status text NOT NULL DEFAULT 'candidate' "
                "CHECK (status IN ('candidate', 'selected', 'rejected')), "
                "loaded_at timestamptz NOT NULL DEFAULT now())"
            ).format(schema)
        )
        _create_table(cur, "train_prepared", prepared)
        _copy(cur, "train_prepared", prepared)
        counts["train_prepared"] = len(prepared)

        for name, dataset in datasets.items():
            table = f"train_imputed_{name}"
            _create_table(cur, table, dataset)
            _copy(cur, table, dataset)
            strategy = STRATEGY_BY_NAME[name]
            metrics = {
                k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                for k, v in comparison.loc[name].items()
                if isinstance(v, (int, float, np.integer, np.floating))
            }
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.imputation_strategy (strategy, label, description, params, "
                    "metrics, table_name, row_count, column_count, dataset_sha256, input_sha256, "
                    "status, loaded_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "'candidate', now()) ON CONFLICT (strategy) DO UPDATE SET "
                    "label = EXCLUDED.label, description = EXCLUDED.description, "
                    "params = EXCLUDED.params, metrics = EXCLUDED.metrics, "
                    "table_name = EXCLUDED.table_name, row_count = EXCLUDED.row_count, "
                    "column_count = EXCLUDED.column_count, "
                    "dataset_sha256 = EXCLUDED.dataset_sha256, "
                    "input_sha256 = EXCLUDED.input_sha256, status = 'candidate', "
                    "loaded_at = now()"
                ).format(schema),
                (
                    name,
                    strategy.label,
                    strategy.description,
                    Jsonb(strategy.params),
                    Jsonb(metrics),
                    table,
                    len(dataset),
                    dataset.shape[1] - 2,
                    comparison.loc[name, "dataset_sha256"],
                    input_sha,
                ),
            )
            counts[table] = len(dataset)
        cur.execute(sql.SQL("DROP VIEW IF EXISTS {}.train_selected").format(schema))
    problems = verify_production(url, datasets, prepared_features)
    if problems:
        raise RuntimeError("; ".join(problems))
    return counts


def verify_production(
    url: str, datasets: dict[str, pd.DataFrame], prepared_features: pd.DataFrame
) -> list[str]:
    """Row counts, NULL counts and source_line join back to raw staging."""
    problems = []
    schema = sql.Identifier(SCHEMA)
    features = list(prepared_features.columns)
    null_sum = sql.SQL(" + ").join(
        sql.SQL("(t.{} IS NULL)::int").format(sql.Identifier(c)) for c in features
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        expected_nulls = int(prepared_features.isna().sum().sum())
        tables = {"train_prepared": expected_nulls} | {
            f"train_imputed_{name}": 0 for name in datasets
        }
        for table, nulls in tables.items():
            cur.execute(
                sql.SQL(
                    "SELECT count(*), coalesce(sum({}), 0), "
                    "count(*) FILTER (WHERE r.source_line IS NULL) "
                    "FROM {}.{} t LEFT JOIN secom_raw_staging.measurements r USING (source_line)"
                ).format(null_sum, schema, sql.Identifier(table))
            )
            rows, found_nulls, orphans = cur.fetchone()
            if rows != len(prepared_features):
                problems.append(f"{table}: {rows} rows, expected {len(prepared_features)}")
            if found_nulls != nulls:
                problems.append(f"{table}: {found_nulls} NULL cells, expected {nulls}")
            if orphans:
                problems.append(f"{table}: {orphans} rows not in secom_raw_staging")
    return problems


def select_strategy(name: str, url: str | None = None) -> None:
    """Mark one candidate as selected (others rejected) and point ``train_selected`` at it."""
    if name not in STRATEGY_BY_NAME:
        raise ValueError(f"unknown strategy {name!r}")
    schema = sql.Identifier(SCHEMA)
    with psycopg.connect(url or database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "UPDATE {}.imputation_strategy SET status = "
                "CASE WHEN strategy = %s THEN 'selected' ELSE 'rejected' END"
            ).format(schema),
            (name,),
        )
        cur.execute(
            sql.SQL("CREATE OR REPLACE VIEW {}.train_selected AS SELECT * FROM {}.{}").format(
                schema, schema, sql.Identifier(f"train_imputed_{name}")
            )
        )


def record_cv_results(summary_path: Path, url: str | None = None) -> int:
    """Store the downstream CV means per model in the registry (``metrics -> 'cv'``)."""
    summary = pd.read_csv(summary_path)
    keep = [c for c in summary.columns if c.endswith("_mean") or c in ("corrected_p",)]
    schema = sql.Identifier(SCHEMA)
    updated = 0
    with psycopg.connect(url or database_url()) as conn, conn.cursor() as cur:
        for name, group in summary.groupby("strategy"):
            cv = {row["model"]: {c: float(row[c]) for c in keep} for _, row in group.iterrows()}
            cur.execute(
                sql.SQL(
                    "UPDATE {}.imputation_strategy "
                    "SET metrics = metrics || jsonb_build_object('cv', %s::jsonb) "
                    "WHERE strategy = %s"
                ).format(schema),
                (json.dumps(cv), name),
            )
            updated += cur.rowcount
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the secom_production schema.")
    parser.add_argument("--select", help="strategy to promote to train_selected")
    parser.add_argument(
        "--record-cv",
        type=Path,
        nargs="?",
        const=Path("reports/imputation_cv/cv_summary.csv"),
        help="store CV results from cv_summary.csv in the registry",
    )
    args = parser.parse_args(argv)
    if args.record_cv:
        print(json.dumps({"cv_results_recorded": record_cv_results(args.record_cv)}))
    if args.select:
        select_strategy(args.select)
        print(json.dumps({"selected": args.select, "view": f"{SCHEMA}.train_selected"}))
    if not (args.select or args.record_cv):
        parser.error("nothing to do: pass --record-cv and/or --select <strategy>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
