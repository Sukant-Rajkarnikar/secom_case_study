"""Load the untouched SECOM source files into the ``secom_raw_staging`` Postgres schema.

Raw staging keeps the source as close to the files as possible:

* ``measurements``  - one row per line of ``secom.data``; ``feature_000`` .. ``feature_589``
  as double precision. The file's ``NaN`` tokens become SQL ``NULL``.
* ``labels``        - one row per line of ``secom_labels.data``: the raw -1/1 ``label``,
  the timestamp text exactly as written, and the parsed ``test_timestamp``.
* ``source_files``  - lineage: file name, SHA-256, size, line count, and the full text of
  ``secom.names``.
* ``load_log``      - one row per load run.

``source_line`` (1-based line number) is the join key between measurements and labels.
Every load is a single transaction that truncates and reloads, so re-running is safe.

Run with::

    python -m secom_analysis.staging --data-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg import sql

from .data import LABELS_FILE, MEASUREMENTS_FILE, TIMESTAMP_FORMAT, resolve_data_dir

SCHEMA = "secom_raw_staging"
NAMES_FILE = "secom.names"
N_FEATURES = 590
FEATURE_COLUMNS = [f"feature_{i:03d}" for i in range(N_FEATURES)]


def database_url(env_file: Path = Path(".env")) -> str:
    """Return ``SECOM_DATABASE_URL`` from the environment or the project ``.env`` file."""
    if url := os.environ.get("SECOM_DATABASE_URL"):
        return url
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "SECOM_DATABASE_URL":
                return value.strip()
    raise RuntimeError("SECOM_DATABASE_URL not set; run scripts/db.sh init first")


def _ddl() -> list[sql.Composed | sql.SQL]:
    schema = sql.Identifier(SCHEMA)
    features = sql.SQL(", ").join(
        sql.SQL("{} double precision").format(sql.Identifier(c)) for c in FEATURE_COLUMNS
    )
    return [
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema),
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {}.load_log ("
            " load_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
            " loaded_at timestamptz NOT NULL DEFAULT now(),"
            " source_dir text NOT NULL,"
            " measurement_rows integer NOT NULL,"
            " label_rows integer NOT NULL)"
        ).format(schema),
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {}.source_files ("
            " file_name text PRIMARY KEY,"
            " sha256 char(64) NOT NULL,"
            " size_bytes bigint NOT NULL,"
            " line_count integer NOT NULL,"
            " content text,"
            " load_id bigint NOT NULL REFERENCES {}.load_log(load_id))"
        ).format(schema, schema),
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {}.measurements ("
            " source_line integer PRIMARY KEY, {},"
            " load_id bigint NOT NULL REFERENCES {}.load_log(load_id))"
        ).format(schema, features, schema),
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {}.labels ("
            " source_line integer PRIMARY KEY,"
            " label smallint NOT NULL CHECK (label IN (-1, 1)),"
            " timestamp_raw text NOT NULL,"
            " test_timestamp timestamp,"
            " load_id bigint NOT NULL REFERENCES {}.load_log(load_id))"
        ).format(schema, schema),
        sql.SQL(
            "COMMENT ON SCHEMA {} IS 'Untransformed UCI SECOM source files (NaN stored as NULL)'"
        ).format(schema),
    ]


def _parse_measurements(path: Path) -> list[list[float | None]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.split()
        if len(tokens) != N_FEATURES:
            raise ValueError(f"{path.name}:{number} has {len(tokens)} fields, expected 590")
        values = []
        for token in tokens:
            value = float(token)  # raises on anything that is not a number or NaN
            values.append(None if math.isnan(value) else value)
        rows.append(values)
    return rows


def _parse_labels(path: Path) -> list[tuple[int, str, datetime]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        label_text, _, timestamp_text = line.strip().partition(" ")
        timestamp_raw = timestamp_text.strip().strip('"')
        label = int(label_text)
        if label not in (-1, 1):
            raise ValueError(f"{path.name}:{number} has label {label}, expected -1 or 1")
        rows.append((label, timestamp_raw, datetime.strptime(timestamp_raw, TIMESTAMP_FORMAT)))
    return rows


def _file_meta(path: Path) -> tuple[str, int, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), len(data.splitlines())


def load_raw_staging(data_dir: Path, url: str) -> dict[str, int]:
    """Validate and load the three SECOM files; return row counts."""
    data_dir = resolve_data_dir(data_dir)
    measurements = _parse_measurements(data_dir / MEASUREMENTS_FILE)
    labels = _parse_labels(data_dir / LABELS_FILE)
    if len(measurements) != len(labels):
        raise ValueError(f"{len(measurements)} measurement rows vs {len(labels)} label rows")

    schema = sql.Identifier(SCHEMA)
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        for statement in _ddl():
            cur.execute(statement)
        cur.execute(
            sql.SQL("TRUNCATE {}.measurements, {}.labels, {}.source_files").format(
                schema, schema, schema
            )
        )
        cur.execute(
            sql.SQL(
                "INSERT INTO {}.load_log (source_dir, measurement_rows, label_rows) "
                "VALUES (%s, %s, %s) RETURNING load_id"
            ).format(schema),
            (str(data_dir), len(measurements), len(labels)),
        )
        load_id = cur.fetchone()[0]

        for file_name in (MEASUREMENTS_FILE, LABELS_FILE, NAMES_FILE):
            path = data_dir / file_name
            if not path.exists():
                continue
            sha, size, lines = _file_meta(path)
            content = path.read_text(encoding="utf-8") if file_name == NAMES_FILE else None
            cur.execute(
                sql.SQL("INSERT INTO {}.source_files VALUES (%s, %s, %s, %s, %s, %s)").format(
                    schema
                ),
                (file_name, sha, size, lines, content, load_id),
            )

        columns = sql.SQL(", ").join(
            sql.Identifier(c) for c in ["source_line", *FEATURE_COLUMNS, "load_id"]
        )
        with cur.copy(
            sql.SQL("COPY {}.measurements ({}) FROM STDIN").format(schema, columns)
        ) as copy:
            for number, values in enumerate(measurements, start=1):
                copy.write_row([number, *values, load_id])
        with cur.copy(
            sql.SQL(
                "COPY {}.labels (source_line, label, timestamp_raw, test_timestamp, load_id) "
                "FROM STDIN"
            ).format(schema)
        ) as copy:
            for number, row in enumerate(labels, start=1):
                copy.write_row([number, *row, load_id])
        cur.execute(sql.SQL("ANALYZE {}.measurements").format(schema))
        cur.execute(sql.SQL("ANALYZE {}.labels").format(schema))

    nan_cells = sum(value is None for row in measurements for value in row)
    return {
        "load_id": load_id,
        "measurement_rows": len(measurements),
        "label_rows": len(labels),
        "null_cells": nan_cells,
    }


def verify_raw_staging(url: str, expected: dict[str, int]) -> list[str]:
    """Compare database contents with what was parsed from the files; return problems."""
    problems = []
    schema = sql.Identifier(SCHEMA)
    null_sum = sql.SQL(" + ").join(
        sql.SQL("({} IS NULL)::int").format(sql.Identifier(c)) for c in FEATURE_COLUMNS
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT count(*), coalesce(sum({}), 0) FROM {}.measurements").format(
                null_sum, schema
            )
        )
        rows, nulls = cur.fetchone()
        cur.execute(
            sql.SQL(
                "SELECT count(*), count(*) FILTER (WHERE m.source_line IS NULL) "
                "FROM {}.labels l LEFT JOIN {}.measurements m USING (source_line)"
            ).format(schema, schema)
        )
        label_rows, orphans = cur.fetchone()
    if rows != expected["measurement_rows"]:
        problems.append(f"measurements: {rows} rows, expected {expected['measurement_rows']}")
    if nulls != expected["null_cells"]:
        problems.append(f"measurements: {nulls} NULL cells, expected {expected['null_cells']}")
    if label_rows != expected["label_rows"]:
        problems.append(f"labels: {label_rows} rows, expected {expected['label_rows']}")
    if orphans:
        problems.append(f"labels: {orphans} rows without a matching measurement row")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load raw SECOM files into Postgres staging.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--database-url", default=None, help="defaults to SECOM_DATABASE_URL")
    args = parser.parse_args(argv)

    url = args.database_url or database_url()
    counts = load_raw_staging(args.data_dir, url)
    problems = verify_raw_staging(url, counts)
    print(f"Loaded into schema {SCHEMA}: {counts}")
    for problem in problems:
        print(f"[FAIL] {problem}")
    if not problems:
        print("[PASS] row counts, NULL counts and label/measurement join verified")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
