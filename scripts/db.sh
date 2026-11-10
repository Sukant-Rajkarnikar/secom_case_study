#!/usr/bin/env bash
# Project-local PostgreSQL 16 cluster for SECOM (no sudo / system service needed).
#
#   scripts/db.sh init     # create the cluster in ./.pgdata, the `secom` database and .env
#   scripts/db.sh start    # start the server (localhost:5433 + unix socket in ./.pgdata/run)
#   scripts/db.sh stop
#   scripts/db.sh status
#   scripts/db.sh psql     # open psql on the secom database
#
# Auth: unix socket uses peer auth (your OS user is the cluster superuser);
# TCP on 127.0.0.1 requires the password stored in .env (git-ignored).
set -euo pipefail

PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGDATA="$ROOT/.pgdata/data"
SOCKET_DIR="$ROOT/.pgdata/run"
LOG_FILE="$ROOT/.pgdata/postgres.log"
PORT="${SECOM_PG_PORT:-5433}"
DB_NAME="secom"
ENV_FILE="$ROOT/.env"

start() {
  if "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    echo "already running"; return
  fi
  "$PG_BIN/pg_ctl" -D "$PGDATA" -l "$LOG_FILE" -w start \
    -o "-p $PORT -k $SOCKET_DIR -c listen_addresses=localhost"
}

init() {
  if [[ -f "$PGDATA/PG_VERSION" ]]; then
    echo "cluster already exists at $PGDATA"; return
  fi
  mkdir -p "$SOCKET_DIR"
  chmod 700 "$ROOT/.pgdata"
  "$PG_BIN/initdb" -D "$PGDATA" -U "$USER" -E UTF8 --locale=C.UTF-8 \
    --auth-local=peer --auth-host=scram-sha-256 >/dev/null
  start
  password="$(openssl rand -hex 24)"
  "$PG_BIN/psql" -h "$SOCKET_DIR" -p "$PORT" -d postgres -v ON_ERROR_STOP=1 -q \
    -c "ALTER ROLE \"$USER\" PASSWORD '$password';" \
    -c "CREATE DATABASE $DB_NAME;"
  umask 077
  cat >"$ENV_FILE" <<EOF
# Local SECOM PostgreSQL (created by scripts/db.sh init). Do not commit.
SECOM_DATABASE_URL=postgresql://$USER:$password@localhost:$PORT/$DB_NAME
EOF
  echo "cluster ready: database '$DB_NAME' on localhost:$PORT; connection string in .env"
}

case "${1:-}" in
  init) init ;;
  start) start ;;
  stop) "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast stop ;;
  status) "$PG_BIN/pg_ctl" -D "$PGDATA" status ;;
  psql) shift; exec "$PG_BIN/psql" -h "$SOCKET_DIR" -p "$PORT" -d "$DB_NAME" "$@" ;;
  *) echo "usage: $0 {init|start|stop|status|psql}" >&2; exit 2 ;;
esac
