#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Kill servers and stop DB on exit
trap 'echo ""; echo "Shutting down..."; kill $(jobs -p) 2>/dev/null; docker compose -f "$ROOT/api/docker-compose.yml" stop db; exit' INT TERM

# Start Postgres if not already running
echo "Starting database..."
docker compose -f "$ROOT/api/docker-compose.yml" up db -d

echo "Waiting for database to be ready..."
until docker compose -f "$ROOT/api/docker-compose.yml" exec -T db pg_isready -U ftp_admin -d finetuning -q; do
  sleep 1
done
echo "Database ready."

echo "Starting API..."
cd "$ROOT/api"
.venv/bin/uvicorn app.main:app --reload --env-file .env --port 8000 &

echo "Starting frontend..."
cd "$ROOT/frontend"
npm run dev &

wait
