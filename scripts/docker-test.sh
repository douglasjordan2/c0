#!/usr/bin/env bash
# Run c0's live integration test in an isolated full stack (Neo4j + Ollama + c0).
# Seeds a realistic graph with real commands and asserts real command output.
# Exits with the integration test's status. See docker-compose.test.yml.
#
# Usage: ./scripts/docker-test.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

COMPOSE="docker compose -f docker-compose.test.yml"

# Tear down containers on exit, but keep the named Ollama-model volume so reruns
# don't re-download the embedding model.
cleanup() { $COMPOSE down >/dev/null 2>&1 || true; }
trap cleanup EXIT

$COMPOSE up --build --abort-on-container-exit --exit-code-from c0-test
