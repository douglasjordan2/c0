#!/usr/bin/env bash
# Live integration test: spin up against a real Neo4j + real Ollama embeddings,
# seed a realistic graph by driving real c0 commands, then assert that real
# commands return the right results. Designed to run inside the c0-test service
# of docker-compose.test.yml (see scripts/docker-test.sh).
#
# Determinism: embeddings and the graph are real; the non-deterministic chat LLM
# (concept extraction / session enrichment) is NOT required — its structural
# inputs (turns + embeddings) are validated here, and the selection logic itself
# is covered by unit tests. Set C0_ENRICH_CHAT_MODEL=<ollama-model> to also run
# the live enrichment extraction and assert it links at least one concept.
set -uo pipefail

NS="demo"
export HOME="${HOME:-/root}"
PROJ="/work/$NS"
OLLAMA="${OLLAMA_HOST:-http://ollama:11434}"
EMBED_MODEL="nomic-embed-text"

mkdir -p "$PROJ" "$HOME/.c0" "$HOME/.claude/projects/$NS"

# Point c0 at the containerized services. Neo4j is wired via NEO4J_* env;
# Ollama host lives in the global config (the embedding path reads it there).
cat > "$HOME/.c0/config.toml" <<EOF
[semantic]
enabled = true

[ollama]
host = "$OLLAMA"
model = "$EMBED_MODEL"
EOF

wait_for() { # name url
  local name="$1" url="$2" i=0
  echo -n "waiting for $name "
  until curl -sf "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 150 ]; then echo " TIMEOUT"; exit 1; fi
    sleep 2; echo -n "."
  done
  echo " ready"
}

wait_for "neo4j" "http://neo4j:7474"
wait_for "ollama" "$OLLAMA/api/tags"

echo "── ensuring embedding model '$EMBED_MODEL' is present ──"
curl -sf "$OLLAMA/api/pull" -d "{\"name\":\"$EMBED_MODEL\"}" >/dev/null || true
until curl -sf "$OLLAMA/api/tags" | grep -q "$EMBED_MODEL"; do sleep 2; done
echo "model ready"

echo "── schema + namespace ──"
c0 migrate
cd "$PROJ"
c0 init --namespace "$NS" || true

bash /c0/scripts/seed-test-graph.sh "$NS" "$PROJ"

echo
echo "════════════════ assertions ════════════════"
FAIL=0
assert_contains() { # description expected -- cmd...
  local desc="$1" expected="$2"; shift 2
  local out; out="$("$@" 2>&1)"
  if grep -qi -- "$expected" <<<"$out"; then
    echo "PASS: $desc"
  else
    echo "FAIL: $desc — expected to find '$expected' in output of: $*"
    sed 's/^/    | /' <<<"$out"
    FAIL=1
  fi
}

# Graph traversal: walk resolves a concept and follows its edges.
assert_contains "walk traverses to neighbour" "hybrid search" \
  c0 walk "reciprocal rank fusion"

# Vector search: real embeddings retrieve a paraphrase with no keyword overlap.
assert_contains "vector search finds concept (real embeddings)" "reciprocal rank fusion" \
  c0 search "combine several ranked result lists into one" --vector-only

# Keyword search: BM25 path.
assert_contains "keyword search finds concept" "bm25" \
  c0 search "bm25" --keyword-only

# Retrieval eval: the full cascade (exact -> fulltext -> hybrid + temporal) over
# real embeddings resolves its golden set. This exercises the vector and temporal
# tiers the fulltext-only CI eval-gate can't reach; --min-recall makes the command
# self-asserting (it exits non-zero, failing this assertion, on a regression).
assert_contains "eval: full cascade recall is high (real embeddings)" "gate passed" \
  c0 eval --seed --k 3 --min-recall 0.9

# Temporal: the superseding concept is present and walkable.
assert_contains "supersession recorded" "app router" \
  c0 walk "app router"

# Sessions: the fixture transcript indexed into the current namespace
# (listed by its first prompt, which the index parsed from the transcript).
assert_contains "session indexed" "reciprocal rank fusion" \
  c0 sessions

# Session turns carry real embeddings — exactly what enrichment ranks over.
assert_contains "session turns are embedded + searchable" "fusion" \
  c0 sessions search "reciprocal rank fusion ranking"

# Optional: live enrichment extraction (needs a chat model; non-deterministic).
if [ -n "${C0_ENRICH_CHAT_MODEL:-}" ]; then
  echo "── running live enrichment with model '$C0_ENRICH_CHAT_MODEL' ──"
  curl -sf "$OLLAMA/api/pull" -d "{\"name\":\"$C0_ENRICH_CHAT_MODEL\"}" >/dev/null || true
  C0_ENRICH_MODEL="$C0_ENRICH_CHAT_MODEL" c0 sessions enrich --force 2>&1 | sed 's/^/    /' || true
  assert_contains "enrichment linked concepts to the session" "concept" \
    c0 find "MATCH (c:Concept)-[:MENTIONED_IN]->(s:Session) RETURN c.name LIMIT 5"
else
  echo "SKIP: live enrichment extraction (no chat model set). Turn embeddings are"
  echo "      validated above; salience-selection logic is covered by unit tests."
  echo "      Set C0_ENRICH_CHAT_MODEL=<ollama-model> to run the full path."
fi

echo "═════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
  echo "✓ all integration assertions passed"
else
  echo "✗ integration assertions failed"
fi
exit "$FAIL"
