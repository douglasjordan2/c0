#!/usr/bin/env bash
# Setup for the reflection-loop demo (assets/demo-reflect.tape).
# Resets the reflector inbox + pending queue and removes the demo concept so the
# loop records as a clean gap -> classify -> commit transition.
# Requires a working classifier (config: classification_provider).
# NOTE: this empties ~/.c0/reflector/pending-commits.jsonl — back it up first if
# the daemon has real pending commits you care about.
set -e
: > "$HOME/.c0/reflector/inbox.jsonl"
: > "$HOME/.c0/reflector/pending-commits.jsonl"
c0 find "MATCH (c:Concept) WHERE c.name STARTS WITH 'biome-javascript-linter' DETACH DELETE c RETURN count(c)" >/dev/null 2>&1 || true
