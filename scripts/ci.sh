#!/usr/bin/env bash
# Run the same checks as .github/workflows/ci.yml locally, in the same order.
# Keep this in sync with that workflow so a green run here means a green CI.
#
# Usage: ./scripts/ci.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

run() {
	echo
	echo "▶ $*"
	"$@"
}

run cargo fmt --all --check
run cargo clippy --all-targets --all-features
run cargo build --verbose
run cargo build --features sessions --verbose
run cargo test --all-features

echo
echo "✓ local CI passed"
