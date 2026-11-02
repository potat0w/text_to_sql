#!/usr/bin/env bash
# ./modal.sh [--detach] [--force] --cmd upload|prepare|baseline|train|test|compare|pull
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .profile ]]; then
  # shellcheck disable=SC1091
  source .profile
fi

detach=()
args=()
for arg in "$@"; do
  if [[ "$arg" == "--detach" ]]; then
    detach+=("--detach")
  else
    args+=("$arg")
  fi
done

if [[ ${#args[@]} -eq 0 ]]; then
  echo "Usage: ./modal.sh [--detach] [--force] --cmd upload|prepare|baseline|train|test|compare|pull" >&2
  exit 1
fi

if [[ "${args[*]}" == "--cmd pull" ]]; then
  # Download into an empty directory, then copy the JSON up. If the destination
  # already exists, `modal volume get` nests a second results/ folder inside it.
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  modal volume get --env text2sql text2sql /results "$tmp/results"
  mkdir -p results
  cp -a "$tmp/results"/. results/
  exit 0
fi

exec modal run --env text2sql "${detach[@]}" -m text2sql.modal_app "${args[@]}"
