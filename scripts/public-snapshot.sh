#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: scripts/public-snapshot.sh [--candidate] OUTPUT_DIR\n' >&2
}

candidate=0
if [[ ${1:-} == --candidate ]]; then
  candidate=1
  shift
fi
if [[ $# -ne 1 || $1 == --help ]]; then
  usage
  exit 2
fi

root=$(git rev-parse --show-toplevel)
parent=$(cd "$(dirname "$1")" && pwd -P)
destination="$parent/$(basename "$1")"
if [[ $parent == "$root" || $parent == "$root/"* || -e $destination || -L $destination ]]; then
  printf 'Refusing a destination in the repository or one that already exists\n' >&2
  exit 1
fi
if [[ $candidate -eq 0 && -n $(git -C "$root" status --porcelain) ]]; then
  printf 'Commit and verify the release tree before making a public snapshot\n' >&2
  exit 1
fi

public=(
  .gitignore Cargo.toml Cargo.lock rust-toolchain.toml dist-workspace.toml README.md
  LICENSE-MIT LICENSE-APACHE COMPATIBILITY.md
  src tests schemas fixtures adapters scripts/public-snapshot.sh
)
if [[ $candidate -eq 1 ]]; then
  public+=(.github/workflows/ci.yml)
elif git -C "$root" cat-file -e HEAD:.github/workflows/ci.yml 2>/dev/null; then
  public+=(.github/workflows/ci.yml)
fi

temporary=$(mktemp -d "$parent/.snapjudge-public.XXXXXXXX")
trap 'rm -r "$temporary"' EXIT
if [[ $candidate -eq 1 ]]; then
  git -C "$root" ls-files --cached --others --exclude-standard -z -- "${public[@]}" \
    | (cd "$root" && tar --null -T - -cf -) \
    | tar -xf - -C "$temporary"
else
  git -C "$root" archive HEAD -- "${public[@]}" | tar -xf - -C "$temporary"
fi

link=$(find "$temporary" -type l -print -quit)
if [[ -n $link ]]; then
  printf 'Refusing a public snapshot containing symlinks\n' >&2
  exit 1
fi
if grep -R -q -E 'docs/(superpowers|research|typesafe)/|scripts/jev-ask.sh' \
  "$temporary/README.md" "$temporary/src"; then
  printf 'Public source still references private development documents\n' >&2
  exit 1
fi

mv "$temporary" "$destination"
trap - EXIT
printf 'Public source snapshot: %s\n' "$destination"
