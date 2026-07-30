#!/usr/bin/env bash
# Fail if anything that looks like a live credential is staged, tracked, or
# anywhere in git history.
#
#   bash scripts/scan-secrets.sh            # staged + tracked files
#   bash scripts/scan-secrets.sh --history  # also every commit ever made (slow)
#
# Install as a pre-commit hook:
#   git config core.hooksPath .githooks
#
# This is a backstop, not a substitute for keeping keys in .env. It catches the
# common provider key formats; it cannot catch a secret that doesn't look like
# one. Assume it will miss something.

set -uo pipefail

PATTERNS='(sk-ant-[A-Za-z0-9_-]{15,}|sk-[A-Za-z0-9]{32,}|gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{25,}|tvly-[A-Za-z0-9]{10,}|eyJhbGciOi[A-Za-z0-9_-]{25,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16})'

fail=0

echo "==> .env must not be tracked"
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "    FAIL: .env is tracked by git. Run: git rm --cached .env"
  fail=1
else
  echo "    ok"
fi

echo "==> scanning tracked files"
if hits=$(git ls-files -z | xargs -0 grep -InE "$PATTERNS" 2>/dev/null); then
  echo "    FAIL:"
  echo "$hits" | sed 's/^/      /'
  fail=1
else
  echo "    ok"
fi

echo "==> scanning staged changes"
if hits=$(git diff --cached -U0 | grep -E "^\+" | grep -InE "$PATTERNS" 2>/dev/null); then
  echo "    FAIL:"
  echo "$hits" | sed 's/^/      /'
  fail=1
else
  echo "    ok"
fi

if [ "${1:-}" = "--history" ]; then
  echo "==> scanning full history (this takes a while)"
  found=$(git rev-list --all | while read -r c; do
    git grep -IlE "$PATTERNS" "$c" 2>/dev/null
  done | sort -u)
  if [ -n "$found" ]; then
    echo "    FAIL: secrets found in history:"
    echo "$found" | sed 's/^/      /'
    echo "    A rotate-then-purge is required. Rotating the key is the part"
    echo "    that actually matters; rewriting history alone does not help,"
    echo "    because GitHub retains unreachable objects."
    fail=1
  else
    echo "    ok"
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "SECRET SCAN FAILED — do not commit or push."
  exit 1
fi

echo
echo "Secret scan clean."
