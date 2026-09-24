#!/usr/bin/env bash
# Installs awslabs/git-secrets hooks + prohibited patterns for this repo.
# Run this after every fresh clone: ./scripts/setup-git-secrets.sh
#
# git-secrets stores hooks in .git/hooks and prohibited patterns in
# .git/config, neither of which are versioned by git itself, so this script
# is the source of truth and must be re-run per clone. Allowed (false
# positive) patterns live in the tracked .gitallowed file instead, since
# git-secrets reads that automatically.
set -euo pipefail

if ! command -v git-secrets >/dev/null 2>&1; then
  echo "git-secrets is not installed. See https://github.com/awslabs/git-secrets#installing-git-secrets" >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel)"

git secrets --install --force
git secrets --register-aws

# git-secrets matches patterns with `grep -Ew` (POSIX ERE, whole-word, no -i)
# so patterns below avoid PCRE syntax like (?i) and spell out case variants.

# PEM-style private key material. Leading dash is bracket-escaped ("[-]---")
# rather than literal ("-----") because git-secrets' internal de-dupe check
# shells out to `grep -F "$pattern"` without a `--` guard, so a pattern
# starting with a bare `-` gets misparsed as a grep option on every re-run.
git secrets --add -- '[-]----BEGIN[[:space:]]?(RSA|DSA|EC|OPENSSH|PGP)?[[:space:]]?PRIVATE[[:space:]]KEY-----'

# Well-known token formats with unique prefixes (low false-positive rate).
git secrets --add -- 'gh[pousr]_[A-Za-z0-9]{36,}'                                    # GitHub tokens
git secrets --add -- 'xox[baprs]-[A-Za-z0-9-]{10,}'                                  # Slack tokens
git secrets --add -- 'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'  # JWTs

# Generic "password/secret/token = <realistic-looking value>" assignments.
# Two branches cover different separator styles:
#   Branch 1: keyword + optional quote + [:=] + optional quote + value
#     Catches: password: value, "password":"value", apiKey='value'
#   Branch 2: keyword + space + quoted value (no colon/equals)
#     Catches: password `value`, reset-admin-password 'value'
# Both require at least one digit in the value: prose that merely talks
# *about* secrets ("password: generate random...") has none, while real
# generated secrets/hashes/tokens virtually always contain a digit somewhere.
# Placeholders like <bcrypt_hash> or CHANGEME, and sealed-secrets ciphertext,
# are excluded via .gitallowed rather than here.
git secrets --add -- '([Pp]assword|[Pp]asswd|[Pp]wd|[Ss]ecret|[Tt]oken|[Aa]pi[_-]?[Kk]ey|[Aa]ccess[_-]?[Kk]ey)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"']?([A-Za-z0-9/+_.@#$%^&*-]{6,}[0-9][A-Za-z0-9/+_.@#$%^&*-]*|[A-Za-z/+_.@#$%^&*-]*[0-9][A-Za-z0-9/+_.@#$%^&*-]{6,})["'"'"']?'
git secrets --add -- '([Pp]assword|[Pp]asswd|[Pp]wd|[Ss]ecret|[Tt]oken|[Aa]pi[_-]?[Kk]ey|[Aa]ccess[_-]?[Kk]ey)[[:space:]]+['"'"'`"]([A-Za-z0-9/+_.@#$%^&*-]{6,}[0-9][A-Za-z0-9/+_.@#$%^&*-]*|[A-Za-z/+_.@#$%^&*-]*[0-9][A-Za-z0-9/+_.@#$%^&*-]{6,})['"'"'`"]'
# Branch 3: keyword + space + optional ( + quoted value + optional )
#   Catches: password (`value`) — prose with paren-wrapped backtick value
git secrets --add -- '([Pp]assword|[Pp]asswd|[Pp]wd|[Ss]ecret|[Tt]oken|[Aa]pi[_-]?[Kk]ey|[Aa]ccess[_-]?[Kk]ey)[[:space:]]+\(?['"'"'`"]([A-Za-z0-9/+_.@#$%^&*-]{6,}[0-9][A-Za-z0-9/+_.@#$%^&*-]*|[A-Za-z/+_.@#$%^&*-]*[0-9][A-Za-z0-9/+_.@#$%^&*-]{6,})['"'"'`"]\)?'

echo "git-secrets installed. Prohibited patterns:"
git secrets --list
