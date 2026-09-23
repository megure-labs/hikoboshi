#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: resolve-pr-author-association.sh OWNER/REPO PULL_NUMBER" >&2
  exit 2
fi
repository="$1"
pull_number="$2"
if [[ ! "${repository}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] ||
   [[ ! "${pull_number}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid repository or pull-request number" >&2
  exit 2
fi

identity="$(gh api --method GET "repos/${repository}/pulls/${pull_number}" \
  --jq '[.author_association, .user.login, (.user.id | tostring)] | @tsv')"
IFS=$'\t' read -r association author author_id extra <<< "${identity}"
if [[ ! "${association}" =~ ^[A-Z_]+$ ]] ||
   [[ ! "${author}" =~ ^[A-Za-z0-9-]+$ ]] ||
   [[ ! "${author_id}" =~ ^[1-9][0-9]*$ ]] ||
   [[ -n "${extra:-}" || "${identity}" == *$'\n'* ]]; then
  echo "GitHub returned an invalid pull-request author identity" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${association}" != OWNER && "${association}" != MEMBER ]] &&
   [[ ! -e "${repo_root}/.provenance/KANAME_ENFORCEMENT_BASE" ]]; then
  # Organization membership may be hidden from GITHUB_TOKEN. During bootstrap
  # verify actual repository administration, without claiming org membership.
  access="$(gh api --method GET \
    "repos/${repository}/collaborators/${author}/permission" \
    --jq '[.permission, .user.login, (.user.id | tostring)] | @tsv')"
  IFS=$'\t' read -r permission access_author access_id extra <<< "${access}"
  if [[ ! "${permission}" =~ ^[a-z_]+$ || "${access_author}" != "${author}" ||
        "${access_id}" != "${author_id}" || -n "${extra:-}" ||
        "${access}" == *$'\n'* ]]; then
    echo "GitHub returned invalid repository permission evidence" >&2
    exit 1
  fi
  if [[ "${permission}" == admin ]]; then
    association=BOOTSTRAP_REPOSITORY_ADMIN
  fi
fi
printf '%s\n' "${association}"
