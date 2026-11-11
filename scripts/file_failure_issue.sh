#!/usr/bin/env bash
# File a failure of the live loop as a GitHub issue, where somebody will see it.
#
#   GH_TOKEN=... scripts/file_failure_issue.sh "<body>"
#
# One issue, commented on while it stays open rather than duplicated, so a long
# outage reads as one thread. Shared by drift-loop.yml (the weekly cycle) and
# heartbeat.yml (the daily check): the title is how an open issue is found
# again, so it lives here once rather than in two workflows that could drift.
set -euo pipefail

title="drift-loop: the weekly cycle is failing"
body="${1:?usage: file_failure_issue.sh <body>}"

number=$(gh issue list --state open --search "$title in:title" --json number --jq '.[0].number // empty')
if [ -n "$number" ]; then
  gh issue comment "$number" --body "$body"
else
  gh issue create --title "$title" --body "$body"
fi
