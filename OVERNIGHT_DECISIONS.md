# Overnight Decisions

## Active Assumptions

- Use `codex/tinker-integration` as the base for new stacked work unless a task specifically depends on #40.
- Keep #40 as the integration branch for DataGen reconciliation, not as a dumping ground for unrelated fixes.
- Prefer small live validations before full pipeline runs.
- If a live test fails because a credential is invalid or a provider rejects access, record the failure and pivot to local/fake validation plus a clear PR note.

## Decisions

- 2026-05-19 05:20 PDT — Created `codex/overnight-coordination` worktree for progress logs so feature branches remain clean.
