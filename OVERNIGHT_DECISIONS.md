# Overnight Decisions

## Active Assumptions

- Use `codex/tinker-integration` as the base for new stacked work unless a task specifically depends on #40.
- Keep #40 as the integration branch for DataGen reconciliation, not as a dumping ground for unrelated fixes.
- Prefer small live validations before full pipeline runs.
- If a live test fails because a credential is invalid or a provider rejects access, record the failure and pivot to local/fake validation plus a clear PR note.

## Decisions

- 2026-05-19 05:20 PDT — Created `codex/overnight-coordination` worktree for progress logs so feature branches remain clean.
- 2026-05-19 05:30 PDT — Treat raw Mode C web acquisition as source material, not trainable SFT data. Curation must reject targetless web rows until a real structuring stage produces assistant targets.
- 2026-05-19 05:30 PDT — Add explicit Mode C backend control: `auto` can fall back, `synthetic` never calls web, `web` fails loudly. `DATA_GENERATOR_SYNTHETIC_OFFLINE=1` overrides all of them for deterministic local tests.
- 2026-05-19 05:31 PDT — Add `testpaths = tests` / `norecursedirs = outputs/reference` because the local gitignored Tinker cookbook reference checkout should not be collected by repo pytest.
- 2026-05-19 05:32 PDT — Keep the full live HF retrieval suite out of broad smoke commands. It enumerates and downloads many datasets and is a separate external validation, not a cheap overnight regression check.
