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
- 2026-05-19 05:36 PDT — For live Tinker smoke budgeting, use conservative $2.50 ledger entries even when local token/cost reports are much lower. This keeps the $100 cap safe until authoritative billing is exposed.
- 2026-05-19 05:45 PDT — Use `LAST_ASSISTANT_MESSAGE` for Tinker chat/SFT V1 and split multi-assistant conversations into one target per assistant turn. This follows the cookbook-safe path for renderers without the extension property.
- 2026-05-19 05:48 PDT — Web acquisition and web structuring remain separate stages. Raw web pages are source material only; teacher structuring must produce schema-valid SFT rows before curation/training can proceed.
- 2026-05-19 05:49 PDT — Before spending more live budget, run a local stacked merge smoke across all active implementation PRs. The current stack merges cleanly and passes the non-live suite, so the next risk to chase is behavior, not branch compatibility.
- 2026-05-19 05:53 PDT — Treat AutoResearch proposal application as two coordinated artifacts: persist the JSON patch for audit/revert, but pass the pending patch into the SDK-native Tinker run directly from graph state.
- 2026-05-19 05:56 PDT — Keep empty or failed DataGen outputs from entering DecisionEngine/AutoResearch. The Manager boundary should explain untrainable data before the failure reaches Tinker.
- 2026-05-19 06:02 PDT — AutoResearch candidate acceptance should be monotonic against the current best score, not merely better than the original baseline. The diary should report the patch relative to the pre-patch config for auditability.
- 2026-05-19 06:06 PDT — Treat LLM-generated hyperparameter patches as untrusted until validated by `TrainingConfig.apply_patch`; prompt constraints are guidance, runtime validation is the guardrail.
- 2026-05-19 06:13 PDT — Keep the spec-site source of truth aligned with the stacked implementation branches. Since reviewers use the Vercel preview to understand intended behavior, stale REST-job language is treated as an implementation risk, not just documentation drift.
- 2026-05-19 06:22 PDT — Keep the first frontend/backend bridge intentionally thin: start a run, poll state, and surface final Manager results. Live per-stage streaming should wait for run-scoped observability rather than fabricating precision.
- 2026-05-19 06:26 PDT — Add an explicit integration PR for the draft stack rather than assuming sibling branches compose. Retarget docs behind this integrated stack so review order matches implementation reality.
