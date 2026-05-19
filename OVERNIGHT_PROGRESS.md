# Overnight Progress

Started: 2026-05-19 05:20:26 PDT

Budget: $100 total. Target average: $2-3 per live experiment. Hard rule: do not spend the whole budget on one large run.

Current context:
- Base stack: #35 Tinker SFT runner.
- DataGen reconciliation: #40 combines #27 web acquisition, #38 synthetic SFT generation, and #37 curation.
- Existing validation before overnight loop: non-live suite on #40 passed with 164 passed / 6 skipped.

Operating rules:
- Track cumulative live spend in `OVERNIGHT_EXPERIMENTS.md`.
- Use one branch/worktree per logical code change.
- Open draft PRs early.
- Do not merge PRs.
- Do not copy or summarize internal company content from external skill repos into this repo.

## Timeline

- 2026-05-19 05:20 PDT — Started overnight loop. Credentials present: Tinker, Anthropic, HF, Tavily. Skill directories found.
- 2026-05-19 05:26 PDT — Live Tinker smoke `tinker-001` passed: `Qwen/Qwen3.5-9B`, 5 steps, 60.92s. Ledger uses conservative $2.50 estimate.
- 2026-05-19 05:29 PDT — Live DataGen scouts passed: Tavily sanity, tiny Mode C web acquisition, and tiny live Claude synthetic generation. Logged conservative $0.32 estimate.
- 2026-05-19 05:31 PDT — Hardened #40 with explicit Mode C backend selection, synthetic/offline no-web behavior, targetless raw-web rejection in curation, stricter synthetic label validation, and Manager programmatic `data_path` handling.
- 2026-05-19 05:32 PDT — Validation after hardening: compileall passed; targeted DataGen/Manager suite `46 passed, 1 skipped`; broad non-live repo suite `169 passed, 4 skipped`.
- 2026-05-19 05:32 PDT — Tiny live Claude strict synthetic validation passed after label enforcement: 8 records, teacher used, validation passed. Logged conservative $0.30 estimate.
- 2026-05-19 05:33 PDT — Pushed #40 commit `ba5e901` and updated the draft PR description with current validation and remaining E2E fake-Tinker boundary gap.
- 2026-05-19 05:36 PDT — Bounded live partial pipeline passed: Mode C synthetic/offline DataGen -> curation JSONL -> DecisionEngine `tinker_sft` plan -> 5-step live Tinker on `Qwen/Qwen3.5-9B`. Tinker local cost report was `$0.000138`, ledger books conservative $2.50.
- 2026-05-19 05:38 PDT — Tiny live web curation guard passed after one retry: 3 Tavily results, 2 crawled pages, 2 targetless web records, curation refused to create trainable rows.
- 2026-05-19 05:39 PDT — Rechecked #40 Vercel failures. `npx vercel inspect ... --logs` still cannot find the failing `ml-agent-demo` / `spr26-team-26` deployments under the current Vercel context; green spec-site projects remain separate.
- 2026-05-19 05:40 PDT — Draft PR #42 opened: test-only offline Manager -> DataGen -> curation -> DecisionEngine -> AutoResearch baseline -> fake Tinker boundary test, stacked on #40.
- 2026-05-19 05:43 PDT — Draft PR #39 updated by worker: CostManager now has in-process spend accounting and AutoResearch records baseline/iteration Tinker costs into it.
- 2026-05-19 05:45 PDT — Draft PR #43 opened: Tinker SFT runner now uses `LAST_ASSISTANT_MESSAGE` per assistant target, eliminating the live cookbook renderer warning. Live 2-step Tinker smoke passed.
- 2026-05-19 05:48 PDT — Draft PR #44 opened: Mode C web sources can now be teacher-structured into schema-valid chat/SFT rows, while raw/unstructured web pages remain invalid for training.
- 2026-05-19 05:49 PDT — Local full-stack merge smoke passed: #39 + #43 + #40 + #42 + #44 merged cleanly, then non-live suite passed with `181 passed, 4 skipped`.
- 2026-05-19 05:53 PDT — Draft PR #45 opened: AutoResearch iteration runs now submit the pending proposal patch to Tinker instead of silently rerunning the baseline hyperparameters. Validation: compileall, targeted runner tests, and AutoResearch/Cost/Tinker cluster `50 passed`.
- 2026-05-19 05:56 PDT — Draft PR #46 opened: Manager now handles no-stdin data prompts as Mode C/no-data and stops before DecisionEngine/Tinker when DataGen curation produces no trainable rows. Validation: Manager/DataGen guard suite `33 passed, 3 skipped`.
- 2026-05-19 05:58 PDT — Expanded local full-stack merge smoke passed: #39 + #45 + #43 + #40 + #42 + #44 + #46 merged cleanly, then broad non-live suite passed with `186 passed, 4 skipped`.
- 2026-05-19 05:58 PDT — Live 1-iteration AutoResearch candidate-config smoke passed on the expanded stack: offline synthetic DataGen -> curation -> DecisionEngine -> baseline Tinker run -> deterministic proposed Tinker run. Captured learning rates were `0.0001` then `0.0002`.
- 2026-05-19 06:02 PDT — Draft PR #47 opened: AutoResearch now compares candidate scores against the current best run, and diary diffs for kept patches use the pre-patch config. Validation: AutoResearch/Cost/Tinker cluster `52 passed`.
- 2026-05-19 06:03 PDT — Expanded stack with #47 merged cleanly and broad non-live suite passed with `188 passed, 4 skipped`.
- 2026-05-19 06:06 PDT — Draft PR #48 opened: AutoResearch JSON config patches now validate through `TrainingConfig.apply_patch` before file writes or Tinker candidate runs. Validation: proposal/AutoResearch/Tinker cluster `85 passed`.
- 2026-05-19 06:06 PDT — Expanded stack with #48 merged cleanly and broad non-live suite passed with `191 passed, 4 skipped`.
- 2026-05-19 06:08 PDT — Closed superseded drafts #36, #37, and #38 with comments and deleted their remote branches. Their scope is now carried by integrated DataGen PR #40.
- 2026-05-19 06:10 PDT — Refreshed PR #40 description to remove stale next-step language, note the closed superseded drafts, and include the latest live/stack validation context.
- 2026-05-19 06:13 PDT — Draft PR #49 opened: spec site now documents the SDK-native Tinker runner, config-patch AutoResearch loop, in-process CostManager accounting, and current shared types. It also fixes an existing spec-site Sidebar lint issue. Validation: `npm run lint`, `npm run build`, and `git diff --check`.
- 2026-05-19 06:22 PDT — Draft PR #50 opened: FastAPI run bridge plus frontend polling hook. The UI remains simulation-only by default and uses the real Manager API when `VITE_API_BASE_URL` is configured. Validation: server compileall, server/Manager pytest `15 passed, 1 skipped`, frontend lint, frontend build.
- 2026-05-19 06:26 PDT — Draft PR #51 opened: integrated full-stack contract smoke on top of AutoResearch patch validation, merging the sibling DataGen/Tinker/Manager guard heads. New test exercises Manager -> real DataGen/curation -> real DecisionEngine -> real AutoResearch graph -> fake Tinker. Broad non-live stack suite passed with `192 passed, 4 skipped`.
- 2026-05-19 06:26 PDT — Retargeted spec docs PR #49 from `codex/datagen-integration` to `codex/full-stack-contract-smoke` so the docs sit behind the code they describe.
- 2026-05-19 06:29 PDT — Draft PR #52 opened: DataGen curation payload now preserves chat `messages` instead of flattening them away. Focused validation passed (`36 passed, 1 skipped` in the broader DataGen/Manager/Tinker cluster).
- 2026-05-19 06:29 PDT — Merged #52 into integration PR #51 and reran the broad non-live stack suite: `193 passed, 4 skipped`.
- 2026-05-19 06:43 PDT — Draft PR #53 opened by parallel worker: AutoResearch compiled graph now has an offline budget-boundary test proving baseline + one fake Tinker candidate stops exactly at the budget before another proposal.
- 2026-05-19 06:53 PDT — Draft PR #54 opened: Mode C web structuring and the Manager trainability gate are now proven together at the Manager boundary. Structured web chat/SFT rows reach DecisionEngine/AutoResearch; raw targetless web is rejected before training.
- 2026-05-19 06:53 PDT — Accidentally started the live HF retrieval suite via an overly broad pytest command and stopped it once it entered the external dataset pull. No Tinker/LLM budget was spent; future broad checks should explicitly ignore `tests/data_generator/test_mode_b_hf_retrieval.py`.
- 2026-05-19 06:54 PDT — Merged #53 and #54 into integration PR #51 and pushed commit `98e2951`. Refreshed #51 description. Non-live stack suite passed with `196 passed, 6 skipped`.
