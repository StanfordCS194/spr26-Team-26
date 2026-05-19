# Overnight PR Stack

Updated: 2026-05-19 10:02 PDT

Purpose: give reviewers a merge/review order for the draft PR stack without
having to infer dependencies from GitHub.

## Core Stack

1. #35 `codex/tinker-integration` -> `main`
   - SDK-native Tinker SFT runner and AutoResearch integration root.
2. AutoResearch chain:
   - #39 `codex/cost-manager-tinker-runs` -> #35
   - #45 `codex/autoresearch-candidate-config` -> #39
   - #47 `codex/autoresearch-compare-best` -> #45
   - #48 `codex/autoresearch-validate-patches` -> #47
   - #59 `codex/autoresearch-sft-early-stop` -> #48
   - #61 `codex/autoresearch-min-improvement` -> #59
   - #63 `codex/autoresearch-dataset-splits` -> #61
   - #66 `codex/tinker-primary-metric-plan` -> #63
   - #69 `codex/autoresearch-budget-preflight` -> #66
   - #70 `codex/tinker-run-cost-estimate` -> #69
   - #72 `codex/autoresearch-estimated-spend-floor` -> #70
   - #79 `codex/tinker-hparam-aliases` -> #72
   - #80 `codex/budget-skip-finite-metrics` -> #79
3. Tinker runner chain:
   - #43 `codex/tinker-single-assistant-rendering` -> #35
   - #57 `codex/tinker-live-loss-metrics` -> #43
   - #60 `codex/tinker-mean-final-metrics` -> #57
   - #62 `codex/tinker-heldout-eval` -> #60
   - #65 `codex/tinker-heldout-batches` -> #62
   - #81 `codex/tinker-runner-strict-json` -> #65
4. DataGen/Manager chain:
   - #40 `codex/datagen-integration` -> #35
   - #42 `codex/e2e-manager-datagen-tinker-boundary` -> #40
   - #44 `codex/mode-c-web-structuring` -> #40
   - #46 `codex/manager-datagen-validation-gate` -> #40
   - #73 `codex/manager-json-robustness` -> #46
   - #50 `codex/frontend-manager-api-bridge` -> #46
   - #74 `codex/api-run-output-isolation` -> #50
   - #76 `codex/frontend-dataset-source` -> #74
   - #77 `codex/api-run-artifacts` -> #76
   - #82 `codex/api-artifact-download-path` -> #77
   - #78 `codex/api-progress-cancel` -> #77
   - #52 `codex/preserve-chat-messages-curation` -> #40
   - #58 `codex/curation-small-splits` -> #52
   - #64 `codex/curation-source-splits` -> #58
   - #67 `codex/hf-live-test-controls` -> #64
   - #68 `codex/hf-parser-free-text` -> #67
   - #75 `codex/mode-c-structuring-leaf` -> #68
5. Integration/docs:
   - #51 `codex/full-stack-contract-smoke` -> #48
   - #49 `codex/spec-tinker-sdk-docs` -> #51
   - #55 `codex/autoresearch-budget-stop-graph-v2` -> #51
   - #56 `codex/mode-c-web-manager-boundary-v2` -> #51
   - #71 `codex/full-stack-budget-contract` -> #51
   - #41 `codex/overnight-coordination` -> #35

## Review Notes

- #51 is an integration proof branch, not a normal isolated feature PR. Review
  it after its ancestor feature branches, or refresh/replace it once ancestors
  land.
- #55 and #56 are intentional replacement PRs for #53 and #54. The earlier PRs
  were accidentally marked merged when their base branch was temporarily pushed
  with leaf commits, then reset. Use #55/#56 for review.
- #49 has been refreshed on top of current #51. Its PR diff should be limited
  to `spec-site/components/Sidebar.tsx` and `spec-site/content/spec.ts`.
- Vercel checks remain noisy/external: some old projects fail or rate-limit
  independently of local validation. Do not treat those as code evidence unless
  the project owners reconnect or reconfigure the Vercel integrations.

## Latest Local Stack Validation

Unpublished local stack currently includes #55 through #78, including #73/#74/#75/#76/#77/#78
on top of #51.

- `python3 -m compileall src`
- Full non-live suite with live Tinker/HF cases skipped by default:
  `225 passed, 9 skipped`
- Live #62 heldout smoke on the 22-row web DataGen dataset completed with
  split 17/2/3, saved checkpoints, `val_loss=3.5944`, and `test_loss=8.8181`.
- Live full AutoResearch graph smoke with #62/#63 completed with split 17/2/3
  reaching both baseline/candidate Tinker calls; `learning_rate=5e-4` improved
  heldout scalar from `0.0675` to `0.2300`.
- Live Mode B partial with #64 pulled `SetFit/sst2` and verified curation
  preserves source split counts as 2/2/2.
- Live #65 direct Tinker smoke completed on the 22-row web DataGen dataset with
  split 17/2/3 and `batch_size=2`.
- Local stack after adding #66 passed compileall and the broad non-live suite
  excluding live Tinker/HF retrieval with `208 passed, 7 skipped`.
- #67 focused suite passed with `21 passed, 3 skipped`; tiny opt-in live HF
  smoke passed with one dataset and a three-row cap.
- Local stack after adding #67 passed the broad non-live suite with only live
  Tinker excluded: `212 passed, 9 skipped`.
- #68 fixed the free-text HF parser false positive found by the live Mode B
  run. Local stack with #68 passed `214 passed, 9 skipped`; corrected live
  Mode B/Tinker rerun selected only `SetFit/sst2`, had no curation issues, and
  kept the `learning_rate=5e-4` candidate.
- #69 added AutoResearch budget preflight before Tinker launch. It uses explicit
  per-launch `estimated_run_cost_usd` when present, writes cancelled artifacts
  for skipped runs, and labels budget skips separately from catastrophic model
  failures. The local stack caught and fixed an initial over-conservative use of
  plan-level `estimated_cost`.
- #70 makes DecisionEngine populate `estimated_run_cost_usd` for bounded Tinker
  runs. #71 updates the integration proof test so its fake budget/cost scale
  remains compatible with the new preflight estimate while still proving the
  budget boundary.
- #72 records launched Tinker runs at a conservative spend floor of
  `max(reported_cost_usd, estimated_run_cost_usd)`. Budget preflight skips
  still record zero because no SDK call happened.
- Live #72 budget-floor validation completed with baseline + one candidate
  Tinker run and stopped at the exact software budget `$2.24`.
- #73 hardens Manager reasoning parsing for fenced JSON responses. Focused
  branch tests passed (`14 passed, 1 skipped`), local stack Manager/full-stack
  contract tests passed (`15 passed, 1 skipped`), and the live Manager Mode B
  validation reached baseline + one candidate Tinker run before stopping at
  the exact `$2.24` software budget.
- #74 adds run-scoped output routing for API/background Manager runs. It is
  stacked on #50 because it depends on the FastAPI bridge. Validation passed
  targeted API/runtime tests, broad no-live branch tests with the old ungated
  HF retrieval file ignored (`176 passed, 7 skipped`), frontend lint, and
  frontend build.
- #76 adds frontend/API dataset-source pass-through on top of #74. It accepts
  existing local paths and Hugging Face sources, preserves canonical `hf://...`
  through Manager, and passed compileall, focused API/Manager tests
  (`26 passed, 1 skipped`), broad no-live branch tests (`187 passed, 7 skipped`),
  frontend lint/build, and browser form sanity.
- #77 surfaces completed-run artifacts through the API/frontend bridge. It adds
  typed artifact summaries, allowlisted downloads for run-local manifest,
  metrics, metrics log, sample, and diary files, and final-results UI/export
  metadata. Validation passed compileall, focused API/DataSource/Manager tests
  (`27 passed, 1 skipped`), broad no-live branch tests (`188 passed, 7 skipped`),
  frontend lint, and frontend build.
- #78 adds live API progress refresh and cooperative cancellation on top of
  #77. It routes observability logs into the run output root, refreshes API
  state from run-local logs/diary/Tinker metrics, exposes
  `POST /api/runs/{run_id}/cancel`, and adds frontend Cancel/Cancelling/
  Cancelled states. Validation passed compileall, focused progress/cancel
  tests (`33 passed`), broad no-live branch tests (`194 passed, 7 skipped`),
  frontend lint/build, and browser cancellation UI sanity.
- #75 resolves the DataGen stack topology issue where #44 was a sibling of the
  latest DataGen leaf chain. It merges the existing Mode C teacher-backed web
  structuring work onto #68 and passed the broad no-live suite (`181 passed,
  9 skipped`).
- The unpublished composition branch now includes #55 through #76, including
  #73/#74/#75/#76. After merging #76 on top of the composed stack, compileall
  and the full no-live test suite passed (`244 passed, 9 skipped`).
- After merging #77 into the same unpublished composition branch, compileall,
  the full no-live Python suite (`245 passed, 9 skipped`), and frontend
  lint/build all passed. The only remaining dirty file in that temporary
  worktree is generated `configs/current.json`.
- After merging #78 into the unpublished composition branch, compileall, the
  full no-live Python suite (`251 passed, 9 skipped`), and frontend lint/build
  all passed. The only conflicts were in `src/autoresearch/autoresearch.py` and
  `src/tinker_api/sft_runner.py`, where the resolution preserved the newer
  stack's budget/split/heldout behavior and wrapped it with #78's cancellation
  hooks.
- A live API Mode B validation on the unpublished stack confirmed #78's progress
  refresh behavior: the API streamed Manager/DataGen/AutoResearch/Tinker progress,
  five Tinker metric points, and partial artifacts before failing on a separate
  Tinker config-name bug (`epochs` vs `num_epochs`). That bug should become the
  next scoped AutoResearch compatibility PR.
- #79 fixes that config-name bug by canonicalizing Tinker aliases at the
  AutoResearch proposal/search boundary. Branch validation passed focused and
  no-live suites; next step is composing it into the unpublished full-stack
  worktree and rerunning the live API Mode B path.
- The unpublished composition branch now includes #79 as well. Compileall and
  the full no-live Python suite passed (`255 passed, 9 skipped`). The live API
  Mode B rerun completed with baseline + one candidate Tinker job, 10 streamed
  metric points, and run artifacts exposed before stopping at the `$2.24`
  software budget limit.
- #80 fixes a follow-on artifact hygiene issue found in that live run:
  budget-preflight skip metrics no longer serialize `Infinity`. The unpublished
  composition branch includes #80 and still passes compileall plus the full
  no-live Python suite (`255 passed, 9 skipped`).
- #81 applies the same strict-artifact principle inside the Tinker runner for
  non-finite SDK training/heldout losses. The unpublished composition branch
  includes #81 and passes the full no-live Python suite (`257 passed, 9 skipped`).
- #82 fixes API-provided artifact download links to match the mounted FastAPI
  route. The unpublished composition branch includes #82 and still passes API
  tests plus the full no-live Python suite (`257 passed, 9 skipped`).
- Live cancellation validation on the unpublished composition branch passed:
  the API cancelled an active baseline Tinker run, returned terminal
  `cancelled`, and kept result null while still exposing downloadable artifacts
  after the runner finished its safe boundary. It also exposed the next API
  artifact-refresh issue: while a newer experiment directory exists but has not
  written files yet, live refresh can temporarily replace working artifact links
  with placeholder/missing files.
- #83 fixes that artifact-refresh issue on top of #78. It keeps metrics/stage
  refresh eager but switches downloadable artifacts only after the newer
  experiment has manifest, metrics, metrics_log, and sample files. It composes
  cleanly with #82 and the later unpublished stack; the composed no-live suite
  passed with `253 passed, 6 skipped`.
- A live cancellation rerun after composing #83 confirmed the intended browser
  contract: no placeholder artifact bundle during `cancelling`, then real
  downloadable artifacts at terminal `cancelled`.

Current conservative live spend: `$83.64 / $100.00`.
