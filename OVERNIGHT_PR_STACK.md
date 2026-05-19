# Overnight PR Stack

Updated: 2026-05-19 14:39 PDT

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
   - #84 `codex/manager-noninteractive-invoke` -> #73
   - #50 `codex/frontend-manager-api-bridge` -> #46
   - #74 `codex/api-run-output-isolation` -> #50
   - #76 `codex/frontend-dataset-source` -> #74
   - #77 `codex/api-run-artifacts` -> #76
   - #82 `codex/api-artifact-download-path` -> #77
   - #78 `codex/api-progress-cancel` -> #77
   - #83 `codex/api-artifact-stable-refresh` -> #78
   - #86 `codex/api-preserve-terminal-metrics` -> #83
   - #87 `codex/frontend-cancelled-artifacts-budget` -> #83
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
   - #85 `codex/mode-c-web-structuring-budget-cap` -> #56
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

Unpublished local stack currently includes #55 through #87, including the API,
DataGen, AutoResearch, Tinker, Manager, and frontend leaves listed above. The
latest composed validation after #86/#87 passed with `260 passed, 6 skipped`.

- `python3 -m compileall src`
- Full non-live suite with live Tinker/HF cases skipped by default:
  `260 passed, 6 skipped`
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
- A live API Mode C/no-data run on the composed stack completed end-to-end with
  live web acquisition, teacher structuring, curation, baseline plus one
  candidate Tinker run, and API artifact downloads. It exposed the legacy stdin
  prompt in the server path.
- #84 fixes that Manager prompt issue on top of #73 by making
  `invoke_manager_graph()` noninteractive by default and preserving an explicit
  `interactive_data_prompt=True` opt-in. It composes cleanly with the local stack
  after resolving only the `typing` import conflict; the composed no-live suite
  passed with `256 passed, 6 skipped`.
- #85 hardens Mode C web teacher structuring on top of #75. It uses explicit
  `max_records`, `DATA_GENERATOR_WEB_STRUCTURING_MAX_RECORDS`,
  `DATA_GENERATOR_SYNTHETIC_EXAMPLES`, or config-level example counts before
  falling back to 24, and it retries malformed teacher JSON once with a repair
  prompt before falling back to unstructured failure. It composes cleanly with
  the local stack; the composed no-live suite passed with `260 passed, 6 skipped`.
  The capped live no-data API rerun completed with requested caps `[8, 8, 8]`,
  5 curated rows, baseline plus one Tinker candidate, and five working artifact
  downloads.
- #86 preserves streamed API metrics when a run reaches terminal complete state;
  the fallback summary metric remains for runs without streamed points. It
  composes cleanly with the local stack; focused API tests passed (`17 passed`)
  and the full composed no-live suite remained `260 passed, 6 skipped`.
- #87 makes cancelled-run artifacts visible in the frontend when the backend
  exposes them and lowers the UI budget minimum to `$1` for cheap smoke runs.
  Frontend lint/build passed on the branch and in the composed stack.
- #88 passes the API/frontend `task_type` selection into Manager planning as a
  normalized operator hint. It deliberately does not override the inferred ML
  task, because `fine-tuning` is a workflow choice rather than a concrete task
  type. Branch validation passed compileall, focused API/Manager tests
  (`31 passed, 1 skipped`), and broad no-live tests (`197 passed, 6 skipped`).
  The unpublished composition branch now includes #88 after resolving overlap
  with #84 by preserving both `interactive_data_prompt` and `task_type_hint`;
  focused API/Manager tests passed (`37 passed, 1 skipped`) and the full
  no-live Python suite passed (`261 passed, 6 skipped`).
- #89 fixes frontend artifact download links on top of #87. Backend artifact
  paths are already rooted at `/api/runs/...`; the frontend now avoids doubling
  `/api` when `VITE_API_BASE_URL=/api`, while still supporting full-origin API
  bases. Branch and composed-stack validation passed frontend lint,
  `VITE_API_BASE_URL=/api npm run build`, and a direct resolver contract check.
- #90 keeps Mode C web structuring `auto` trainable when teacher structuring is
  unavailable or produces zero valid rows. It falls back to synthetic chat/SFT
  rows, preserves web source URLs and structuring issues in metadata/report, and
  leaves `required` mode strict. Branch no-live suite passed (`200 passed,
  6 skipped`); composed stack through #90 passed (`262 passed, 6 skipped`).
- #49 now also updates root `README.md` and `Home.md` to describe the current
  Tinker chat/SFT V1 scope instead of stale image-classification, pretraining,
  and REST-job promises. Spec-site lint/build passed after the docs refresh.
- #91 guards AutoResearch eval-suite adaptation so iteration 10 does not
  unexpectedly require or spend Anthropic credits in default no-live runs.
  `AUTORESEARCH_EVAL_ADAPTATION=auto` now adapts only when
  `ANTHROPIC_API_KEY` is configured, `off` disables it, and `required` preserves
  the explicit live path. Branch no-live suite passed (`168 passed, 4 skipped`);
  composed stack through #91 passed (`264 passed, 6 skipped`). Follow-up update:
  `NO_SPEND=1` now skips eval adaptation before any live path even when a key is
  loaded or adaptation is marked required; full branch AutoResearch tests passed
  (`42 passed`).
- #92 adds a local Manager reasoner on top of the noninteractive Manager path.
  `MANAGER_REASONER=auto` keeps Claude when a key is configured, but falls back
  to deterministic local planning without a key or under `NO_SPEND=1`.
- #93 makes Mode B's offline fallback produce trainable `input`/`output` rows
  instead of inert examples, and curation preserves dataset/source metadata.
- #94 adds `TINKER_BACKEND=dry_run` so the API/full graph can exercise the
  Tinker artifact contract without constructing the SDK client or spending live
  budget.
- #95 adds a local AutoResearch proposer. After compose smoke, it was updated
  to avoid no-op proposals by sampling alternate random values, stepping
  discrete candidates to neighbors, and forcing integer perturbations to move
  when bounds allow. Branch validation passed (`67 passed, 3 skipped`); composed
  AutoResearch/Tinker passed (`74 passed`). Follow-up update: after rebasing on
  #91, `NO_SPEND=1` now forces the local proposer even if
  `AUTORESEARCH_PROPOSER=claude` is accidentally set; branch AutoResearch/Tinker
  validation passed (`65 passed`).
- #71 was updated for compatibility with #95: the fake-Claude full-stack budget
  contract now explicitly sets `AUTORESEARCH_PROPOSER=claude` when it
  monkeypatches `propose_hypothesis()`. This keeps the test's intended path
  stable while default no-key runs use the local proposer.
- The unpublished composition branch through #95 now passes the full no-live
  Python suite (`281 passed, 8 skipped`) and a no-spend API product smoke with
  local Manager/proposer plus dry-run Tinker.
- #96 fixes a data-contract hole in Mode A. Plain `.txt` lines are now
  source-only rather than fabricated `output="unknown"` supervised examples,
  and both curation and the Tinker runner reject chat records with no assistant
  target after a user. It is stacked on #90 and composes with the local stack
  after preserving later Tinker runner split/heldout/dry-run behavior.
- #97 fixes a DecisionEngine contract mismatch on top of #79. Low-budget plans
  no longer switch to `strategy="pre-train"` while keeping `backend="tinker_sft"`;
  they stay on the supported Tinker SFT path and let budget preflight handle
  affordability.
- #94 was updated with a graph-level dry-run smoke that exercises the compiled
  AutoResearch graph against the real dry-run Tinker backend. It explicitly
  forces the fake-Claude proposal path for compatibility with #95, blocks live
  SDK dependency loading, and now composes with the local stack (`286 passed,
  8 skipped` full no-live suite).
- #98 is an integration-only stacked draft PR against the pushed
  `codex/local-leaf-stack-smoke` base branch. It intentionally is not based on
  `main` because its API-level HF dry-run smoke depends on the current draft
  stack (#92-#97 plus the earlier API/data-source/runtime branches). Diff
  against that integration base is one focused test in `tests/test_server_api.py`.
- #99 is stacked on #98 and adds the parallel local-file API dry-run smoke.
  Together #98/#99 cover the two user-data entry paths that should work without
  live credentials: Hugging Face source and existing local JSONL.
- #100 is now stacked on #99 for the no-data path, making the API dry-run smoke
  chain linear: #98 Hugging Face source -> #99 local JSONL -> #100 no-data Mode
  C. Local composition of all three passed the full no-live Python suite (`289
  passed, 8 skipped`), and #100 focused validation after retargeting passed
  (`21 passed`).
- #101 is stacked on #90 and hardens Mode C offline/no-spend semantics. A shared
  policy now makes `NO_SPEND=1`, `DATA_GENERATOR_OFFLINE=1`, and
  `DATA_GENERATOR_SYNTHETIC_OFFLINE=1` all block web acquisition and teacher
  calls, including `scrape_web()` and web structuring. Focused and boundary
  validation passed (`35 passed`), with no live spend.
- #102 is an independent main-based guard for the legacy low-level
  `src/tinker_api/tinker_api.py` wrapper. It blocks direct SDK operations under
  `NO_SPEND` or `TINKER_BACKEND=dry_run`, including `run_training_loop`, while
  leaving ledger/cancel helpers available. Focused validation passed
  (`26 passed`).
- #103 is an independent main-based guard for legacy `autoresearch/prepare.py`.
  Offline/no-spend mode now permits cached shards but blocks both `download_data`
  and direct `download_single_shard` network downloads. Focused no-network
  validation passed (`7 passed`).
- Local composition of #98/#99/#100 with #101/#102/#103 passed the focused
  no-spend guard cluster (`81 passed`) and the full no-live suite with live
  credentials unset plus `UV_NO_NETWORK=1` (`308 passed, 8 skipped`).
- #104 is stacked on #100 and hardens the API dry-run smokes by installing
  explicit failure fences around hidden live seams: stdin, Claude, HF dataset
  fetch, web search/crawl, Mode C teacher structuring, synthetic teacher
  creation, `requests.get`, and Tinker SDK loading. Local composition through
  #104 again passed the focused guard cluster (`81 passed`) and full no-live
  suite (`308 passed, 8 skipped`).
- #105 is stacked on #89 and closes the frontend's default simulation gap. The
  app now targets the real Manager API at `/api` unless
  `VITE_USE_SIMULATION=1` is set; the README documents default backend dev,
  alternate API bases, and explicit static demo mode. Frontend lint and three
  build modes passed.
- #94 was updated again so `NO_SPEND=1` alone selects the Tinker dry-run
  backend. This closes the SDK-native SFT runner gap that the low-level
  `tinker_api.py` guard did not cover.
- #104 now includes the latest #94 runner guard in its stack and asserts the
  three API dry-run smokes do not set `TINKER_BACKEND`; `NO_SPEND=1` is the only
  Tinker dry-run selector. Focused branch validation passed (`52 passed`), and
  the local no-spend guard composition passed the full no-live suite (`309
  passed, 8 skipped`).
- #92 was updated so `NO_SPEND=1` overrides explicit Claude Manager mode before
  a client can be constructed. Focused Manager validation passed (`21 passed, 1
  skipped`).
- #93 was updated so Mode B Hugging Face acquisition treats both `NO_SPEND` and
  `DATA_GENERATOR_OFFLINE` as hard offline flags. Focused Mode B validation
  passed (`12 passed, 2 skipped`).
- #101 was updated so direct Mode C search/crawl helpers honor the same offline
  policy as the graph nodes. Focused Mode C validation passed (`29 passed`).
- #49 was updated to document local/no-spend Manager, AutoResearch, Tinker, Mode
  B, and Mode C behavior in the spec site. Spec-site lint and build passed.
- The refreshed local no-spend guard composition merged the latest #92, #93,
  #94, #95, #101, #104, and related guard branches. Full no-live suite with
  live credentials unset and `UV_NO_NETWORK=1` passed (`318 passed, 8 skipped, 5
  warnings`).
- #86 was updated after a product UI smoke exposed budget-preflight sentinel
  metrics leaking into the dashboard as `1e9` loss. The API now filters
  budget-preflight skipped experiment/diary rows from user-facing metric and
  iteration views while leaving the strict artifacts intact on disk.
- Local product UI composition of the no-spend backend stack plus #105 frontend
  and updated #86 passed a browser smoke: FastAPI+Vite completed a `NO_SPEND=1`
  run, the dashboard displayed real final metrics, all five artifact endpoints
  returned 200, frontend build passed, and the full no-live Python suite passed
  (`319 passed, 8 skipped, 5 warnings`).
- #91 was updated again to close direct Anthropic seams outside the LangGraph
  path. `propose_hypothesis()`, `adapt_eval_suite()`, CLI `--strategy claude`,
  and direct `ClaudeProposalStrategy.propose()` now fail before client
  construction under `NO_SPEND=1`.
- #102 was updated so the legacy low-level Tinker wrapper lazy-loads `tinker`
  only after no-spend/dry-run guards pass. This prevents dry-run/no-spend
  imports from touching the SDK at all.
- The refreshed local guard composition including #91 and #102 passed focused
  AutoResearch/Tinker guard validation (`132 passed, 3 skipped`) and full
  no-live validation (`322 passed, 8 skipped, 5 warnings`).
- #102 was updated again after composed-suite ordering exposed stale Tinker
  cancel/spend state across module reloads. Its test fixture now restores the
  previous `src.tinker_api.tinker_api` module object and the cached parent
  package attribute after each mocked import. Branch validation passed
  (`27 passed`), the refreshed guard composition passed selected validation
  (`152 passed, 3 skipped`) plus full no-live validation (`322 passed, 8
  skipped, 5 warnings`), and the product UI composition passed the selected
  backend/API suite (`153 passed, 3 skipped`) plus frontend build.
- #106 is a new main-based policy guard for live tests already present on
  `main`: live Claude proposal integration and live Hugging Face retrieval skip
  under `NO_SPEND=1`, even if credentials are loaded. The newer live-test
  branches were patched with the same policy: #67 for Hugging Face live
  retrieval, #91 for Claude proposal integration, #94 for live Tinker smoke,
  and #101 for live Mode C web tests. Targeted no-spend validation with fake
  credentials and live opt-ins produced `7 skipped`, and the refreshed guard
  composition still passed the full no-live suite (`322 passed, 8 skipped, 5
  warnings`).
- #49 was updated again to clean stale source-level proposal-loop docstrings.
  The docs now distinguish the legacy proposal-only CLI scaffold from the full
  LangGraph path that owns SDK-native Tinker execution/evaluation/decisions.
  Validation passed `compileall` and proposal-loop tests (`23 passed, 3
  skipped`).
- #102 was updated again so `TINKER_BACKEND=dry-run` normalizes to the same
  no-live guard as `TINKER_BACKEND=dry_run`. Focused low-level Tinker API guard
  validation passed (`29 passed`).
- #107 is a new stacked draft PR on #104 for cancellation semantics in the
  SDK-native Tinker runner. It keeps active job registration through
  finalization, skips post-cancel live SDK save/sample/heldout calls, mirrors
  cancellation in dry-run/NO_SPEND, and adds FastAPI cancellation coverage.
  Branch validation passed the Tinker runner suite (`32 passed, 1 skipped`) and
  full server API suite with required optional deps (`21 passed`). Local
  compositions with the latest #102/#107 changes passed targeted
  guard/cancellation/API clusters (`66 passed` each) and the refreshed full
  no-live guard suite (`326 passed, 8 skipped, 5 warnings`).
- #108 is a new stacked draft PR on #105 for frontend product honesty. Completed
  runs no longer show a fake deploy action, and user-facing cost labels now say
  `Budget Used` with tooltip copy that distinguishes budget accounting from
  provider-billed spend in dry-run/no-spend mode. Branch and product composition
  frontend lint/build passed, plus a simulation browser smoke.
- #109 is a new stacked draft PR on #108 for metric-label honesty. Backend run
  state now exposes `primaryMetric` and `primaryMetricLabel` for metric points
  and AutoResearch iteration rows while keeping legacy `accuracy`/`f1` fields
  as compatibility shims. The frontend renders the API-provided metric label in
  cards, charts, final results, and cancelled artifacts, so Tinker SFT proxy
  scores are no longer presented as validation accuracy or F1. Branch
  validation passed compileall, server API tests (`17 passed`), frontend
  lint/build, and a simulation browser smoke. Product UI composition through
  #109 passed server API tests (`22 passed`) plus frontend lint/build.
- #110 is a new stacked draft PR on #109 for run provenance. API state derives
  compact spend/backend/data-mode/budget-skip/live-service evidence from
  Tinker manifests and DataGen debug artifacts, and the dashboard renders it as
  provenance badges. Branch validation passed compileall, no-network server API
  tests (`18 passed`), frontend lint/build, and a simulation browser smoke with
  no new console errors.
- #92 was updated after audit to close the local/no-spend half of #88's API
  task-type hint. The local Manager planner now accepts `task_type_hint`:
  `classification -> text-classification`, `regression -> custom`, and
  `fine-tuning -> prompt/data inference`. Branch validation passed Manager
  tests (`25 passed, 1 skipped`). A throwaway #88 + updated #92 composition
  resolved the expected Manager conflicts and passed Manager/API tests (`44
  passed, 1 skipped`) plus a direct `NO_SPEND=1` fake-key check.
- #86 was updated after audit to distinguish a Tinker budget-preflight skip
  from successful training. When the result points at a manifest with
  `budget_preflight_skipped=true` and `termination_reason=budget_limit`, the
  API now returns terminal `cancelled`, keeps artifacts/downloads and the budget
  reason, and avoids repopulating sentinel metrics from the Manager result.
  Manager logs now say skipped/stopped by budget rather than always training
  complete. Validation passed server API tests (`19 passed`) and Manager tests
  (`13 passed, 1 skipped`).

Current conservative live spend: `$94.14 / $100.00`.
