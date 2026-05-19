# Overnight PR Stack

Updated: 2026-05-19 08:50 PDT

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
3. Tinker runner chain:
   - #43 `codex/tinker-single-assistant-rendering` -> #35
   - #57 `codex/tinker-live-loss-metrics` -> #43
   - #60 `codex/tinker-mean-final-metrics` -> #57
   - #62 `codex/tinker-heldout-eval` -> #60
   - #65 `codex/tinker-heldout-batches` -> #62
4. DataGen/Manager chain:
   - #40 `codex/datagen-integration` -> #35
   - #42 `codex/e2e-manager-datagen-tinker-boundary` -> #40
   - #44 `codex/mode-c-web-structuring` -> #40
   - #46 `codex/manager-datagen-validation-gate` -> #40
   - #50 `codex/frontend-manager-api-bridge` -> #46
   - #52 `codex/preserve-chat-messages-curation` -> #40
   - #58 `codex/curation-small-splits` -> #52
   - #64 `codex/curation-source-splits` -> #58
   - #67 `codex/hf-live-test-controls` -> #64
   - #68 `codex/hf-parser-free-text` -> #67
5. Integration/docs:
   - #51 `codex/full-stack-contract-smoke` -> #48
   - #49 `codex/spec-tinker-sdk-docs` -> #51
   - #55 `codex/autoresearch-budget-stop-graph-v2` -> #51
   - #56 `codex/mode-c-web-manager-boundary-v2` -> #51
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

Unpublished local stack currently includes #55, #56, #57, #58, #59, #60, #61,
#62, #63, #64, #65, #66, #67, #68, and #69 on top of #51.

- `python3 -m compileall src`
- Full non-live suite with live Tinker/HF cases skipped by default:
  `219 passed, 9 skipped`
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

Current conservative live spend: `$58.54 / $100.00`.
