# Overnight PR Stack

Updated: 2026-05-19 08:08 PDT

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
3. Tinker runner chain:
   - #43 `codex/tinker-single-assistant-rendering` -> #35
   - #57 `codex/tinker-live-loss-metrics` -> #43
   - #60 `codex/tinker-mean-final-metrics` -> #57
   - #62 `codex/tinker-heldout-eval` -> #60
4. DataGen/Manager chain:
   - #40 `codex/datagen-integration` -> #35
   - #42 `codex/e2e-manager-datagen-tinker-boundary` -> #40
   - #44 `codex/mode-c-web-structuring` -> #40
   - #46 `codex/manager-datagen-validation-gate` -> #40
   - #50 `codex/frontend-manager-api-bridge` -> #46
   - #52 `codex/preserve-chat-messages-curation` -> #40
   - #58 `codex/curation-small-splits` -> #52
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
and #62 on top of #51. #63 has been opened on the AutoResearch chain and should
be included in the next local stack smoke.

- `python3 -m compileall src`
- Broad non-live suite excluding live Tinker and live Hugging Face retrieval:
  `204 passed, 7 skipped`
- Live #62 heldout smoke on the 22-row web DataGen dataset completed with
  split 17/2/3, saved checkpoints, `val_loss=3.5944`, and `test_loss=8.8181`.

Current conservative live spend: `$41.04 / $100.00`.
