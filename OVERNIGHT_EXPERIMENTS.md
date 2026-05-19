# Overnight Experiments

Budget cap: $100 total.

| ID | Time PDT | Branch/PR | Kind | Live Services | Status | Estimated Spend | Cumulative Spend | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| setup-001 | 2026-05-19 05:20 | local | credential sanity | none | complete | $0.00 | $0.00 | Keys present without printing values. |
| tinker-001 | 2026-05-19 05:24 | #40/#35 | live 5-step smoke | Tinker | passed | $2.50 | $2.50 | `Qwen/Qwen3.5-9B`, 5 steps, pytest live smoke passed in 60.92s. Artifacts were written under pytest tmpdir. |
| datagen-001 | 2026-05-19 05:29 | #40 | live web/synthetic scout | Tavily, Anthropic | passed | $0.32 | $2.82 | Tavily sanity plus tiny Mode C web acquisition used 2 Tavily requests; tiny Claude synthetic path generated 8 valid records. |
| datagen-002 | 2026-05-19 05:32 | #40 | live synthetic validation | Anthropic | passed | $0.30 | $3.12 | Re-ran tiny strict Claude synthetic generation after schema-label validation hardening: 8 records, validation passed. |
| tinker-002 | 2026-05-19 05:36 | #40/#35 | live partial pipeline | Tinker | passed | $2.50 | $5.62 | Mode C synthetic/offline -> curation JSONL -> DecisionEngine -> 5-step Tinker. Local Tinker cost report: `$0.000138`; conservative ledger keeps smoke estimate. |
| datagen-003 | 2026-05-19 05:38 | #40 | live web curation guard | Tavily | passed | $0.02 | $5.64 | Two tiny Tavily queries total after first crawl miss; final run got 3 results, 2 pages, and curation rejected targetless web records. |
| tinker-003 | 2026-05-19 05:45 | #43/#35 | live renderer validation | Tinker | passed | $1.00 | $6.64 | 2-step live Tinker smoke after switching to `LAST_ASSISTANT_MESSAGE`; passed in 53.74s with no renderer warning. |
| datagen-004 | 2026-05-19 05:47 | #44/#40 | live web structuring validation | Anthropic | passed | $0.30 | $6.94 | One manual web excerpt -> teacher structuring -> 4 schema-valid chat/SFT rows. No Tavily/Tinker used. |
| tinker-004 | 2026-05-19 05:58 | expanded stack #39/#45/#43/#40/#42/#44/#46 | live candidate-config graph smoke | Tinker | passed | $2.00 | $8.94 | Offline synthetic DataGen -> curation -> DecisionEngine -> AutoResearch baseline plus one deterministic proposal. Captured Tinker learning rates: `0.0001` then `0.0002`; both capped at 2 steps. Local Tinker cost report: `$0.000284`. |
| docs-001 | 2026-05-19 06:13 | #49 | spec-site validation | none | passed | $0.00 | $8.94 | `npm run lint`, `npm run build`, and `git diff --check` passed after updating the Tinker/AutoResearch spec text and fixing the existing Sidebar lint issue. |
| api-001 | 2026-05-19 06:22 | #50 | backend/frontend API validation | none | passed | $0.00 | $8.94 | FastAPI run bridge tests plus Manager tests passed (`15 passed, 1 skipped`); frontend `npm run lint` and `npm run build` passed. |
| stack-001 | 2026-05-19 06:26 | #51 | full-stack offline contract | none | passed | $0.00 | $8.94 | New Manager -> DataGen -> DecisionEngine -> real AutoResearch graph -> fake Tinker smoke passed; broad non-live suite passed with `192 passed, 4 skipped`. |
| datagen-005 | 2026-05-19 06:29 | #52/#51 | chat-message curation contract | none | passed | $0.00 | $8.94 | DataGen preserves chat `messages` through curation; focused cluster passed `36 passed, 1 skipped`; updated #51 broad stack passed `193 passed, 4 skipped`. |
| stack-002 | 2026-05-19 06:43 | #53 | AutoResearch budget-boundary graph validation | none | passed | $0.00 | $8.94 | Parallel worker added offline compiled-graph coverage proving baseline plus one fake Tinker candidate stops at the budget boundary before another proposal. |
| stack-003 | 2026-05-19 06:53 | #54 | Mode C web/Manager boundary validation | none | passed | $0.00 | $8.94 | Structured web chat/SFT rows reach DecisionEngine/AutoResearch; raw targetless web is rejected before training. Focused and neighboring suites passed. |
| hf-001 | 2026-05-19 06:53 | #54/#51 | accidental broad HF retrieval start | Hugging Face public dataset pull | stopped | $0.00 | $8.94 | An overly broad pytest command entered the existing live HF retrieval suite. Stopped it once identified; no Tinker, Anthropic, or Tavily spend. |
| stack-004 | 2026-05-19 06:54 | #51 temporary local merge | refreshed full-stack non-live validation | none | passed | $0.00 | $8.94 | Temporarily folded #53 and #54 into #51; `python3 -m compileall src`; non-live suite excluding live Tinker and live HF retrieval passed with `196 passed, 6 skipped`. This integration push was reset because it auto-marked leaf drafts as merged. |
| hygiene-001 | 2026-05-19 06:58 | #55/#56/#51 | PR hygiene correction | none | complete | $0.00 | $8.94 | Reset #51 back to `680c5c5`; opened replacement drafts #55 and #56 so the leaf changes remain reviewable and unmerged; deleted obsolete remote heads for #53/#54. |
| datagen-006 | 2026-05-19 07:04 | #51 stack | live web structuring partial | Tavily, Anthropic | passed | $0.35 | $9.29 | One Tavily query -> 3 results -> 2 crawled pages -> 6 Claude-structured chat/SFT rows -> curation passed. |
| tinker-005 | 2026-05-19 07:07 | #51 stack/#43 | live web-data Tinker smoke | Tinker | passed with metric bug | $1.00 | $10.29 | 2-step `Qwen/Qwen3.5-9B` run on live web-structured JSONL completed and saved checkpoint/sample, but runner reported bogus zero loss. Local Tinker cost report: `$0.000047`. |
| tinker-006 | 2026-05-19 07:13 | #57 | live loss metric validation | Tinker | passed | $0.50 | $10.79 | 1-step `Qwen/Qwen3.5-9B` validation after #57 fix reported real `train_loss=24.81933307647705`. Local Tinker cost report: `$0.000025`. |
| datagen-007 | 2026-05-19 07:19 | #58 | curation split validation | none | passed | $0.00 | $10.79 | Local tests prove 3-9 record curated datasets reserve val/test rows; adjacent DataGen/Manager suite passed `43 passed, 4 skipped`. |
| stack-005 | 2026-05-19 07:22 | local only #55/#56/#57/#58 | unpublished stack composition smoke | none | passed | $0.00 | $10.79 | Throwaway worktree merged active leaf PRs on top of #51 without pushing; compileall passed; non-live suite excluding live Tinker/HF passed `193 passed, 6 skipped`. |

## Spend Ledger

Current estimated cumulative spend: **$10.79 / $100.00**

Notes:
- Tinker billing may not be available through the local code, so per-run spend will be estimated conservatively from run type and observed duration unless a real cost is exposed.
- Tavily budget provided by user: $16 ~= 2000 requests. Track requests qualitatively unless API exposes exact usage.
