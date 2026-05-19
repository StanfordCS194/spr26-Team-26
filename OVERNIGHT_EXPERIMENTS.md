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

## Spend Ledger

Current estimated cumulative spend: **$8.94 / $100.00**

Notes:
- Tinker billing may not be available through the local code, so per-run spend will be estimated conservatively from run type and observed duration unless a real cost is exposed.
- Tavily budget provided by user: $16 ~= 2000 requests. Track requests qualitatively unless API exposes exact usage.
