# Overnight Experiments

Budget cap: $100 total.

| ID | Time PDT | Branch/PR | Kind | Live Services | Status | Estimated Spend | Cumulative Spend | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| setup-001 | 2026-05-19 05:20 | local | credential sanity | none | complete | $0.00 | $0.00 | Keys present without printing values. |
| tinker-001 | 2026-05-19 05:24 | #40/#35 | live 5-step smoke | Tinker | passed | $2.50 | $2.50 | `Qwen/Qwen3.5-9B`, 5 steps, pytest live smoke passed in 60.92s. Artifacts were written under pytest tmpdir. |
| datagen-001 | 2026-05-19 05:29 | #40 | live web/synthetic scout | Tavily, Anthropic | passed | $0.32 | $2.82 | Tavily sanity plus tiny Mode C web acquisition used 2 Tavily requests; tiny Claude synthetic path generated 8 valid records. |
| datagen-002 | 2026-05-19 05:32 | #40 | live synthetic validation | Anthropic | passed | $0.30 | $3.12 | Re-ran tiny strict Claude synthetic generation after schema-label validation hardening: 8 records, validation passed. |
| tinker-002 | 2026-05-19 05:36 | #40/#35 | live partial pipeline | Tinker | passed | $2.50 | $5.62 | Mode C synthetic/offline -> curation JSONL -> DecisionEngine -> 5-step Tinker. Local Tinker cost report: `$0.000138`; conservative ledger keeps smoke estimate. |

## Spend Ledger

Current estimated cumulative spend: **$5.62 / $100.00**

Notes:
- Tinker billing may not be available through the local code, so per-run spend will be estimated conservatively from run type and observed duration unless a real cost is exposed.
- Tavily budget provided by user: $16 ~= 2000 requests. Track requests qualitatively unless API exposes exact usage.
