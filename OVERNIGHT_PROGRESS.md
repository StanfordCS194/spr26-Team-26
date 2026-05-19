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
