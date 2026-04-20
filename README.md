# spr26-Team-26 — Nemoral: An Autonomous Self-Improving ML Training Agent

Welcome to the repository for **Team Nemoral** — Stanford CS194, Spring 2026.

> Give it a plain-English prompt and a budget. Get back a trained model. No infra knowledge required.

For full project details, team information, and documentation, visit our **[Wiki Home Page](https://github.com/StanfordCS194/spr26-Team-26/wiki)**.

---

## Quick Links

- [Wiki Home](https://github.com/StanfordCS194/spr26-Team-26/wiki)
- [Team Member Matrix](https://github.com/StanfordCS194/spr26-Team-26/wiki#team-member-matrix)
- [Project Synopsis](https://github.com/StanfordCS194/spr26-Team-26/wiki#project-synopsis)
- [Contact & Communication](https://github.com/StanfordCS194/spr26-Team-26/wiki#communication)

## Retrieval Agent

The repo includes a spec-driven `retrieval_agent` module focused on external data acquisition. It:

- accepts a data-acquisition spec from the manager/planner stage
- discovers source candidates and ranks them
- downloads raw artifacts (for example `CSV`, `JSON`, `JSONL`, `TXT`, `ZIP`, `PDF`, `PARQUET`, `HTML`)
- stores provenance/metadata and generates a human-readable bundle
- outputs a handoff manifest for downstream structuring/curation

Run it with:

```bash
python -m retrieval_agent.agent path/to/spec.json path/to/retrieval_report.json --raw-output-dir path/to/raw
```
