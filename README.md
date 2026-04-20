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

## Data Agent

The repo includes a query-aware `data_agent` module for turning mixed raw user data into a trainable dataset. It:

- accepts a natural-language training query plus one or more files or folders
- ingests `CSV`, `JSON`, `JSONL`, `TXT`, and `MD`
- infers a target schema from the query and sampled source evidence
- extracts examples from tabular files, key-value text blocks, and class-labeled directories
- standardizes and exports `train.jsonl`, `val.jsonl`, `test.jsonl`, and `metadata.json`

The Data Agent is set up to use free/local schema-planning resources first via LangChain + Ollama or direct Ollama, and fall back to its built-in heuristic planner otherwise.

Run it with:

```bash
python -m data_agent.agent "classify support ticket urgency as urgent or normal" path/to/output_dir path/to/raw_data
```
