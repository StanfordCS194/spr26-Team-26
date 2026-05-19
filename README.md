# Nemoral: An Autonomous Self-Improving ML Training Agent

Welcome to the repository for **Team Nemoral** — Stanford CS194, Spring 2026.

> Give it a plain-English prompt and a budget. Get back a trained model. No infra knowledge required.

Nemoral is a multi-agent training pipeline for chat-style supervised fine-tuning. The current V1 path is:

1. Manager interprets the prompt, budget, and optional dataset source.
2. DataGen retrieves or creates trainable chat/SFT JSONL records.
3. DecisionEngine emits a Tinker SFT plan.
4. AutoResearch runs bounded baseline and candidate LoRA experiments.
5. The SDK-native Tinker runner writes metrics, manifest, sample, and checkpoint artifacts.

The main implementation target is local JSONL chat/SFT data with records in either `messages` form or `input`/`output`-style pairs. Image classification, RL/GRPO, and full pre-training are outside the current V1 execution path.

---

## Current Entry Points

- Manager API: `src/server/app.py`
- Manager graph: `src/manager/manager.py`
- Data generation and curation: `src/data_generator/`
- AutoResearch graph: `src/autoresearch/autoresearch.py`
- Tinker SFT runner: `src/tinker_api/sft_runner.py`
- Frontend dashboard: `ml-agent-frontend/`
- Spec site: `spec-site/`

## Local Validation

```bash
python -m compileall src
python -m pytest tests -q --ignore=tests/integration/test_tinker_live_smoke.py
```

Live Tinker validation is opt-in and requires credentials:

```bash
RUN_LIVE_TINKER=1 TINKER_API_KEY=... TINKER_SMOKE_MODEL="Qwen/Qwen3.5-9B" TINKER_SMOKE_STEPS=5 \
  python -m pytest tests/integration/test_tinker_live_smoke.py -q -s
```

## Project Docs

- [Wiki Home](https://github.com/StanfordCS194/spr26-Team-26/wiki)
- [Spec Site](./spec-site)
- [Frontend README](./ml-agent-frontend/README.md)
