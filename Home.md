# Team Nemoral

> *Stanford CS194 — Senior Capstone Project, Spring 2026*

---

## Team Logo

```
   _       _   _ _____ ___   _____  ____    _    ___ _   _ 
  / \  _  | | | |_   _/ _ \ |_   _|  _ \  / \  |_ _| \ | |
 / _ \| | | |_| | | || | | |  | | | |_) |/ _ \  | ||  \| |
/ ___ \ |_|  _  | | || |_| |  | | |  _ // ___ \ | || |\  |
/_/   \_(_)_| |_| |_| \___/   |_| |_| \_/_/   \_|___|_| \_|

      A G E N T      ·      T e a m  2 6      ·      S P 2 6
```

*Replace with a real logo if you have one — drop a PNG in and swap the block above.*

---

## Project Synopsis

**Nemoral** is an end-to-end autonomous ML training agent that turns a plain-English prompt, optional dataset source, and hard budget cap into a trained model artifact.

**The problem:** Training a custom language model today requires stitching together data wrangling, dataset validation, model choice, LoRA configuration, GPU execution, hyperparameter search, evaluation, and budget controls. For students and researchers, any one of those steps can derail an entire project.

**Current V1 scope:** chat-style supervised fine-tuning on Tinker. The supported execution path is JSONL chat/SFT data -> DataGen curation -> DecisionEngine Tinker plan -> AutoResearch baseline/candidate loop -> SDK-native Tinker LoRA runner -> metrics, checkpoint, sample, and manifest artifacts.

**Out of scope for V1:** image classification, RL/GRPO, and full pre-training. The repository may still contain compatibility stubs or older notes for those paths, but the active implementation target is Tinker chat/SFT.

**Pipeline components:**

| Feature | What it does |
|---------|--------------|
| **Manager Agent** | Orchestrates the entire pipeline end-to-end |
| **Data Generator** | Finds Hugging Face data, structures web sources, or creates synthetic chat/SFT rows from the prompt |
| **Decision Engine** | Chooses the Tinker SFT plan, base model, LoRA config, run estimate, and compatibility training artifact |
| **AutoResearch Loop** | Iterates hypothesis -> config patch -> bounded Tinker run -> heldout score; keeps material wins and reverts losses |
| **Cost Manager** | Guards launches with budget preflight, records in-process spend, and wraps SDK-native Tinker runs with monitoring hooks |

**June Demo Target:** A working chat/SFT agent that can run from prompt plus optional dataset source, stay inside a budget, execute real Tinker LoRA experiments on `Qwen/Qwen3.5-9B`, and return inspectable artifacts.

**Tech Stack:** Python · FastAPI · LangGraph · Tinker SDK · Tinker Cookbook · Hugging Face Hub API · Tavily · React/Vite

---

## Team Member Matrix

| Name | Photo | Role | Strengths | Gaps / Learning Areas | GitHub |
|------|-------|------|-----------|-----------------------|--------|
| [Member 1] | <!-- photo --> | Manager Agent / Orchestration | [e.g. LLM tooling, agent frameworks] | [e.g. distributed systems] | [@handle](https://github.com/handle) |
| [Member 2] | <!-- photo --> | Data Generator | [e.g. data pipelines, HF ecosystem] | [e.g. frontend] | [@handle](https://github.com/handle) |
| [Member 3] | <!-- photo --> | Decision Engine + AutoResearch | [e.g. ML training, PyTorch] | [e.g. cloud billing APIs] | [@handle](https://github.com/handle) |
| [Member 4] | <!-- photo --> | Cost Manager + Tinker Infra | [e.g. cloud APIs, DevOps] | [e.g. ML architecture] | [@handle](https://github.com/handle) |
| [Member 5] | <!-- photo --> | Observability + UI | [e.g. dashboards, full stack] | [e.g. ML training internals] | [@handle](https://github.com/handle) |

### Expertise Coverage

| Area | Status | Owner |
|------|--------|-------|
| LLM / Agent frameworks | ✅ Strong | TBD |
| PyTorch / model training | ✅ Strong | TBD |
| Hugging Face ecosystem | ✅ Strong | TBD |
| Tinker GPU API / cloud infra | ⚠️ Learning | TBD |
| Data scraping & synthetic gen | ⚠️ Learning | TBD |
| Observability / dashboards | ⚠️ Gap | TBD |
| Cost optimization at scale | ⚠️ Gap | TBD |

---

## Communication

| Channel | Link / Info |
|---------|-------------|
| Slack | [#team-26](https://stanford-cs194.slack.com) *(update with your channel link)* |
| Team Email | [team26@stanford.edu](mailto:team26@stanford.edu) *(update if different)* |
| Weekly Standup | *(add day/time and Zoom link)* |
| GitHub Issues | [Issues](https://github.com/StanfordCS194/spr26-Team-26/issues) |
| Project Board | [GitHub Projects](https://github.com/StanfordCS194/spr26-Team-26/projects) |

---

## Useful Links

- [Repository](https://github.com/StanfordCS194/spr26-Team-26)
- [Product Requirements Doc (PRD)](https://github.com/StanfordCS194/spr26-Team-26/blob/main/prd.md)
- [Hugging Face Hub API Docs](https://huggingface.co/docs/hub/api)
- [Karpathy's autoresearch (inspiration)](https://github.com/karpathy/autoresearch)
- [CS194 Course Page](https://web.stanford.edu/class/cs194/)

---

*Last updated: April 2026*
