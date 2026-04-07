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

## Theme Music

🎵 **["Harder, Better, Faster, Stronger" — Daft Punk](https://www.youtube.com/watch?v=yydNF8tuVmU)**

*Training loops, hyperparameter tuning, AutoResearch — harder, better, faster, stronger, one epoch at a time.*

---

## Project Synopsis

**AutoTrain Agent** is an end-to-end autonomous ML training agent that turns a plain-English prompt and a hard budget cap into a fully trained, saved model — with no infrastructure knowledge required.

**The problem:** Training a custom ML model today requires stitching together data wrangling, architecture decisions, cloud GPU setup, hyperparameter tuning, and cost monitoring. For students and researchers, any one of those steps can derail an entire project.

**Our solution:** A multi-agent pipeline with five components:

| Feature | What it does |
|---------|--------------|
| **Manager Agent** | Orchestrates the entire pipeline end-to-end |
| **Data Generator** | Finds data on Hugging Face, scrapes the web, or synthetically generates a dataset from the prompt |
| **Decision Engine** | Decides whether to fine-tune (LoRA) or pre-train from scratch based on the task and budget |
| **AutoResearch Loop** | Iterates hypothesis → code edit → Tinker run → validation loss check; keeps wins, reverts losses (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) |
| **Cost Manager** | Polls Tinker's billing API; saves `state_dict` and kills the instance the moment the budget is hit |

**June Demo Target:** A working agent that handles 3 task types (image classification, LLM fine-tuning, custom pre-training), runs fully autonomously on Tinker GPUs, and outputs saved model weights.

**Tech Stack:** Python · Tinker Distributed GPU API · Hugging Face Hub API · PyTorch · LoRA / PEFT

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
