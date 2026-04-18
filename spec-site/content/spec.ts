export const TEAM = [
  "Sid Potti",
  "Matthew Torre",
  "Ron Polonsky",
  "Angel Raychev",
  "Hayley Antczak",
];

export const LAST_UPDATED = "April 12, 2026";
export const VERSION = "1.0";

export type Section = {
  id: string;
  title: string;
  owner?: string;
  content: string; // markdown
};

export const sections: Section[] = [
  {
    id: "overview",
    title: "Overview",
    content: `
The **Autonomous ML Training Agent** is an end-to-end system that takes a plain-English prompt (e.g., *"build a model that classifies handwritten digits"*) and a strict student budget cap (e.g., $50), then automatically handles:

- Data discovery / generation
- Model architecture selection
- Hyperparameter tuning
- Training orchestration on Tinker's distributed GPU infrastructure

The system enforces **hard budget guardrails** and terminates runs before exceeding allocated costs, eliminating the financial anxiety that paralyzes student ML projects.

> **Core Innovation:** An autonomous ML "co-pilot" that collapses weeks of data wrangling, model selection, and infrastructure setup into a single API call — making custom model training accessible to non-ML researchers, students, and bootstrapped teams.
    `,
  },
  {
    id: "problem",
    title: "The Problem & Market Opportunity",
    content: `
### Why We're Solving This

Current ML project workflows are broken for students and non-specialists:

| Pain Point | Description |
|---|---|
| **Data wrangling dominates** | Finding, cleaning, and formatting data consumes 80% of project time. Our agent automates discovery (HuggingFace, synthetic generation, intelligent scraping) and standardization. |
| **Architecture paralysis** | Fine-tune or train from scratch? LoRA or full weights? 8B or 235B? Without domain expertise, teams waste weeks on decisions. Our Decision Engine chooses the right strategy based on task and budget. |
| **Cloud costs are terrifying** | A forgotten GPU run = $400+ charges. Students have no financial buffer. We provide a hard kill-switch tied to real-time billing. |
| **Hyperparameter tuning is tedious** | Manual iteration (tweak → wait → check loss → repeat) takes days. Our AutoResearch loop runs dozens of controlled experiments in hours. |

---

### Market Size & Opportunity

**SOM (Serviceable Obtainable Market)**
Academic researchers (bio, econ, social science labs) who need custom ML models for papers but lack ML infrastructure knowledge. Stanford has 500+ faculty-led research groups; even 5% adoption = significant user base.

**SAM (Serviceable Available Market)**
As the agent's AutoResearch loop stabilizes, the market expands to bootstrapped startups, indie hackers, and small agencies. Roughly **5–10 million developers globally** want to train custom AI models but cannot afford $150k+/year MLOps engineers.

**TAM (Total Addressable Market)**
The global software engineering workforce (~30 million developers). Only 2 million are specialized in ML. Our agent bridges this gap, giving the remaining **26+ million** standard developers autonomous infrastructure to train and deploy custom models.
    `,
  },
  {
    id: "users",
    title: "User Segments & Value Proposition",
    content: `
### Primary Segments

| Segment | Description |
|---|---|
| **Student ML Builders** | CS students needing quick, budget-conscious training for class projects or portfolios. |
| **Non-ML Academic Researchers** | PhD students in bio, econ, linguistics, etc. who need custom classifiers for thesis data but lack infrastructure expertise. |
| **Lightweight Prototyping** | Teams validating ML feasibility in days, not weeks. |

### Secondary Segments

| Segment | Description |
|---|---|
| **Bootstrapped Founders / Small Startups** | Teams needing custom models without dedicated ML talent or GPU clusters. |
| **General-Purpose AI Experimenters** | Hackers, indie researchers, and hobbyists exploring new model architectures. |

---

### Core Value Proposition

> A fully autonomous AI "co-pilot" that transforms a plain-English idea and a hard budget cap into a deployed, trained model by automatically searching, curating, and/or generating data; choosing the right training strategy (LoRA, fine-tune, or pre-train); running an intelligent AutoResearch loop; and enforcing strict cost guardrails.

**Ship useful models without touching infrastructure or blowing up cloud bills.**
    `,
  },
  {
    id: "feature-0",
    title: "Feature 0: The Manager (Orchestrator)",
    owner: "Sid Potti",
    content: `
**Central orchestrator** that receives the user's training prompt and budget, then sequences the downstream features while monitoring the Cost Manager for budget violations.

### Inputs

\`\`\`
prompt: string          # plain-English task description
budget: float           # hard dollar cap
data_path?: string      # optional user-provided data
\`\`\`

### Output JSON Schema

\`\`\`json
{
  "data": bool,
  "prompt": string,
  "compute_budget": float,
  "training_procedure": {
    "task_type": "classification | fine-tune | pre-train",
    "data_format": "...",
    "training_type": "SFT | RL | ...",
    "base_model": "...",
    "hyperparameters": {
      "learning_rate": float,
      "batch_size": int,
      "epochs": int,
      "...": "..."
    },
    "notes": "..."
  }
}
\`\`\`

### Responsibilities

1. Parse and validate user prompt + budget
2. Query user if they have existing data to provide
3. Reason about task type, data requirements, and training strategy
4. Emit the orchestration JSON consumed by all downstream agents
5. Sequence: **Data Generator → Decision Engine → AutoResearch Loop**
6. Monitor Cost Manager for budget violations at each step
7. Maintain full audit trail / decision log

### Key Decisions

- Manager runs locally; training jobs are submitted to Tinker remotely
- Uses Claude API for prompt interpretation and reasoning
- Falls back to rule-based defaults if API quota exhausted

> **Blocker:** Tinker API docs must be confirmed by **Apr 18** (Sid).
    `,
  },
  {
    id: "feature-1",
    title: "Feature 1: The Data Generator",
    owner: "Ron Polonsky, Angel Raychev",
    content: `
Intelligently discovers or creates training data. Operates in three modes determined by the Manager.

---

### Mode A — User-Provided Data

**Input:** Messy user data (CSV, JSON, images, etc.)

**Pipeline:**
1. Detect format and data type
2. Normalize, clean, and filter
3. Augment with synthetic data if needed
4. Output standardized training dataset

---

### Mode B — HuggingFace Discovery

**Trigger:** No user data; suitable dataset likely exists on HuggingFace Hub

**Pipeline:**
1. Search HuggingFace database for candidates using task description
2. Rank by relevance, license, and dataset quality
3. Select best match, download, and validate
4. Return standardized dataset

---

### Mode C — Synthetic Generation

**Trigger:** No user data; HuggingFace search fails or returns no suitable results

**Pipeline:**
1. Determine data format and input/output schema from \`training_procedure\`
2. Intelligently scrape web **OR** generate synthetic data using LLM-based teacher
3. Morph collected data into standardized format

---

### Output Schema

\`\`\`json
{
  "dataset_path": "string",
  "format": "jsonl | csv | parquet | ...",
  "num_examples": int,
  "split": {
    "train": int,
    "val": int,
    "test": int
  },
  "mode_used": "A | B | C",
  "quality_notes": "string"
}
\`\`\`

---

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Web scraping is fragile | Prioritize HuggingFace + synthetic generation; scraping is fallback only |
| Data quality unclear | Validate via: accurate labels, distribution check, no missing values; monitor training curves |
| HuggingFace rate limits | Implement backoff/retry logic |

### MVP Scope

- Mode B (HuggingFace) is primary path
- Mode C (synthetic generation) is fallback
- Mode A (user-provided cleanup) is secondary
- Web scraping **deprioritized for MVP**
    `,
  },
  {
    id: "feature-2",
    title: "Feature 2: The Decision Engine",
    owner: "Ron Polonsky, Angel Raychev",
    content: `
Analyzes the user's task and budget to recommend and configure a training strategy.

---

### Case A — Fine-Tune (preferred)

**Trigger:** Suitable pre-trained base model exists on HuggingFace

**Steps:**
1. Select pre-trained base model from HuggingFace (size appropriate for budget)
2. Configure LoRA-based fine-tuning script
3. Hand off to AutoResearch loop with baseline config

---

### Case B — Pre-Train from Scratch

**Trigger:** Novel architecture or custom objective with no suitable base model

**Steps:**
1. Write custom \`model.py\` and \`train.py\` from scratch
2. Define architecture based on task requirements
3. Hand off to AutoResearch loop for continuous improvement

---

### Decision Logic

\`\`\`
if task_has_pretrained_base AND budget >= fine_tune_cost_estimate:
    → Case A (Fine-Tune with LoRA)
elif novel_architecture OR budget < fine_tune_cost_estimate:
    → Case B (Pre-Train)
\`\`\`

### Output Schema

\`\`\`json
{
  "strategy": "fine-tune | pre-train",
  "base_model": "string | null",
  "lora_config": { "rank": int, "alpha": int, "target_modules": [...] },
  "estimated_cost": float,
  "estimated_time_min": int,
  "training_script_path": "string"
}
\`\`\`
    `,
  },
  {
    id: "feature-3",
    title: "Feature 3: AutoResearch Loop",
    owner: "Matthew Torre, Hayley Antczak",
    content: `
The core innovation: **autonomous hyperparameter and architectural tuning**. After receiving a baseline model from the Decision Engine, the loop continuously proposes, tests, and adopts improvements — operating like a fast, budget-conscious ML engineer running dozens of controlled experiments.

---

### Loop Lifecycle

\`\`\`
┌─────────────┐
│   PROPOSE   │  Generate code/config diff encoding a single hypothesis
│             │  e.g. "increase context length to 1,024"
└──────┬──────┘
       ↓
┌─────────────┐
│     RUN     │  Schedule constrained experiment on Tinker
│             │  e.g. 5-minute budgeted run with standardized logging
└──────┬──────┘
       ↓
┌─────────────┐
│  EVALUATE   │  Score run using Evaluator sub-feature
│             │  Returns scalar score + natural-language critique
└──────┬──────┘
       ↓
┌─────────────┐
│   DECIDE    │  Merge winning diffs; revert losing ones
│             │  Log to research diary
└─────────────┘
\`\`\`

---

### Search Strategy & Safeguards

- **Random search** over bounded hyperparameter ranges
- **Local perturbations** around current best config
- **Heuristic playbooks** from prior successful runs
- **Early stopping:** catastrophic degradation (exploding loss, NaNs, accuracy collapse) within first few steps → terminate + reallocate budget
- All edits tracked as discrete patches to training script / config
- Human-readable **research diary** logs every iteration: what was tried, cost, metric delta

### Research Diary Schema

\`\`\`json
{
  "iteration": int,
  "hypothesis": "string",
  "diff": "string",
  "cost_usd": float,
  "metrics": {
    "train_loss": float,
    "val_loss": float,
    "test_loss": float,
    "primary_metric": float
  },
  "decision": "KEPT | REVERTED",
  "notes": "string"
}
\`\`\`

---

### Evaluator Sub-Feature

Manages the evaluation surface for every iteration.

**Responsibilities:**
1. Create or select the right eval suite (hold-out validation split, synthetic stress tests, LLM-graded benchmarks)
2. Run evals automatically after each experiment
3. Return a single scalar score + natural-language critique to the loop
4. Flag regressions (accuracy drop, calibration worsening, poor cost-per-gain) → trigger revert
5. Adapt eval set over time by adding harder examples when systematic weaknesses are detected

**MVP evals:** validation accuracy, F1, loss curves (LLM-based grading is optional and budgeted separately)

---

### Matthew Torre Action Items

- Manager Agent Prompt
- Validation Loss, Training Loss, Test Loss → Code Iterator
- Mini runs for hyperparam tuning → until longer run after tuned

---

### Success Criteria

- AutoResearch beats baseline by **≥ 3%** on primary metric
- **3–5 meaningful tests per budget hour**
- All experiments reproducible via patch log
    `,
  },
  {
    id: "feature-4",
    title: "Feature 4: Cost Manager",
    owner: "Sid Potti",
    content: `
Hard financial guardrail enforcing the user's budget cap in real time.

---

### Behavior

1. **Continuously polls** Tinker billing API every **30 seconds**
2. Compares cumulative spend against user budget
3. At **90% of budget**: emit warning, save model checkpoint
4. At **100% of budget**: immediately save \`state_dict\`, kill GPU instance, return results

### Polling Loop

\`\`\`python
while True:
    spent = tinker.get_cumulative_spend(job_id)
    if spent >= budget * 0.9:
        save_checkpoint()
    if spent >= budget:
        kill_instance()
        return best_checkpoint, cost_breakdown
    time.sleep(30)
\`\`\`

### Output Schema

\`\`\`json
{
  "final_model_path": "string",
  "best_checkpoint_path": "string",
  "cost_breakdown": {
    "data_generation_usd": float,
    "training_usd": float,
    "llm_calls_usd": float,
    "total_usd": float
  },
  "termination_reason": "budget_limit | training_complete | error"
}
\`\`\`

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Tinker billing API latency | Poll every 30s; use 90% threshold as buffer |
| Run exceeds budget before kill | Save \`state_dict\` every 5–10 minutes during training |
| Billing SLA unclear | **Confirm Tinker's billing SLA by Apr 18** |

> **Blocker:** Must confirm Tinker billing API rate limits and latency SLA with Tinker team.
    `,
  },
  {
    id: "feature-5",
    title: "Feature 5: Observability",
    owner: "Team",
    content: `
Real-time visibility into all agent decisions, training metrics, and budget usage.

---

### MVP: Structured CLI Logging

All agents emit structured JSON logs plus human-readable CLI output:

\`\`\`
[Agent] Task: Classify movie review sentiment
[Agent] Budget: $50.00 | Spent: $0.00
[Manager] → Initializing system...
[DataGen] Searching HuggingFace...
[DataGen] ✓ Found 'movie_reviews' dataset (50K examples)
[DataGen] Downloading and validating...
[DecisionEngine] Task Analysis: Sentiment classification (NLP)
[DecisionEngine] Recommendation: Fine-tune BERT (cost: $12, time: 8m)
[AutoResearch] Iteration 1/8: Testing LoRA rank=16...
  → Loss: 0.35 → Val F1: 0.89 ✓ [KEPT]
[AutoResearch] Iteration 2/8: Testing learning_rate=2e-5...
  → Loss: 0.33 → Val F1: 0.91 ✓ [KEPT]
[CostManager] Spend: $9.20 / $50.00 (18% used)
[Agent] Training complete!
[Agent] Final Model: F1 = 0.93 | Cost: $11.50
[Agent] Saved to: ./models/sentiment_classifier.pt
\`\`\`

### Log Schema (JSON)

\`\`\`json
{
  "timestamp": "ISO8601",
  "agent": "Manager | DataGen | DecisionEngine | AutoResearch | CostManager",
  "level": "INFO | WARN | ERROR",
  "message": "string",
  "metadata": {}
}
\`\`\`

---

### Stretch Goal: Web Dashboard (Week 8 if bandwidth)

- Monitor in-flight runs and AutoResearch diary
- Real-time metrics charts (loss curves, budget gauge)
- Backend: REST API exposing system state
- Frontend: Simple React dashboard
- **Not required for June demo** — CLI logs are sufficient
    `,
  },
  {
    id: "metrics",
    title: "Metrics for Success",
    content: `
### Objective Function

> Minimize **time-to-trained-model** for students and researchers building custom classifiers, while maintaining cost ≤ user-specified budget cap, with **zero human infrastructure intervention** required.

---

### Internal (Development) Metrics

| Metric | Target by June | Owner |
|---|---|---|
| % tasks completed end-to-end without human intervention | 80% for baseline classification tasks | Team |
| Time from prompt → trainable model | < 20 min (classification), < 45 min (fine-tuning) | Ron + Angel |
| Cost prediction accuracy | Within 10% of actual spend | Sid |
| AutoResearch iterations per hour | 3–5 meaningful tests per budget hour | Matthew + Hayley |
| Baseline vs. improved model lift | AutoResearch beats baseline by ≥ 3% | Matthew |

---

### Adoption Metrics (If Live User Testing)

| Metric | Target |
|---|---|
| Unique users (grad students, researchers) who test the system | ≥ 3 by demo day |
| Models trained end-to-end | ≥ 5 |
| Average cost per model | < $15 per student project |
| User "Would use again" sentiment | ≥ 70% |
| Data Generator success rate (≥ 1 mode succeeds for any prompt) | ≥ 90% |
    `,
  },
  {
    id: "timeline",
    title: "Timeline & Milestones",
    content: `
### 10-Week Development Plan

| Phase | Timeline | Deliverables | Owner(s) |
|---|---|---|---|
| **MVP Build** | Apr 15 – May 5 (Weeks 1–3) | Feature 0 (Manager), Feature 1 (basic DataGen), Feature 2 (fine-tune path), Feature 4 (cost kill-switch) | Ron / Angel / Sid |
| **AutoResearch Build** | May 6 – May 26 (Weeks 4–6) | Feature 3 (loop + search strategy), Evaluator sub-feature, early testing | Matthew (lead) |
| **Integration & Polish** | May 27 – Jun 9 (Weeks 7–8) | End-to-end system test, Feature 5 (observability MVP), bug fixes, documentation | Team |
| **Demo & Final Polish** | Jun 10 – Jun 20 (Weeks 9–10) | Demo script, case study/video walkthrough, presentation | Team |

---

### Key Milestones

| Date | Milestone |
|---|---|
| **May 5** | MVP complete. Features 0–2 and Feature 4 functional. Can train a basic classifier end-to-end on Tinker. |
| **May 26** | AutoResearch loop stable. Feature 3 fully integrated; Evaluator working. System can improve baseline by ≥ 3%. |
| **Jun 9** | System hardened. All features integrated. Pass 3 diverse test scenarios. |
| **Jun 15–20** | Demo day ready. Live demo running on Tinker. 10-minute walkthrough of entire pipeline. |

---

### Current Blockers

> ⚠️ **Apr 18 deadline:** Sid must confirm updated Tinker job submission + billing API docs. This is blocking all team work in Week 1.
    `,
  },
  {
    id: "architecture",
    title: "Technical Architecture",
    content: `
### System Architecture Overview

\`\`\`
                    ┌─────────────────────────────────────────────┐
                    │              Manager Agent                  │
                    │  • Parses prompt + budget                   │
                    │  • Emits orchestration JSON                 │
                    │  • Sequences downstream features            │
                    │  • Monitors Cost Manager                    │
                    └──────┬──────────────────────────────────────┘
                           │ orchestration JSON
          ┌────────────────┼────────────────────┐
          ↓                ↓                    ↓
  ┌──────────────┐  ┌──────────────┐   ┌──────────────────┐
  │ Data         │  │ Decision     │   │ Cost Manager     │
  │ Generator    │  │ Engine       │   │ • Polls Tinker   │
  │              │  │              │   │   billing API    │
  │ Mode A/B/C   │  │ Fine-tune or │   │ • Hard kill at   │
  │              │  │ Pre-train    │   │   budget limit   │
  └──────┬───────┘  └──────┬───────┘   └──────────────────┘
         │                 │
         └────────┬────────┘
                  ↓
        ┌──────────────────┐
        │  AutoResearch    │
        │  Loop            │
        │  Propose→Run→    │
        │  Evaluate→Decide │
        └──────────────────┘
                  │
                  ↓
        ┌──────────────────┐
        │  Tinker GPU      │
        │  Infrastructure  │
        │  (training jobs) │
        └──────────────────┘
\`\`\`

---

### Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.10+ | Modern async, type hints, ML ecosystem |
| Training Framework | PyTorch | Flexibility for custom architectures; matches CS231N tools |
| Model Sourcing | HuggingFace Transformers | Largest open-source model hub |
| AutoResearch Scaffold | Adapted from Karpathy's minimal autoresearch repo | Single mutable training script + research diary (JSON) |
| LLM-powered decisions | Claude API (Anthropic SDK) | Prompt interpretation, data generation, architectural suggestions |
| State Management | In-memory agent state + persistent JSON logs | Research diary, cost ledger, model checkpoints |
| GPU Infrastructure | Tinker | Job submission + billing API |

---

### Key Technical Decisions

**Manager runs locally; training jobs submitted to Tinker remotely.**
This avoids needing the full agent stack on Tinker's infrastructure.

**Single mutable training script + patch log (Karpathy autoresearch pattern)**
Each AutoResearch iteration edits one file; diffs are tracked. One thread writes at a time; Git used internally for versioning. Revert on any error.

**Claude API for decisions; rule-based fallback**
Per-call budget set. Falls back to deterministic rules if quota exhausted.
    `,
  },
  {
    id: "open-questions",
    title: "Open Questions & Assumptions",
    content: `
### Product & Feature Scope

| Question | Status | Owner |
|---|---|---|
| What is the minimum viable dataset size? When does DataGen switch from HF search to synthetic generation? | **Needs decision by Week 2** | Ron + Angel |
| How complex should AutoResearch be for MVP? Random search or Bayesian optimization? | Start with **random search** for MVP | Matthew |
| Should we support deployment (inference)? | **No** — MVP trains and saves weights only. Deployment is post-June. | Team |
| How do we handle data privacy for proprietary researcher uploads? | Needs policy decision by **Week 3** | Sid |

---

### Technical & Integration

| Question | Status | Owner |
|---|---|---|
| Tinker API blockers: updated job submission + billing API docs | **Blocking all Week 1 work — due Apr 18** | Sid |
| Tinker GPU cluster: dev/test account separate from prod? | Pending clarification | Sid |
| LLM call cost management for AutoResearch proposals + Evaluator grading | Use **Claude Haiku** for cheap calls; set per-request budgets | Matthew |
| Manager runs locally; training jobs submitted to Tinker? | **Yes** — confirmed architecture | Sid |

---

### Go/No-Go Decisions

| Decision | Status |
|---|---|
| Open-source after June? | **Deferred to Week 8** — evaluate after beta feedback |
| Web dashboard (Feature 5 stretch) | **Only if bandwidth in Week 8** |
| Deployment / inference support | **Post-June roadmap** |
| Multi-modal support | **Post-June** — not needed for any demo use case |
    `,
  },
  {
    id: "demo-criteria",
    title: "Demo Day Success Criteria",
    content: `
### Success Criteria for June 20 Demo

| Criteria | Target |
|---|---|
| **Live system demo** | User enters prompt + budget → system trains a model end-to-end on Tinker without human intervention |
| **Cost guardrails working** | System respects budget cap and terminates gracefully if approaching limit |
| **AutoResearch delivers** | Final model beats baseline by ≥ 3% via automated hyperparameter + architectural tuning |
| **Task diversity** | Supports ≥ 3 task types: basic classification, fine-tuning LLM, custom architecture |
| **User evidence** | Real or realistic case study showing tangible value |
| **Polished presentation** | 10-min walkthrough, clear messaging, architectural clarity |

---

### Example CLI Session (Demo Script)

\`\`\`bash
python agent.py \\
  --prompt "classify sentiment in movie reviews" \\
  --budget 50 \\
  --task classification
\`\`\`

Expected output:

\`\`\`
[Agent] Task: Classify movie review sentiment
[Agent] Budget: $50.00 | Spent: $0.00
[Manager] → Initializing system...
[DataGen] Searching HuggingFace...
[DataGen] ✓ Found 'movie_reviews' dataset (50K examples)
[DataGen] Downloading and validating...
[DecisionEngine] Recommendation: Fine-tune BERT (cost: $12, time: 8m)
[AutoResearch] Iteration 1/8: Testing LoRA rank=16...
  → Loss: 0.35 → Val F1: 0.89 ✓ [KEPT]
[AutoResearch] Iteration 2/8: Testing learning_rate=2e-5...
  → Loss: 0.33 → Val F1: 0.91 ✓ [KEPT]
[CostManager] Spend: $9.20 / $50.00 (18% used)
[Agent] Training complete!
[Agent] Final Model: F1 = 0.93 | Cost: $11.50
[Agent] Saved to: ./models/sentiment_classifier.pt
\`\`\`
    `,
  },
];
