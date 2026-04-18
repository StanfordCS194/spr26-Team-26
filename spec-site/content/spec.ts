export const TEAM = [
  "Sid Potti",
  "Matthew Torre",
  "Ron Polonsky",
  "Angel Raychev",
  "Hayley Antczak",
];

export type Param = {
  name: string;
  type: string;
  description: string;
  optional?: boolean;
};

export type FunctionSpec = {
  name: string;
  signature: string;
  description: string;
  params: Param[];
  returns: { type: string; description: string };
  notes?: string;
};

export type FeatureSpec = {
  id: string;
  title: string;
  owner: string;
  description: string;
  /** Plain-text or pseudo-code control flow diagram shown before functions */
  architecture: string;
  /** ASCII/text flow diagram (optional) */
  flowDiagram?: string;
  functions: FunctionSpec[];
};

// ─── OVERALL SYSTEM ARCHITECTURE ─────────────────────────────────────────────
export const systemArchitecture = {
  overview: `The system is a linear pipeline with one autonomous feedback loop. The user provides a single prompt and a budget. The Manager reasons about the task and emits a config object that every other agent reads from. Control then flows through three sequential stages — data, decisions, training — with the Cost Manager running as a background watchdog throughout.`,
  flowDiagram: `
User
  │  prompt: str
  │  budget: float
  │  data_path?: str
  ▼
┌─────────────────────────────────────┐
│          Manager Agent              │
│  1. query_user_for_data()           │
│  2. reason_about_task()  [Claude]   │
│  3. build_orchestration_config()    │
│  4. log_decision()                  │
└───────────────┬─────────────────────┘
                │  OrchestrationConfig
                │  (passed to ALL agents)
                ▼
┌─────────────────────────────────────┐
│         Data Generator              │
│  Mode A: clean user data            │
│  Mode B: search HuggingFace         │◄── HuggingFace Hub API
│  Mode C: synthesize / scrape        │◄── Claude API (teacher)
└───────────────┬─────────────────────┘
                │  DatasetResult
                ▼
┌─────────────────────────────────────┐
│         Decision Engine             │
│  analyze_task()                     │
│  find_base_model()    ◄─── HF Hub   │
│  estimate_training_cost()           │
│  write_finetune_script() OR         │
│  write_pretrain_script()            │
└───────────────┬─────────────────────┘
                │  TrainingPlan
                │  (script path + config)
                ▼
┌─────────────────────────────────────┐   ┌───────────────────────┐
│       AutoResearch Loop             │   │    Cost Manager        │
│                                     │   │  (background thread)   │
│  baseline run → EvalScore           │   │                        │
│  ┌──────────────────────────────┐   │   │  poll_spend() / 30s    │
│  │  propose_hypothesis()        │   │──►│  save_checkpoint()@90% │
│  │  apply_patch()               │   │   │  kill_job() @ 100%     │
│  │  submit_experiment() ──────────────► Tinker GPU              │
│  │  wait_for_experiment()       │   │   └───────────────────────┘
│  │  check_early_stop()          │   │
│  │  run_evals()                 │   │
│  │  compare_scores()            │   │
│  │  decide_keep_or_revert()     │   │
│  │  log_iteration()             │   │
│  └──────────────────────────────┘   │
│  (repeat until budget exhausted     │
│   or no improvement for N iters)    │
└───────────────┬─────────────────────┘
                │  TrainedModel
                ▼
           User receives:
           • weights_path
           • final EvalScore
           • CostBreakdown
           • research_diary.jsonl
  `,
  keyContracts: [
    {
      from: "User → Manager",
      data: "prompt: str, budget: float, data_path?: str",
    },
    {
      from: "Manager → All agents",
      data: "OrchestrationConfig — single source of truth for task, budget, and training procedure",
    },
    {
      from: "Data Generator → Decision Engine",
      data: "DatasetResult — standardized dataset with split sizes and quality report",
    },
    {
      from: "Decision Engine → AutoResearch",
      data: "TrainingPlan — training strategy, base model, LoRA config, script path, eval metric",
    },
    {
      from: "AutoResearch → Cost Manager",
      data: "job_id — each Tinker job ID is registered with Cost Manager before submission",
    },
    {
      from: "AutoResearch → User",
      data: "TrainedModel — best checkpoint path, eval score, cost breakdown, research diary",
    },
  ],
  observabilityNote:
    "Every agent calls log_event() from the Observability module on every significant action. This writes a JSON line to disk and prints a color-coded CLI line. No agent writes to stdout directly.",
};

export const features: FeatureSpec[] = [
  // ─── SHARED TYPES ────────────────────────────────────────────────────────────
  // (shown as a reference panel, not a feature)

  // ─── FEATURE 0: MANAGER ──────────────────────────────────────────────────────
  {
    id: "manager",
    title: "Feature 0 — Manager Agent",
    owner: "Sid Potti",
    description:
      "Central orchestrator. Takes the user's raw prompt and budget, reasons about the task, queries for optional data, and emits the OrchestrationConfig JSON consumed by every downstream agent.",
    architecture: `The Manager is the only agent the user interacts with directly. It runs entirely locally — no GPU required. Its job is to turn an ambiguous human prompt into a precise, structured config that every downstream agent can execute independently.

Control flow:
1. run_manager() is called with the user's prompt and budget.
2. It calls query_user_for_data() to ask if the user has existing data. This sets the "data" bool in the config.
3. It calls reason_about_task() which sends the prompt + context to the Claude API. The LLM infers task type, data format, training type (SFT/RL/pre-train), a suggested base model, and starting hyperparameters.
4. build_orchestration_config() assembles all of this into the OrchestrationConfig dict.
5. log_decision() records the reasoning to decisions.jsonl.
6. orchestrate() is called with the config. This is the only function that calls into the other features — it sequences Data Generator → Decision Engine → AutoResearch, passing results between stages and registering each Tinker job with the Cost Manager.

The Manager never writes to stdout directly. All output goes through log_event() from the Observability module.`,
    flowDiagram: `
run_manager(prompt, budget, data_path?)
  │
  ├─► query_user_for_data()
  │     └─► has_data: bool
  │
  ├─► reason_about_task(prompt, budget, has_data)  [Claude API]
  │     └─► TaskReasoning
  │
  ├─► build_orchestration_config(reasoning, ...)
  │     └─► OrchestrationConfig
  │
  ├─► log_decision("task_reasoning", rationale, config)
  │
  └─► orchestrate(config)
        │
        ├─► run_data_generator(config, data_path?)  → DatasetResult
        ├─► run_decision_engine(config, dataset)    → TrainingPlan
        ├─► start_cost_monitor(job_id, budget)      → background thread
        └─► run_autoresearch(plan, config, cost_mgr)→ TrainedModel
    `,
    functions: [
      {
        name: "run_manager",
        signature: "run_manager(prompt: str, budget: float, data_path: str | None = None) -> OrchestrationConfig",
        description:
          "Main entry point. Parses the user prompt, optionally queries for existing data, calls the Claude API to reason about task type and training strategy, then returns the OrchestrationConfig used by all downstream features.",
        params: [
          { name: "prompt", type: "str", description: "Plain-English task description from the user (e.g. 'classify handwritten digits')." },
          { name: "budget", type: "float", description: "Hard dollar cap for the entire run (e.g. 50.0)." },
          { name: "data_path", type: "str | None", description: "Path to user-provided data, if any. None triggers data discovery.", optional: true },
        ],
        returns: { type: "OrchestrationConfig", description: "Structured config JSON consumed by DataGenerator, DecisionEngine, AutoResearch, and CostManager." },
      },
      {
        name: "query_user_for_data",
        signature: "query_user_for_data() -> str | None",
        description:
          "Interactively asks the user whether they have existing training data. Returns a file path if provided, otherwise None.",
        params: [],
        returns: { type: "str | None", description: "Absolute path to the user's data directory/file, or None if they have none." },
      },
      {
        name: "reason_about_task",
        signature: "reason_about_task(prompt: str, budget: float, has_data: bool) -> TaskReasoning",
        description:
          "Calls the Claude API with a structured system prompt to infer task type, data format, training procedure, starting hyperparameters, and estimated cost from the user's prompt.",
        params: [
          { name: "prompt", type: "str", description: "Raw user task description." },
          { name: "budget", type: "float", description: "Budget cap in USD." },
          { name: "has_data", type: "bool", description: "Whether the user is supplying their own data." },
        ],
        returns: { type: "TaskReasoning", description: "Structured reasoning output: task_type, data_format, training_type (SFT/RL), suggested base model, starting hyperparameters, notes." },
      },
      {
        name: "build_orchestration_config",
        signature: "build_orchestration_config(reasoning: TaskReasoning, prompt: str, budget: float, has_data: bool) -> OrchestrationConfig",
        description: "Assembles the final OrchestrationConfig dict from manager reasoning output. This JSON is passed to every downstream agent.",
        params: [
          { name: "reasoning", type: "TaskReasoning", description: "Output from reason_about_task." },
          { name: "prompt", type: "str", description: "Original user prompt." },
          { name: "budget", type: "float", description: "Budget cap in USD." },
          { name: "has_data", type: "bool", description: "Whether user supplied data." },
        ],
        returns: {
          type: "OrchestrationConfig",
          description: `Dict with shape:\n{ data: bool, prompt: str, compute_budget: float, training_procedure: { task_type, data_format, training_type, base_model, hyperparameters, notes } }`,
        },
      },
      {
        name: "log_decision",
        signature: "log_decision(step: str, rationale: str, config: OrchestrationConfig) -> None",
        description: "Appends a timestamped entry to the audit trail log file (decisions.jsonl).",
        params: [
          { name: "step", type: "str", description: "Name of the pipeline step (e.g. 'task_reasoning')." },
          { name: "rationale", type: "str", description: "Human-readable explanation of the decision." },
          { name: "config", type: "OrchestrationConfig", description: "Current config snapshot at time of decision." },
        ],
        returns: { type: "None", description: "Writes to disk only." },
      },
      {
        name: "orchestrate",
        signature: "orchestrate(config: OrchestrationConfig) -> TrainedModel",
        description:
          "Sequences all downstream features: DataGenerator → DecisionEngine → AutoResearch loop, passing results between stages and monitoring CostManager for budget violations at each handoff.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "The orchestration config emitted by build_orchestration_config." },
        ],
        returns: { type: "TrainedModel", description: "Final trained model artifact including weights path, metrics, and cost breakdown." },
      },
    ],
  },

  // ─── FEATURE 1: DATA GENERATOR ───────────────────────────────────────────────
  {
    id: "data-generator",
    title: "Feature 1 — Data Generator",
    owner: "Ron Polonsky, Angel Raychev",
    description:
      "Discovers or creates training data in three modes: (A) clean user-provided data, (B) search HuggingFace Hub, (C) synthesize with an LLM teacher or scrape the web. Outputs a standardized dataset regardless of mode.",
    architecture: `The Data Generator is stateless and purely functional — given an OrchestrationConfig and an optional data path, it always returns a DatasetResult. The caller (Manager's orchestrate()) doesn't need to know which mode ran.

The top-level function run_data_generator() is the only entry point. It decides the mode and routes accordingly:

Mode A (user provided data): The user gave us a file/directory. We load it, detect its format, normalize and clean it into the standard schema, and optionally augment it with synthetic examples if it's too small.

Mode B (HuggingFace): No user data. We search HuggingFace Hub for a dataset matching the task description. If we find a good candidate we download it, normalize it, and validate it. This is the preferred path — fast, free, and high quality.

Mode C (synthesize): HuggingFace search came up empty. We use the Claude API as an LLM teacher to generate synthetic (input, output) pairs from scratch, or fall back to web scraping if synthesis isn't feasible. All paths funnel into morph_to_standard().

After all three modes, validate_dataset() is called to check label quality and distribution. The DatasetResult includes a ValidationReport that the Decision Engine uses to size the model appropriately.`,
    flowDiagram: `
run_data_generator(config, data_path?)
  │
  ├─[data_path provided]──────────────────── MODE A
  │   ├─► load_raw_data(data_path)
  │   ├─► detect_format(data_path)
  │   ├─► normalize_and_clean(raw, schema)
  │   └─► augment_with_synthetic()?  (if n < 500)
  │
  ├─[no data_path]──────────────────────────  try MODE B
  │   ├─► search_huggingface(task, task_type)
  │   ├─► rank_hf_candidates(candidates, config)
  │   │
  │   ├─[candidate found]──────────────────── MODE B ✓
  │   │   ├─► download_hf_dataset(candidate)
  │   │   └─► normalize_and_clean(raw, schema)
  │   │
  │   └─[no candidate]─────────────────────── MODE C
  │       ├─► determine_data_schema(config)  [Claude API]
  │       ├─► generate_synthetic_data(schema, n)  [Claude API]
  │       │         OR
  │       ├─► scrape_web(query, schema)       (fallback)
  │       └─► morph_to_standard(raw, schema)
  │
  └─► validate_dataset(dataset, schema)
        └─► DatasetResult  (returned to Manager)
    `,
    functions: [
      {
        name: "run_data_generator",
        signature: "run_data_generator(config: OrchestrationConfig, data_path: str | None) -> DatasetResult",
        description:
          "Top-level dispatcher. Routes to Mode A if data_path is provided, Mode B if a suitable HuggingFace dataset is found, Mode C otherwise.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config from the Manager (used for schema, task type, etc.)." },
          { name: "data_path", type: "str | None", description: "Path to user-provided raw data, or None." },
        ],
        returns: { type: "DatasetResult", description: "Standardized dataset with path, format, split sizes, mode used, and quality notes." },
      },
      // ── Mode A ──
      {
        name: "load_raw_data",
        signature: "load_raw_data(data_path: str) -> RawData",
        description: "Loads data from a file path, auto-detecting format (CSV, JSON, JSONL, Parquet, image dir, etc.).",
        params: [
          { name: "data_path", type: "str", description: "Absolute path to the user's data file or directory." },
        ],
        returns: { type: "RawData", description: "In-memory data object with raw records and detected format metadata." },
      },
      {
        name: "detect_format",
        signature: "detect_format(data_path: str) -> DataFormat",
        description: "Inspects file extension, magic bytes, and sample rows to determine format (csv, json, jsonl, parquet, image_dir, etc.) and data modality (text, image, tabular).",
        params: [
          { name: "data_path", type: "str", description: "Path to the data file or directory." },
        ],
        returns: { type: "DataFormat", description: "Enum + metadata: { modality: 'text'|'image'|'tabular', file_type: str, encoding: str }." },
      },
      {
        name: "normalize_and_clean",
        signature: "normalize_and_clean(raw: RawData, schema: DataSchema) -> StandardDataset",
        description: "Normalizes field names, removes null/malformed rows, deduplicates, and reindexes data into the standard {input, output, split} format.",
        params: [
          { name: "raw", type: "RawData", description: "Output of load_raw_data." },
          { name: "schema", type: "DataSchema", description: "Expected input/output schema from OrchestrationConfig." },
        ],
        returns: { type: "StandardDataset", description: "Clean dataset in standard format, split into train/val/test (80/10/10 by default)." },
      },
      {
        name: "augment_with_synthetic",
        signature: "augment_with_synthetic(dataset: StandardDataset, n_extra: int, schema: DataSchema) -> StandardDataset",
        description: "Optionally augments an existing dataset with LLM-generated synthetic examples when the dataset is too small (< 500 training examples by default).",
        params: [
          { name: "dataset", type: "StandardDataset", description: "Cleaned dataset from normalize_and_clean." },
          { name: "n_extra", type: "int", description: "Number of synthetic examples to generate and append." },
          { name: "schema", type: "DataSchema", description: "Input/output schema used to prompt the LLM teacher." },
        ],
        returns: { type: "StandardDataset", description: "Augmented dataset with original + synthetic examples merged." },
      },
      // ── Mode B ──
      {
        name: "search_huggingface",
        signature: "search_huggingface(task_description: str, task_type: str) -> list[HFCandidate]",
        description: "Queries the HuggingFace Hub datasets API for candidates matching the task description and type. Returns up to 10 ranked candidates.",
        params: [
          { name: "task_description", type: "str", description: "Natural-language task description from OrchestrationConfig." },
          { name: "task_type", type: "str", description: "Structured task type (e.g. 'text-classification', 'token-classification', 'seq2seq')." },
        ],
        returns: { type: "list[HFCandidate]", description: "Up to 10 candidates with id, name, num_examples, license, task_categories, download_size." },
      },
      {
        name: "rank_hf_candidates",
        signature: "rank_hf_candidates(candidates: list[HFCandidate], config: OrchestrationConfig) -> HFCandidate | None",
        description: "Scores candidates by relevance to task, dataset size, license compatibility, and download size. Returns the best candidate or None if all fail the minimum quality bar.",
        params: [
          { name: "candidates", type: "list[HFCandidate]", description: "Output of search_huggingface." },
          { name: "config", type: "OrchestrationConfig", description: "Used to weight task relevance and size requirements." },
        ],
        returns: { type: "HFCandidate | None", description: "Best-ranked candidate, or None if no candidate meets quality threshold." },
      },
      {
        name: "download_hf_dataset",
        signature: "download_hf_dataset(candidate: HFCandidate) -> RawData",
        description: "Downloads the selected HuggingFace dataset to disk and loads it into a RawData object.",
        params: [
          { name: "candidate", type: "HFCandidate", description: "Selected candidate from rank_hf_candidates." },
        ],
        returns: { type: "RawData", description: "Downloaded dataset as RawData, ready for normalize_and_clean." },
      },
      {
        name: "validate_dataset",
        signature: "validate_dataset(dataset: StandardDataset, schema: DataSchema) -> ValidationReport",
        description: "Checks label accuracy (spot-checks via LLM), distribution coverage, and completeness (no missing input/output fields). Returns a pass/fail report.",
        params: [
          { name: "dataset", type: "StandardDataset", description: "Normalized dataset." },
          { name: "schema", type: "DataSchema", description: "Expected schema for validation checks." },
        ],
        returns: { type: "ValidationReport", description: "{ passed: bool, issues: list[str], sample_accuracy_estimate: float }." },
      },
      // ── Mode C ──
      {
        name: "determine_data_schema",
        signature: "determine_data_schema(config: OrchestrationConfig) -> DataSchema",
        description: "Uses the Claude API to infer the input/output schema for synthetic data generation from the OrchestrationConfig training_procedure.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config with training_procedure details." },
        ],
        returns: { type: "DataSchema", description: "{ input_format: str, output_format: str, input_description: str, output_description: str, example_pair: dict }." },
      },
      {
        name: "generate_synthetic_data",
        signature: "generate_synthetic_data(schema: DataSchema, n_examples: int, teacher_model: str = 'claude-haiku-4-5-20251001') -> RawData",
        description: "Generates n_examples synthetic (input, output) pairs using an LLM teacher prompted with the DataSchema. Batches requests to stay within API rate limits.",
        params: [
          { name: "schema", type: "DataSchema", description: "Input/output schema from determine_data_schema." },
          { name: "n_examples", type: "int", description: "Number of examples to generate." },
          { name: "teacher_model", type: "str", description: "Claude model ID for the LLM teacher. Defaults to Haiku for cost efficiency.", optional: true },
        ],
        returns: { type: "RawData", description: "Generated examples as RawData, ready for morph_to_standard." },
      },
      {
        name: "scrape_web",
        signature: "scrape_web(query: str, schema: DataSchema, max_examples: int = 500) -> RawData",
        description: "Fallback web scraping path. Searches for pages matching the query, extracts candidate (input, output) pairs, and returns raw scraped data. Includes retry/backoff logic.",
        params: [
          { name: "query", type: "str", description: "Search query derived from task description." },
          { name: "schema", type: "DataSchema", description: "Used to filter and extract relevant content." },
          { name: "max_examples", type: "int", description: "Maximum number of examples to collect before stopping.", optional: true },
        ],
        returns: { type: "RawData", description: "Scraped examples as RawData. May be noisy; passed to morph_to_standard for cleanup." },
      },
      {
        name: "morph_to_standard",
        signature: "morph_to_standard(raw: RawData, schema: DataSchema) -> StandardDataset",
        description: "Transforms scraped or generated raw data into the standard {input, output, split} format, deduplicates, and removes low-quality rows.",
        params: [
          { name: "raw", type: "RawData", description: "Output of generate_synthetic_data or scrape_web." },
          { name: "schema", type: "DataSchema", description: "Target schema for the output dataset." },
        ],
        returns: { type: "StandardDataset", description: "Clean, split dataset ready for training." },
      },
    ],
  },

  // ─── FEATURE 2: DECISION ENGINE ──────────────────────────────────────────────
  {
    id: "decision-engine",
    title: "Feature 2 — Decision Engine",
    owner: "Ron Polonsky, Angel Raychev",
    description:
      "Analyzes the task and budget to choose a training strategy (fine-tune vs. pre-train), select a base model, configure LoRA if applicable, and write the training script handed to AutoResearch.",
    architecture: `The Decision Engine is a pure decision function — no LLM calls, no side effects beyond writing train.py to disk. It takes the OrchestrationConfig + DatasetResult and produces a TrainingPlan.

The key decision is fine-tune vs. pre-train. The engine checks whether a suitable pretrained model exists on HuggingFace and whether fine-tuning it would fit within the budget. If both are true, it takes the fine-tune path (Case A). Otherwise it writes a model from scratch (Case B).

Case A — Fine-tune with LoRA:
Find the best pretrained base model → estimate the cost of fine-tuning it → configure LoRA parameters appropriate for the model architecture → generate train.py that wraps the base model with LoRA adapters, sets up the optimizer and data loaders from the DatasetResult, and includes standardized checkpoint saves and metric logging hooks.

Case B — Pre-train from scratch:
Generate both model.py (architecture definition) and train.py (training loop) targeting the task type. The architecture is sized to fit within the remaining budget.

In both cases the output is a TrainingPlan with the path to train.py. AutoResearch will treat this script as mutable and apply patches to it during the search loop — so the script must follow a consistent, patchable structure with clearly separated config, model, and training loop sections.`,
    flowDiagram: `
run_decision_engine(config, dataset)
  │
  ├─► analyze_task(config)
  │     └─► TaskAnalysis  { task_type, modality, has_pretrained_base, eval_metric }
  │
  ├─► find_base_model(task, budget)  ◄── HuggingFace Hub API
  │     └─► model_id: str | None
  │
  ├─► estimate_training_cost(model_id, dataset, strategy)
  │     └─► CostEstimate  { estimated_usd, estimated_time_min }
  │
  ├─[model found AND cost fits budget]──────── CASE A: Fine-tune
  │   ├─► configure_lora(base_model, task)
  │   │     └─► LoRAConfig  { rank, alpha, dropout, target_modules }
  │   └─► write_finetune_script(base_model, dataset, lora, config)
  │         └─► train.py path
  │
  └─[no model OR cost too high]────────────── CASE B: Pre-train
      └─► write_pretrain_script(task, dataset, config)
            └─► train.py path

  └─► TrainingPlan  (returned to Manager → AutoResearch)
    `,
    functions: [
      {
        name: "run_decision_engine",
        signature: "run_decision_engine(config: OrchestrationConfig, dataset: DatasetResult) -> TrainingPlan",
        description: "Top-level dispatcher. Runs task analysis, model selection, cost estimation, and script generation. Returns a complete TrainingPlan.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config from Manager." },
          { name: "dataset", type: "DatasetResult", description: "Output of run_data_generator." },
        ],
        returns: { type: "TrainingPlan", description: "{ strategy, base_model, lora_config, estimated_cost, estimated_time_min, training_script_path, eval_metric }." },
      },
      {
        name: "analyze_task",
        signature: "analyze_task(config: OrchestrationConfig) -> TaskAnalysis",
        description: "Classifies the task into a canonical type and determines whether a suitable pretrained base model likely exists on HuggingFace.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config." },
        ],
        returns: { type: "TaskAnalysis", description: "{ task_type: str, modality: str, has_pretrained_base: bool, eval_metric: str, complexity: 'low'|'medium'|'high' }." },
      },
      {
        name: "find_base_model",
        signature: "find_base_model(task: TaskAnalysis, budget: float) -> str | None",
        description: "Searches HuggingFace Hub for the best pretrained base model matching the task type and budget constraints. Returns the model ID or None if pre-training is required.",
        params: [
          { name: "task", type: "TaskAnalysis", description: "Output of analyze_task." },
          { name: "budget", type: "float", description: "Remaining budget in USD." },
        ],
        returns: { type: "str | None", description: "HuggingFace model ID (e.g. 'bert-base-uncased') or None." },
      },
      {
        name: "estimate_training_cost",
        signature: "estimate_training_cost(model_id: str | None, dataset: DatasetResult, strategy: str) -> CostEstimate",
        description: "Estimates GPU-hours and USD cost for a training run based on model size, dataset size, and strategy (fine-tune or pre-train).",
        params: [
          { name: "model_id", type: "str | None", description: "HuggingFace model ID, or None for pre-train." },
          { name: "dataset", type: "DatasetResult", description: "Dataset result (used for num_examples and input length)." },
          { name: "strategy", type: "str", description: "'fine-tune' or 'pre-train'." },
        ],
        returns: { type: "CostEstimate", description: "{ estimated_usd: float, estimated_gpu_hours: float, estimated_time_min: int, confidence: 'low'|'medium'|'high' }." },
      },
      {
        name: "configure_lora",
        signature: "configure_lora(base_model: str, task: TaskAnalysis) -> LoRAConfig",
        description: "Determines LoRA hyperparameters (rank, alpha, dropout, target_modules) appropriate for the base model architecture and task type.",
        params: [
          { name: "base_model", type: "str", description: "HuggingFace model ID." },
          { name: "task", type: "TaskAnalysis", description: "Task analysis output." },
        ],
        returns: { type: "LoRAConfig", description: "{ rank: int, alpha: int, dropout: float, target_modules: list[str] }." },
      },
      {
        name: "write_finetune_script",
        signature: "write_finetune_script(base_model: str, dataset: DatasetResult, lora: LoRAConfig, config: OrchestrationConfig) -> str",
        description: "Generates a complete train.py fine-tuning script with LoRA, standard logging hooks, and checkpoint saves. Returns the path to the written script.",
        params: [
          { name: "base_model", type: "str", description: "HuggingFace model ID." },
          { name: "dataset", type: "DatasetResult", description: "Dataset result used to set data paths in the script." },
          { name: "lora", type: "LoRAConfig", description: "LoRA configuration." },
          { name: "config", type: "OrchestrationConfig", description: "Used for hyperparameters from training_procedure." },
        ],
        returns: { type: "str", description: "Absolute path to the generated train.py script." },
      },
      {
        name: "write_pretrain_script",
        signature: "write_pretrain_script(task: TaskAnalysis, dataset: DatasetResult, config: OrchestrationConfig) -> str",
        description: "Generates a from-scratch model.py + train.py for novel architecture pre-training. Returns the path to the written script.",
        params: [
          { name: "task", type: "TaskAnalysis", description: "Task analysis output." },
          { name: "dataset", type: "DatasetResult", description: "Dataset result." },
          { name: "config", type: "OrchestrationConfig", description: "Used for architecture details and hyperparameters." },
        ],
        returns: { type: "str", description: "Absolute path to the generated train.py script." },
      },
    ],
  },

  // ─── FEATURE 3: AUTORESEARCH LOOP ────────────────────────────────────────────
  {
    id: "autoresearch",
    title: "Feature 3 — AutoResearch Loop",
    owner: "Matthew Torre, Hayley Antczak",
    description:
      "Autonomous hyperparameter and architecture search. Continuously proposes, runs, evaluates, and merges/reverts experiments until budget is exhausted or convergence is reached.",
    architecture: `The AutoResearch loop is the most complex part of the system. It implements a research-diary-driven search over the space of training configurations and architectures.

Before the loop starts, create_eval_suite() is called once to build the fixed evaluation harness. A baseline experiment is submitted and scored — this gives us the starting EvalScore that all future iterations are compared against.

Each iteration of the loop does exactly four things:
  1. PROPOSE: Call the Claude API with the full research diary and ask for one hypothesis as a unified diff. The LLM has context on what's been tried and what worked.
  2. RUN: Apply the patch to train.py, submit a short (default 5-minute) experiment to Tinker, wait for results. If early_stop_check() detects catastrophic failure, revert immediately.
  3. EVALUATE: Run the fixed eval suite against the new checkpoint. Get a scalar score and a natural-language critique.
  4. DECIDE: Compare to the current best. If improved → keep the patch and update the best score. If not → revert the patch. Either way, log the iteration to the research diary.

The loop exits when: budget is exhausted (CostManager kills the job), no improvement for N consecutive iterations, or the target metric is reached.

A key invariant: only one thread ever modifies train.py at a time. apply_patch() saves the original content, revert_patch() restores it. There is no concurrent patching.

The Evaluator sub-feature (create_eval_suite, run_evals, adapt_eval_suite) is called by the loop but can be developed independently — it has a clean interface: given a model path and an EvalSuite, return an EvalScore.`,
    flowDiagram: `
run_autoresearch(plan, config, cost_manager)
  │
  ├─► create_eval_suite(task, dataset)   ← built once, reused every iter
  │     └─► EvalSuite
  │
  ├─► submit_experiment(script, plan)    ← baseline run
  ├─► wait_for_experiment(job_id)
  ├─► run_evals(model_path, eval_suite)
  │     └─► baseline_score: EvalScore
  │
  └─► LOOP  (until budget exhausted or N iters no improvement)
        │
        ├─► propose_hypothesis(config, diary, task)  [Claude API]
        │     └─► Hypothesis  { patch, description, expected_effect }
        │
        ├─► apply_patch(script_path, patch)
        │     └─► original_content  (saved for revert)
        │
        ├─► submit_experiment(script_path, plan, timeout_min=5)
        │     └─► job_id
        │
        ├─► wait_for_experiment(job_id, timeout)
        │     └─► ExperimentResult  { metrics, model_path, cost_usd }
        │
        ├─► check_early_stop(metrics)
        │     └─[True]──► revert_patch() → continue next iter
        │
        ├─► run_evals(model_path, eval_suite)
        │     └─► new_score: EvalScore
        │
        ├─► compare_scores(new_score, baseline)
        │     └─► ScoreDelta  { relative_pct, improved }
        │
        ├─► flag_regression(delta)
        │     └─[True]──► revert_patch() → log REVERTED → continue
        │
        ├─► decide_keep_or_revert(delta)
        │     ├─[KEEP]──► update baseline_score
        │     └─[REVERT]► revert_patch()
        │
        ├─► log_iteration(diary, IterationRecord)
        │
        └─► (every 10 iters) adapt_eval_suite(suite, weaknesses)

  └─► return best TrainedModel
    `,
    functions: [
      {
        name: "run_autoresearch",
        signature: "run_autoresearch(plan: TrainingPlan, config: OrchestrationConfig, cost_manager: CostManager) -> TrainedModel",
        description: "Top-level loop. Runs baseline experiment, then iterates Propose → Run → Evaluate → Decide until budget is exhausted or no improvement for N iterations.",
        params: [
          { name: "plan", type: "TrainingPlan", description: "Output of run_decision_engine." },
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config (used for budget, task type, eval metric)." },
          { name: "cost_manager", type: "CostManager", description: "Running CostManager instance for budget enforcement." },
        ],
        returns: { type: "TrainedModel", description: "Best model found: { weights_path, metrics, cost_usd, n_iterations, research_diary_path }." },
      },
      // Propose
      {
        name: "propose_hypothesis",
        signature: "propose_hypothesis(current_config: dict, diary: ResearchDiary, task: TaskAnalysis) -> Hypothesis",
        description: "Calls the Claude API with the research diary and current config to generate a single testable hypothesis as a code/config diff. Uses random search over bounded ranges, local perturbations around the current best, and playbook heuristics.",
        params: [
          { name: "current_config", type: "dict", description: "Current training hyperparameters and architecture settings." },
          { name: "diary", type: "ResearchDiary", description: "All past iterations and their outcomes." },
          { name: "task", type: "TaskAnalysis", description: "Task context for domain-relevant proposals." },
        ],
        returns: { type: "Hypothesis", description: "{ description: str, patch: str, expected_effect: str, search_strategy: 'random'|'local'|'playbook' }." },
      },
      {
        name: "apply_patch",
        signature: "apply_patch(script_path: str, patch: str) -> str",
        description: "Applies a unified diff patch to the training script. Saves the original content before patching so revert_patch can restore it.",
        params: [
          { name: "script_path", type: "str", description: "Path to the current train.py." },
          { name: "patch", type: "str", description: "Unified diff string from propose_hypothesis." },
        ],
        returns: { type: "str", description: "The original script content (to pass to revert_patch if needed)." },
      },
      {
        name: "revert_patch",
        signature: "revert_patch(script_path: str, original_content: str) -> None",
        description: "Restores train.py to its pre-patch content when a hypothesis is rejected.",
        params: [
          { name: "script_path", type: "str", description: "Path to the patched train.py." },
          { name: "original_content", type: "str", description: "Original file content returned by apply_patch." },
        ],
        returns: { type: "None", description: "Writes to disk only." },
      },
      // Run
      {
        name: "submit_experiment",
        signature: "submit_experiment(script_path: str, plan: TrainingPlan, timeout_min: int = 5) -> str",
        description: "Submits a constrained (short, budgeted) training run to Tinker for a single hypothesis test. Returns the Tinker job ID.",
        params: [
          { name: "script_path", type: "str", description: "Path to the patched train.py to submit." },
          { name: "plan", type: "TrainingPlan", description: "Training plan (used for resource config)." },
          { name: "timeout_min", type: "int", description: "Max wall-clock minutes for this experiment. Defaults to 5 for mini runs.", optional: true },
        ],
        returns: { type: "str", description: "Tinker job ID." },
      },
      {
        name: "wait_for_experiment",
        signature: "wait_for_experiment(job_id: str, timeout_min: int) -> ExperimentResult",
        description: "Polls Tinker for job completion and returns the result. Raises TimeoutError if timeout_min is exceeded.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID from submit_experiment." },
          { name: "timeout_min", type: "int", description: "Maximum wait time in minutes." },
        ],
        returns: { type: "ExperimentResult", description: "{ job_id, status, metrics: TrainingMetrics, model_path, cost_usd, logs_path }." },
      },
      {
        name: "check_early_stop",
        signature: "check_early_stop(metrics: TrainingMetrics) -> bool",
        description: "Returns True if the run should be terminated early due to catastrophic failure: exploding loss (> 10× baseline), NaN values, or accuracy collapse (< chance level).",
        params: [
          { name: "metrics", type: "TrainingMetrics", description: "Live metrics from the experiment result." },
        ],
        returns: { type: "bool", description: "True if the run should be killed immediately." },
      },
      // Evaluate
      {
        name: "run_evals",
        signature: "run_evals(model_path: str, eval_suite: EvalSuite) -> EvalScore",
        description: "Runs the evaluation suite against the model at model_path and returns a scalar score plus per-metric breakdown.",
        params: [
          { name: "model_path", type: "str", description: "Path to saved model weights/checkpoint." },
          { name: "eval_suite", type: "EvalSuite", description: "Evaluation suite from create_eval_suite." },
        ],
        returns: { type: "EvalScore", description: "{ scalar: float, metrics: dict[str, float], critique: str }." },
      },
      {
        name: "compare_scores",
        signature: "compare_scores(new_score: EvalScore, baseline_score: EvalScore) -> ScoreDelta",
        description: "Computes the relative improvement of new_score vs baseline_score on the primary eval metric.",
        params: [
          { name: "new_score", type: "EvalScore", description: "Score from the current hypothesis run." },
          { name: "baseline_score", type: "EvalScore", description: "Best score achieved so far." },
        ],
        returns: { type: "ScoreDelta", description: "{ absolute: float, relative_pct: float, improved: bool }." },
      },
      // Decide
      {
        name: "decide_keep_or_revert",
        signature: "decide_keep_or_revert(delta: ScoreDelta) -> Literal['KEEP', 'REVERT']",
        description: "Returns KEEP if the hypothesis improved the primary metric, REVERT otherwise. A tie (delta.relative_pct == 0) defaults to REVERT to avoid script complexity growth.",
        params: [
          { name: "delta", type: "ScoreDelta", description: "Output of compare_scores." },
        ],
        returns: { type: "Literal['KEEP', 'REVERT']", description: "'KEEP' or 'REVERT'." },
      },
      {
        name: "log_iteration",
        signature: "log_iteration(diary: ResearchDiary, record: IterationRecord) -> ResearchDiary",
        description: "Appends an IterationRecord to the research diary and writes the updated diary to disk as a JSONL file.",
        params: [
          { name: "diary", type: "ResearchDiary", description: "Current research diary (list of past IterationRecords)." },
          { name: "record", type: "IterationRecord", description: "{ iteration, hypothesis, patch, cost_usd, metrics, decision, notes }." },
        ],
        returns: { type: "ResearchDiary", description: "Updated diary with the new record appended." },
      },
      // Evaluator sub-feature
      {
        name: "create_eval_suite",
        signature: "create_eval_suite(task: TaskAnalysis, dataset: DatasetResult) -> EvalSuite",
        description: "Creates the evaluation suite for this task: selects appropriate metrics (accuracy, F1, perplexity, etc.), holds out a fixed test split, and optionally adds LLM-graded stress tests.",
        params: [
          { name: "task", type: "TaskAnalysis", description: "Task analysis output from DecisionEngine." },
          { name: "dataset", type: "DatasetResult", description: "Dataset result with split sizes." },
        ],
        returns: { type: "EvalSuite", description: "{ primary_metric: str, metrics: list[str], test_split_path: str, use_llm_grading: bool }." },
      },
      {
        name: "adapt_eval_suite",
        signature: "adapt_eval_suite(suite: EvalSuite, weaknesses: list[str]) -> EvalSuite",
        description: "Adds harder evaluation examples targeting systematic weaknesses detected across recent iterations (e.g. if the model fails on long inputs, add more long-input test cases).",
        params: [
          { name: "suite", type: "EvalSuite", description: "Current evaluation suite." },
          { name: "weaknesses", type: "list[str]", description: "Natural-language descriptions of systematic failure patterns." },
        ],
        returns: { type: "EvalSuite", description: "Updated EvalSuite with additional test cases targeting the weaknesses." },
      },
      {
        name: "flag_regression",
        signature: "flag_regression(delta: ScoreDelta, threshold: float = -0.01) -> bool",
        description: "Returns True if the score degraded beyond the threshold, triggering an automatic revert.",
        params: [
          { name: "delta", type: "ScoreDelta", description: "Output of compare_scores." },
          { name: "threshold", type: "float", description: "Maximum tolerated relative score drop (default -1%).", optional: true },
        ],
        returns: { type: "bool", description: "True if regression is detected." },
      },
    ],
  },

  // ─── FEATURE 4: COST MANAGER ─────────────────────────────────────────────────
  {
    id: "cost-manager",
    title: "Feature 4 — Cost Manager",
    owner: "Sid Potti",
    description:
      "Hard financial guardrail. Continuously polls Tinker billing API every 30 seconds. Saves checkpoint at 90% of budget; kills the GPU instance at 100%. Returns final weights and a cost breakdown.",
    architecture: `The Cost Manager runs as a background thread — it never blocks the main training loop and never needs to be awaited. It is started once per Tinker job via start_cost_monitor(), which spawns the monitor thread and returns immediately.

The monitor thread runs a tight polling loop:
  every 30 seconds → poll_spend(job_id) → check_budget_status(spent, budget)
  → OK: do nothing
  → WARNING (≥90%): call save_checkpoint() and emit a log warning
  → EXCEEDED (≥100%): call save_checkpoint(), then kill_job(), then stop the thread

In addition to the budget-triggered saves, the training script itself calls save_checkpoint() every 5–10 minutes during the training loop. This ensures we always have a recent checkpoint even if the kill happens between polling cycles.

The Cost Manager also runs during AutoResearch mini-runs, not just the final training run. Each 5-minute experiment is registered with start_cost_monitor() so it can be killed if it somehow overruns.

At the end of the run (whether natural completion or budget kill), generate_cost_report() is called to produce the final CostBreakdown that gets returned to the user as part of TrainedModel.`,
    flowDiagram: `
start_cost_monitor(job_id, budget, poll_interval=30)
  └─► spawns background thread, returns immediately

BACKGROUND THREAD:
  loop every 30s:
    ├─► poll_spend(job_id)           ← Tinker billing API
    │     └─► spent: float
    │
    ├─► check_budget_status(spent, budget)
    │     └─► BudgetStatus: OK | WARNING | EXCEEDED
    │
    ├─[OK]──► continue polling
    │
    ├─[WARNING ≥90%]──────────────────────────────────
    │   ├─► save_checkpoint(job_id, output_dir)
    │   └─► log_event(COST_MANAGER, WARN, "90% budget used")
    │
    └─[EXCEEDED ≥100%]────────────────────────────────
        ├─► save_checkpoint(job_id, output_dir)
        ├─► kill_job(job_id)          ← Tinker API: terminate instance
        ├─► log_event(COST_MANAGER, WARN, "Budget limit reached")
        └─► stop thread

TRAINING SCRIPT (every 5-10 min):
  └─► save_checkpoint()  ← called from within train.py directly

END OF RUN:
  └─► generate_cost_report(job_id)
        └─► CostBreakdown  { data_gen_usd, training_usd, llm_calls_usd, total_usd }
    `,
    functions: [
      {
        name: "start_cost_monitor",
        signature: "start_cost_monitor(job_id: str, budget: float, poll_interval_sec: int = 30) -> threading.Thread",
        description: "Starts a background thread that polls Tinker billing every poll_interval_sec seconds. The thread calls save_checkpoint at 90% budget and kill_job at 100%.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID to monitor." },
          { name: "budget", type: "float", description: "Hard budget cap in USD." },
          { name: "poll_interval_sec", type: "int", description: "Polling interval in seconds. Default 30.", optional: true },
        ],
        returns: { type: "threading.Thread", description: "Running background monitor thread. Call thread.join() to block until the job ends." },
      },
      {
        name: "poll_spend",
        signature: "poll_spend(job_id: str) -> float",
        description: "Calls Tinker billing API to fetch cumulative USD spend for a job.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID." },
        ],
        returns: { type: "float", description: "Cumulative USD spend so far." },
      },
      {
        name: "check_budget_status",
        signature: "check_budget_status(spent: float, budget: float) -> BudgetStatus",
        description: "Returns the current budget status given spend vs. budget.",
        params: [
          { name: "spent", type: "float", description: "Current cumulative spend in USD." },
          { name: "budget", type: "float", description: "Budget cap in USD." },
        ],
        returns: { type: "BudgetStatus", description: "Enum: OK (< 90%), WARNING (90–99%), EXCEEDED (>= 100%)." },
      },
      {
        name: "save_checkpoint",
        signature: "save_checkpoint(job_id: str, output_dir: str) -> str",
        description: "Saves the current model state_dict to disk. Called automatically at 90% budget and every 5–10 minutes during training.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID (used to locate the running process)." },
          { name: "output_dir", type: "str", description: "Directory to write the checkpoint file." },
        ],
        returns: { type: "str", description: "Absolute path to the saved checkpoint file." },
      },
      {
        name: "kill_job",
        signature: "kill_job(job_id: str) -> None",
        description: "Calls Tinker API to immediately terminate the GPU instance for job_id. Called when spend >= budget.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID to kill." },
        ],
        returns: { type: "None", description: "Side effect only: terminates the Tinker job." },
      },
      {
        name: "generate_cost_report",
        signature: "generate_cost_report(job_id: str) -> CostBreakdown",
        description: "Fetches the final cost breakdown from Tinker and splits it into components (data generation, training, LLM calls). Returned to the user at end of run.",
        params: [
          { name: "job_id", type: "str", description: "Completed or terminated Tinker job ID." },
        ],
        returns: { type: "CostBreakdown", description: "{ data_gen_usd, training_usd, llm_calls_usd, total_usd, termination_reason }." },
      },
    ],
  },

  // ─── FEATURE 5: OBSERVABILITY ────────────────────────────────────────────────
  {
    id: "observability",
    title: "Feature 5 — Observability",
    owner: "Team",
    description:
      "Structured logging for all agent decisions, training metrics, and budget usage. Emits human-readable CLI output in real time and writes machine-readable JSON logs to disk.",
    architecture: `Observability is a shared utility — every other agent imports and calls log_event(). No agent writes to stdout or disk directly. This keeps all output consistent and means you can add structured logging to any new function by adding one line.

The call chain is always: log_event() → format_cli_line() + emit_cli() (stdout) AND write_json_log() (disk). Both happen synchronously on every call, so log output is always up to date.

log_event() takes an AgentName enum so the CLI output is color-coded by agent, making it easy to see which part of the system is running. The metadata dict is written to the JSON log as structured data, so metrics, diffs, and cost figures are all machine-readable.

get_budget_display() is a pure formatting utility used by the Cost Manager to produce the budget status line that appears in CLI output after every experiment.

There's no singleton or global state — log_event() takes a log_path argument so tests can redirect logs to a temp file without monkeypatching.`,
    flowDiagram: `
Any agent calls:
  log_event(agent, level, message, metadata)
    │
    ├─► format_cli_line(entry)
    │     └─► "[DataGen] ✓ Found dataset (50K examples)"
    │
    ├─► emit_cli(entry)
    │     └─► print to stdout with ANSI color by agent
    │
    └─► write_json_log(entry, log_path)
          └─► append JSON line to run.jsonl

Cost Manager calls:
  get_budget_display(spent, budget)
    └─► "Spend: $9.20 / $50.00 (18% used)"
    `,
    functions: [
      {
        name: "log_event",
        signature: "log_event(agent: AgentName, level: LogLevel, message: str, metadata: dict = {}) -> None",
        description: "Central logging function called by every agent. Writes a structured JSON entry to the log file and calls emit_cli to display it.",
        params: [
          { name: "agent", type: "AgentName", description: "Enum: MANAGER | DATA_GEN | DECISION_ENGINE | AUTORESEARCH | COST_MANAGER." },
          { name: "level", type: "LogLevel", description: "Enum: INFO | WARN | ERROR." },
          { name: "message", type: "str", description: "Human-readable log message." },
          { name: "metadata", type: "dict", description: "Optional structured payload (e.g. metrics, cost, config diff).", optional: true },
        ],
        returns: { type: "None", description: "Writes to disk and stdout." },
      },
      {
        name: "format_cli_line",
        signature: "format_cli_line(entry: LogEntry) -> str",
        description: "Formats a LogEntry into the standard CLI output format: [AgentName] message.",
        params: [
          { name: "entry", type: "LogEntry", description: "Structured log entry." },
        ],
        returns: { type: "str", description: "Formatted string, e.g. '[DataGen] ✓ Found dataset (50K examples)'." },
      },
      {
        name: "emit_cli",
        signature: "emit_cli(entry: LogEntry) -> None",
        description: "Prints the formatted CLI line to stdout with ANSI color coding by agent and level.",
        params: [
          { name: "entry", type: "LogEntry", description: "Log entry to display." },
        ],
        returns: { type: "None", description: "Writes to stdout." },
      },
      {
        name: "write_json_log",
        signature: "write_json_log(entry: LogEntry, log_path: str) -> None",
        description: "Appends the LogEntry as a JSON line to the structured log file for programmatic access.",
        params: [
          { name: "entry", type: "LogEntry", description: "Log entry to persist." },
          { name: "log_path", type: "str", description: "Path to the .jsonl log file." },
        ],
        returns: { type: "None", description: "Appends to disk." },
      },
      {
        name: "get_budget_display",
        signature: "get_budget_display(spent: float, budget: float) -> str",
        description: "Returns a formatted budget string for CLI display.",
        params: [
          { name: "spent", type: "float", description: "Current spend in USD." },
          { name: "budget", type: "float", description: "Budget cap in USD." },
        ],
        returns: { type: "str", description: "e.g. 'Spend: $9.20 / $50.00 (18% used)'." },
      },
    ],
  },

  // ─── TINKER API WRAPPER ───────────────────────────────────────────────────────
  {
    id: "tinker-api",
    title: "Tinker API Wrapper",
    owner: "Sid Potti",
    description:
      "Thin wrapper around Tinker's job submission and billing REST APIs. All GPU training runs and cost monitoring go through these functions.",
    architecture: `The Tinker API Wrapper is a pure HTTP client — no business logic, no state. It exists to give the rest of the system a typed, mockable interface to Tinker's REST APIs so that every other feature doesn't need to know about auth headers, retry logic, or response parsing.

Two Tinker APIs are used:
  Job API: submit_job, get_job_status, cancel_job, get_job_logs, list_jobs
  Billing API: get_cumulative_spend

Callers:
  AutoResearch calls submit_job() and wait_for_experiment() (which polls get_job_status()).
  Cost Manager calls get_cumulative_spend() every 30 seconds and cancel_job() when budget is exceeded.
  Observability optionally calls get_job_logs() to stream training output.

All functions raise a TinkerAPIError on non-2xx responses. Retry logic (exponential backoff, max 3 attempts) is handled internally so callers don't need to implement it.

NOTE: Tinker API docs + auth credentials are a hard dependency. This wrapper cannot be built until Sid confirms the API spec (target: Apr 18). All other features can be built with a mock implementation of this module in the meantime.`,
    flowDiagram: `
AutoResearch:
  submit_job(script_path, job_config)  ──►  POST /jobs
    └─► job_id: str (UUID)

  get_job_status(job_id)  ──────────►  GET /jobs/{id}/status
    └─► JobStatus: PENDING | RUNNING | COMPLETED | FAILED

Cost Manager:
  get_cumulative_spend(job_id)  ──────►  GET /billing/{id}/spend
    └─► float (USD)

  cancel_job(job_id)  ────────────────►  POST /jobs/{id}/cancel

Observability (optional):
  get_job_logs(job_id, tail=100)  ─────►  GET /jobs/{id}/logs?tail=100
    └─► list[str]

All functions:
  - Raise TinkerAPIError on non-2xx
  - Retry up to 3× with exponential backoff
  - Log every call via log_event(TINKER_API, INFO, ...)
    `,
    functions: [
      {
        name: "submit_job",
        signature: "submit_job(script_path: str, job_config: JobConfig) -> str",
        description: "Submits a training script to Tinker for execution on a GPU instance. Returns the Tinker job ID.",
        params: [
          { name: "script_path", type: "str", description: "Absolute path to the train.py script." },
          { name: "job_config", type: "JobConfig", description: "{ gpu_type, num_gpus, timeout_min, env_vars, output_dir }." },
        ],
        returns: { type: "str", description: "Tinker job ID (UUID string)." },
      },
      {
        name: "get_job_status",
        signature: "get_job_status(job_id: str) -> JobStatus",
        description: "Fetches the current status of a Tinker job.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID." },
        ],
        returns: { type: "JobStatus", description: "Enum: PENDING | RUNNING | COMPLETED | FAILED | CANCELLED." },
      },
      {
        name: "get_cumulative_spend",
        signature: "get_cumulative_spend(job_id: str) -> float",
        description: "Returns the cumulative USD spend for a job from the Tinker billing API.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID." },
        ],
        returns: { type: "float", description: "Cumulative spend in USD." },
      },
      {
        name: "cancel_job",
        signature: "cancel_job(job_id: str) -> None",
        description: "Immediately cancels and terminates a Tinker job, releasing the GPU instance.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID to cancel." },
        ],
        returns: { type: "None", description: "Side effect: Tinker terminates the GPU instance." },
      },
      {
        name: "get_job_logs",
        signature: "get_job_logs(job_id: str, tail: int = 100) -> list[str]",
        description: "Fetches the last N lines of stdout/stderr from a running or completed Tinker job.",
        params: [
          { name: "job_id", type: "str", description: "Tinker job ID." },
          { name: "tail", type: "int", description: "Number of log lines to return from the end. Default 100.", optional: true },
        ],
        returns: { type: "list[str]", description: "List of log line strings." },
      },
      {
        name: "list_jobs",
        signature: "list_jobs(limit: int = 20) -> list[JobSummary]",
        description: "Lists recent Tinker jobs for the current account, ordered by submission time descending.",
        params: [
          { name: "limit", type: "int", description: "Maximum number of jobs to return.", optional: true },
        ],
        returns: { type: "list[JobSummary]", description: "List of JobSummary: { job_id, status, submitted_at, cost_usd, script_name }." },
      },
    ],
  },
];

// ─── SHARED TYPES REFERENCE ───────────────────────────────────────────────────
export type TypeDef = {
  name: string;
  description: string;
  fields: { name: string; type: string; description: string }[];
};

export const sharedTypes: TypeDef[] = [
  {
    name: "OrchestrationConfig",
    description: "Central JSON passed from Manager to all downstream agents.",
    fields: [
      { name: "data", type: "bool", description: "Whether the user provided their own data." },
      { name: "prompt", type: "str", description: "Original user task description." },
      { name: "compute_budget", type: "float", description: "Hard budget cap in USD." },
      { name: "training_procedure", type: "TrainingProcedure", description: "Nested object with all training details." },
    ],
  },
  {
    name: "TrainingProcedure",
    description: "Nested inside OrchestrationConfig. Describes the training strategy.",
    fields: [
      { name: "task_type", type: "str", description: "e.g. 'text-classification', 'seq2seq', 'custom'." },
      { name: "data_format", type: "str", description: "Expected format for training data." },
      { name: "training_type", type: "str", description: "'SFT', 'RL', or 'pre-train'." },
      { name: "base_model", type: "str | None", description: "HuggingFace model ID or None for pre-train." },
      { name: "hyperparameters", type: "dict", description: "Starting lr, batch_size, epochs, etc." },
      { name: "notes", type: "str", description: "Free-text notes from manager reasoning." },
    ],
  },
  {
    name: "StandardDataset",
    description: "Normalized training dataset output by any DataGenerator mode.",
    fields: [
      { name: "path", type: "str", description: "Path to dataset on disk." },
      { name: "format", type: "str", description: "'jsonl' | 'csv' | 'parquet'." },
      { name: "train_size", type: "int", description: "Number of training examples." },
      { name: "val_size", type: "int", description: "Number of validation examples." },
      { name: "test_size", type: "int", description: "Number of test examples." },
    ],
  },
  {
    name: "DatasetResult",
    description: "Return type of run_data_generator.",
    fields: [
      { name: "dataset", type: "StandardDataset", description: "The standardized dataset." },
      { name: "mode_used", type: "str", description: "'A' | 'B' | 'C'." },
      { name: "quality_notes", type: "str", description: "Human-readable quality assessment." },
      { name: "validation_report", type: "ValidationReport", description: "Pass/fail quality check." },
    ],
  },
  {
    name: "TrainingPlan",
    description: "Return type of run_decision_engine.",
    fields: [
      { name: "strategy", type: "str", description: "'fine-tune' | 'pre-train'." },
      { name: "base_model", type: "str | None", description: "HuggingFace model ID or None." },
      { name: "lora_config", type: "LoRAConfig | None", description: "LoRA settings if fine-tuning." },
      { name: "estimated_cost", type: "float", description: "Estimated USD for full training run." },
      { name: "estimated_time_min", type: "int", description: "Estimated wall-clock minutes." },
      { name: "training_script_path", type: "str", description: "Path to generated train.py." },
      { name: "eval_metric", type: "str", description: "Primary metric to optimize." },
    ],
  },
  {
    name: "EvalScore",
    description: "Return type of run_evals.",
    fields: [
      { name: "scalar", type: "float", description: "Single scalar score on primary metric (higher = better)." },
      { name: "metrics", type: "dict[str, float]", description: "Full per-metric breakdown." },
      { name: "critique", type: "str", description: "Natural-language analysis of model weaknesses." },
    ],
  },
  {
    name: "IterationRecord",
    description: "A single entry in the research diary.",
    fields: [
      { name: "iteration", type: "int", description: "Iteration number (1-indexed)." },
      { name: "hypothesis", type: "str", description: "Description of what was tested." },
      { name: "patch", type: "str", description: "The unified diff that was applied." },
      { name: "cost_usd", type: "float", description: "USD cost of this experiment." },
      { name: "metrics", type: "TrainingMetrics", description: "train_loss, val_loss, test_loss, primary_metric." },
      { name: "decision", type: "str", description: "'KEPT' | 'REVERTED'." },
      { name: "notes", type: "str", description: "Free-text notes." },
    ],
  },
  {
    name: "CostBreakdown",
    description: "Final cost report returned to the user.",
    fields: [
      { name: "data_gen_usd", type: "float", description: "Cost of data generation (API calls, downloads)." },
      { name: "training_usd", type: "float", description: "Cost of GPU training time on Tinker." },
      { name: "llm_calls_usd", type: "float", description: "Cost of Claude API calls during AutoResearch." },
      { name: "total_usd", type: "float", description: "Total spend." },
      { name: "termination_reason", type: "str", description: "'budget_limit' | 'training_complete' | 'error'." },
    ],
  },
  {
    name: "TrainedModel",
    description: "Final output returned to the user after the full pipeline.",
    fields: [
      { name: "weights_path", type: "str", description: "Path to the best model checkpoint." },
      { name: "metrics", type: "EvalScore", description: "Final evaluation score." },
      { name: "cost", type: "CostBreakdown", description: "Full cost breakdown." },
      { name: "n_iterations", type: "int", description: "Number of AutoResearch iterations run." },
      { name: "research_diary_path", type: "str", description: "Path to the JSONL research diary." },
    ],
  },
];
