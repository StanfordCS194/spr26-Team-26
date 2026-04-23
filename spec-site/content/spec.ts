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
  overview: `The system is a linear pipeline with one autonomous feedback loop. The user provides a single prompt and a budget. The Manager reasons about the task and emits a config object that every other agent reads from. Control then flows through three sequential stages — data, decisions, training — with the Cost Manager running as a background watchdog throughout.

LangGraph is used for the three stateful, multi-step agent processes: the Manager (linear graph with one Claude call), the Data Generator (conditional routing graph), and the AutoResearch loop (cyclic graph). The Decision Engine, Cost Manager, Observability module, and Tinker API wrapper are plain Python — they have no branching agent logic that would benefit from a graph runtime.

LangGraph gives us: (1) built-in checkpointing so a long AutoResearch run can resume after a crash, (2) a typed state object (TypedDict) that makes inter-node data flow explicit and auditable, and (3) conditional edges that make the Mode A/B/C and KEEP/REVERT branching readable and testable.`,
  flowDiagram: `
User
  │  prompt: str
  │  budget: float
  │  data_path?: str
  ▼
┌─────────────────────────────────────┐
│   Manager Agent  [LangGraph]        │
│   StateGraph(ManagerState)          │
│                                     │
│   query_data ──► reason ──►         │
│   build_config ──► orchestrate      │
└───────────────┬─────────────────────┘
                │  OrchestrationConfig  (in ManagerState)
                │  (passed to ALL agents)
                ▼
┌─────────────────────────────────────┐
│   Data Generator  [LangGraph]       │
│   StateGraph(DataGenState)          │
│                                     │
│   Sub-agent 1 (Acquisition)         │
│   route ──► acquire_user_data       │
│         └─► acquire_hf_data         │◄── HuggingFace Hub API
│               └─► acquire_web_data  │◄── Web/LLM fallback
│   emits handoff payload             │
└───────────────┬─────────────────────┘
                │  handoff payload
                ▼
┌─────────────────────────────────────┐
│   Data Curation Sub-agent           │
│   Sub-agent 2 (Curation)            │
│                                     │
│   structure_data / validate_hf_data │
│   validate + (optional) synth aug   │
└───────────────┬─────────────────────┘
                │  DatasetResult
                ▼
┌─────────────────────────────────────┐
│   Decision Engine  [plain Python]   │
│   analyze_task()                    │
│   find_base_model()   ◄─── HF Hub   │
│   estimate_training_cost()          │
│   write_finetune_script() OR        │
│   write_pretrain_script()           │
└───────────────┬─────────────────────┘
                │  TrainingPlan
                ▼
┌─────────────────────────────────────┐   ┌───────────────────────┐
│   AutoResearch  [LangGraph]         │   │  Cost Manager          │
│   StateGraph(AutoResearchState)     │   │  (background thread)   │
│                                     │   │                        │
│   init ──► baseline                 │   │  poll_spend() / 30s    │
│   ┌──────────────────────────────┐  │──►│  save_checkpoint()@90% │
│   │  propose ──► run             │  │   │  kill_job() @ 100%     │
│   │    ├─[early stop]──► revert  │  │   └───────────────────────┘
│   │    └─► evaluate ──► decide   │  │
│   │          ├─[KEEP]────────────┤  │
│   │          └─[REVERT]──► log ──┘  │
│   └── (loop or END on budget/conv.) ┘  │
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
    architecture: `The Manager is implemented as a LangGraph StateGraph(ManagerState). It is the only agent the user interacts with directly and runs entirely locally — no GPU required.

Why LangGraph here: the Manager makes a Claude API call whose output determines what happens next (does the task need fine-tuning or pre-training? does the user have data?). Modeling this as a graph makes the state transitions explicit, checkpointable, and easy to extend with human-in-the-loop pauses later.

Graph nodes (4 nodes, linear with no cycles):
  1. query_data_node — asks the user if they have existing data; writes has_data + data_path into ManagerState
  2. reason_node — calls Claude API with the prompt; writes task_reasoning into ManagerState
  3. build_config_node — assembles OrchestrationConfig from ManagerState fields; writes config
  4. orchestrate_node — calls the downstream pipeline (DataGen → DecisionEngine → AutoResearch) using the config from state; writes result into ManagerState

The graph is compiled once with build_manager_graph() and invoked via invoke_manager_graph(). The compiled graph handles state passing between nodes automatically — no manual threading of return values.

Existing helper functions (query_user_for_data, reason_about_task, build_orchestration_config, log_decision) are called inside their respective node functions. Their signatures don't change.`,
    flowDiagram: `
build_manager_graph() → CompiledStateGraph[ManagerState]

ManagerState = TypedDict:
  prompt, budget, data_path,
  has_data, task_reasoning, config, result

Graph nodes and edges:
  START
    │
    ▼
  query_data_node          calls: query_user_for_data()
    │  writes: has_data, data_path
    ▼
  reason_node              calls: reason_about_task()  [Claude API]
    │  writes: task_reasoning
    ▼
  build_config_node        calls: build_orchestration_config(), log_decision()
    │  writes: config (OrchestrationConfig)
    ▼
  orchestrate_node         calls: DataGen graph → DecisionEngine → AutoResearch graph
    │  writes: result (TrainedModel)
    ▼
  END

invoke_manager_graph(prompt, budget, data_path?) → TrainedModel
  └─► graph.invoke({ prompt, budget, data_path })["result"]
    `,
    functions: [
      {
        name: "build_manager_graph",
        signature: "build_manager_graph() -> CompiledStateGraph[ManagerState]",
        description: "Constructs and compiles the Manager LangGraph StateGraph. Returns the compiled graph ready to invoke. Called once at startup.",
        params: [],
        returns: { type: "CompiledStateGraph[ManagerState]", description: "Compiled LangGraph graph with nodes: query_data → reason → build_config → orchestrate." },
      },
      {
        name: "invoke_manager_graph",
        signature: "invoke_manager_graph(prompt: str, budget: float, data_path: str | None = None) -> TrainedModel",
        description: "Main entry point for the entire system. Invokes the compiled Manager graph with initial state and returns the final TrainedModel.",
        params: [
          { name: "prompt", type: "str", description: "Plain-English task description from the user." },
          { name: "budget", type: "float", description: "Hard dollar cap for the entire run." },
          { name: "data_path", type: "str | None", description: "Path to user-provided data, or None.", optional: true },
        ],
        returns: { type: "TrainedModel", description: "Final trained model with weights path, eval score, and cost breakdown." },
      },
      {
        name: "query_data_node",
        signature: "query_data_node(state: ManagerState) -> dict",
        description: "LangGraph node. Calls query_user_for_data() and returns a partial state update with has_data and data_path.",
        params: [{ name: "state", type: "ManagerState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { has_data: bool, data_path: str | None }." },
      },
      {
        name: "reason_node",
        signature: "reason_node(state: ManagerState) -> dict",
        description: "LangGraph node. Calls reason_about_task() with state.prompt, state.budget, state.has_data. Returns partial state with task_reasoning.",
        params: [{ name: "state", type: "ManagerState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { task_reasoning: TaskReasoning }." },
      },
      {
        name: "build_config_node",
        signature: "build_config_node(state: ManagerState) -> dict",
        description: "LangGraph node. Calls build_orchestration_config() and log_decision(). Returns partial state with the final OrchestrationConfig.",
        params: [{ name: "state", type: "ManagerState", description: "Current graph state (needs task_reasoning populated)." }],
        returns: { type: "dict", description: "Partial state update: { config: OrchestrationConfig }." },
      },
      {
        name: "orchestrate_node",
        signature: "orchestrate_node(state: ManagerState) -> dict",
        description: "LangGraph node. Sequences the downstream pipeline: invokes the DataGen graph, then calls DecisionEngine functions, then invokes the AutoResearch graph. Returns partial state with the final result.",
        params: [{ name: "state", type: "ManagerState", description: "Current graph state (needs config populated)." }],
        returns: { type: "dict", description: "Partial state update: { result: TrainedModel }." },
      },
      {
        name: "query_user_for_data",
        signature: "query_user_for_data() -> str | None",
        description: "Helper (called inside query_data_node). Interactively asks the user whether they have existing training data.",
        params: [],
        returns: { type: "str | None", description: "Absolute path to the user's data, or None." },
      },
      {
        name: "reason_about_task",
        signature: "reason_about_task(prompt: str, budget: float, has_data: bool) -> TaskReasoning",
        description: "Helper (called inside reason_node). Calls the Claude API to infer task type, data format, training type, base model, and starting hyperparameters.",
        params: [
          { name: "prompt", type: "str", description: "Raw user task description." },
          { name: "budget", type: "float", description: "Budget cap in USD." },
          { name: "has_data", type: "bool", description: "Whether the user is supplying their own data." },
        ],
        returns: { type: "TaskReasoning", description: "Structured reasoning: task_type, data_format, training_type (SFT/RL), suggested base model, hyperparameters, notes." },
      },
      {
        name: "build_orchestration_config",
        signature: "build_orchestration_config(reasoning: TaskReasoning, prompt: str, budget: float, has_data: bool) -> OrchestrationConfig",
        description: "Helper (called inside build_config_node). Assembles the OrchestrationConfig dict passed to all downstream agents.",
        params: [
          { name: "reasoning", type: "TaskReasoning", description: "Output from reason_about_task." },
          { name: "prompt", type: "str", description: "Original user prompt." },
          { name: "budget", type: "float", description: "Budget cap in USD." },
          { name: "has_data", type: "bool", description: "Whether user supplied data." },
        ],
        returns: { type: "OrchestrationConfig", description: "{ data, prompt, compute_budget, training_procedure: { task_type, data_format, training_type, base_model, hyperparameters, notes } }." },
      },
      {
        name: "log_decision",
        signature: "log_decision(step: str, rationale: str, config: OrchestrationConfig) -> None",
        description: "Helper (called inside build_config_node). Appends a timestamped entry to the audit trail log (decisions.jsonl).",
        params: [
          { name: "step", type: "str", description: "Pipeline step name." },
          { name: "rationale", type: "str", description: "Human-readable explanation of the decision." },
          { name: "config", type: "OrchestrationConfig", description: "Current config snapshot." },
        ],
        returns: { type: "None", description: "Writes to disk only." },
      },
    ],
  },

  // ─── FEATURE 1: DATA GENERATOR ───────────────────────────────────────────────
  {
    id: "data-generator",
    title: "Feature 1 — Data Generator",
    owner: "Ron Polonsky, Angel Raychev",
    description:
      "Refactored into two sub-agents: Sub-agent 1 does acquisition/routing (implemented now), then hands off to Sub-agent 2 for curation and structuring.",
    architecture: `The Data Generator is split into two sub-agents.

Sub-agent 1 (implemented in current src/data_generator):
  - LangGraph StateGraph(DataGenState) for Mode A/B/C acquisition
  - Produces a handoff payload for Sub-agent 2

Sub-agent 2 (data curation stage):
  - Consumes the handoff payload
  - Performs structuring, validation, and optional synthetic augmentation
  - Produces DatasetResult for downstream DecisionEngine

Current implementation scope in this repo path is Sub-agent 1 only:
  route_node → acquire_user_data_node / acquire_hf_data_node / acquire_web_data_node
  then select_curation_edge routes to:
    - handoff_structure_data_node (strict)
    - handoff_validate_hf_node (light)

invoke_data_generator_graph(config, data_path) currently returns the handoff payload for Sub-agent 2.`,
    flowDiagram: `
Sub-agent 1 (implemented now)
build_data_generator_graph() → CompiledStateGraph[DataGenState]

DataGenState = TypedDict:
  config, data_path, mode,
  raw_data, hf_candidates, selected_candidate,
  schema, dataset, validation_report, handoff

Graph nodes and edges:
  START
    │
    ▼
  route_node                 inspects data_path/explicit HF IDs → sets mode
    │
    ├─[mode == "A"]────────► acquire_user_data_node ─┐
    ├─[mode == "B"]────────► acquire_hf_data_node   ├─► select_curation_edge
    └─[mode == "C"]────────► acquire_web_data_node  ┘
                                                      │
                                                      ├─[A/C]► handoff_structure_data_node
                                                      └─[B] ─► handoff_validate_hf_node
                                                                │
                                                                ▼
                                                               END
                                  writes handoff payload to state["handoff"]

Sub-agent 2 (separate curation stage)
  input: handoff payload
  output: DatasetResult (structured + validated dataset)
    `,
    functions: [
      {
        name: "build_data_generator_graph",
        signature: "build_data_generator_graph() -> CompiledStateGraph[DataGenState]",
        description: "Constructs and compiles Sub-agent 1 acquisition graph with route/acquire/handoff nodes.",
        params: [],
        returns: { type: "CompiledStateGraph[DataGenState]", description: "Compiled graph with nodes: route → acquire(A/B/C) → handoff(A/C or B)." },
      },
      {
        name: "invoke_data_generator_graph",
        signature: "invoke_data_generator_graph(config: OrchestrationConfig, data_path: str | None) -> dict",
        description: "Entry point for Sub-agent 1. Invokes acquisition graph and returns the handoff payload for Sub-agent 2.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config from the Manager." },
          { name: "data_path", type: "str | None", description: "Path to user-provided raw data, or None." },
        ],
        returns: { type: "dict", description: "Handoff payload: { target_subagent, action, verification_level, mode_used, raw_data, ... }." },
      },
      {
        name: "route_node",
        signature: "route_node(state: DataGenState) -> dict",
        description: "Determines acquisition mode: A (user data path), B (explicit HF IDs), or C (fallback web acquisition).",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { mode: 'A' | 'B' | 'C' }." },
      },
      {
        name: "select_mode_edge",
        signature: "select_mode_edge(state: DataGenState) -> Literal['acquire_user_data', 'acquire_hf_data', 'acquire_web_data']",
        description: "Conditional edge after route_node.",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state (mode must be set)." }],
        returns: { type: "Literal['acquire_user_data', 'acquire_hf_data', 'acquire_web_data']", description: "Target node name." },
      },
      {
        name: "select_curation_edge",
        signature: "select_curation_edge(state: DataGenState) -> Literal['structure_data', 'validate_hf_data']",
        description: "Conditional edge after acquisition nodes. Routes by mode: A/C -> structure_data, B -> validate_hf_data.",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state after acquisition." }],
        returns: { type: "Literal['structure_data', 'validate_hf_data']", description: "Target handoff node." },
      },
      {
        name: "acquire_user_data_node",
        signature: "acquire_user_data_node(state: DataGenState) -> dict",
        description: "Mode A acquisition node. Loads raw data from user-provided path.",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state (data_path must be set)." }],
        returns: { type: "dict", description: "Partial state update: { raw_data: RawData }." },
      },
      {
        name: "acquire_hf_data_node",
        signature: "acquire_hf_data_node(state: DataGenState) -> dict",
        description: "Mode B acquisition node. Parses explicit HF IDs, builds candidates, fetches HF rows.",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { hf_candidates, selected_candidate, raw_data }." },
      },
      {
        name: "acquire_web_data_node",
        signature: "acquire_web_data_node(state: DataGenState) -> dict",
        description: "Mode C acquisition node. Uses web retrieval fallback when no explicit/user data is available.",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { raw_data: RawData }." },
      },
      {
        name: "handoff_structure_data_node",
        signature: "handoff_structure_data_node(state: DataGenState) -> dict",
        description: "Builds strict handoff payload for Sub-agent 2 (used by Mode A/C).",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { handoff: {...action: 'structure_data', verification_level: 'strict'} }." },
      },
      {
        name: "handoff_validate_hf_node",
        signature: "handoff_validate_hf_node(state: DataGenState) -> dict",
        description: "Builds light-verification handoff payload for Sub-agent 2 (used by Mode B).",
        params: [{ name: "state", type: "DataGenState", description: "Current graph state." }],
        returns: { type: "dict", description: "Partial state update: { handoff: {...action: 'validate_hf_dataset', verification_level: 'light'} }." },
      },
      {
        name: "load_raw_data",
        signature: "load_raw_data(data_path: str) -> RawData",
        description: "Loads user-provided raw data into RawData contract.",
        params: [
          { name: "data_path", type: "str", description: "Path to user-provided dataset." },
        ],
        returns: { type: "RawData", description: "Raw records + format metadata." },
      },
      {
        name: "parse_explicit_hf_dataset_ids",
        signature: "parse_explicit_hf_dataset_ids(config: OrchestrationConfig, data_path: str | None = None) -> list[str]",
        description: "Extracts and normalizes explicit HF dataset IDs/URLs from config and prompt.",
        params: [
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config from Manager." },
          { name: "data_path", type: "str | None", description: "Optional path shortcut (supports hf://...).", optional: true },
        ],
        returns: { type: "list[str]", description: "Normalized HF dataset IDs." },
      },
      {
        name: "build_explicit_hf_candidates",
        signature: "build_explicit_hf_candidates(dataset_ids: list[str], task_type: str) -> list[HFCandidate]",
        description: "Builds Mode B candidate metadata for explicit HF IDs.",
        params: [
          { name: "dataset_ids", type: "list[str]", description: "Explicit normalized HF IDs." },
          { name: "task_type", type: "str", description: "Task type for metadata tags." },
        ],
        returns: { type: "list[HFCandidate]", description: "Candidate list for Mode B handoff." },
      },
      {
        name: "fetch_hf_datasets",
        signature: "fetch_hf_datasets(candidates: list[HFCandidate]) -> RawData",
        description: "Fetches rows from explicit HF datasets and merges into RawData.",
        params: [
          { name: "candidates", type: "list[HFCandidate]", description: "Mode B candidates to fetch." },
        ],
        returns: { type: "RawData", description: "Fetched records with format metadata." },
      },
      {
        name: "acquire_web_data",
        signature: "acquire_web_data(query: str) -> RawData",
        description: "Mode C helper for web retrieval fallback.",
        params: [
          { name: "query", type: "str", description: "Prompt-derived query." },
        ],
        returns: { type: "RawData", description: "Web-acquired raw records." },
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
    architecture: `The AutoResearch loop is implemented as a LangGraph StateGraph(AutoResearchState) with a cycle. This is the most natural use of LangGraph in the system — the loop is literally a graph: propose → run → evaluate → decide, with conditional edges that either loop back to propose or exit to END.

Why LangGraph here (most important use):
  - The loop must be resumable. If a Tinker job crashes mid-run or the process is killed, LangGraph's built-in checkpointing lets the loop resume from the last completed node rather than restarting from scratch.
  - The conditional branching (early stop vs. evaluate, keep vs. revert, continue vs. stop) maps directly to LangGraph conditional edges, making the control flow readable and independently testable.
  - AutoResearchState holds all mutable loop data (diary, current_script, best_score, iteration count) in one typed dict. No hidden globals or class state.

Graph structure (cyclic):
  init_node → baseline_node → propose_node → run_node → [early_stop_edge] → evaluate_node → [decision_edge] → log_node → [continue_edge] → propose_node (loop) or END

Conditional edges:
  early_stop_edge — after run_node: if metrics show catastrophic failure, go to revert_and_continue_node (revert patch + increment iter) else go to evaluate_node.
  decision_edge — after evaluate_node: if delta.improved → keep_node, else → revert_node. Both then go to log_node.
  continue_edge — after log_node: if budget exhausted OR no improvement for N iters → END, else → propose_node.

The Evaluator sub-feature nodes (create_eval_suite inside init_node, run_evals inside evaluate_node, adapt_eval_suite called periodically from log_node) can be built and tested as standalone functions — the nodes just call them.`,
    flowDiagram: `
build_autoresearch_graph() → CompiledStateGraph[AutoResearchState]

AutoResearchState = TypedDict:
  plan, config, eval_suite,
  current_script, current_config, original_content,
  diary, baseline_score, best_score,
  last_result, last_score, last_delta,
  iteration, no_improve_streak, should_stop

Graph nodes and edges:
  START
    │
    ▼
  init_node              create_eval_suite()
    │
    ▼
  baseline_node          submit_experiment → wait → run_evals
    │                    sets baseline_score + best_score
    ▼
  propose_node  ◄────────────────────────────────────┐
    │           propose_hypothesis()  [Claude API]   │
    │           apply_patch() → saves original_content│
    ▼                                                 │
  run_node               submit_experiment()          │
    │                    wait_for_experiment()         │
    │                                                 │
    ├─[early_stop_edge: catastrophic failure]──────   │
    │   ▼                                             │
    │   revert_and_continue_node                      │
    │   revert_patch() → increment iter ─────────────┘
    │
    └─[early_stop_edge: normal]──────────────────────
        ▼
      evaluate_node       run_evals() → compare_scores() → flag_regression()
        │
        ├─[decision_edge: KEEP]──────────────────────
        │   ▼
        │   keep_node     update best_score, best_script
        │   └──────────────────────────────────────────┐
        │                                              │
        └─[decision_edge: REVERT]────────────────────  │
            ▼                                          │
            revert_node   revert_patch()               │
            └──────────────────────────────────────────┤
                                                       ▼
                                                    log_node
                                                    log_iteration()
                                                    adapt_eval_suite()? (every 10)
                                                       │
                                          ┌────────────┘
                                          │
                            ┌─[continue_edge: continue]──► propose_node (loop)
                            └─[continue_edge: stop]──────► END
                                                           returns best TrainedModel
    `,
    functions: [
      {
        name: "build_autoresearch_graph",
        signature: "build_autoresearch_graph() -> CompiledStateGraph[AutoResearchState]",
        description: "Constructs and compiles the AutoResearch LangGraph StateGraph with all nodes, conditional edges, and LangGraph checkpointer for resumability. Called once at startup.",
        params: [],
        returns: { type: "CompiledStateGraph[AutoResearchState]", description: "Compiled cyclic graph: init → baseline → [propose → run → evaluate → decide → log] × N → END." },
        notes: "Pass a LangGraph checkpointer (e.g. SqliteSaver) to graph.compile() so the loop can be resumed after crashes.",
      },
      {
        name: "invoke_autoresearch_graph",
        signature: "invoke_autoresearch_graph(plan: TrainingPlan, config: OrchestrationConfig, cost_manager: CostManager) -> TrainedModel",
        description: "Entry point called by the Manager's orchestrate_node. Invokes the compiled AutoResearch graph and returns the best TrainedModel from final state.",
        params: [
          { name: "plan", type: "TrainingPlan", description: "Output of run_decision_engine." },
          { name: "config", type: "OrchestrationConfig", description: "Orchestration config (budget, task type, eval metric)." },
          { name: "cost_manager", type: "CostManager", description: "Running CostManager instance for budget enforcement." },
        ],
        returns: { type: "TrainedModel", description: "Best model: weights_path, metrics, cost, n_iterations, research_diary_path." },
      },
      {
        name: "init_node",
        signature: "init_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls create_eval_suite() and sets up the initial state for the loop (eval_suite, current_script from plan, iteration=0).",
        params: [{ name: "state", type: "AutoResearchState", description: "Initial state with plan and config populated." }],
        returns: { type: "dict", description: "Partial state update: { eval_suite, current_script, current_config, iteration: 0 }." },
      },
      {
        name: "baseline_node",
        signature: "baseline_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Submits and runs the unmodified baseline training script, evaluates it, and sets baseline_score and best_score in state.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after init_node." }],
        returns: { type: "dict", description: "Partial state update: { baseline_score, best_score }." },
      },
      {
        name: "propose_node",
        signature: "propose_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls propose_hypothesis() with the research diary and current config. Applies the returned patch to the script and saves original_content for revert.",
        params: [{ name: "state", type: "AutoResearchState", description: "Current loop state." }],
        returns: { type: "dict", description: "Partial state update: { current_script (patched), original_content, last_hypothesis }." },
      },
      {
        name: "run_node",
        signature: "run_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls submit_experiment and wait_for_experiment with the patched script. Writes ExperimentResult to state. early_stop_edge reads from state after this node.",
        params: [{ name: "state", type: "AutoResearchState", description: "Current loop state (current_script must be patched)." }],
        returns: { type: "dict", description: "Partial state update: { last_result: ExperimentResult }." },
      },
      {
        name: "early_stop_edge",
        signature: "early_stop_edge(state: AutoResearchState) -> Literal['evaluate', 'revert_and_continue']",
        description: "LangGraph conditional edge after run_node. Calls check_early_stop() on state.last_result.metrics. Returns 'revert_and_continue' on catastrophic failure, 'evaluate' otherwise.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after run_node." }],
        returns: { type: "Literal['evaluate', 'revert_and_continue']", description: "Next node to execute." },
      },
      {
        name: "revert_and_continue_node",
        signature: "revert_and_continue_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls revert_patch() with state.original_content, logs the early-stopped iteration as REVERTED, and increments the iteration counter. Loops back to propose_node.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after run_node (early stop detected)." }],
        returns: { type: "dict", description: "Partial state update: { current_script (reverted), diary, iteration }." },
      },
      {
        name: "evaluate_node",
        signature: "evaluate_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls run_evals, compare_scores, and flag_regression. Writes last_score and last_delta to state. decision_edge reads from state after this node.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after run_node (normal completion)." }],
        returns: { type: "dict", description: "Partial state update: { last_score: EvalScore, last_delta: ScoreDelta }." },
      },
      {
        name: "decision_edge",
        signature: "decision_edge(state: AutoResearchState) -> Literal['keep', 'revert']",
        description: "LangGraph conditional edge after evaluate_node. Calls decide_keep_or_revert(state.last_delta). Returns 'keep' or 'revert'.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after evaluate_node." }],
        returns: { type: "Literal['keep', 'revert']", description: "Next node to execute." },
      },
      {
        name: "keep_node",
        signature: "keep_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Updates best_score and best_script to the current iteration's values. Resets no_improve_streak.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after evaluate_node (decision: keep)." }],
        returns: { type: "dict", description: "Partial state update: { best_score, best_script, no_improve_streak: 0 }." },
      },
      {
        name: "revert_node",
        signature: "revert_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls revert_patch() to restore original_content, increments no_improve_streak.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after evaluate_node (decision: revert)." }],
        returns: { type: "dict", description: "Partial state update: { current_script (reverted), no_improve_streak }." },
      },
      {
        name: "log_node",
        signature: "log_node(state: AutoResearchState) -> dict",
        description: "LangGraph node. Calls log_iteration() to append the IterationRecord to the diary. Calls adapt_eval_suite() every 10 iterations. Increments iteration counter. continue_edge reads from state after this node.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after keep_node or revert_node." }],
        returns: { type: "dict", description: "Partial state update: { diary, eval_suite (possibly updated), iteration }." },
      },
      {
        name: "continue_edge",
        signature: "continue_edge(state: AutoResearchState) -> Literal['propose', '__end__']",
        description: "LangGraph conditional edge after log_node. Returns '__end__' if budget is exhausted, no_improve_streak >= N, or target metric is reached. Otherwise returns 'propose' to loop.",
        params: [{ name: "state", type: "AutoResearchState", description: "State after log_node." }],
        returns: { type: "Literal['propose', '__end__']", description: "'propose' to continue the loop, '__end__' to exit." },
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
    name: "ManagerState",
    description: "LangGraph TypedDict for the Manager StateGraph. Passed between all Manager nodes.",
    fields: [
      { name: "prompt", type: "str", description: "User's plain-English task description." },
      { name: "budget", type: "float", description: "Hard budget cap in USD." },
      { name: "data_path", type: "str | None", description: "Path to user-provided data, or None." },
      { name: "has_data", type: "bool", description: "Set by query_data_node." },
      { name: "task_reasoning", type: "TaskReasoning", description: "Set by reason_node after Claude API call." },
      { name: "config", type: "OrchestrationConfig", description: "Set by build_config_node. Passed to all downstream agents." },
      { name: "result", type: "TrainedModel | None", description: "Set by orchestrate_node. Final output." },
    ],
  },
  {
    name: "DataGenState",
    description: "LangGraph TypedDict for the Data Generator StateGraph. Passed between all DataGen nodes.",
    fields: [
      { name: "config", type: "OrchestrationConfig", description: "Orchestration config from Manager." },
      { name: "data_path", type: "str | None", description: "User-provided data path, or None." },
      { name: "mode", type: "str | None", description: "Set by route_node: 'A' | 'B' | 'C'." },
      { name: "raw_data", type: "RawData | None", description: "Set by acquisition nodes after loading/fetching/generating data." },
      { name: "hf_candidates", type: "list[HFCandidate]", description: "Set by acquire_hf_data_node." },
      { name: "selected_candidate", type: "HFCandidate | None", description: "Optional top candidate for Mode B." },
      { name: "schema", type: "DataSchema | None", description: "Reserved for Sub-agent 2 curation outputs." },
      { name: "dataset", type: "StandardDataset | None", description: "Reserved for Sub-agent 2 curation outputs." },
      { name: "validation_report", type: "ValidationReport | None", description: "Reserved for Sub-agent 2 validation output." },
      { name: "handoff", type: "dict | None", description: "Final Sub-agent 1 handoff payload for Sub-agent 2." },
    ],
  },
  {
    name: "AutoResearchState",
    description: "LangGraph TypedDict for the AutoResearch cyclic StateGraph. Persisted across iterations via LangGraph checkpointer.",
    fields: [
      { name: "plan", type: "TrainingPlan", description: "Training plan from Decision Engine." },
      { name: "config", type: "OrchestrationConfig", description: "Orchestration config." },
      { name: "eval_suite", type: "EvalSuite", description: "Set by init_node. Reused every iteration." },
      { name: "current_script", type: "str", description: "Path to the current (possibly patched) train.py." },
      { name: "current_config", type: "dict", description: "Current hyperparameter config dict." },
      { name: "original_content", type: "str | None", description: "Pre-patch file content saved by propose_node for revert." },
      { name: "diary", type: "ResearchDiary", description: "Append-only list of IterationRecords. Updated by log_node." },
      { name: "baseline_score", type: "EvalScore", description: "Score of unmodified baseline. Set by baseline_node." },
      { name: "best_score", type: "EvalScore", description: "Best score seen so far. Updated by keep_node." },
      { name: "best_script", type: "str", description: "Path to the script that achieved best_score." },
      { name: "last_result", type: "ExperimentResult | None", description: "Set by run_node after each experiment." },
      { name: "last_score", type: "EvalScore | None", description: "Set by evaluate_node." },
      { name: "last_delta", type: "ScoreDelta | None", description: "Set by evaluate_node." },
      { name: "iteration", type: "int", description: "Current iteration counter. Incremented by log_node." },
      { name: "no_improve_streak", type: "int", description: "Consecutive iterations without improvement. Reset by keep_node." },
      { name: "should_stop", type: "bool", description: "Set to True by continue_edge when stop conditions are met." },
    ],
  },
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
