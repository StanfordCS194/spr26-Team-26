"""
CLI entry point for the PROPOSE phase of the AutoResearch Loop.

── Algorithmic strategies (no API cost) ─────────────────────────────────────
    python -m src.autoresearch --strategy random --n-iters 5
    python -m src.autoresearch --strategy local  --n-iters 8 --seed 42

── Real Claude Haiku proposals (costs ~$0.001 per iteration) ─────────────────
    ANTHROPIC_API_KEY=sk-... python -m src.autoresearch --strategy claude --n-iters 3

── Watch logs live in a second terminal ──────────────────────────────────────
    tail -f outputs/logs/run.jsonl | python -m json.tool
    tail -f outputs/logs/research_diary.jsonl | python -m json.tool

── Inspect the diary after a run ─────────────────────────────────────────────
    python -m src.autoresearch --show-diary
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.autoresearch.config import TrainingConfig
from src.autoresearch.loop import AutoResearchLoop
from src.autoresearch.proposer import (
    LocalPerturbationProposalStrategy,
    Proposal,
    ProposalStrategy,
    RandomSearchProposalStrategy,
)
from src.types import IterationRecord

_DEFAULT_CONFIG = Path("configs/current.json")
_DEFAULT_DIARY  = Path("outputs/logs/research_diary.jsonl")
_DEFAULT_LOG    = Path("outputs/logs/run.jsonl")

# ANSI
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"


# ─── Claude strategy wrapper ──────────────────────────────────────────────────

class ClaudeProposalStrategy(ProposalStrategy):
    """
    Calls propose_hypothesis() (Claude Haiku) to generate each proposal.
    Requires ANTHROPIC_API_KEY to be set in the environment.
    """

    def __init__(self, task_type: str = "text-classification", eval_metric: str = "f1") -> None:
        self._task = {
            "task_type": task_type,
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": eval_metric,
            "complexity": "medium",
        }

    def propose(self, config: TrainingConfig, history: list[IterationRecord]) -> Proposal:
        from src.autoresearch.autoresearch import propose_hypothesis
        hypothesis = propose_hypothesis(config.to_dict(), history, self._task)
        patch_dict = json.loads(hypothesis["patch"])
        return Proposal(
            hypothesis=hypothesis["description"],
            patch=patch_dict,
            metadata={
                "strategy": "claude",
                "expected_effect": hypothesis["expected_effect"],
                "search_strategy": hypothesis["search_strategy"],
            },
        )


# ─── Pretty output helpers ────────────────────────────────────────────────────

def _print_header(strategy: str, n_iters: int, config_path: Path) -> None:
    print(f"\n{_BOLD}{'═' * 68}{_RESET}")
    print(f"{_BOLD}  AutoResearch PROPOSE Loop — Live Demo{_RESET}")
    print(f"{'═' * 68}")
    print(f"  strategy : {_CYAN}{strategy}{_RESET}")
    print(f"  iters    : {_CYAN}{n_iters}{_RESET}")
    print(f"  config   : {_DIM}{config_path}{_RESET}")
    print(f"  diary    : {_DIM}{_DEFAULT_DIARY}{_RESET}")
    print(f"  log      : {_DIM}{_DEFAULT_LOG}{_RESET}")
    print(f"\n  {_DIM}Tip: run in a second terminal to watch live:{_RESET}")
    print(f"  {_DIM}  tail -f {_DEFAULT_LOG} | python -m json.tool{_RESET}")
    print(f"{'─' * 68}\n")


def _print_diary_summary(diary_path: Path) -> None:
    if not diary_path.exists():
        return
    lines = [l for l in diary_path.read_text().splitlines() if l.strip()]
    if not lines:
        return

    records: list[IterationRecord] = [json.loads(l) for l in lines]

    print(f"\n{'═' * 68}")
    print(f"{_BOLD}  RESEARCH DIARY  —  {len(records)} iteration(s){_RESET}")
    print(f"{'═' * 68}\n")

    for r in records:
        decision = r.get("decision", "PENDING")
        if decision == "KEPT":
            badge = f"{_GREEN}✓ KEPT    {_RESET}"
        elif decision == "REVERTED":
            badge = f"{_YELLOW}✗ REVERTED{_RESET}"
        else:
            badge = f"{_BLUE}⏳ PENDING {_RESET}"

        hypothesis = r.get("hypothesis", "")
        print(f"  [{badge}] Iter {r['iteration']:>2}  {hypothesis[:62]}")

        for line in r.get("patch", "").splitlines():
            if line.startswith("+"):
                print(f"             {_GREEN}{line}{_RESET}")
            elif line.startswith("-"):
                print(f"             {_RED}{line}{_RESET}")

        cost = r.get("cost_usd", 0.0)
        notes = r.get("notes", "")
        if cost or notes:
            print(f"             {_DIM}cost=${cost:.4f}  {notes}{_RESET}")
        print()

    pending = sum(1 for r in records if r.get("decision") == "PENDING")
    kept    = sum(1 for r in records if r.get("decision") == "KEPT")
    reverted = sum(1 for r in records if r.get("decision") == "REVERTED")
    print(f"  {_DIM}Total: {kept} KEPT  {reverted} REVERTED  {pending} PENDING{_RESET}")
    print(f"  {_DIM}Diary written to: {_DEFAULT_DIARY}{_RESET}\n")


def _show_diary(diary_path: Path) -> None:
    """--show-diary mode: pretty-print the existing diary and exit."""
    if not diary_path.exists():
        print(f"{_RED}No diary found at {diary_path}{_RESET}")
        sys.exit(1)
    _print_diary_summary(diary_path)


# ─── Argument parser ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.autoresearch",
        description="Run N PROPOSE iterations of the AutoResearch Loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--strategy",
        choices=["random", "local", "claude"],
        default="random",
        help=(
            "'random' — sample full range (no API); "
            "'local' — ±20%% perturbation (no API); "
            "'claude' — real Claude Haiku proposals (requires ANTHROPIC_API_KEY)."
        ),
    )
    p.add_argument(
        "--n-iters",
        type=int,
        default=1,
        metavar="N",
        help="Number of PROPOSE iterations to run (default: 1).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        metavar="PATH",
        help=f"Training config JSON (default: {_DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--diary",
        type=Path,
        default=_DEFAULT_DIARY,
        metavar="PATH",
        help=f"Research diary JSONL (default: {_DEFAULT_DIARY}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducible proposals (ignored with --strategy claude).",
    )
    p.add_argument(
        "--perturbation",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help="Perturbation factor for local strategy (default: 0.2 = ±20%%).",
    )
    p.add_argument(
        "--task-type",
        default="text-classification",
        metavar="STR",
        help="Task type passed to Claude (default: text-classification).",
    )
    p.add_argument(
        "--show-diary",
        action="store_true",
        help="Pretty-print the existing research diary and exit.",
    )
    return p


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    if args.show_diary:
        _show_diary(args.diary)
        return

    if args.strategy == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print(f"{_RED}Error: ANTHROPIC_API_KEY is not set.{_RESET}")
            print(f"Export it and re-run:")
            print(f"  export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        proposer: ProposalStrategy = ClaudeProposalStrategy(task_type=args.task_type)
    elif args.strategy == "local":
        proposer = LocalPerturbationProposalStrategy(
            perturbation_factor=args.perturbation,
            seed=args.seed,
        )
    else:
        proposer = RandomSearchProposalStrategy(seed=args.seed)

    _print_header(args.strategy, args.n_iters, args.config)

    loop = AutoResearchLoop(
        proposer=proposer,
        diary_path=args.diary,
        current_config_path=args.config,
    )

    for i in range(args.n_iters):
        print(f"{_BOLD}{'─' * 68}{_RESET}")
        print(f"{_BOLD}  ITERATION {i + 1} / {args.n_iters}{_RESET}")
        print(f"{'─' * 68}")
        loop.run_iteration()

    _print_diary_summary(args.diary)


if __name__ == "__main__":
    main()
