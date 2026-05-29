"""N-workspace routing eval (M8).

Given N passages each loaded into its own workspace, and a question whose
answer lives in exactly one of them, measure how often the model:
  - routes to the correct workspace (indexer-level accuracy)
  - answers correctly (downstream task accuracy)

Default N values per spec: {2, 4, 8, 16}. Distractor similarity is
controlled by the caller (random vs same-topic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict


@dataclass
class RoutingResult:
    n_workspaces: int
    n_examples: int
    n_correct_routing: int
    n_correct_answer: int

    @property
    def routing_accuracy(self) -> float:
        return self.n_correct_routing / max(1, self.n_examples)

    @property
    def answer_accuracy(self) -> float:
        return self.n_correct_answer / max(1, self.n_examples)


@dataclass
class RoutingItem:
    workspaces: Dict[str, str]      # name -> passage
    target_wks: str
    question: str
    answer: str


def workspace_routing_eval(
    items: List[RoutingItem],
    predict_with_routing: Callable[[RoutingItem], tuple[str, str]],
    *,
    judge: Callable[[str, str], bool] = lambda gold, got: gold.lower().strip() in got.lower(),
) -> RoutingResult:
    """`predict_with_routing(item) → (chosen_wks_name, answer_string)`."""
    n = len(items)
    nr = 0
    na = 0
    for it in items:
        chosen, answer = predict_with_routing(it)
        if chosen == it.target_wks:
            nr += 1
        if judge(it.answer, answer):
            na += 1
    return RoutingResult(
        n_workspaces=len(items[0].workspaces) if items else 0,
        n_examples=n, n_correct_routing=nr, n_correct_answer=na,
    )
