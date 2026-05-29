"""Synthetic factoid world for M0.5 knowledge-displacement experiments.

Allocates a small alphabet from the model's existing vocab to build a
controlled (subject, predicate, object) world. This lets us measure
exactly how many facts can be learned in-weights vs how many can be
mounted via a workspace, on a model with essentially no priors.

Design (plan §6.6):
  - 16 subject tokens × 16 predicate tokens × 32 object tokens
    → 8192 possible triples
  - sample N triples without replacement
  - render each as 5 templated sentences (vary phrasing)
  - question form: "What is the {predicate} of {subject}?"
                   → 1-token answer (object)

The alphabet uses contiguous token ids near the end of vocab so we
don't collide with special tokens at the start.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import torch


@dataclass
class FactoidWorld:
    """One sampled world of facts + a renderer/QA generator."""
    subjects: List[int]
    predicates: List[int]
    objects: List[int]
    facts: List[Tuple[int, int, int]]  # (s, p, o)
    template_tokens: Dict[str, List[int]] = field(default_factory=dict)
    eos_id: int = 0

    @property
    def n_facts(self) -> int:
        return len(self.facts)

    def fact_lookup(self) -> Dict[Tuple[int, int], int]:
        return {(s, p): o for (s, p, o) in self.facts}


def build_world(*, vocab_size: int, n_facts: int,
                n_subjects: int = 16, n_predicates: int = 16, n_objects: int = 32,
                template_offset: int = 200, alphabet_offset: int | None = None,
                eos_id: int = 0, seed: int = 0) -> FactoidWorld:
    """Pick contiguous token blocks for the s/p/o alphabet + templating glue.

    The alphabet sits in the high end of the vocab so it doesn't collide
    with special tokens at low ids. Template glue ("is the", "of", "?")
    just borrows a handful of arbitrary mid-vocab tokens; the model treats
    them as opaque markers — this is intentional (it's a controlled world).
    """
    rng = random.Random(seed)
    total_alphabet = n_subjects + n_predicates + n_objects
    if alphabet_offset is None:
        alphabet_offset = vocab_size - total_alphabet - 1
    subjects = list(range(alphabet_offset, alphabet_offset + n_subjects))
    predicates = list(range(alphabet_offset + n_subjects,
                            alphabet_offset + n_subjects + n_predicates))
    objects = list(range(alphabet_offset + n_subjects + n_predicates,
                         alphabet_offset + total_alphabet))

    # 5 template "glue tokens" — small fixed ids; we don't care about their meaning.
    template_tokens = {
        "is_the": [template_offset + 0, template_offset + 1],         # "is the"
        "of": [template_offset + 2],                                  # "of"
        "q_what": [template_offset + 3, template_offset + 0],         # "What is"
        "q_mark": [template_offset + 4],                              # "?"
        "sep": [template_offset + 5],                                 # "."
    }

    all_triples = [(s, p, o) for s in subjects for p in predicates for o in objects]
    rng.shuffle(all_triples)
    if n_facts > len(all_triples):
        raise ValueError(f"n_facts={n_facts} > capacity {len(all_triples)}")
    facts = all_triples[:n_facts]
    return FactoidWorld(subjects=subjects, predicates=predicates, objects=objects,
                        facts=facts, template_tokens=template_tokens, eos_id=eos_id)


_RENDERERS = [
    # 5 surface forms per fact; all are 1-token-answer-friendly.
    lambda w, s, p, o: w.template_tokens["is_the"][:1] + [p] + w.template_tokens["of"]
                       + [s] + w.template_tokens["is_the"][1:] + [o] + w.template_tokens["sep"],
    lambda w, s, p, o: [s] + w.template_tokens["is_the"][:1] + [p] + [o] + w.template_tokens["sep"],
    lambda w, s, p, o: [s] + [p] + [o] + w.template_tokens["sep"],
    lambda w, s, p, o: w.template_tokens["q_what"] + [p] + w.template_tokens["of"]
                       + [s] + w.template_tokens["q_mark"] + [o] + w.template_tokens["sep"],
    lambda w, s, p, o: [o] + w.template_tokens["is_the"][:1] + [p] + w.template_tokens["of"]
                       + [s] + w.template_tokens["sep"],
]


def render_corpus(world: FactoidWorld, *, repeats_per_fact: int = 5,
                  seed: int = 0) -> List[int]:
    """Render every fact `repeats_per_fact` times (cycling templates).

    Returns a flat token id stream. We shuffle the per-fact sentence
    order so the model sees facts interleaved (otherwise it could memorize
    by position).
    """
    rng = random.Random(seed)
    sentences: List[List[int]] = []
    for (s, p, o) in world.facts:
        for r in range(repeats_per_fact):
            renderer = _RENDERERS[r % len(_RENDERERS)]
            sentences.append(renderer(world, s, p, o))
    rng.shuffle(sentences)
    out: List[int] = []
    for sent in sentences:
        out.extend(sent)
    return out


def build_qa_pairs(world: FactoidWorld) -> List[Tuple[List[int], int]]:
    """Build one Q/A pair per fact.

    Format: prompt = "What is the {p} of {s} ?"  (token ids)
            answer = object token id (single token)
    """
    out: List[Tuple[List[int], int]] = []
    for (s, p, o) in world.facts:
        prompt = (world.template_tokens["q_what"] + [p]
                  + world.template_tokens["of"] + [s]
                  + world.template_tokens["q_mark"])
        out.append((prompt, o))
    return out


def evaluate_qa(model, qa_pairs, *, device, k_eval: int = 0) -> Dict[str, float]:
    """Compute top-1 accuracy on QA pairs (optionally a held-out k_eval slice).

    Returns dict {accuracy, n_evaluated}. The model is called with the
    prompt; we score on the next-token logits.
    """
    model.eval()
    if k_eval and len(qa_pairs) > k_eval:
        qa_pairs = qa_pairs[:k_eval]
    correct = 0
    total = 0
    with torch.no_grad():
        for prompt_ids, answer_id in qa_pairs:
            x = torch.tensor(prompt_ids, device=device).unsqueeze(0)
            logits = model(input_ids=x).logits
            pred = int(logits[0, -1].argmax())
            correct += (pred == answer_id)
            total += 1
    return {"accuracy": correct / max(total, 1), "n_evaluated": total}


def partition_facts(world: FactoidWorld, n_partitions: int,
                    *, seed: int = 0) -> List[FactoidWorld]:
    """Split facts evenly across N partitions for multi-workspace experiments."""
    rng = random.Random(seed)
    facts = list(world.facts)
    rng.shuffle(facts)
    chunk = len(facts) // n_partitions
    out: List[FactoidWorld] = []
    for i in range(n_partitions):
        sub = facts[i * chunk:(i + 1) * chunk]
        out.append(FactoidWorld(
            subjects=world.subjects, predicates=world.predicates,
            objects=world.objects, facts=sub,
            template_tokens=world.template_tokens, eos_id=world.eos_id,
        ))
    return out


def make_lm_batches(token_stream: List[int], *, batch_size: int, seq_len: int,
                    device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pack a flat token stream into (input_ids, labels) batches."""
    out = []
    # Pad to a multiple of (batch_size * seq_len)
    total = (len(token_stream) // (batch_size * seq_len)) * (batch_size * seq_len)
    if total == 0:
        return out
    ids = torch.tensor(token_stream[:total], device=device, dtype=torch.long)
    ids = ids.view(-1, batch_size, seq_len)  # (N_batches, B, T)
    for b in ids:
        out.append((b, b.clone()))
    return out
