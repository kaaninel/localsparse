"""Unit tests for the factoid synthetic world."""
import torch
import pytest

from localsparse.training.factoid_world import (
    build_world, render_corpus, build_qa_pairs, partition_facts, make_lm_batches,
)


def test_world_shape():
    w = build_world(vocab_size=4096, n_facts=200, seed=1)
    assert w.n_facts == 200
    assert len(set(w.facts)) == 200  # unique triples
    assert len(w.subjects) == 16 and len(w.predicates) == 16 and len(w.objects) == 32


def test_corpus_contains_all_facts():
    w = build_world(vocab_size=4096, n_facts=20, seed=2)
    stream = render_corpus(w, repeats_per_fact=3, seed=2)
    text = set(stream)
    for (s, p, o) in w.facts:
        assert s in text and p in text and o in text


def test_qa_pair_answer_is_object():
    w = build_world(vocab_size=4096, n_facts=10, seed=3)
    pairs = build_qa_pairs(w)
    assert len(pairs) == 10
    for (prompt_ids, ans), (s, p, o) in zip(pairs, w.facts):
        assert ans == o
        assert s in prompt_ids and p in prompt_ids


def test_partition_disjoint():
    w = build_world(vocab_size=4096, n_facts=80, seed=4)
    parts = partition_facts(w, n_partitions=4, seed=4)
    assert len(parts) == 4
    sets = [set(p.facts) for p in parts]
    for i in range(4):
        for j in range(i + 1, 4):
            assert sets[i].isdisjoint(sets[j])


def test_make_lm_batches_shapes():
    w = build_world(vocab_size=4096, n_facts=50, seed=5)
    stream = render_corpus(w, repeats_per_fact=5, seed=5)
    batches = make_lm_batches(stream, batch_size=2, seq_len=64,
                              device=torch.device("cpu"))
    assert len(batches) > 0
    inp, lbl = batches[0]
    assert inp.shape == (2, 64)
    assert (inp == lbl).all()


def test_alphabet_fits_in_vocab():
    with pytest.raises(ValueError):
        # n_facts > capacity should raise
        build_world(vocab_size=4096, n_facts=99_999, seed=6)
