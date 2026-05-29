# LocalSparse

A small (~1B) dense LLM with DeepSeek-V4-style 3-branch sparse attention and a disk-backed, soft-mounted workspace context system. Knowledge lives in agent-curated workspaces, not weights.

See `~/.copilot/session-state/<id>/plan.md` for the full architecture spec (v0.1).

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[training,dev]"
pytest                                # run local tests
localsparse --help                    # CLI
```

## Status

**Implementation: complete v0 scaffold ready for Colab M1 training.**

| Layer | Files | Tests | Status |
|---|---|---|---|
| Storage | slab · registry · access_log | 14 | ✅ |
| Workspace | manager · eviction · consolidation | 13 | ✅ |
| Attention | sparse_three_branch · indexer · yarn · kv_quant | 11 | ✅ |
| Tools | parser · registry · workspace_tools · web_tools | 9 | ✅ |
| Chat | template (MiniCPM5 + `<workspace_context>`) | 5 | ✅ |
| Agent | agent loop · MockBackend | 4 | ✅ |
| Model | surgery (drop-in LlamaAttention → ThreeBranchAttention) · HFBackend | 3 | ✅ |
| Training | losses · data · milestone1 trainer | 9 | ✅ |
| Eval | RULER · mount_vs_flat · routing | (within training tests) | ✅ |
| Integration | end-to-end surgery + train + agent + consolidation | 2 | ✅ |
| **Total** | | **70 tests** | **all green on CPU** |

`scripts/run_m1_surgery.py` smoke-tested with `--base toy`.
`notebooks/colab_m1.ipynb` ready: clone → install → test → download MiniCPM5 → baseline PPL → surgery → 200-step heal → post PPL gate (≤1.3×) → save.

## Layout

```
localsparse/
  storage/       slab.py · registry.py · access_log.py
  workspace/     manager.py · eviction.py · consolidation.py
  attention/     sparse_three_branch.py · indexer.py · yarn.py · kv_quant.py
  model/         surgery.py · localsparse_model.py
  tools/         parser.py · registry.py · workspace_tools.py · web_tools.py
  chat/          template.py
  agent/         agent.py
  training/      losses.py · data.py · milestone1.py
  eval/          ruler.py · niah.py · mount_vs_flat.py · routing.py
  cli.py
tests/           CPU-runnable unit + integration tests
scripts/         download/surgery/train helpers
notebooks/       Colab-ready training notebooks (M1…M9)
```

## Training milestones

| # | Goal | Budget | Pass criterion |
|---|---|---|---|
| 1 | Attention surgery sanity | $5 | PPL/HumanEval/MMLU within 5% of base |
| 2 | All 3 branches alive | $10 | per-branch attention mass > 5% |
| 3 | Compressed branch contributes | $15 | long-doc PPL improves |
| 4 | Native long-ctx 128K | $30 | RULER@128K ≥ 85% |
| 5 | YaRN ctx extension to 256K | $30 | RULER@256K ≥ 70% |
| 6 | Workspace mount = flat ctx | $40 | mount-vs-flat ≥ 70% |
| 7 | Eviction preserves recall | $30 | gist-recall ≥ 60% post-evict |
| 8 | Cross-wks routing | $50 | routing acc ≥ 75% @ N=8 |
| 9 | Agentic SFT + headline eval | $90–115 | hit ≥3 of 5 M9 targets |
| 10 (opt) | mHC retrofit | $50–150 | uplift vs M9 |

Total v1: **~$340–365** on Colab A100 ($0.50/hr Turkey pricing).

## Status

This is the v0.1 implementation scaffold. Storage / workspace / attention / tools / agent modules are implemented and locally tested. Training scripts are Colab-ready but unrun.
