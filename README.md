# AI Infra Notes

English · [中文](README_CN.md)

Source-level notes on inference engines, KV caches, scheduling, communication, model-serving protocols, and agent runtimes.

I follow concrete execution paths: what a design avoids computing, which assumptions make it work, and how to distinguish implementation checks from performance or quality evidence. These are independent engineering notes, not official project documentation.

## DeepInfer / deepseek-recipe

Five source walkthroughs in Chinese, with pinned code links and explicit validation scope:

| Article | Focus |
| --- | --- |
| [01 · Where YoCo actually removes work](notes/deepinfer/01-yoco-bounded-replay.md) | Visibility versus row compaction, placeholder rows, ring writes, full-source data, and context-parallel layout changes |
| [02 · When DSpark chooses its verification budget](notes/deepinfer/02-dspark-verification-budget.md) | Cumulative confidence, graph padding versus useful work, draft-inclusive sample timing, and stale-confidence scheduling |
| [03 · Why recipe keeps token identity](notes/deepinfer/03-recipe-token-identity.md) | Explicit special-token ranges, a runnable tokenizer comparison, token-aware output parsing, and truncated tool calls |
| [04 · A finished request is not necessarily a successful one](notes/deepinfer/04-offline-generation-failures.md) | Terminal notifications, batch draining, borrowed output lifetimes and rejected admission |
| [05 · Tool identity survives removal from declarations](notes/deepinfer/05-history-tool-identity.md) | Historical names, namespace aliases, order-independent reservation and native regression evidence |
| [Initial reading map](notes/deepinfer/00-reading-map.md) | Entry points, first-round scope and subsequent verification updates |

Source snapshots: DeepInfer [`8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b) and recipe [`935817c6`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f).

Runnable companion: [Core DeepInfer](https://github.com/he-yufeng/core-deepinfer) follows requests through prefix recovery, verification budgets and KV rollback with a random, untrained NumPy CPU model. It is also [on PyPI](https://pypi.org/project/core-deepinfer/0.1.0/). This reference does not load DeepSeek weights or implement recipe's protocol codec; the notes retain their separate evidence scope.

### Run the token-identity example

The example uses a local copy of recipe's V4.1 tokenizer. No model weights or API key are needed.

```bash
python -m pip install 'tokenizers>=0.23,<0.24'
python notes/examples/token_identity.py /path/to/deepseek-recipe/static/tokenizers/v41/tokenizer.json
```

It compares whole-string encoding with encoding after removing the added vocabulary. This is a Python tokenizer experiment, not a run of recipe's Rust implementation.

## Evidence and scope

Production figures are attributed to their publishers. The first three notes retain their initial reading scope. Notes 04 and 05 separately document the 2026-10-20 entrypoint checks with test doubles and 85 native recipe Rust tests on Rust 1.92. They distinguish original baselines from still-open fix branches; an unexecuted rerun is not counted as new validation. No DeepInfer native-scheduler/GPU/NPU model run or cluster benchmark is claimed.

Short source excerpts retain links to their original versions and licenses. If something is wrong, please open an issue with the version, code location, or a reproduction.

More articles and source walkthroughs will be published here, including other AI infrastructure projects. If the notes are useful, a Star is welcome.

Author: [Yufeng He](https://github.com/he-yufeng).
