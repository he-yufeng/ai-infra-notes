# AI Infra Notes

English · [中文](README_CN.md)

Source-level notes on inference engines, KV caches, scheduling, communication, model-serving protocols, and agent runtimes.

I follow concrete execution paths: what a design avoids computing, which assumptions make it work, and how to distinguish implementation checks from performance or quality evidence. These are independent engineering notes, not official project documentation.

## DeepInfer / deepseek-recipe

The first three articles are in Chinese, with pinned source links and explicit validation scope:

| Article | Focus |
| --- | --- |
| [01 · Where YoCo actually removes work](notes/deepinfer/01-yoco-bounded-replay.md) | Visibility versus row compaction, placeholder rows, ring writes, full-source data, and context-parallel layout changes |
| [02 · When DSpark chooses its verification budget](notes/deepinfer/02-dspark-verification-budget.md) | Cumulative confidence, graph padding versus useful work, draft-inclusive sample timing, and stale-confidence scheduling |
| [03 · Why recipe keeps token identity](notes/deepinfer/03-recipe-token-identity.md) | Explicit special-token ranges, a runnable tokenizer comparison, token-aware output parsing, and truncated tool calls |
| [Initial reading map](notes/deepinfer/00-reading-map.md) | Entry points and the first round of source checks |

Source snapshots: DeepInfer [`8b161497`](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b) and recipe [`935817c6`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f).

### Run the token-identity example

The example uses a local copy of recipe's V4.1 tokenizer. No model weights or API key are needed.

```bash
python -m pip install 'tokenizers>=0.23,<0.24'
python notes/examples/token_identity.py /path/to/deepseek-recipe/static/tokenizers/v41/tokenizer.json
```

It compares whole-string encoding with encoding after removing the added vocabulary. This is a Python tokenizer experiment, not a run of recipe's Rust implementation.

## Evidence and scope

Production figures are attributed to their publishers. Source derivations, executed local experiments, and unrun upstream tests are kept separate. No DeepInfer GPU/NPU model run, cluster benchmark, or recipe Rust test run is claimed in the current series.

Short source excerpts retain links to their original versions and licenses. If something is wrong, please open an issue with the version, code location, or a reproduction.

More articles and source walkthroughs will be published here, including other AI infrastructure projects. If the notes are useful, a Star is welcome.

Author: [Yufeng He](https://github.com/he-yufeng).
