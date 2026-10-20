"""Compare added-token matching with ordinary BPE. Requires tokenizers>=0.23."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def compare(path: Path) -> dict:
    definition = json.loads(path.read_text(encoding="utf-8"))
    original = Tokenizer.from_str(json.dumps(definition))
    definition["added_tokens"] = []
    plain = Tokenizer.from_str(json.dumps(definition))
    marker = "<｜User｜>"
    control_id = original.token_to_id(marker)
    if control_id is None:
        raise ValueError("This example needs a tokenizer with the V4.1 User marker")
    text = "file contains " + marker
    whole_ids = original.encode(text, add_special_tokens=False).ids
    ordinary_ids = plain.encode(text, add_special_tokens=False).ids
    assert control_id in whole_ids
    assert control_id not in ordinary_ids
    return {"text": text, "control_id": control_id,
            "whole_text_ids": whole_ids, "plain_text_ids": ordinary_ids,
            "scope": "Tokenizer comparison only; not a recipe Rust or model test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path, help="Local V4.1 tokenizer.json")
    args = parser.parse_args()
    print(json.dumps(compare(args.tokenizer), ensure_ascii=False, indent=2))
