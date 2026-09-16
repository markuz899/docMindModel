"""Group-aware train/validation/test split.

A naive per-example split leaks: the same documentation section, reworded or
re-issued with extra distractors, lands on both sides and the test score
measures memorisation.

Policy (group_by="project"):
  * test  = whole projects, held out entirely. This is the only honest measure
            of "works on documentation it has never seen".
  * train/validation = the remaining projects, split by *family* so that an
            example and its distractor variants never straddle the boundary.
            Validation exists for early stopping, not for generalisation.

The requested ratios are targets, not guarantees: one project is a large,
indivisible block. The achieved ratios are reported by the caller.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from src.config import SplitConfig
from src.dataset.schema import Example
from src.text import jaccard

_VARIANT_SUFFIX = re.compile(r"-(?:noise|var|aug)\d+$")


def group_key(example: Example, mode: str) -> str:
    if mode == "project":
        return example.project or (example.context[0].source if example.context else "unknown")
    if mode == "document":
        return example.context[0].source if example.context else "unknown"
    return example.id


def family_key(example: Example) -> str:
    """Example plus its generated variants -- they must not straddle a split."""
    return _VARIANT_SUFFIX.sub("", example.id)


def _dedupe_exact(examples: list[Example]) -> list[Example]:
    """Same question, same answer *and* same sources -> genuine duplicate.

    Context is part of the key on purpose: distractor variants legitimately
    repeat a question and answer over a noisier context.
    """
    seen, out = set(), []
    for ex in examples:
        key = (
            ex.question.strip().lower(),
            ex.answer.strip().lower(),
            tuple(sorted(f"{s}#{h}" for s, h in ex.context_keys())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def _fingerprint(ex: Example) -> str:
    return f"{ex.question} {ex.answer}"


def drop_near_duplicates(
    candidates: list[Example], reference: list[Example], threshold: float
) -> tuple[list[Example], list[Example]]:
    """Remove candidates too similar to anything in reference."""
    ref_prints = [_fingerprint(r) for r in reference]
    kept, dropped = [], []
    for ex in candidates:
        fp = _fingerprint(ex)
        if any(jaccard(fp, rp) >= threshold for rp in ref_prints):
            dropped.append(ex)
        else:
            kept.append(ex)
    return kept, dropped


def _split_by_family(
    examples: list[Example], validation_share: float, rng: random.Random
) -> tuple[list[Example], list[Example]]:
    families: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        families[family_key(ex)].append(ex)
    keys = sorted(families)
    rng.shuffle(keys)
    target = validation_share * len(examples)
    validation: list[Example] = []
    train: list[Example] = []
    for key in keys:
        if len(validation) < target:
            validation.extend(families[key])
        else:
            train.extend(families[key])
    return train, validation


def split_dataset(
    examples: list[Example], cfg: SplitConfig | None = None, seed: int = 42
) -> dict[str, list[Example]]:
    cfg = cfg or SplitConfig()
    examples = _dedupe_exact(examples)
    rng = random.Random(seed)

    if cfg.group_by == "none":
        train, rest = _split_by_family(examples, cfg.validation + cfg.test, rng)
        test_share = cfg.test / max(cfg.validation + cfg.test, 1e-9)
        validation, test = _split_by_family(rest, test_share, rng)
        buckets = {"train": train, "validation": validation, "test": test}
    else:
        groups: dict[str, list[Example]] = defaultdict(list)
        for ex in examples:
            groups[group_key(ex, cfg.group_by)].append(ex)
        if len(groups) < 2:
            print(
                f"[split] only {len(groups)} group(s) for group_by={cfg.group_by}; "
                "falling back to a family-aware per-example split -- the test "
                "score will NOT measure generalisation to unseen documentation"
            )
            return split_dataset(examples, cfg.model_copy(update={"group_by": "none"}), seed)

        # Smallest groups first: hold out whole groups until the test target is met.
        ordered = sorted(groups, key=lambda k: (len(groups[k]), k))
        test_target = cfg.test * len(examples)
        test, remaining = [], []
        for name in ordered:
            take_for_test = (
                not remaining          # held-out groups are a contiguous prefix
                and len(ordered) > 1   # never hand every group to the test split
                and len(test) < test_target
            )
            (test if take_for_test else remaining).extend(groups[name])
        if not remaining:  # degenerate: everything went to test
            remaining, test = test, []
        pool_share = cfg.validation / max(cfg.train + cfg.validation, 1e-9)
        train, validation = _split_by_family(remaining, pool_share, rng)
        buckets = {"train": train, "validation": validation, "test": test}

    for name in ("validation", "test"):
        buckets[name], dropped = drop_near_duplicates(
            buckets[name], buckets["train"], cfg.near_duplicate_threshold
        )
        if dropped:
            print(f"[split] dropped {len(dropped)} near-duplicate(s) from {name}")

    for items in buckets.values():
        rng.shuffle(items)
    return buckets
