import pytest

from src.config import QualityConfig, SplitConfig, resolve_path
from src.dataset.io import read_jsonl, write_jsonl
from src.dataset.quality import check_example
from src.dataset.schema import ContextChunk, Example
from src.dataset.split import family_key, split_dataset

DATASET_FILES = ["data/train/train.jsonl", "data/validation/validation.jsonl",
                 "data/test/test.jsonl"]
BENCHMARK_FILES = ["benchmarks/hallucination.jsonl", "benchmarks/retrieval_noise.jsonl",
                   "benchmarks/bug_investigation.jsonl"]


# --- schema ----------------------------------------------------------------

def test_derived_metadata(good_example):
    assert good_example.source_count == 2
    assert good_example.contains_code is True
    assert good_example.relevant_keys() == good_example.context_keys()


def test_context_needs_at_least_one_chunk():
    with pytest.raises(Exception):
        Example(id="x", question="q?", context=[], answer="a")


# --- quality gate ----------------------------------------------------------

def test_good_example_passes(good_example):
    assert check_example(good_example).ok


def test_fabricated_citation_is_rejected(good_example):
    bad = good_example.model_copy(
        update={"answer": good_example.answer + " Vedi [99-ghost.md — Nowhere]."}
    )
    issues = check_example(bad).issues
    assert any("fabricated citation" in i for i in issues)


def test_misattributed_citation_is_rejected(good_example):
    bad = good_example.model_copy(
        update={"answer": "Risposta [04-api-reference.md — Wrong Heading]."}
    )
    assert any("heading not in context" in i for i in check_example(bad).issues)


def test_invented_identifier_is_rejected(good_example):
    bad = good_example.model_copy(
        update={"answer": good_example.answer + " Il problema è in `PaymentGateway`."}
    )
    assert any("absent from context" in i for i in check_example(bad).issues)


def test_unanswerable_must_state_the_gap(chunks):
    silent = Example(
        id="n-1", question="Qual è la latenza p99?", context=chunks,
        answer="La latenza p99 è di 120 ms.", answerable="none", project="demo",
    )
    assert any("does not state the gap" in i for i in check_example(silent).issues)


def test_answerable_needs_a_citation(chunks):
    uncited = Example(
        id="u-1", question="Come funziona GET /me?", context=chunks,
        answer="È gestita da UserService e restituisce l'utente autenticato.",
        answerable="full", project="demo",
    )
    assert any("without any citation" in i for i in check_example(uncited).issues)


def test_must_not_include_is_enforced(chunks):
    lying = Example(
        id="f-1", question="Perché 500?", context=chunks,
        answer="La documentazione non contiene la causa, ma è un database timeout.",
        answerable="none", project="demo", must_not_include=["database timeout"],
    )
    assert any("ungrounded fact" in i for i in check_example(lying).issues)


# --- split -----------------------------------------------------------------

def _mk(i: int, project: str, chunks) -> Example:
    # Lexically distinct on purpose: near-identical wording across projects is
    # (correctly) treated as a near duplicate and dropped from the eval splits.
    token = f"{project}{i}"
    return Example(
        id=f"{project}-{i}",
        question=f"Come funziona il componente {token} del servizio {project}?",
        context=chunks,
        answer=(
            f"Il componente {token} appartiene al servizio {project} e segue il "
            f"flusso {token} descritto [04-api-reference.md — GET /me]."
        ),
        answerable="full",
        project=project,
    )


def test_split_holds_out_whole_projects(chunks):
    examples = [_mk(i, p, chunks) for p in ("a", "b", "c", "d") for i in range(10)]
    splits = split_dataset(examples, SplitConfig(), seed=1)
    train_projects = {e.project for e in splits["train"]}
    test_projects = {e.project for e in splits["test"]}
    assert test_projects and not (train_projects & test_projects)


def test_variants_never_straddle_a_split(chunks):
    base = [_mk(i, "solo", chunks) for i in range(20)]
    variants = [e.model_copy(update={"id": f"{e.id}-noise1"}) for e in base]
    splits = split_dataset(base + variants, SplitConfig(group_by="none"), seed=3)
    where = {}
    for name, items in splits.items():
        for ex in items:
            where.setdefault(family_key(ex), set()).add(name)
    assert all(len(v) == 1 for v in where.values())


def test_no_id_overlap_between_splits(chunks):
    examples = [_mk(i, p, chunks) for p in ("a", "b", "c") for i in range(9)]
    splits = split_dataset(examples, SplitConfig(), seed=2)
    ids = [{e.id for e in items} for items in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


# --- the shipped dataset ---------------------------------------------------

@pytest.mark.parametrize("path", DATASET_FILES + BENCHMARK_FILES)
def test_shipped_files_are_valid_and_clean(path):
    if not resolve_path(path).exists():
        pytest.skip(f"{path} not built yet (run scripts/build_dataset.py)")
    examples = read_jsonl(path)
    assert examples, f"{path} is empty"
    cfg = QualityConfig()
    failures = [(e.id, check_example(e, cfg).issues) for e in examples]
    failures = [f for f in failures if f[1]]
    assert not failures, failures[:3]


def test_splits_do_not_share_projects():
    paths = [resolve_path(p) for p in DATASET_FILES]
    if not all(p.exists() for p in paths):
        pytest.skip("dataset not built yet")
    train = {e.project for e in read_jsonl(DATASET_FILES[0])}
    test = {e.project for e in read_jsonl(DATASET_FILES[2])}
    assert not (train & test)


def test_hallucination_benchmark_is_large_enough():
    path = resolve_path(BENCHMARK_FILES[0])
    if not path.exists():
        pytest.skip("benchmarks not built yet")
    items = read_jsonl(path)
    assert len(items) >= 100
    assert all(e.answerable == "none" for e in items)


def test_jsonl_roundtrip(tmp_path, good_example):
    target = tmp_path / "x.jsonl"
    write_jsonl(target, [good_example])
    assert read_jsonl(target)[0].answer == good_example.answer
