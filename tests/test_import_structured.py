"""The RAG-chat importer: chunks parsed back out, [Dx] rewritten to the real
citation token, patched answers preferred over the originals."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.dataset.quality import check_example
from src.config import load_dataset_config

_spec = importlib.util.spec_from_file_location(
    "import_structured", Path(__file__).resolve().parent.parent / "scripts" / "import_structured.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def record(style: str, context: str, answer: str = "Vedi [Policy v3 — D1 · 2026-09-09]."):
    return {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": f"{context}\n\nDOMANDA: Come funziona?"},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"id": "rag-0001", "lang": "it", "domain": "expense_policy",
                 "context_style": style, "n_chunks": 1, "traits": ["clean"]},
    }


JSON_CTX = "CONTESTO RECUPERATO (RAG):\n" + json.dumps(
    [{"doc_id": "D1", "source": "Policy v3", "date": "2026-09-09", "score": 0.8,
      "content": "Termine: 30 giorni."}]
)
MD_CTX = "--- [D1] Policy v3 · 2026-09-09 · rilevanza 0.80 ---\nTermine: 30 giorni."
XML_CTX = ('<contesto_recuperato>\n<doc id="D1" source="Policy v3" date="2026-09-09" '
           'score="0.8">\nTermine: 30 giorni.\n</doc>\n</contesto_recuperato>')


@pytest.mark.parametrize("style,context", [("json", JSON_CTX), ("md", MD_CTX), ("xml", XML_CTX)])
def test_every_context_style_yields_the_same_chunk(style, context):
    example = mod.convert(record(style, context), {})
    assert example.question == "Come funziona?"
    assert [(c.source, c.heading, c.content) for c in example.context] == [
        ("Policy v3", "D1 · 2026-09-09", "Termine: 30 giorni.")
    ]


def test_citations_are_rewritten_and_resolve_against_the_context():
    example = mod.convert(record("json", JSON_CTX, "Il termine è 30 giorni [D1]."), {})
    assert example.answer == "Il termine è 30 giorni [Policy v3 — D1 · 2026-09-09]."
    assert check_example(example, load_dataset_config("configs/dataset.real.yaml").quality).ok


def test_a_patch_replaces_the_assistant_turn():
    patched = mod.convert(
        record("json", JSON_CTX, "vecchia risposta [D1]"),
        {"rag-0001": {"answer": "nuova risposta [D1]"}},
    )
    assert patched.answer.startswith("nuova risposta")


def test_a_citation_with_no_matching_chunk_is_refused():
    with pytest.raises(ValueError, match="D9"):
        mod.convert(record("json", JSON_CTX, "inventata [D9]"), {})


def test_a_source_containing_the_separator_stays_parseable():
    context = JSON_CTX.replace("Policy v3", "IT note — printers")
    example = mod.convert(record("json", context, "vedi [D1]"), {})
    assert example.context[0].source == "IT note / printers"
    assert check_example(example, load_dataset_config("configs/dataset.real.yaml").quality).ok
