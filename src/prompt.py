"""The stable prompt contract shared by dataset, training, inference and eval.

Everything that touches the model formats its input here. If this file and the
training data ever disagree, the fine-tune silently degrades -- so there is
exactly one implementation.
"""

from __future__ import annotations

from typing import Iterable, Mapping

SYSTEM_PROMPT = (
    "You are a technical documentation assistant.\n"
    "Answer using only the provided project documentation.\n"
    "Do not use your pretrained knowledge to invent project-specific facts.\n"
    "Clearly separate documented facts from hypotheses.\n"
    "If the documentation does not contain enough information, say so explicitly.\n"
    "When suggesting investigation steps, explain why they follow from the "
    "supplied documentation.\n"
    "Cite the relevant sources using the [file — heading] form shown in the "
    "documentation blocks.\n"
    "Answer in the language of the question."
)

CITE_SEP = " — "  # em dash, matches the [file — heading] citation format


def citation(source: str, heading: str) -> str:
    """Canonical citation token, e.g. ``[04-api-reference.md — GET /me]``."""
    return f"[{source}{CITE_SEP}{heading}]"


def render_context(chunks: Iterable[Mapping[str, str]]) -> str:
    """Render retrieved documentation chunks as labelled blocks.

    The citation token is printed verbatim above each block so the model can
    copy it instead of reconstructing (and mangling) file names.
    """
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"{citation(chunk['source'], chunk['heading'])}\n{chunk['content'].strip()}"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: Iterable[Mapping[str, str]]) -> str:
    return (
        "## Project documentation\n\n"
        f"{render_context(chunks)}\n\n"
        "## Question\n\n"
        f"{question.strip()}"
    )


def build_messages(
    question: str,
    chunks: Iterable[Mapping[str, str]],
    answer: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages
