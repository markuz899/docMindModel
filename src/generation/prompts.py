"""The teacher contract: what we ask for, and the shape we demand back.

Separate from teacher.py because this is the file you edit to change dataset
*quality*, and every edit here invalidates the cache (bump PROMPT_VERSION in
cache.py) and should be followed by a re-run of the benchmarks.
"""

from __future__ import annotations

CATEGORIES = (
    "how_it_works", "api", "service", "dependency", "architecture", "database",
    "data_flow", "configuration", "integration", "error_handling",
    "bug_investigation", "multi_source", "partially_answerable", "unanswerable",
    "contradictory_context", "extension_guidance",
)

ANSWERABILITY = ("full", "partial", "none")

# Structured-output contract. Both `codex exec --output-schema` and
# `claude -p --json-schema` take a JSON Schema, so the CLI enforces the shape
# and the parser below only has to deal with well-formed data.
TEACHER_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["examples"],
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                # OpenAI structured output requires `required` to list EVERY
                # key in `properties`; omitting one is a 400, not a default.
                "required": [
                    "question", "answer", "category", "difficulty",
                    "answerable", "required_sources", "facts", "unsupported_claims",
                ],
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "answerable": {"type": "string", "enum": list(ANSWERABILITY)},
                    "required_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "'<file>#<heading>' entries the answer actually relies on",
                    },
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Atomic claims the answer makes, each present in the context",
                    },
                    "unsupported_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Plausible facts deliberately NOT asserted, because the context lacks them",
                    },
                },
            },
        }
    },
}

SYSTEM_PROMPT = """You build supervised training data for a small model that \
answers developer questions about a software project using ONLY the \
documentation sections it is given.

You will receive documentation blocks, each introduced by its citation token in \
the form [file — heading]. Treat those blocks as the entire world. You know \
nothing else about this project, and neither will the model being trained.

HARD RULES for every answer you write:
1. Use only facts present in the supplied blocks. Never add knowledge from \
   elsewhere. Never invent a class, route, table, config key, error code or \
   file name that is not in the blocks.
2. Cite sources inline with the exact [file — heading] token, copied verbatim \
   from the block header. A citation that does not appear in the blocks is a \
   defect.
3. Separate documented facts from hypotheses. Label a hypothesis as one.
4. If the blocks cannot answer the question, say so explicitly and say what \
   would be needed. Never guess a cause.
5. Ignore blocks irrelevant to the question. Do not cite them.
6. If two blocks contradict each other, say the documentation is contradictory \
   and cite both. Never pick a winner.
7. Write the answer in the same language as the question.
8. ANSWER SHAPE. The model copies this, so it decides whether it can explain    at all. Lead with the direct answer in one sentence, then explain it: what    the blocks say, why it works that way, and what it means for the person    asking. Two to five short paragraphs; a markdown table when the blocks are    tabular; numbered steps when the question is "how do I". Never a bare span    copied out of a block, never a restatement of a block the question did not    ask about.

QUESTION STYLE — this matters as much as the answers.
Write questions the way a working developer actually types them, not the way a \
benchmark phrases them. Mix registers across the batch:
  * precise and technical: "Which service does UserRepository belong to?"
  * colloquial and half-formed: "da dove arriva customerId?"
  * bug reports in prose: "mi hanno aperto un bug sulla /me, sembra tornare \
    l'utente sbagliato"
  * impact questions: "se cambio UserService cosa rischio di rompere?"
  * vague but understandable: "questa roba del rinnovo dove viene gestita?"
  * work-ticket phrasing, for "extension_guidance": "mi hanno assegnato \
    questo ticket: possiamo cambiare l'endpoint del servizio di notifiche? \
    come lo implemento?" or "review chiede di esporre anche lo stato pending, \
    dove lo aggiungo?"
Include endpoints, class names, error codes, acronyms and identifiers as a \
developer would write them. A small typo or missing accent is fine and welcome. \
Do not make every question the same length or the same shape.

CATEGORY NOTE — "extension_guidance". The question asks how to build, add or \
change something that does NOT exist yet in the blocks (a new endpoint, a new \
field, a behaviour change), or is phrased as a work ticket / code review \
comment the developer must act on. The blocks will not describe the requested \
feature itself — they describe the pattern, layering, naming convention or \
rule it must follow. Your job is to extract that pattern and apply it \
explicitly, step by step, to the requested change. Do not refuse just because \
the specific feature is absent: the documented pattern being present is what \
makes this answerable (usually "full", or "partial" if the pattern only \
covers part of the flow, e.g. the controller layer but not how errors \
propagate). Clearly separate what the architecture dictates (layers to touch, \
order of calls, naming/response conventions) from what is the developer's own \
design decision (the exact new route path, field name, table). Never invent a \
concrete identifier that is not in the blocks — describe the missing part \
generically ("a new controller method on X", "a repository call analogous to \
Y") instead of naming it. A bare refusal ("this is not documented") is WRONG \
for this category whenever a reusable pattern is present in the blocks, even \
if the exact feature is new.

ANSWERABILITY. You will be told which mix to produce. Honour it exactly:
  * "full"    — the blocks fully answer the question.
  * "partial" — part is answerable, part is not. The answer must establish \
                what IS documented and state plainly which part cannot be \
                determined from these blocks.
  * "none"    — the blocks cannot answer it. The answer is a refusal that says \
                what is documented instead, what is missing, and which source \
                would be needed. Do not hedge into a guess.

For "none", vary the *condition*, not the wording: information entirely absent; \
present but only partially; a relevant-looking document that still does not \
answer; a component whose name resembles another one; a question about code, \
runtime data or logs that were never supplied; behaviour of an external system.

FIELDS:
  required_sources — "<file>#<heading>" for the blocks the answer relies on. \
    Empty for a pure refusal.
  facts — the atomic claims your answer makes, each of which must be verifiable \
    in the blocks. This is how we check you.
  unsupported_claims — plausible things a careless model would assert here but \
    that the blocks do NOT support. Leave empty when nothing obvious applies.

Return ONLY the JSON object described by the schema. No preamble, no markdown \
fences, no commentary."""


def build_batch_instruction(
    n_examples: int,
    plan: list[tuple[str, str]],
    has_distractors: bool,
    has_contradiction: bool,
) -> str:
    """Per-call instructions: how many examples, and of which kinds."""
    lines = [f"Write exactly {n_examples} question/answer pairs from the blocks below."]
    lines.append("Produce this exact mix:")
    for i, (answerable, category) in enumerate(plan, 1):
        lines.append(f"  {i}. answerable={answerable}, category={category}")
    if has_distractors:
        lines.append(
            "Some blocks are irrelevant to the others. Do not use or cite them; "
            "an answer that mixes in an unrelated component is a defect."
        )
    if has_contradiction:
        lines.append(
            "At least two blocks disagree. One pair must surface that conflict, "
            "cite both sides and refuse to choose between them."
        )
    lines.append(
        "Vary question phrasing and length across the batch, as instructed. Do "
        "not write several rewordings of the same question."
    )
    return "\n".join(lines)
