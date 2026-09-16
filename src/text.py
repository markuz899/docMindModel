"""Lexical helpers shared by the quality filters and the evaluators.

Deliberately dependency-free and deterministic: groundedness and hallucination
have to be measurable on a laptop, offline, with no judge model in the loop.
"""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")

# Identifier shapes that carry project-specific facts. Anything matching one of
# these in an answer but absent from the context is a candidate hallucination.
_IDENT_PATTERNS = [
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),          # UserRepository
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b"),     # user_profile, AUTH_TIMEOUT
    re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b"),            # config.timeout, app.yaml
    re.compile(r"(?<![\w/])/[A-Za-z][A-Za-z0-9_\-{}:/]*"),         # /auth/login, /me
    re.compile(r"\b[A-Z]{3,}\b"),                                  # JWT, POSTGRES
]
_BACKTICK_RE = re.compile(r"`([^`\n]{1,80})`")
_CITATION_RE = re.compile(r"\[([^\[\]\n]+?)\s+[—–-]\s+([^\[\]\n]+?)\]")

# Vocabulary every technical answer may use without it being a project fact.
GENERIC_IDENTIFIERS = {
    "http", "https", "api", "apis", "rest", "json", "yaml", "yml", "xml", "sql",
    "get", "post", "put", "patch", "delete", "head", "options", "crud", "url",
    "uri", "id", "ids", "uuid", "db", "sdk", "cli", "ui", "ux", "ok", "todo",
    "fixme", "note", "readme", "md", "env", "ci", "cd", "tls", "ssl", "dns",
    "cpu", "gpu", "ram", "io", "os", "pr", "mr", "utc", "iso", "rfc", "faq",
    "and/or", "e/o", "n/a", "ndr",
}

STOPWORDS = {
    # english
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "be", "been", "was", "were", "it", "its", "this", "that", "these", "those",
    "as", "at", "by", "from", "with", "not", "no", "but", "if", "then", "than",
    "so", "such", "can", "could", "may", "might", "will", "would", "should",
    "does", "do", "did", "has", "have", "had", "there", "which", "what", "when",
    "where", "who", "how", "why", "into", "about", "also", "only", "any", "all",
    "we", "you", "they", "he", "she", "i", "us", "them", "their", "our", "your",
    # italian
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "da", "del",
    "della", "dei", "delle", "dello", "degli", "al", "allo", "alla", "ai",
    "agli", "alle", "dal", "dalla", "dai", "nel", "nella", "nei", "nelle",
    "sul", "sulla", "con", "per", "tra", "fra", "che", "chi", "cui", "non",
    "come", "dove", "quando", "quale", "quali", "questo", "questa", "questi",
    "queste", "quel", "quello", "quella", "essere", "sono", "e", "ed", "o",
    "ma", "se", "anche", "piu", "più", "molto", "puo", "può", "deve", "viene",
    "vengono", "essa", "esso", "si", "ci", "ne", "su", "una", "ha", "hanno",
}

_REFUSAL_MARKERS = (
    # english
    "does not contain", "do not contain", "doesn't contain", "not contain",
    "not enough information", "insufficient information", "no information",
    "does not specify", "doesn't specify", "not specified", "not documented",
    "is not covered", "no mention", "cannot be determined", "cannot determine",
    "can't determine", "does not say", "doesn't say", "not available in the",
    "outside the provided", "not present in the provided",
    # italian
    "non contiene", "non contengono", "non specifica", "non specificano",
    "non indica", "non indicano", "non riporta", "non riportano",
    "non documenta", "non è documentat", "non sono documentat",
    "non sufficient", "non è sufficiente", "non sono sufficienti",
    "informazioni insufficienti", "non permette di determinare",
    "non consente di determinare", "non sono present", "non è present",
    "non viene specificat", "non compare", "non emerge dalla documentazione",
    "nessuna informazione", "non e possibile stabilire", "non è possibile stabilire",
    "non e possibile determinare", "non è possibile determinare",
    "non e possibile rispondere", "non è possibile rispondere",
    "non ho elementi", "non posso rispondere", "non posso dedur",
    "non posso escludere", "non posso dire", "non compare in nessuna",
    "non descrivono", "non dicono nulla", "non c'e nulla", "non c'è nulla",
    "non ci sono", "va verificato", "non dice nulla", "non e trattat",
    "non è trattat", "non e ricavabile", "non è ricavabile", "non risulta document",
    # english
    "is not stated", "are not described", "do not describe", "cannot be answered",
    "no reference to", "would require", "cannot answer", "none of them",
    "is not enough", "are not enough", "not mentioned", "none of the supplied",
    "none of the provided", "is silent on", "are silent on", "will not guess",
    "is not covered by", "not documented in",
)

_CONTRADICTION_MARKERS = (
    "contradict", "conflicting", "in conflict", "inconsistent",
    "contraddi", "contrastant", "in conflitto", "incoerent", "discordant",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return text.replace("—", "-").replace("–", "-").lower()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def extract_identifiers(text: str) -> set[str]:
    """Project-specific looking symbols: classes, routes, tables, env vars."""
    text = text or ""
    found: set[str] = set()
    for pattern in _IDENT_PATTERNS:
        found.update(m.group(0) for m in pattern.finditer(text))
    for m in _BACKTICK_RE.finditer(text):
        token = m.group(1).strip()
        if token and len(token.split()) <= 3:
            found.add(token)
    cleaned = set()
    for token in found:
        token = token.strip(" .,:;()[]{}\"'`").rstrip("/")
        if len(token) < 2 or token.lower() in GENERIC_IDENTIFIERS:
            continue
        if token.lower() in STOPWORDS:
            continue
        cleaned.add(token)
    return cleaned


def unsupported_identifiers(answer: str, context: str) -> set[str]:
    """Identifiers used in the answer that never appear in the context."""
    haystack = normalize(context)
    out = set()
    for ident in extract_identifiers(answer):
        needle = normalize(ident)
        if needle in haystack:
            continue
        # tolerate plural / possessive / path variations
        if needle.rstrip("s") and needle.rstrip("s") in haystack:
            continue
        out.add(ident)
    return out


def _is_technical_token(tok: str) -> bool:
    """Digits, path/underscore punctuation or an internal capital.

    A leading capital does not count: otherwise every sentence-initial Italian
    word would be treated as a project identifier.
    """
    if len(tok) < 2:
        return False
    if any(ch.isdigit() or ch in "_./" for ch in tok):
        return True
    return any(ch.isupper() for ch in tok[1:])


def technical_tokens(text: str) -> set[str]:
    """Tokens that carry project facts: identifiers, numbers, versions, paths.

    Plain prose is excluded on purpose -- an Italian answer over English docs
    shares almost no prose with its sources, but every identifier, status code
    and config key it names must still be in there.
    """
    out = {t for t in extract_identifiers(text)}
    for raw in WORD_RE.findall(text or ""):
        tok = raw.strip("./-")
        if not tok or tok.lower() in GENERIC_IDENTIFIERS or tok.lower() in STOPWORDS:
            continue
        if _is_technical_token(tok):
            out.add(tok)
    return {t for t in out if len(t) >= 2}


def technical_groundedness(answer: str, context: str) -> float:
    """Share of the answer's technical tokens that occur in the context."""
    tokens = technical_tokens(answer)
    if not tokens:
        return 1.0  # a pure-prose answer asserts no project facts
    haystack = normalize(context)
    hits = sum(1 for t in tokens if normalize(t) in haystack or normalize(t).rstrip("s") in haystack)
    return hits / len(tokens)


def groundedness(answer: str, context: str) -> float:
    """Share of the answer's content words that occur in the context."""
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_vocab = set(content_tokens(context))
    hits = sum(1 for t in answer_tokens if t in context_vocab)
    return hits / len(answer_tokens)


def parse_citations(text: str) -> list[tuple[str, str]]:
    """Extract ``[file — heading]`` pairs, order preserved, duplicates kept."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _CITATION_RE.finditer(text or "")]


def looks_like_refusal(text: str) -> bool:
    t = normalize(text)
    return any(marker in t for marker in _REFUSAL_MARKERS)


def leads_with_refusal(text: str, window: int = 160) -> bool:
    """Does the answer *open* by declining?

    A real refusal leads with the gap ("the documentation does not contain..."),
    while a substantive answer that happens to flag a caveat raises it later.
    That position is what separates "I cannot answer" from "here is the answer,
    and note this one unknown" -- and neither the words nor the citations do.
    """
    return looks_like_refusal(normalize(text)[:window])


def flags_contradiction(text: str) -> bool:
    t = normalize(text)
    return any(marker in t for marker in _CONTRADICTION_MARKERS)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_f1(prediction: str, reference: str) -> float:
    """Unigram F1 over content words -- the cheap stand-in for answer overlap."""
    pred, ref = content_tokens(prediction), content_tokens(reference)
    if not pred or not ref:
        return 0.0
    common: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if ref_counts.get(t, 0) > common.get(t, 0):
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(text: str, keywords: list[str]) -> float:
    """Share of required key facts present in the text (substring, normalised)."""
    if not keywords:
        return 1.0
    haystack = normalize(text)
    hits = sum(1 for kw in keywords if normalize(kw) in haystack)
    return hits / len(keywords)
