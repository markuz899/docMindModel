"""Refusal phrasings observed in real teacher output.

This file is a ledger, not a brainstorm. Every string here was produced by an
actual teacher run over real documentation and was, at the time, misread by the
detector -- which silently discarded a correct example. Add to it only from
observed output.
"""

import pytest

from src.text import leads_with_refusal, looks_like_refusal, unsupported_identifiers

OBSERVED_REFUSALS = [
    "Questi blocchi non permettono di stabilirlo. È documentato che il seriale compare"
    " in `Cerca dispositivi` quando le credenziali cloud sono configurate.",
    "Questi blocchi non permettono di determinare la causa del tuo errore: servono"
    " l'output del comando e il backup coinvolto.",
    "Non è determinabile dai blocchi: mancano i log della richiesta.",
    "Il funzionamento effettivo del citofono con credenziali valide non è determinabile"
    " da questi blocchi. Servirebbero risultati di test su hardware reale.",
    "The provided documentation does not contain this information.",
    "This cannot be determined from the supplied sections.",
]

NOT_REFUSALS = [
    "La rotta risponde HTTP 200 con il payload CurrentUser [a.md — GET /me].",
    "`UserService` compone account e profilo [a.md — UserService].",
    "The gateway applies rate limits per caller and rejects the request with 429.",
]


@pytest.mark.parametrize("text", OBSERVED_REFUSALS)
def test_observed_refusals_are_detected(text):
    assert looks_like_refusal(text)


@pytest.mark.parametrize("text", NOT_REFUSALS)
def test_substantive_answers_are_not_refusals(text):
    assert not looks_like_refusal(text)


def test_a_refusal_that_opens_with_context_still_leads_with_the_decline():
    text = OBSERVED_REFUSALS[0]
    assert leads_with_refusal(text)


def test_a_caveat_at_the_end_does_not_make_an_answer_a_refusal():
    text = ("`UserService` compone account e profilo [a.md — UserService]. "
            "Non documentato: le fonti non citano la cache.")
    assert looks_like_refusal(text), "the caveat is still a gap statement"
    assert not leads_with_refusal(text), "but the answer does not open with it"


def test_identifiers_introduced_by_the_question_are_not_hallucinations():
    """Observed: a bug report naming NOT_ON had its answer rejected for using it."""
    question = "mandando NOT_ON il relè va HIGH, dal codice si capisce perché?"
    context = 'onMessage uses msg.indexOf("ON") >= 0 to decide.'
    answer = "`NOT_ON` contiene `ON`, quindi `indexOf` la trova."
    assert unsupported_identifiers(answer, context, question) == set()
    # without the question it would still look invented
    assert "NOT_ON" in unsupported_identifiers(answer, context)
