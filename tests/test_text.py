from src.text import (
    extract_identifiers,
    groundedness,
    keyword_recall,
    looks_like_refusal,
    flags_contradiction,
    parse_citations,
    technical_groundedness,
    token_f1,
    unsupported_identifiers,
)

CONTEXT = (
    "GET /me is handled by UserController.getMe which calls UserService.getCurrentUser. "
    "display_name lives on the user_profiles table. DB_STATEMENT_TIMEOUT_MS defaults to 5000."
)


def test_extracts_project_identifiers():
    found = extract_identifiers("UserService reads /me and user_profiles via `ProfileRepository`")
    assert {"UserService", "/me", "user_profiles", "ProfileRepository"} <= found


def test_sentence_initial_words_are_not_identifiers():
    # Regression: a leading capital used to make every Italian sentence start
    # look like a class name.
    assert "Percorso" not in extract_identifiers("Percorso documentato: la rotta risponde.")
    assert technical_groundedness("Percorso documentato per /me.", CONTEXT) == 1.0


def test_unsupported_identifiers_flags_inventions():
    answer = "Causato da PaymentGateway e da un NullPointerException in OrderService."
    assert unsupported_identifiers(answer, CONTEXT) >= {"PaymentGateway", "OrderService"}


def test_technical_groundedness_is_language_independent():
    italian = "La rotta /me passa da UserController.getMe e legge user_profiles."
    assert technical_groundedness(italian, CONTEXT) == 1.0
    assert groundedness(italian, CONTEXT) < 1.0  # prose overlap is lower, by design


def test_citation_parsing_roundtrip():
    assert parse_citations("vedi [04-api-reference.md — GET /me] e basta") == [
        ("04-api-reference.md", "GET /me")
    ]


def test_refusal_detection_both_languages():
    assert looks_like_refusal("La documentazione fornita non contiene queste informazioni.")
    assert looks_like_refusal("The provided documentation does not specify the cause.")
    assert not looks_like_refusal("La rotta risponde HTTP 200 con il payload CurrentUser.")


def test_contradiction_detection():
    assert flags_contradiction("La documentazione contiene informazioni contrastanti.")
    assert flags_contradiction("The documentation is contradictory here.")


def test_scoring_helpers():
    assert token_f1("user service reads profile", "user service reads profile") == 1.0
    assert token_f1("nothing alike", "user service") == 0.0
    assert keyword_recall("uses UserService today", ["userservice"]) == 1.0
    assert keyword_recall("nothing", ["UserService", "displayName"]) == 0.0
