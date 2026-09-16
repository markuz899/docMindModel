import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.schema import ContextChunk, Example  # noqa: E402


@pytest.fixture
def chunks() -> list[ContextChunk]:
    return [
        ContextChunk(
            source="04-api-reference.md",
            heading="GET /me",
            content="Returns the authenticated user. Handled by UserController.getMe, "
                    "which calls UserService.getCurrentUser. displayName comes from the profile.",
        ),
        ContextChunk(
            source="05-main-flows.md",
            heading="Current User Flow",
            content="authMiddleware verifies the token, then UserService merges the account "
                    "record with the profile record into CurrentUser.",
        ),
    ]


@pytest.fixture
def good_example(chunks) -> Example:
    return Example(
        id="t-001",
        question="Come funziona GET /me?",
        context=chunks,
        answer=(
            "La rotta è gestita da `UserController.getMe`, che chiama "
            "`UserService.getCurrentUser` [04-api-reference.md — GET /me]. "
            "Il token è verificato da `authMiddleware` [05-main-flows.md — Current User Flow]."
        ),
        category="route",
        difficulty="easy",
        answerable="full",
        project="demo",
        relevant_sources=["04-api-reference.md#GET /me", "05-main-flows.md#Current User Flow"],
        must_include=["UserService"],
    )
