"""Engine loop: happy path, validate→repair retries, guardrail refusal,
fence stripping, and cross-retry usage accounting — all with fake
providers/adapters (no network, no transpiler)."""

from __future__ import annotations

from typing import Any

from arango_query_core.nl import (
    GroundedEntity,
    GroundedPredicate,
    GuardrailVerdict,
    LabelIndex,
    NLQueryEngine,
    PredicateIndex,
    ValidationResult,
)
from arango_query_core.nl.fewshot import FewShotIndex, _NoopRetriever


class FakeProvider:
    """Scripted provider: returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        self.calls.append((system, user))
        content = self._responses.pop(0)
        return content, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cached_tokens": 2,
        }


class FakeAdapter:
    """Minimal QueryLanguageAdapter: valid iff the query contains 'SELECT'."""

    language = "sparql"

    def __init__(self, *, allow: bool = True) -> None:
        self._allow = allow
        self.repair_calls: list[str] = []

    def grammar_prompt_section(self, schema_context: str) -> str:
        return f"## Grammar\nWrite SPARQL only.\n{schema_context}".strip()

    def few_shot_index(self) -> FewShotIndex | None:
        return FewShotIndex(_NoopRetriever(), examples=[])

    def grounding_index(self):
        return None

    def predicate_index(self):
        return None

    def validate(self, query: str) -> ValidationResult:
        if "SELECT" in query:
            return ValidationResult(ok=True)
        return ValidationResult(ok=False, error="not a SELECT", code="E_PARSE")

    def repair_hint(self, query: str, failure: ValidationResult) -> str:
        self.repair_calls.append(query)
        return "Emit a SELECT query."

    def guardrails(self, query: str, context: dict[str, Any]) -> GuardrailVerdict:
        if self._allow:
            return GuardrailVerdict(allowed=True)
        return GuardrailVerdict(allowed=False, reasons=["tenant scope violation"])


def test_happy_path_first_attempt() -> None:
    engine = NLQueryEngine(provider=FakeProvider(["SELECT ?s WHERE { ?s ?p ?o }"]), adapter=FakeAdapter())
    result = engine.generate("show everything")
    assert result.ok and result.query == "SELECT ?s WHERE { ?s ?p ?o }"
    assert result.retries == 0
    assert result.total_tokens == 15


def test_repair_loop_recovers_and_accumulates_usage() -> None:
    provider = FakeProvider(["MATCH (n) RETURN n", "SELECT ?s WHERE { ?s ?p ?o }"])
    adapter = FakeAdapter()
    engine = NLQueryEngine(provider=provider, adapter=adapter, max_retries=2)
    result = engine.generate("show everything")
    assert result.ok and result.retries == 1
    # Usage summed across both attempts.
    assert result.total_tokens == 30 and result.cached_tokens == 4
    # The retry prompt carried the adapter's corrective hint.
    assert adapter.repair_calls == ["MATCH (n) RETURN n"]
    assert "Emit a SELECT query." in provider.calls[1][1]


def test_retries_exhausted_reports_last_failure() -> None:
    provider = FakeProvider(["bad1", "bad2", "bad3"])
    engine = NLQueryEngine(provider=provider, adapter=FakeAdapter(), max_retries=2)
    result = engine.generate("show everything")
    assert not result.ok and result.query == ""
    assert result.error == "not a SELECT"
    assert result.retries == 2


def test_guardrail_refusal_is_surfaced_not_silent() -> None:
    engine = NLQueryEngine(
        provider=FakeProvider(["SELECT ?s WHERE { ?s ?p ?o }"]),
        adapter=FakeAdapter(allow=False),
    )
    result = engine.generate("show everything")
    assert not result.ok
    assert "tenant scope violation" in result.error
    assert result.guardrail is not None and not result.guardrail.allowed


def test_fenced_response_is_stripped() -> None:
    fenced = "```sparql\nSELECT ?s WHERE { ?s ?p ?o }\n```"
    engine = NLQueryEngine(provider=FakeProvider([fenced]), adapter=FakeAdapter())
    result = engine.generate("show everything")
    assert result.ok and result.query == "SELECT ?s WHERE { ?s ?p ?o }"


def test_system_prompt_contains_grammar_section() -> None:
    provider = FakeProvider(["SELECT ?s WHERE { ?s ?p ?o }"])
    engine = NLQueryEngine(provider=provider, adapter=FakeAdapter())
    engine.generate("show everything", schema_context="Classes: Person")
    system = provider.calls[0][0]
    assert "## Grammar" in system and "Classes: Person" in system


class GroundedFakeAdapter(FakeAdapter):
    """FakeAdapter + a populated grounding index (seam 6)."""

    _SENTINEL_LABEL = "Sentinel Widget XYZ123"

    def grounding_index(self) -> LabelIndex | None:
        return LabelIndex.from_items(
            [GroundedEntity(id="http://ex.org/w1", labels=(self._SENTINEL_LABEL,), type="Widget")]
        )

    def grounding_prompt_section(self, question: str, index: LabelIndex, k: int = 20) -> str:
        return index.format_prompt_section(
            question,
            k=k,
            header="## Known entities",
            instruction="Use the exact IDs below.",
            id_prefix="<",
            id_suffix=">",
        )


def test_engine_composes_grounding_block() -> None:
    provider = FakeProvider(["SELECT ?s WHERE { ?s ?p ?o }"])
    adapter = GroundedFakeAdapter()
    engine = NLQueryEngine(provider=provider, adapter=adapter, grounding_k=20)
    engine.generate(f"find the {GroundedFakeAdapter._SENTINEL_LABEL}")
    system = provider.calls[0][0]
    assert "## Known entities" in system
    assert GroundedFakeAdapter._SENTINEL_LABEL in system
    # Grounding lands AFTER the grammar/few-shot sections.
    assert system.index("## Grammar") < system.index("## Known entities")


class PredicateGroundedFakeAdapter(GroundedFakeAdapter):
    """GroundedFakeAdapter + a populated predicate index (seam 7)."""

    _SENTINEL_PREDICATE_LABEL = "pv:sentinelPredicateXYZ123"

    def predicate_index(self) -> PredicateIndex | None:
        return PredicateIndex.from_items(
            [
                GroundedPredicate(
                    iri="http://ex.org/pv/sentinelPredicateXYZ123",
                    label=self._SENTINEL_PREDICATE_LABEL,
                    kind="datatype",
                    domain="Widget",
                    range="xsd:string",
                    shape="literal",
                )
            ]
        )

    def predicate_prompt_section(self, question: str, index: PredicateIndex, k: int = 20) -> str:
        return index.format_prompt_section(
            question,
            k=k,
            header="## Known schema predicates",
            instruction="Use only these predicates.",
        )


def test_engine_composes_predicate_block_after_entities() -> None:
    provider = FakeProvider(["SELECT ?s WHERE { ?s ?p ?o }"])
    adapter = PredicateGroundedFakeAdapter()
    engine = NLQueryEngine(provider=provider, adapter=adapter, grounding_k=20, predicate_k=20)
    engine.generate(
        f"find the {GroundedFakeAdapter._SENTINEL_LABEL} "
        f"{PredicateGroundedFakeAdapter._SENTINEL_PREDICATE_LABEL}"
    )
    system = provider.calls[0][0]
    assert "## Known schema predicates" in system
    assert PredicateGroundedFakeAdapter._SENTINEL_PREDICATE_LABEL in system
    # Ordering: grammar -> few-shot -> entities -> predicates, all post-cache-boundary.
    assert system.index("## Grammar") < system.index("## Known entities")
    assert system.index("## Known entities") < system.index("## Known schema predicates")

    # D-07 cache-boundary: the predicate block must never leak into the
    # cacheable grammar_prompt_section (static prefix), same rule seam 6
    # already obeys.
    standalone_prompt = adapter.grammar_prompt_section("")
    assert PredicateGroundedFakeAdapter._SENTINEL_PREDICATE_LABEL not in standalone_prompt
    assert "## Known schema predicates" not in standalone_prompt
