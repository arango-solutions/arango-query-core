"""Language-agnostic NL→query generation loop.

Composes the three shared pieces — an :class:`~arango_query_core.nl.providers.LLMProvider`,
a :class:`~arango_query_core.nl.fewshot.FewShotIndex`, and a
:class:`~arango_query_core.nl.seams.QueryLanguageAdapter` — into the
generate → validate → repair retry loop both ``nl2cypher`` and
``nl2sparql`` need. The engine owns *flow* and *accounting*; every
judgement about the target language lives behind the adapter seams.

This is deliberately the minimal faithful loop. ``nl2cypher``'s full
pipeline carries extra stages (entity resolution, tenant prompt
sections, Anthropic cache splitting) that migrate here incrementally
as the re-point lands, gated by that repo's eval suite — the engine's
API is designed so those arrive as optional collaborators, not
signature breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .providers import LLMProvider
from .seams import GuardrailVerdict, QueryLanguageAdapter, ValidationResult

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")


@dataclass
class NLResult:
    """Outcome of one NL→query generation.

    ``query`` is empty when generation failed (``ok`` False); ``error``
    then carries the last validation failure or guardrail refusal.
    Token fields accumulate across retries so cost accounting sees the
    whole conversation, not the last attempt.
    """

    query: str = ""
    ok: bool = False
    error: str = ""
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    guardrail: GuardrailVerdict | None = None
    validation: ValidationResult | None = None


@dataclass
class NLQueryEngine:
    """Generate a target-language query from a natural-language question.

    ``few_shot_k`` examples are retrieved per question from the
    adapter's corpus (seam 2) and rendered under ``## Examples`` — the
    section :func:`~arango_query_core.nl.providers.split_system_for_anthropic_cache`
    treats as the cache boundary, so the static prefix (grammar +
    schema) stays cacheable across questions. ``grounding_k`` entities
    are likewise retrieved per question from the adapter's grounding
    index (seam 6) and rendered after the few-shot section — also
    post-cache-boundary, per-question dynamic content. ``predicate_k``
    predicates are retrieved per question from the adapter's predicate
    index (seam 7) and rendered AFTER the entity section — same
    post-cache-boundary, per-question dynamic content discipline.
    """

    provider: LLMProvider
    adapter: QueryLanguageAdapter
    few_shot_k: int = 3
    grounding_k: int = 20
    predicate_k: int = 20
    max_retries: int = 2
    guardrail_context: dict[str, Any] = field(default_factory=dict)

    def generate(self, question: str, *, schema_context: str = "") -> NLResult:
        result = NLResult()
        system = self._system_prompt(question, schema_context)
        user = question
        failure: ValidationResult | None = None
        candidate = ""

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                assert failure is not None
                hint = self.adapter.repair_hint(candidate, failure) or failure.error
                user = (
                    f"{question}\n\n"
                    f"Your previous {self.adapter.language} query was rejected:\n"
                    f"```{self.adapter.language}\n{candidate}\n```\n"
                    f"Problem: {hint}\n"
                    f"Return a corrected query only."
                )
                result.retries = attempt
            content, usage = self.provider.generate(system, user)
            for key in _USAGE_KEYS:
                setattr(result, key, getattr(result, key) + int(usage.get(key, 0) or 0))
            candidate = _strip_code_fence(content, self.adapter.language)
            failure = self.adapter.validate(candidate)
            result.validation = failure
            if failure.ok:
                break
        else:  # pragma: no cover — loop always breaks or exhausts via range
            pass

        if failure is None or not failure.ok:
            result.error = failure.error if failure else "provider returned nothing"
            return result

        verdict = self.adapter.guardrails(candidate, dict(self.guardrail_context))
        result.guardrail = verdict
        if not verdict.allowed:
            result.error = "; ".join(verdict.reasons) or "guardrail refused the query"
            return result

        result.query = candidate
        result.ok = True
        return result

    def _system_prompt(self, question: str, schema_context: str) -> str:
        sections = [self.adapter.grammar_prompt_section(schema_context)]
        index = self.adapter.few_shot_index()
        if index is not None:
            examples = index.format_prompt_section(
                question, k=self.few_shot_k, language=self.adapter.language
            )
            if examples:
                sections.append(examples)
        grounding = self.adapter.grounding_index()
        if grounding is not None:
            block = self.adapter.grounding_prompt_section(question, grounding, k=self.grounding_k)
            if block:
                sections.append(block)
        predicates = self.adapter.predicate_index()
        if predicates is not None:
            pred_block = self.adapter.predicate_prompt_section(
                question, predicates, k=self.predicate_k
            )
            if pred_block:
                sections.append(pred_block)
        return "\n\n".join(s for s in sections if s)


def _strip_code_fence(content: str, language: str) -> str:
    """Extract the query from a fenced code block, tolerating bare text.

    LLMs frequently wrap the answer in ```<lang> fences despite the
    output-format contract; accept fenced, language-tagged-fenced, and
    bare responses uniformly so the validator sees only query text.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline == -1:
        return text.strip("`").strip()
    body = text[first_newline + 1 :]
    closing = body.rfind("```")
    if closing != -1:
        body = body[:closing]
    return body.strip()
