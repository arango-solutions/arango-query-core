"""Language-agnostic NL→query engine (providers, few-shot retrieval, seams).

Extracted from ``arango_cypher.nl2cypher`` so ``nl2cypher``,
``nl2sparql``, and the contextual-data-fabric's NL front-end share one
engine instead of maintaining three forks. Language-specific behavior
enters exclusively through :class:`~arango_query_core.nl.seams.QueryLanguageAdapter`.
"""

from .engine import NLQueryEngine, NLResult
from .fewshot import BM25Retriever, DenseRetriever, FewShotIndex, Retriever, cached_few_shot_index
from .grounding import GroundedEntity, GroundedPredicate, LabelIndex, PredicateIndex
from .providers import (
    AnthropicProvider,
    LLMProvider,
    OpenAIProvider,
    OpenRouterProvider,
    get_llm_provider,
    split_system_for_anthropic_cache,
)
from .seams import GuardrailVerdict, QueryLanguageAdapter, ValidationResult

__all__ = [
    "AnthropicProvider",
    "BM25Retriever",
    "DenseRetriever",
    "FewShotIndex",
    "GroundedEntity",
    "GroundedPredicate",
    "GuardrailVerdict",
    "LabelIndex",
    "LLMProvider",
    "NLQueryEngine",
    "NLResult",
    "OpenAIProvider",
    "OpenRouterProvider",
    "PredicateIndex",
    "QueryLanguageAdapter",
    "Retriever",
    "ValidationResult",
    "cached_few_shot_index",
    "get_llm_provider",
    "split_system_for_anthropic_cache",
]
