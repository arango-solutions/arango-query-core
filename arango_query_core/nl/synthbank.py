"""Query-first synthetic few-shot **template catalog** + slot-filling core.

This is the pure, dependency-free construction core promoted out of
``arango-sparql-py``'s eval-side ``bank_generator`` (Phase 07.5, Stage 2):
a typed catalog of ontology-agnostic compositional query shapes
(:data:`SHAPE_CATALOG`), each pairing an ``applies`` gate (over a
:class:`~arango_query_core.nl.grounding.GroundedPredicate` + its home
:class:`~arango_query_core.nl.grounding.PredicateIndex` + data-driven
signals) with a ``build_sparql`` slot-filling renderer (a pre-bound
``binding`` dict -> gold SPARQL text).

**Boundary (Phase 07.5 OQ-2, ``promote-template-core-only``).** Only the
*pure* surface lives here: the ``ShapeTemplate`` type, the shape gates,
and the SPARQL slot-filling renderers. Everything that needs an RDF store
-- data-binding (sampling real fillers), execution-non-empty filtering,
the strict-extremum probe -- stays caller-side (in ``arango-sparql-py``'s
``tests/nl2sparql/eval/bank_generator.py``), because it depends on
``pyoxigraph`` (a test-only dependency that MUST NOT enter this engine
package). The onboarding caller supplies each shape a pre-bound
``binding`` dict and calls ``shape.build_sparql(binding)``.

This module is intentionally ``pyoxigraph``-free and stdlib-only
(mirroring the rest of ``arango_query_core.nl``), so it imports cleanly
in any transpiler. The ``applies`` gates read ONLY ``GroundedPredicate``
fields + a duck-typed ``signals`` mapping (``.orderable`` /
``.optional_relation`` booleans), never a hardcoded vocabulary term --
so the same catalog runs unmodified across ontologies (CK25, QALD, ...).

**Cypher inheritance.** The ``ShapeTemplate`` type and the ``applies``
gates are target-language-agnostic (they read only the shared
``GroundedPredicate`` shape), so a Cypher front-end reuses them directly
and supplies its own ``build_*`` renderers; the ``build_sparql`` closures
here are the SPARQL-specific half.

Name-anchoring discipline (carried from the eval-side generator): every
entity slot is resolved via the well-known, dataset-independent
``rdfs:label`` (:data:`RDFS_LABEL_IRI`), never a per-schema term -- so
every emitted triple references only the ontology's OWN declared
vocabulary IRIs plus ``rdfs:label``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .grounding import GroundedPredicate

# Well-known, dataset-INDEPENDENT RDF vocabulary term -- never a per-schema
# hint. ``rdfs:label`` covers a strict superset of any per-schema name
# predicate, so this ALONE is a sufficient, schema-agnostic name-anchor
# for every shape below.
RDFS_LABEL_IRI = "http://www.w3.org/2000/01/rdf-schema#label"

# The two ``GroundedPredicate.shape`` values whose RANGE side is a real,
# enumerable, label-anchorable "entity" class (as opposed to ``"literal"``
# -- a plain datatype value -- or ``"value_object"`` -- an intermediate
# node reached only via an extra hop). Mechanical, derived from the walker's
# own classification -- never a term-name special case.
RELATIONAL_SHAPES = frozenset({"category_instance", "linked_entity"})


@dataclass(frozen=True)
class ShapeTemplate:
    """One compositional query shape in the generator's typed catalog.

    ``applies`` gates whether a given ``GroundedPredicate`` (+ its home
    ``PredicateIndex`` + a duck-typed ``signals`` mapping) is a candidate
    for this shape. ``build_sparql`` renders the slot-filled, name-anchored
    SPARQL text for a matched, pre-bound ``binding``. Both callables read
    ONLY ``GroundedPredicate`` fields (``kind``/``domain``/``range``/
    ``shape``/``shape_detail``), the data-driven ``signals``
    (``.orderable``/``.optional_relation``), and pre-bound ``binding``
    values -- NEVER a hardcoded vocabulary term, so the same catalog runs
    unmodified across ontologies.

    ``semantic_slots`` are the shape's first-class, machine-readable
    faithfulness ground truth: the named fillers a paraphrase must
    preserve. ``intent_lexicon`` is the shape's own paraphrase-guard
    vocabulary (e.g. ``top_n`` -> a superlative token; ``negation`` ->
    "without"/"no"/"lacking"). Both are consumed by the caller-side
    ``slot_preserving`` faithfulness guard.
    """

    name: str
    applies: Callable[..., bool]
    build_sparql: Callable[..., str]
    question_template: str
    semantic_slots: tuple[str, ...]
    intent_lexicon: tuple[str, ...]


# Ordered catalog registry -- order is the generator's deterministic
# per-shape generation order.
SHAPE_CATALOG: list[ShapeTemplate] = []


def _register(template: ShapeTemplate) -> ShapeTemplate:
    """Append *template* to the ordered ``SHAPE_CATALOG`` and return it."""
    SHAPE_CATALOG.append(template)
    return template


# --------------------------------------------------------------------------
# Small, dependency-free render/index helpers (no pyoxigraph).
# --------------------------------------------------------------------------


def _lit(value: str) -> str:
    """Render *value* as an escaped SPARQL string literal (name-anchor)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _predicates_of(index: Any) -> list[GroundedPredicate]:
    """The flat ``list[GroundedPredicate]`` backing *index* (``PredicateIndex``
    exposes no public iterator; the same private-field access the eval-side
    predicate tests already established)."""
    return list(getattr(index, "_predicates", []))


def _sorted_predicates(index: Any) -> list[GroundedPredicate]:
    """Deterministic (IRI-sorted) predicate list -- generation order must
    not depend on the store's own (unordered) query-result iteration."""
    return sorted(_predicates_of(index), key=lambda p: p.iri)


# --------------------------------------------------------------------------
# Shape 1: lookup -- ``?x rdfs:label "V" . ?x P ?result`` (a direct literal
# hop off a name-anchored subject). Schema-agnostic gate: any DatatypeProperty
# with a declared domain (an undeclared domain cannot be anchored, so it is
# correctly excluded).
# --------------------------------------------------------------------------


def _applies_lookup(pred: Any, index: Any, signals: Mapping[str, Any]) -> bool:
    return pred.kind == "datatype" and bool(pred.domain)


def _build_lookup_sparql(binding: dict[str, Any]) -> str:
    # Deliberately NO ``?x a <domain_iri>`` type constraint: several
    # TBox-declared domains are abstract superclasses with ZERO direct
    # instances -- only their concrete subclasses are ever actually typed.
    # The predicate + its label anchor is a sufficient, precise real-world
    # constraint.
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?x <{RDFS_LABEL_IRI}> {_lit(binding['filler_label'])} .\n"
        f"  ?x <{binding['predicate_iri']}> ?result .\n"
        "}"
    )


# --------------------------------------------------------------------------
# Shape 2: value_object -- ``?x rdfs:label "V" . ?x P ?mid . ?mid Q ?result``
# (the "extra hop" pattern: P's range class C has ONLY datatype-property
# children -- the ``"value_object"`` classification -- Q is one of them).
# --------------------------------------------------------------------------


def _applies_value_object(pred: Any, index: Any, signals: Mapping[str, Any]) -> bool:
    return pred.shape == "value_object"


def _build_value_object_sparql(binding: dict[str, Any]) -> str:
    # Same no-domain-type-constraint discipline as ``_build_lookup_sparql``.
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?x <{RDFS_LABEL_IRI}> {_lit(binding['filler_label'])} .\n"
        f"  ?x <{binding['predicate_iri']}> ?mid .\n"
        f"  ?mid <{binding['hop_predicate_iri']}> ?result .\n"
        "}"
    )


# --------------------------------------------------------------------------
# Shape 3: category_filter -- ``?c rdfs:label "V" . ?result P ?c`` (anchor
# the RANGE side by label; project the DOMAIN-side members).
# --------------------------------------------------------------------------


def _applies_relational(pred: Any, index: Any, signals: Mapping[str, Any]) -> bool:
    return pred.shape in RELATIONAL_SHAPES


def _build_category_filter_sparql(binding: dict[str, Any]) -> str:
    # RANGE side keeps its type constraint (disambiguates the label match to
    # the correct class -- the anchor). DOMAIN side deliberately has NO type
    # constraint (same abstract-domain rationale as lookup/value_object).
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?c a <{binding['range_iri']}> .\n"
        f"  ?c <{RDFS_LABEL_IRI}> {_lit(binding['filler_label'])} .\n"
        f"  ?result <{binding['predicate_iri']}> ?c .\n"
        "}"
    )


# --------------------------------------------------------------------------
# Shape 4: scalar_count -- same predicate pool as category_filter, COUNT
# aggregate instead of a listing.
# --------------------------------------------------------------------------


def _build_scalar_count_sparql(binding: dict[str, Any]) -> str:
    # Same RANGE-anchor-only discipline as category_filter.
    return (
        "SELECT (COUNT(DISTINCT ?member) AS ?result) WHERE {\n"
        f"  ?c a <{binding['range_iri']}> .\n"
        f"  ?c <{RDFS_LABEL_IRI}> {_lit(binding['filler_label'])} .\n"
        f"  ?member <{binding['predicate_iri']}> ?c .\n"
        "}"
    )


# --------------------------------------------------------------------------
# Shape 5: grouped_aggregation -- ``SELECT ?result WHERE {...} GROUP BY
# ?result HAVING (COUNT(?x) > K)``. DISTINCT from scalar_count. K is
# data-bound by the caller (never a hardcoded constant).
# --------------------------------------------------------------------------


def _build_grouped_aggregation_sparql(binding: dict[str, Any]) -> str:
    # No domain type constraint: the predicate alone determines real group
    # membership.
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?x <{binding['predicate_iri']}> ?result .\n"
        "}\n"
        "GROUP BY ?result\n"
        f"HAVING (COUNT(?x) > {binding['threshold']})"
    )


# --------------------------------------------------------------------------
# Shapes 6/7: top_n / offset -- ``ORDER BY DESC(?v) LIMIT 1 [OFFSET 1]``.
# Requires the data-driven ``orderable`` signal (a datatype predicate whose
# range is an ordered XSD type). No name-anchor slot -- the domain class
# alone provides the "member_type" context (ranks across ALL its instances).
# The caller's generation-time strict-extremum probe drops ties.
# --------------------------------------------------------------------------


def _applies_orderable(pred: Any, index: Any, signals: Mapping[str, Any]) -> bool:
    sig = signals.get(pred.iri)
    return pred.kind == "datatype" and bool(pred.domain) and bool(sig and sig.orderable)


def _build_top_n_sparql(binding: dict[str, Any]) -> str:
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?result a <{binding['domain_iri']}> .\n"
        f"  ?result <{binding['predicate_iri']}> ?v .\n"
        "}\n"
        "ORDER BY DESC(?v)\nLIMIT 1"
    )


def _build_offset_sparql(binding: dict[str, Any]) -> str:
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?result a <{binding['domain_iri']}> .\n"
        f"  ?result <{binding['predicate_iri']}> ?v .\n"
        "}\n"
        "ORDER BY DESC(?v)\nLIMIT 1 OFFSET 1"
    )


# --------------------------------------------------------------------------
# Shape 8: negation -- ``?result a C . FILTER NOT EXISTS { ?result P ?v }``.
# Requires the data-driven ``optional_relation`` signal (both P-present and
# P-absent C-instances exist) -- guarantees non-empty. Unavailable on
# TBox-only ontologies -- degrades to False, never crashes.
# --------------------------------------------------------------------------


def _applies_negation(pred: Any, index: Any, signals: Mapping[str, Any]) -> bool:
    sig = signals.get(pred.iri)
    return pred.kind == "object" and bool(pred.domain) and bool(sig and sig.optional_relation)


def _build_negation_sparql(binding: dict[str, Any]) -> str:
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?result a <{binding['domain_iri']}> .\n"
        f"  FILTER NOT EXISTS {{ ?result <{binding['predicate_iri']}> ?v }}\n"
        "}"
    )


# --------------------------------------------------------------------------
# Shape 9: two_hop -- ``?c rdfs:label "V" . ?x P ?c . ?x Q ?result`` (the
# richest bucket: category_filter's range-anchor direction, PLUS a second
# forward hop off the SAME domain instance via a sibling predicate Q sharing
# P's domain). ``applies`` may consult *index* for a same-domain relational
# sibling -- still schema-agnostic (structural lookup, no term-name special
# casing).
# --------------------------------------------------------------------------


def _applies_two_hop(pred: Any, index: Any, signals: Mapping[str, Any]) -> bool:
    if pred.shape not in RELATIONAL_SHAPES:
        return False
    return any(
        other.iri != pred.iri and other.domain == pred.domain and other.shape in RELATIONAL_SHAPES
        for other in _predicates_of(index)
    )


def _build_two_hop_sparql(binding: dict[str, Any]) -> str:
    return (
        "SELECT DISTINCT ?result WHERE {\n"
        f"  ?c a <{binding['range_iri']}> .\n"
        f"  ?c <{RDFS_LABEL_IRI}> {_lit(binding['filler_label'])} .\n"
        f"  ?x <{binding['predicate_iri']}> ?c .\n"
        f"  ?x <{binding['hop_predicate_iri']}> ?result .\n"
        "}"
    )


# The 9 shapes. ``grouped_aggregation`` is a DISTINCT shape from
# ``scalar_count`` (treating them as one re-introduces a known regression:
# scalar-COUNT examples distract a HAVING case).

_register(
    ShapeTemplate(
        name="lookup",
        applies=_applies_lookup,
        build_sparql=_build_lookup_sparql,
        question_template="What is the {predicate} of {entity}?",
        semantic_slots=("entity", "predicate"),
        intent_lexicon=(),
    )
)

_register(
    ShapeTemplate(
        name="value_object",
        applies=_applies_value_object,
        build_sparql=_build_value_object_sparql,
        question_template="What is the {predicate} {hop_predicate} of {entity}?",
        semantic_slots=("entity", "predicate", "hop_predicate"),
        intent_lexicon=(),
    )
)

_register(
    ShapeTemplate(
        name="category_filter",
        applies=_applies_relational,
        build_sparql=_build_category_filter_sparql,
        question_template="Which {member_type} are in the {category} category?",
        semantic_slots=("category", "member_type"),
        intent_lexicon=(),
    )
)

_register(
    ShapeTemplate(
        name="scalar_count",
        applies=_applies_relational,
        build_sparql=_build_scalar_count_sparql,
        question_template="How many {member_type} are there for {category}?",
        semantic_slots=("category", "member_type"),
        intent_lexicon=("how many", "number of", "count"),
    )
)

_register(
    ShapeTemplate(
        name="grouped_aggregation",
        applies=_applies_relational,
        build_sparql=_build_grouped_aggregation_sparql,
        question_template="Which {group_type} have more than {threshold} {member_type}?",
        semantic_slots=("group_type", "member_type", "threshold"),
        intent_lexicon=(
            "more than",
            "at least",
            "per",
            "over",
            "greater than",
            "greater than or equal",
            "exceeding",
            "at minimum",
        ),
    )
)

_register(
    ShapeTemplate(
        name="top_n",
        applies=_applies_orderable,
        build_sparql=_build_top_n_sparql,
        question_template="Which {member_type} has the {superlative} {order_predicate}?",
        semantic_slots=("member_type", "order_predicate", "superlative", "direction"),
        intent_lexicon=("most", "least", "highest", "lowest", "largest", "smallest"),
    )
)

_register(
    ShapeTemplate(
        name="offset",
        applies=_applies_orderable,
        build_sparql=_build_offset_sparql,
        question_template="Which {member_type} has the {ordinal}-{superlative} {order_predicate}?",
        semantic_slots=("member_type", "order_predicate", "superlative", "ordinal", "direction"),
        intent_lexicon=("second", "third", "next", "after the"),
    )
)

_register(
    ShapeTemplate(
        name="negation",
        applies=_applies_negation,
        build_sparql=_build_negation_sparql,
        question_template="Which {member_type} do not have a {predicate}?",
        semantic_slots=("member_type", "predicate"),
        intent_lexicon=(
            "without",
            "no",
            "lacking",
            "don't have",
            "missing",
            "lack",
            "lacks",
            "do not have",
            "have no",
            "not have",
        ),
    )
)

_register(
    ShapeTemplate(
        name="two_hop",
        applies=_applies_two_hop,
        build_sparql=_build_two_hop_sparql,
        question_template=(
            "What is the {far_predicate} of the {member_type} whose {near_predicate} is {entity}?"
        ),
        semantic_slots=("entity", "near_predicate", "far_predicate", "far_type", "member_type"),
        intent_lexicon=(),
    )
)

# Name -> ShapeTemplate, built once after every shape is registered above.
_SHAPES_BY_NAME: dict[str, ShapeTemplate] = {t.name: t for t in SHAPE_CATALOG}


__all__ = [
    "RDFS_LABEL_IRI",
    "RELATIONAL_SHAPES",
    "SHAPE_CATALOG",
    "ShapeTemplate",
]
