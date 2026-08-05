"""Tests for the promoted synthetic-bank template catalog (``nl.synthbank``).

Pure, dependency-free coverage of the catalog surface promoted out of
arango-sparql-py's eval-side ``bank_generator`` (Phase 07.5, Stage 2):
the shape gates and the SPARQL slot-filling renderers. No RDF store, no
pyoxigraph, no network -- exactly the boundary the promotion draws.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from arango_query_core.nl import SHAPE_CATALOG, ShapeTemplate
from arango_query_core.nl.grounding import GroundedPredicate, PredicateIndex
from arango_query_core.nl.synthbank import (
    RDFS_LABEL_IRI,
    _predicates_of,
    _sorted_predicates,
)

PV = "http://example.org/pv#"


@dataclass(frozen=True)
class _Sig:
    """Duck-typed stand-in for the eval-side ``PredicateSignals`` -- the
    ``applies`` gates read only these two booleans."""

    orderable: bool = False
    optional_relation: bool = False


def _pred(
    local: str,
    *,
    kind: str,
    domain: str,
    range_: str,
    shape: str,
) -> GroundedPredicate:
    return GroundedPredicate(
        iri=f"{PV}{local}",
        label=local,
        kind=kind,
        domain=domain,
        range=range_,
        shape=shape,
    )


# --------------------------------------------------------------------------
# Catalog shape / integrity
# --------------------------------------------------------------------------


def test_catalog_has_nine_uniquely_named_shapes() -> None:
    names = [t.name for t in SHAPE_CATALOG]
    assert len(names) == 9
    assert len(set(names)) == 9
    assert set(names) == {
        "lookup",
        "value_object",
        "category_filter",
        "scalar_count",
        "grouped_aggregation",
        "top_n",
        "offset",
        "negation",
        "two_hop",
    }


def test_every_shape_is_a_frozen_shapetemplate_with_callables() -> None:
    for t in SHAPE_CATALOG:
        assert isinstance(t, ShapeTemplate)
        assert callable(t.applies)
        assert callable(t.build_sparql)
        # applies signature: (pred, index, signals)
        assert len(inspect.signature(t.applies).parameters) == 3
        # build_sparql signature: (binding)
        assert len(inspect.signature(t.build_sparql).parameters) == 1


def test_module_is_pyoxigraph_free() -> None:
    import re

    import arango_query_core.nl.synthbank as mod

    src = inspect.getsource(mod)
    # Guard the import path, not prose: the docstring legitimately *names*
    # pyoxigraph to explain why it must stay out.
    assert not re.search(r"^\s*(import pyoxigraph|from pyoxigraph)", src, re.MULTILINE)


# --------------------------------------------------------------------------
# applies gates (over synthetic GroundedPredicates + duck-typed signals)
# --------------------------------------------------------------------------


def _shape(name: str) -> ShapeTemplate:
    return next(t for t in SHAPE_CATALOG if t.name == name)


def test_applies_lookup_needs_datatype_with_domain() -> None:
    applies = _shape("lookup").applies
    with_domain = _pred("price", kind="datatype", domain="Product", range_="decimal", shape="literal")
    no_domain = _pred("name", kind="datatype", domain="", range_="string", shape="literal")
    an_object = _pred("hasCat", kind="object", domain="Product", range_="Category", shape="category_instance")
    idx = PredicateIndex([with_domain, no_domain, an_object])
    assert applies(with_domain, idx, {}) is True
    assert applies(no_domain, idx, {}) is False
    assert applies(an_object, idx, {}) is False


def test_applies_relational_gates_on_shape() -> None:
    applies = _shape("category_filter").applies
    rel = _pred("hasCat", kind="object", domain="Product", range_="Category", shape="category_instance")
    vo = _pred("price", kind="object", domain="Product", range_="Money", shape="value_object")
    idx = PredicateIndex([rel, vo])
    assert applies(rel, idx, {}) is True
    assert applies(vo, idx, {}) is False


def test_applies_orderable_requires_signal() -> None:
    applies = _shape("top_n").applies
    p = _pred("weight", kind="datatype", domain="Product", range_="decimal", shape="literal")
    idx = PredicateIndex([p])
    assert applies(p, idx, {p.iri: _Sig(orderable=True)}) is True
    assert applies(p, idx, {p.iri: _Sig(orderable=False)}) is False
    assert applies(p, idx, {}) is False  # no signal -> False, never a crash


def test_applies_negation_requires_optional_relation_signal() -> None:
    applies = _shape("negation").applies
    p = _pred("hasManager", kind="object", domain="Employee", range_="Person", shape="linked_entity")
    idx = PredicateIndex([p])
    assert applies(p, idx, {p.iri: _Sig(optional_relation=True)}) is True
    assert applies(p, idx, {p.iri: _Sig(optional_relation=False)}) is False
    assert applies(p, idx, {}) is False


def test_applies_two_hop_needs_same_domain_relational_sibling() -> None:
    applies = _shape("two_hop").applies
    a = _pred("hasCat", kind="object", domain="Product", range_="Category", shape="category_instance")
    sibling = _pred("hasBrand", kind="object", domain="Product", range_="Brand", shape="linked_entity")
    lonely = _pred("hasMgr", kind="object", domain="Employee", range_="Person", shape="linked_entity")
    assert applies(a, PredicateIndex([a, sibling]), {}) is True
    assert applies(a, PredicateIndex([a, lonely]), {}) is False  # no same-domain sibling


# --------------------------------------------------------------------------
# build_sparql slot-filling renderers (pre-bound bindings -> SPARQL text)
# --------------------------------------------------------------------------


def test_build_lookup_renders_name_anchored_literal_hop() -> None:
    sparql = _shape("lookup").build_sparql({"predicate_iri": f"{PV}price", "filler_label": 'Widget "X"'})
    assert f"<{RDFS_LABEL_IRI}>" in sparql
    assert f"<{PV}price>" in sparql
    # embedded quote is escaped in the SPARQL string literal (name-anchor)
    assert r'"Widget \"X\""' in sparql
    assert sparql.startswith("SELECT DISTINCT ?result WHERE {")


def test_build_scalar_count_uses_count_aggregate() -> None:
    sparql = _shape("scalar_count").build_sparql(
        {"range_iri": f"{PV}Category", "predicate_iri": f"{PV}hasCat", "filler_label": "Tools"}
    )
    assert "COUNT(DISTINCT ?member)" in sparql
    assert f"?c a <{PV}Category>" in sparql


def test_build_grouped_aggregation_binds_threshold_into_having() -> None:
    sparql = _shape("grouped_aggregation").build_sparql({"predicate_iri": f"{PV}hasCat", "threshold": 5})
    assert "GROUP BY ?result" in sparql
    assert "HAVING (COUNT(?x) > 5)" in sparql


def test_build_top_n_and_offset_order_descending() -> None:
    top = _shape("top_n").build_sparql({"domain_iri": f"{PV}Product", "predicate_iri": f"{PV}weight"})
    off = _shape("offset").build_sparql({"domain_iri": f"{PV}Product", "predicate_iri": f"{PV}weight"})
    assert top.endswith("ORDER BY DESC(?v)\nLIMIT 1")
    assert off.endswith("ORDER BY DESC(?v)\nLIMIT 1 OFFSET 1")


def test_build_negation_uses_filter_not_exists() -> None:
    sparql = _shape("negation").build_sparql(
        {"domain_iri": f"{PV}Employee", "predicate_iri": f"{PV}hasManager"}
    )
    assert "FILTER NOT EXISTS" in sparql
    assert f"?result a <{PV}Employee>" in sparql


def test_build_two_hop_chains_near_and_far_predicates() -> None:
    sparql = _shape("two_hop").build_sparql(
        {
            "range_iri": f"{PV}Category",
            "predicate_iri": f"{PV}hasCat",
            "hop_predicate_iri": f"{PV}hasBrand",
            "filler_label": "Tools",
        }
    )
    assert f"?x <{PV}hasCat> ?c" in sparql
    assert f"?x <{PV}hasBrand> ?result" in sparql


# --------------------------------------------------------------------------
# index helpers
# --------------------------------------------------------------------------


def test_sorted_predicates_is_iri_ordered() -> None:
    b = _pred("bbb", kind="datatype", domain="C", range_="string", shape="literal")
    a = _pred("aaa", kind="datatype", domain="C", range_="string", shape="literal")
    idx = PredicateIndex([b, a])
    assert [p.iri for p in _sorted_predicates(idx)] == [a.iri, b.iri]
    assert set(_predicates_of(idx)) == {a, b}
