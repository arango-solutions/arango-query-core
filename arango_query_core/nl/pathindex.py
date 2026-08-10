"""Relationship-path grounding (seam 8): bounded shortest predicate paths
between two already-anchored classes.

Seam-6 (``LabelIndex``) and seam-7 (``PredicateIndex``) retrieve
*things* (instances, predicates) that lexically overlap a question.
Seam-8 (:class:`ClassPathIndex`) is different: it does NOT self-score
against a question at all (D-02). It consumes anchor classes already
resolved by seam-6 (the grounded entity's ``rdf:type``) and target
predicates/classes already resolved by seam-7's own token scorer, and
answers a purely structural question over the TBox's own class-
connectivity graph: "what is the shortest predicate path connecting
this anchor class to that target?"

This is the fix for the composed ``grounded + few-shot`` arm's 16
verified "right-entity, wrong-path" CK25 failures (root-caused
2026-08): the model grounds the named entity correctly (seam 6 already
works) but invents a predicate path that executes to an empty result
set because it never sees the schema's actual navigation from that
entity's class to what the question asks for. Surfacing the ACTUAL
mechanical path (not the whole schema — that was 07.4's distraction
regression) is the targeted fix.

Construction discipline mirrors :class:`~arango_query_core.nl.grounding.LabelIndex`/
:class:`~arango_query_core.nl.grounding.PredicateIndex` byte-for-byte:
frozen/caller-owned, stdlib-only, no memoization, ``from_items(...)``
classmethod, ``""``-on-no-match renderer contract. The one deliberate
departure (Pitfall 4 / D-02): :meth:`ClassPathIndex.shortest_paths`
takes PRE-RESOLVED anchor classes and targets, never a raw question
string — a third independent question-parser inside this module would
duplicate seam-6/7's already-proven retrieval.

Graph build (mechanical, TBox-only, D-9/D-2/D-10):

- Every declared object property ``(pred, domain, range)`` contributes
  a forward edge ``domain -[pred]-> range`` AND its inverse
  ``range -[pred^-1]-> domain`` (D-2 — required for cases like the
  ck25-12-shaped "anchor on the range side, target on the domain
  side" navigation).
- ``rdfs:subClassOf`` pairs are unioned into connected components (D-9):
  a subclass and its ancestors/descendants share ONE effective edge
  set, so a predicate declared with ``domain: Employee`` is reachable
  from an ``Agent``-typed anchor and vice versa — without this, the
  canonical inverse-join case (``Employee ⊑ Agent``, ``memberOf``
  domain=``Agent``, ``hasManager`` domain=``Employee``) is unreachable
  from a naive exact-class graph.
- Bounded self-revisit (D-10): a simple-path constraint (never revisit
  an already-visited class) would incorrectly reject even a length-1
  self-referential edge (``domain == range``, e.g. a "compatible
  product" relation) because the start class is trivially "already
  visited". Self-loop traversal is allowed, but bounded to AT MOST ONE
  use per path — a second traversal of the same self-loop edge within
  one path is refused, so the bound never turns into an unbounded cycle
  even though depth alone (≤ 3) would otherwise permit revisiting
  it.

Node keys are class LOCAL NAMES (e.g. ``"Department"``), never full
IRIs — consistent with how ``GroundedEntity.type``/``GroundedPredicate.
domain``/``.range`` are already local names elsewhere in this package
(Pitfall 3): a full-IRI join key would silently break anchor resolution
since seam-6 never renders one.

The path-surface budget (D-03/D-05, and the direct control for the
07.4 distraction regression): every candidate path across every
(anchor, target) combination is pooled into ONE global list, ranked
deterministically, and capped at ``k`` (default 5) — never a
per-(anchor, target) fair share.

Deterministic ordering: (path length, target rank, lexical signature).
"Target rank" is the position of the matching entry within the
caller-supplied ``targets`` list — the caller (the adapter, added in a
later plan) is expected to pass seam-7's OWN retrieval order (already
question-relevance-ranked), so this module never needs to re-derive a
"question-token overlap" score itself; it only needs to respect an
already-ranked input list. "Lexical signature" (anchor, edges, target)
is the final, fully deterministic tiebreak.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from arango_query_core.nl.grounding import _sanitize_label

# D-1: depth ≤ 3 (locked design decision; the two 4-hop supply-chain
# cases are a documented, accepted known limitation — never relaxed here).
_MAX_DEPTH = 3

# D-3/D-5: at most 5 paths surfaced, globally, regardless of how many
# (anchor, target) combinations exist. This is the anti-distraction
# safety valve — see the module docstring.
_DEFAULT_K = 5

# Suffix marking an edge traversed against its declared direction
# (D-2, inverse). Chosen to be trivially strippable/detectable and to
# never collide with a real predicate local name (``^`` is not a legal
# character in an IRI local name / QName suffix).
_INVERSE_SUFFIX = "^-1"


@dataclass(frozen=True)
class ClassPath:
    """One retrievable class-to-class navigation.

    ``anchor``/``target`` are class LOCAL NAMES. ``edges`` is the
    ordered predicate-local-name hop sequence connecting them; an entry
    ending in ``"^-1"`` means that hop was traversed against the
    predicate's declared ``rdfs:domain``/``rdfs:range`` direction
    (D-2, inverse). ``length`` is ``len(edges)`` (≤ 3, D-1).
    """

    anchor: str
    edges: tuple[str, ...]
    target: str

    @property
    def length(self) -> int:
        return len(self.edges)


def _strip_inverse(edge: str) -> tuple[str, bool]:
    """``"memberOf^-1"`` -> ``("memberOf", True)``; ``"hasManager"`` -> ``("hasManager", False)``."""
    if edge.endswith(_INVERSE_SUFFIX):
        return edge[: -len(_INVERSE_SUFFIX)], True
    return edge, False


class ClassPathIndex:
    """Bounded shortest-path retrieval over a mechanically-built,
    subclass-aware, inverse-edged class-connectivity graph.

    Construction is caller-owned: pass the raw edge/subclass-pair lists
    (via ``__init__`` or :meth:`from_items`) that the caller's own
    TBox walker extracted. There is no file I/O and no memoization at
    this layer, mirroring :class:`~arango_query_core.nl.grounding.LabelIndex`/
    :class:`~arango_query_core.nl.grounding.PredicateIndex`.
    """

    def __init__(
        self,
        edges: list[tuple[str, str, str]],
        subclass_of: list[tuple[str, str]],
    ) -> None:
        """``edges`` is a list of ``(predicate_local_name, domain_local_name,
        range_local_name)`` triples for every declared object property.
        ``subclass_of`` is a list of ``(sub_local_name, super_local_name)``
        pairs. Both are plain tuples of local-name strings — this class
        never inspects a full IRI."""
        self._raw_edges: list[tuple[str, str, str]] = list(edges)
        self._subclass_of: list[tuple[str, str]] = list(subclass_of)
        self._adj: dict[str, list[tuple[str, str]]] = self._build_adjacency()

    @classmethod
    def from_items(
        cls,
        edges: list[tuple[str, str, str]],
        subclass_of: list[tuple[str, str]],
    ) -> ClassPathIndex:
        return cls(edges, subclass_of)

    # -- graph build ---------------------------------------------------

    def _build_adjacency(self) -> dict[str, list[tuple[str, str]]]:
        """Build the effective adjacency map used for traversal: every
        class's OWN declared edges (forward + inverse, D-2) unioned with
        every subclass-linked relative's edges (D-9 — see module
        docstring)."""
        direct: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for pred, dom, rng in self._raw_edges:
            if not pred or not dom or not rng:
                continue
            direct[dom].append((pred, rng))
            direct[rng].append((f"{pred}{_INVERSE_SUFFIX}", dom))

        all_classes: set[str] = set(direct)
        for sub, sup in self._subclass_of:
            if sub:
                all_classes.add(sub)
            if sup:
                all_classes.add(sup)

        # Union-find over the subclass graph (D-9): a subclass and its
        # ancestors/descendants form ONE connected component that shares
        # every member's own edge set — the mechanical fix for the
        # canonical "domain declared on the subclass, anchor resolved as
        # the superclass" (or vice versa) miss (Pitfall 2).
        parent: dict[str, str] = {c: c for c in all_classes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for sub, sup in self._subclass_of:
            if sub and sup:
                union(sub, sup)

        components: dict[str, set[str]] = defaultdict(set)
        for c in all_classes:
            components[find(c)].add(c)

        effective: dict[str, list[tuple[str, str]]] = {}
        for c in all_classes:
            comp = components[find(c)]
            seen: set[tuple[str, str]] = set()
            merged: list[tuple[str, str]] = []
            for member in sorted(comp):  # deterministic merge order
                for edge, dest in direct.get(member, []):
                    key = (edge, dest)
                    if key not in seen:
                        seen.add(key)
                        merged.append((edge, dest))
            effective[c] = merged
        return effective

    # -- path enumeration ------------------------------------------------

    def _enumerate_paths(self, start: str) -> list[tuple[tuple[str, ...], str]]:
        """All ``(edges, end_class)`` pairs reachable from *start* within
        :data:`_MAX_DEPTH` hops (D-1), honoring the bounded-self-revisit
        rule (D-10 — see module docstring).

        Visited-tracking is on the LITERAL destination class name, not
        its D-9 component id: two different literal classes merged into
        the same component by subclass linking (e.g. ``Agent``/
        ``Manager``) must both remain independently reachable as
        distinct path steps (``Agent -[hasManager]-> Manager`` is a
        real, meaningful narrowing hop, not a revisit) — this is exactly
        the canonical inverse-join shape (D-9 + D-2) seam-8 exists to
        recover. A component-level visited set would incorrectly refuse
        this transition the moment ANY member of that component had been
        touched. The (rare, harmless) side effect is that a path may
        re-enter a *different* literal class that happens to share a
        component with an earlier hop (e.g. bouncing back through an
        inverse edge to a subclass alias of the start); such a path is
        never wrong, merely non-minimal, and the deterministic
        ``(length, ...)`` sort in :meth:`shortest_paths` always ranks a
        genuinely shorter path for the same target ahead of it.
        """
        results: list[tuple[tuple[str, ...], str]] = []
        # stack: (node, edges_so_far, visited_classes, self_loop_used)
        stack: list[tuple[str, tuple[str, ...], tuple[str, ...], bool]] = [
            (start, (), (start,), False)
        ]
        while stack:
            node, edges, visited, self_used = stack.pop()
            if edges:
                results.append((edges, node))
            if len(edges) >= _MAX_DEPTH:
                continue
            for edge, dest in self._adj.get(node, []):
                is_self_loop = dest == node
                if dest in visited:
                    if not (is_self_loop and not self_used):
                        continue  # a real revisit of an already-visited literal class — refused
                    new_visited = visited
                    new_self_used = True
                else:
                    new_visited = visited + (dest,)
                    new_self_used = self_used or is_self_loop
                stack.append((dest, edges + (edge,), new_visited, new_self_used))
        return results

    @staticmethod
    def _match_target(edges: tuple[str, ...], end_class: str, targets: list[str]) -> str | None:
        """First entry of *targets* (in caller-supplied order) this path
        satisfies — either the path's end class, or any hop's bare
        (direction-stripped) predicate name. Returns ``None`` on no match."""
        hop_names = {_strip_inverse(e)[0] for e in edges}
        for t in targets:
            if t == end_class or t in hop_names:
                return t
        return None

    def shortest_paths(
        self, anchor_classes: list[str], targets: list[str], k: int = _DEFAULT_K
    ) -> list[ClassPath]:
        """Bounded shortest predicate paths from any of *anchor_classes* to
        any of *targets* (D-02: both are already-resolved identifiers, not
        a question string — see the module docstring's Pitfall-4 note).

        Pools every candidate across every (anchor, target) combination
        into ONE global list (D-03), ranked by
        ``(length, target_rank, lexical_signature)`` and capped at ``k``
        (D-05). ``target_rank`` is the position of the matching target
        within the caller-supplied ``targets`` order — this module never
        re-derives question relevance itself, it only respects an
        already-ranked input (the caller's seam-7 retrieval order).
        """
        if not anchor_classes or not targets:
            return []

        target_rank = {t: i for i, t in enumerate(targets)}
        pool: dict[tuple[str, tuple[str, ...], str], ClassPath] = {}
        for anchor in anchor_classes:
            for edges, end_class in self._enumerate_paths(anchor):
                matched = self._match_target(edges, end_class, targets)
                if matched is None:
                    continue
                key = (anchor, edges, end_class)
                if key not in pool:
                    pool[key] = ClassPath(anchor=anchor, edges=edges, target=end_class)

        def sort_key(p: ClassPath) -> tuple[int, int, tuple[str, tuple[str, ...], str]]:
            matched = self._match_target(p.edges, p.target, targets)
            rank = target_rank.get(matched, len(targets)) if matched is not None else len(targets)
            return (p.length, rank, (p.anchor, p.edges, p.target))

        ranked = sorted(pool.values(), key=sort_key)
        return ranked[:k]

    # -- renderer ----------------------------------------------------------

    def format_prompt_section(
        self,
        anchor_classes: list[str],
        targets: list[str],
        k: int = _DEFAULT_K,
        *,
        header: str = "## Known navigation paths (use a SHARED variable per hop)",
        instruction: str = (
            "These are the ONLY valid multi-hop joins from the grounded entity above "
            "to the requested target. Render each as a shared-variable join (a single "
            "intermediate variable reused across hops), never as separate, unconnected "
            "triples."
        ),
    ) -> str:
        """Render at most ``k`` navigation hints as shared-variable join/star
        patterns (D-04 — NOT a directed A-to-B-to-C walk).

        Returns ``""`` when anchor/target is unresolved or no path exists
        within depth 3 (D-8, mirrors seam-6/7's own no-match contract) so
        the caller can omit the section entirely. ``<ANCHOR>`` is a literal
        placeholder token the caller substitutes with the actual grounded
        instance IRI (seam 6 already renders the real IRI in its own
        block; this renderer only knows about CLASSES, never instances).
        Every rendered label is sanitized (control-char strip, length cap)
        so a maliciously-labeled TBox predicate cannot break the
        prompt-block structure (mirrors T-07.3-01/T-07.4-01).
        """
        paths = self.shortest_paths(anchor_classes, targets, k=k)
        if not paths:
            return ""
        lines = [header, instruction, ""]
        for p in paths:
            hop_labels = " -> ".join(_sanitize_label(_strip_inverse(e)[0]) for e in p.edges)
            pattern = self._render_join_pattern(p)
            lines.append(
                f"- {_sanitize_label(p.anchor)} -> {hop_labels} -> {_sanitize_label(p.target)}: `{pattern}`"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_join_pattern(path: ClassPath) -> str:
        """Render *path* as a chain of shared-variable triples (D-04).

        Each hop becomes one triple; consecutive triples share the
        intermediate variable (a join), so a 2-hop inverse+forward path
        like ``memberOf^-1, hasManager`` renders as
        ``?hop1 memberOf <ANCHOR> . ?hop1 hasManager ?result .`` — a
        star/join around the shared intermediate, never a directed
        A-to-B-to-C walk.
        """
        triples: list[str] = []
        prev_var = "<ANCHOR>"
        for i, raw_edge in enumerate(path.edges):
            pred, inverse = _strip_inverse(raw_edge)
            pred = _sanitize_label(pred)
            is_last = i == len(path.edges) - 1
            next_var = "?result" if is_last else f"?hop{i + 1}"
            if inverse:
                triples.append(f"{next_var} {pred} {prev_var} .")
            else:
                triples.append(f"{prev_var} {pred} {next_var} .")
            prev_var = next_var
        return " ".join(triples)
