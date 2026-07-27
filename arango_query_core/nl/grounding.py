"""Entity/instance grounding + predicate/schema-convention grounding
retrieval for NL→query prompts.

At prompt-construction time, retrieve the top-K instances from the
target's own instance data whose label(s) share tokens with the user's
question, and inject them into the adapter's prompt as a "Known
entities" block naming their exact opaque IDs (IRIs for SPARQL, node
keys for Cypher, ...). This lets the LLM reference a specific
individual by its exact ID instead of guessing a name-literal match —
a corpus/CK25 spike measured this lifting execution-graded accuracy
from 12.2% to 24.5% (McNemar p=0.031, 0 regressions).

Mirrors :class:`~arango_query_core.nl.fewshot.FewShotIndex`'s shape
exactly: a retrieval index built from caller-owned data (no file
loading, no memoization at this layer — the caller decides how/when to
build and cache the index for its own corpus/deployment), a
``retrieve(question, k)`` method, and a ``format_prompt_section(...)``
renderer that returns ``""`` on no matches so the caller can omit the
section entirely.

The retrieval/scoring machinery is target-language-agnostic — ``id``
is an opaque string the scorer never inspects. The exact prompt
wording (e.g. "use these EXACT IRIs" vs. "use these EXACT node IDs")
is intentionally NOT owned by this module; ``format_prompt_section``
accepts ``header``/``instruction``/``id_prefix``/``id_suffix`` so
callers supply their own language-specific phrasing.

:class:`GroundedPredicate`/:class:`PredicateIndex` (seam 7) extend the
same pattern one level up the stack: instead of instance data, they
retrieve over the TBox's own predicate declarations (label + domain +
range + object-vs-datatype + a mechanically-derived usage "shape"), so
the LLM gets a distilled, imperative usage cheat-sheet for schema
conventions it otherwise has to infer from raw ``rdfs:domain``/
``rdfs:range`` triples buried in the grammar prompt. Same contract:
caller-owned construction, no memoization at this layer, a
``retrieve(question, k)`` method, and a ``format_prompt_section(...)``
renderer returning ``""`` on no matches. The token-substring scorer is
shared with :class:`LabelIndex` via a single private helper
(``_token_substring_retrieve``) — not duplicated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

_T = TypeVar("_T")

_STOP = frozenset(
    "the a an of in is are on to for me my i give need all every who what "
    "which where when how does do not no and or with by their his her its "
    "please list show find get".split()
)

# C0 control chars (0x00-0x1F) + DEL (0x7F), including \n and \r — a
# maliciously-labeled instance (e.g. rdfs:label containing "\nignore
# previous instructions") must render on a single bullet line and
# cannot inject a new prompt section (T-07.3-01).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RUN_RE = re.compile(r"\s+")
_LABEL_MAX_LEN = 200


def _sanitize_label(label: str) -> str:
    """Strip control chars, collapse whitespace, cap length.

    A no-op on already-clean labels: no control chars, single spaces,
    under the length cap all pass through byte-identical (verified in
    tests) so ordinary CK25 label text is never altered.
    """
    cleaned = _CONTROL_CHARS_RE.sub(" ", label)
    cleaned = _WHITESPACE_RUN_RE.sub(" ", cleaned).strip()
    if len(cleaned) > _LABEL_MAX_LEN:
        cleaned = cleaned[:_LABEL_MAX_LEN]
    return cleaned


def _token_substring_retrieve(
    items: list[_T],
    get_labels: Callable[[_T], tuple[str, ...]],
    question: str,
    k: int,
    *,
    dump: bool = False,
) -> list[_T]:
    """Top-k items whose ``get_labels(item)`` tokens appear (as
    substrings) in the question, or vice versa.

    Both-direction substring matching survives plural/inflection
    mismatches (e.g. "Transistors" in the question vs. a "Transistor"
    label token). Ranked by hit count (desc), then shortest matching
    label (tiebreak — prefers the more specific/precise match). Ported
    verbatim from the spike's scorer
    (scratchpad/nl-grounding-spike/grounding_spike.py::retrieve),
    generalized over an item-type-agnostic ``get_labels`` accessor so
    both :class:`LabelIndex` (entity labels) and :class:`PredicateIndex`
    (predicate label/domain/range) share one scorer — not two.

    ``dump`` (default ``False``) is CR-01's real dump-mode escape hatch:
    when ``False``, an item scoring zero token hits against the question is
    dropped regardless of ``k`` (today's behavior, unchanged -- this is
    what :class:`LabelIndex` always gets, and what :class:`PredicateIndex`
    gets by default). When ``True``, zero-hit items are ALSO appended to
    ``scored`` (with ``hits=0``) instead of being dropped, so after the
    same ``(hits, -best_len)`` sort they rank after every real match and
    ``[:k]`` can return the full item list when ``k >= len(items)`` -- a
    genuine "show me everything" dump, not merely a widened retrieval cap
    (widening ``k`` alone can never add back a zero-hit item, since the
    old ``if hits:`` guard dropped it before ``k`` was ever applied).
    """
    ql = question.lower()
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", ql) if len(t) >= 3 and t not in _STOP}
    scored: list[tuple[int, int, _T]] = []
    for item in items:
        hits = 0
        best_len = 999
        for lab in get_labels(item):
            labl = lab.lower()
            lab_tokens = [t for t in re.findall(r"[a-z0-9]+", labl) if len(t) >= 3]
            h = sum(1 for t in lab_tokens if t in ql)
            h += sum(1 for t in q_tokens if t in labl and not any(t == lt for lt in lab_tokens))
            if h > hits:
                hits, best_len = h, len(labl)
        if hits or dump:
            scored.append((hits, -best_len, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in scored[:k]]


@dataclass(frozen=True)
class GroundedEntity:
    """One retrievable instance: an opaque id + its human-readable labels.

    ``id`` is opaque to this module (an RDF IRI for SPARQL, a node key
    for Cypher, ...) — the retrieval scorer never inspects its shape.
    """

    id: str
    labels: tuple[str, ...]
    type: str = ""


class LabelIndex:
    """Substring-token retrieval over a fixed list of :class:`GroundedEntity`.

    Construction is caller-owned: pass a pre-built ``list[GroundedEntity]``
    (via ``__init__`` or :meth:`from_items`). There is no file-loading
    or memoization at this layer — unlike
    :func:`~arango_query_core.nl.fewshot.cached_few_shot_index`, there
    is no single canonical "bank path" for grounding data (it varies
    per corpus/deployment); callers own the build-once discipline.
    """

    def __init__(self, entities: list[GroundedEntity]) -> None:
        self._entities: list[GroundedEntity] = list(entities)

    @classmethod
    def from_items(cls, items: list[GroundedEntity]) -> LabelIndex:
        return cls(items)

    def retrieve(self, question: str, k: int = 20) -> list[GroundedEntity]:
        """Top-k entities whose label tokens appear (as substrings) in
        the question, or vice versa. See :func:`_token_substring_retrieve`
        for the ranking rule (shared with :class:`PredicateIndex`)."""
        return _token_substring_retrieve(self._entities, lambda e: e.labels, question, k)

    def format_prompt_section(
        self,
        question: str,
        k: int = 20,
        *,
        header: str,
        instruction: str,
        id_prefix: str = "",
        id_suffix: str = "",
    ) -> str:
        """Render a generic "known entities" prompt block for *question*.

        Returns the empty string when no entities match, matching
        :meth:`~arango_query_core.nl.fewshot.FewShotIndex.format_prompt_section`'s
        contract so callers can omit the section entirely.

        ``header``/``instruction``/``id_prefix``/``id_suffix`` are
        passed through unmodified — this renderer stays language-
        agnostic; the exact wording ("EXACT IRIs" vs. "EXACT node IDs")
        is the adapter's responsibility, mirroring how
        ``grammar_prompt_section`` (not ``FewShotIndex``'s header) owns
        target-specific wording.

        Every rendered label is sanitized (control chars stripped to
        spaces, length-capped) so a maliciously-labeled instance cannot
        break the prompt-block structure (T-07.3-01).
        """
        matches = self.retrieve(question, k=k)
        if not matches:
            return ""
        lines = [header, instruction, ""]
        for e in matches:
            labels = " / ".join(sorted(_sanitize_label(lab) for lab in e.labels))
            lines.append(f'- {id_prefix}{e.id}{id_suffix} — "{labels}" ({e.type or "?"})')
        return "\n".join(lines)


@dataclass(frozen=True)
class GroundedPredicate:
    """One retrievable TBox predicate: label + domain + range + kind +
    a mechanically-derived usage ``shape``.

    ``iri`` is opaque to this module, mirroring :attr:`GroundedEntity.id`.
    ``kind`` is ``"object"`` or ``"datatype"`` (``owl:ObjectProperty`` vs
    ``owl:DatatypeProperty``). ``shape`` is one of ``"value_object"``,
    ``"category_instance"``, ``"linked_entity"``, or ``"literal"`` — the
    caller (adapter/eval-harness TBox walker) derives it purely from
    ``rdfs:domain``/``rdfs:range`` declarations, never hand-curated per
    schema. ``shape_detail`` carries ``(child_label, child_range)``
    pairs for ``"value_object"`` predicates only (the datatype-property
    children of the range class, used to render the extra-hop example
    triple) — empty for every other shape.
    """

    iri: str
    label: str
    kind: str
    domain: str
    range: str
    shape: str
    shape_detail: tuple[tuple[str, str], ...] = ()


class PredicateIndex:
    """Substring-token retrieval over a fixed list of :class:`GroundedPredicate`.

    Mirrors :class:`LabelIndex` exactly: caller-owned construction (via
    ``__init__`` or :meth:`from_items`), no file-loading or memoization
    at this layer, a ``retrieve(question, k)`` method, and a
    ``format_prompt_section(...)`` renderer returning ``""`` on no
    matches. Retrieval reuses the SAME shared scorer
    (:func:`_token_substring_retrieve`) — not a second hand-rolled one.

    ``retrieve``'s ``k`` only caps candidates; it does NOT by itself decide
    dump-vs-retrieve mode (CR-01 correction: widening ``k`` to
    ``len(predicates)`` alone is a no-op against the shared scorer's
    zero-hit filter — a predicate with no lexical overlap against the
    question is dropped regardless of ``k``). Genuine dump mode is the
    explicit ``dump=True`` keyword on :meth:`retrieve`/
    :meth:`format_prompt_section`, which bypasses that filter and returns
    every predicate. The caller (adapter/eval-harness) still owns choosing
    which ``k``/``dump`` to pass based on schema size, mirroring
    :class:`LabelIndex`'s "no business logic in this class" ethos.
    """

    def __init__(self, predicates: list[GroundedPredicate]) -> None:
        self._predicates: list[GroundedPredicate] = list(predicates)

    @classmethod
    def from_items(cls, items: list[GroundedPredicate]) -> PredicateIndex:
        return cls(items)

    def retrieve(self, question: str, k: int = 20, *, dump: bool = False) -> list[GroundedPredicate]:
        """Top-k predicates whose label/domain/range tokens appear (as
        substrings) in the question, or vice versa — identical ranking
        rule to :meth:`LabelIndex.retrieve` (shared scorer), scored over
        ``(label, domain, range)`` rather than a single label tuple.

        ``dump=True`` (CR-01 fix) bypasses the shared scorer's zero-hit
        filter, returning every predicate (ranked, real hits first) instead
        of only the ones that lexically overlap the question — the genuine
        "dump the whole schema" mode a small-enough TBox needs, which
        widening ``k`` alone can never achieve (see
        :func:`_token_substring_retrieve`'s docstring). Default ``False``
        keeps prior behavior byte-identical."""
        return _token_substring_retrieve(
            self._predicates, lambda p: (p.label, p.domain, p.range), question, k, dump=dump
        )

    def format_prompt_section(
        self,
        question: str,
        k: int = 20,
        *,
        header: str,
        instruction: str,
        dump: bool = False,
    ) -> str:
        """Render a two-tier "known schema predicates" prompt block for
        *question*.

        Returns the empty string when no predicates match, matching
        :meth:`LabelIndex.format_prompt_section`'s contract so callers
        can omit the section entirely.

        Every predicate gets a terse ``label (domain -> range) [kind]``
        line. Predicates whose ``shape`` is ``"value_object"`` or
        ``"category_instance"`` additionally get an expanded, bracketed
        shape tag plus an example triple pattern demonstrating the
        required extra hop — the two conventions CK25's convention-bound
        failures actually need spelled out. ``"linked_entity"``/
        ``"literal"`` predicates are already unambiguous 1-hop edges the
        raw ontology block states elsewhere, so they stay terse (token
        budget discipline).

        Every rendered label/domain/range/shape-detail string is
        sanitized (control chars stripped to spaces, length-capped) so
        a maliciously-labeled TBox predicate cannot break the
        prompt-block structure (T-07.4-01, extending T-07.3-01).
        """
        matches = self.retrieve(question, k=k, dump=dump)
        if not matches:
            return ""
        lines = [header, instruction, ""]
        for p in matches:
            label = _sanitize_label(p.label)
            domain = _sanitize_label(p.domain) if p.domain else "?"
            range_ = _sanitize_label(p.range) if p.range else "?"
            lines.append(f"- {label} ({domain} -> {range_}) [{p.kind}]")
            if p.shape == "value_object":
                hop_labels = [_sanitize_label(child_label) for child_label, _ in p.shape_detail]
                triple = " . ".join(f"?v {hop} ?c{i}" for i, hop in enumerate(hop_labels))
                example = f"?x {label} ?v . {triple}".strip()
                lines.append(f"  [VALUE OBJECT] extra hop required, e.g. `{example}`")
            elif p.shape == "category_instance":
                lines.append(
                    f"  [CATEGORY] bind directly to a known instance IRI, e.g. "
                    f"`?x {label} <IRI>` — see Known entities for the exact IRI"
                )
        return "\n".join(lines)
