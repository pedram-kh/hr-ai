"""The answer-provider interface and its plain data shapes (ADR-0015).

These dataclasses are deliberately framework-free (not pydantic): the provider
layer is a thin port that knows nothing about FastAPI or hr-backend. `app/main.py`
validates the HTTP body with pydantic and maps it onto these.

`api_key` is a per-call argument — never an instance field — so a key can never
outlive the single synthesis call it was handed for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkInput:
    """One eligible retrieved chunk passed to synthesis.

    `authority_level` is load-bearing for the precedence rule: the employee's
    convenio governs the topics it addresses; `national_law` (the Estatuto) is
    the baseline that applies only where the convenio is silent. hr-backend
    orders convenio chunks before national_law chunks before handing them here.

    `source_type` (Sprint 7c Q7) discriminates a vector chunk from a structured
    reference fact: a `reference_fact` source has chunk_id=None (it is not a
    chunk, ADR-0006) and rides the SAME ordered source list + citation-mapping +
    precedence as any chunk — it is just another typed, authority-labelled source.
    """

    chunk_id: int | None
    document_id: int
    page_from: int | None
    page_to: int | None
    content: str
    score: float
    authority_level: str | None
    source_type: str = "chunk"  # 'chunk' | 'reference_fact'


@dataclass
class ProviderConfig:
    """Non-secret provider settings (which model, which EU endpoint). Comes from
    hr-ai's own config — NOT from hr-backend, and NOT a secret."""

    provider: str
    model: str
    endpoint: str | None = None


@dataclass
class SynthesisResult:
    """What a provider returns for one synthesis call.

    `authority_used` is computed deterministically from the authority levels of
    the cited chunks (not trusted from the model) so an auditor can see whether an
    answer was drawn from the convenio or from the national-law baseline.
    """

    answer: str
    citations: list[dict] = field(default_factory=list)
    grounding_signal: dict = field(default_factory=dict)
    confidence: float = 0.0
    authority_used: list[str] = field(default_factory=list)
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class GroundChunk:
    """One CITED chunk passed to the per-claim grounding check (Sprint 2b-2, §5).

    `is_tabular` lets the grounding prompt be table-aware: digit-presence in a
    tabular/columnar chunk is NOT entailment of a claim (Q5's lesson) — the
    salary path is SQL, but a prose answer that happens to cite a wage-table
    chunk must not be ruled "grounded" just because the digits appear.
    """

    chunk_id: int | None = None
    content: str = ""
    authority_level: str | None = None
    is_tabular: bool = False
    source_type: str = "chunk"  # 'chunk' | 'reference_fact' (Sprint 7c Q7)


@dataclass
class RouterResult:
    """The router's classification of ONE question (Sprint 2b-2, ADR-0016).

    `label` ∈ salary | prose | off_domain. `subqueries` is non-empty only for a
    compound question (used by hr-backend's recall hardening: one /retrieve per
    sub-query, unioned — §6). The router sees the QUESTION only, never the chunks.
    """

    label: str
    confidence: float = 0.0
    subqueries: list[str] = field(default_factory=list)
    reason: str = ""
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class VocabularyCandidate:
    """One closed-vocabulary value hr-backend passes to the tagging tier so the
    model BINDS to a real id rather than inventing free text (Sprint 7a). The
    full territory/sector/document_type lists are tiny + closed; convenios are a
    parser-hint shortlist. `aliases` lets the model match a spelling variant."""

    id: int
    name: str
    aliases: list[str] = field(default_factory=list)
    code: str | None = None  # document_type code / territory code


@dataclass
class TagProposalResult:
    """The document-level facet proposal for ONE document (Sprint 7a, ADR-0020).

    The AI is a STRICT PROPOSER and is INERT: every value here is a SUGGESTION
    that hr-backend persists as `ai_agent` provenance ONLY — it never writes the
    authoritative scope FKs, and the document stays `under_review` (the embedding
    gate keeps it unretrievable) until a human verifies. hr-ai writes NOTHING.

    - `facets`: each {facet, value_id|value_code|value, confidence}. The model
      resolves convenio/territory/sector/document_type to a candidate ID; it
      records validity as a string. It BINDS into the provided vocabulary only.
    - `topics`: existing APPROVED topic ids the document covers (never new ones).
    - `raw_unmatched_values`: values it could not resolve, each with an optional
      `variant_of` hint for the propose-new-vocabulary flow (it never invents).
    - `overall_confidence`: the min across facets — drives the review queue order.

    This is DOCUMENT-LEVEL facet tagging only — never multi-scope fact
    segmentation (that is Sprint 7b). One document → one set of facets.
    """

    facets: list[dict] = field(default_factory=list)
    topics: list[dict] = field(default_factory=list)
    raw_unmatched_values: list[dict] = field(default_factory=list)
    overall_confidence: float = 0.0
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class JobCategoryCandidate:
    """One convenio-scoped job category in the segmentation candidate vocabulary
    (Sprint 7b-2). Sparse: `convenio_job_categories` is salary-derived, so many
    periodo convenios have none — then the model leaves `job_category_id` null and
    the group lives in `group_label`."""

    id: int
    name: str
    group_code: str | None = None


@dataclass
class ConvenioCandidate:
    """One convenio in the CLOSED candidate vocabulary for segmentation (Sprint
    7b-2, ADR-0022). Convenio-centric: scope rides the convenio, so each carries
    its DERIVED territory + sector (+ sparse job categories) inline. The model
    picks a `convenio_id` whose (territory, sector) matches the carried
    TERRITORY/SECTOR header — territory/sector are never returned (they derive).
    `aliases` (incl. COEAS ≡ "Ocio Educativo y Animación Sociocultural", and the
    territory spelling variants Gipuzkoa/Guipúzcoa, Bizkaia/Vizcaya) let the model
    resolve the real-world naming variance in the fixtures."""

    id: int
    name: str
    numero: str | None = None
    aliases: list[str] = field(default_factory=list)
    territory_name: str = ""
    territory_aliases: list[str] = field(default_factory=list)
    sector_name: str = ""
    sector_aliases: list[str] = field(default_factory=list)
    job_categories: list[JobCategoryCandidate] = field(default_factory=list)


@dataclass
class SegmentedFactsResult:
    """The array of per-scope facts the segmentation agent proposes for ONE
    reference_source (Sprint 7b-2, ADR-0022).

    The AI is a STRICT, INERT proposer: every fact is a SUGGESTION hr-backend
    persists as `ai_agent`/`needs_review` — not answerable, never verified by the
    agent, never a salary row, never new vocabulary. hr-ai writes NOTHING.

    Each fact dict carries: `convenio_id` (bound — territory/sector derive),
    `job_category_id` (or null), `group_label` (the group AS WRITTEN — a
    first-class identity discriminator so per-group facts don't collide on the
    logical key), `topic_id` (or null), `value` + `raw_values` (the rule, multi-
    value breakdown INSIDE — one fact per scope), `confidence`, `uncertainty`
    ({field, reason} or null — flag-don't-guess), `source_locator` +
    `source_excerpt` (the exact line + header trail, the review-UX defense).

    The agent does NOT propose validity (hr-backend derives it from the source
    document — Q7) and does NOT propose authority (hr-backend forces the floor).
    """

    facts: list[dict] = field(default_factory=list)
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class GroupProposalResult:
    """The group STRUCTURE proposed for ONE convenio (Sprint 7f, ADR-0028).

    A strict, inert proposer, exactly like `SegmentedFactsResult`: hr-backend
    persists every node as `ai_agent`/`needs_review`, nothing is comparable by
    the answer path until a human approves it, and hr-ai writes NOTHING.

    `groups` is a flat list of nodes, each:
      `code_label`      the group AS PRINTED ("Grupo 2", "Grupo I") — hr-backend
                        derives `code_normalized` with `GroupCodeNormalizer`, so
                        normalization has exactly ONE implementation and it is
                        not the model's job.
      `parent_code_label` null for a group, the parent's `code_label` for a
                        SUB-AREA. Names rather than ids because nothing exists
                        yet — hr-backend resolves the tree on persist.
      `source_excerpt`  the convenio line that justifies this node. REQUIRED for
                        a sub-area: a slice of a group that no text supports is
                        precisely the inference this sprint routes to a human.
      `source_locator`  where it was read (e.g. "p.12").
      `job_category_ids` EXISTING category ids only, validated against the closed
                        set (ADR-0011 — the category vocabulary is never minted).
      `confidence`, `uncertainty` ({field, reason} or null — flag, don't guess).

    ⚠ THE GRANULARITY RULE, which is the whole reason this endpoint is not a
    generic "list the groups" call: A SUB-AREA IS PROPOSED ONLY WHERE THE TEXT
    ASSIGNS THE SLICES DIFFERENT VALUES. Hostelería Navarra prices `área 5` of
    Grupo 2 at 90/75/60 días and `resto áreas` at 60/45/30, so those two
    sub-areas exist. A convenio that merely mentions áreas without pricing them
    differently gets ONE node for the group — an unnecessary split would make the
    Phase 3 matcher demand a distinction the convenio never made, turning a
    correct answer into an escalation.
    """

    groups: list[dict] = field(default_factory=list)
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class OcrPageResult:
    """One page's OCR transcription (Sprint 7e, ADR-0026) — the vision call's
    return shape. hr-ai READS the page image and RETURNS this; it writes NOTHING
    to the DB (the S3 sidecar write happens in `app/ocr.py`'s orchestration, not
    here — this is the pure provider call, same separation as every other
    capability: the provider never touches storage or the DB).

    `columns`/`table_rows`/`article_headers` follow the PINNED table-placement
    contract (Sprint 7e Round-2 Adjustment 2, review.md §1.6/§2.2): a table
    page's title lives ONLY in `article_headers`; a footnote/plus-line block
    lives ONLY in one `columns` `es` entry; `table_rows` holds ONLY the grid.
    `bilingual` is derived the same way `extract_columns.py`'s native two-column
    path derives it: a two-column layout where the two columns' `language`
    differ. `trace_fragment` carries `cost_usd`/`sec_per_page`/`model` for the
    caller to log and persist as `document_pages.ocr_cost_usd`/`ocr_engine`.
    """

    layout: str
    columns: list[dict] = field(default_factory=list)
    table_rows: list[list] = field(default_factory=list)
    article_headers: list[str] = field(default_factory=list)
    bilingual: bool = False
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class GroundingResult:
    """The per-claim entailment verdict for one prose answer (Sprint 2b-2, §5).

    `grounded` is the gate signal: True only when EVERY load-bearing claim is
    entailed by a cited chunk. `claims` records each claim + its verdict +
    the supporting chunk (for the audit trail / the trace). hr-backend escalates
    the whole turn (low_confidence) when `grounded` is False — never edits.
    """

    grounded: bool
    claims: list[dict] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    trace_fragment: dict = field(default_factory=dict)


class AnswerProvider(ABC):
    """A pluggable answer provider (synthesis + routing + grounding). Default: Claude.

    All three calls reuse the SAME hr-backend-owned key path (ADR-0015): the key
    is a per-call argument, never an instance field, never persisted. The MODEL
    differs by call (ADR-0016 / Sprint 2b-2 §5): `route` uses the small/fast
    ROUTER_MODEL; `synthesise` and `ground` use the capable ANSWER_MODEL
    (entailment is subtle — never judged by the cheap classifier).
    """

    @abstractmethod
    def synthesise(
        self,
        question: str,
        chunks: list[ChunkInput],
        api_key: str,
        config: ProviderConfig,
    ) -> SynthesisResult:
        """Compose a cited answer grounded ONLY in `chunks`, honouring the
        convenio-over-baseline precedence rule. `api_key` is used for this one
        call and never persisted."""
        raise NotImplementedError

    @abstractmethod
    def classify(
        self,
        question: str,
        api_key: str,
        config: ProviderConfig,
    ) -> RouterResult:
        """Classify the question salary | prose | off_domain and, for a compound
        question, decompose it into `subqueries` (ADR-0016). Small/fast model."""
        raise NotImplementedError

    @abstractmethod
    def ground(
        self,
        question: str,
        answer: str,
        chunks: list[GroundChunk],
        api_key: str,
        config: ProviderConfig,
    ) -> GroundingResult:
        """Per-claim entailment of `answer` against the CITED `chunks`. The real
        grounding gate (Sprint 2b-2 §5). Table-aware. Capable answer model."""
        raise NotImplementedError

    @abstractmethod
    def propose_tags(
        self,
        page_text: str,
        candidate_vocabulary: dict[str, list[VocabularyCandidate]],
        api_key: str,
        config: ProviderConfig,
    ) -> TagProposalResult:
        """Read a document's `page_text` and PROPOSE document-level facets +
        confidence, binding into `candidate_vocabulary` ONLY (Sprint 7a). The AI
        is a strict, inert proposer: it returns suggestions, never writes, never
        invents vocabulary (unresolvable values become raw_unmatched_values with
        an optional variant hint). Document-level facet tagging only — NOT
        multi-scope fact segmentation (7b)."""
        raise NotImplementedError

    @abstractmethod
    def segment_facts(
        self,
        pages_text: str,
        candidate_convenios: list[ConvenioCandidate],
        candidate_topics: list[VocabularyCandidate],
        api_key: str,
        config: ProviderConfig,
    ) -> SegmentedFactsResult:
        """Read a multi-scope reference_source's text and SEGMENT it into per-
        scope facts (Sprint 7b-2, ADR-0022). THE load-bearing instruction is
        header-carry: TERRITORY then SECTOR headers govern the value lines
        beneath them until the next header; the scope RESETS on each new header.
        The model re-derives the hierarchy FROM THE TEXT (the reader's section
        split is unreliable) and BINDS each fact's `convenio_id` to the closed
        candidate list (never invents). One fact per scope (multi-value breakdown
        inside `value`/`raw_values`). A strict, inert proposer — it returns
        suggestions, writes nothing, proposes no validity/authority, and flags
        uncertainty rather than guessing scope."""
        raise NotImplementedError

    @abstractmethod
    def propose_groups(
        self,
        convenio: ConvenioCandidate,
        pages_text: str,
        observed_group_labels: list[str],
        api_key: str,
        config: ProviderConfig,
    ) -> GroupProposalResult:
        """Read ONE convenio's own text and propose its GROUP STRUCTURE (Sprint
        7f, ADR-0028) — the vocabulary the answer path will later compare
        exactly, replacing a bare-digit regex.

        Three inputs, with deliberately unequal authority:
          `pages_text`  the convenio's own text — THE ONLY SOURCE OF TRUTH. Every
                        node must carry an excerpt from it.
          `convenio.job_categories` existing categories, with `group_code` as
                        EVIDENCE ONLY, never as truth: across the real corpus 72
                        of 94 rows hold nothing to normalize and 9 of the
                        remaining 22 hold a salary figure or a year. A code that
                        the text contradicts is ignored; the categories
                        themselves are a CLOSED set and are never minted.
          `observed_group_labels` the `group_label` strings human-verified
                        reference facts already use for this convenio. These are
                        a CHECKLIST, not a source: a structure that cannot
                        express a label a verified fact already uses would leave
                        that fact unbindable and permanently escalating.

        A sub-area is proposed ONLY where the text assigns the slices different
        values (see `GroupProposalResult`). A strict, inert proposer: it returns
        suggestions, writes nothing, mints no category, normalizes no code, and
        flags uncertainty rather than guessing."""
        raise NotImplementedError

    @abstractmethod
    def ocr_page(
        self,
        image_bytes: bytes,
        api_key: str,
        config: ProviderConfig,
    ) -> OcrPageResult:
        """OCR one already-rendered page image (Sprint 7e, ADR-0026). Reads the
        page and TRANSCRIBES it, literally — no cleanup/normalization, no
        summarization, an illegible word becomes `[ilegible]` rather than a
        guess. Binds to the pinned table-placement contract (see
        `OcrPageResult`). `api_key` is a per-call argument, used for this one
        page only, never persisted (same discipline as every other call)."""
        raise NotImplementedError
