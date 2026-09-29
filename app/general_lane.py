"""The `general_knowledge` lane's web-fetching + PII-defence layer (Sprint 13,
step 9, plan.md §B.6.2/§B.6.5). hr-backend's `PiiScrubber` (PHP) is the
PRIMARY scrub — it knows who the employee is and replaces name/email/DNI/
NIE/NAF/IBAN/phone/convenio-name/territory-name/money/date tokens with typed
placeholders BEFORE the question ever reaches this service. This module is
DEFENCE IN DEPTH: it re-applies the pattern-level part (the parts that don't
need to know who the employee is) and REFUSES the call outright if any
pattern is still present, and it is the ONLY thing in hr-ai that is allowed to
reach the public internet, on a tight SSRF-safe leash.

Two independent responsibilities, deliberately kept in one small module so
both are exercised by the same test (`scripts/general_lane_fetch_test.py`):

1. `refuse_if_pii(text)` — the defence-in-depth pattern re-check.
2. `fetch_source(url, allowed_domains)` — ONE bounded, allowlisted, SSRF-safe
   HTML fetch, returning extracted text. No cookies, no auto-redirect-follow
   (manual, re-checked, capped hops), public-IP-only, https-only, `text/html`
   only, fixed User-Agent, no query string, no request body — there is
   nothing in this fetch for a URL to leak the question through even if a
   catalogue URL were ever attacker-influenced (it isn't: the model picks a
   catalogue id, never a URL — see `select_sources`).
"""

from __future__ import annotations

import functools
import ipaddress
import re
import socket
import ssl
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

# --- 1. PII defence in depth (§B.6.2) -------------------------------------
# Only the PATTERN-recognisable kinds — the ones that don't require knowing
# who the employee is (name, convenio name/numero, territory names are
# hr-backend-only: they need the employee/scope record). A hit here means
# hr-backend's PiiScrubber missed something and this call must be refused,
# never silently "best-effort" sent on.
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "dni": re.compile(r"\b\d{8}[A-Za-z]\b"),
    "nie": re.compile(r"\b[XYZxyz]\d{7}[A-Za-z]\b"),
    "naf": re.compile(r"\b\d{2}[\/ ]?\d{8}[\/ ]?\d{2}\b"),
    "iban": re.compile(r"\bES\d{2}(\s?\d{4}){5}\b"),
    "phone": re.compile(r"(\+34\s?)?[6789]\d{2}(\s?\d{3}){2}\b"),
}


class GeneralLanePiiRefused(ValueError):
    """Raised when `question_scrubbed` still carries a recognisable PII
    pattern — the call must never reach the provider."""

    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(f"question_scrubbed still contains a '{kind}' pattern — refusing to call provider")


def refuse_if_pii(text: str) -> None:
    """Raise `GeneralLanePiiRefused` if any pattern-recognisable PII kind is
    still present in `text`. Defence in depth — hr-backend's `PiiScrubber` is
    the primary scrub; this never trusts that it ran correctly."""
    for kind, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            raise GeneralLanePiiRefused(kind)


# --- 2. Catalogue selection (model picks ids only, never URLs) -----------
def select_sources(question_scrubbed: str, catalogue: list[dict], limit: int = 2) -> list[dict]:
    """Deterministic, LOCAL keyword-overlap match between the (already-
    scrubbed) question and each catalogue entry's `topics` — never an extra
    model call, never a URL the model invented. Returns at most `limit`
    entries (§B.6.5: ≤2 fetches per turn), ranked by overlap count, ties
    broken by catalogue order (stable sort) so results are reproducible."""
    scored: list[tuple[int, dict]] = []
    for entry in catalogue:
        overlap = len(matching_topics(question_scrubbed, entry))
        if overlap > 0:
            scored.append((overlap, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def matching_topics(question_scrubbed: str, entry: dict) -> list[str]:
    """The catalogue entry's topic terms that overlap the question — the
    'priority' terms whose excerpt windows are kept first when the excerpt
    budget is exhausted (see `windowed_excerpt`)."""
    words = set(re.findall(r"[a-záéíóúñü]{4,}", question_scrubbed.lower()))
    topics = [str(t).lower() for t in entry.get("topics", [])]
    return [t for t in topics if any(t in w or w in t for w in words)]


# --- 2b. Keyword-windowed excerpts (replaces the 20k prefix cap) ----------
# A 20k-char PREFIX of a long legal page (BOE's Estatuto is ~940KB of text)
# never reaches the article the question is about. Instead the fetched page
# text is reduced to WINDOWS around every occurrence of a catalogue topic
# term, merged when they overlap, with a total budget. A page with no topic
# term at all (a generic hub / landing page) yields NO excerpt — that is
# "no material", not a thin page to answer from.
WINDOW_CHARS = 500  # context kept each side of a match
MERGE_GAP_CHARS = 120  # windows closer than this merge into one
MAX_EXCERPT_CHARS = 12_000  # total budget for one page's excerpt
WINDOW_SEPARATOR = " […] "


def _fold(text: str) -> str:
    """Lower-case + accent-stripped copy with EXACTLY the same length as the
    input (so match offsets index the original): each char is mapped to its
    first NFD code point, lower-cased. 'ñ'->'n', 'Á'->'a'."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFD", ch)[0].lower()
        out.append(base[0] if base else ch)
    return "".join(out)


@dataclass
class WindowedExcerpt:
    text: str
    windows: int
    matched_terms: list[str] = field(default_factory=list)
    matches: int = 0


def windowed_excerpt(
    text: str,
    terms: list[str],
    priority_terms: list[str] | None = None,
    *,
    window_chars: int = WINDOW_CHARS,
    merge_gap: int = MERGE_GAP_CHARS,
    max_chars: int = MAX_EXCERPT_CHARS,
) -> WindowedExcerpt:
    """Keyword-windowed excerpt of `text`: every window around a topic-term
    match, overlapping windows merged, cut on word boundaries, total capped at
    `max_chars`. When the budget cannot hold every window, windows containing a
    `priority_terms` match (the terms the QUESTION used) are kept first, then
    earlier ones; the survivors are emitted in document order. Zero matches →
    an empty excerpt (`windows == 0`)."""
    folded = _fold(text)
    term_list = [t for t in dict.fromkeys(_fold(t).strip() for t in terms) if t]
    priority = {_fold(t).strip() for t in (priority_terms or [])}

    # (start, end, term) for every match. Terms of ≤3 chars ('it') need a
    # right word boundary too, else 'it' would match 'item'/'italia'.
    hits: list[tuple[int, int, str]] = []
    for term in term_list:
        right = r"(?![a-z0-9])" if len(term) <= 3 else ""
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(term) + right, folded):
            hits.append((m.start(), m.end(), term))
    if not hits:
        return WindowedExcerpt(text="", windows=0)
    hits.sort()

    # Build windows around each hit, merging those within `merge_gap`.
    merged: list[list] = []  # [start, end, {terms}, has_priority]
    for start, end, term in hits:
        lo, hi = max(0, start - window_chars), min(len(text), end + window_chars)
        if merged and lo <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2].add(term)
            merged[-1][3] = merged[-1][3] or term in priority
        else:
            merged.append([lo, hi, {term}, term in priority])

    # Snap edges to word boundaries (never cut mid-word).
    windows: list[tuple[int, int, set[str], bool]] = []
    for lo, hi, ts, pr in merged:
        while lo > 0 and not text[lo - 1].isspace():
            lo -= 1
        while hi < len(text) and not text[hi].isspace():
            hi += 1
        windows.append((lo, hi, ts, pr))

    # Budget: priority windows first, then document order; a window that does
    # not fit whole is truncated to the remaining budget only if nothing has
    # been kept yet (a single huge window still yields SOMETHING).
    order = sorted(range(len(windows)), key=lambda i: (not windows[i][3], windows[i][0]))
    kept: list[int] = []
    used = 0
    for i in order:
        lo, hi, _, _ = windows[i]
        size = hi - lo + len(WINDOW_SEPARATOR)
        if used + size > max_chars:
            if kept:
                continue
            hi = lo + max(0, max_chars - len(WINDOW_SEPARATOR))
            windows[i] = (lo, hi, windows[i][2], windows[i][3])
            size = hi - lo
        kept.append(i)
        used += size
    kept.sort(key=lambda i: windows[i][0])

    pieces = [text[windows[i][0]:windows[i][1]].strip() for i in kept]
    matched = sorted({t for i in kept for t in windows[i][2]})
    return WindowedExcerpt(
        text=WINDOW_SEPARATOR.join(pieces),
        windows=len(kept),
        matched_terms=matched,
        matches=len(hits),
    )


# --- 2c. TLS trust: certifi + bundled intermediates (never verify=False) ---
CERTS_DIR = Path(__file__).parent / "certs"


def build_ssl_context(*, extra_cert_dir: Path | None = CERTS_DIR, cafile: str | None = None) -> ssl.SSLContext:
    """The lane's TLS context: the certifi (Mozilla) root bundle PLUS any PEM
    intermediates shipped in `app/certs/`. Verification is always on
    (CERT_REQUIRED + check_hostname) — this only fixes servers that send an
    incomplete chain (leaf without its intermediate), which browsers paper
    over with AIA fetching and OpenSSL/Python do not. Each shipped PEM is an
    intermediate whose root is in certifi, so trust still terminates at a
    public root; it is added to the store as a chain-building aid, and a
    certificate can only ever validate if its signature chain is intact.

    `cafile` overrides the root bundle (tests use a synthetic root)."""
    import certifi

    ctx = ssl.create_default_context(cafile=cafile or certifi.where())
    if extra_cert_dir and Path(extra_cert_dir).is_dir():
        for pem in sorted(Path(extra_cert_dir).glob("*.pem")):
            ctx.load_verify_locations(cafile=str(pem))
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx


@functools.lru_cache(maxsize=1)
def _default_ssl_context() -> ssl.SSLContext:
    return build_ssl_context()


# --- 3. The SSRF-safe fetcher (§B.6.5) ------------------------------------
CONNECT_TIMEOUT_S = 3.0
READ_TIMEOUT_S = 5.0
TOTAL_TIMEOUT_S = 8.0
MAX_BODY_BYTES = 1_500_000  # 1.5MB streamed cap
MAX_TEXT_CHARS = 20_000  # per-fetch cap into the prompt
MAX_REDIRECT_HOPS = 3
USER_AGENT = "hr-ai-general-lane/1.0 (+internal; no cookies; read-only)"


@dataclass
class FetchResult:
    url: str
    status: int | None
    bytes: int
    ms: int
    text: str = ""
    error: str | None = None
    # Windowed-excerpt bookkeeping (trace only — never the page text itself).
    text_chars: int = 0  # full extracted page text length
    windows: int = 0
    matched_terms: list[str] = field(default_factory=list)


class _TextExtractor(HTMLParser):
    """Minimal HTML→text: drop tags/scripts/styles, keep visible text, collapse
    whitespace. stdlib-only (no lxml/bs4 dependency) — good enough for the
    curated boe.es/mites.gob.es/seg-social.es/sepe.es pages this ever touches."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _default_resolve(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def host_is_public(host: str, *, resolve=_default_resolve) -> bool:
    """SSRF guard: every resolved IP for `host` must be a public, routable
    address — no loopback/private/link-local/reserved range. Resolves ALL
    addresses (A + AAAA) and rejects if ANY is non-public.

    `resolve` is injectable so `scripts/general_lane_fetch_test.py` can
    exercise this against a FAKE (host -> [ip, ...]) resolver, without any
    real DNS lookup and without depending on what a given sandbox's network
    policy happens to allow."""
    try:
        addrs = resolve(host)
    except OSError:
        return False
    if not addrs:
        return False
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False
    return True


def _host_allowed(host: str, allowed_domains: list[str]) -> bool:
    host = host.lower()
    for domain in allowed_domains:
        domain = domain.lower()
        if host == domain or host.endswith("." + domain):
            return True
    return False


def fetch_source(
    url: str,
    allowed_domains: list[str],
    *,
    topics: list[str] | None = None,
    priority_topics: list[str] | None = None,
    transport=None,
    resolve=_default_resolve,
    ssl_context: ssl.SSLContext | None = None,
) -> FetchResult:
    """One bounded, allowlisted, SSRF-safe HTML fetch (§B.6.5).

    `topics` (the catalogue entry's topic terms) turns the extracted page text
    into keyword WINDOWS (`windowed_excerpt`) instead of a 20k prefix, and makes
    relevance a fetch-level fact: a page containing none of the topic terms is
    `error="no_topic_match"` with no text — a generic hub page is no material.
    With `topics=None` (no terms to window on) the legacy bounded prefix is
    returned. `priority_topics` are the terms the question used.

    `transport` is an optional httpx transport override and `resolve` an
    optional (host -> [ip, ...]) resolver override, used ONLY by
    `scripts/general_lane_fetch_test.py`'s `httpx.MockTransport` + fake
    resolver — production callers never pass either. The SSRF check ALWAYS
    runs (real DNS in production, the injected fake resolver in tests) so the
    guard itself is exercised by the same code path either way.
    """
    import time

    import httpx

    started = time.monotonic()
    current = url
    hops = 0

    while True:
        parts = urlsplit(current)
        if parts.scheme != "https":
            return FetchResult(url=current, status=None, bytes=0, ms=_elapsed_ms(started), error="scheme_not_https")
        host = parts.hostname or ""
        if not _host_allowed(host, allowed_domains):
            return FetchResult(url=current, status=None, bytes=0, ms=_elapsed_ms(started), error="domain_not_allowed")
        if not host_is_public(host, resolve=resolve):
            return FetchResult(url=current, status=None, bytes=0, ms=_elapsed_ms(started), error="ssrf_blocked")

        client = httpx.Client(
            transport=transport,
            verify=ssl_context if ssl_context is not None else _default_ssl_context(),
            timeout=httpx.Timeout(TOTAL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S),
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    hops += 1
                    if not location or hops > MAX_REDIRECT_HOPS:
                        return FetchResult(url=current, status=resp.status_code, bytes=0, ms=_elapsed_ms(started), error="redirect_limit")
                    current = location
                    continue

                # An error page is not evidence. Found live on staging (step
                # 10): a catalogue URL that 404s still answers with an HTML
                # "No encontrada" page, and this fetcher used to hand that
                # text on as if it were the source's content. Any non-2xx is
                # a failed fetch — the tool then sees no usable source.
                if resp.status_code >= 400:
                    return FetchResult(url=current, status=resp.status_code, bytes=0, ms=_elapsed_ms(started), error=f"http_{resp.status_code}")

                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type.split(";")[0].strip():
                    return FetchResult(url=current, status=resp.status_code, bytes=0, ms=_elapsed_ms(started), error="not_html")

                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BODY_BYTES:
                        break

                extractor = _TextExtractor()
                extractor.feed(body.decode("utf-8", errors="replace"))
                full_text = extractor.text()

                if topics is None:
                    return FetchResult(
                        url=current,
                        status=resp.status_code,
                        bytes=len(body),
                        ms=_elapsed_ms(started),
                        text=full_text[:MAX_TEXT_CHARS],
                        text_chars=len(full_text),
                    )

                excerpt = windowed_excerpt(full_text, topics, priority_topics)
                if excerpt.windows == 0:
                    # Generic hub / landing page: reachable, but nothing about
                    # the topic. No material — never a page to answer from.
                    return FetchResult(
                        url=current,
                        status=resp.status_code,
                        bytes=len(body),
                        ms=_elapsed_ms(started),
                        error="no_topic_match",
                        text_chars=len(full_text),
                    )
                return FetchResult(
                    url=current,
                    status=resp.status_code,
                    bytes=len(body),
                    ms=_elapsed_ms(started),
                    text=excerpt.text,
                    text_chars=len(full_text),
                    windows=excerpt.windows,
                    matched_terms=excerpt.matched_terms,
                )
        except httpx.HTTPError as exc:
            return FetchResult(url=current, status=None, bytes=0, ms=_elapsed_ms(started), error=str(exc))
        finally:
            client.close()


def _elapsed_ms(started: float) -> int:
    import time

    return int((time.monotonic() - started) * 1000)
