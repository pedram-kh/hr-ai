"""Sprint 13, step 9 (plan.md §B.6.5) — the `general_knowledge` lane's
SSRF-safe fetcher contract. No live network, no live API (`httpx.MockTransport`
+ a fake DNS resolver, same house style as `planner_contract_test.py`).

Asserts:
  1. A non-allowlisted domain is refused WITHOUT any HTTP call being made.
  2. A redirect to an allowlisted domain is followed (bounded, manual hops)
     and the allowlist is RE-CHECKED on every hop.
  3. A redirect chain longer than the cap is refused (`redirect_limit`).
  4. A redirect to a non-allowlisted domain is refused mid-chain.
  5. The SSRF guard (`host_is_public`) blocks a private/loopback/link-local
     resolved address even for an otherwise-allowlisted host.
  6. Every outbound request carries NO query string and NO body — the
     question text never leaves in the URL or the request (the fetcher
     signature itself has no question parameter at all).
  7. `select_sources` never returns more than the fetch cap and only returns
     catalogue entries, never invents a URL.
  8. Basic HTML→text extraction strips `<script>`/`<style>` and collapses
     whitespace.

Run:
    python3 scripts/general_lane_fetch_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app.general_lane import fetch_source, host_is_public, select_sources  # noqa: E402

ALLOWED = ["boe.es", "seg-social.es"]

failures: list[str] = []


def _resolver_public(host: str) -> list[str]:
    return ["93.184.216.34"]  # an arbitrary public unicast address


def _resolver_private(host: str) -> list[str]:
    return ["10.0.0.5"]


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        failures.append(f"{label} FAILED: {detail}")
    print(label, "FAIL" if not condition else "OK")


def main() -> int:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        url = str(request.url)
        if url == "https://boe.es/pagina":
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=(
                b"<html><body><script>evil()</script><style>.x{}</style>"
                b"<p>Texto   con   espacios</p></body></html>"
            ))
        if url == "https://boe.es/redirect-once":
            return httpx.Response(302, headers={"location": "https://seg-social.es/destino"})
        if url == "https://seg-social.es/destino":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>Destino final</p>")
        if url == "https://boe.es/redirect-loop":
            return httpx.Response(302, headers={"location": "https://boe.es/redirect-loop-2"})
        if url == "https://boe.es/redirect-loop-2":
            return httpx.Response(302, headers={"location": "https://boe.es/redirect-loop-3"})
        if url == "https://boe.es/redirect-loop-3":
            return httpx.Response(302, headers={"location": "https://boe.es/redirect-loop-4"})
        if url == "https://boe.es/redirect-loop-4":
            return httpx.Response(302, headers={"location": "https://boe.es/redirect-loop-5"})
        if url == "https://boe.es/redirect-away":
            return httpx.Response(302, headers={"location": "https://evil.example/steal"})
        if url == "https://boe.es/missing":
            # A real 404 still carries an HTML body ("No encontrada ...").
            return httpx.Response(404, headers={"content-type": "text/html"}, content=b"<p>No encontrada</p>")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    # --- (1) non-allowlisted domain refused, no HTTP call made ---
    requested_urls.clear()
    result = fetch_source("https://evil.example/page", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(1) non-allowlisted domain refused", result.error == "domain_not_allowed", result.error)
    check("(1) no HTTP call was made", requested_urls == [], str(requested_urls))

    # --- (2) redirect to an allowlisted domain is followed, allowlist re-checked ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/redirect-once", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(2) redirect followed to final allowlisted host", result.text == "Destino final", repr(result))
    check("(2) both hops requested", requested_urls == ["https://boe.es/redirect-once", "https://seg-social.es/destino"], str(requested_urls))

    # --- (3) redirect chain longer than the cap is refused ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/redirect-loop", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(3) redirect chain over the cap is refused", result.error == "redirect_limit", result.error)
    check("(3) hop count is bounded", len(requested_urls) <= 5, str(requested_urls))

    # --- (4) redirect to a non-allowlisted domain refused mid-chain ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/redirect-away", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(4) redirect to non-allowlisted domain refused", result.error == "domain_not_allowed", result.error)
    check("(4) the disallowed hop was never actually requested", requested_urls == ["https://boe.es/redirect-away"], str(requested_urls))

    # --- (5) SSRF guard blocks a private resolved address ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/pagina", ALLOWED, transport=transport, resolve=_resolver_private)
    check("(5) private resolved IP is blocked", result.error == "ssrf_blocked", result.error)
    check("(5) no HTTP call was made when SSRF-blocked", requested_urls == [], str(requested_urls))
    check("(5) host_is_public itself rejects a private address", host_is_public("boe.es", resolve=_resolver_private) is False)
    check("(5) host_is_public itself accepts a public address", host_is_public("boe.es", resolve=_resolver_public) is True)

    # --- (6) no query string, no body, on the successful request ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/pagina", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(6) fetch succeeded", result.status == 200 and result.error is None, repr(result))
    check("(6) request URL carries no query string", "?" not in requested_urls[0], requested_urls[0])
    import inspect
    sig = inspect.signature(fetch_source)
    check("(6) fetch_source has no question/text parameter at all", "question" not in sig.parameters and "text" not in sig.parameters, str(sig))

    # --- (7) select_sources never invents a URL, respects the fetch cap ---
    catalogue = [
        {"id": "a", "url": "https://boe.es/a", "title": "Excedencias", "topics": ["excedencia", "excedencias"]},
        {"id": "b", "url": "https://seg-social.es/b", "title": "IT", "topics": ["incapacidad temporal", "baja"]},
        {"id": "c", "url": "https://boe.es/c", "title": "Vacaciones", "topics": ["vacaciones"]},
    ]
    picked = select_sources("¿qué es una excedencia voluntaria?", catalogue, limit=2)
    check("(7) select_sources picks only catalogue entries", all(p in catalogue for p in picked), str(picked))
    check("(7) select_sources respects the fetch cap", len(picked) <= 2, str(picked))
    check("(7) select_sources matched the relevant topic", any(p["id"] == "a" for p in picked), str(picked))
    picked_none = select_sources("preguntas totalmente ajenas sobre repostería", catalogue, limit=2)
    check("(7) select_sources returns nothing on no overlap", picked_none == [], str(picked_none))

    # --- (8) HTML→text strips script/style and collapses whitespace ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/pagina", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(8) script content stripped", "evil()" not in result.text, result.text)
    check("(8) whitespace collapsed", result.text == "Texto con espacios", repr(result.text))

    # --- (9) a non-2xx response is a failed fetch, never content (found live, step 10) ---
    requested_urls.clear()
    result = fetch_source("https://boe.es/missing", ALLOWED, transport=transport, resolve=_resolver_public)
    check("(9) 404 is an error, not content", result.error == "http_404", repr(result))
    check("(9) no text handed on from an error page", not result.text, repr(result.text))

    windowed_checks(transport)
    tls_checks()
    provider_checks()

    if failures:
        for f in failures:
            print(f)
        return 1
    print("all general-lane fetch checks passed")
    return 0


# ---------------------------------------------------------------------------
# CP-1 fetcher fixes: (a) keyword-windowed excerpts, (c) hub page = no material
# ---------------------------------------------------------------------------
def windowed_checks(_unused_transport) -> None:
    from app.general_lane import MAX_EXCERPT_CHARS, MAX_TEXT_CHARS, windowed_excerpt

    filler = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 40  # ~2.2k chars, no topic term
    deep = (
        "Artículo 46. Excedencias. 1. La EXCEDENCIA podrá ser voluntaria o forzosa. "
        "El trabajador con al menos una antigüedad de un año tiene derecho a excedencia voluntaria."
    )
    page = filler * 150 + deep + filler * 3  # topic sits ~330k chars in: far beyond the 20k prefix
    assert len(page) > 10 * MAX_TEXT_CHARS

    # (10) windows reach text the old 20k prefix never could; accent/case-insensitive.
    ex = windowed_excerpt(page, ["excedencia"])
    check("(10) deep match is captured (prefix cap would have missed it)", "Artículo 46" in ex.text and "voluntaria o forzosa" in ex.text, ex.text[:200])
    check("(10) matching is case-insensitive", ex.matches >= 3, str(ex.matches))
    check("(10) overlapping hits merge into ONE window", ex.windows == 1, str(ex.windows))
    check("(10) excerpt is bounded (nowhere near the page)", len(ex.text) < 3_000, str(len(ex.text)))
    check("(10) no unrelated filler far from a match", ex.text.count("Lorem ipsum") < 30, str(ex.text.count("Lorem ipsum")))
    check("(10) never cut mid-word", not ex.text.startswith(("orem", "psum")) and not ex.text.endswith(("Lore", "ips")), repr(ex.text[:12] + ".." + ex.text[-12:]))
    check("(10) accent-insensitive: 'baja medica' finds 'baja médica'", windowed_excerpt("Sobre la BAJA MÉDICA y su duración.", ["baja medica"]).windows == 1)
    check("(10) short term 'it' needs a word boundary ('item'/'italia' don't count)", windowed_excerpt("item italia digital", ["it"]).windows == 0)
    check("(10) short term 'it' matches the word 'IT'", windowed_excerpt("La IT es una situación", ["it"]).windows == 1)
    check("(10) plural stem matches ('excedencia' finds 'excedencias')", windowed_excerpt("varias excedencias", ["excedencia"]).windows == 1)

    # Every window around every match (not just the first), total capped, priority-first.
    spaced = ""
    for i in range(30):
        spaced += (f"bloque {i} sobre EXCEDENCIA. " + "relleno neutro sin tema " * 120)
        spaced += f"bloque {i} sobre permisos retribuidos. " + "relleno neutro sin tema " * 120
    all_windows = windowed_excerpt(spaced, ["excedencia", "permisos"])
    check("(10) all windows around matches are kept when they fit the budget or are capped", all_windows.windows >= 1 and len(all_windows.text) <= MAX_EXCERPT_CHARS, f"{all_windows.windows} {len(all_windows.text)}")
    check("(10) total excerpt never exceeds the cap", len(all_windows.text) <= MAX_EXCERPT_CHARS, str(len(all_windows.text)))
    prio = windowed_excerpt(spaced, ["excedencia", "permisos"], priority_terms=["permisos"], max_chars=3_000)
    check("(10) under a tight budget the question's terms win", "permisos retribuidos" in prio.text and len(prio.text) <= 3_000, f"{len(prio.text)} {prio.matched_terms}")
    check("(10) matched terms are reported", "permisos" in prio.matched_terms, str(prio.matched_terms))
    check("(10) no match -> empty excerpt", windowed_excerpt(filler, ["excedencia"]).windows == 0)

    # (11) end to end through fetch_source with a body far larger than the old cap.
    body = ("<html><body><nav>Inicio | Servicios | Contacto</nav><p>" + (filler * 120) + "</p><p>" + deep + "</p></body></html>").encode()
    assert len(body) < 1_400_000
    big = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=body))
    r = fetch_source("https://boe.es/estatuto", ALLOWED, topics=["excedencia", "excedencias"], transport=big, resolve=_resolver_public)
    check("(11) windowed fetch reaches an article 250k chars deep", r.error is None and "voluntaria o forzosa" in r.text, repr(r.error))
    check("(11) fetch reports full-page size vs excerpt (trace bookkeeping)", r.text_chars > 100_000 and 0 < len(r.text) < 5_000 and r.windows == 1, f"{r.text_chars} {len(r.text)} {r.windows}")
    check("(11) matched terms surfaced", r.matched_terms == ["excedencia", "excedencias"], str(r.matched_terms))
    legacy = fetch_source("https://boe.es/estatuto", ALLOWED, transport=big, resolve=_resolver_public)
    check("(11) without topics the legacy bounded prefix is returned (and misses the article)", len(legacy.text) == MAX_TEXT_CHARS and "voluntaria o forzosa" not in legacy.text)

    # (12) a generic hub page (reachable, but no topic term) is NO material.
    hub = b"<html><body><h1>Portal del Ciudadano</h1><nav>Prestaciones | Pensiones | Empleo | Salud</nav><p>Bienvenido al portal.</p></body></html>"
    hub_t = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=hub))
    r = fetch_source("https://seg-social.es/hub", ALLOWED, topics=["incapacidad temporal", "baja medica", "it"], transport=hub_t, resolve=_resolver_public)
    check("(12) hub page with no topic term is an error, not content", r.error == "no_topic_match" and r.text == "", repr(r))
    check("(12) the fetch itself succeeded (status kept for the trace)", r.status == 200 and r.bytes > 0, repr(r))
    thin = b"<html><body><p>La baja medica se tramita en el centro de salud.</p></body></html>"
    thin_t = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=thin))
    r = fetch_source("https://seg-social.es/thin", ALLOWED, topics=["incapacidad temporal", "baja medica"], transport=thin_t, resolve=_resolver_public)
    check("(12) a page with at least one topic term in the excerpt passes", r.error is None and "baja medica" in r.text, repr(r))


# ---------------------------------------------------------------------------
# CP-1 fetcher fix (b): TLS chain trust — certifi + bundled intermediates.
# ---------------------------------------------------------------------------
FNMT_INTERMEDIATE_SHA256 = "F0:38:42:1F:07:F2:0D:63:A2:0D:36:91:E5:A1:78:AB:84:59:EB:E5:70:C1:64:7B:76:90:55:4E:F2:38:76:AB"


def _pem_fingerprint(path) -> str:
    import hashlib
    import ssl

    der = ssl.PEM_cert_to_DER_cert(open(path).read())
    h = hashlib.sha256(der).hexdigest().upper()
    return ":".join(h[i:i + 2] for i in range(0, len(h), 2))


def tls_checks() -> None:
    import shutil
    import socket
    import ssl
    import subprocess
    import tempfile
    import threading
    from pathlib import Path

    import certifi

    from app.general_lane import CERTS_DIR, build_ssl_context

    ctx = build_ssl_context()
    check("(13) verification is never disabled (CERT_REQUIRED + check_hostname)", ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True)

    # The mites.gob.es fix: FNMT's server sends only the leaf; the missing
    # intermediate ships in app/certs/ and chains to a certifi (Mozilla) root.
    pem = CERTS_DIR / "fnmt-ac-componentes-informaticos.pem"
    check("(13) bundled FNMT intermediate is present", pem.is_file(), str(pem))
    if pem.is_file():
        check("(13) bundled intermediate is the pinned FNMT certificate", _pem_fingerprint(pem) == FNMT_INTERMEDIATE_SHA256, _pem_fingerprint(pem))
        subjects = [str(c.get("subject")) for c in ctx.get_ca_certs()]
        check("(13) intermediate is loaded into the context's store", any("AC Componentes" in s for s in subjects))
    check("(13) the intermediate's root (AC RAIZ FNMT-RCM) is in certifi", "AC RAIZ FNMT-RCM" in open(certifi.where()).read())

    # General mechanism, proven on a synthetic chain (needs the openssl CLI).
    if shutil.which("openssl") is None:
        print("(13) SKIP synthetic chain handshake: openssl CLI not available")
        return
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        def sh(*args: str) -> None:
            subprocess.run(args, cwd=d, check=True, capture_output=True)

        (d / "ca.cnf").write_text("basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n")
        (d / "leaf.cnf").write_text("basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n")
        sh("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "root.key", "-out", "root.pem", "-subj", "/CN=Test Root", "-days", "2", "-extensions", "v3_ca", "-config", str(_minimal_cnf(d)))
        sh("openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", "int.key", "-out", "int.csr", "-subj", "/CN=Test Intermediate")
        sh("openssl", "x509", "-req", "-in", "int.csr", "-CA", "root.pem", "-CAkey", "root.key", "-CAcreateserial", "-out", "int.pem", "-days", "2", "-extfile", "ca.cnf")
        sh("openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", "leaf.key", "-out", "leaf.csr", "-subj", "/CN=localhost")
        sh("openssl", "x509", "-req", "-in", "leaf.csr", "-CA", "int.pem", "-CAkey", "int.key", "-CAcreateserial", "-out", "leaf.pem", "-days", "2", "-extfile", "leaf.cnf")

        server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_ctx.load_cert_chain(str(d / "leaf.pem"), str(d / "leaf.key"))  # LEAF ONLY: the misconfigured-server shape
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        port = srv.getsockname()[1]

        def serve() -> None:
            for _ in range(3):
                try:
                    srv.settimeout(20)
                    conn, _addr = srv.accept()
                    with server_ctx.wrap_socket(conn, server_side=True) as tls:
                        tls.recv(1)
                except (ssl.SSLError, OSError):
                    pass

        threading.Thread(target=serve, daemon=True).start()

        def handshake(client_ctx: ssl.SSLContext) -> str:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
                    with client_ctx.wrap_socket(raw, server_hostname="localhost") as tls:
                        tls.send(b"x")
                return "ok"
            except ssl.SSLCertVerificationError as exc:
                return "verify_failed: " + str(exc.verify_message)

        no_int = build_ssl_context(extra_cert_dir=None, cafile=str(d / "root.pem"))
        res = handshake(no_int)
        check("(13) leaf-only server FAILS verification without the intermediate (the mites.gob.es bug)", res.startswith("verify_failed"), res)

        (d / "certs").mkdir()
        (d / "certs" / "int.pem").write_text((d / "int.pem").read_text())
        with_int = build_ssl_context(extra_cert_dir=d / "certs", cafile=str(d / "root.pem"))
        res = handshake(with_int)
        check("(13) same server VERIFIES once the intermediate is in the store", res == "ok", res)

        # A different, untrusted root must still be rejected: adding an
        # intermediate never loosens verification to 'accept anything'.
        sh("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "other.key", "-out", "other.pem", "-subj", "/CN=Other Root", "-days", "2", "-extensions", "v3_ca", "-config", str(_minimal_cnf(d)))
        wrong = build_ssl_context(extra_cert_dir=d / "certs", cafile=str(d / "other.pem"))
        res = handshake(wrong)
        check("(13) an untrusted root is still rejected", res.startswith("verify_failed"), res)
        srv.close()


# ---------------------------------------------------------------------------
# Step 11 finding: the REAL ClaudeProvider.general_knowledge() had never run —
# `GeneralKnowledgeResult` was not imported in claude.py, so every live call
# died with NameError -> provider_error -> "unavailable" (the step-9 tests
# stopped at the endpoint/provider seam with mocks). This drives the actual
# method with a fake `anthropic` client so the seam itself is covered.
# ---------------------------------------------------------------------------
def provider_checks() -> None:
    import types

    from app.providers.base import GeneralKnowledgeResult, ProviderConfig
    from app.providers.claude import ClaudeProvider

    seen: dict = {}

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Resp:
        stop_reason = "end_turn"

        def __init__(self, text):
            self.content = [_Block(text)]
            self.usage = types.SimpleNamespace(input_tokens=11, output_tokens=7)

    def fake_module(reply):
        class _Messages:
            def create(self, **kwargs):
                seen["kwargs"] = kwargs
                return _Resp(reply)

        class _Anthropic:
            def __init__(self, **kwargs):
                self.messages = _Messages()

        return types.SimpleNamespace(Anthropic=_Anthropic)

    cfg = ProviderConfig(provider="claude", model="claude-sonnet-5", endpoint=None)
    excerpts = [{"id": "sepe-x", "title": "SEPE", "text": "El periodo de prueba es un tramo inicial del contrato."}]

    saved = sys.modules.get("anthropic")
    try:
        sys.modules["anthropic"] = fake_module('{"answer": "Es un tramo inicial del contrato.", "sources_used": ["sepe-x"]}')
        res = ClaudeProvider().general_knowledge("¿Qué es el periodo de prueba?", excerpts, "sk-test", cfg)
        check("(14) real provider method returns a GeneralKnowledgeResult (no NameError)", isinstance(res, GeneralKnowledgeResult), repr(res))
        check("(14) web source carried with its excerpt", res.sources and res.sources[0]["kind"] == "web" and "tramo inicial" in res.sources[0]["excerpt"], repr(res.sources))
        check("(14) excerpts reach the model prompt", "EXTRACTOS DISPONIBLES" in seen["kwargs"]["messages"][0]["content"], "")
        check("(14) no temperature param sent", "temperature" not in seen["kwargs"], "")

        sys.modules["anthropic"] = fake_module('{"answer": "Un ERTE es un procedimiento.", "sources_used": []}')
        res = ClaudeProvider().general_knowledge("¿Qué es un ERTE?", [], "sk-test", cfg)
        check("(14) no excerpts -> model_knowledge source", res.sources == [{"kind": "model_knowledge", "title": "conocimiento general"}], repr(res.sources))

        sys.modules["anthropic"] = fake_module("not json at all")
        res = ClaudeProvider().general_knowledge("¿Qué es un ERTE?", [], "sk-test", cfg)
        check("(14) unparseable output -> empty answer, parse_error traced", res.answer == "" and res.trace_fragment.get("parse_error") is True, repr(res))
    finally:
        if saved is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = saved


def _minimal_cnf(d):
    """openssl.cnf with a v3_ca section (LibreSSL/OpenSSL-portable)."""
    p = d / "openssl-min.cnf"
    p.write_text(
        "[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=x\n"
        "[v3_ca]\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n"
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
