"""PDF extraction: per-page text + per-page image (ADR-0010).

Input is the S3 key of an uploaded original PDF. We read it from S3, extract
text per page and render each page to a JPEG written back to S3, and return the
per-page data. hr-backend writes the documents/document_pages rows from the
response — hr-ai never touches the database.

Sprint 1 was PDF-only: an image-only (scanned) page yielded empty text and
stayed that way (the page image was still produced so the source view worked,
and hr-backend visibly flagged a document whose text was entirely empty).

Sprint 7e (ADR-0026) adds the OCR fallback's *marker* only — never the OCR call
itself. When `ocr=True`, a text-less page (up to `ocr_page_cap`, counted in
page order) is flagged `extraction_source="ocr_pending"` instead of the default
`"text_layer"`; every page WITH text is always `"text_layer"`, `ocr` or not.
This endpoint still never calls the vision model — that would make a
multi-page scan's `/extract` request run minutes long inside a single HTTP
request/response cycle (measured 9-90 s/page, review.md §1.3/§1.5). hr-backend
reads `extraction_source="ocr_pending"` off this response and dispatches the
queued job that actually drives the OCR calls (review.md §2.1) — `/extract`'s
own latency is completely unchanged by the `ocr` flag.
"""

import fitz  # PyMuPDF

from .config import settings
from .storage import get_object_bytes, put_object_bytes


def page_image_key(document_uuid: str, page_number: int) -> str:
    return f"documents/{document_uuid}/pages/{page_number:04d}.jpg"


def extract_pdf(
    storage_key: str,
    document_uuid: str,
    ocr: bool = False,
    ocr_page_cap: int = 60,
) -> dict:
    pdf_bytes = get_object_bytes(storage_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = settings.extract_image_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pages = []
        ocr_pending_count = 0
        for index in range(doc.page_count):
            page = doc.load_page(index)
            page_number = index + 1

            text = page.get_text("text") or ""
            text = text.strip()

            pixmap = page.get_pixmap(matrix=matrix)
            image_key = page_image_key(document_uuid, page_number)
            put_object_bytes(image_key, pixmap.tobytes("jpeg"), "image/jpeg")

            extraction_source = "text_layer"
            if text == "" and ocr and ocr_pending_count < ocr_page_cap:
                extraction_source = "ocr_pending"
                ocr_pending_count += 1

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "image_key": image_key,
                    "extraction_source": extraction_source,
                }
            )

        return {"page_count": doc.page_count, "pages": pages}
    finally:
        doc.close()
