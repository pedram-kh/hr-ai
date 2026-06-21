"""PDF extraction: per-page text + per-page image (ADR-0010).

Input is the S3 key of an uploaded original PDF. We read it from S3, extract
text per page and render each page to a JPEG written back to S3, and return the
per-page data. hr-backend writes the documents/document_pages rows from the
response — hr-ai never touches the database.

Sprint 1 is PDF-only. Image-only (scanned) pages yield empty text (no OCR this
sprint); the page image is still produced so the source view works, and
hr-backend visibly flags a document whose text is entirely empty.
"""

import fitz  # PyMuPDF

from .config import settings
from .storage import get_object_bytes, put_object_bytes


def page_image_key(document_uuid: str, page_number: int) -> str:
    return f"documents/{document_uuid}/pages/{page_number:04d}.jpg"


def extract_pdf(storage_key: str, document_uuid: str) -> dict:
    pdf_bytes = get_object_bytes(storage_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = settings.extract_image_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pages = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            page_number = index + 1

            text = page.get_text("text") or ""

            pixmap = page.get_pixmap(matrix=matrix)
            image_key = page_image_key(document_uuid, page_number)
            put_object_bytes(image_key, pixmap.tobytes("jpeg"), "image/jpeg")

            pages.append(
                {
                    "page_number": page_number,
                    "text": text.strip(),
                    "image_key": image_key,
                }
            )

        return {"page_count": doc.page_count, "pages": pages}
    finally:
        doc.close()
