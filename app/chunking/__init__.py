"""Column-aware bilingual extraction + de-spacing + article-aware chunking.

Per ADR-0013: re-extract the original PDF column-aware (PyMuPDF block bboxes)
to separate Euskara (left) from Spanish (right) — never blend languages in a
chunk; normalize the BOG intra-word spacing artifact on the text that gets
embedded; chunk on article boundaries with a size cap. The Sprint-1
`document_pages` text + page images are left UNTOUCHED (citation surface).
"""
