import re
from collections import Counter

from pypdf import PdfReader

# Table-of-contents lines end in leader dots: "1.1 About this Guide. . . . . 4"
LEADER_DOTS = re.compile(r"(?:\.\s?){4,}")

# Running headers/footers are looked for in the last lines of each page
FOOTER_LINES = 6


def _clean(text):
    # Soft hyphens split words across lines: "Opti\xad\nmizer" -> "Optimizer"
    text = text.replace("\xad\n", "").replace("\xad", "")
    # SAP Note links leave a stray period on its own line: "see 617547\n ." -> "see 617547."
    return re.sub(r"\n\s*\.(?=\s|$)", ".", text)


def _footer_key(line):
    # Page numbers differ on every page, so compare footer lines without digits
    return re.sub(r"\d+", "#", line.strip())


def extract_pages(pdf):
    """Return [(page_number, text)] for a PDF, without running footers or TOC leader lines.

    pdf is a file path or a binary file object.
    """
    reader = PdfReader(pdf)
    pages = [_clean(page.extract_text() or "").split("\n") for page in reader.pages]

    # A line that shows up at the bottom of several pages is a running footer,
    # e.g. "C U S T O M E R", "© 2015 SAP SE ... All rights reserved. 11", the chapter name
    counts = Counter(
        key
        for lines in pages
        for key in {_footer_key(line) for line in lines[-FOOTER_LINES:] if line.strip()}
    )
    footers = {key for key, n in counts.items() if n >= 3}

    result = []
    for number, lines in enumerate(pages, start=1):
        tail = [line for line in lines[-FOOTER_LINES:] if _footer_key(line) not in footers]
        body = [
            line for line in lines[:-FOOTER_LINES] + tail
            if line.strip() and not LEADER_DOTS.search(line)
        ]
        result.append((number, "\n".join(body)))
    return result
