import re

# Numbered section headings such as "3.2.3.1 Data Growth and Data Archiving Monitors in EHS"
HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z].{2,120})$")

# Sentence boundary: end punctuation followed by the start of a new sentence or bullet
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z●(\"'])")


def split_sections(pages):
    """Group the document's lines under their numbered section headings.

    pages is a list of (page_number, text). Returns a list of
    {"title": str, "lines": [(page_number, line), ...]} in document order.
    """
    sections = [{"title": "Front matter", "lines": []}]
    last_number = ()

    for page, text in pages:
        # pypdf marks soft-wrapped lines with a trailing space; rejoin them so
        # sentences and wrapped headings are not cut at a line break, but never
        # glue a heading onto the end of the previous line
        text = re.sub(r"[ \t]+\n(?!\d{1,2}(?:\.\d{1,2}){0,3}\s+[A-Z])", " ", text)

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            match = HEADING.match(line)
            if match:
                number = tuple(int(n) for n in match.group(1).split("."))
                # Headings must advance in order (3.1.9 -> 3.2 -> 4), which filters out
                # table rows and stray numbers that only look like headings
                previous_chapter = last_number[0] if last_number else 0
                if number > last_number and number[0] - previous_chapter <= 1:
                    last_number = number
                    title = re.sub(r"\s+", " ", f"{match.group(1)} {match.group(2)}")
                    sections.append({"title": title, "lines": []})
                    continue

            sections[-1]["lines"].append((page, line))

    return [s for s in sections if s["lines"]]


def _split_units(lines, count_tokens, budget):
    """Break section lines into sentences; table rows and bullets stay separate lines."""
    units = []

    for page, line in lines:
        for i, sentence in enumerate(SENTENCE_END.split(line)):
            pieces = [sentence]

            # A single sentence or table row longer than the budget is split by words
            if count_tokens(sentence) > budget:
                pieces, piece = [], []
                for word in sentence.split():
                    if piece and count_tokens(" ".join(piece + [word])) > budget:
                        pieces.append(" ".join(piece))
                        piece = []
                    piece.append(word)
                pieces.append(" ".join(piece))

            for j, piece in enumerate(pieces):
                units.append({
                    "page": page,
                    "text": piece,
                    "tokens": count_tokens(piece),
                    "new_line": i == 0 and j == 0,
                })

    return units


def _overlap(units, overlap_tokens):
    """Return the trailing whole sentences of a chunk that fit in overlap_tokens."""
    carried, total = [], 0
    for unit in reversed(units):
        if total + unit["tokens"] > overlap_tokens:
            break
        carried.insert(0, unit)
        total += unit["tokens"]
    return carried


def _make_chunk(title, units):
    body = ""
    for unit in units:
        if body:
            body += "\n" if unit["new_line"] else " "
        body += unit["text"]

    return {
        "text": f"{title}\n{body}",
        "section": title,
        "page": units[0]["page"],
        "page_end": units[-1]["page"],
    }


def chunk_document(pages, count_tokens, max_tokens=180, overlap_tokens=40):
    """Split a document into section-aware chunks made of whole sentences.

    Chunks never cross a section boundary. Within a section, sentences, bullets
    and table rows are packed until max_tokens, and each chunk repeats up to
    overlap_tokens of whole sentences from the end of the previous one. Every
    chunk starts with its section title, so both the embedding and the LLM know
    where the text comes from.

    max_tokens must stay below the embedding model's input limit (256 tokens for
    all-MiniLM-L6-v2): anything longer is silently truncated before embedding.
    """
    chunks = []

    for section in split_sections(pages):
        title = section["title"]
        budget = max_tokens - count_tokens(title)
        units = _split_units(section["lines"], count_tokens, budget)

        current = []
        for unit in units:
            if current and sum(u["tokens"] for u in current) + unit["tokens"] > budget:
                chunks.append(_make_chunk(title, current))
                current = _overlap(current, overlap_tokens)
                if sum(u["tokens"] for u in current) + unit["tokens"] > budget:
                    current = []
            current.append(unit)

        if current:
            chunks.append(_make_chunk(title, current))

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index

    return chunks
