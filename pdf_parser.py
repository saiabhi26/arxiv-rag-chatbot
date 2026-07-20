"""Turn a paper PDF into clean, section-labelled text.

Knows nothing about arXiv — it takes a path to a PDF and returns
`[{section, text}, ...]`. `scraper.py` is what feeds it.

Extracting a research paper's body text is the fiddliest part of this project,
because a PDF describes *ink on a page*, not a document structure. Three
problems have to be solved before the text is usable, and each one fails
silently rather than loudly:

  1. **Reading order.** Default extraction reads top-to-bottom, which shuffles
     the two columns of an ACL/ICML paper together.
  2. **Headings.** They arrive split across text blocks ("2.1" on one line, the
     title on the next), or unnumbered and identifiable *only* by their font.
  3. **Furniture.** Page numbers and running headers are interleaved with the
     prose, and look exactly like section headings once you start guessing.
"""

import re
from collections import Counter, namedtuple

import fitz  # PyMuPDF

# Sections that are citation dumps or boilerplate rather than content. The old
# Wikipedia corpus was ~30% navigation junk ("See also", "External links",
# reference lists), which polluted retrieval; papers have the same disease.
JUNK_SECTIONS = re.compile(
    r"^(references?|bibliography|acknowledge?ments?)\b", re.IGNORECASE
)

# A heading with its number attached: "3 Method", "3.1 Encoder".
NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+([A-Z][^\n]{2,70}?)\s*$")

# A line holding *only* a section number. Typesetters routinely put the number
# and the title in separate PDF text blocks, so these arrive alone on a line
# and have to be stitched back onto the title that follows.
NUMBER_ONLY = re.compile(r"^(\d+)((?:\.\d+)*)\.?$")

# Unnumbered headings that appear in nearly every paper, whatever the template.
BARE_HEADING = re.compile(
    r"^\s*(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"experiments?|results?|discussion|conclusions?|limitations?|"
    r"references?|bibliography|acknowledge?ments?|appendix)\s*$",
    re.IGNORECASE,
)

CAPTION = re.compile(r"^(figure|table|algorithm|eq(uation)?)\s*\d", re.IGNORECASE)

# Author bylines are set bold, sit above the first real heading, and are short —
# so they sail through every other test and get mistaken for section headings,
# and the paper's whole body then accumulates under a person's name. Commas and
# affiliation markers are what give them away; note "and" alone does not, since
# plenty of real headings read "Conclusion and Future Work".
AUTHOR_LINE = re.compile(r",|\bIEEE\b|\bUniversity\b|\bMember\b|\bInstitute\b", re.IGNORECASE)

MIN_SECTION_WORDS = 30
FRONT_MATTER = "__front_matter__"

# A line of a PDF, carrying the typography that identifies headings.
Line = namedtuple("Line", "text size bold")


def extract_sections(pdf_path):
    """Split a paper's PDF into [{section, text}], dropping junk sections."""
    doc = fitz.open(pdf_path)
    pages = [page_lines(page) for page in doc]
    doc.close()

    pages = drop_running_headers(pages)
    lines = rejoin_split_headings([line for page in pages for line in page])
    lines = [line for line in lines if not is_noise(line.text)]
    body_size = body_font_size(lines)

    sections = []
    # Everything before the first heading is the title/author block — metadata
    # we already hold per-record. FRONT_MATTER is dropped on flush.
    current = FRONT_MATTER
    buffer = []

    def flush():
        if not buffer or current == FRONT_MATTER:
            return
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        text = re.sub(r"(\w)-\s(\w)", r"\1\2", text)  # rejoin hyphenated line breaks
        if len(text.split()) >= MIN_SECTION_WORDS and is_prose(text):
            sections.append({"section": current, "text": text})

    for line in lines:
        heading = heading_of(line, body_size)
        if heading is None:
            buffer.append(line.text)
            continue

        # The bibliography runs to the end of the paper, and everything from
        # there on is citations. Stop.
        if JUNK_SECTIONS.match(heading):
            flush()
            return sections

        flush()
        current = heading
        buffer = []

    flush()
    return sections


def page_lines(page):
    """Extract a page's lines, in human reading order, with their typography.

    PyMuPDF's default extraction reads strictly top-to-bottom, which interleaves
    the two columns of an ACL/ICML-style paper into nonsense. So we pull text
    blocks with coordinates and, when they separate cleanly into a left and a
    right half, read the left column fully before the right.
    """
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    if not blocks:
        return []

    mid = page.rect.width / 2
    left = [b for b in blocks if b["bbox"][2] <= mid + 20]   # ends before the midline
    right = [b for b in blocks if b["bbox"][0] >= mid - 20]  # starts after the midline
    two_column = len(left) + len(right) > 0.8 * len(blocks) and left and right

    if two_column:
        blocks.sort(key=lambda b: (
            0 if (b["bbox"][0] + b["bbox"][2]) / 2 < mid else 1, b["bbox"][1]
        ))
    else:
        blocks.sort(key=lambda b: b["bbox"][1])

    lines = []
    for block in blocks:
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            # Carry the typography through: it is the only thing that marks an
            # *unnumbered* heading ("Introduction", "Background") as a heading
            # rather than a short sentence. The text alone cannot tell you.
            lines.append(Line(
                text=clean_text("".join(s["text"] for s in spans)),
                size=max(s["size"] for s in spans),
                bold=any("bold" in s["font"].lower() for s in spans),
            ))
    return lines


def clean_text(text):
    """Strip the artifacts that PDF text extraction reliably introduces."""
    text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text.strip()


def is_noise(text):
    """Page furniture: page numbers and the arXiv margin stamp."""
    if not text:
        return True
    if re.fullmatch(r"\d{1,3}", text):  # bare page number
        return True
    if text.startswith("arXiv:"):  # the vertical stamp down the first page
        return True
    return False


def drop_running_headers(pages):
    """Remove the running header/footer repeated on most pages.

    These have to go *before* heading detection: a running header sits directly
    against the page number, so "7" + "Takahashi et al. | PHINN-EEG" is
    indistinguishable from a numbered section heading — and page numbers count
    up just like section numbers do, so the ordering check in
    rejoin_split_headings() can't save us either. Repetition across pages is
    what actually identifies furniture.
    """
    if len(pages) < 3:
        return pages

    # Count each distinct line once per page, so a phrase repeated *within* a
    # page doesn't look like furniture.
    seen = Counter(
        text for page in pages for text in {line.text for line in page if line.text}
    )
    threshold = max(3, int(0.3 * len(pages)))
    repeated = {text for text, count in seen.items() if count >= threshold}

    return [[line for line in page if line.text not in repeated] for page in pages]


def rejoin_split_headings(lines):
    """Stitch number-only lines back onto the heading title that follows.

    The ambiguity: a lone "3" is either a section number or a page number, and
    nothing about the line itself distinguishes them. So we only accept a bare
    integer that is the next section number we expect — a paper's top-level
    sections run 1, 2, 3 in order. This is a second line of defence behind
    drop_running_headers().
    """
    joined = []
    next_top_level = 1
    consumed = set()

    for i, line in enumerate(lines):
        if i in consumed:
            continue

        match = NUMBER_ONLY.match(line.text)
        if not match:
            joined.append(line)
            continue

        top_level, sub = int(match.group(1)), match.group(2)

        # The title is the next non-empty line.
        title_index = next(
            (j for j in range(i + 1, min(i + 3, len(lines))) if lines[j].text),
            None,
        )
        title = lines[title_index] if title_index is not None else None

        believable = title is not None and looks_like_title(title.text) and (
            # "3.1" — a subsection of whichever section we're currently inside.
            top_level == next_top_level - 1 if sub
            # "3" — only believable as the top-level section we expect next.
            else top_level == next_top_level
        )
        if not believable:
            joined.append(line)
            continue

        if not sub:
            next_top_level += 1
        joined.append(Line(
            text=f"{top_level}{sub} {title.text}",
            size=title.size,
            bold=title.bold,
        ))
        consumed.add(title_index)

    return joined


def looks_like_title(text):
    """Could this text be a heading title?"""
    if not (2 < len(text) <= 70) or not text[0].isupper():
        return False
    if text.endswith((".", ",", ";", ":")) or NUMBER_ONLY.match(text):
        return False
    if len(text.split()) > 12:  # headings are short; sentences are not
        return False
    if any(c in text for c in "*=$|∗†"):  # math fragments and author footnotes
        return False
    if CAPTION.match(text):  # "Figure 3", "Table 1", "Algorithm 2"
        return False
    if AUTHOR_LINE.search(text):
        return False
    # Headings are words. Equations and captions are not.
    letters = sum(c.isalpha() or c.isspace() or c == "-" for c in text)
    return letters / len(text) >= 0.85


def is_prose(text):
    """Is this mostly sentences, or is it a mangled equation/table dump?

    Extraction turns dense math into character soup. Those chunks are unretrievable
    noise: they match nothing, and they take up space in the index.
    """
    return sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1) >= 0.75


def body_font_size(lines):
    """The paper's body-text size — whatever size most of its prose is set in.

    Inferred per paper rather than hardcoded, because it varies by template.
    """
    sizes = Counter(
        round(line.size, 1) for line in lines if len(line.text.split()) >= 5
    )
    return sizes.most_common(1)[0][0] if sizes else 0.0


def heading_of(line, body_size):
    """The section title this line announces, or None if it isn't a heading."""
    match = NUMBERED_HEADING.match(line.text) or BARE_HEADING.match(line.text)
    if match:
        return normalize_heading(match.group(1))

    # Unnumbered headings ("Introduction", "Background") in AISTATS/JMLR-style
    # papers are marked *only* by typography — nothing in the text distinguishes
    # them from a short sentence. Bold or oversized short lines are headings.
    styled = line.bold or line.size > body_size + 0.5
    if styled and looks_like_title(line.text):
        return normalize_heading(line.text)

    return None


def normalize_heading(text):
    """Papers set headings in ALL CAPS, Title Case, or Sentence case.

    These strings become visible citation text ("{title} — {section}"), so
    INTRODUCTION and Introduction should not be two different sections.
    """
    text = text.strip()
    return text.title() if text.isupper() else text
