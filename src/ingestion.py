# src/ingestion.py
import pymupdf as fitz  # pymupdf
from pathlib import Path
from dataclasses import dataclass
import re

@dataclass
class PageContent:
    doc_name: str
    page_num: int
    raw_text: str

def extract_pdf_pages(pdf_path: Path) -> list[PageContent]:
    """Extract raw text per page from a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")  # "text" mode preserves reading order better than default
        pages.append(PageContent(doc_name=pdf_path.stem, page_num=i, raw_text=text))
    doc.close()
    return pages

def detect_repeated_lines(pages: list[PageContent], threshold: float = 0.6) -> set[str]:
    """Find lines that repeat across many pages — likely headers/footers, not content."""
    from collections import Counter
    line_counts = Counter()
    for page in pages:
        lines = set(line.strip() for line in page.raw_text.split("\n") if line.strip())
        line_counts.update(lines)
    n_pages = len(pages)
    return {line for line, count in line_counts.items() if count / n_pages > threshold}

def clean_page_text(raw_text: str, boilerplate_lines: set[str]) -> str:
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in boilerplate_lines:
            continue
        if re.match(r"^\d+$", stripped):  # standalone page numbers
            continue
        cleaned_lines.append(stripped)

    # Join lines that were likely broken mid-sentence (line doesn't end in punctuation
    # and next line starts lowercase → probably a wrapped sentence, not a new one)
    joined = []
    buffer = ""
    for line in cleaned_lines:
        if buffer and not buffer.endswith((".", ":", "?", "!", ";")) and line[:1].islower():
            buffer += " " + line
        else:
            if buffer:
                joined.append(buffer)
            buffer = line
    if buffer:
        joined.append(buffer)

    return "\n".join(joined)

def is_near_empty(text: str, min_words: int = 15) -> bool:
    return len(text.split()) < min_words


def process_document(pdf_path: Path, output_dir: Path) -> Path:
    pages = extract_pdf_pages(pdf_path)
    boilerplate = detect_repeated_lines(pages)

    cleaned_pages = []
    for page in pages:
        cleaned = clean_page_text(page.raw_text, boilerplate)
        if not is_near_empty(cleaned):
            cleaned_pages.append(cleaned)

    full_text = "\n\n".join(cleaned_pages)
    output_path = output_dir / f"{pdf_path.stem}.txt"
    output_path.write_text(full_text, encoding="utf-8")
    return output_path

def process_all(raw_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in raw_dir.glob("*.pdf"):
        out = process_document(pdf_path, output_dir)
        print(f"Processed {pdf_path.name} → {out.name} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    process_all(Path("data/raw_pdfs"), Path("data/cleaned_text"))

