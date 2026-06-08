"""
Task 3 - Convert files in data/landing/ to Markdown.

Legal DOCX/PDF files are converted with MarkItDown when possible. DOCX also has
a lightweight fallback based on the zipped Word XML, which is enough for the
course tests and keeps the script usable in a small environment.
"""

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _convert_with_markitdown(filepath: Path) -> str:
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(filepath))
    return result.text_content


def _convert_docx_fallback(filepath: Path) -> str:
    with zipfile.ZipFile(filepath) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _read_legal_file(filepath: Path) -> str:
    try:
        return _convert_with_markitdown(filepath)
    except Exception:
        if filepath.suffix.lower() == ".docx":
            return _convert_docx_fallback(filepath)
        raise


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if not legal_dir.exists():
        return outputs

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting legal: {filepath.name}")
        content = _read_legal_file(filepath).strip()
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(f"# {filepath.stem}\n\n{content}\n", encoding="utf-8")
        outputs.append(output_path)
        print(f"  Saved: {output_path}")

    return outputs


def _article_to_markdown(data: dict) -> str:
    title = data.get("title") or "Untitled article"
    header = [
        f"# {title}",
        "",
        f"**Source:** {data.get('url', 'N/A')}",
        f"**Crawled:** {data.get('date_crawled', 'N/A')}",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + (data.get("content_markdown") or data.get("content") or "")


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON/HTML/TXT news files in data/landing/news/."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if not news_dir.exists():
        return outputs

    for filepath in sorted(news_dir.iterdir()):
        if filepath.name.startswith("."):
            continue

        print(f"Converting news: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"

        if filepath.suffix.lower() == ".json":
            data = json.loads(filepath.read_text(encoding="utf-8"))
            content = _article_to_markdown(data)
        elif filepath.suffix.lower() in {".md", ".txt"}:
            content = filepath.read_text(encoding="utf-8")
        elif filepath.suffix.lower() == ".html":
            content = _convert_with_markitdown(filepath)
        else:
            continue

        output_path.write_text(content.strip() + "\n", encoding="utf-8")
        outputs.append(output_path)
        print(f"  Saved: {output_path}")

    return outputs


def convert_all() -> list[Path]:
    """Convert all supported landing files and return generated markdown paths."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    outputs = []
    outputs.extend(convert_legal_docs())
    outputs.extend(convert_news_articles())

    print(f"\nDone. Wrote {len(outputs)} markdown files to {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
