"""
Task 2 - Crawl news articles about artists related to drug cases.

The assignment asks for at least 5 articles. This repo uses 7 URLs so one weak
article/file will not drop the task below the threshold.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Create data/landing/news/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_SPECS = [
    (
        "01_chi_dan.json",
        "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    ),
    (
        "02_chu_bin.json",
        "https://vnexpress.net/ca-si-chu-bin-bi-tam-giu-vi-lien-quan-ma-tuy-4755275.html",
    ),
    (
        "03_huu_tin.json",
        "https://vnexpress.net/dien-vien-hai-huu-tin-bi-de-nghi-truy-to-7-15-nam-tu-4530802.html",
    ),
    (
        "04_le_hang.json",
        "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    ),
    (
        "05_nhikolai_dinh.json",
        "https://ngoisao.vnexpress.net/nam-than-lai-nga-nhikolai-dinh-bi-bat-4762594.html",
    ),
    (
        "06_andrea_aybar.json",
        "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    ),
    (
        "07_binh_gold.json",
        "https://cuoi.tuoitre.vn/rapper-nhieu-tat-binh-gold-vua-bi-bat-vi-duong-tinh-ma-tuy-lang-lach-tren-cao-toc-20250724092146502.htm",
    ),
]
ARTICLE_URLS = [url for _, url in ARTICLE_SPECS]


class TextExtractor(HTMLParser):
    """Small fallback extractor for article-like HTML."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        text = " ".join(unescape(data).split())
        if not text:
            return
        if self._in_title:
            self.title += text
        elif not self._skip_depth and len(text) > 30:
            self.parts.append(text)


def _slugify(url: str, index: int) -> str:
    stem = re.sub(r"^https?://", "", url).split("?")[0].rstrip("/")
    stem = Path(stem).stem or f"article_{index:02d}"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower()
    return stem or f"article_{index:02d}"


def _html_to_markdown(html: str, url: str) -> dict:
    parser = TextExtractor()
    parser.feed(html)
    title = (
        parser.title.replace(" - VnExpress", "")
        .replace(" - Tuoi Tre Online", "")
        .strip()
    )
    body = "\n\n".join(dict.fromkeys(parser.parts))
    return {
        "url": url,
        "title": title or "Untitled article",
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": f"# {title or 'Untitled article'}\n\n{body}",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return metadata plus markdown content.

    Crawl4AI is used when available. A requests + HTMLParser fallback keeps this
    script usable in a small local environment.
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            metadata = getattr(result, "metadata", {}) or {}
            return {
                "url": url,
                "title": metadata.get("title") or "Untitled article",
                "date_crawled": datetime.now(timezone.utc).isoformat(),
                "content_markdown": getattr(result, "markdown", "") or "",
            }
    except Exception:
        import requests

        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Day08RAG/1.0)"},
        )
        response.raise_for_status()
        return _html_to_markdown(response.text, url)


async def crawl_all():
    """Crawl all URLs in ARTICLE_URLS and save one JSON file per article."""
    setup_directory()

    for i, (filename, url) in enumerate(ARTICLE_SPECS, 1):
        print(f"[{i}/{len(ARTICLE_SPECS)}] Crawling: {url}")
        filepath = DATA_DIR / filename
        try:
            article = await crawl_article(url)
        except Exception as exc:
            if filepath.exists():
                print(f"  Skipped after error, kept existing file: {exc}")
                continue
            raise
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
