from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse
import io

from src.data_generator.mode_c.offline import mode_c_offline, mode_c_offline_reason

try:
    import requests
except ModuleNotFoundError:
    class _MissingRequests:
        def get(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests")

    requests = _MissingRequests()


def crawl_and_extract_pages(
    search_results: list[dict[str, Any]],
    web_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Real crawling and real text extraction.
    """
    if mode_c_offline():
        return []

    max_pages = int(web_plan.get("max_pages", 12))
    min_chars = int(web_plan.get("min_extracted_chars", 500))

    pages: list[dict[str, Any]] = []

    for result in search_results:
        if len(pages) >= max_pages:
            break

        page = fetch_and_extract_one(result)
        if page.get("error"):
            continue
        if len(str(page.get("content", ""))) < min_chars:
            continue

        pages.append(page)
        time.sleep(0.25)

    return pages


def fetch_and_extract_one(result: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    url = result["url"]
    source_type = classify_url(url)

    if mode_c_offline():
        return {
            **result,
            "source": "web_page",
            "source_type": source_type,
            "content": "",
            "metadata": {
                "extraction_method": "offline_guard",
            },
            "error": f"Mode C crawling disabled by {mode_c_offline_reason()}.",
        }

    headers = {
        "User-Agent": "Mozilla/5.0 compatible; MLDataAcquisitionBot/0.1"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        status = response.status_code

        if status >= 400:
            return {
                **result,
                "source": "web_page",
                "content": "",
                "metadata": {
                    "http_status": status,
                    "extraction_method": "trafilatura",
                },
                "error": f"HTTP {status}",
            }
        content_type = response.headers.get("content-type", "")

        if source_type == "pdf":
            pdf_text = extract_pdf_text(response.content)

            return {
                "source": "web_asset",
                "source_type": "pdf",
                "url": url,
                "domain": result.get("domain") or urlparse(url).netloc,
                "title": result.get("title", ""),
                "query": result.get("query", ""),
                "snippet": result.get("snippet", ""),
                "content": pdf_text,
                "metadata": {
                    "provider": result.get("provider"),
                    "provider_score": result.get("provider_score"),
                    "http_status": status,
                    "content_type": content_type,
                    "content_chars": len(pdf_text),
                    "extraction_method": "pymupdf",
                    "num_bytes": len(response.content),
                },
                "error": None,
            }

        if source_type == "image":
            return {
                "source": "web_asset",
                "source_type": "image",
                "url": url,
                "domain": result.get("domain") or urlparse(url).netloc,
                "title": result.get("title", ""),
                "query": result.get("query", ""),
                "snippet": result.get("snippet", ""),
                "content": result.get("snippet", ""),
                "metadata": {
                    "provider": result.get("provider"),
                    "provider_score": result.get("provider_score"),
                    "http_status": status,
                    "content_type": content_type,
                    "content_chars": len(str(result.get("snippet", ""))),
                    "extraction_method": "image_metadata_only",
                    "num_bytes": len(response.content),
                    "requires_vision_processing": True,
                },
                "error": None,
            }

        if source_type in {"csv", "json"}:
            text = response.text or ""
            return {
                "source": "web_asset",
                "source_type": source_type,
                "url": url,
                "domain": result.get("domain") or urlparse(url).netloc,
                "title": result.get("title", ""),
                "query": result.get("query", ""),
                "snippet": result.get("snippet", ""),
                "content": text[:20000],
                "metadata": {
                    "provider": result.get("provider"),
                    "provider_score": result.get("provider_score"),
                    "http_status": status,
                    "content_type": content_type,
                    "content_chars": min(len(text), 20000),
                    "extraction_method": "direct_text_asset",
                },
                "error": None,
            }
        html = response.text or ""
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        ) or ""

        content = clean_text(extracted)

        return {
            "source": "web_page",
            "url": url,
            "domain": result.get("domain"),
            "title": result.get("title", ""),
            "query": result.get("query", ""),
            "snippet": result.get("snippet", ""),
            "content": content,
            "metadata": {
                "provider": result.get("provider"),
                "provider_score": result.get("provider_score"),
                "http_status": status,
                "html_chars": len(html),
                "content_chars": len(content),
                "extraction_method": "trafilatura",
            },
            "error": None,
        }

    except Exception as exc:
        return {
            **result,
            "source": "web_page",
            "content": "",
            "metadata": {
                "http_status": None,
                "extraction_method": "trafilatura",
            },
            "error": str(exc)[:300],
        }


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def classify_url(url: str) -> str:
    lower = url.lower().split("?")[0]
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith((".json", ".jsonl")):
        return "json"
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return "image"
    return "html"



def extract_pdf_text(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    chunks = []
    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            chunks.append(f"\n\n--- Page {page_idx + 1} ---\n{text}")
    return clean_text("\n".join(chunks))
