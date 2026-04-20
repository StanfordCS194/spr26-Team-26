from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from ..inspector import assess_artifact_reasonableness, evaluate_artifact_relevance
from ..models import AcquisitionSpec, CollectedArtifact, CollectionSummary, SourceCandidate

SUPPORTED_EXTENSIONS = {
    ".zip",
    ".pdf",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".txt",
    ".md",
    ".xml",
    ".html",
    ".htm",
}

CONTENT_TYPE_EXTENSION_MAP = {
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "text/csv": ".csv",
    "application/json": ".json",
    "application/x-parquet": ".parquet",
    "application/parquet": ".parquet",
    "text/plain": ".txt",
    "text/html": ".html",
}


@dataclass
class CollectorConfig:
    max_sources: int = 6
    max_links_per_source: int = 10
    max_artifacts_per_source: int = 4
    timeout_seconds: int = 25
    max_download_bytes: int = 40 * 1024 * 1024
    relevance_threshold: float = 0.2
    ssl_verify: bool = True
    ca_bundle: str | None = None
    collect_search_portal_candidates: bool = False


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def collect_raw_artifacts(
    spec: AcquisitionSpec,
    ranked_candidates: list[SourceCandidate],
    keyword_bank: list[str],
    output_dir: Path,
    config: CollectorConfig | None = None,
) -> tuple[list[CollectedArtifact], CollectionSummary]:
    config = config or CollectorConfig()
    config = _apply_ssl_env_overrides(config)
    ssl_context = _build_ssl_context(config.ssl_verify, config.ca_bundle)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[CollectedArtifact] = []
    attempted_sources = 0
    seen_artifact_urls: set[str] = set()

    for candidate in ranked_candidates[: config.max_sources]:
        if (
            candidate.collection_method == "search_portal"
            and candidate.source_type != "explicit"
            and not config.collect_search_portal_candidates
        ):
            # Discovery/search pages are noisy; treat as discovery-only for now.
            continue
        attempted_sources += 1
        target_urls = _expand_artifact_urls(candidate, config, ssl_context=ssl_context)
        if not target_urls:
            artifacts.append(
                CollectedArtifact(
                    artifact_id=_artifact_id(candidate.url + "#resolution_failed"),
                    source_id=candidate.source_id,
                    source_url=candidate.url,
                    artifact_url=candidate.url,
                    local_path="",
                    content_type="unknown",
                    size_bytes=0,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    relevance_score=0.0,
                    matched_terms=[],
                    is_relevant=False,
                    is_reasonable=False,
                    reasonableness_reason="resolution_failed",
                    reasonableness_details=["Could not resolve downloadable artifact URLs from this source."],
                    status="failed",
                    notes=["Artifact URL resolution failed for source."],
                )
            )
            continue
        successful_from_source = 0
        for artifact_url in target_urls:
            if successful_from_source >= config.max_artifacts_per_source:
                break
            if artifact_url in seen_artifact_urls:
                continue
            seen_artifact_urls.add(artifact_url)
            artifact = _download_and_inspect(
                spec=spec,
                source_id=candidate.source_id,
                source_url=candidate.url,
                artifact_url=artifact_url,
                keyword_bank=keyword_bank,
                output_dir=output_dir,
                config=config,
                ssl_context=ssl_context,
            )
            artifacts.append(artifact)
            if artifact.status in {"downloaded", "filtered_out"}:
                successful_from_source += 1

    summary = CollectionSummary(
        attempted_sources=attempted_sources,
        attempted_artifacts=len(artifacts),
        downloaded_artifacts=sum(1 for a in artifacts if a.status in {"downloaded", "filtered_out"}),
        relevant_artifacts=sum(1 for a in artifacts if a.is_relevant),
        reasonable_artifacts=sum(1 for a in artifacts if a.is_reasonable),
        failed_artifacts=sum(1 for a in artifacts if a.status == "failed"),
        output_dir=str(output_dir),
    )
    return artifacts, summary


def _expand_artifact_urls(
    candidate: SourceCandidate,
    config: CollectorConfig,
    ssl_context: ssl.SSLContext,
) -> list[str]:
    if _is_landing_or_listing_url(candidate.url):
        return []

    if _is_hf_collection_url(candidate.url):
        collection_urls = _expand_huggingface_collection_urls(candidate.url, config, ssl_context=ssl_context)
        return collection_urls[: config.max_links_per_source]

    if _is_hf_dataset_url(candidate.url):
        hf_urls = _expand_huggingface_dataset_urls(candidate.url, config, ssl_context=ssl_context)
        # For HF dataset pages, only return direct artifact URLs; do not fall back to landing-page HTML.
        return hf_urls[: config.max_links_per_source]

    if _looks_like_artifact_url(candidate.url):
        return [candidate.url]

    html = _fetch_text(candidate.url, timeout_seconds=config.timeout_seconds, ssl_context=ssl_context)
    if not html:
        return [candidate.url]

    links = _extract_links(html)
    resolved: list[str] = []
    for link in links:
        absolute = urljoin(candidate.url, link)
        absolute = _normalize_search_link(absolute)
        if config.collect_search_portal_candidates and candidate.collection_method == "search_portal":
            if absolute.startswith(("http://", "https://")):
                resolved.append(absolute)
        elif _looks_like_artifact_url(absolute):
            resolved.append(absolute)
        if len(resolved) >= config.max_links_per_source:
            break

    if not resolved:
        return [candidate.url]
    return _dedupe_preserve_order(resolved)


def _expand_huggingface_dataset_urls(
    source_url: str,
    config: CollectorConfig,
    ssl_context: ssl.SSLContext,
) -> list[str]:
    parsed = urlparse(source_url)
    if "huggingface.co" not in parsed.netloc:
        return []

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 3 or path_parts[0] != "datasets":
        return []

    dataset_id = f"{path_parts[1]}/{path_parts[2]}"
    api_urls = [
        f"https://huggingface.co/api/datasets/{dataset_id}/parquet",
        f"https://datasets-server.huggingface.co/parquet?dataset={quote(dataset_id, safe='')}",
    ]
    urls: list[str] = []

    for api_url in api_urls:
        text = _fetch_text(api_url, timeout_seconds=config.timeout_seconds, ssl_context=ssl_context)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if url.startswith(("http://", "https://")):
                    urls.append(url)
        elif isinstance(payload, dict):
            for key in ("parquet_files", "files"):
                value = payload.get(key)
                if not isinstance(value, list):
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url", "")).strip()
                    if url.startswith(("http://", "https://")):
                        urls.append(url)

    if urls:
        return _dedupe_preserve_order(urls)[: config.max_links_per_source]

    # Fallback: use dataset metadata endpoint and construct resolve URLs from file listing.
    dataset_meta_url = f"https://huggingface.co/api/datasets/{dataset_id}"
    meta_text = _fetch_text(dataset_meta_url, timeout_seconds=config.timeout_seconds, ssl_context=ssl_context)
    if meta_text:
        try:
            payload = json.loads(meta_text)
            if isinstance(payload, dict):
                siblings = payload.get("siblings")
                if isinstance(siblings, list):
                    for sibling in siblings:
                        if not isinstance(sibling, dict):
                            continue
                        filename = str(sibling.get("rfilename", "")).strip()
                        if not filename:
                            continue
                        lowered = filename.lower()
                        if lowered.endswith((".parquet", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".zip", ".pdf")):
                            urls.append(f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{quote(filename, safe='/')}")
        except Exception:
            pass

    if urls:
        return _dedupe_preserve_order(urls)[: config.max_links_per_source]

    # Fallback: scrape links from HF dataset page HTML and keep resolve/parquet-like targets.
    html = _fetch_text(source_url, timeout_seconds=config.timeout_seconds, ssl_context=ssl_context)
    if not html:
        return []

    links = _extract_links(html)
    resolved: list[str] = []
    for link in links:
        absolute = _normalize_search_link(urljoin(source_url, link))
        lowered = absolute.lower()
        if "/resolve/" in lowered or lowered.endswith((".parquet", ".json", ".jsonl", ".csv", ".zip")):
            resolved.append(absolute)
        if len(resolved) >= config.max_links_per_source:
            break
    return _dedupe_preserve_order(resolved)


def _is_hf_dataset_url(url: str) -> bool:
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    return len(path_parts) >= 3 and path_parts[0] == "datasets"


def _is_hf_collection_url(url: str) -> bool:
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    return len(path_parts) >= 3 and path_parts[0] == "collections"


def _expand_huggingface_collection_urls(
    source_url: str,
    config: CollectorConfig,
    ssl_context: ssl.SSLContext,
) -> list[str]:
    html = _fetch_text(source_url, timeout_seconds=config.timeout_seconds, ssl_context=ssl_context)
    if not html:
        return []

    links = _extract_links(html)
    dataset_page_urls: list[str] = []
    for link in links:
        absolute = _normalize_search_link(urljoin(source_url, link))
        if _is_hf_dataset_url(absolute):
            dataset_page_urls.append(absolute)
        if len(dataset_page_urls) >= config.max_links_per_source:
            break

    resolved_artifacts: list[str] = []
    for dataset_url in _dedupe_preserve_order(dataset_page_urls):
        artifacts = _expand_huggingface_dataset_urls(dataset_url, config, ssl_context=ssl_context)
        resolved_artifacts.extend(artifacts)
        if len(resolved_artifacts) >= config.max_links_per_source:
            break
    return _dedupe_preserve_order(resolved_artifacts)[: config.max_links_per_source]


def _download_and_inspect(
    spec: AcquisitionSpec,
    source_id: str,
    source_url: str,
    artifact_url: str,
    keyword_bank: list[str],
    output_dir: Path,
    config: CollectorConfig,
    ssl_context: ssl.SSLContext,
) -> CollectedArtifact:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    artifact_id = _artifact_id(artifact_url)

    try:
        file_path, content_type, size_bytes, truncated = _download_file(
            artifact_url,
            output_dir,
            timeout_seconds=config.timeout_seconds,
            max_download_bytes=config.max_download_bytes,
            ssl_context=ssl_context,
        )
        relevance_score, matched_terms, is_relevant = evaluate_artifact_relevance(
            file_path=file_path,
            keyword_bank=keyword_bank,
            label_space=spec.target_schema.label_space,
            min_score=config.relevance_threshold,
        )
        reasonableness = assess_artifact_reasonableness(file_path)
        is_reasonable = bool(reasonableness.get("is_reasonable", False))
        reasonableness_reason = str(reasonableness.get("reason", "unknown"))
        reasonableness_details = [str(item) for item in reasonableness.get("details", [])]

        notes: list[str] = []
        if truncated:
            notes.append("Download was truncated due to max_download_bytes limit.")

        # Keep every successfully fetched artifact in final output.
        # Relevance/reasonableness are still attached as metadata for downstream agents.
        status = "downloaded"
        if not is_relevant:
            notes.append("Artifact was collected and kept, but marked not relevant to the acquisition spec.")
        if not is_reasonable:
            notes.append(f"Artifact was collected and kept, but failed reasonableness check: {reasonableness_reason}.")

        return CollectedArtifact(
            artifact_id=artifact_id,
            source_id=source_id,
            source_url=source_url,
            artifact_url=artifact_url,
            local_path=str(file_path),
            content_type=content_type,
            size_bytes=size_bytes,
            retrieved_at=retrieved_at,
            relevance_score=relevance_score,
            matched_terms=matched_terms,
            is_relevant=is_relevant,
            is_reasonable=is_reasonable,
            reasonableness_reason=reasonableness_reason,
            reasonableness_details=reasonableness_details,
            status=status,
            notes=notes,
        )
    except Exception as exc:
        return CollectedArtifact(
            artifact_id=artifact_id,
            source_id=source_id,
            source_url=source_url,
            artifact_url=artifact_url,
            local_path="",
            content_type="unknown",
            size_bytes=0,
            retrieved_at=retrieved_at,
            relevance_score=0.0,
            matched_terms=[],
            is_relevant=False,
            is_reasonable=False,
            reasonableness_reason="download_failed",
            reasonableness_details=["Reasonableness check skipped because download failed."],
            status="failed",
            notes=[f"Download failed: {exc}"],
        )


def _download_file(
    url: str,
    output_dir: Path,
    timeout_seconds: int,
    max_download_bytes: int,
    ssl_context: ssl.SSLContext,
) -> tuple[Path, str, int, bool]:
    req = Request(url, headers={"User-Agent": "retrieval-agent/0.1 (+raw-data-collector)"})
    with urlopen(req, timeout=timeout_seconds, context=ssl_context) as response:  # nosec B310 - intended URL fetcher
        content_type = (response.headers.get("Content-Type") or "application/octet-stream").split(";")[0].strip().lower()
        filename = _build_filename(url, content_type)
        destination = output_dir / filename

        total = 0
        truncated = False
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_download_bytes:
                    handle.write(chunk[: max_download_bytes - (total - len(chunk))])
                    truncated = True
                    total = max_download_bytes
                    break
                handle.write(chunk)

    return destination, content_type, total, truncated


def _build_filename(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    basename = Path(parsed.path).name or "artifact"
    safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", basename)

    ext = Path(safe_base).suffix.lower()
    if not ext:
        ext = CONTENT_TYPE_EXTENSION_MAP.get(content_type) or mimetypes.guess_extension(content_type) or ".bin"

    stem = Path(safe_base).stem[:60] or "artifact"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}{ext}"


def _artifact_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _fetch_text(url: str, timeout_seconds: int, ssl_context: ssl.SSLContext) -> str:
    req = Request(url, headers={"User-Agent": "retrieval-agent/0.1 (+source-expander)"})
    try:
        with urlopen(req, timeout=timeout_seconds, context=ssl_context) as response:  # nosec B310 - intended URL fetcher
            content_type = (response.headers.get("Content-Type") or "").lower()
            if (
                "text/html" not in content_type
                and "text/plain" not in content_type
                and "application/json" not in content_type
                and "text/json" not in content_type
                and "application/problem+json" not in content_type
            ):
                return ""
            data = response.read(1024 * 512)
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_links(html: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(html)
    return parser.links


def _looks_like_artifact_url(url: str) -> bool:
    lowered = url.lower()
    if _is_landing_or_listing_url(url):
        return False
    if any(lowered.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        return True
    return any(token in lowered for token in ("download", "export", "/resolve/"))


def _is_landing_or_listing_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()

    if "huggingface.co" in host:
        if path in {"/datasets", "/models", "/spaces"}:
            return True
        if path.startswith("/collections/"):
            return True
        if path.startswith("/datasets") and "search=" in query:
            return True
    if "duckduckgo.com" in host:
        return True
    if "kaggle.com" in host and path.startswith("/datasets"):
        return True
    if "github.com" in host and path == "/search":
        return True
    return False


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_search_link(url: str) -> str:
    """Normalize search result redirect links into direct targets when possible."""
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path in {"/l/", "/"}:
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
    return url


def _build_ssl_context(ssl_verify: bool, ca_bundle: str | None) -> ssl.SSLContext:
    if not ssl_verify:
        return ssl._create_unverified_context()

    bundle = ca_bundle
    if bundle:
        return ssl.create_default_context(cafile=bundle)

    # Prefer certifi bundle when available to avoid local Python CA-store issues.
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _apply_ssl_env_overrides(config: CollectorConfig) -> CollectorConfig:
    ssl_verify = config_ssl_verify_env(default=config.ssl_verify)
    ca_bundle = config_ca_bundle_env(default=config.ca_bundle)
    return CollectorConfig(
        max_sources=config.max_sources,
        max_links_per_source=config.max_links_per_source,
        max_artifacts_per_source=config.max_artifacts_per_source,
        timeout_seconds=config.timeout_seconds,
        max_download_bytes=config.max_download_bytes,
        relevance_threshold=config.relevance_threshold,
        ssl_verify=ssl_verify,
        ca_bundle=ca_bundle,
        collect_search_portal_candidates=config.collect_search_portal_candidates,
    )


def config_ssl_verify_env(default: bool = True) -> bool:
    value = os.environ.get("RETRIEVAL_AGENT_SSL_VERIFY")
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


def config_ca_bundle_env(default: str | None = None) -> str | None:
    value = os.environ.get("RETRIEVAL_AGENT_CA_BUNDLE")
    if value is None:
        return default
    value = value.strip()
    return value or default
