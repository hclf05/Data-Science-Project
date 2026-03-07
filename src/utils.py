"""Shared utility functions for the UFC empirical project."""

from __future__ import annotations

import re
from pathlib import Path
from threading import local
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_FIGURES = ROOT / "output" / "figures"

UFCSTATS_DOMAIN = "ufcstats.com"
UFCSTATS_BASE_URL = f"http://{UFCSTATS_DOMAIN}"
UFCSTATS_URL_CANDIDATES = (
    f"http://{UFCSTATS_DOMAIN}",
    f"http://www.{UFCSTATS_DOMAIN}",
    f"https://{UFCSTATS_DOMAIN}",
    f"https://www.{UFCSTATS_DOMAIN}",
)

REQUEST_TIMEOUT = 30
HTML_PARSER = "lxml"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Referer": UFCSTATS_BASE_URL,
}

_THREAD_LOCAL = local()


def ensure_directories() -> None:
    """Create project directories if they do not already exist."""
    for path in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUTPUT_FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    """Create a requests session with retry behaviour suitable for scraping."""
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_thread_session() -> requests.Session:
    """Reuse one session per worker thread."""
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = build_session()
        _THREAD_LOCAL.session = session
    return session


def normalise_whitespace(value: str) -> str:
    """Collapse repeated whitespace and strip ends."""
    return " ".join(value.split()).strip()


def safe_text(element: Tag | None) -> str:
    """Extract stripped text from a tag, or return an empty string."""
    if element is None:
        return ""
    return normalise_whitespace(element.get_text(" ", strip=True))


def canonicalize_ufcstats_url(url: str) -> str:
    """Normalise UFCStats URLs to the working HTTP endpoint."""
    if not url:
        return ""

    absolute_url = urljoin(UFCSTATS_BASE_URL, url)
    parsed = urlparse(absolute_url)

    if not parsed.netloc:
        parsed = urlparse(urljoin(UFCSTATS_BASE_URL, absolute_url))

    query = urlencode(parse_qsl(parsed.query, keep_blank_values=True))
    return urlunparse(
        ("http", UFCSTATS_DOMAIN, parsed.path, parsed.params, query, "")
    )


def candidate_ufcstats_urls(url: str) -> list[str]:
    """Generate HTTP/HTTPS and bare/www host variants for a UFCStats URL."""
    canonical_url = canonicalize_ufcstats_url(url)
    parsed = urlparse(canonical_url)
    query = urlencode(parse_qsl(parsed.query, keep_blank_values=True))
    candidates: list[str] = []

    for base_url in UFCSTATS_URL_CANDIDATES:
        base = urlparse(base_url)
        candidate = urlunparse(
            (base.scheme, base.netloc, parsed.path, parsed.params, query, "")
        )
        if candidate not in candidates:
            candidates.append(candidate)

    return candidates


def fetch_response(
    session: requests.Session,
    url: str,
    timeout: int = REQUEST_TIMEOUT,
) -> requests.Response:
    """Fetch a UFCStats page, trying the working host/scheme variants."""
    last_error: Exception | None = None

    for candidate_url in candidate_ufcstats_urls(url):
        try:
            response = session.get(candidate_url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc

    raise requests.ConnectionError(
        f"Failed to fetch UFCStats URL after trying multiple variants: {url}"
    ) from last_error


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch a page and return a parsed BeautifulSoup object."""
    response = fetch_response(session, url)
    return BeautifulSoup(response.text, HTML_PARSER)


def parse_labelled_value(value: str) -> tuple[str, str]:
    """Split strings such as 'Height: 5\\' 7\"' into key/value pairs."""
    cleaned = normalise_whitespace(value)
    if ":" not in cleaned:
        return cleaned.lower(), ""

    label, remainder = cleaned.split(":", maxsplit=1)
    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return key, remainder.strip()


def parse_height_to_inches(value: str) -> float | None:
    """Convert height strings like 5' 7\" to inches."""
    cleaned = normalise_whitespace(value)
    match = re.search(r"(?P<feet>\d+)\s*'\s*(?P<inches>\d+)", cleaned)
    if match is None:
        return None
    return float(int(match.group("feet")) * 12 + int(match.group("inches")))


def parse_reach_to_inches(value: str) -> float | None:
    """Convert reach strings like 70\" to inches."""
    cleaned = normalise_whitespace(value)
    match = re.search(r"(?P<reach>\d+(?:\.\d+)?)", cleaned)
    if match is None:
        return None
    return float(match.group("reach"))


def parse_weight_to_lbs(value: str) -> float | None:
    """Convert weight strings like 125 lbs. to pounds."""
    cleaned = normalise_whitespace(value)
    match = re.search(r"(?P<weight>\d+(?:\.\d+)?)", cleaned)
    if match is None:
        return None
    return float(match.group("weight"))
