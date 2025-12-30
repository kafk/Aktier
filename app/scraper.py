import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import logging
import asyncio
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Swedish timezone
SWEDISH_TZ = ZoneInfo("Europe/Stockholm")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def parse_relative_time(time_str: str) -> Optional[datetime]:
    """Parse relative time strings like '1h ago', '2d ago', '30m ago'."""
    if not time_str:
        return None

    time_str = time_str.lower().strip()
    now = datetime.now(SWEDISH_TZ)

    # Match patterns like "1h ago", "2 hours ago", "30m ago", "1d ago"
    patterns = [
        (r'(\d+)\s*m(?:in(?:ute)?s?)?\s*ago', 'minutes'),
        (r'(\d+)\s*h(?:our)?s?\s*ago', 'hours'),
        (r'(\d+)\s*d(?:ay)?s?\s*ago', 'days'),
        (r'(\d+)\s*w(?:eek)?s?\s*ago', 'weeks'),
        (r'yesterday', 'yesterday'),
    ]

    for pattern, unit in patterns:
        match = re.search(pattern, time_str)
        if match:
            if unit == 'yesterday':
                return now - timedelta(days=1)
            value = int(match.group(1))
            if unit == 'minutes':
                return now - timedelta(minutes=value)
            elif unit == 'hours':
                return now - timedelta(hours=value)
            elif unit == 'days':
                return now - timedelta(days=value)
            elif unit == 'weeks':
                return now - timedelta(weeks=value)

    return None


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse various date string formats."""
    if not date_str:
        return None

    # Try relative time first
    relative = parse_relative_time(date_str)
    if relative:
        return relative

    # Try various date formats
    date_formats = [
        "%a, %B %d, %Y at %I:%M %p",  # Mon, December 29, 2025 at 4:48 PM
        "%B %d, %Y at %I:%M %p",       # December 29, 2025 at 4:48 PM
        "%Y-%m-%dT%H:%M:%S",           # ISO format
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y",                    # Dec 29, 2025
        "%B %d, %Y",                    # December 29, 2025
    ]

    # Clean up the string
    clean_str = re.sub(r'\s+GMT[+-]\d+', '', date_str)  # Remove GMT offset
    clean_str = re.sub(r'\s+', ' ', clean_str).strip()

    for fmt in date_formats:
        try:
            parsed = datetime.strptime(clean_str, fmt)
            return parsed.replace(tzinfo=SWEDISH_TZ)
        except ValueError:
            continue

    return None


class NewsArticle:
    def __init__(
        self,
        title: str,
        url: str,
        summary: Optional[str] = None,
        source: Optional[str] = None,
        published_at: Optional[datetime] = None,
        stock_symbol: Optional[str] = None,
    ):
        self.title = title
        self.url = url
        self.summary = summary
        self.source = source
        self.published_at = published_at
        self.stock_symbol = stock_symbol


async def scrape_yahoo_finance_news(symbol: str) -> list[NewsArticle]:
    """Scrape news for a specific stock symbol from Yahoo Finance."""
    articles = []
    url = f"https://finance.yahoo.com/quote/{symbol}/news/"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=HEADERS)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Find news articles in the news section
            news_items = soup.find_all("li", class_=re.compile(r"stream-item|js-stream-content"))

            if not news_items:
                # Try alternative selectors
                news_items = soup.find_all("div", {"data-testid": re.compile(r"news-stream|storyitem")})

            if not news_items:
                # Another fallback - look for article links
                news_items = soup.select("section[data-testid='news-stream'] li, .news-stream li, article")

            for item in news_items[:10]:  # Limit to 10 most recent
                try:
                    # Try multiple selectors for the link
                    link = item.find("a", href=re.compile(r"/news/|/m/"))
                    if not link:
                        link = item.find("a", class_=re.compile(r"title|headline"))
                    if not link:
                        link = item.select_one("h3 a, h2 a, a[data-testid='title']")

                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    href = link.get("href", "")

                    if not title or not href:
                        continue

                    # Make URL absolute
                    if href.startswith("/"):
                        href = f"https://finance.yahoo.com{href}"

                    # Get summary if available
                    summary_elem = item.find("p") or item.find(class_=re.compile(r"summary|description|snippet"))
                    summary = summary_elem.get_text(strip=True) if summary_elem else None

                    # Get source if available
                    source_elem = item.find(class_=re.compile(r"source|provider|publisher"))
                    source = source_elem.get_text(strip=True) if source_elem else "Yahoo Finance"

                    # Try to get publication time
                    published_at = None
                    # Look for time element
                    time_elem = item.find("time")
                    if time_elem:
                        # Try datetime attribute first
                        datetime_attr = time_elem.get("datetime")
                        if datetime_attr:
                            published_at = parse_date_string(datetime_attr)
                        if not published_at:
                            published_at = parse_date_string(time_elem.get_text(strip=True))

                    # Try other common time patterns
                    if not published_at:
                        time_span = item.find(class_=re.compile(r"time|date|ago|timestamp"))
                        if time_span:
                            published_at = parse_date_string(time_span.get_text(strip=True))

                    # Look for text containing "ago" anywhere in the item
                    if not published_at:
                        item_text = item.get_text()
                        ago_match = re.search(r'(\d+\s*[hmd](?:ours?|ins?|ays?)?\s*ago)', item_text, re.IGNORECASE)
                        if ago_match:
                            published_at = parse_relative_time(ago_match.group(1))

                    # Fallback to now if no date found
                    if not published_at:
                        published_at = datetime.now(SWEDISH_TZ)

                    articles.append(
                        NewsArticle(
                            title=title,
                            url=href,
                            summary=summary,
                            source=source,
                            published_at=published_at,
                            stock_symbol=symbol,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error parsing news item: {e}")
                    continue

    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching news for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Error scraping news for {symbol}: {e}")

    logger.info(f"Found {len(articles)} articles for {symbol}")
    return articles


async def scrape_google_finance_news(symbol: str) -> list[NewsArticle]:
    """Scrape news from Google Finance as a backup source."""
    articles = []
    url = f"https://www.google.com/finance/quote/{symbol}:NYSE"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=HEADERS)

            if response.status_code != 200:
                # Try NASDAQ
                url = f"https://www.google.com/finance/quote/{symbol}:NASDAQ"
                response = await client.get(url, headers=HEADERS)

            soup = BeautifulSoup(response.text, "lxml")

            # Find news section
            news_section = soup.find("div", class_=re.compile(r"news|related"))
            if news_section:
                links = news_section.find_all("a", href=re.compile(r"^https?://"))

                for link in links[:10]:
                    title = link.get_text(strip=True)
                    href = link.get("href", "")

                    if title and href and len(title) > 10:
                        articles.append(
                            NewsArticle(
                                title=title,
                                url=href,
                                source="Google Finance",
                                stock_symbol=symbol,
                            )
                        )
    except Exception as e:
        logger.error(f"Error scraping Google Finance for {symbol}: {e}")

    return articles


async def scrape_all_sources(symbol: str) -> list[NewsArticle]:
    """Scrape news from all available sources for a symbol."""
    yahoo_articles = await scrape_yahoo_finance_news(symbol)

    # If Yahoo didn't return enough, try Google Finance
    if len(yahoo_articles) < 3:
        google_articles = await scrape_google_finance_news(symbol)
        yahoo_articles.extend(google_articles)

    return yahoo_articles


def check_keywords(text: str, keywords: list[str]) -> list[str]:
    """Check if any keywords are present in the text."""
    if not text or not keywords:
        return []

    text_lower = text.lower()
    matched = []

    for keyword in keywords:
        if keyword.lower() in text_lower:
            matched.append(keyword)

    return matched


def get_news_cutoff_date() -> datetime:
    """Get the cutoff date for news (start of yesterday in Swedish time)."""
    now_swedish = datetime.now(SWEDISH_TZ)
    yesterday = now_swedish - timedelta(days=1)
    # Start of yesterday
    cutoff = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    # Convert to UTC for database comparison
    return cutoff.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def is_article_recent(published_at: Optional[datetime]) -> bool:
    """Check if an article is from today or yesterday."""
    if not published_at:
        return True  # If no date, assume it's recent

    cutoff = get_news_cutoff_date()
    return published_at >= cutoff
