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

                    articles.append(
                        NewsArticle(
                            title=title,
                            url=href,
                            summary=summary,
                            source=source,
                            published_at=datetime.utcnow(),  # Yahoo doesn't always show exact time
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
