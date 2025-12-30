"""
Impact scoring system for stock news alerts.
Calculates a 1-10 impact score based on multiple factors.
"""

import json
import re
from typing import Dict, Optional, Tuple


# Base Event Impact scores (1-5)
EVENT_IMPACT_SCORES = {
    # High impact (5)
    "EARNINGS_BEAT": 5,
    "EARNINGS_MISS": 5,
    "GUIDANCE_RAISE": 5,
    "GUIDANCE_CUT": 5,
    "FDA_APPROVAL": 5,
    "FDA_REJECTION": 5,
    # Medium-high impact (4)
    "ACQUISITION": 4,
    "SEC_INVESTIGATION": 4,
    "BANKRUPTCY": 4,
    # Medium impact (3)
    "CEO_CHANGE": 3,
    "CFO_CHANGE": 3,
    "LAYOFFS": 3,
    "STOCK_BUYBACK": 3,
    "DIVIDEND": 3,
    "STOCK_SPLIT": 3,
    "LAWSUIT": 3,
    # Lower impact (2)
    "UPGRADE": 2,
    "DOWNGRADE": 2,
    "PRODUCT_LAUNCH": 2,
    "PARTNERSHIP": 2,
    "IPO": 2,
    "INSIDER_BUYING": 2,
    "INSIDER_SELLING": 2,
    "INVESTMENT": 2,
}

# Sentiment intensity words
STRONG_POSITIVE_WORDS = [
    "exceptional", "breakthrough", "dominant", "unprecedented", "soars",
    "skyrockets", "explodes", "massive", "blowout", "crushes", "smashes",
    "record-breaking", "phenomenal", "extraordinary", "stellar"
]

MODERATE_POSITIVE_WORDS = [
    "strong", "record", "significant", "accelerating", "solid",
    "robust", "impressive", "beat", "exceeds", "surges", "jumps",
    "rallies", "gains", "growth", "outperforms"
]

STRONG_NEGATIVE_WORDS = [
    "severe", "material", "significant decline", "collapse", "crash",
    "plunge", "disaster", "crisis", "devastating", "catastrophic",
    "freefall", "implosion", "meltdown", "tanking"
]

MODERATE_NEGATIVE_WORDS = [
    "weak", "slowing", "pressure", "disappointing", "miss", "decline",
    "drops", "falls", "concerns", "warning", "cuts", "struggles",
    "underperforms", "soft", "challenging"
]

# Source credibility scores
SOURCE_CREDIBILITY = {
    # Tier 1: Official sources (+2)
    "sec.gov": 2.0,
    "ir.": 2.0,  # Investor relations
    "investor.": 2.0,

    # Tier 1.5: Major financial news (+1.5-2)
    "reuters.com": 2.0,
    "bloomberg.com": 2.0,
    "wsj.com": 1.5,
    "ft.com": 1.5,
    "cnbc.com": 1.5,

    # Tier 2: Established financial media (+1)
    "marketwatch.com": 1.0,
    "barrons.com": 1.0,
    "finance.yahoo.com": 1.0,
    "seekingalpha.com": 1.0,
    "investing.com": 1.0,
    "thestreet.com": 1.0,
    "fool.com": 1.0,
    "benzinga.com": 1.0,
    "investors.com": 1.0,

    # Tier 3: General news (+0.5)
    "news.google.com": 0.5,
    "bbc.com": 0.5,
    "cnn.com": 0.5,
    "nytimes.com": 0.5,
}


def calculate_base_event_score(event_type: Optional[str]) -> float:
    """
    Calculate base event impact score (1-5).
    Some events always move stocks more than others.
    """
    if not event_type:
        return 1.0  # General news mention
    return EVENT_IMPACT_SCORES.get(event_type, 1.0)


def calculate_sentiment_strength(text: str) -> float:
    """
    Calculate sentiment strength modifier (-2 to +2).
    Detects language intensity, not just polarity.
    """
    text_lower = text.lower()

    # Count intensity words
    strong_pos = sum(1 for word in STRONG_POSITIVE_WORDS if word in text_lower)
    mod_pos = sum(1 for word in MODERATE_POSITIVE_WORDS if word in text_lower)
    strong_neg = sum(1 for word in STRONG_NEGATIVE_WORDS if word in text_lower)
    mod_neg = sum(1 for word in MODERATE_NEGATIVE_WORDS if word in text_lower)

    # Calculate score
    positive_score = min(2.0, strong_pos * 2 + mod_pos * 1)
    negative_score = min(2.0, strong_neg * 2 + mod_neg * 1)

    # Net sentiment
    if positive_score > negative_score:
        return min(2.0, positive_score)
    elif negative_score > positive_score:
        return max(-2.0, -negative_score)
    return 0.0


def calculate_source_credibility(source: Optional[str], url: Optional[str] = None) -> float:
    """
    Calculate source credibility score (0-2).
    Who said it matters.
    """
    if not source and not url:
        return 0.5  # Unknown source

    check_text = f"{source or ''} {url or ''}".lower()

    # Check for known sources
    for source_pattern, score in SOURCE_CREDIBILITY.items():
        if source_pattern in check_text:
            return score

    # Check for official filings
    if any(term in check_text for term in ["8-k", "10-k", "10-q", "sec filing", "press release"]):
        return 2.0

    return 0.5  # Default for unknown sources


def calculate_surprise_factor(
    event_type: Optional[str],
    stock_symbol: str,
    historical_events: list = None
) -> float:
    """
    Calculate surprise factor (0-2).
    Markets move on unexpected news.

    TODO: Implement with historical data:
    - Compare against previous earnings/guidance
    - Check if this is a first-time event
    - Check if it was rumored before
    """
    # For now, return a moderate score
    # Future: integrate with historical data
    if not historical_events:
        return 1.0  # Assume moderate surprise without history

    # Check if this event type has occurred recently
    recent_same_events = [e for e in historical_events if e.get("event_type") == event_type]
    if not recent_same_events:
        return 2.0  # First time event - high surprise

    return 0.5  # Repeated event - low surprise


def calculate_company_sensitivity(stock_symbol: str, market_cap: str = None) -> float:
    """
    Calculate company sensitivity (0-2).
    Small or volatile stocks move more.

    TODO: Implement with market data:
    - Get market cap
    - Get beta
    - Get short interest
    """
    # Known mega-caps (less sensitive)
    mega_caps = ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "BRK.A", "BRK.B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC"]

    # Known volatile sectors (more sensitive)
    biotech_patterns = ["BIO", "GENE", "PHARMA", "MED", "THERAPEUTICS"]

    symbol_upper = stock_symbol.upper()

    if symbol_upper in mega_caps:
        return 0.5

    # Check if it looks like a biotech (often more volatile)
    if any(pattern in symbol_upper for pattern in biotech_patterns):
        return 2.0

    # Default: moderate sensitivity
    return 1.0


def calculate_market_context() -> float:
    """
    Calculate market context modifier (-1 to +1).
    Same news ≠ same reaction in different markets.

    TODO: Implement with market data:
    - Check if market is in bear/bull mode
    - Check sector performance
    - Check VIX level
    """
    # For now, return neutral
    # Future: integrate with market data APIs
    return 0.0


def calculate_impact_score(
    event_type: Optional[str],
    sentiment: Optional[str],
    title: str,
    summary: str = "",
    source: Optional[str] = None,
    url: Optional[str] = None,
    stock_symbol: str = "",
    historical_events: list = None
) -> Tuple[int, Dict]:
    """
    Calculate the total impact score (1-10) and return breakdown.

    Components:
    1. Base Event Impact (1-5)
    2. Sentiment Strength (-2 to +2)
    3. Source Credibility (0-2)
    4. Surprise Factor (0-2) - Future enhancement
    5. Company Sensitivity (0-2) - Partial implementation
    6. Market Context (-1 to +1) - Future enhancement
    """
    text = f"{title} {summary}"

    # Calculate each component
    base_event = calculate_base_event_score(event_type)
    sentiment_strength = calculate_sentiment_strength(text)
    source_cred = calculate_source_credibility(source, url)
    surprise = calculate_surprise_factor(event_type, stock_symbol, historical_events)
    company_sens = calculate_company_sensitivity(stock_symbol)
    market_ctx = calculate_market_context()

    # Calculate raw total
    raw_score = (
        base_event +
        sentiment_strength +
        source_cred +
        surprise +
        company_sens +
        market_ctx
    )

    # Normalize to 1-10 scale
    # Max possible: 5 + 2 + 2 + 2 + 2 + 1 = 14
    # Min possible: 1 + (-2) + 0 + 0 + 0 + (-1) = -2
    # Normalize: ((raw + 2) / 16) * 9 + 1
    normalized = ((raw_score + 2) / 16) * 9 + 1
    final_score = max(1, min(10, round(normalized)))

    # Create breakdown for transparency
    breakdown = {
        "base_event": round(base_event, 1),
        "sentiment_strength": round(sentiment_strength, 1),
        "source_credibility": round(source_cred, 1),
        "surprise_factor": round(surprise, 1),
        "company_sensitivity": round(company_sens, 1),
        "market_context": round(market_ctx, 1),
        "raw_total": round(raw_score, 1),
        "final_score": final_score
    }

    return final_score, breakdown


def get_impact_band(score: int) -> Dict:
    """
    Get the impact band info for a score.

    Score bands:
    1-2: Noise (Ignore)
    3-4: Low (Digest)
    5-6: Medium (Watch)
    7-8: High (Alert)
    9-10: Critical (Push notification)
    """
    bands = {
        (1, 2): {"label": "Noise", "action": "Ignore", "color": "gray"},
        (3, 4): {"label": "Low", "action": "Digest", "color": "blue"},
        (5, 6): {"label": "Medium", "action": "Watch", "color": "yellow"},
        (7, 8): {"label": "High", "action": "Alert", "color": "orange"},
        (9, 10): {"label": "Critical", "action": "Push", "color": "red"},
    }

    for (low, high), info in bands.items():
        if low <= score <= high:
            return info

    return {"label": "Unknown", "action": "Review", "color": "gray"}


def score_breakdown_to_json(breakdown: Dict) -> str:
    """Convert breakdown dict to JSON string for storage."""
    return json.dumps(breakdown)


def json_to_score_breakdown(json_str: str) -> Dict:
    """Parse JSON string back to breakdown dict."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}
