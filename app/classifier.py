"""
Rule-based event classifier for stock news.
Classifies news headlines into event types and sentiment.
"""

import re
from typing import Tuple, Optional, List, Dict


# Event type definitions (display names for built-in rules)
EVENT_TYPES = {
    "EARNINGS_BEAT": "Earnings Beat",
    "EARNINGS_MISS": "Earnings Miss",
    "GUIDANCE_RAISE": "Guidance Raised",
    "GUIDANCE_CUT": "Guidance Cut",
    "ACQUISITION": "Acquisition/Merger",
    "CEO_CHANGE": "CEO Change",
    "CFO_CHANGE": "CFO Change",
    "LAYOFFS": "Layoffs",
    "FDA_APPROVAL": "FDA Approval",
    "FDA_REJECTION": "FDA Rejection",
    "SEC_INVESTIGATION": "SEC Investigation",
    "LAWSUIT": "Lawsuit",
    "STOCK_BUYBACK": "Stock Buyback",
    "DIVIDEND": "Dividend",
    "STOCK_SPLIT": "Stock Split",
    "UPGRADE": "Analyst Upgrade",
    "DOWNGRADE": "Analyst Downgrade",
    "PRODUCT_LAUNCH": "Product Launch",
    "PARTNERSHIP": "Partnership",
    "BANKRUPTCY": "Bankruptcy",
    "IPO": "IPO",
    "INSIDER_BUYING": "Insider Buying",
    "INSIDER_SELLING": "Insider Selling",
    "INVESTMENT": "Investment",
}


def classify_event(title: str, summary: str = "", custom_rules: List[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Classify a news headline into an event type and sentiment.

    Args:
        title: The news headline
        summary: Optional article summary
        custom_rules: List of custom rules from database

    Returns:
        Tuple of (event_type, sentiment) or (None, None) if no match
    """
    text = f"{title} {summary}".lower()

    # Check custom rules from database first
    if custom_rules:
        for rule in custom_rules:
            if not rule.get("active", True):
                continue
            keywords = [kw.strip() for kw in rule.get("keywords", "").split(",")]
            if any(kw in text for kw in keywords if kw):
                return rule.get("event_type"), rule.get("sentiment", "neutral")

    # Earnings
    if _matches(text, ["beats", "topped", "exceeds", "surpass"]) and _matches(text, ["earnings", "eps", "profit", "estimates", "expectations"]):
        return "EARNINGS_BEAT", "positive"

    if _matches(text, ["misses", "missed", "falls short", "below"]) and _matches(text, ["earnings", "eps", "profit", "estimates", "expectations"]):
        return "EARNINGS_MISS", "negative"

    # Guidance
    if _matches(text, ["raises", "raised", "lifts", "lifted", "boosts", "boosted", "increases"]) and _matches(text, ["guidance", "outlook", "forecast", "target"]):
        return "GUIDANCE_RAISE", "positive"

    if _matches(text, ["cuts", "lowers", "lowered", "reduces", "slashes", "trims"]) and _matches(text, ["guidance", "outlook", "forecast", "target"]):
        return "GUIDANCE_CUT", "negative"

    if _matches(text, ["weak", "disappointing", "soft"]) and _matches(text, ["guidance", "outlook", "forecast"]):
        return "GUIDANCE_CUT", "negative"

    # M&A
    if _matches(text, ["acquires", "acquire", "acquisition", "to buy", "buying", "purchase", "takeover", "bid for"]):
        return "ACQUISITION", "neutral"

    if _matches(text, ["merger", "merge", "merging", "combines with"]):
        return "ACQUISITION", "neutral"

    # Leadership changes
    if _matches(text, ["ceo"]) and _matches(text, ["resigns", "steps down", "departs", "leaves", "out", "fired", "replaced", "new", "appoints", "names"]):
        return "CEO_CHANGE", "neutral"

    if _matches(text, ["cfo"]) and _matches(text, ["resigns", "steps down", "departs", "leaves", "out", "fired", "replaced", "new", "appoints", "names"]):
        return "CFO_CHANGE", "neutral"

    # Layoffs
    if _matches(text, ["layoffs", "laying off", "job cuts", "workforce reduction", "downsizing", "restructuring"]) or \
       (_matches(text, ["cuts"]) and _matches(text, ["jobs", "employees", "workers", "staff"])):
        return "LAYOFFS", "negative"

    # FDA (Biotech)
    if _matches(text, ["fda"]) and _matches(text, ["approves", "approved", "approval", "clears", "cleared", "green light"]):
        return "FDA_APPROVAL", "positive"

    if _matches(text, ["fda"]) and _matches(text, ["rejects", "rejected", "rejection", "denies", "denied", "refuses"]):
        return "FDA_REJECTION", "negative"

    # SEC / Legal
    if _matches(text, ["sec"]) and _matches(text, ["investigation", "investigates", "probe", "probing", "inquiry", "subpoena"]):
        return "SEC_INVESTIGATION", "negative"

    if _matches(text, ["lawsuit", "sued", "sues", "suing", "litigation", "legal action", "class action"]):
        return "LAWSUIT", "negative"

    # Stock actions
    if _matches(text, ["buyback", "repurchase", "buy back"]) and _matches(text, ["stock", "shares", "billion", "million"]):
        return "STOCK_BUYBACK", "positive"

    if _matches(text, ["dividend"]) and _matches(text, ["raises", "increases", "hikes", "boosts", "declares", "announces"]):
        return "DIVIDEND", "positive"

    if _matches(text, ["dividend"]) and _matches(text, ["cuts", "slashes", "suspends", "eliminates"]):
        return "DIVIDEND", "negative"

    if _matches(text, ["stock split", "share split"]) or (_matches(text, ["split"]) and _matches(text, ["for-1", "for 1", "-for-"])):
        return "STOCK_SPLIT", "positive"

    # Analyst ratings
    if _matches(text, ["upgrades", "upgraded", "raises rating", "raised rating"]) or \
       (_matches(text, ["upgrade"]) and _matches(text, ["buy", "outperform", "overweight"])):
        return "UPGRADE", "positive"

    if _matches(text, ["downgrades", "downgraded", "cuts rating", "lowers rating"]) or \
       (_matches(text, ["downgrade"]) and _matches(text, ["sell", "underperform", "underweight"])):
        return "DOWNGRADE", "negative"

    if _matches(text, ["price target"]) and _matches(text, ["raises", "raised", "lifts", "increases", "boosts"]):
        return "UPGRADE", "positive"

    if _matches(text, ["price target"]) and _matches(text, ["cuts", "lowers", "reduces", "slashes"]):
        return "DOWNGRADE", "negative"

    # Product & Business
    if _matches(text, ["launches", "launched", "unveils", "unveiled", "introduces", "announces", "reveals"]) and \
       _matches(text, ["product", "service", "platform", "device", "app", "feature"]):
        return "PRODUCT_LAUNCH", "positive"

    if _matches(text, ["partnership", "partners with", "teams up", "collaboration", "alliance", "deal with"]):
        return "PARTNERSHIP", "positive"

    # Negative events
    if _matches(text, ["bankruptcy", "chapter 11", "chapter 7", "insolvency", "insolvent"]):
        return "BANKRUPTCY", "negative"

    # IPO
    if _matches(text, ["ipo", "initial public offering", "goes public", "going public", "public debut"]):
        return "IPO", "neutral"

    # Insider trading
    if _matches(text, ["insider"]) and _matches(text, ["buys", "buying", "purchased", "acquires"]):
        return "INSIDER_BUYING", "positive"

    if _matches(text, ["insider"]) and _matches(text, ["sells", "selling", "sold", "dumps"]):
        return "INSIDER_SELLING", "negative"

    # No match - try to determine sentiment only
    sentiment = _detect_sentiment(text)
    return None, sentiment


def _matches(text: str, patterns: list) -> bool:
    """Check if any pattern exists in the text."""
    return any(pattern in text for pattern in patterns)


def _detect_sentiment(text: str) -> Optional[str]:
    """Detect overall sentiment from text when no specific event is matched."""
    positive_words = [
        "surges", "soars", "jumps", "rallies", "gains", "rises", "climbs",
        "record", "strong", "growth", "profit", "success", "wins", "positive",
        "beats", "exceeds", "outperforms", "bullish", "optimistic"
    ]

    negative_words = [
        "plunges", "crashes", "tumbles", "falls", "drops", "declines", "sinks",
        "loss", "weak", "warns", "concerns", "fears", "risk", "trouble",
        "fails", "misses", "underperforms", "bearish", "pessimistic", "slump"
    ]

    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"

    return "neutral"


def get_event_display_name(event_type: str) -> str:
    """Get the display name for an event type."""
    return EVENT_TYPES.get(event_type, event_type)


def get_sentiment_color(sentiment: str) -> str:
    """Get the color class for a sentiment."""
    colors = {
        "positive": "green",
        "negative": "red",
        "neutral": "gray"
    }
    return colors.get(sentiment, "gray")
