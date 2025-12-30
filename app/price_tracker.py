"""
Price tracking module for backtesting alert accuracy.
Uses yfinance to fetch stock prices.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Try to import yfinance, with fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Price tracking will use manual input.")


def get_current_price(symbol: str) -> Optional[float]:
    """Get the current/latest price for a stock symbol."""
    if not YFINANCE_AVAILABLE:
        return None

    try:
        ticker = yf.Ticker(symbol)
        # Get the most recent price
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return None


def get_price_at_time(symbol: str, target_time: datetime) -> Optional[float]:
    """
    Get the stock price at or near a specific time.
    Uses intraday data if available, otherwise daily close.
    """
    if not YFINANCE_AVAILABLE:
        return None

    try:
        ticker = yf.Ticker(symbol)

        # Calculate days since target time
        now = datetime.now(ZoneInfo("UTC"))
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=ZoneInfo("UTC"))

        days_ago = (now - target_time).days

        if days_ago <= 7:
            # Use intraday data for recent dates
            hist = ticker.history(period="7d", interval="1h")
        else:
            # Use daily data for older dates
            hist = ticker.history(period=f"{days_ago + 5}d")

        if hist.empty:
            return None

        # Find the closest price to target time
        hist.index = hist.index.tz_convert("UTC")
        target_time_utc = target_time.astimezone(ZoneInfo("UTC")) if target_time.tzinfo else target_time

        # Find nearest index
        time_diffs = abs(hist.index - target_time_utc)
        nearest_idx = time_diffs.argmin()

        return float(hist['Close'].iloc[nearest_idx])

    except Exception as e:
        logger.error(f"Error fetching historical price for {symbol}: {e}")
        return None


def calculate_price_change(
    price_before: float,
    price_after: float
) -> Tuple[float, float]:
    """
    Calculate absolute and percentage change.
    Returns (absolute_change, percent_change)
    """
    if price_before == 0:
        return 0.0, 0.0

    absolute_change = price_after - price_before
    percent_change = (absolute_change / price_before) * 100

    return round(absolute_change, 2), round(percent_change, 2)


def price_change_to_impact(percent_change: float, sentiment: str = "neutral") -> int:
    """
    Convert a percentage price change to an impact score (1-10).

    The mapping considers:
    - Direction matches sentiment (positive news + price up = confirmed)
    - Magnitude of the move

    Impact Scale:
    - < 0.5%: 1-2 (Noise)
    - 0.5-1%: 3-4 (Low)
    - 1-2%: 5-6 (Medium)
    - 2-5%: 7-8 (High)
    - > 5%: 9-10 (Critical)
    """
    abs_change = abs(percent_change)

    # Base impact from magnitude
    if abs_change < 0.5:
        base_impact = 1
    elif abs_change < 1.0:
        base_impact = 3
    elif abs_change < 2.0:
        base_impact = 5
    elif abs_change < 3.0:
        base_impact = 6
    elif abs_change < 5.0:
        base_impact = 7
    elif abs_change < 7.0:
        base_impact = 8
    elif abs_change < 10.0:
        base_impact = 9
    else:
        base_impact = 10

    # Adjust based on sentiment alignment
    if sentiment == "positive":
        # Positive news should move price up
        if percent_change > 0:
            return min(10, base_impact + 1)  # Aligned - boost score
        else:
            return max(1, base_impact - 1)  # Misaligned - reduce score
    elif sentiment == "negative":
        # Negative news should move price down
        if percent_change < 0:
            return min(10, base_impact + 1)  # Aligned - boost score
        else:
            return max(1, base_impact - 1)  # Misaligned - reduce score
    else:
        # Neutral - just use magnitude
        return base_impact


def get_weight_adjustment_recommendation(
    event_type: str,
    avg_prediction_error: float,
    sample_count: int
) -> dict:
    """
    Generate a recommendation for adjusting event weights.

    Args:
        event_type: The event type (e.g., "EARNINGS_BEAT")
        avg_prediction_error: Average (predicted - actual), positive = over-predict
        sample_count: Number of samples

    Returns:
        Dict with recommendation details
    """
    if sample_count < 3:
        return {
            "event_type": event_type,
            "recommendation": "Need more data",
            "confidence": "low",
            "adjustment": 0,
            "reason": f"Only {sample_count} samples. Need at least 3 for recommendation."
        }

    confidence = "medium" if sample_count < 10 else "high"

    if avg_prediction_error > 1.5:
        return {
            "event_type": event_type,
            "recommendation": "Reduce weight",
            "confidence": confidence,
            "adjustment": -1,
            "reason": f"Over-predicting by {avg_prediction_error:.1f} points on average"
        }
    elif avg_prediction_error < -1.5:
        return {
            "event_type": event_type,
            "recommendation": "Increase weight",
            "confidence": confidence,
            "adjustment": +1,
            "reason": f"Under-predicting by {abs(avg_prediction_error):.1f} points on average"
        }
    else:
        return {
            "event_type": event_type,
            "recommendation": "Keep current weight",
            "confidence": confidence,
            "adjustment": 0,
            "reason": f"Prediction error is acceptable ({avg_prediction_error:+.1f} points)"
        }


def is_market_hours() -> bool:
    """Check if US markets are currently open."""
    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Weekday check (Monday=0, Sunday=6)
    if now_et.weekday() >= 5:
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close


def get_historical_prices_for_alert(
    symbol: str,
    alert_time: datetime
) -> dict:
    """
    Get historical prices for backtesting an alert.
    Returns prices at alert time, +1h, and +1d.

    Note: yfinance limitations:
    - Intraday (hourly) data: only ~7 days back
    - Daily data: available for years

    For older alerts, we approximate using daily open/close.
    """
    if not YFINANCE_AVAILABLE:
        return {"error": "yfinance not available"}

    try:
        ticker = yf.Ticker(symbol)

        # Make alert_time timezone-aware if it isn't
        if alert_time.tzinfo is None:
            alert_time = alert_time.replace(tzinfo=ZoneInfo("UTC"))

        now = datetime.now(ZoneInfo("UTC"))
        days_ago = (now - alert_time).days

        result = {
            "symbol": symbol,
            "alert_time": alert_time.isoformat(),
            "price_at_alert": None,
            "price_1h": None,
            "price_1d": None,
            "data_quality": "unknown"
        }

        # Determine what data we can get
        if days_ago <= 7:
            # Can use intraday data for better precision
            hist = ticker.history(period="7d", interval="1h")
            result["data_quality"] = "hourly"
        elif days_ago <= 60:
            # Use daily data
            hist = ticker.history(period="3mo", interval="1d")
            result["data_quality"] = "daily"
        else:
            # Older data
            hist = ticker.history(period="1y", interval="1d")
            result["data_quality"] = "daily"

        if hist.empty:
            result["error"] = "No historical data available"
            return result

        # Convert index to UTC for comparison
        hist.index = hist.index.tz_convert("UTC")

        # Find price at alert time
        time_diffs = abs(hist.index - alert_time)
        if len(time_diffs) > 0:
            nearest_idx = time_diffs.argmin()
            result["price_at_alert"] = float(hist['Close'].iloc[nearest_idx])

        # Find price at +1h (if hourly data available)
        time_1h = alert_time + timedelta(hours=1)
        if time_1h <= now:
            time_diffs_1h = abs(hist.index - time_1h)
            if len(time_diffs_1h) > 0:
                nearest_idx_1h = time_diffs_1h.argmin()
                result["price_1h"] = float(hist['Close'].iloc[nearest_idx_1h])

        # Find price at +1d
        time_1d = alert_time + timedelta(days=1)
        if time_1d <= now:
            time_diffs_1d = abs(hist.index - time_1d)
            if len(time_diffs_1d) > 0:
                nearest_idx_1d = time_diffs_1d.argmin()
                result["price_1d"] = float(hist['Close'].iloc[nearest_idx_1d])

        return result

    except Exception as e:
        logger.error(f"Error fetching historical prices for {symbol}: {e}")
        return {"error": str(e)}
