from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class StockCreate(BaseModel):
    symbol: str
    name: Optional[str] = None


class StockResponse(BaseModel):
    id: int
    symbol: str
    name: Optional[str]
    active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class KeywordCreate(BaseModel):
    word: str


class KeywordResponse(BaseModel):
    id: int
    word: str
    active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class NewsArticleResponse(BaseModel):
    id: int
    title: str
    summary: Optional[str]
    url: str
    source: Optional[str]
    published_at: Optional[datetime]
    fetched_at: datetime
    stock_symbol: Optional[str]

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    id: int
    stock_id: int
    article_id: int
    matched_keywords: str
    event_type: Optional[str]
    sentiment: Optional[str]
    impact_score: Optional[int]
    score_breakdown: Optional[str]
    is_read: bool
    is_notified: bool
    created_at: datetime
    stock: Optional[StockResponse]
    article: Optional[NewsArticleResponse]

    class Config:
        from_attributes = True


class AlertMarkRead(BaseModel):
    is_read: bool = True


class ScrapeStatus(BaseModel):
    last_scrape: Optional[datetime]
    next_scrape: Optional[datetime]
    stocks_monitored: int
    keywords_active: int
    total_alerts: int
    unread_alerts: int
    within_scrape_hours: bool
    scrape_hours: str
    scrape_interval_minutes: int


class ScrapeIntervalUpdate(BaseModel):
    interval_minutes: int


class NoteUpdate(BaseModel):
    content: str


class NoteResponse(BaseModel):
    id: int
    content: str
    updated_at: datetime

    class Config:
        from_attributes = True


class ClassificationRuleCreate(BaseModel):
    event_type: str
    display_name: str
    keywords: str  # Comma-separated
    sentiment: str = "neutral"


class ClassificationRuleResponse(BaseModel):
    id: int
    event_type: str
    display_name: str
    keywords: str
    sentiment: str
    is_builtin: bool
    active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PriceTrackingResponse(BaseModel):
    id: int
    alert_id: int
    stock_symbol: str
    price_at_alert: Optional[float]
    price_1h: Optional[float]
    price_1d: Optional[float]

    # Market index prices
    index_at_alert: Optional[float]
    index_1d: Optional[float]

    # Raw percentage changes
    change_1h_percent: Optional[float]
    change_1d_percent: Optional[float]

    # Market-adjusted metrics
    stock_abs_move_pct: Optional[float]
    market_abs_move_pct: Optional[float]
    market_adjusted_move_pct: Optional[float]
    baseline_move_pct: Optional[float]
    news_impact_pct: Optional[float]
    impact_class: Optional[str]

    # Legacy fields
    actual_impact: Optional[int]
    prediction_error: Optional[int]
    status: str
    alert_time: datetime
    tracked_1h_at: Optional[datetime]
    tracked_1d_at: Optional[datetime]

    class Config:
        from_attributes = True


class PriceTrackingWithAlert(PriceTrackingResponse):
    """Price tracking with full alert details."""
    alert: Optional[AlertResponse]


class BacktestStats(BaseModel):
    """Statistics for backtesting analysis."""
    total_tracked: int
    complete_tracking: int
    avg_prediction_error: Optional[float]
    over_predictions: int
    under_predictions: int
    accurate_predictions: int
    event_type_stats: dict


class WeightRecommendation(BaseModel):
    """Recommendation for adjusting event weights."""
    event_type: str
    recommendation: str
    confidence: str
    adjustment: int
    reason: str
    sample_count: int
    avg_error: float
