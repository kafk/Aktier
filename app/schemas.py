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
