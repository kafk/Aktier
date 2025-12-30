import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import get_db, init_db, seed_classification_rules, Stock, Keyword, NewsArticle, Alert, Note, ClassificationRule, PriceTracking
from .schemas import (
    StockCreate,
    StockResponse,
    KeywordCreate,
    KeywordResponse,
    NewsArticleResponse,
    AlertResponse,
    AlertMarkRead,
    ScrapeStatus,
    ScrapeIntervalUpdate,
    NoteUpdate,
    NoteResponse,
    ClassificationRuleCreate,
    ClassificationRuleResponse,
    PriceTrackingResponse,
    BacktestStats,
    WeightRecommendation,
)
from .scraper import scrape_all_sources, check_keywords, get_news_cutoff_date
from .scorer import calculate_impact_score, score_breakdown_to_json
from .classifier import classify_event
from .price_tracker import (
    get_current_price,
    get_price_at_time,
    calculate_price_change,
    price_change_to_impact,
    get_weight_adjustment_recommendation,
)

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler
scheduler: Optional[AsyncIOScheduler] = None
last_scrape_time: Optional[datetime] = None
scrape_interval_minutes: int = 5  # Default: every 5 minutes

# Swedish timezone for schedule
SWEDISH_TZ = ZoneInfo("Europe/Stockholm")
SCRAPE_START_TIME = time(8, 30)  # 08:30
SCRAPE_END_TIME = time(22, 0)    # 22:00


def is_within_scrape_hours() -> bool:
    """Check if current time is within allowed scraping hours (08:30-22:00 Swedish time)."""
    now_swedish = datetime.now(SWEDISH_TZ).time()
    return SCRAPE_START_TIME <= now_swedish <= SCRAPE_END_TIME


async def cleanup_old_news():
    """Remove news articles older than yesterday."""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        cutoff = get_news_cutoff_date()
        # Delete old alerts first (due to foreign key)
        old_articles = db.query(NewsArticle).filter(NewsArticle.fetched_at < cutoff).all()
        old_article_ids = [a.id for a in old_articles]

        if old_article_ids:
            db.query(Alert).filter(Alert.article_id.in_(old_article_ids)).delete(synchronize_session=False)
            db.query(NewsArticle).filter(NewsArticle.id.in_(old_article_ids)).delete(synchronize_session=False)
            db.commit()
            logger.info(f"Cleaned up {len(old_article_ids)} old articles")
    except Exception as e:
        logger.error(f"Error cleaning up old news: {e}")
        db.rollback()
    finally:
        db.close()


async def scrape_news_job(force: bool = False):
    """Background job to scrape news for all monitored stocks."""
    global last_scrape_time

    # Check if within allowed hours (skip check if forced)
    if not force and not is_within_scrape_hours():
        logger.info("Outside scraping hours (08:30-22:00 Swedish time), skipping...")
        return

    # Cleanup old news first
    await cleanup_old_news()

    from .database import SessionLocal

    db = SessionLocal()
    try:
        # Get all active stocks
        stocks = db.query(Stock).filter(Stock.active == True).all()
        if not stocks:
            logger.info("No active stocks to monitor")
            return

        # Get all active keywords
        keywords = db.query(Keyword).filter(Keyword.active == True).all()
        keyword_list = [k.word for k in keywords]

        # Get custom classification rules
        custom_rules = db.query(ClassificationRule).filter(ClassificationRule.active == True).all()
        custom_rules_list = [
            {"event_type": r.event_type, "keywords": r.keywords, "sentiment": r.sentiment, "active": r.active}
            for r in custom_rules
        ]

        logger.info(f"Scraping news for {len(stocks)} stocks with {len(keyword_list)} keywords")

        for stock in stocks:
            try:
                articles = await scrape_all_sources(stock.symbol)

                for article_data in articles:
                    # Check if article already exists
                    existing = db.query(NewsArticle).filter(NewsArticle.url == article_data.url).first()
                    if existing:
                        continue

                    # Save article
                    article = NewsArticle(
                        title=article_data.title,
                        summary=article_data.summary,
                        url=article_data.url,
                        source=article_data.source,
                        published_at=article_data.published_at,
                        stock_symbol=stock.symbol,
                    )
                    db.add(article)
                    db.flush()

                    # Check for keyword matches
                    text_to_check = f"{article_data.title} {article_data.summary or ''}"
                    matched_keywords = check_keywords(text_to_check, keyword_list)

                    if matched_keywords:
                        # Classify the event
                        event_type, sentiment = classify_event(
                            article_data.title,
                            article_data.summary or "",
                            custom_rules_list
                        )

                        # Calculate impact score
                        impact_score, score_breakdown = calculate_impact_score(
                            event_type=event_type,
                            sentiment=sentiment,
                            title=article_data.title,
                            summary=article_data.summary or "",
                            source=article_data.source,
                            url=article_data.url,
                            stock_symbol=stock.symbol
                        )

                        alert = Alert(
                            stock_id=stock.id,
                            article_id=article.id,
                            matched_keywords=", ".join(matched_keywords),
                            event_type=event_type,
                            sentiment=sentiment,
                            impact_score=impact_score,
                            score_breakdown=score_breakdown_to_json(score_breakdown),
                        )
                        db.add(alert)
                        logger.info(f"Alert created for {stock.symbol}: {matched_keywords} (event: {event_type}, score: {impact_score})")

                db.commit()
            except Exception as e:
                logger.error(f"Error scraping {stock.symbol}: {e}")
                db.rollback()

        last_scrape_time = datetime.now(SWEDISH_TZ)
        logger.info("Scraping job completed")

    except Exception as e:
        logger.error(f"Error in scrape job: {e}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler

    # Initialize database
    init_db()
    seed_classification_rules()
    logger.info("Database initialized")

    # Start scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scrape_news_job,
        trigger=IntervalTrigger(minutes=5),
        id="scrape_news",
        name="Scrape news every 5 minutes",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started - scraping every 5 minutes")

    # Run initial scrape
    asyncio.create_task(scrape_news_job())

    yield

    # Shutdown
    if scheduler:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


app = FastAPI(
    title="Stock News Monitor",
    description="Monitor stock news and get alerts for keywords",
    version="1.5.0",
    lifespan=lifespan,
)

# Serve static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(static_path, "index.html"))


@app.get("/classifications")
async def classifications_page():
    """Serve the classifications management page."""
    return FileResponse(os.path.join(static_path, "classifications.html"))


# ============== Stock Endpoints ==============


@app.get("/api/stocks", response_model=list[StockResponse])
def get_stocks(db: Session = Depends(get_db)):
    """Get all monitored stocks."""
    return db.query(Stock).all()


@app.post("/api/stocks", response_model=StockResponse)
def add_stock(stock: StockCreate, db: Session = Depends(get_db)):
    """Add a new stock to monitor."""
    symbol = stock.symbol.upper().strip()

    existing = db.query(Stock).filter(Stock.symbol == symbol).first()
    if existing:
        if not existing.active:
            existing.active = True
            db.commit()
            db.refresh(existing)
            return existing
        raise HTTPException(status_code=400, detail="Stock already exists")

    db_stock = Stock(symbol=symbol, name=stock.name)
    db.add(db_stock)
    db.commit()
    db.refresh(db_stock)
    return db_stock


@app.delete("/api/stocks/{stock_id}")
def delete_stock(stock_id: int, db: Session = Depends(get_db)):
    """Remove a stock from monitoring."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # Delete related alerts first
    db.query(Alert).filter(Alert.stock_id == stock_id).delete()
    # Delete the stock
    db.delete(stock)
    db.commit()
    return {"message": "Stock removed"}


@app.put("/api/stocks/{stock_id}/toggle")
def toggle_stock(stock_id: int, db: Session = Depends(get_db)):
    """Toggle stock active status."""
    stock = db.query(Stock).filter(Stock.id == stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    stock.active = not stock.active
    db.commit()
    db.refresh(stock)
    return {"active": stock.active}


# ============== Keyword Endpoints ==============


@app.get("/api/keywords", response_model=list[KeywordResponse])
def get_keywords(db: Session = Depends(get_db)):
    """Get all keywords."""
    return db.query(Keyword).all()


@app.post("/api/keywords", response_model=KeywordResponse)
def add_keyword(keyword: KeywordCreate, db: Session = Depends(get_db)):
    """Add a new keyword to monitor."""
    word = keyword.word.strip().lower()

    existing = db.query(Keyword).filter(Keyword.word == word).first()
    if existing:
        if not existing.active:
            existing.active = True
            db.commit()
            db.refresh(existing)
            # Scan existing articles for this reactivated keyword
            scan_existing_articles_for_keyword(db, word)
            return existing
        raise HTTPException(status_code=400, detail="Keyword already exists")

    db_keyword = Keyword(word=word)
    db.add(db_keyword)
    db.commit()
    db.refresh(db_keyword)

    # Scan existing articles for this new keyword
    scan_existing_articles_for_keyword(db, word)

    return db_keyword


def scan_existing_articles_for_keyword(db: Session, keyword: str):
    """Scan all existing articles for a keyword and create alerts."""
    from .scraper import check_keywords

    # Get all articles
    articles = db.query(NewsArticle).all()
    # Get all stocks (to link alerts)
    stocks = {s.symbol: s for s in db.query(Stock).all()}

    # Get custom classification rules
    custom_rules = db.query(ClassificationRule).filter(ClassificationRule.active == True).all()
    custom_rules_list = [
        {"event_type": r.event_type, "keywords": r.keywords, "sentiment": r.sentiment, "active": r.active}
        for r in custom_rules
    ]

    alerts_created = 0
    for article in articles:
        text_to_check = f"{article.title} {article.summary or ''}"
        if keyword.lower() in text_to_check.lower():
            # Check if alert already exists for this article
            stock = stocks.get(article.stock_symbol)
            if stock:
                existing_alert = db.query(Alert).filter(
                    Alert.article_id == article.id,
                    Alert.stock_id == stock.id
                ).first()

                if existing_alert:
                    # Update matched keywords if not already included
                    if keyword.lower() not in existing_alert.matched_keywords.lower():
                        existing_alert.matched_keywords += f", {keyword}"
                else:
                    # Classify the event
                    event_type, sentiment = classify_event(
                        article.title,
                        article.summary or "",
                        custom_rules_list
                    )

                    # Calculate impact score
                    impact_score, score_breakdown = calculate_impact_score(
                        event_type=event_type,
                        sentiment=sentiment,
                        title=article.title,
                        summary=article.summary or "",
                        source=article.source,
                        url=article.url,
                        stock_symbol=stock.symbol
                    )

                    # Create new alert
                    alert = Alert(
                        stock_id=stock.id,
                        article_id=article.id,
                        matched_keywords=keyword,
                        event_type=event_type,
                        sentiment=sentiment,
                        impact_score=impact_score,
                        score_breakdown=score_breakdown_to_json(score_breakdown),
                    )
                    db.add(alert)
                    alerts_created += 1

    if alerts_created > 0:
        db.commit()
        logger.info(f"Created {alerts_created} alerts for keyword '{keyword}'")


@app.delete("/api/keywords/{keyword_id}")
def delete_keyword(keyword_id: int, db: Session = Depends(get_db)):
    """Remove a keyword."""
    keyword = db.query(Keyword).filter(Keyword.id == keyword_id).first()
    if not keyword:
        raise HTTPException(status_code=404, detail="Keyword not found")

    # Delete the keyword
    db.delete(keyword)
    db.commit()
    return {"message": "Keyword removed"}


# ============== News & Alerts Endpoints ==============


@app.get("/api/news", response_model=list[NewsArticleResponse])
def get_news(
    stock: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    """Get news articles (only today and yesterday)."""
    cutoff = get_news_cutoff_date()
    query = db.query(NewsArticle).filter(
        NewsArticle.fetched_at >= cutoff
    ).order_by(NewsArticle.fetched_at.desc())

    if stock:
        query = query.filter(NewsArticle.stock_symbol == stock.upper())

    return query.limit(limit).all()


@app.get("/api/alerts", response_model=list[AlertResponse])
def get_alerts(
    unread_only: bool = Query(False),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    """Get keyword match alerts."""
    query = db.query(Alert).order_by(Alert.created_at.desc())

    if unread_only:
        query = query.filter(Alert.is_read == False)

    return query.limit(limit).all()


@app.put("/api/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int, body: AlertMarkRead, db: Session = Depends(get_db)):
    """Mark an alert as read/unread."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_read = body.is_read
    db.commit()
    return {"is_read": alert.is_read}


@app.put("/api/alerts/read-all")
def mark_all_alerts_read(db: Session = Depends(get_db)):
    """Mark all alerts as read."""
    db.query(Alert).filter(Alert.is_read == False).update({"is_read": True})
    db.commit()
    return {"message": "All alerts marked as read"}


@app.get("/api/alerts/unnotified", response_model=list[AlertResponse])
def get_unnotified_alerts(db: Session = Depends(get_db)):
    """Get alerts that haven't been sent as notifications yet."""
    return db.query(Alert).filter(Alert.is_notified == False).order_by(Alert.created_at.desc()).all()


@app.put("/api/alerts/{alert_id}/notified")
def mark_alert_notified(alert_id: int, db: Session = Depends(get_db)):
    """Mark an alert as notified (notification was sent)."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_notified = True
    db.commit()
    return {"is_notified": alert.is_notified}


@app.put("/api/alerts/notified-all")
def mark_all_alerts_notified(db: Session = Depends(get_db)):
    """Mark all unnotified alerts as notified."""
    count = db.query(Alert).filter(Alert.is_notified == False).update({"is_notified": True})
    db.commit()
    return {"message": f"Marked {count} alerts as notified"}


# ============== Status Endpoints ==============


@app.get("/api/status", response_model=ScrapeStatus)
def get_status(db: Session = Depends(get_db)):
    """Get current monitoring status."""
    global last_scrape_time, scheduler, scrape_interval_minutes

    next_scrape = None
    if scheduler:
        job = scheduler.get_job("scrape_news")
        if job and job.next_run_time:
            next_scrape = job.next_run_time

    return ScrapeStatus(
        last_scrape=last_scrape_time,
        next_scrape=next_scrape,
        stocks_monitored=db.query(Stock).filter(Stock.active == True).count(),
        keywords_active=db.query(Keyword).filter(Keyword.active == True).count(),
        total_alerts=db.query(Alert).count(),
        unread_alerts=db.query(Alert).filter(Alert.is_read == False).count(),
        within_scrape_hours=is_within_scrape_hours(),
        scrape_hours="08:30-22:00 (Swedish time)",
        scrape_interval_minutes=scrape_interval_minutes,
    )


@app.post("/api/scrape-now")
async def scrape_now():
    """Trigger an immediate scrape (bypasses time restrictions)."""
    asyncio.create_task(scrape_news_job(force=True))
    return {"message": "Scrape started"}


@app.put("/api/settings/interval")
def update_scrape_interval(body: ScrapeIntervalUpdate):
    """Update the scraping interval."""
    global scrape_interval_minutes, scheduler

    # Validate interval (minimum 1 minute, maximum 60 minutes)
    if body.interval_minutes < 1 or body.interval_minutes > 60:
        raise HTTPException(status_code=400, detail="Interval must be between 1 and 60 minutes")

    scrape_interval_minutes = body.interval_minutes

    # Reschedule the job with new interval
    if scheduler:
        scheduler.reschedule_job(
            "scrape_news",
            trigger=IntervalTrigger(minutes=scrape_interval_minutes)
        )
        logger.info(f"Scraping interval updated to {scrape_interval_minutes} minutes")

    return {"interval_minutes": scrape_interval_minutes}


# ============== Notes Endpoints ==============


@app.get("/api/notes", response_model=NoteResponse)
def get_notes(db: Session = Depends(get_db)):
    """Get the notepad content."""
    note = db.query(Note).first()
    if not note:
        # Create default empty note
        note = Note(content="")
        db.add(note)
        db.commit()
        db.refresh(note)
    return note


@app.put("/api/notes", response_model=NoteResponse)
def update_notes(body: NoteUpdate, db: Session = Depends(get_db)):
    """Update the notepad content."""
    note = db.query(Note).first()
    if not note:
        note = Note(content=body.content)
        db.add(note)
    else:
        note.content = body.content
    db.commit()
    db.refresh(note)
    return note


# ============== Classification Rules Endpoints ==============


@app.get("/api/classifications", response_model=list[ClassificationRuleResponse])
def get_classification_rules(db: Session = Depends(get_db)):
    """Get all classification rules."""
    return db.query(ClassificationRule).order_by(ClassificationRule.display_name).all()


@app.post("/api/classifications", response_model=ClassificationRuleResponse)
def create_classification_rule(rule: ClassificationRuleCreate, db: Session = Depends(get_db)):
    """Create a new classification rule."""
    # Check if event_type already exists
    existing = db.query(ClassificationRule).filter(
        ClassificationRule.event_type == rule.event_type.upper()
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Event type already exists")

    db_rule = ClassificationRule(
        event_type=rule.event_type.upper().replace(" ", "_"),
        display_name=rule.display_name,
        keywords=rule.keywords.lower(),
        sentiment=rule.sentiment,
        is_builtin=False,
    )
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    return db_rule


@app.delete("/api/classifications/{rule_id}")
def delete_classification_rule(rule_id: int, db: Session = Depends(get_db)):
    """Delete a classification rule (only non-builtin)."""
    rule = db.query(ClassificationRule).filter(ClassificationRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.is_builtin:
        raise HTTPException(status_code=400, detail="Cannot delete built-in rules")

    db.delete(rule)
    db.commit()
    return {"message": "Rule deleted"}


@app.put("/api/classifications/{rule_id}/toggle")
def toggle_classification_rule(rule_id: int, db: Session = Depends(get_db)):
    """Toggle a classification rule active/inactive."""
    rule = db.query(ClassificationRule).filter(ClassificationRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    rule.active = not rule.active
    db.commit()
    return {"active": rule.active}


# ============ Backtesting Endpoints ============

@app.get("/backtesting")
async def backtesting_page():
    """Serve the backtesting page."""
    return FileResponse(os.path.join(static_path, "backtesting.html"))


@app.get("/api/backtesting/tracking", response_model=list[PriceTrackingResponse])
def get_price_tracking(
    limit: int = Query(default=50, le=200),
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all price tracking records."""
    query = db.query(PriceTracking).order_by(PriceTracking.alert_time.desc())

    if status:
        query = query.filter(PriceTracking.status == status)

    return query.limit(limit).all()


@app.post("/api/backtesting/track/{alert_id}")
def start_tracking_alert(alert_id: int, db: Session = Depends(get_db)):
    """Start price tracking for a specific alert."""
    # Check if alert exists
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Check if already tracking
    existing = db.query(PriceTracking).filter(PriceTracking.alert_id == alert_id).first()
    if existing:
        return {"message": "Already tracking", "tracking": existing}

    # Get current price
    stock = db.query(Stock).filter(Stock.id == alert.stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    current_price = get_current_price(stock.symbol)

    # Create tracking record
    tracking = PriceTracking(
        alert_id=alert_id,
        stock_symbol=stock.symbol,
        price_at_alert=current_price,
        alert_time=alert.created_at,
        status="pending" if current_price else "error"
    )
    db.add(tracking)
    db.commit()
    db.refresh(tracking)

    return {"message": "Tracking started", "tracking": tracking}


@app.post("/api/backtesting/track-all-untracked")
def track_all_untracked_alerts(db: Session = Depends(get_db)):
    """Start tracking for all alerts that don't have tracking yet."""
    # Get alerts without tracking
    tracked_alert_ids = db.query(PriceTracking.alert_id).subquery()
    untracked_alerts = db.query(Alert).filter(
        ~Alert.id.in_(tracked_alert_ids)
    ).all()

    tracked_count = 0
    for alert in untracked_alerts:
        stock = db.query(Stock).filter(Stock.id == alert.stock_id).first()
        if not stock:
            continue

        current_price = get_current_price(stock.symbol)

        tracking = PriceTracking(
            alert_id=alert.id,
            stock_symbol=stock.symbol,
            price_at_alert=current_price,
            alert_time=alert.created_at,
            status="pending" if current_price else "error"
        )
        db.add(tracking)
        tracked_count += 1

    db.commit()
    return {"message": f"Started tracking {tracked_count} alerts"}


@app.post("/api/backtesting/update-prices")
def update_tracked_prices(db: Session = Depends(get_db)):
    """Update prices for all pending/partial tracking records."""
    from datetime import timedelta

    now = datetime.utcnow()
    updated_count = 0

    # Get all tracking records that need updates
    tracking_records = db.query(PriceTracking).filter(
        PriceTracking.status.in_(["pending", "partial"])
    ).all()

    for tracking in tracking_records:
        alert = db.query(Alert).filter(Alert.id == tracking.alert_id).first()
        if not alert:
            continue

        alert_time = tracking.alert_time
        time_since_alert = now - alert_time

        # Update 1h price if enough time has passed and not yet recorded
        if time_since_alert >= timedelta(hours=1) and tracking.price_1h is None:
            price_1h = get_current_price(tracking.stock_symbol)
            if price_1h and tracking.price_at_alert:
                tracking.price_1h = price_1h
                _, tracking.change_1h_percent = calculate_price_change(
                    tracking.price_at_alert, price_1h
                )
                tracking.tracked_1h_at = now
                tracking.status = "partial"
                updated_count += 1

        # Update 1d price if enough time has passed and not yet recorded
        if time_since_alert >= timedelta(days=1) and tracking.price_1d is None:
            price_1d = get_current_price(tracking.stock_symbol)
            if price_1d and tracking.price_at_alert:
                tracking.price_1d = price_1d
                _, tracking.change_1d_percent = calculate_price_change(
                    tracking.price_at_alert, price_1d
                )
                tracking.tracked_1d_at = now

                # Calculate actual impact based on 1d change
                tracking.actual_impact = price_change_to_impact(
                    tracking.change_1d_percent,
                    alert.sentiment or "neutral"
                )

                # Calculate prediction error
                if alert.impact_score:
                    tracking.prediction_error = alert.impact_score - tracking.actual_impact

                tracking.status = "complete"
                updated_count += 1

    db.commit()
    return {"message": f"Updated {updated_count} tracking records"}


@app.get("/api/backtesting/stats")
def get_backtest_stats(db: Session = Depends(get_db)):
    """Get backtesting statistics."""
    from sqlalchemy import func

    # Total tracked
    total_tracked = db.query(PriceTracking).count()

    # Complete tracking
    complete = db.query(PriceTracking).filter(
        PriceTracking.status == "complete"
    ).all()
    complete_count = len(complete)

    # Calculate averages and counts
    over_predictions = 0
    under_predictions = 0
    accurate = 0
    total_error = 0
    event_stats = {}

    for tracking in complete:
        alert = db.query(Alert).filter(Alert.id == tracking.alert_id).first()
        if not alert or tracking.prediction_error is None:
            continue

        error = tracking.prediction_error
        total_error += error

        if error > 1:
            over_predictions += 1
        elif error < -1:
            under_predictions += 1
        else:
            accurate += 1

        # Group by event type
        event_type = alert.event_type or "UNKNOWN"
        if event_type not in event_stats:
            event_stats[event_type] = {"count": 0, "total_error": 0, "errors": []}
        event_stats[event_type]["count"] += 1
        event_stats[event_type]["total_error"] += error
        event_stats[event_type]["errors"].append(error)

    # Calculate averages per event type
    for event_type, stats in event_stats.items():
        if stats["count"] > 0:
            stats["avg_error"] = round(stats["total_error"] / stats["count"], 2)
        else:
            stats["avg_error"] = 0
        del stats["errors"]  # Remove raw data

    avg_error = round(total_error / complete_count, 2) if complete_count > 0 else None

    return {
        "total_tracked": total_tracked,
        "complete_tracking": complete_count,
        "avg_prediction_error": avg_error,
        "over_predictions": over_predictions,
        "under_predictions": under_predictions,
        "accurate_predictions": accurate,
        "event_type_stats": event_stats
    }


@app.get("/api/backtesting/recommendations", response_model=list[WeightRecommendation])
def get_weight_recommendations(db: Session = Depends(get_db)):
    """Get recommendations for adjusting event weights based on backtest data."""
    # Get complete tracking with alerts
    complete = db.query(PriceTracking).filter(
        PriceTracking.status == "complete"
    ).all()

    # Group by event type
    event_errors = {}
    for tracking in complete:
        alert = db.query(Alert).filter(Alert.id == tracking.alert_id).first()
        if not alert or tracking.prediction_error is None:
            continue

        event_type = alert.event_type or "UNKNOWN"
        if event_type not in event_errors:
            event_errors[event_type] = []
        event_errors[event_type].append(tracking.prediction_error)

    # Generate recommendations
    recommendations = []
    for event_type, errors in event_errors.items():
        avg_error = sum(errors) / len(errors) if errors else 0
        rec = get_weight_adjustment_recommendation(event_type, avg_error, len(errors))
        rec["sample_count"] = len(errors)
        rec["avg_error"] = round(avg_error, 2)
        recommendations.append(rec)

    # Sort by confidence and magnitude of recommendation
    recommendations.sort(key=lambda x: (
        x["confidence"] != "high",
        x["adjustment"] == 0,
        abs(x["avg_error"])
    ))

    return recommendations


@app.post("/api/backtesting/manual-entry/{tracking_id}")
def manual_price_entry(
    tracking_id: int,
    price_1h: Optional[float] = None,
    price_1d: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Manually enter price data for a tracking record."""
    tracking = db.query(PriceTracking).filter(PriceTracking.id == tracking_id).first()
    if not tracking:
        raise HTTPException(status_code=404, detail="Tracking record not found")

    alert = db.query(Alert).filter(Alert.id == tracking.alert_id).first()

    if price_1h is not None and tracking.price_at_alert:
        tracking.price_1h = price_1h
        _, tracking.change_1h_percent = calculate_price_change(
            tracking.price_at_alert, price_1h
        )
        tracking.tracked_1h_at = datetime.utcnow()
        if tracking.status == "pending":
            tracking.status = "partial"

    if price_1d is not None and tracking.price_at_alert:
        tracking.price_1d = price_1d
        _, tracking.change_1d_percent = calculate_price_change(
            tracking.price_at_alert, price_1d
        )
        tracking.tracked_1d_at = datetime.utcnow()

        # Calculate actual impact
        tracking.actual_impact = price_change_to_impact(
            tracking.change_1d_percent,
            alert.sentiment if alert else "neutral"
        )

        # Calculate prediction error
        if alert and alert.impact_score:
            tracking.prediction_error = alert.impact_score - tracking.actual_impact

        tracking.status = "complete"

    db.commit()
    db.refresh(tracking)
    return tracking


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
