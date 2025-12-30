import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stock_monitor.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts = relationship("Alert", back_populates="stock")


class Keyword(Base):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String(100), unique=True, index=True, nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    url = Column(String(1000), unique=True, nullable=False)
    source = Column(String(200), nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    stock_symbol = Column(String(20), index=True)

    alerts = relationship("Alert", back_populates="article")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    matched_keywords = Column(String(500), nullable=False)
    event_type = Column(String(50), nullable=True)  # e.g., EARNINGS_BEAT, GUIDANCE_CUT
    sentiment = Column(String(20), nullable=True)   # positive, negative, neutral
    impact_score = Column(Integer, nullable=True)   # 1-10 impact score
    score_breakdown = Column(Text, nullable=True)   # JSON breakdown of score components
    is_read = Column(Boolean, default=False)
    is_notified = Column(Boolean, default=False)    # Track if notification was sent
    created_at = Column(DateTime, default=datetime.utcnow)

    stock = relationship("Stock", back_populates="alerts")
    article = relationship("NewsArticle", back_populates="alerts")


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClassificationRule(Base):
    __tablename__ = "classification_rules"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), unique=True, nullable=False)  # e.g., INVESTMENT
    display_name = Column(String(100), nullable=False)  # e.g., "Investment"
    keywords = Column(Text, nullable=False)  # Comma-separated: "invests,investment,invested"
    sentiment = Column(String(20), nullable=False, default="neutral")  # positive, negative, neutral
    is_builtin = Column(Boolean, default=False)  # Built-in rules can't be deleted
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)


def seed_classification_rules():
    """Seed built-in classification rules into the database."""
    db = SessionLocal()
    try:
        # Check if any rules exist
        existing = db.query(ClassificationRule).first()
        if existing:
            return  # Already seeded

        # Built-in rules with their keywords
        builtin_rules = [
            ("EARNINGS_BEAT", "Earnings Beat", "beats,topped,exceeds,surpasses,earnings,eps,profit", "positive"),
            ("EARNINGS_MISS", "Earnings Miss", "misses,missed,falls short,below expectations,earnings miss", "negative"),
            ("GUIDANCE_RAISE", "Guidance Raised", "raises guidance,raised guidance,lifts outlook,boosts forecast", "positive"),
            ("GUIDANCE_CUT", "Guidance Cut", "cuts guidance,lowers outlook,weak guidance,disappointing forecast", "negative"),
            ("ACQUISITION", "Acquisition/Merger", "acquires,acquisition,to buy,takeover,merger,merge", "neutral"),
            ("CEO_CHANGE", "CEO Change", "ceo resigns,ceo steps down,new ceo,ceo departs,ceo fired", "neutral"),
            ("CFO_CHANGE", "CFO Change", "cfo resigns,cfo steps down,new cfo,cfo departs,cfo fired", "neutral"),
            ("LAYOFFS", "Layoffs", "layoffs,laying off,job cuts,workforce reduction,downsizing,restructuring", "negative"),
            ("FDA_APPROVAL", "FDA Approval", "fda approves,fda approved,fda approval,fda clears,fda cleared", "positive"),
            ("FDA_REJECTION", "FDA Rejection", "fda rejects,fda rejected,fda rejection,fda denies,fda denied", "negative"),
            ("SEC_INVESTIGATION", "SEC Investigation", "sec investigation,sec investigates,sec probe,sec inquiry,sec subpoena", "negative"),
            ("LAWSUIT", "Lawsuit", "lawsuit,sued,sues,suing,litigation,legal action,class action", "negative"),
            ("STOCK_BUYBACK", "Stock Buyback", "buyback,repurchase,buy back,share repurchase", "positive"),
            ("DIVIDEND", "Dividend", "dividend raises,dividend increases,dividend hikes,declares dividend", "positive"),
            ("STOCK_SPLIT", "Stock Split", "stock split,share split,split for", "positive"),
            ("UPGRADE", "Analyst Upgrade", "upgrades,upgraded,raises rating,upgrade to buy,price target raised", "positive"),
            ("DOWNGRADE", "Analyst Downgrade", "downgrades,downgraded,cuts rating,downgrade to sell,price target cut", "negative"),
            ("PRODUCT_LAUNCH", "Product Launch", "launches,launched,unveils,unveiled,introduces,new product,new service", "positive"),
            ("PARTNERSHIP", "Partnership", "partnership,partners with,teams up,collaboration,alliance,deal with", "positive"),
            ("BANKRUPTCY", "Bankruptcy", "bankruptcy,chapter 11,chapter 7,insolvency,insolvent", "negative"),
            ("IPO", "IPO", "ipo,initial public offering,goes public,going public,public debut", "neutral"),
            ("INSIDER_BUYING", "Insider Buying", "insider buys,insider buying,insider purchased,insider acquires", "positive"),
            ("INSIDER_SELLING", "Insider Selling", "insider sells,insider selling,insider sold,insider dumps", "negative"),
            ("INVESTMENT", "Investment", "invests,investment,invested,investing in", "neutral"),
        ]

        for event_type, display_name, keywords, sentiment in builtin_rules:
            rule = ClassificationRule(
                event_type=event_type,
                display_name=display_name,
                keywords=keywords,
                sentiment=sentiment,
                is_builtin=True,
                active=True
            )
            db.add(rule)

        db.commit()
        print("Seeded built-in classification rules")
    except Exception as e:
        print(f"Error seeding classification rules: {e}")
        db.rollback()
    finally:
        db.close()
