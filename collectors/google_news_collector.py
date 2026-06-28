import sys
import time
import json
from pathlib import Path
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import feedparser
from bs4 import BeautifulSoup
from loguru import logger
from duckdb import query
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import TICKERS, TICKER_KEYWORDS, RAW_SENTIMENT_DIR
from config.database import insert_df

GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"
REQUEST_DELAY_SECONDS = 1.5

def clean_html(text: str) -> str:
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        return text
    
def parse_date(entry) -> str:
    for field in ["published", "updated"]:
        raw = getattr(entry, field, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def build_query_url(ticker: str) -> str:
    keywords = TICKER_KEYWORDS.get(ticker, [ticker])
    primary_name = keywords[0]
    query = f'"{primary_name}" stock OR share OR shares OR NSE'
    encoded_query = quote_plus(query)
    return f"{GOOGLE_NEWS_BASE}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
def fetch_ticker_news(ticker: str, run_date: str) -> list[dict]:
    url = build_query_url(ticker)
    records: list[dict] = []
    
    try:
        feed = feedparser.parse(url)
        
        if feed.bozo and not feed.entries:
            logger.warning(f"[gnews] Malfound feed for {ticker}")
            return []
        
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = clean_html(entry.get("summary", ""))
            full_text = f"{title} {summary}".strip()
            
            if len(full_text) < 15:
                continue
            
            published_at = parse_date(entry)
            article_url = entry.get("link", "")
            article_id = entry.get("id", article_url)[:180]
            
            source_field = entry.get("source")
            author = (
                source_field.get("title", "Google News")
                if isinstance(source_field, dict)
                else "Google News"
            )
            
            records.append({
                "source": "google_news",
                "source_id": f"{article_id}_{ticker}",
                "ticker": ticker,
                "published_at": published_at,
                "collected_date": run_date,
                "raw_text": full_text[:4000],
                "author": author,
                "upvotes": 0,
                "url": article_url
            })
    
    except Exception as e:
        logger.error(f"[gnews] Failed to fetch {ticker}: {e}")
    
    return records

def collect_google_news(run_date: str = None) -> dict[str, int]:
    if run_date is None:
        run_date = date.today().isoformat()
        
    logger.info(f"[gnews] Starting Google News collection for {run_date}")
    
    all_records: list[dict] = []
    results: dict[str, int] = {}
    seen_ids: set[str] = set()
    
    for i, ticker in enumerate(TICKERS):
        records = fetch_ticker_news(ticker, run_date)
        
        new_records = [r for r in records if r["source_id"] not in seen_ids]
        for r in new_records:
            seen_ids.add(r["source_id"])
            
        all_records.extend(new_records)
        results[ticker] = len(new_records)
        
        if new_records:
            logger.info(f"[gnews] {ticker}: {len(new_records)} articles")
            
        if i < len(TICKERS) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)
            
    raw_path = RAW_SENTIMENT_DIR / f"google_news_{run_date}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"[gnews] Saved {len(all_records)} articles -> {raw_path}")
    
    if all_records:
        df = pd.DataFrame(all_records)
        insert_df("raw_sentiment", df, mode="ignore")
        
    total = sum(results.values())
    logger.info(f"[gnews] Collection complete: {total} articles across {len(TICKERS)} tickers")
    return results

if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = collect_google_news(run_date)
    for ticker, count in sorted(result.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"{ticker}: {count} articles")