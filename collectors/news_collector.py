import sys
import re
import json
import socket
from pathlib import Path
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import feedparser
from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd

socket.setdefaulttimeout(10.0)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    TICKERS, TICKER_KEYWORDS, NEWS_RSS_FEEDS, RAW_NEWS_DIR
)
from config.database import insert_df

def clean_html(text:str) -> str:
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(
            separator=" ", strip=True
        )
    except Exception:
        return text
    
def parse_date(entry) -> str:
    for field in ["published", "updated", "created"]:
        raw = getattr(entry, field, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def match_tickers_in_text(text: str) -> list[str]:
    text_upper = text.upper()
    matched: list[str] = []
    
    for ticker, keywords in TICKER_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.upper()) + r'\b'
            if re.search(pattern, text_upper):
                matched.append(ticker)
                break
    
    return list(set(matched))

def parse_feed(feed_url: str, run_date: str) -> list[dict]:
    records: list[dict] = []
    
    try:
        feed = feedparser.parse(feed_url)
        
        if feed.bozo and not feed.entries:
            logger.warning(f"[news] Malformed feed: {feed_url}")
            return []
        
        feed_name = feed.feed.get("title", feed_url[:50])
        logger.info(f"[news] Parsing '{feed_name}' - {len(feed.entries)} entries")
        
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = clean_html(entry.get("summary", ""))
            full_text = f"{title} {summary}".strip()
            
            if len(full_text) < 20:
                continue
            
            matched_tickers = match_tickers_in_text(full_text)
            if not matched_tickers:
                continue
            
            published_at = parse_date(entry)
            article_url = entry.get("link", "")
            article_id = entry.get("id", article_url)[:180]
            
            for ticker in matched_tickers:
                records.append({
                    "source": "news",
                    "source_id": f"{article_id}_{ticker}",
                    "ticker": ticker,
                    "published_at": published_at,
                    "collected_date": run_date,
                    "raw_text": full_text[:4000],
                    "author": entry.get("author", feed_name[:100]),
                    "upvotes": 0,
                    "url": article_url
                })
                
    except Exception as e:
        logger.error(f"[news] Feed error {feed_url}: {e}")
        
    return records

def collect_news_data(run_date: str = None) -> dict[str, int]:
    if run_date is None:
        run_date = date.today().isoformat()

    logger.info(f"[news] Starting news collection for {run_date}")

    all_records: list[dict] = []
    seen_ids: set[str] = set()

    for feed_url in NEWS_RSS_FEEDS:
        records = parse_feed(feed_url, run_date)
        for r in records:
            if r["source_id"] not in seen_ids:
                seen_ids.add(r["source_id"])
                all_records.append(r)

    raw_path = RAW_NEWS_DIR / f"news_{run_date}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"[news] Saved {len(all_records)} articles -> {raw_path}")

    ticker_counts: dict[str, int] = {t: 0 for t in TICKERS}
    for r in all_records:
        if r["ticker"] in ticker_counts:
            ticker_counts[r["ticker"]] += 1

    if all_records:
        df = pd.DataFrame(all_records)
        insert_df("raw_sentiment", df, mode="ignore")

    total = len(all_records)
    logger.info(f"[news] Collection complete: {total} articles inserted")
    return ticker_counts

if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = collect_news_data(run_date)
    for ticker, count in sorted(result.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"{ticker}: {count} articles")