import sys
import json
from pathlib import Path
from datetime import date, datetime, timezone
from pandas.core.ops.docstrings import key
import yfinance as yf
from loguru import logger
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import TICKERS, TICKER_NSE_MAP, RAW_SENTIMENT_DIR
from config.database import insert_df

def extract_article(article: dict, ticker: str, run_date: str) -> dict | None:
    try:
        content = article.get("content", article)
        
        title = content.get("title", "")
        summary = content.get("summary", "") or content.get("description", "")
        full_text = f"{title} {summary}".strip()
        
        if len(full_text) < 15:
            return None
        
        pub_date_raw = content.get("pubDate", "")
        try:
            published_at = datetime.fromisoformat(
                pub_date_raw.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:%M:%S")
            
        except (ValueError, AttributeError):
            published_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
        provider = content.get("provider", {})
        author = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
        
        url_obj = content.get("canonicalUrl", {})
        url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
        
        article_id = article.get("id", url)[:180]
        
        return {
            "source": "yahoo_finance",
            "source_id": f"{article_id}_{ticker}",
            "ticker": ticker,
            "published_at": published_at,
            "collected_date": run_date,
            "raw_text": full_text[:4000],
            "author": author,
            "upvotes": 0,
            "url": url,
        }
    except Exception as e:
        logger.warning(f"[yf_news] Article extraction error for {ticker}: {e}")
        return None
    
def fetch_ticker_name(ticker: str, run_date: str) -> list[dict]:
    ticker_nse = TICKER_NSE_MAP[ticker]
    records: list[dict] = []
    
    try:
        yf_ticker = yf.Ticker(ticker_nse)
        articles = yf_ticker.news or []
        
        for article in articles:
            record = extract_article(article, ticker, run_date)
            if record:
                records.append(record)
    
    except Exception as e:
        logger.error(f"[yf_news] Failed to fetch news for {ticker_nse}: {e}")
        
    return records

def collect_yfinance_news(run_date: str = None) -> dict[str, int]:
    if run_date is None:
        run_date = date.today().isoformat()
        
    logger.info(f"[tf_news] Starting Yahoo Finance news collection for {run_date}")
    
    all_records: list[dict] = []
    results: dict[str, int] = {}
    seen_ids: set[str] = set()
    
    for ticker in TICKERS:
        records = fetch_ticker_name(ticker, run_date)
        
        new_records = [r for r in records if r["source_id"] not in seen_ids]
        for r in new_records:
            seen_ids.add(r["source_id"])
            
        all_records.extend(new_records)
        results[ticker] = len(new_records)
        
        if new_records:
            logger.info(f"[yf_news] {ticker}: {len(new_records)} articles")
            
    raw_path = RAW_SENTIMENT_DIR / f"yfinance_news_{run_date}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f ,indent=2, ensure_ascii=False, default=str)
    logger.info(f"[yf_news] Saved {len(all_records)} articles -> {raw_path}")
    
    if all_records:
        df = pd.DataFrame(all_records)
        insert_df("raw_sentiment", df, mode="ignore")
        
    total = sum(results.values())
    logger.info(f"[yf_news] Collection complete: {total} articles across {len(TICKERS)} tickers")
    
    return results

if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = collect_yfinance_news(run_date)
    for ticker, count in sorted(result.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"{ticker}: {count} articles")
            