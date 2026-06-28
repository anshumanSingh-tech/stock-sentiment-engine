import sys
import argparse
from pathlib import Path
from datetime import date, timedelta
from typing import Optional
from loguru import logger
 
from prefect import task, flow
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import test_connection
 
 
 
@task(name="collect-market-data", retries=2, retry_delay_seconds=30)
def collect_market_task(run_date: str, lookback_days: int = 1) -> dict:
    """Fetches OHLCV price data for all 25 NSE tickers from yfinance."""
    from collectors.market_collector import collect_market_data
    logger.info(f"[flow] Starting market data collection for {run_date}")
    result = collect_market_data(run_date, lookback_days)
    total  = sum(result.values())
    logger.info(f"[flow] Market collection done: {total} rows across {len(result)} tickers")
    return result
 
 
@task(name="collect-google-news", retries=2, retry_delay_seconds=30)
def collect_google_news_task(run_date: str) -> dict:
    """Collects ticker-specific news via Google News RSS search."""
    from collectors.google_news_collector import collect_google_news
    logger.info(f"[flow] Starting Google News collection for {run_date}")
    result = collect_google_news(run_date)
    total  = sum(result.values())
    logger.info(f"[flow] Google News done: {total} articles")
    return result
 
 
@task(name="collect-yfinance-news", retries=2, retry_delay_seconds=30)
def collect_yfinance_news_task(run_date: str) -> dict:
    """Collects curated news directly from Yahoo Finance per ticker."""
    from collectors.yfinance_news_collector import collect_yfinance_news
    logger.info(f"[flow] Starting Yahoo Finance news collection for {run_date}")
    result = collect_yfinance_news(run_date)
    total  = sum(result.values())
    logger.info(f"[flow] Yahoo Finance news done: {total} articles")
    return result
 
 
@task(name="collect-news-sentiment", retries=2, retry_delay_seconds=30)
def collect_news_task(run_date: str) -> dict:
    """Fetches financial news from fixed Indian RSS feeds."""
    from collectors.news_collector import collect_news_data
    logger.info(f"[flow] Starting fixed RSS news collection for {run_date}")
    result = collect_news_data(run_date)
    total  = sum(result.values())
    logger.info(f"[flow] Fixed RSS news done: {total} articles")
    return result
 
 
@task(name="validate-collection", retries=0)
def validate_collection_task(run_date: str) -> bool:
    """Data quality gate - runs after collection, before scoring."""
    from validation.validator import run_collection_checks
    logger.info(f"[flow] Running collection validation for {run_date}")
    report = run_collection_checks(run_date)
    logger.info(
        f"[flow] Validation passed: "
        f"{report.passed}/{len(report.checks)} checks"
    )
    return True
 
 
@task(name="score-sentiment", retries=1, retry_delay_seconds=60)
def score_sentiment_task(run_date: str) -> dict:
    """Runs FinBERT on all unscored raw_sentiment records."""
    from nlp.finbert_scorer import score_raw_sentiment
    logger.info(f"[flow] Starting FinBERT scoring for {run_date}")
    result = score_raw_sentiment(run_date)
    logger.info(f"[flow] Scored {result.get('scored', 0)} records")
    return result
 
 
@task(name="aggregate-sentiment", retries=1, retry_delay_seconds=30)
def aggregate_sentiment_task(run_date: str) -> dict:
    """Aggregates individual FinBERT scores into daily_sentiment table."""
    from nlp.finbert_scorer import aggregate_daily_sentiment
    logger.info(f"[flow] Aggregating daily sentiment for {run_date}")
    result = aggregate_daily_sentiment(run_date)
    logger.info(f"[flow] Aggregated {result.get('aggregated', 0)} tickers")
    return result
 
 
@task(name="validate-scoring", retries=0)
def validate_scoring_task(run_date: str) -> bool:
    """Second quality gate - validates FinBERT output before correlation."""
    from validation.validator import (
        check_scored_sentiment, check_daily_sentiment, ValidationReport
    )
    logger.info(f"[flow] Validating scoring output for {run_date}")
    report = ValidationReport(run_date=run_date)
    report.checks.extend(check_scored_sentiment(run_date))
    report.checks.extend(check_daily_sentiment(run_date))
    report.print_summary()
    if not report.pipeline_ok:
        failures = [c.name for c in report.critical_failures]
        raise ValueError(f"Scoring validation failed: {failures}")
    return True
 
 
@task(name="run-correlation-analysis", retries=1, retry_delay_seconds=30)
def correlation_task(run_date: str, lookback_days: int = 90) -> dict:
    """Computes Pearson lag correlations using DuckDB window functions."""
    from analysis.correlation_engine import run_correlation_analysis
    logger.info(f"[flow] Running correlation analysis for {run_date}")
    result = run_correlation_analysis(run_date, lookback_days)
    logger.info(f"[flow] Correlations computed for {result.get('computed', 0)} tickers")
    return result
 
 
@task(name="pipeline-summary", retries=0)
def summary_task(
    run_date: str,
    market_result:       dict,
    google_news_result:  dict,
    yf_news_result:      dict,
    news_result:         dict,
    score_result:        dict,
    corr_result:         dict,
) -> None:
    from config.database import query_df
 
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE - {run_date}")
    logger.info("=" * 60)
    logger.info(f"  Market rows:           {sum(market_result.values())}")
    logger.info(f"  Google News articles:  {sum(google_news_result.values())}")
    logger.info(f"  Yahoo Finance news:    {sum(yf_news_result.values())}")
    logger.info(f"  Fixed RSS articles:    {sum(news_result.values())}")
    logger.info(f"  FinBERT scored:        {score_result.get('scored', 0)}")
    logger.info(f"  Correlations:          {corr_result.get('computed', 0)} tickers")
 
    df = query_df("""
        SELECT ticker, optimal_lag_days, optimal_lag_corr, sample_size
        FROM sentiment_price_correlation
        WHERE analysis_date = ?
        ORDER BY ABS(optimal_lag_corr) DESC
        LIMIT 3
    """, [run_date])
 
    if not df.empty:
        logger.info("\n  TOP SENTIMENT-PRICE FINDINGS:")
        for _, row in df.iterrows():
            direction = "positive" if row["optimal_lag_corr"] > 0 else "negative"
            logger.info(
                f"    {row['ticker']:<12} "
                f"lag={int(row['optimal_lag_days'])}d  "
                f"r={row['optimal_lag_corr']:+.4f} ({direction})  "
                f"n={int(row['sample_size'])}"
            )
    logger.info("=" * 60)
 
@flow(
    name="stock-sentiment-daily",
    description="Daily pipeline: collect -> score -> correlate",
)
def daily_pipeline(
    run_date: Optional[str]  = None,
    lookback_days: int  = 1,
):
   
    if run_date is None:
        run_date = (date.today() - timedelta(days=1)).isoformat()
 
    logger.info(f"[flow] Starting daily pipeline for {run_date}")
    
    from config.config import is_market_holiday
    if is_market_holiday(run_date):
        logger.info(
            f"[flow] {run_date} is known NSE/BSE market holiday - "
            f"skipping pipeline run. No price data exists to collect "
            f"today, so there is nothing for this run to do."
        )
 
    market_result       = collect_market_task(run_date, lookback_days)
    google_news_result  = collect_google_news_task(run_date)
    yf_news_result       = collect_yfinance_news_task(run_date)
    news_result          = collect_news_task(run_date)
 
    validate_collection_task(run_date)
 
    score_result = score_sentiment_task(run_date)
    agg_result   = aggregate_sentiment_task(run_date)
 
    validate_scoring_task(run_date)
 
    corr_result = correlation_task(run_date, lookback_days=90)
 
    summary_task(
        run_date,
        market_result,
        google_news_result,
        yf_news_result,
        news_result,
        score_result,
        corr_result,
    )
 
 
@flow(name="stock-sentiment-backfill")
def backfill_pipeline(
    start_date: str,
    end_date:   Optional[str] = None,
):
    if end_date is None:
        end_date = date.today().isoformat()
 
    current = date.fromisoformat(start_date)
    end     = date.fromisoformat(end_date)
    total   = (end - current).days + 1
 
    logger.info(
        f"[backfill] Processing {total} days: "
        f"{start_date} -> {end_date}"
    )
 
    processed = 0
    while current <= end:
        run_date_str = current.isoformat()
        logger.info(f"[backfill] Processing {run_date_str} ({processed+1}/{total})")
 
        try:
            daily_pipeline(run_date=run_date_str, lookback_days=1)
        except Exception as e:
            logger.warning(f"[backfill] {run_date_str} failed: {e} - continuing")
 
        current  += timedelta(days=1)
        processed += 1
 
    logger.info(f"[backfill] Complete - processed {processed} days")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Sentiment Pipeline")
    parser.add_argument("--date",      default=None,  help="Run date YYYY-MM-DD")
    parser.add_argument("--backfill",  action="store_true")
    parser.add_argument("--start",     default=None,  help="Backfill start date")
    parser.add_argument("--end",       default=None,  help="Backfill end date")
    parser.add_argument("--serve",     action="store_true",
                        help="Run on cron schedule (keeps running)")
    args = parser.parse_args()
 
    if not test_connection():
        logger.error("Database not ready - run: python config/database.py")
        sys.exit(1)
 
    if args.backfill:
        if not args.start:
            logger.error("--backfill requires --start date")
            sys.exit(1)
        backfill_pipeline(start_date=args.start, end_date=args.end)
 
    elif args.serve:
        from config.config import COLLECT_MARKET_CRON
        logger.info(f"[flow] Serving on schedule: {COLLECT_MARKET_CRON}")
        daily_pipeline.serve(
            name="stock-sentiment-scheduled",
            cron=COLLECT_MARKET_CRON,
        )
 
    else:
        daily_pipeline(run_date=args.date)