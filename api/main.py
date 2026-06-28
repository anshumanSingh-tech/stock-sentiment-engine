import sys
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import query_df
from config.config import TICKERS

app = FastAPI(
    title="Stock Sentiment Engine API",
    description=(
        "Correlates social/news sentiment with NSE stock price movements. "
        "Read-only API serving pre-computed FinBERT sentiment scores and "
        "Pearson lag-correlation results."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

class TickerInfo(BaseModel):
    ticker: str
    company_name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    
class SentimentSnapshot(BaseModel):
    ticker: str
    date: str
    avg_sentiment_score: Optional[float] = None
    engagement_weighted_score: Optional[float] = None
    total_mentions: int
    positive_ratio: Optional[float] = None
    negative_ratio: Optional[float] = None
    
class PricePoint(BaseModel):
    ticker: str
    date: str
    close_price: float
    pct_change: Optional[float] = None
    volume: Optional[int] = None
    
class CorrelationResult(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    analysis_date: str
    optimal_lag_days: Optional[int] = None
    optimal_lag_corr: Optional[float] = None
    corr_lag_0d: Optional[float] = None
    corr_lag_1d: Optional[float] = None
    corr_lag_2d: Optional[float] = None
    corr_lag_3d: Optional[float] = None
    corr_lag_7d: Optional[float] = None
    sample_size: Optional[int] = None
    
def _validate_ticker(ticker: str) -> str:
    ticker_upper = ticker.upper()
    if ticker_upper not in TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not tracked. "
            f"See GET /tickers for the full list."
        )
    return ticker_upper

@app.get("/", tags=["Health"])
def health_check():
    try:
        df = query_df("SELECT COUNT(*) AS cnt FROM tickers")
        ticker_count = int(df["cnt"].iloc[0])
        return{
            "status": "ok",
            "service": "Stock Sentiment Engine API",
            "tickers_tracked": ticker_count,
        }
    except Exception as e:
        logger.error(f"[api] Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    
@app.get("/tickers", response_model=list[TickerInfo], tags=["Reference"])
def list_tickers():
    df = query_df("""
                  SELECT ticker, company_name, sector, industry
                  FROM tickers
                  WHERE is_active = TRUE
                  ORDER BY ticker
    """)
    return df.to_dict(orient="records")

@app.get(
    "/sentiment/{ticker}",
    response_model=list[SentimentSnapshot],
    tags=["Sentiment"]
)
def get_sentiment_history(
    ticker: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
):
    ticker = _validate_ticker(ticker)
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    
    df = query_df("""
                SELECT
                     ticker,
                     sentiment_date AS date,
                     avg_sentiment_score,
                     engagement_weighted_score,
                     total_mentions,
                     positive_ratio,
                     negative_ratio
                FROM daily_sentiment
                WHERE ticker = ?
                    AND sentiment_date >= ?
                ORDER BY sentiment_date DESC
    """, [ticker, cutoff])
    
    if df.empty:
        return []
    
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")

@app.get(
    "/sentiment/{ticker}/latest",
    response_model=SentimentSnapshot,
    tags=["Sentiment"]
)
def get_latest_sentiment(ticker: str):
    ticker = _validate_ticker(ticker)
    
    df = query_df("""
        SELECT
          ticker,
          sentiment_date AS date,
          avg_sentiment_score,
          engagement_weighted_score,
          total_mentions,
          positive_ratio,
          negative_ratio
        FROM daily_sentiment
        WHERE ticker = ?
        ORDER BY sentiment_date DESC
        LIMIT 1
    """, [ticker])
    
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for {ticker} yet"
        )
        
    df["date"] = df["date"].astype(str)
    return df._to_dict(orient="records")[0]

@app.get(
    "/price/{ticker}",
    response_model=list[PricePoint],
    tags=["Price"],
)
def get_price_history(
    ticker: str,
    days: int = Query(30, ge=1, le=365),
):
    ticker = _validate_ticker(ticker)
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    
    df = query_df("""
            SELECT
               ticker,
               trade_date AS date,
               close_price,
               pct_change,
               volume,
            FROM price_daily
            WHERE ticker = ?
              AND trade_date >= ?
            ORDER BY trade_date DESC
    """, [ticker, cutoff])
    
    if df.empty:
        return []
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")

@app.get(
    "/correlation/{ticker}",
    response_model=CorrelationResult,
    tags=["Correlation"],
)
def get_correlation(ticker: str):
    ticker = _validate_ticker(ticker)
    
    df= query_df("""
            SELECT
              spc.ticker,
              t.company_name,
              spc.analysis_date,
              spc.optimal_lag_days,
              spc.optimal_lag_corr,
              spc.corr_lag_0d,
              spc.corr_lag_1d,
              spc.corr_lag_2d,
              spc.corr_lag_3d,
              spc.corr_lag_7d,
              spc.sample_size
            FROM sentiment_price_correlation spc
            JOIN tickers t ON spc.ticker = t.ticker
            WHERE spc.ticker = ?
            ORDER BY spc.analysis_date DESC
            LIMIT 1
    """, [ticker])
    
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No correlation data found for {ticker} yet. "
                   f"Needs at least 10 days of overlapping sentiment "
                   f"and price data."
        )
    df["analysis_date"] = df["analysis_date"].astype(str)
    return df.to_dict(orient="records")[0]

@app.get(
    "/correlation/top",
    response_model=list[CorrelationResult],
    tags=["Correlation"],
) 
def get_top_correlation(
    limit: int = Query(10, ge=1, le=25),
):
    df = query_df("""
            SELECT * FROM vw_latest_correlations
            LIMIT ?
    """, [limit])
    
    if df.empty:
        return []
    
    df["analysis_date"] = df["analysis_date"].astype(str)
    return df.to_dict(orient="records")

@app.get("/sector/{sector}", tags=["Sector"])
def get_sector_sentiment(
    sector: str,
    days: int = Query(30, ge=1, le=365),
):
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    
    df = query_df("""
                SELECT *
                FROM vw_sector_sentiment
                WHERE sector = ?
                  AND sentiment_date >= ?
                ORDER BY sentiment_date DESC
    """, [sector, cutoff])
    
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for sector '{sector}'."
        )
        
    df["sentiment_date"] = df["sentiment_date"].astype(str)
    return df.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)