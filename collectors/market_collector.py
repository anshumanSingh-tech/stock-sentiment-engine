import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import TICKERS, TICKER_NSE_MAP, RAW_MARKET_DIR
from config.database import insert_df, query_df

def fetch_ticker(ticker_nse: str, start: str, end: str) -> pd.DataFrame:
    
    try:
        df = yf.download(
            ticker_nse,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        
        if df.empty:
            logger.warning(f"[market] No data for {ticker_nse} ({start} -> {end})")
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    
    except Exception as e:
        logger.error(f"[market] yfinance failed for {ticker_nse}: {e}")
        return pd.DataFrame()
    
def process_ticker_df(df: pd.DataFrame, ticker:str) -> pd.DataFrame:
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.sort_values("date").reset_index(drop=True)
    
    df["pct_change"] = df["close"].pct_change() * 100
    
    df["intraday_range"] = ((df["high"] - df["low"]) / df["low"]) * 100
    
    out = pd.DataFrame({
        "ticker": ticker,
        "trade_date": pd.to_datetime(df["date"]).dt.date,
        "open_price": df["open"].round(4),
        "high_price": df["high"].round(4),
        "low_price": df["low"].round(4),
        "close_price": df["close"].round(4),
        "adj_close": df.get("adj_close", df["close"]).round(4),
        "volume": df["volume"].fillna(0).astype(int),
        "pct_change": df["pct_change"].round(4),
        "intraday_range": df["intraday_range"].round(4),
    }) 
    
    out = out.dropna(subset=["pct_change"])
    return out

def collect_market_data(
    run_date: str = None,
    lookback_days: int = 1
) -> dict[str, int]:
    
    if run_date is None:
        run_date = (date.today() - timedelta(days=1)).isoformat()
        
    start = (
        date.fromisoformat(run_date) - timedelta(days=lookback_days + 1)
    ).isoformat()
    end = date.today().isoformat()
    
    logger.info(
        f"[market] Collectig {len(TICKERS)} ticker | "
        f"{start} -> {end} | lookback={lookback_days}d"
    )
    
    results: dict[str, int] = {}
    all_frames: list[pd.DataFrame] = []
    
    for ticker in TICKERS:
        ticker_nse = TICKER_NSE_MAP[ticker]
        df_raw = fetch_ticker(ticker_nse, start, end)
        
        if df_raw.empty:
            results[ticker] = 0
            continue
        
        raw_path = RAW_MARKET_DIR / f"{ticker}_{run_date}.csv"
        df_raw.to_csv(raw_path, index=False)
        
        df_processed = process_ticker_df(df_raw, ticker)
        if not df_processed.empty:
            all_frames.append(df_processed)
            results[ticker] = len(df_processed)
            logger.info(f"[market] {ticker}: {len(df_processed)} rows")
        else:
            results[ticker] = 0
    
    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        inserted = insert_df("price_daily", df_all, mode="ignore")
        logger.info(
            f"[market] Done - {inserted} rows inserted across "
            f"{len(TICKERS)} tickers"
            )
    else:
        logger.warning("[market] No data collected for any ticker")
        
    return results

if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = collect_market_data(run_date, lookback_days=7)
    for ticker, count in result.items():
        print(f"{ticker}: {count} rows")
        