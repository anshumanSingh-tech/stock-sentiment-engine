import sys
from pathlib import Path
from datetime import date
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import query_df, insert_df


def compute_correlations(
    run_date: str = None,
    lookback_days: int = 90
) -> pd.DataFrame:
    """Computes Pearson correlation between sentiment and price across lag windows."""
    if run_date is None:
        run_date = date.today().isoformat()

    lookback_start = (
        pd.Timestamp(run_date) - pd.Timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    logger.info(
        f"[corr] Computing correlations | "
        f"{lookback_start} -> {run_date} | "
        f"lookback={lookback_days}d"
    )

    df_joined = query_df("""
        WITH base AS (
            SELECT
                ds.ticker,
                ds.sentiment_date                               AS date,
                ds.avg_sentiment_score                         AS sentiment,
                ds.engagement_weighted_score,
                ds.total_mentions,
                pd.pct_change                                  AS price_change,
                pd.close_price,
                pd.volume,
                LAG(ds.avg_sentiment_score, 1) OVER (
                    PARTITION BY ds.ticker ORDER BY ds.sentiment_date
                )                                              AS sentiment_lag1,
                LAG(ds.avg_sentiment_score, 2) OVER (
                    PARTITION BY ds.ticker ORDER BY ds.sentiment_date
                )                                              AS sentiment_lag2,
                LAG(ds.avg_sentiment_score, 3) OVER (
                    PARTITION BY ds.ticker ORDER BY ds.sentiment_date
                )                                              AS sentiment_lag3,
                LAG(ds.avg_sentiment_score, 7) OVER (
                    PARTITION BY ds.ticker ORDER BY ds.sentiment_date
                )                                              AS sentiment_lag7,
                AVG(ds.avg_sentiment_score) OVER (
                    PARTITION BY ds.ticker
                    ORDER BY ds.sentiment_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                )                                              AS rolling_30d_sentiment,
                AVG(pd.pct_change) OVER (
                    PARTITION BY ds.ticker
                    ORDER BY ds.sentiment_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                )                                              AS rolling_30d_price
            FROM daily_sentiment ds
            JOIN price_daily pd
              ON ds.ticker = pd.ticker
             AND ds.sentiment_date = pd.trade_date
            WHERE ds.sentiment_date BETWEEN ? AND ?
        )

        SELECT
            ticker,
            ? AS analysis_date,
            ROUND(corr(sentiment,      price_change), 6)  AS corr_lag_0d,
            ROUND(corr(sentiment_lag1, price_change), 6)  AS corr_lag_1d,
            ROUND(corr(sentiment_lag2, price_change), 6)  AS corr_lag_2d,
            ROUND(corr(sentiment_lag3, price_change), 6)  AS corr_lag_3d,
            ROUND(corr(sentiment_lag7, price_change), 6)  AS corr_lag_7d,
            ROUND(AVG(rolling_30d_sentiment), 4)          AS rolling_30d_avg_sent,
            ROUND(AVG(rolling_30d_price),     4)          AS rolling_30d_price_ret,
            COUNT(*)                                      AS sample_size
        FROM base
        WHERE sentiment IS NOT NULL
          AND price_change IS NOT NULL
        GROUP BY ticker
        HAVING COUNT(*) >= 10
    """, [lookback_start, run_date, run_date])

    if df_joined.empty:
        logger.warning("[corr] No data for correlation. Collect more data first.")
        return pd.DataFrame()

    logger.info(f"[corr] Raw correlations computed for {len(df_joined)} tickers")
    return df_joined


def find_optimal_lag(df: pd.DataFrame) -> pd.DataFrame:
    """For each ticker, finds which lag window has the strongest absolute correlation."""
    lag_cols = {
        0: "corr_lag_0d",
        1: "corr_lag_1d",
        2: "corr_lag_2d",
        3: "corr_lag_3d",
        7: "corr_lag_7d",
    }

    optimal_lags  = []
    optimal_corrs = []

    for _, row in df.iterrows():
        best_lag  = 0
        best_corr = 0.0

        for lag, col in lag_cols.items():
            val = row.get(col)
            if val is not None and not pd.isna(val):
                if abs(float(val)) > abs(best_corr):
                    best_corr = float(val)
                    best_lag  = lag

        optimal_lags.append(best_lag)
        optimal_corrs.append(round(best_corr, 6))

    df = df.copy()
    df["optimal_lag_days"] = optimal_lags
    df["optimal_lag_corr"] = optimal_corrs
    return df


def run_correlation_analysis(
    run_date: str = None,
    lookback_days: int = 90
) -> dict:
    """Main function - called by Prefect @task."""
    if run_date is None:
        run_date = date.today().isoformat()

    df = compute_correlations(run_date, lookback_days)
    if df.empty:
        return {"computed": 0}

    df = find_optimal_lag(df)
    insert_df("sentiment_price_correlation", df, mode="replace", replace_keys=["ticker", "analysis_date"])

    logger.info("=" * 60)
    logger.info("TOP SENTIMENT-PRICE CORRELATIONS")
    logger.info("=" * 60)

    for _, row in df.nlargest(5, "sample_size").iterrows():
        lag  = int(row["optimal_lag_days"])
        corr = float(row["optimal_lag_corr"])
        n    = int(row["sample_size"])
        logger.info(
            f"  {row['ticker']:<12} "
            f"optimal_lag={lag}d  "
            f"corr={corr:+.4f}  "
            f"n={n}"
        )

    logger.info("=" * 60)
    return {"computed": len(df), "date": run_date}


if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_correlation_analysis(run_date, lookback_days=90)
    print(f"Computed correlations for {result['computed']} tickers")