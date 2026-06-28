import sys
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import TICKERS, DUCKDB_PATH, DUCKDB_FILENAME, HF_DATASET_REPO, HF_TOKEN

@st.cache_resource(ttl=300)
def ensure_latest_database():
    if not HF_DATASET_REPO:
        return
    
    try:
        from huggingface_hub import hf_hub_download
        
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=DUCKDB_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN if HF_TOKEN else None,
            force_download=True
        )
        
        DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(downloaded_path, DUCKDB_PATH)
        
    except Exception as e:
        st.warning(
            f"Could not refresh data from Hugging Face Dataset repo "
            f"({e}). Showing last available data, if any."
        )
        
ensure_latest_database()

from config.database import query_df



st.set_page_config(
    page_title="Stock Sentiment Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(ttl=300)
def load_tickers() -> pd.DataFrame:
    return query_df("""
            SELECT ticker, company_name, sector, industry
            FROM tickers
            WHERE is_active = TRUE
            ORDER BY ticker
    """)
    
@st.cache_data(ttl=300)
def load_sentiment_price(ticker: str, days: int) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    return query_df("""
            SELECT
              ds.sentiment_date AS date,
              ds.avg_sentiment_score,
              ds.engagement_weighted_score,
              ds.total_mentions,
              ds.positive_ratio,
              ds.negative_ratio,
              pd.close_price,
              pd.pct_change
            FROM daily_sentiment ds
            JOIN price_daily pd
             ON ds.ticker = pd.ticker
             AND ds.sentiment_date = pd.trade_date
            WHERE ds.ticker = ?
             AND ds.sentiment_date >= ?
            ORDER BY ds.sentiment_date
    """, [ticker, cutoff])
    
@st.cache_data(ttl=300)
def load_correlation(ticker: str) -> pd.DataFrame:
    return query_df("""
            SELECT *
            FROM sentiment_price_correlation
            WHERE ticker = ?
            ORDER BY analysis_date DESC
            LIMIT 1
    """, [ticker])
    
@st.cache_data(ttl=300)
def load_top_correlations(limit: int = 10) -> pd.DataFrame:
    return query_df("""
            SELECT * FROM vw_latest_correlations LIMIT ?
    """, [limit])
    
@st.cache_data(ttl=300)
def load_sector_overview() -> pd.DataFrame:
    return query_df("""
            SELECT
               sector,
               AVG(avg_sector_sentiment)   AS avg_sentiment,
               AVG(avg_sector_return)      AS avg_return,
               SUM(total_sector_mentions)  AS total_mentions
            FROM vw_sector_sentiment
            WHERE sentiment_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY sector
            ORDER BY avg_sentiment DESC
    """)
    
@st.cache_data(ttl=300)
def get_data_freshness() -> dict:
    df = query_df("""
            SELECT
               MAX(trade_date) AS latest_price_date,
               (SELECT MAX(sentiment_date) FROM daily_sentiment) AS latest_sentiment_date,
               (SELECT MAX(analysis_date) FROM sentiment_price_correlation) AS latest_correlation_date
            FROM price_daily
    """)
    if df.empty:
        return {}
    return df.to_dict(orient="records")[0]

st.sidebar.title("📈 Stock Sentiment Engine")
st.sidebar.markdown(
    "Correlates Indian NSE stock sentiment from Google News, Yahoo "
    "Finance, and financial RSS feeds with price movements using "
    "FinBERT and lag-correlation analysis."
)

freshness = get_data_freshness()
if freshness:
    st.sidebar.markdown("----")
    st.sidebar.caption("**Data Freshness**")
    st.sidebar.caption(f"Latest price data: {freshness.get('latest_price_date', 'N/A')}")
    st.sidebar.caption(f"Latest sentiment: {freshness.get('latest_sentiment_date', 'N/A')}")
    latest_corr = freshness.get("latest_correlation_date")
    latest_corr_display = latest_corr if pd.notna(latest_corr) else "Not yet available"
    st.sidebar.caption(f"Latest correlation run: {latest_corr_display}")
    
st.sidebar.markdown("----")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Ticker Deep Dive", "Correlation Leaderboard", "Sector Analysis"],
)

st.sidebar.markdown("----")
st.sidebar.caption(
    "Built with Prefect, DuckDB, FinBERT, FastAPI, and Streamlit."
)

if page == "Overview":
    st.title("Stock Sentiment Engine")
    st.markdown(
        "An end-to-end pipeline that scores financial news sentiment "
        "with **FinBERT** and tests whether it predicts next-day NSE "
        "price movement, across 25 tracked stocks."
    )
    
    tickers_df = load_tickers()
    top_corr_df = load_top_correlations(limit=5)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tickers Tracked", len(tickers_df))
    with col2:
        if not top_corr_df.empty:
            best = top_corr_df.iloc[0]
            st.metric(
                "Strongest Signal",
                best["ticker"],
                f"r = {best['optimal_lag_corr']:+.3f} (lag {int(best['optimal_lag_days'])}d)"
            )
        else:
            st.metric("Strongest Signal", "Collecting data...")
    with col3:
        sectors_df = load_sector_overview()
        st.metric("Sectors Covered", len(sectors_df) if not sectors_df.empty else 0)
        
    st.markdown("----")
    st.subheader("Top Sentiment-Price Correlations")
    st.caption(
        "How many days does sentiment lead price movement, and how "
        "strong is that relationship (Pearson r)?"
    )
    
    if not top_corr_df.empty:
        fig = px.bar(
            top_corr_df,
            x="ticker",
            y="optimal_lag_corr",
            color="optimal_lag_corr",
            color_continuous_scale=["#d62728", "#cccccc", "#2ca02c"],
            color_continuous_midpoint=0,
            labels={"optimal_lag_corr": "Correlation (r)", "ticker": "Ticker"},
            hover_data=["company_name", "optimal_lag_days", "sample_size"],
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width="stretch")
        
        st.dataframe(
            top_corr_df[[
                "ticker", "company_name", "sector",
                "optimal_lag_days", "optimal_lag_corr", "sample_size"
            ]].rename(columns={
                "optimal_lag_days": "Optimal Lag (days)",
                "optimal_lag_corr": "Correlation (r)",
                "sample_size": "Sample Size",
            }),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No correlation data yet. This pipeline collect live news "
            "and price data once per day - correlation analysis needs "
            "at least 10 days of overlapping history to be statistically "
            "meaningful, since each ticker's news sources only return "
            "current articles, not historical archives. Check back as "
            "daily collection accumulates."
        )
        
elif page == "Ticker Deep Dive":
    st.title("Ticker Deep Dive")
    
    tickers_df = load_tickers()
    ticker_options = tickers_df["ticker"].tolist() if not tickers_df.empty else TICKERS
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_ticker = st.selectbox("Select a ticker", ticker_options)
    with col2:
        days_back = st.slider("Days of history", 7, 180, 30)
        
    data = load_sentiment_price(selected_ticker, days_back)
    
    if data.empty:
        st.warning(
            f"No overlapping sentiment and price data for "
            f"{selected_ticker} yet. Run the pipeline for more days."
        )
    else:
        company_row = tickers_df[tickers_df["ticker"] == selected_ticker]
        company_name = (
            company_row["company_name"].iloc[0]
            if not company_row.empty else selected_ticker
        )
        st.subheader(company_name)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["avg_sentiment_score"],
            name="Sentiment Score", line=dict(color="#1f77b4"),
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["close_price"],
            name="Close Price (₹)", line=dict(color="#ff7f0e"),
            yaxis="y2"
        ))
        fig.update_layout(
            title="Sentiment Score vs Closing Price",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Sentiment Score", side="left", range=[-1, 1]),
            yaxis2=dict(title="Close Price (₹)", side="right", overlaying="y"),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Avg Sentiment",
                f"{data['avg_sentiment_score'].mean():+.3f}"
            )
        with col2:
            st.metric(
                "Total Mentions",
                int(data["total_mentions"].sum())
            )
        with col3:
            st.metric(
                "Avg Daily Return",
                f"{data['pct_change'].mean():+.2f}%"
            )
        with col4:
            corr_df = load_correlation(selected_ticker)
            if not corr_df.empty:
                row = corr_df.iloc[0]
                st.metric(
                    "Optimal Lag",
                    f"{int(row['optimal_lag_days'])} day(s)",
                    f"r = {row['optimal_lag_corr']:+.3f}"
                )
            else:
                st.metric("Optimal Lag", "N/A")
        
        st.markdown("----")
        st.subheader("Mention Volume Over Time")
        fig2 = px.bar(
            data, x="date", y="total_mentions",
            labels={"total_mentions": "Mentions", "date": "Date"},
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")
        
        with st.expander("Raw data"):
            st.dataframe(data, width="stretch", hide_index=True)
            
elif page == "Correlation Leaderboard":
    st.title("Correlation Leaderboard")
    st.markdown(
        "Ranks every tracked stock by how strongly its sentiment "
        "predicts future price movement, and at what lag."
    )
    
    n = st.slider("Number of tickers to show", 5, 25, 15)
    top_df = load_top_correlations(limit=n)
    
    if top_df.empty:
        st.info(
            "No correlation data yet - accumulation daily history. "
            "Check back once 10+ days of overlapping sentiment and "
            "price data are available."
        )
    else:
        fig = px.scatter(
            top_df,
            x="optimal_lag_days",
            y="optimal_lag_corr",
            size="sample_size",
            color="sector",
            hover_name="ticker",
            hover_date=["company_name", "sample_size"],
            labels={
                "optimal_lag_days": "Optimal Lag (days)",
                "optimal_lag_corr": "Correlation Strength (r)",
            },
            title="Lag vs Correlation Strength by Sector",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        
        st.dataframe(
            top_df[[
                "ticker", "company_name", "sector", "optimal_lag_days",
                "optimal_lag_corr", "corr_lag_0d", "corr_lag_1d",
                "corr_lag_2d", "corr_lag_3d", "corr_lag_7d", "sample_size" 
            ]],
            width="stretch",
            hide_index=True,
        )
        
elif page == "Sector Analysis":
    st.title("Sector Analysis")
    st.markdown("Aggregated sentiment and average return by sector.")
    
    sector_df = load_sector_overview()
    
    if sector_df.empty:
        st.info("No sector data yet. Run the pipeline first.")
    else:
        fig = px.bar(
            sector_df,
            x="sector",
            y="avg_sentiment",
            color="avg_return",
            color_continuous_scale="Rdylgn",
            labels={
                "avg_sentiment": "Avg Sentiment Score",
                "sector": "Sector",
                "avg_return": "Avg Daily Return (%)",
            },
            title="Average Sentiment by Sector (colored by avg return)",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")
        
        st.dataframe(
            sector_df.rename(columns={
                "avg_sentiment": "Avg Sentiment",
                "avg_return": "Avg Daily Return (%)",
                "total_mentions": "Total Mentions",
            }),
            width="stretch",
            hide_index=True,
        )