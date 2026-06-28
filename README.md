---
title: Stock Sentiment Engine
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Stock Sentiment Engine

An end-to-end data pipeline that scores Indian NSE stock sentiment from
news sources using **FinBERT**, then tests whether that sentiment
predicts next-day price movement using Pearson lag-correlation analysis.

**Live dashboard:** you're looking at it. Pick a ticker in the sidebar.

## What this project demonstrates

- **Data engineering**: a daily Prefect pipeline collects price data
  (yfinance), and news sentiment from three independent sources
  (Google News RSS, Yahoo Finance, fixed financial RSS feeds) - all
  free, all requiring zero API approval.
- **NLP**: FinBERT (a BERT model fine-tuned on financial text) scores
  every collected article as positive / negative / neutral, far more
  accurately than general-purpose sentiment tools on financial language.
- **Analytics engineering**: DuckDB window functions compute lag
  correlations (0/1/2/3/7-day) between sentiment and price returns -
  no Spark cluster needed for this data volume, just well-written SQL.
- **Data quality**: a custom validation layer gates the pipeline between
  collection and scoring, and again before correlation - bad data never
  silently reaches the analysis layer.
- **Full-stack delivery**: a FastAPI REST API and this Streamlit
  dashboard both read from the same DuckDB file.

## Architecture

```
Sources (yfinance, Google News, Yahoo Finance, RSS feeds)
        |
        v
   Prefect pipeline (collection tasks)
        |
        v
   Data quality gate
        |
        v
   FinBERT scoring (per article)
        |
        v
   Daily aggregation (per ticker, engagement-weighted)
        |
        v
   Data quality gate
        |
        v
   DuckDB lag-correlation engine
        |
        v
   FastAPI  +  Streamlit dashboard (this Space)
```

## Why no Reddit data

Reddit now requires pre-approval for all apps, including personal
portfolio projects, with high rejection rates reported for solo
developers. Rather than wait on an uncertain approval process, this
project uses three independent news-based sentiment sources instead,
none of which require authentication or registration.

## Tech stack

Prefect • DuckDB • FinBERT (HuggingFace Transformers) • FastAPI •
Streamlit • Plotly • pandas - zero paid cloud services, zero Docker
required for local development.

## Running the full pipeline locally

This deployed Space only shows pre-computed results. To run the
actual data collection and scoring pipeline:

```bash
git clone https://github.com/anshumanSingh-tech/stock-sentiment-engine.git
cd stock-sentiment-engine
pip install -r requirements.txt
cp .env.example .env
python config/database.py
python -m flows.main_flow
```

Note: this pipeline collects *live* news once per day - each ticker's
news sources only return current articles, not historical archives,
so there is no `--backfill` option for generating historical sentiment
data. Correlation analysis becomes statistically meaningful once 10+
days of real daily collection have accumulated. See `scripts/setup_task_scheduler.md`
for automating daily collection via Windows Task Scheduler.

See the full repository on GitHub:
https://github.com/anshumanSingh-tech/stock-sentiment-engine