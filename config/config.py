import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

DUCKDB_PATH = BASE_DIR / os.getenv("DUCKDB_PATH", "data/sentiment.duckdb")
DUCKDB_FILENAME = "sentiment.duckdb"

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", None)

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LEN = 512
FINBERT_BATCH_SIZE = 16

TICKERS = [
    "RELIANCE", "HINDUNILVR", "ADANIGREEN", "NTPC", "JSWINFRA",
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "TMCV", "M&M", "MARUTI", "HYUNDAI", "BAJAJ-AUTO",
    "GOLDBEES", "SILVERBEES", "BANKBEES", "SUNPHARMA", "ONGC"
]

TICKER_NSE_MAP = {t: f"{t}.NS" for t in TICKERS}

TICKER_KEYWORDS: dict[str, list[str]] = {
    "RELIANCE": ["Reliance", "RIL", "Mukesh Ambani", "Jio"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"],
    "ADANIGREEN": ["Adani Green Energy", "Adani Green"],
    "NTPC": ["NTPC Limited", "NTPC"],
    "JSWINFRA": ["JSW Infrastructure", "JSW"],
    "TCS": ["Tata Consultancy Services", "TCS", "Tata"],
    "INFY": ["Infosys", "Narayana Murthy"],
    "HCLTECH": ["HCL Technologies", "HCL"],
    "WIPRO": ["Wipro Limited", "WIPRO"],
    "TECHM": ["Tech Mahindra Ltd", "TECHM"],
    "HDFCBANK": ["HDFC BANK", "HDFC"],
    "ICICIBANK": ["ICICI BANK", "ICICI"],
    "KOTAKBANK": ["KOTAK BANK", "KOTAK"],
    "AXISBANK": ["AXIS BANK", "AXIS"],
    "SBIN": ["SBI BANK OF INDIA", "SBI"],
    "TMCV": ["TATA MOTORS", "Tata"],
    "M&M": ["Mahindra & Mahindra"],
    "MARUTI": ["MARUTI SUZUKI"],
    "HYUNDAI": ["HYUNDAI MOTOR INDIA", "HYU"],
    "BAJAJ-AUTO": ["BAJAJ Auto", "Bajaj"],
    "GOLDBEES": ["Nippon India ETF Gold BeES", "Gold ETF"],
    "SILVERBEES": ["Nippon India Silver ETF", "Silver ETF"],
    "BANKBEES": ["Nippon India ETF Nifty Bank BeES", "Bank ETF"],
    "SUNPHARMA": ["Sun Pharmaceutical Industries", "Sun Pharma"],
    "ONGC": ["Oil and Natural Gas Corporation"],
}

for t in TICKERS:
    if t not in TICKER_KEYWORDS:
        TICKER_KEYWORDS[t] = [t]
        
NEWS_RSS_FEEDS = [
    
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    
]

NSE_HOLIDAYS_2026 = {
    "2026-09-14",
    "2026-10-02",
    "2026-10-20",
    "2026-11-10",
    "2026-11-24",
    "2026-12-25",
}

def is_market_holiday(date_str: str) -> bool:
    """
    Returns if there's a market on holiday on the given dates.
    Does not check weekends.
    """
    return date_str in NSE_HOLIDAYS_2026

DATA_DIR = BASE_DIR / "data"
RAW_MARKET_DIR = DATA_DIR / "raw" / "market"
RAW_SENTIMENT_DIR = DATA_DIR / "raw" / "sentiment"
RAW_NEWS_DIR = DATA_DIR / "raw" / "news"
PROCESSED_DIR = DATA_DIR / "processed"


for _dir in [RAW_MARKET_DIR, RAW_SENTIMENT_DIR, 
             RAW_NEWS_DIR, PROCESSED_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
    
COLLECT_MARKET_CRON = "30 18 * * 1-5"
COLLECT_SENTIMENT_CRON = "0 19 * * 1-5"
SCORE_SENTIMENT_CRON = "0 20 * * 1-5"
CORRELATE_CRON = "0 21 * * 1-5"

DASHBOARD_PORT = 8501
API_PORT = 8000
PREFECT_PORT = 4200