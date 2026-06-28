import sys
from pathlib import Path 


root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from config.config import DUCKDB_PATH

import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger

def get_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DUCKDB_PATH))
    return conn

def init_database():
    logger.info(f"[DB] Initalising database at {DUCKDB_PATH}")
    conn = get_connection()
    
    conn.execute("""
                CREATE TABLE IF NOT EXISTS tickers(
                     ticker          VARCHAR PRIMARY KEY,
                     company_name    VARCHAR NOT NULL,
                     sector          VARCHAR,
                     industry        VARCHAR,
                     exchange        VARCHAR DEFAULT 'NSE',
                     is_active       BOOLEAN DEFAULT TRUE,
                     added_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
    """)
    
    conn.execute("""
                CREATE TABLE IF NOT EXISTS price_daily(
                     ticker          VARCHAR   NOT NULL,
                     trade_date      DATE      NOT NULL,
                     open_price      DOUBLE,
                     high_price      DOUBLE,
                     low_price       DOUBLE,
                     close_price     DOUBLE,
                     adj_close       DOUBLE,
                     volume          BIGINT,
                     pct_change      DOUBLE,
                     intraday_range  DOUBLE,
                     ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     PRIMARY KEY (ticker, trade_date)
                )
    """)
    conn.execute("""
                 CREATE SEQUENCE IF NOT EXISTS seq_raw_sentiment_id START 1
    """)
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS raw_sentiment (
                     id               INTEGER  PRIMARY KEY DEFAULT nextval('seq_raw_sentiment_id'),
                     source           VARCHAR  NOT NULL,
                     source_id        VARCHAR,
                     ticker           VARCHAR  NOT NULL,
                     published_at     TIMESTAMP,
                     collected_date   VARCHAR  NOT NULL,
                     raw_text         VARCHAR  NOT NULL,
                     author           VARCHAR,
                     upvotes          INTEGER  DEFAULT 0,
                     url              VARCHAR,
                     ingested_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     UNIQUE (source, source_id, ticker)
                )
    """)
    
    conn.execute("""
                 CREATE SEQUENCE IF NOT EXISTS seq_sentiment_scored_id START 1
    """)
    
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS sentiment_scored (
                     id               INTEGER  PRIMARY KEY DEFAULT nextval('seq_sentiment_scored_id'),
                     raw_id           INTEGER  REFERENCES raw_sentiment(id),
                     ticker           VARCHAR  NOT NULL,
                     scored_date      DATE     NOT NULL,
                     label            VARCHAR  NOT NULL,
                     score_positive   DOUBLE   NOT NULL,
                     score_negative   DOUBLE   NOT NULL,
                     score_neutral    DOUBLE   NOT NULL,
                     confidence       DOUBLE   NOT NULL,
                     compound_score   DOUBLE   NOT NULL,
                     scored_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
    """)
    
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS daily_sentiment (
                     ticker                      VARCHAR  NOT NULL,
                     sentiment_date              DATE     NOT NULL,
                     total_mentions              INTEGER  DEFAULT 0,
                     news_mentions               INTEGER  DEFAULT 0,
                     google_news_mentions        INTEGER  DEFAULT 0,
                     yahoo_finance_mentions      INTEGER  DEFAULT 0,
                     avg_sentiment_score         DOUBLE,
                     engagement_weighted_score   DOUBLE,
                     positive_ratio              DOUBLE,
                     negative_ratio              DOUBLE,
                     neutral_ratio               DOUBLE,
                     sentiment_std               DOUBLE,
                     computed_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     PRIMARY KEY (ticker, sentiment_date)
                 )
    """)
    
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS sentiment_price_correlation (
                     ticker                     VARCHAR  NOT NULL,
                     analysis_date              DATE     NOT NULL,
                     corr_lag_0d                DOUBLE,
                     corr_lag_1d                DOUBLE,
                     corr_lag_2d                DOUBLE,
                     corr_lag_3d                DOUBLE,
                     corr_lag_7d                DOUBLE,
                     optimal_lag_days           INTEGER,
                     optimal_lag_corr           DOUBLE,
                     rolling_30d_avg_sent       DOUBLE,
                     rolling_30d_price_ret      DOUBLE,
                     sample_size                INTEGER,
                     computed_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     PRIMARY KEY (ticker, analysis_date)
                 )
     """)
    
    conn.execute("""
                 CREATE OR REPLACE VIEW vw_sentiment_price AS
                 SELECT
                     ds.ticker,
                     ds.sentiment_date               AS date,
                     ds.avg_sentiment_score,
                     ds.engagement_weighted_score,
                     ds.total_mentions,
                     ds.positive_ratio,
                     ds.negative_ratio,
                     ds.sentiment_std,
                     pd.close_price,
                     pd.pct_change                   AS price_change_pct,
                     pd.volume,
                     pd.intraday_range,
                     
                     LEAD(pd.pct_change, 1) OVER (
                            PARTITION BY ds.ticker ORDER BY ds.sentiment_date
                            )                        AS next_day_return,
                     LEAD(pd.pct_change, 2) OVER (
                            PARTITION BY ds.ticker ORDER BY ds.sentiment_date
                            )                        AS next_2day_return
                 FROM daily_sentiment ds
                 JOIN price_daily pd
                     ON ds.ticker = pd.ticker
                     AND ds.sentiment_date = pd.trade_date
     """)
    
    conn.execute("""
                 CREATE OR REPLACE VIEW vw_latest_correlations AS 
                 SELECT
                     spc.*,
                     t.company_name,
                     t.sector
                 FROM sentiment_price_correlation spc
                 JOIN tickers t ON spc.ticker = t.ticker
                 WHERE spc.analysis_date = (
                         SELECT MAX(analysis_date)
                         FROM sentiment_price_correlation
                )
                ORDER BY ABS(spc.optimal_lag_corr) DESC
     """)
    
    conn.execute("""
                 CREATE OR REPLACE VIEW vw_sector_sentiment AS
                 SELECT
                     t.sector,
                     ds.sentiment_date,
                     COUNT(DISTINCT ds.ticker)         AS stocks_tracked,
                     AVG(ds.avg_sentiment_score)       AS avg_sector_sentiment,
                     SUM(ds.total_mentions)            AS total_sector_mentions,
                     AVG(pd.pct_change)                AS avg_sector_return
                 FROM daily_sentiment ds
                 JOIN tickers t ON ds.ticker = t.ticker
                 JOIN price_daily pd
                     ON ds.ticker = pd.ticker
                     AND ds.sentiment_date = pd.trade_date
                 GROUP BY t.sector, ds.sentiment_date
     """)
    
    conn.execute("""
                 INSERT OR IGNORE INTO tickers
                     (ticker, company_name, sector, industry)
                 VALUES 
                     ('RELIANCE',   'Reliance Industries',        'Energy',         'Oil & Gas'),
                     ('HINDUNILVR',  'Hindustan Unilever',         'FMCG',           'Consumer Goods'),
                     ('ADANIGREEN', 'Adani Green Energy',         'Utilities',      'Power Generation'),
                     ('NTPC',       'NTPC Limited',               'Utilties',       'Power Generation'),
                     ('JSWINFRA',   'JSW Infrastructure',         'Infrastructure', 'Port'),
                     ('TCS',        'Tata Consultancy Services',  'IT',             'Software'),
                     ('INFY',       'Infosys',                    'IT',             'Software'),
                     ('HCLTECH',    'HCL Technologies',           'IT',             'Software'),
                     ('WIPRO',      'Wipro',                      'IT',             'Software'),
                     ('TECHM',      'Tech Mahindra Ltd',          'IT',             'software'),
                     ('HDFCBANK',   'HDFC Bank',                  'Finance',        'Banking'),
                     ('ICICIBANK',  'ICICI Bank',                 'Finance',        'Banking'),
                     ('KOTAKBANK',  'Kotak Mahindra Bank',        'Finance',        'Banking'),
                     ('AXISBANK',   'Axis Bank',                  'Finance',        'Banking'),
                     ('SBIN',       'State Bank of India',        'Finance',        'Banking'),
                     ('TMCV',       'Tata Motors',                'Auto',           'Commercial Vehicles'),
                     ('M&M',        'Mahindra and Mahindra',      'Auto',           'Passenger Vehicles'),
                     ('MARUTI',     'Maruti Suzuki',              'Auto',           'Passenger Vehicles'),
                     ('HYUNDAI',    'Hyundai',                    'Auto',           'Passenger Vehicles'),
                     ('BAJAJ-AUTO', 'Bajaj auto',                 'Auto',           'Commercial Vehicles'),
                     ('GOLDBEES',   'Nippon India ETF Gold BeES', 'Metals',         'Commodities'),
                     ('SILVERBEES', 'Nippon India Silver ETF',    'Metals',         'Commodities'),
                     ('BANKBEES',   'Nippon India ETF Bank BeES', 'Finance',        'Banking'),
                     ('SUNPHARMA',  'Sun Pharmaceutical',         'Healthcare',     'Pharma'),
                     ('ONGC',        'ONGC',                      'Energy',         'Oil & Gas' )
     """)
    
    conn.commit()
    conn.close()
    logger.info("[DB] Databse initialised successfully - all tables and views are created")
    
def query_df(sql: str, params: list = None) -> pd.DataFrame:
    conn = get_connection()
    try:
        if params:
            result = conn.execute(sql, params).df()
        else:
            result = conn.execute(sql).df()
        return result
    except Exception as e:
        logger.error(f"[DB] query_df failed: {e}\nsql: {sql[:200]}")
        raise
    finally:
        conn.close()


def execute_many(sql: str, records: list[list]) -> int:
    if not records:
        return 0
    conn = get_connection()
    try:
        conn.executemany(sql, records)
        conn.commit()
        logger.info(f"[DB] Inserted {len(records)} rows")
        return len(records)
    except Exception as e:
        logger.error(f"[DB] execute_many failed: {e}")
        raise
    finally:
        conn.close()
        
def insert_df(
    table: str,
    df: pd.DataFrame,
    mode: str = "ignore",
    replace_keys: list[str] = None,
) -> int:
    
    if df.empty:
        logger.warning(f"[DB] insert_df called with empty DataFrame for {table}")
        return 0
 
    if mode == "replace" and not replace_keys:
        raise ValueError(
            "insert_df(mode='replace') requires replace_keys, e.g. "
            "replace_keys=['ticker', 'analysis_date'], so it knows "
            "which existing rows to delete before re-inserting."
        )
 
    conn = get_connection()
    try:
        conn.register("incoming_df", df)
 
        columns = list(df.columns)
        column_list = ", ".join(columns)
 
        if mode == "ignore":
            conn.execute(
                f"INSERT OR IGNORE INTO {table} ({column_list}) "
                f"SELECT {column_list} FROM incoming_df"
            )
 
        elif mode == "replace":
          
            key_values_df = df[replace_keys].drop_duplicates()
            conn.register("keys_to_replace", key_values_df)
 
            join_condition = " AND ".join(
                f"{table}.{k} = keys_to_replace.{k}" for k in replace_keys
            )
            conn.execute(f"""
                DELETE FROM {table}
                WHERE EXISTS (
                    SELECT 1 FROM keys_to_replace
                    WHERE {join_condition}
                )
            """)
            conn.unregister("keys_to_replace")
 
            conn.execute(
                f"INSERT INTO {table} ({column_list}) "
                f"SELECT {column_list} FROM incoming_df"
            )
 
        else:
            conn.execute(
                f"INSERT INTO {table} ({column_list}) "
                f"SELECT {column_list} FROM incoming_df"
            )
 
        conn.unregister("incoming_df")
        conn.commit()
        logger.info(f"[DB] Inserted {len(df)} rows into {table}")
        return len(df)
    except Exception as e:
        logger.error(f"[DB] insert_df failed for {table}: {e}")
        raise
    finally:
        conn.close()
 
def run_sql(sql: str) -> None:
    conn = get_connection()
    try:
        conn.execute(sql)
        conn.commit()
    except Exception as e:
        logger.error(f"[DB] run_sql failed: {e}")
        raise
    finally:
        conn.close()
        
def test_connection() -> bool:
    try:
        result = query_df("SELECT COUNT(*) AS ticker_count FROM tickers")
        count = result["ticker_count"].iloc[0]
        logger.info(f"[DB] Connection is OK - {count} tickers in database")
        return True
    except Exception as e:
        logger.error(f"[DB] Connection Failed: {e}")
        return False

if __name__ == "__main__":
    init_database()
    test_connection()   
    
    