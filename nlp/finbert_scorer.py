import sys
from pathlib import Path
from datetime import date
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import FINBERT_MODEL, FINBERT_MAX_LEN, FINBERT_BATCH_SIZE
from config.database import query_df, insert_df

_tokenizer = None
_model = None
_device = None

def load_finbert():
    global _tokenizer, _model, _device
    
    if _tokenizer is not None:
        return
    
    logger.info(f"[finbert] Loading {FINBERT_MODEL}")
    logger.info("[finbert] First run downloads _450 MB (cached after that)")
    
    _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)
    _model.eval()
    
    logger.info(f"[finbert] Model ready on {_device}")
    
def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s%.,!?₹$\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:800]

def score_texts(texts: list[str]) -> list[dict]:
    load_finbert()
    
    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=FINBERT_MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = _model(**inputs).logits
        
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    
    results = []
    for prob in probs:
        pos, neg, nev = float(prob[0]), float(prob[1]), float(prob[2])
        label_idx = int(prob.argmax())
        label = ["positive", "negative", "neutral"][label_idx]
        results.append({
            "label": label,
            "score_positive": round(pos, 4),
            "score_negative": round(neg, 4),
            "score_neutral": round(nev, 4),
            "confidence": round(float(prob[label_idx]), 4),
            "compound_score": round(pos - neg, 4),
        })
    return results

def score_raw_sentiment(run_date: str = None) -> dict:
    if run_date is None:
        run_date = date.today().isoformat()
        
    logger.info(f"[finbert] Scoring raw sentiment for {run_date}")
    
    df = query_df("""
        SELECT
           rs.id AS raw_id,
           rs.ticker,
           rs.raw_text,
           rs.source,
           rs.upvotes
        FROM raw_sentiment rs
        LEFT JOIN sentiment_scored ss ON rs.id = ss.raw_id
        WHERE rs.collected_date = ?
            AND ss.raw_id IS NULL
            AND rs.raw_text IS NOT NULL
            AND LENGTH(rs.raw_text) > 15
        ORDER BY rs.id
    """, [run_date])
   
    if df.empty:
       logger.info(f"[finbert] No unscored records found for {run_date}")
       return {"scored": 0}
   
    logger.info(f"[finbert] Scoring {len(df)} records in batches of {FINBERT_BATCH_SIZE}") 
    
    scored_rows = []
    n_batches = (len(df) + FINBERT_BATCH_SIZE - 1) // FINBERT_BATCH_SIZE
    
    for i in range(0, len(df), FINBERT_BATCH_SIZE):
        batch_df = df.iloc[i : i + FINBERT_BATCH_SIZE]
        texts = [clean_text(t) for t in batch_df["raw_text"]]
        
        valid = [(idx, row, txt)
                 for (idx, row), txt in zip(batch_df.iterrows(), texts)
                 if txt]
        if not valid:
            continue
        
        _, rows, clean = zip(*valid)
        
        try:
            scores = score_texts(list(clean))
            for row, score in zip(rows, scores):
                scored_rows.append({
                    "raw_id": int(row["raw_id"]),
                    "ticker": row["ticker"],
                    "scored_date": run_date,
                    "label": score["label"],
                    "score_positive": score["score_positive"],
                    "score_negative": score["score_negative"],
                    "score_neutral": score["score_neutral"],
                    "confidence": score["confidence"],
                    "compound_score": score["compound_score"],
                })
                
            batch_num = i // FINBERT_BATCH_SIZE + 1
            logger.info(f"[finbert] Batch {batch_num}/{n_batches} complete")
        
        except Exception as e:
            logger.error(f"[finbert] Batch {i} failed: {e}")
            continue
        
    if scored_rows:
        df_scored = pd.DataFrame(scored_rows)
        insert_df("sentiment_scored", df_scored, mode="ignore")
        logger.info(f"[finbert] Scored and saved {len(scored_rows)} recors")
        
    return {"scored": len(scored_rows), "date": run_date}

def aggregate_daily_sentiment(run_date: str = None) -> dict:
    if run_date is None:
        run_date = date.today().isoformat()
        
    logger.info(f"[finbert] Aggregating daily sentiment for {run_date}")
    
    df = query_df("""
        SELECT
            ss.ticker,
            ? AS sentiment_date,
            COUNT(*)                                       AS total_mentions,
            SUM(CASE WHEN rs.source = 'news'          THEN 1 ELSE 0 END) AS news_mentions,
            SUM(CASE WHEN rs.source = 'google_news'   THEN 1 ELSE 0 END) AS google_news_mentions,
            SUM(CASE WHEN rs.source = 'yahoo_finance' THEN 1 ELSE 0 END) AS yahoo_finance_mentions,
            ROUND(AVG(ss.compound_score), 4)               AS avg_sentiment_score,
            ROUND(
                SUM(ss.compound_score * LN(rs.upvotes + 1.0))
                / NULLIF(SUM(LN(rs.upvotes + 1.0)), 0)
            , 4)                                           AS engagement_weighted_score,
            ROUND(AVG(ss.score_positive), 4)               AS positive_ratio,
            ROUND(AVG(ss.score_negative), 4)                     AS negative_ratio,
            ROUND(AVG(ss.score_neutral), 4)                AS neutral_ratio,
            ROUND(STDDEV(ss.compound_score), 4)            AS sentiment_std
        FROM sentiment_scored ss
        JOIN raw_sentiment rs ON ss.raw_id = rs.id
        WHERE ss.scored_date = ?
        GROUP BY ss.ticker
        HAVING COUNT(*) >= 2
    """, [run_date, run_date])
    
    if df.empty:
        logger.warning(f"[finbert] No scored data to aggregate for {run_date}")
        return {"aggregated": 0}
    
    insert_df("daily_sentiment", df, mode="replace", replace_keys=["ticker", "sentiment_date"])
    logger.info(f"[finbert] Aggregated {len(df)} tickers for {run_date}")
    return {"aggregated": len(df), "date": run_date}

if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    score_raw_sentiment(run_date)
    aggregate_daily_sentiment(run_date)