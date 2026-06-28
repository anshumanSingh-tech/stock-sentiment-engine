import sys
from pathlib import Path
from loguru import logger
from huggingface_hub import HfApi, login

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import DUCKDB_PATH, DUCKDB_FILENAME, HF_DATASET_REPO, HF_TOKEN
from config.database import query_df

def verify_no_stale_correlation_rows() -> bool:
    try:
        df = query_df(
            "SELECT COUNT(*) AS cnt FROM sentiment_price_correlation "
            "WHERE sample_size < 10"
        )
        stale_count = int(df["cnt"].iloc[0])
    except Exception as e:
        logger.warning(f"[upload] Could not check correlation table: {e}")
        return True
    
    if stale_count > 0:
        logger.error(
            f"[upload] Blocked: {stale_count} rows in "
            f"sentiment_price_correlation have sample_size < 10. "
            f"Clean those up before uploading: "
            f"DELETE FROM sentiment_price_correlation WHERE sample_size < 10"
        )
        return False
    
    logger.info("[upload] Correlation table check passed - no stale rows")
    return True

def upload_database() -> bool:
    if not HF_DATASET_REPO:
        logger.error(
            "[upload] HF_DATASET_REPO is not set in your .env file. "
            "Add: HF_DATASET_REPO=Anshuman0301/stock-sentiment-data"
        )
        return False
    
    if not DUCKDB_PATH.exists():
        logger.error(f"[upload] Database file not found at {DUCKDB_PATH}")
        return False
    
    if not verify_no_stale_correlation_rows():
        return False
    
    try:
        if HF_TOKEN:
            login(token=HF_TOKEN)
            
        api = HfApi()
        logger.info(
            f"[upload] Uploading {DUCKDB_PATH} -> "
            F"datasets/{HF_DATASET_REPO}/{DUCKDB_FILENAME}"
        )
        
        api.upload_file(
            path_or_fileobj=str(DUCKDB_PATH),
            path_in_repo=DUCKDB_FILENAME,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Daily data refresh",
        )
        
        logger.info("[upload] Upload complete")
        return True
    
    except Exception as e:
        logger.error(f"[upload] Upload failed: {e}")
        return False
    
if __name__ == "__main__":
    success = upload_database()
    sys.exit(0 if success else 1)