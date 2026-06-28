import sys
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field
from loguru import logger
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import query_df
 
 
@dataclass
class CheckResult:
    name:     str
    passed:   bool
    critical: bool
    message:  str
    value:    float = 0.0
 
 
@dataclass
class ValidationReport:
    run_date:  str
    checks:    list = field(default_factory=list)
 
    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)
 
    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)
 
    @property
    def critical_failures(self) -> list:
        return [c for c in self.checks if not c.passed and c.critical]
 
    @property
    def pipeline_ok(self) -> bool:
        return len(self.critical_failures) == 0
 
    def print_summary(self):
        logger.info("=" * 55)
        logger.info(f"DATA QUALITY REPORT - {self.run_date}")
        logger.info("=" * 55)
        for c in self.checks:
            icon = "PASS" if c.passed else ("FAIL-CRITICAL" if c.critical else "WARN")
            logger.info(f"  [{icon}] {c.name}: {c.message}")
        logger.info("-" * 55)
        logger.info(
            f"  {self.passed}/{len(self.checks)} checks passed "
            f"| Pipeline OK: {self.pipeline_ok}"
        )
        logger.info("=" * 55)
 
 
def check_price_data(run_date: str) -> list:
    results = []
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM price_daily WHERE trade_date = ?",
        [run_date]
    )
    count = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="price_row_count",
        passed=count >= 20,
        critical=True,
        message=f"{count} rows (need >=20 tickers)",
        value=count,
    ))
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM price_daily "
        "WHERE trade_date = ? AND close_price IS NULL",
        [run_date]
    )
    nulls = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="price_no_null_close",
        passed=nulls == 0,
        critical=True,
        message=f"{nulls} null close prices",
        value=nulls,
    ))
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM price_daily "
        "WHERE trade_date = ? AND close_price <= 0",
        [run_date]
    )
    bad = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="price_positive_values",
        passed=bad == 0,
        critical=True,
        message=f"{bad} rows with close_price <= 0",
        value=bad,
    ))
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM price_daily "
        "WHERE trade_date = ? AND ABS(pct_change) > 30",
        [run_date]
    )
    extreme = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="price_pct_change_range",
        passed=extreme == 0,
        critical=False,
        message=f"{extreme} rows with |pct_change| > 30%",
        value=extreme,
    ))
 
    return results
 
 
def check_raw_sentiment(run_date: str) -> list:
    results = []
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM raw_sentiment "
        "WHERE collected_date = ?",
        [run_date]
    )
    count = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="sentiment_row_count",
        passed=count >= 10,
        critical=True,
        message=f"{count} posts collected (need >=10)",
        value=count,
    ))
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM raw_sentiment "
        "WHERE collected_date = ? AND (raw_text IS NULL OR LENGTH(raw_text) < 10)",
        [run_date]
    )
    empty = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="sentiment_no_empty_text",
        passed=empty == 0,
        critical=False,
        message=f"{empty} records with empty text",
        value=empty,
    ))
 
    df = query_df(
        "SELECT COUNT(DISTINCT source) AS cnt FROM raw_sentiment "
        "WHERE collected_date = ?",
        [run_date]
    )
    sources = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="sentiment_multiple_sources",
        passed=sources >= 1,
        critical=False,
        message=f"{sources} distinct sources",
        value=sources,
    ))
 
    df = query_df(
        "SELECT COUNT(DISTINCT ticker) AS cnt FROM raw_sentiment "
        "WHERE collected_date = ?",
        [run_date]
    )
    tickers = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="sentiment_ticker_coverage",
        passed=tickers >= 5,
        critical=True,
        message=f"{tickers} tickers have sentiment data (need >=5)",
        value=tickers,
    ))
 
    return results
 
 
def check_scored_sentiment(run_date: str) -> list:
    """Validates sentiment_scored table after FinBERT scoring."""
    results = []
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM sentiment_scored WHERE scored_date = ?",
        [run_date]
    )
    count = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="scored_row_count",
        passed=count >= 5,
        critical=True,
        message=f"{count} scored records (need >=5)",
        value=count,
    ))
 
    df = query_df("""
        SELECT AVG(ABS(score_positive + score_negative + score_neutral - 1.0)) AS avg_error
        FROM sentiment_scored
        WHERE scored_date = ?
    """, [run_date])
    error = float(df["avg_error"].iloc[0] or 0)
    results.append(CheckResult(
        name="scored_probability_sum",
        passed=error < 0.01,
        critical=True,
        message=f"avg probability sum error: {error:.6f} (need <0.01)",
        value=error,
    ))
 
    df = query_df("""
        SELECT COUNT(*) AS cnt
        FROM sentiment_scored
        WHERE scored_date = ?
          AND (compound_score < -1.0 OR compound_score > 1.0)
    """, [run_date])
    bad = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="scored_compound_range",
        passed=bad == 0,
        critical=True,
        message=f"{bad} records with compound_score outside [-1, 1]",
        value=bad,
    ))
 
    return results
 
 
def check_daily_sentiment(run_date: str) -> list:
    """Validates daily_sentiment aggregation."""
    results = []
 
    df = query_df(
        "SELECT COUNT(*) AS cnt FROM daily_sentiment WHERE sentiment_date = ?",
        [run_date]
    )
    count = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="daily_agg_row_count",
        passed=count >= 3,
        critical=True,
        message=f"{count} tickers aggregated (need >=3)",
        value=count,
    ))
 
    df = query_df("""
        SELECT COUNT(*) AS cnt FROM daily_sentiment
        WHERE sentiment_date = ?
          AND (avg_sentiment_score < -1.0 OR avg_sentiment_score > 1.0)
    """, [run_date])
    bad = int(df["cnt"].iloc[0])
    results.append(CheckResult(
        name="daily_agg_score_range",
        passed=bad == 0,
        critical=True,
        message=f"{bad} rows with avg_sentiment outside [-1,1]",
        value=bad,
    ))
 
    return results
 
def run_collection_checks(run_date: str = None) -> ValidationReport:
    if run_date is None:
        run_date = date.today().isoformat()
 
    logger.info(f"[validation] Running collection checks for {run_date}")
    report = ValidationReport(run_date=run_date)
 
    report.checks.extend(check_price_data(run_date))
    report.checks.extend(check_raw_sentiment(run_date))
 
    report.print_summary()
 
    if not report.pipeline_ok:
        failures = [c.name for c in report.critical_failures]
        raise ValueError(
            f"[validation] CRITICAL COLLECTION FAILURES for {run_date}: {failures}\n"
        )
 
    return report

def run_all_checks(run_date: str = None) -> ValidationReport:
    if run_date is None:
        run_date = date.today().isoformat()
 
    logger.info(f"[validation] Running all checks for {run_date}")
    report = ValidationReport(run_date=run_date)
 
    report.checks.extend(check_price_data(run_date))
    report.checks.extend(check_raw_sentiment(run_date))
    report.checks.extend(check_scored_sentiment(run_date))
    report.checks.extend(check_daily_sentiment(run_date))
 
    report.print_summary()
 
    if not report.pipeline_ok:
        failures = [c.name for c in report.critical_failures]
        raise ValueError(
            f"[validation] CRITICAL FAILURES for {run_date}: {failures}\n"
        )
 
    return report
 
 
if __name__ == "__main__":
    run_date = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        report = run_all_checks(run_date)
        print(f"\nAll checks passed: {report.passed}/{len(report.checks)}")
    except ValueError as e:
        print(f"\nValidation failed: {e}")