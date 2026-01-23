from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    aws_region: str = os.getenv("AWS_REGION", "eu-west-1")

    bucket: str = os.getenv("S3_BUCKET", "alpha-edge-algo")

    # Athena
    athena_workgroup: str = os.getenv("ATHENA_WORKGROUP", "primary")
    athena_output: str = os.getenv(
        "ATHENA_OUTPUT",
        f"s3://{bucket}/_athena/results/"
    )

    # Databases (Glue/Athena)
    db_curated: str = os.getenv("DB_CURATED", "curated")
    db_staged: str = os.getenv("DB_STAGED", "staged")
    db_analytics: str = os.getenv("DB_ANALYTICS", "analytics")

    # Table names
    tbl_dim_assets: str = os.getenv("TBL_DIM_ASSETS", "dim_assets")
    tbl_prices_iceberg: str = os.getenv("TBL_PRICES_ICEBERG", "prices_ohlcv_1d")
    tbl_prices_stage: str = os.getenv("TBL_PRICES_STAGE", "prices_ohlcv_1d_stage")

    # S3 prefixes
    raw_prices_prefix: str = os.getenv("RAW_PRICES_PREFIX", "raw/prices_1d")
    staged_prices_prefix: str = os.getenv("STAGED_PRICES_PREFIX", "staged/prices_1d")

    # yfinance tuning
    yf_chunk_size: int = int(os.getenv("YF_CHUNK_SIZE", "200"))
    yf_lookback_days: int = int(os.getenv("YF_LOOKBACK_DAYS", "7"))  # fetch last N days to avoid gaps


settings = Settings()
