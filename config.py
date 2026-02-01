# config.py
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple

@dataclass
class Config:
    # runtime
    dry_run: bool = os.getenv("DRY_RUN", "1") == "1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # strategy gates
    min_balance_usd: float = 110.0

    # arb/risk
    buffer: float = 0.01            # 1%
    max_pm_slippage: float = 0.005   # 0.5%
    max_hedge_latency_ms: int = 800
    max_net_exposure_shares: float = 100.0
    OP_MIN_TAKER_FEE_USD = Decimal("0.5")

    # market filters
    max_end_days: int = 365 * 1
    min_volume_24h: float = 50.0
    meta_refresh_sec: int = 30 * 60
    pm_page_size: int = 500
    pm_max_market: int = 30000

    # scanning cadence
    loop_sleep_sec: float = 0.0
    price_timeout_sec: float = 10.0
    op_price_timeout_sec: float = 20.0
    book_timeout_sec: float = 6.0

    # mapping
    mapping_topk_fuzz: int = 3
    mapping_topk_sbert: int = 3
    mapping_topk_rerank: int = 5
    mapping_min_confidence: float = 0.90
    end_time_max_diff_hours: int = 24

    # order sizing ladder (USD notional)
    size_ladder_usd: Tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)

    # ===== execution constraints =====
    min_order_usd: float = float(os.getenv("MIN_ORDER_USD", "5.0"))
    target_order_usd: float = float(os.getenv("TARGET_ORDER_USD", "50.0"))
    max_order_usd: float = float(os.getenv("MAX_ORDER_USD", "50.0"))  # 先固定50，后续你想加再改
    topic_rate: float = float(os.getenv("OP_TOPIC_RATE", "0.76"))  # fee公式里的 topic_rate
    max_book_levels: int = int(os.getenv("MAX_BOOK_LEVELS", "3"))  # ask[0..2]
    hedge_poll_sec: float = float(os.getenv("HEDGE_POLL_SEC", "0.5"))  # maker填充轮询频率
    maker_max_life_sec: float = float(os.getenv("MAKER_MAX_LIFE_SEC", "30.0"))  # maker单最长挂多久

    # endpoints / keys
    opinion_api_key: str = os.getenv("OPINION_API_KEY", "")
    opinion_openapi_base: str = os.getenv("OPINION_OPENAPI_BASE", "")
    opinion_clob_host: str = os.getenv("OPINION_CLOB_HOST", "")
    opinion_private_key: str = os.getenv("OPINION_PRIVATE_KEY", "")
    opinion_multi_sig_address: str = os.getenv("OPINION_MULTI_SIG_ADDRESS", "")
    opinion_rpc_url: str = os.getenv("OPINION_RPC_URL", "")
    opinion_chain_id: int = int(os.getenv("OPINION_CHAIN_ID", "56"))
    opinion_conditional_token: str = os.getenv("CONDITIONAL_TOKEN_ADDR", "")

    poly_gamma_base: str = os.getenv("POLY_GAMMA_BASE", "https://gamma-api.polymarket.com")
    poly_clob_host: str = os.getenv("POLY_CLOB_HOST", "https://clob.polymarket.com")
    poly_private_key: str = os.getenv("POLY_PRIVATE_KEY", "")
    poly_funder: str = os.getenv("POLY_ADDRESS", "")
    poly_chain_id: int = int(os.getenv("POLY_CHAIN_ID", "137"))
    poly_rpc_url: str = os.getenv("POLYGON_RPC_URL", "")


CFG = Config()