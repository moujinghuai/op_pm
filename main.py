# main.py
import time
import logging
from datetime import datetime

import opinion_clob_sdk
import requests
from py_clob_client import BalanceAllowanceParams, AssetType
from web3 import Web3
from typing import Any, Dict, Optional, Tuple, List
from decimal import Decimal as D, Decimal, ROUND_DOWN, ROUND_UP

from dotenv import load_dotenv
from rapidfuzz import process, fuzz

from net import HTTP

load_dotenv()

from config import CFG
from clients import OpinionOpenAPI, OpinionTrader, PolymarketData, PolymarketTrader
from markets import (
    iso_to_ts, filter_pm_markets, filter_op_markets,
    normalize_pm_token_ids, pm_key,
    flatten_opinion_markets,
)
from arb import (
    parse_op_orderbook, invert_yes_book_to_no,
    find_buy_opportunity,
    take_top_levels,
    build_refined_plan, _D,
    clamp01d as _clamp01,
    _cost_and_last_price_from_asks as _pm_cost_and_last_from_asks,
    opinion_taker_fee_rate as _op_taker_fee_rate, _revenue_and_last_price_from_bids, make_ev_key,
)
from mapper import EventMapper, _HAS_ST, normalize_text, ManualEntry
from pm_balance import get_wallet_usdc_balance

BASE = "https://data-api.polymarket.com"


def norm_cond(x) -> str:
    return str(x or "").strip().lower()


def setup_logging(level: str) -> None:
    """
    Unified logging setup:
    - App logs -> file + console
    - Third-party noisy logs (httpx/httpcore) -> WARNING+
    """
    level = level.upper()

    # ===== root logger =====
    root = logging.getLogger()
    root.handlers.clear()  # 防止重复 addHandler
    root.setLevel(getattr(logging, level, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # ===== file handler =====
    fh = logging.FileHandler("arb_bot.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(fh)

    # ===== console handler =====
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(ch)

    # ===== suppress noisy libraries =====
    noisy_libs = [
        "httpx",
        "httpcore",
        "urllib3",
        "web3",
        "asyncio",
    ]
    for name in noisy_libs:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ===== SDKs (可按需加) =====
    logging.getLogger("py_clob_client").setLevel(logging.WARNING)
    logging.getLogger("opinion_clob_sdk").setLevel(logging.WARNING)

    root.info(
        "[LOG] logging initialized "
        f"(level={level}, file=arb_bot.log, console=on)"
    )


setup_logging(CFG.log_level)
log = logging.getLogger(__name__)


def _dec(x) -> D:
    try:
        return D(str(x))
    except Exception:
        return D("0")


def _with_retry(
        fn,
        *,
        what: str,
        max_retries: int = 3,
        retry_delay: float = 0.2,
) -> bool:
    """
    与旧脚本保持一致：
    - 尝试 max_retries 次
    - 每次失败记录日志
    - 成功一次就返回 True
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            ok = bool(fn())
            if ok:
                return True
            raise RuntimeError("returned falsy")
        except Exception as e:
            last_err = e
            log.error(f"[RETRY] {what} failed attempt={attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)

    log.warning(f"[RETRY] {what} failed after {max_retries} attempts: {last_err}")
    return False


def now_ts() -> int:
    return int(time.time())


def get_pm_end_ts(pm: Dict[str, Any]) -> Optional[int]:
    return iso_to_ts(pm.get("endDate") or pm.get("end_date") or "")


def get_op_end_ts(op: Dict[str, Any]) -> Optional[int]:
    """
    Opinion cutoffAt: epoch seconds (e.g. 1798675200)
    """
    try:
        v = op.get("cutoffAt")
        if v in (None, "", 0, "0"):
            return None
        return int(v)
    except Exception:
        return None


class Bot:
    def __init__(self):
        self.op_open = OpinionOpenAPI(CFG.opinion_api_key, CFG.opinion_openapi_base)
        self.op_trader = OpinionTrader(CFG)

        self.pm_trader = PolymarketTrader(CFG)
        self.pm_data = PolymarketData(CFG, clob_client=self.pm_trader.client)

        self.mapper = EventMapper(
            topk_fuzz=CFG.mapping_topk_fuzz,
            topk_sbert=CFG.mapping_topk_sbert,
            topk_rerank=CFG.mapping_topk_rerank,
            min_conf=CFG.mapping_min_confidence,
        )

        self.event_map: Dict[str, Dict[str, Any]] = {}
        self.last_meta_refresh = 0

        # risk/exposure
        self.risk_mode: Dict[str, bool] = {}
        self.net_exposure: Dict[str, float] = {}

        # maker orders
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.trade_seen: Dict[Tuple[str, int], set] = {}

        # ev_key -> 双边仓位
        # op_shares / pm_shares：两边“可用仓位真相”（每次cap都会刷新）
        # pair_shares：可双边一起卖的份额 = min(op_shares, pm_shares)
        self.positions: Dict[str, Dict[str, Any]] = {}

        # sell maker orders（OP 卖出挂单 + PM 对冲）
        self.active_sell_orders: Dict[str, Dict[str, Any]] = {}
        self.sell_trade_seen: Dict[Tuple[str, int], set] = {}

        self.last_scan_ts = 0
        self.scan_interval_sec = 120  # 3 分钟

        self.last_pm_orders_housekeep_ts = 0
        self.pm_housekeep_interval_sec = 20 * 60  # 8分钟
        self.pm_repost_age_sec = 20 * 60

        self._positions_rebuilt: bool = False

        self.rebuild_positions()

    def manage_pm_live_orders(self) -> None:
        """
        每8分钟跑一次：
        - 拉 PM 当前 open orders（get_orders）
        - 对 LIVE 且 created_at >= 10分钟 的订单：cancel + repost（按剩余份额）
        - BUY: repost price=0.97
        - SELL: repost price=0.03
        """
        now = now_ts()
        if self.last_pm_orders_housekeep_ts and (
                now - self.last_pm_orders_housekeep_ts) < self.pm_housekeep_interval_sec:
            return
        self.last_pm_orders_housekeep_ts = now

        if not self.pm_trader.enabled or not getattr(self.pm_trader, "client", None):
            return

        try:
            orders = self.pm_trader.client.get_orders()  # 不传 params -> 全账户 open orders
        except Exception as e:
            log.warning(f"[PM][HOUSEKEEP] get_orders failed: {e}")
            return

        if not orders:
            log.info("[PM][HOUSEKEEP] open_orders=0")
            return

        cancel_cnt = 0
        repost_cnt = 0
        skip_cnt = 0

        for od in orders:
            try:
                status = str(od.get("status") or "").upper()
                if status != "LIVE":
                    continue

                created_at = int(od.get("created_at") or 0)
                if created_at <= 0:
                    continue

                age = now - created_at
                if age < self.pm_repost_age_sec:
                    continue  # 未满10分钟

                order_id = str(od.get("id") or "")
                side = str(od.get("side") or "").upper()
                token_id = str(od.get("asset_id") or "")
                if not order_id or side not in ("BUY", "SELL") or not token_id:
                    continue

                original = _dec(od.get("original_size") or 0)
                matched = _dec(od.get("size_matched") or 0)
                remaining = original - matched

                # clamp
                if remaining <= D("0"):
                    skip_cnt += 1
                    log.info(
                        f"[PM][HOUSEKEEP][SKIP] oid={order_id[:10]} side={side} age={age}s "
                        f"remaining<=0 (orig={float(original):.6f}, matched={float(matched):.6f})"
                    )
                    continue

                # 你 PM 侧 shares 最常见支持到 4 位（保守一点，避免被拒单）
                remaining = remaining.quantize(D("0.0001"), rounding=ROUND_DOWN)
                if remaining <= D("0"):
                    skip_cnt += 1
                    continue

                # 1) cancel old
                try:
                    _ = self.pm_trader.client.cancel(order_id)
                    cancel_cnt += 1
                except Exception as e:
                    log.warning(f"[PM][HOUSEKEEP][CANCEL_FAIL] oid={order_id[:10]} side={side} err={e}")
                    continue

                # 2) repost by side
                if side == "BUY":
                    px = D("0.97")
                    ok = self.pm_trader.safe_buy(
                        token_id=str(token_id),
                        shares=float(remaining),
                        limit_price=float(px),
                        max_retries=3,
                        retry_delay=0.2,
                    )
                else:
                    px = D("0.03")
                    ok = self.pm_trader.safe_sell(
                        token_id=str(token_id),
                        shares=float(remaining),
                        limit_price=float(px),
                        max_retries=3,
                        retry_delay=0.2,
                    )
                
                if ok:
                    repost_cnt += 1
                    log.info(
                        f"[PM][HOUSEKEEP][REPOST_OK] side={side} age={age}s "
                        f"old_oid={order_id[:10]} rem={float(remaining):.6f} px={float(px):.2f}"
                    )
                else:
                    log.warning(
                        f"[PM][HOUSEKEEP][REPOST_FAIL] side={side} age={age}s "
                        f"old_oid={order_id[:10]} rem={float(remaining):.6f} px={float(px):.2f}"
                    )

            except Exception as e:
                log.exception(f"[PM][HOUSEKEEP] loop exception: {e}")

        log.info(
            f"[PM][HOUSEKEEP] checked={len(orders)} cancel={cancel_cnt} repost={repost_cnt} skip={skip_cnt}"
        )


    def fetch_all_op_positions(self, op_api) -> list:
        out = []
        page = 1
        limit = 10

        while True:
            resp = op_api.get_my_positions(page=page, limit=limit)
            data = getattr(resp, "result", None)
            lst = getattr(data, "list", []) if data else []

            if not lst:
                break

            for p in lst:
                if p.market_status_enum != "Activated":
                    continue
                if float(p.current_value_in_quote_token) < 0.5:
                    continue

                out.append({
                    "text": normalize_text(
                        (p.root_market_title or "") + " " + (p.market_title or "")
                    ),
                    "market_id": int(p.market_id),
                    "token_id": str(p.token_id),
                    "outcome": p.outcome,
                    "shares": float(p.shares_owned),
                })

            page += 1

        return out

    def fetch_all_pm_positions(self, user_addr: str) -> list:
        out = []
        limit = 50
        offset = 0

        while True:
            params = {
                "user": user_addr,
                "limit": limit,
                "offset": offset,
            }
            rows = HTTP.get(
                "https://data-api.polymarket.com/positions",
                params=params,
                timeout=10,
            )
            if not rows:
                break

            for r in rows:
                if float(r.get("currentValue", 0)) < 0.5:
                    continue
                if r.get("redeemable") is True:
                    continue

                out.append({
                    "ev_key": r["slug"],
                    "text": normalize_text(r["title"]),
                    "pm_token_id": r["asset"],
                    "pm_shares": float(r["size"]),
                    "pm_outcome": r.get("outcome"),
                    "pm_condition_id": str(r.get("conditionId") or ""),
                })

            offset += limit

        return out

    def norm_outcome(self, x) -> str:
        if x is None:
            return ""
        s = str(x).strip().lower()
        if s in ("yes", "y", "1", "true"):
            return "yes"
        if s in ("no", "n", "0", "false"):
            return "no"
        # 不是二元（比如 UP/DOWN）直接返回原值，后面会匹配失败
        return s

    def is_opposite(self, pm_outcome, op_outcome) -> bool:
        pm = self.norm_outcome(pm_outcome)
        op = self.norm_outcome(op_outcome)
        return (pm == "yes" and op == "no") or (pm == "no" and op == "yes")

    def match_pm_op(self, pm_list, op_list):
        used_op = set()
        matches = []

        for pm in pm_list:
            pm_cid = norm_cond(pm.get("pm_condition_id"))

            # =====  自动匹配：fuzz==100 + outcome相反 =====
            best = None
            for i, op in enumerate(op_list):
                if i in used_op:
                    continue

                # 先 outcome 相反，再比文本（更省算力）
                if not self.is_opposite(pm.get("pm_outcome"), op.get("outcome")):
                    continue

                s = fuzz.token_set_ratio(pm["text"], op["text"])
                if s == 100:
                    best = (i, op)
                    break

            if best:
                idx, op = best
                used_op.add(idx)
                matches.append((pm, op))

        return matches

    def rebuild_positions(self):
        op_pos = self.fetch_all_op_positions(self.op_trader.client)
        pm_pos = self.fetch_all_pm_positions(user_addr=CFG.poly_funder)

        pairs = self.match_pm_op(pm_pos, op_pos)
        now = int(time.time())
        self.positions = {}

        for pm, op in pairs:
            ev_key = pm["ev_key"]

            if ev_key in self.positions:
                log.debug(f"[REBUILD][SKIP] duplicate ev_key={ev_key}")
                continue

            pair_shares = min(op["shares"], pm["pm_shares"])

            self.positions[pm["ev_key"]] = {
                "op_market_id": op["market_id"],
                "op_token_id": op["token_id"],
                "pm_token_id": pm["pm_token_id"],
                "op_outcome": op["outcome"],
                "op_shares": op["shares"],
                "pm_shares": pm["pm_shares"],
                "pair_shares": pair_shares,
                "shares": pair_shares,
                "target_shares": pair_shares,
                "sold_total": 0.0,
                "isRebuild": True,
                "updated_ts": now,
            }

        log.info(f"[REBUILD] positions rebuilt: {len(self.positions)}")

    def refresh_meta(self):
        t0 = time.time()
        now = now_ts()
        max_end_ts = now + CFG.max_end_days * 86400

        # PM markets
        pm_all = []
        offset = 0
        while True:
            start = time.time()
            lst = self.pm_data.list_markets(offset=offset, timeout=CFG.price_timeout_sec)
            log.info(f"[PM]offset={offset}, len={len(lst)}, delay={(time.time() - start) * 1000} ms")
            if not lst:
                break
            pm_all.extend(lst)
            if len(lst) < CFG.pm_page_size:
                break
            offset += CFG.pm_page_size
            if len(pm_all) > CFG.pm_max_market:
                break

        # OP markets
        op_parents = []
        page = 1
        while True:
            start = time.time()
            lst = self.op_open.list_markets(
                market_type=2,
                page=page,
                limit=20,
                status="activated",
                timeout=CFG.op_price_timeout_sec,
            )
            log.info(f"[OP]page={page}, len={len(lst)}, delay={(time.time() - start) * 1000} ms")
            if not lst:
                break
            op_parents.extend(lst)
            if len(lst) < 20:
                break
            page += 1
            if page > 20:
                break

        op_rows = flatten_opinion_markets(op_parents)
        op_f = filter_op_markets(op_rows, max_end_ts)

        pm_f = filter_pm_markets(pm_all, now, max_end_ts, CFG.min_volume_24h)

        log.info(f"[META] pm={len(pm_all)} -> {len(pm_f)} | op={len(op_rows)} -> {len(op_f)}")

        self.event_map = self.mapper.build_map(
            pm_markets=pm_f,
            op_markets=op_f,
            end_time_max_diff_hours=CFG.end_time_max_diff_hours,
            get_pm_end_ts=get_pm_end_ts,
            get_op_end_ts=get_op_end_ts,
            pm_key_fn=pm_key,
            now_ts=now_ts(),
        )
        valid_items = [(k, v) for k, v in self.event_map.items() if v.get("valid")]
        valid_cnt = len(valid_items)

        log.info(
            f"[META] event_map total={len(self.event_map)} "
            f"valid={valid_cnt} in {time.time() - t0:.1f}s"
        )

        # ===== 打印每一个成功匹配的事件（含匹配方式 & 时间差）=====
        for ev_key, em in valid_items:
            pm = em["sources"]["polymarket"]
            op = em["sources"]["opinion"]

            pm_title = pm.get("question") or pm.get("title") or pm.get("slug") or ""
            op_title = op.get("marketTitle") or op.get("title") or ""

            reason = em.get("reason", "?")
            diff_sec = em.get("end_time_diff_sec")
            diff_h = em.get("end_time_diff_hours")

            # 时间差格式化（更易读）
            if diff_sec is None:
                diff_str = "N/A"
            else:
                sign = "+" if diff_sec > 0 else "-" if diff_sec < 0 else "±0"
                diff_str = f"{sign}{abs(diff_h):.2f}h"

            log.info(
                "[MATCH] "
                f"reason={reason} "
                f"dt={diff_str} | "
                f"PM='{pm_title}' | "
                f"OP='{op_title}'"
            )

        self.last_meta_refresh = now_ts()

    def check_balances(self) -> bool:
        # Opinion (CLOB) available quote balance
        op_bal = self.op_trader.get_my_balances_usd()
        if op_bal < CFG.min_balance_usd:
            log.warning(f"[BAL] Opinion balance too low: {op_bal:.2f} < {CFG.min_balance_usd}")
            return False

        # Polymarket (on-chain USDC on Polygon)
        pm_usdc = get_wallet_usdc_balance(CFG.poly_rpc_url, CFG.poly_funder)

        if pm_usdc is None:
            log.warning("[BAL] PM chain USDC balance unavailable (rpc/addr issue).")
            pm_usdc_str = "N/A"
        else:
            pm_usdc_str = f"{float(pm_usdc):.2f}"
            if float(pm_usdc) < CFG.min_balance_usd:
                log.warning(f"[BAL] PM wallet USDC too low: {pm_usdc:.2f} < {CFG.min_balance_usd}")
                return False

        log.info(f"[BAL] op_bal={op_bal:.2f}, pm_usdc={pm_usdc_str}")
        return True

    # =========================
    # Position truth source (OP / PM)
    # =========================

    def _op_get_available_shares(self, *, market_id: int, token_id: str) -> D:
        """
        Opinion：查某个 token 的可卖 shares = owned - frozen
        - 使用 SDK: op_trader.client.get_my_positions(market_id=xxx)
        - 返回 Decimal（shares）
        失败则返回 0（保守）
        """
        try:
            if not self.op_trader.enabled or not self.op_trader.client:
                return D("0")
            res = self.op_trader.client.get_my_positions(market_id=int(market_id))
            payload = getattr(res, "result", None) or getattr(res, "data", None) or res
            items = getattr(payload, "list", None) or getattr(payload, "items", None) or []
            if items is None:
                return D("0")

            tid = str(token_id)
            for it in items:
                it_tid = str(getattr(it, "token_id", None) or getattr(it, "tokenId", None) or "")
                if it_tid != tid:
                    continue

                owned_s = str(getattr(it, "shares_owned", "0") or "0")
                frozen_s = str(getattr(it, "shares_frozen", "0") or "0")
                owned = _dec(owned_s)
                frozen = _dec(frozen_s)
                avail = owned
                if avail < D("0"):
                    avail = D("0")
                return avail

            return D("0")
        except Exception as e:
            log.warning(f"[POS][OP] get_available failed market={market_id} token={str(token_id)[:12]} err={e}")
            return D("0")

    def _pm_get_available_shares(self, *, token_id: str) -> D:
        """
        Polymarket：查某个 conditional token 的可卖 shares
        - 使用 CLOB: get_balance_allowance(BalanceAllowanceParams)
        - 返回 Decimal（shares）
        失败则返回 0（保守）
        """
        try:
            if not self.pm_trader.enabled or not self.pm_trader.client:
                return D("0")

            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=str(token_id),
                signature_type=2,
            )
            j = self.pm_trader.client.get_balance_allowance(params)
            # 你实测：{'balance': '46000000', 'allowances': {...}}
            bal_raw = None
            if isinstance(j, dict):
                bal_raw = j.get("balance")
            else:
                bal_raw = getattr(j, "balance", None)

            if bal_raw is None:
                return D("0")

            # PM conditional token shares 通常 1e6 精度
            # 46 shares -> '46000000'
            bal_int = D(str(bal_raw))
            return bal_int / D("1000000")
        except Exception as e:
            log.warning(f"[POS][PM] get_available failed token={str(token_id)[:12]} err={e}")
            return D("0")

    def _cap_sell_pair_shares(
            self,
            *,
            ev_key: Optional[str] = None,
            market_id: int,
            op_token_id: str,
            pm_token_id: str,
            want: D,
    ) -> D:
        """
        卖出双边（OP 卖 + PM 卖）时：两边都要有货
        sell = min(want, op_avail, pm_avail)

        新增语义：
        - 若传入 ev_key 且 positions 里存在该事件，则把 op_av/pm_av 写回 positions：
          op_shares / pm_shares / pair_shares(+shares兼容)
        """
        want = _dec(want)
        if want <= D("0"):
            return D("0")

        op_av = self._op_get_available_shares(market_id=market_id, token_id=op_token_id)
        pm_av = self._pm_get_available_shares(token_id=pm_token_id)

        sell = want
        if op_av < sell:
            sell = op_av
        if pm_av < sell:
            sell = pm_av

        sell = sell.quantize(D("0.000001"), rounding=ROUND_DOWN)
        if sell < D("0"):
            sell = D("0")

        # ✅ 回写 positions（真相同步）
        if ev_key and ev_key in self.positions:
            pos = self.positions[ev_key]
            pos["op_shares"] = float(op_av)
            pos["pm_shares"] = float(pm_av)
            pos["pair_shares"] = float(min(op_av, pm_av))
            # 兼容字段：如果你后面还用 pos["shares"] 当“可退出份额”
            pos["shares"] = pos["pair_shares"]
            pos["updated_ts"] = now_ts()
            self.positions[ev_key] = pos

        if sell < want:
            log.info(
                f"[SELL][CheckRealPos] ev={ev_key or '-'} want={float(want):.6f} -> sell={float(sell):.6f} "
                f"(op_av={float(op_av):.6f}, pm_av={float(pm_av):.6f})"
            )
        return sell

    def _read_trade_shares(self, t: Dict[str, Any]) -> float:
        """
        从 Opinion trade 里尽量解析出“成交 shares”。
        Opinion OpenAPI 常见字段就是 shares。
        """
        for k in ("shares", "filledShares", "filled_shares", "size", "baseSize", "base_size",
                  "filledSize", "filled_size", "amount", "baseAmount"):
            v = t.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                pass
        return 0.0

    def _op_fee_usd_for_sell(self, *, topic_rate: D, price: D, notional: D) -> D:
        """
        Opinion 卖出若成 taker（价格 <= bid0），按 taker fee 估算；
        maker 则 fee=0（按你当前假设）。
        """
        r = _op_taker_fee_rate(topic_rate, price)
        fee = notional * r
        min_fee = _dec(getattr(CFG, "OP_MIN_TAKER_FEE_USD", 0.5))
        if fee < min_fee:
            fee = min_fee
        return fee

    def _cap_sell_price_to_stay_maker(self, *, px: D, bid0: Optional[D]) -> D:
        """
        卖出挂单要避免跨过 bid0 变成 taker：
        - 若 px <= bid0，则抬到 bid0 + 0.01（Opinion tick=0.01）
        """
        if bid0 is None:
            return px
        step = D("0.001")
        if px <= bid0:
            px = bid0 + step
        return px

    def _collect_new_fills(
            self,
            ev_key: str,
            market_id: int,
            *,
            want_side: str,
            want_outcome: str,
            want_order_no: Optional[str] = None,
    ) -> float:
        """
        用 get_my_trades 做“新成交检测”（分页安全版），返回本轮新增成交 shares（近似）。

        - want_side: "BUY" / "SELL"
        - want_outcome: "YES" / "NO"
        """
        want_side = str(want_side).upper()
        want_outcome = str(want_outcome).upper()

        seen_key = (ev_key, market_id, want_side, want_outcome)
        seen = self.trade_seen.get(seen_key)
        if seen is None:
            seen = set()
            self.trade_seen[seen_key] = seen

        PAGE_LIMIT = 20
        MAX_PAGES = 10

        new_sh = 0.0

        for page in range(1, MAX_PAGES + 1):
            trades = self.op_trader.get_my_trades(market_id=market_id, page=page, limit=PAGE_LIMIT)
            if not trades:
                break

            new_cnt = 0
            for t in trades:
                if want_order_no:
                    if str(t.get("order_no") or "") != str(want_order_no):
                        continue

                # 唯一 id：优先 trade_no（你示例里有）
                uid = t.get("trade_no")
                if uid in seen:
                    continue
                seen.add(uid)
                new_cnt += 1

                # === 用接口返回 side 判断 ===
                side = str(t.get("side") or "").upper()  # "BUY"/"SELL"（示例是 "Sell"）
                if side != want_side:
                    continue

                # === 用 outcome 判断 YES/NO ===
                outcome = str(t.get("outcome") or t.get("outcome_side_enum") or "").upper()
                # 兼容 "No"/"Yes"
                if outcome in ("NO", "N"):
                    outcome = "NO"
                elif outcome in ("YES", "Y"):
                    outcome = "YES"

                if outcome != want_outcome:
                    continue

                new_sh += self._read_trade_shares(t)

            # 这一页没有新增 trade -> 早停
            if new_cnt == 0:
                break
            # 返回数量 < limit -> 到底了
            if len(trades) < PAGE_LIMIT:
                break

        return float(new_sh)

    def _is_buy_order_fully_filled(self, *, op_filled: D, target: D, status_enum: str = "") -> bool:
        EPS_DONE = D("0.1")
        s = (status_enum or "").lower()
        if s in ("finished", "filled", "done", "success"):
            return True
        if target > D("0") and (op_filled + EPS_DONE) >= target:
            return True
        return False

    def _arb_ok(self, op_price: D, pm_price: D) -> bool:
        """
        判断当前价格组合是否仍有套利空间：
          op_price + pm_price < 1 - buffer
        注意：跟你信号层一致（buffer=CFG.buffer）
        """
        buf = _dec(CFG.buffer)
        return (op_price + pm_price) < (D("1") - buf)

    def _safe_cancel_op(self, order_id: str) -> bool:
        return _with_retry(
            lambda: self.op_trader.cancel_order(order_id),
            what=f"[OP] cancel order={order_id}",
            max_retries=3,
            retry_delay=0.2,
        )

    def _op_extract_order_detail(self, res: Any) -> Optional[Dict[str, Any]]:
        """
        把 Opinion get_order_by_id 的返回，抽成一个简单 dict:
          {
            filled_shares: Decimal,
            status_enum: str,
            side_enum: str,
            outcome: str,
          }
        """
        if res is None:
            return None

        # SDK: res.result.order_data
        payload = getattr(res, "result", None) or getattr(res, "data", None) or res
        order_data = getattr(payload, "order_data", None) or getattr(payload, "data", None) or payload
        if order_data is None:
            return None

        def _get(x, name: str, default=None):
            if x is None:
                return default
            return getattr(x, name, default)

        filled_shares = _dec(_get(order_data, "filled_shares", "0") or "0")
        status_enum = str(_get(order_data, "status_enum", "") or "")
        side_enum = str(_get(order_data, "side_enum", "") or "")
        outcome = str(_get(order_data, "outcome", "") or "").upper()

        return {
            "filled_shares": filled_shares,
            "status_enum": status_enum,
            "side_enum": side_enum,
            "outcome": outcome,
        }

    def _sync_and_hedge_order_fill(self, ev_key: str, od: Dict[str, Any]) -> bool:
        """
        修改点：
        - 只有当“新增未对冲的成交名义金额”>= 1.1u 才去 PM 对冲
        - 名义金额用 od["price"]（当前挂单价）估算：delta_usd = delta_shares * op_price
        - 未到阈值时不更新 filled_total，让 delta 累积
        """
        if self.risk_mode.get(ev_key):
            return False

        order_id = str(od.get("order_id") or "")
        if not order_id:
            return True

        # ===== 1) 读 OP 订单真相 =====
        try:
            res = self.op_trader.get_order_by_id(order_id)
        except Exception as e:
            log.warning(f"[nonono] ev={ev_key} get_order_by_id failed oid={order_id}: {e}")
            return True  # 网络抖动：先别乱动挂单

        info = self._op_extract_order_detail(res)
        if not info:
            return True

        op_filled = _dec(info["filled_shares"])
        status_enum = info.get("status_enum")
        side_enum = info.get("side_enum")
        outcome = info.get("outcome")

        # ===== 2) 一致性校验（防止串单）=====
        if side_enum and str(side_enum).lower() != "buy":
            log.warning(f"[SYNC] ev={ev_key} oid={order_id} not BUY side={side_enum}")
            return True

        want_outcome = str(od.get("op_outcome") or "").upper()
        if want_outcome and outcome and str(outcome).upper() != want_outcome:
            log.warning(f"[SYNC] ev={ev_key} oid={order_id} outcome mismatch got={outcome} want={want_outcome}")
            return True

        # ===== 3) 计算 delta（OP 新成交但未对冲）=====
        hedged_total = _dec(od.get("filled_total") or 0)  # 语义：已对冲 shares
        delta = op_filled - hedged_total

        # clamp: 避免偶发精度反噬
        if delta.copy_abs() < D("0.001"):
            delta = D("0")

        if delta <= D("0"):
            # 没新增成交
            return True

        # ===== 3.1) 新增：至少填充 1.1u 才对冲 =====
        HEDGE_MIN_USD = D("1.1")
        INIT_POS_MIN_USD = D("5.1")

        # 用当前挂单价估名义（你也可以换成 order_detail 的均价字段，如果 SDK 有提供）
        pm_px = _dec(od.get("pm_limit_price") or 0)
        delta_usd = delta * pm_px

        if delta_usd < HEDGE_MIN_USD:
            # 未到阈值：不对冲、不更新 filled_total，让 delta 累积
            log.info(
                f"[SYNC][HEDGE][HOLD] ev={ev_key} oid={order_id} "
                f"delta={float(delta):.6f} (~${float(delta_usd):.4f}) < {float(HEDGE_MIN_USD):.2f} "
                f"op_filled={float(op_filled):.6f} hedged={float(hedged_total):.6f}"
            )
            return True

        # ===== 3.2) 到阈值才对冲 =====
        pm_token_id = str(od.get("pm_token_id") or "")
        if not pm_token_id or pm_px <= 0:
            log.error(f"[SYNC][HEDGE] ev={ev_key} missing pm_token/pm_lim -> risk_mode")
            self.risk_mode[ev_key] = True
            return False

        log.info(
            f"[SYNC][HEDGE] ev={ev_key} oid={order_id} "
            f"delta={float(delta):.6f} (~${float(delta_usd):.4f}) "
            f"op_filled={float(op_filled):.6f} hedged={float(hedged_total):.6f}"
        )

        ok_pm = self.pm_trader.safe_buy(
            token_id=str(pm_token_id),
            shares=float(delta),
            limit_price=float(pm_px),
            max_retries=3,
            retry_delay=0.2,
        )
        if not ok_pm:
            log.error(f"[nonono] PM buy failed ev={ev_key} delta={float(delta):.6f} -> risk_mode")

            # 对冲失败：立刻撤 OP 剩余，避免继续成交扩大裸露
            try:
                self._safe_cancel_op(order_id)
            except Exception:
                pass
            return False

        # 对冲成功：把本地“已对冲”对齐到 OP 真实成交
        od["filled_total"] = float(op_filled)
        self.active_orders[ev_key] = od

        # ---- 4.1 首次 position 需要满足 OP 已填充价值 >= 5.1u ----
        # 估 OP 已填充价值：优先用订单详情均价/成交均价字段（如果你 _op_extract_order_detail 有给）
        # 否则退化为 od["price"]（你的挂单价）
        op_px = _dec(od.get("price") or 0)
        op_value = op_filled * op_px  # 注意：用“已填充总价值”做 gate（不是 delta）

        can_update_pos = True
        status_enum_l = str(status_enum or "")
        if ev_key not in self.positions:
            if op_value < INIT_POS_MIN_USD:
                can_update_pos = False
                log.info(
                    f"[SYNC][POS][HOLD_INIT] ev={ev_key} oid={order_id} "
                    f"op_filled={float(op_filled):.6f} op_px={float(op_px):.4f} "
                    f"(~OP ${float(op_value):.4f}) < {float(INIT_POS_MIN_USD):.2f} (init gate)"
                )

        if can_update_pos:
            # 用“当前已填充份额”更新/写入 position（不是等 fully filled）
            # 这里沿用你已有提交函数；如果 _commit_position_full 名字不合适但能覆盖更新逻辑，就继续用它
            # 关键：filled_shares=op_filled（或你希望的 final_sh）
            self._commit_position_full(ev_key, od, filled_shares=op_filled, status_enum=status_enum_l)
            log.info(
                f"[SYNC][POS][UPDATE] ev={ev_key} oid={order_id} "
                f"pos_shares={float(op_filled):.6f} (~OP ${float(op_value):.4f})"
            )

        # ===== 5) 如果填充完毕：从 active_orders 清除 =====
        target = _dec(od.get("target_shares") or 0)

        if self._is_buy_order_fully_filled(op_filled=op_filled, target=target, status_enum=status_enum_l):
            log.info(
                f"[DONE][BUY] ev={ev_key} oid={order_id} "
                f"filled={float(op_filled):.6f}/{float(target):.6f} status={status_enum_l} -> drop active_order"
            )
            self.active_orders.pop(ev_key, None)
            return False

        return True

    def _commit_position_full(self, ev_key: str, od: Dict[str, Any], filled_sh: D = None, *,
                              filled_shares: D = None, status_enum: str = "") -> None:
        """
        写入/更新 positions 的“基础元信息”，并初始化/更新 target_shares。
        - filled_sh / filled_shares: 兼容两种调用方式
        """
        if filled_shares is None:
            filled_shares = filled_sh

        market_id = int(od.get("market_id") or 0)
        op_token_id = str(od.get("op_token_id") or "")
        pm_token_id = str(od.get("pm_token_id") or "")
        op_outcome = str(od.get("op_outcome") or "").upper()

        if market_id <= 0 or not op_token_id or not pm_token_id:
            return

        fs = _dec(filled_shares).quantize(D("0.000001"), rounding=ROUND_DOWN)
        if fs <= D("0"):
            return

        now = now_ts()

        # 允许覆盖更新，但保留 target/sold 的语义
        old = self.positions.get(ev_key) or {}

        # target_shares：初始化为 fs，或如果已有则取 max(旧, fs)
        old_target = _dec(old.get("target_shares") or 0)
        new_target = old_target if old_target > fs else fs

        self.positions[ev_key] = {
            "op_market_id": market_id,
            "op_token_id": op_token_id,
            "pm_token_id": pm_token_id,
            "op_outcome": op_outcome,

            # 真相仓位：这里仅用于“初始化”，后续以 _cap_sell_pair_shares 刷新为准
            "op_shares": float(fs),
            "pm_shares": float(fs),
            "pair_shares": float(fs),

            # 兼容旧字段
            "shares": float(fs),

            # 策略层目标&已卖累计
            "target_shares": float(new_target),
            "sold_total": float(_dec(old.get("sold_total") or 0)),

            "updated_ts": now,
        }

        s = (status_enum or "").lower()
        if s in ("finished", "filled", "done", "success"):
            felled = min(fs, new_target)
            self.positions[ev_key]["target_shares"] = float(felled)
            self.positions[ev_key]["isActiveOrderFull"] = True

    def _reprice_op_order(
            self,
            ev_key: str,
            od: Dict[str, Any],
            *,
            new_price: D,
            remaining_shares: D,
    ) -> bool:
        """
        Opinion 没有“改价”，用 cancel + place 新单模拟。
        只对剩余份额重挂（避免重复对冲）。
        """
        old_price = _dec(od.get("price") or 0)
        old_id = str(od.get("order_id") or "")
        market_id = int(od.get("market_id") or 0)
        token_id = str(od.get("op_token_id") or "")
        if not old_id or market_id <= 0 or not token_id:
            return False
        if remaining_shares <= D("0"):
            # 没剩余了，直接删
            self._safe_cancel_op(old_id)
            self.active_orders.pop(ev_key, None)
            return True

        if not self._sync_and_hedge_order_fill(ev_key, od):
            return True # risk_mode 或已完成清理

        # 先撤旧
        cancel_ok = self._safe_cancel_op(old_id)

        if not cancel_ok:
            try:
                op_token_id = str(od.get("op_token_id") or token_id)
                pm_token_id = str(od.get("pm_token_id") or "")
                if not pm_token_id or not op_token_id:
                    self.risk_mode[ev_key] = True
                    log.warning(f"[MAKER][CANCEL_FAIL][NO_TOKENS] ev={ev_key} -> risk_mode")
                    return False

                op_av = D(str(self._op_get_available_shares(market_id=market_id, token_id=op_token_id)))
                pm_av = D(str(self._pm_get_available_shares(token_id=pm_token_id)))
                diff = op_av - pm_av

                pm_px = D("0.9")
                ok_pm = self.pm_trader.safe_buy(
                    token_id=str(pm_token_id),
                    shares=float(remaining_shares),
                    limit_price=float(pm_px),
                    max_retries=3,
                    retry_delay=0.2,
                )
                log.warning(
                    f"[MAKER][CANCEL_FAIL][PM_BUY_{'OK' if ok_pm else 'FAIL'}] ev={ev_key} "
                    f"op_av={float(op_av):.6f} pm_av={float(pm_av):.6f} remaining_shares={float(remaining_shares):.6f} px=0.99 -> risk_mode"
                )
                if ok_pm:
                    self.active_orders.pop(ev_key, None)
                    log.info(
                        f"[MAKER][DROP] ev={ev_key}"
                    )
            except Exception as e:
                self.risk_mode[ev_key] = True
                log.exception(f"[MAKER][CANCEL_FAIL][EXC] ev={ev_key} -> risk_mode: {e}")

            return False

        # 再下新单（只下剩余 shares）
        new_oid = self.op_trader.safe_place_limit_buy_by_shares(
            market_id=market_id,
            token_id=token_id,
            price=float(new_price),
            shares=float(remaining_shares),
            max_retries=3,
            retry_delay=0.2,
        )
        if not new_oid:
            log.warning(f"[MAKER][REPRICE] failed ev={ev_key} new_px={float(new_price):.4f} -> risk_mode")
            self.risk_mode[ev_key] = True
            return False

        od["order_id"] = new_oid
        od["price"] = float(new_price)
        od["last_reprice_ts"] = now_ts()
        self.active_orders[ev_key] = od

        log.info(
            f"[MAKER][REPRICE] ev={ev_key} price {old_price} -> {float(new_price):.4f} "
            f"remain_sh={float(remaining_shares):.6f} oid={new_oid}"
        )
        return True

    def manage_active_orders_before_hedge(self) -> None:
        """
        在第二阶段对冲前，重新评估 self.active_orders 中所有事件：
        分析1：若当前事件不再有套利空间 -> cancel OP 限价单并移除
        分析2：根据你给的规则决定是否需要改价（cancel+recreate）
        """
        REPRICE_COOLDOWN = 2  # 秒，防止每轮都抖动（可按需调）
        TOL = D("0.0001")

        for ev_key, od in list(self.active_orders.items()):
            log.info(ev_key)
            if self.risk_mode.get(ev_key):
                log.info(f"[nonono] ev={ev_key} in risk_mode")
                continue

            # 事件是否还在映射里（或已失效）
            em = self.event_map.get(ev_key)
            if not em or not em.get("valid"):
                oid = str(od.get("order_id") or "")
                if oid:
                    self._safe_cancel_op(oid)
                self.active_orders.pop(ev_key, None)
                log.info(f"[not in event_map] ev={ev_key} 取消当前订单并移出active_orders")
                continue

            # 基础字段
            order_id = str(od.get("order_id") or "")
            market_id = int(od.get("market_id") or 0)
            op_token_id = str(od.get("op_token_id") or "")
            pm_token_id = str(od.get("pm_token_id") or "")
            if not order_id or market_id <= 0 or not op_token_id or not pm_token_id:
                log.info(f"ev_key:{ev_key} not order_id or market_id <= 0 or not op_token_id or not pm_token_id")
                continue

            # 防抖（可选）
            '''
            last_reprice_ts = int(od.get("last_reprice_ts") or 0)
            if last_reprice_ts and (now_ts() - last_reprice_ts) < REPRICE_COOLDOWN:
                continue
            '''

            # ===== 0) 关键：先同步 OP 成交并补对冲，再做任何撤单/改价 =====
            if not self._sync_and_hedge_order_fill(ev_key, od):
                continue  # risk_mode 或已完成清理

            # 只对“剩余未成交部分”做管理
            target = _dec(od.get("target_shares") or 0)
            op_filled = _dec(od.get("filled_total") or 0)  # 此时已经被 sync 对齐为 OP真实成交(=已对冲)
            remaining = target - op_filled

            if remaining <= D("0.1"):
                # 已经满了就别管
                continue

            my_px = _dec(od.get("price") or 0)

            # 拉 OP orderbook（当前 outcome 对应 token）
            try:
                ob = self.op_open.get_orderbook(op_token_id, timeout=CFG.book_timeout_sec)
            except Exception:
                continue

            bids, asks = parse_op_orderbook(ob)
            if not bids:
                # 没有买盘，基本没法按你规则调价；保守撤单
                self._safe_cancel_op(order_id)
                self.active_orders.pop(ev_key, None)
                log.info(f"[nonono] ev={ev_key} no bids -> cancel")
                continue

            bid0, bid0_sz = bids[0][0], bids[0][1]
            bid1 = bids[1][0] if len(bids) >= 2 else None
            ask0 = asks[0][0] if asks else None

            # ===== 1) 拉 PM 完整 orderbook（用于深度一致性判断）=====
            try:
                pm_ob = self.pm_data.get_order_book(pm_token_id, timeout=CFG.book_timeout_sec)
            except Exception:
                continue

            pm_bids, pm_asks = self.pm_data.parse_orderbook_summary_levels(pm_ob)
            if not pm_asks:
                continue

            # 与 build_refined_plan 一致：只取 topN 档（你前面也是这么做的）
            pm_asksN = take_top_levels(pm_asks, CFG.max_book_levels)
            if not pm_asksN:
                continue

            # ===== 2) 用“深度 + 费用 + exec_buffer”一致性判断是否仍可套利 =====
            min_usd = _dec(CFG.min_order_usd)
            max_usd = _dec(CFG.max_order_usd)
            exec_buffer = _dec("0.01")  # build_refined_plan 里同样用 0.01
            OP_MIN_FEE = _dec(getattr(CFG, "OP_MIN_TAKER_FEE_USD", 0.5))  # 你 arb.py 里就是 0.5

            def _feasible_pm_oplim_by_depth(op_px: D):
                """
                返回 (ok, pm_last, sum_per_share)
                ok=False 时 pm_last/sum 为 None
                """
                if remaining <= D("0.1") or op_px <= D("0"):
                    return (False, None, None)

                pm_cost, pm_last = _pm_cost_and_last_from_asks(pm_asksN, remaining)
                if pm_cost is None or pm_last is None:
                    return (False, None, None)

                op_quote = remaining * op_px
                if pm_cost < min_usd or op_quote < min_usd:
                    return (False, None, None)
                if pm_cost > max_usd or op_quote > max_usd:
                    return (False, None, None)

                fee_usd = D("0")  # 你目前这里没算 fee，就先保持一致

                sum_per_share = (pm_cost + op_quote + fee_usd) / remaining
                ok = sum_per_share <= (D("1") - exec_buffer)
                return (ok, pm_last, sum_per_share)

            # ========= 分析1：当前事件是否还存在套利空间（考虑 PM 深度 + fee + buffer 一致性） =========
            ok, pm_last, ssum = _feasible_pm_oplim_by_depth(bid0)
            if not ok:
                ok, pm_last, ssum = _feasible_pm_oplim_by_depth(bid1)
                if ok:
                    if my_px != (bid1):
                        self._reprice_op_order(ev_key, od, new_price=(bid1), remaining_shares=remaining)
                else:
                    cancel_ok = self._safe_cancel_op(order_id)
                    if not cancel_ok:
                        try:
                            op_av = D(str(self._op_get_available_shares(market_id=market_id, token_id=op_token_id)))
                            pm_av = D(str(self._pm_get_available_shares(token_id=pm_token_id)))
                            diff = op_av - pm_av
                            if op_av >= D("7") and diff >= D("7"):
                                delta = diff
                                pm_px = D("0.9")
                                ok_pm = self.pm_trader.safe_buy(
                                    token_id=str(pm_token_id),
                                    shares=float(delta),
                                    limit_price=float(pm_px),
                                    max_retries=3,
                                    retry_delay=0.2,
                                )
                                log.warning(
                                    f"[MAKER][CANCEL_FAIL][PM_BUY_{'OK' if ok_pm else 'FAIL'}] ev={ev_key} "
                                    f"op_av={float(op_av):.6f} pm_av={float(pm_av):.6f} delta={float(delta):.6f} px=0.99 -> risk_mode"
                                )
                                if ok_pm:
                                    self.active_orders.pop(ev_key, None)
                                    log.info(
                                        f"[MAKER][DROP] ev={ev_key}"
                                    )
                            else:
                                self.risk_mode[ev_key] = True
                                log.warning(
                                    f"[MAKER][CANCEL_FAIL][NO_PM_BUY] ev={ev_key} "
                                    f"op_av={float(op_av):.6f} pm_av={float(pm_av):.6f} diff={float(diff):.6f} -> risk_mode"
                                )
                        except Exception as e:
                            self.risk_mode[ev_key] = True
                            log.exception(f"[MAKER][CANCEL_FAIL][EXC] ev={ev_key} -> risk_mode: {e}")

                        continue

                    self.active_orders.pop(ev_key, None)
                    log.info(
                        f"[MAKER][DROP] ev={ev_key} no edge by depth now "
                        f"(remain={float(remaining):.6f} my_px={float(my_px):.4f} bid0={bid0})"
                    )
                    continue

            # still ok -> 刷新 PM 对冲价格（pm_last 是“吃到 remaining 的最后触达价”）
            if pm_last is not None:
                old_pm_lim = float(od.get("pm_limit_price") or 0.0)
                new_pm_lim = float(pm_last)
                if abs(old_pm_lim - new_pm_lim) > 1e-9:
                    od["pm_limit_price"] = new_pm_lim
                    self.active_orders[ev_key] = od
                    log.info(
                        f"[HEDGE][PM_LIM][UPDATE] ev={ev_key} "
                        f"pm_lim {old_pm_lim:.4f} -> {new_pm_lim:.4f} (sum={float(ssum):.6f})"
                    )

            if (my_px != bid0) and ((bid0_sz - remaining) > _dec("20")):
                spread = (ask0 - bid0) if ask0 is not None else None
                if spread is not None and spread >= _dec("0.002"):
                    ok, pm_last, ssum = _feasible_pm_oplim_by_depth(bid0 + _dec(0.001))
                    if ok:
                        self._reprice_op_order(ev_key, od, new_price=(bid0 + _dec(0.001)), remaining_shares=remaining)
                    else:
                        self._reprice_op_order(ev_key, od, new_price=bid0, remaining_shares=remaining)
                else:
                    self._reprice_op_order(ev_key, od, new_price=bid0, remaining_shares=remaining)


    def process_active_orders_hedge(self) -> None:
        """
        第二阶段：遍历所有 active_orders
        直接用 get_order_by_id 的 filled_shares 作为真相源，补齐 delta 对冲
        """
        for ev_key, od in list(self.active_orders.items()):
            if self.risk_mode.get(ev_key):
                continue
            # 复用同一套同步+对冲逻辑
            self._sync_and_hedge_order_fill(ev_key, od)

    def try_execute_sell_for_positions(self) -> None:
        """
        对所有已有仓位事件：
        - 若已存在 active_sell_orders 则跳过
        - 拉 OP/PM 订单薄，判断：
          1) sum_bid(taker 含 fee) > 1.01 -> 双边市价卖
          2) 否则 -> OP 挂限价卖（maker），PM 走对冲（按填充轮询）

        修改点：
        - 先用真相源拿到 op_av / pm_av（可卖 shares）
        - 若 |op_av - pm_av| < 0.1：下单时允许“OP 用 op_av，PM 用 pm_av”
        - 否则：统一按 min(op_av, pm_av) 下单（两边对齐）
        """
        THRESH = D("1.01")
        exec_buffer = D("0.00")
        topic_rate = _dec(CFG.topic_rate)

        DIFF_EPS = D("5")  # 你要求的阈值
        MIN_SH = D("3")

        for ev_key, pos in list(self.positions.items()):
            if self.risk_mode.get(ev_key):
                log.info(f"[nonono] risk_mode ev_key={ev_key}")
                continue

            # 1) ids
            market_id = int(pos.get("op_market_id") or 0)
            op_token_id = str(pos.get("op_token_id") or "")
            pm_token_id = str(pos.get("pm_token_id") or "")
            if market_id <= 0 or not op_token_id or not pm_token_id:
                continue

            # 2) 真相仓位（同时回写 positions）
            #    want 用一个很大的数，目的就是“拿到真实 op_av/pm_av 并计算 pair_shares”
            if not pos.get("isRebuild"):
                want = D("1000000000")
                _ = self._cap_sell_pair_shares(
                    ev_key=ev_key,
                    market_id=market_id,
                    op_token_id=op_token_id,
                    pm_token_id=pm_token_id,
                    want=want,
                )
                if pos.get("isActiveOrderFull"):
                    self.positions[ev_key]["isRebuild"] = True

            # 从 positions 里取回写后的真相
            pos2 = self.positions.get(ev_key, pos)
            op_av = _dec(pos2.get("op_shares") or 0)
            pm_av = _dec(pos2.get("pm_shares") or 0)
            op_av = _dec(op_av).quantize(D("0.1"), rounding=ROUND_DOWN)
            pm_av = _dec(pm_av).quantize(D("0.1"), rounding=ROUND_DOWN)

            if op_av <= MIN_SH and pm_av <= MIN_SH:
                continue

            pos_target = _dec(pos2.get("target_shares") or 0)
            pos_sold = _dec(pos2.get("sold_total") or 0)
            exit_remaining = pos_target - pos_sold
            if exit_remaining < D("0"):
                exit_remaining = D("0")

            diff = (op_av - pm_av).copy_abs()
            use_each_side_qty = diff < DIFF_EPS

            if use_each_side_qty:
                op_sh = op_av
                pm_sh = pm_av
                # 用最小的一边做“阈值评估”的 shares（保守，不会高估）
                sh_eval = op_sh if op_sh < pm_sh else pm_sh
            else:
                # 统一按较小者
                sh = op_av if op_av < pm_av else pm_av
                op_sh = sh
                pm_sh = sh
                sh_eval = sh

            # ===== 检查是否已有卖单，但 target_shares 不够（需要重挂） =====
            if ev_key in self.active_sell_orders:
                od = self.active_sell_orders[ev_key]
                old_target = _dec(od.get("target_shares") or 0)

                # position 中的份额明显变大
                if pos_target > old_target + DIFF_EPS:
                    log.info(
                        f"[已存在的卖单,份额增加了] ev={ev_key} "
                        f"op_sh increased {float(old_target):.6f} -> {float(pos_target):.6f}, {float(op_sh):.6f}, repost sell order"
                    )

                    # 1) 撤掉旧 OP 卖单
                    old_oid = od.get("order_id")
                    if old_oid:
                        try:
                            ok = self._safe_cancel_op(old_oid)
                            if ok:
                                self.active_sell_orders.pop(ev_key, None)
                            else:
                                log.warning(f"[nonono] cancel returned falsy ev={ev_key} oid={old_oid}")
                        except Exception:
                            log.exception(f"[nonono] cancel old order failed ev={ev_key}")
                else:
                    # 旧单仍然覆盖当前仓位，无需任何操作
                    continue

            # 太小就不动
            if sh_eval <= MIN_SH:
                log.info(
                    f"[EXIT][SKIP] ev={ev_key} sh too small "
                    f"(op_av={float(op_av):.6f}, pm_av={float(pm_av):.6f})"
                )
                continue

            # ===== 1) 拉 OP orderbook =====
            try:
                ob = self.op_open.get_orderbook(op_token_id, timeout=CFG.book_timeout_sec)
            except Exception:
                continue
            op_bids, op_asks = parse_op_orderbook(ob)
            if not op_bids or not op_asks:
                continue
            bid0 = op_bids[0][0]
            ask0 = op_asks[0][0]

            # ===== 2) 拉 PM orderbook =====
            pm_ob = self.pm_data.get_order_book(pm_token_id, timeout=CFG.book_timeout_sec)
            if not pm_ob:
                continue
            pm_bids, pm_asks = self.pm_data.parse_orderbook_summary_levels(pm_ob)
            if not pm_bids:
                continue

            pm_bidsN = take_top_levels(pm_bids, CFG.max_book_levels)
            if not pm_bidsN:
                continue

            # ===== 3) 先判断“能否双边市价卖” =====
            # 评估：按 sh_eval（最小边）做阈值判断，避免高估
            pm_rev_eval, pm_last_eval = _revenue_and_last_price_from_bids(pm_bidsN, sh_eval)
            if pm_rev_eval is None or pm_last_eval is None:
                continue
            pm_vwap_eval = pm_rev_eval / sh_eval

            op_bidsN = take_top_levels(op_bids, 3)
            if not op_bidsN:
                continue
            op_rev_eval, op_last_eval = _revenue_and_last_price_from_bids(op_bidsN, sh_eval)
            if op_rev_eval is None or op_last_eval is None:
                continue
            op_vwap_eval = op_rev_eval / sh_eval

            op_fee_eval = self._op_fee_usd_for_sell(
                topic_rate=topic_rate,
                price=op_vwap_eval,
                notional=op_rev_eval,
            )

            sum_bid_mkt_eval = (pm_rev_eval + (op_rev_eval - op_fee_eval)) / sh_eval

            if sum_bid_mkt_eval >= (THRESH + exec_buffer):
                # ===== Case 1: 双边市价卖 =====
                log.info(
                    f"[SELL][MKT] ev={ev_key} "
                    f"use_each={use_each_side_qty} diff={float(diff):.6f} "
                    f"op_sh={float(op_sh):.6f} pm_sh={float(pm_sh):.6f} eval_sh={float(sh_eval):.6f} "
                    f"sum_bid≈{float(sum_bid_mkt_eval):.6f} "
                    f"pm_vwap≈{float(pm_vwap_eval):.4f} pm_last≈{float(pm_last_eval):.4f} "
                    f"op_vwap≈{float(op_vwap_eval):.4f} op_last≈{float(op_last_eval):.4f} fee≈{float(op_fee_eval):.4f}"
                )

                # OP：用自己的份额卖
                if op_sh > MIN_SH:
                    oid = self.op_trader.safe_place_limit_sell_by_shares(
                        market_id=market_id,
                        token_id=op_token_id,
                        price=float(op_last_eval),  # marketable
                        shares=float(op_sh),
                        max_retries=3,
                        retry_delay=0.2,
                    )
                    if not oid:
                        log.info(f"OP Sell Fail!!!")
                        continue
                    log.info(f"OP Sell Success!!!")

                # PM：用自己的份额卖（price 用 eval 得到的 last，保守足够）
                if pm_sh > MIN_SH:
                    ok_pm = self.pm_trader.safe_sell(
                        token_id=pm_token_id,
                        shares=float(pm_sh),
                        limit_price=float(pm_last_eval),
                        max_retries=3,
                        retry_delay=0.2,
                    )
                    if not ok_pm:
                        self.risk_mode[ev_key] = True
                        log.error(f"[EXIT][MKT] PM sell failed ev={ev_key} -> risk_mode")
                        continue
                    log.info(f"PM Sell Success!!!")

                # 卖出后：刷新一次真相；若两边都接近 0 再删 positions
                _ = self._cap_sell_pair_shares(
                    ev_key=ev_key,
                    market_id=market_id,
                    op_token_id=op_token_id,
                    pm_token_id=pm_token_id,
                    want=want,
                )
                pos3 = self.positions.get(ev_key)
                if pos3:
                    op_left = _dec(pos3.get("op_shares") or 0)
                    pm_left = _dec(pos3.get("pm_shares") or 0)
                    if op_left <= MIN_SH and pm_left <= MIN_SH:
                        self.positions.pop(ev_key, None)
                else:
                    self.positions.pop(ev_key, None)

                continue

            # ===== Case 2: 不满足市价阈值 -> OP 挂 maker 卖，等填充后 PM 对冲 =====
            # 目标：选 OP_limit 使得 (PM市价卖vwap + OP_limit) > 1.01
            # 这里 PM_vwap 用 eval_sh 的（保守），OP挂单 shares 用 op_sh
            log.info(
                f"[SELL][LIMIT] ev={ev_key} "
            )
            need_op_px = (THRESH + exec_buffer) - pm_vwap_eval
            if need_op_px <= D("0"):
                need_op_px = ask0

            op_limit = self._cap_sell_price_to_stay_maker(px=need_op_px, bid0=bid0)
            op_limit = _dec(op_limit).quantize(D("0.001"), rounding=ROUND_UP)

            if op_sh <= MIN_SH:
                # OP 没货就没法挂卖单（即使 PM 有货也没意义）
                continue

            oid = self.op_trader.safe_place_limit_sell_by_shares(
                market_id=market_id,
                token_id=op_token_id,
                price=float(op_limit),
                shares=float(op_sh),  # ✅ OP 用自己的份额
                max_retries=3,
                retry_delay=0.2,
            )
            if not oid:
                log.info(f"OP Limit Sell Order Fail!!!")
                continue
            log.info(f"OP Limit Sell Order_ID:{oid}")

            # pm_limit_price：用“评估份额 sh_eval”在 PM bids 上的 last（保守）
            self.active_sell_orders[ev_key] = {
                "order_id": oid,
                "market_id": market_id,
                "op_token_id": op_token_id,
                "pm_token_id": pm_token_id,

                # ✅ target_shares 代表 OP 这边的卖出目标（因为 sell-fill 以 OP 为真相源）
                "target_shares": float(op_sh),
                "op_balance": op_av,
                "pm_balance": pm_av,

                "filled_total": 0.0,
                "price": float(op_limit),
                "pm_limit_price": float(pm_last_eval),

                "created_ts": now_ts(),
                "last_reprice_ts": 0,

                # 可选：把当时的 pm_av 记一下，便于日志/排查
                "pm_shares_snapshot": float(pm_av),
                "use_each_side_qty": bool(use_each_side_qty),
                "diff_snapshot": float(diff),
            }

            log.info(
                f"[EXIT][LIM] ev={ev_key} use_each={use_each_side_qty} diff={float(diff):.6f} "
                f"op_sh={float(op_sh):.6f} pm_sh={float(pm_sh):.6f} eval_sh={float(sh_eval):.6f} "
                f"op_lim={float(op_limit):.4f} (bid0={float(bid0):.4f} ask0={float(ask0):.4f}) "
                f"pm_lim≈{float(pm_last_eval):.4f} pm_vwap≈{float(pm_vwap_eval):.4f}"
            )

    def _reprice_op_sell_order(self, ev_key: str, od: Dict[str, Any], *, new_price: D, remaining_shares: D) -> bool:
        old_id = str(od.get("order_id") or "")
        market_id = int(od.get("market_id") or 0)
        token_id = str(od.get("op_token_id") or "")
        if not old_id or market_id <= 0 or not token_id:
            return False

        if remaining_shares <= D("0"):
            self._safe_cancel_op(old_id)
            self.active_sell_orders.pop(ev_key, None)
            return True

        if not self._sync_and_hedge_sell_order_fill(ev_key, od):
            return True

        cancel_ok = self._safe_cancel_op(old_id)

        if not cancel_ok:
            # ===== cancel 失败 -> 查真相仓位 -> PM 0.001 dump =====
            try:
                pos = self.positions.get(ev_key) or {}
                pm_token_id = str(pos.get("pm_token_id") or "")
                op_token_id = str(pos.get("op_token_id") or "") or token_id  # 兜底用 od 的

                if not pm_token_id or not op_token_id:
                    self.risk_mode[ev_key] = True
                    log.warning(f"[SELL][CANCEL_FAIL][NO_TOKENS] ev={ev_key} -> risk_mode")
                    return False

                op_av = D(str(self._op_get_available_shares(market_id=market_id, token_id=op_token_id)))
                pm_av = D(str(self._pm_get_available_shares(token_id=pm_token_id)))

                diff = pm_av - op_av

                # 条件：pm_av - op_av > 7
                if diff > D("7"):
                    if op_av <= D("3"):
                        # op_av<=3 -> PM 卖出 pm_av
                        delta_cap = pm_av
                    else:
                        # op_av>3 -> PM 卖出 (pm_av - op_av)
                        delta_cap = diff

                    if delta_cap > D("0"):
                        pm_lim = D("0.02")
                        ok_pm = self.pm_trader.safe_sell(
                            token_id=pm_token_id,
                            shares=float(delta_cap),
                            limit_price=float(pm_lim),
                            max_retries=3,
                            retry_delay=0.2,
                        )
                        if ok_pm:
                            # 建议：直接进 risk_mode，避免 OP 侧继续抖动/重复撤单失败
                            log.warning(
                                f"[SELL][CANCEL_FAIL][PM_DUMP_OK] ev={ev_key} "
                                f"pm_av={float(pm_av):.6f} op_av={float(op_av):.6f} diff={float(diff):.6f} "
                                f"dump={float(delta_cap):.6f} px=0.001 -> risk_mode"
                            )
                        else:
                            self.risk_mode[ev_key] = True
                            log.warning(
                                f"[SELL][CANCEL_FAIL][PM_DUMP_FAIL] ev={ev_key} "
                                f"pm_av={float(pm_av):.6f} op_av={float(op_av):.6f} diff={float(diff):.6f} "
                                f"dump={float(delta_cap):.6f} px=0.001 -> risk_mode"
                            )
                    else:
                        self.risk_mode[ev_key] = True
                        log.warning(
                            f"[SELL][CANCEL_FAIL][ZERO_DUMP] ev={ev_key} pm_av={float(pm_av):.6f} op_av={float(op_av):.6f} -> risk_mode"
                        )
                else:
                    # cancel 失败但没满足 dump 条件：也进 risk_mode 更安全
                    self.risk_mode[ev_key] = True
                    log.warning(
                        f"[SELL][CANCEL_FAIL][NO_DUMP] ev={ev_key} pm_av={float(pm_av):.6f} op_av={float(op_av):.6f} diff={float(diff):.6f} -> risk_mode"
                    )

            except Exception as e:
                self.risk_mode[ev_key] = True
                log.exception(f"[SELL][CANCEL_FAIL][EXC] ev={ev_key} -> risk_mode: {e}")

            return False  # cancel 失败：本次不继续 reprice（避免重复挂）

        new_oid = self.op_trader.safe_place_limit_sell_by_shares(
            market_id=market_id,
            token_id=token_id,
            price=float(new_price),
            shares=float(remaining_shares),
            max_retries=3,
            retry_delay=0.2,
        )
        if not new_oid:
            self.risk_mode[ev_key] = True
            log.warning(f"[SELL][REPRICE] failed ev={ev_key} -> risk_mode")
            return False

        od["order_id"] = new_oid
        od["price"] = float(new_price)
        od["last_reprice_ts"] = now_ts()
        self.active_sell_orders[ev_key] = od
        log.info(f"[SELL][REPRICE] ev={ev_key} px->{float(new_price):.4f} remain={float(remaining_shares):.6f}")
        return True

    def manage_active_sell_orders_before_hedge(self) -> None:
        TOL = D("0.0001")

        for ev_key, od in list(self.active_sell_orders.items()):
            if self.risk_mode.get(ev_key):
                log.info(f"[nonono] risk_mode ev_key={ev_key}")
                continue

            # 先同步成交+补对冲
            if not self._sync_and_hedge_sell_order_fill(ev_key, od):
                continue
            target = _dec(od.get("target_shares") or 0)
            filled = _dec(od.get("filled_total") or 0)
            remaining = target - filled
            if remaining <= D("0.1"):
                continue

            if od.get("isNeedCheckReal"):
                # NEW: remaining 也要按真仓位 cap（避免重挂/对冲失败）
                remaining_want = remaining
                try:
                    remaining = self._cap_sell_pair_shares(
                        ev_key=ev_key,
                        market_id=int(od.get("market_id") or 0),
                        op_token_id=str(od.get("op_token_id") or ""),
                        pm_token_id=str(od.get("pm_token_id") or ""),
                        want=remaining_want,
                    )
                except Exception:
                    # 不清 flag，让下轮还能再查一次
                    continue

                od["isNeedCheckReal"] = False
                self.active_sell_orders[ev_key] = od

                if remaining <= D("0.1"):
                    log.info(
                        f"[SELL][SKIP] ev={ev_key} remaining capped too small "
                        f"(want={float(remaining_want):.6f})"
                    )
                    continue
            else:
                pos = self.positions.get(ev_key) or {}
                op_av = _dec(pos.get("op_shares") or 0)
                pm_av = _dec(pos.get("pm_shares") or 0)
                remaining = min(op_av, pm_av, remaining)

            op_token_id = str(od.get("op_token_id") or "")
            pm_token_id = str(od.get("pm_token_id") or "")
            order_id = str(od.get("order_id") or "")
            my_px = _dec(od.get("price") or 0)

            # OP book
            try:
                ob = self.op_open.get_orderbook(op_token_id, timeout=CFG.book_timeout_sec)
            except Exception:
                continue
            bids, asks = parse_op_orderbook(ob)
            if not bids or not asks:
                continue

            bid0, bid0_sz = bids[0][0], bids[0][1]
            ask0, ask0_sz = asks[0][0], asks[0][1]
            ask1 = asks[1][0] if len(asks) >= 2 else None

            # PM book
            pm_ob = self.pm_data.get_order_book(pm_token_id, timeout=CFG.book_timeout_sec)
            if not pm_ob:
                continue
            pm_bids, pm_asks = self.pm_data.parse_orderbook_summary_levels(pm_ob)
            if not pm_bids:
                continue
            pm_bidsN = take_top_levels(pm_bids, CFG.max_book_levels)
            if not pm_bidsN:
                continue

            # 刷新 pm_limit_price：按 remaining 卖出吃 bids 的 last
            pm_rev, pm_last = _revenue_and_last_price_from_bids(pm_bidsN, remaining)
            if pm_rev is None or pm_last is None:
                continue

            pm_vwap = pm_rev / remaining  # 关键：用深度算出来的 PM 卖出 VWAP

            old_pm = float(od.get("pm_limit_price") or 0.0)
            new_pm = float(pm_last)

            pm_changed = abs(old_pm - new_pm) > 0.0001
            if pm_changed:
                od["pm_limit_price"] = new_pm
                self.active_sell_orders[ev_key] = od
                log.info(f"[SELL][PM_LIM][UPDATE] ev={ev_key} {old_pm:.4f}->{new_pm:.4f}")

                # 只在 PM 变动时：重新计算 OP 限价卖价，使得 sum_bid >= 1.01（+buffer）
                THRESH = D("1.01")
                exec_buffer = D("0.00")  # 你想更保守可改成 0.002 等

                need_op_px = (THRESH + exec_buffer) - pm_vwap  # op_limit 的理论下界
                if need_op_px <= D("0"):
                    # 极端兜底：不可能出现但防一下
                    need_op_px = ask0

                # 卖出 maker：不能 <= bid0，否则会立刻成交变 taker
                new_op_px = self._cap_sell_price_to_stay_maker(px=need_op_px, bid0=bid0)

                # 按 tick=0.01 向下取整（你之前的风格）
                new_op_px = new_op_px.quantize(D("0.001"), rounding=ROUND_UP)

                # 只重挂“剩余份额”
                self._reprice_op_sell_order(
                    ev_key, od,
                    new_price=new_op_px,
                    remaining_shares=remaining,
                )
                log.info(
                    f"[SELL][RECALC_OP] ev={ev_key} "
                    f"pm_vwap={float(pm_vwap):.4f} -> need_op={float(need_op_px):.4f} "
                    f"op_px {float(my_px):.4f}->{float(new_op_px):.4f} "
                    f"(bid0={float(bid0):.4f} ask0={float(ask0):.4f})"
                )

    def _sync_and_hedge_sell_order_fill(self, ev_key: str, od: Dict[str, Any]) -> bool:
        """
        卖出对冲同步：
        - OP get_order_by_id -> op_filled_shares（卖出成交份额）
        - delta = op_filled - hedged_total
        - delta 的名义金额 >= 1.1U 才去 PM 卖出对冲
        - PM 卖出成功 -> 更新 filled_total = op_filled
        """
        if self.risk_mode.get(ev_key):
            return False

        order_id = str(od.get("order_id") or "")
        if not order_id:
            return True

        try:
            res = self.op_trader.get_order_by_id(order_id)
        except Exception as e:
            log.warning(f"[SYNC][SELL] ev={ev_key} get_order_by_id failed oid={order_id}: {e}")
            return True

        info = self._op_extract_order_detail(res)
        if not info:
            return True

        op_filled = _dec(info["filled_shares"])
        side_enum = str(info.get("side_enum") or "")
        status_enum = str(info.get("status_enum") or "")
        if side_enum and side_enum.lower() != "sell":
            # 如果 SDK side_enum 不是 sell，这里就先不动（防串单）
            log.warning(f"[SYNC][SELL] ev={ev_key} oid={order_id} not SELL side={side_enum}")
            return True

        hedged_total = _dec(od.get("filled_total") or 0)
        delta = op_filled - hedged_total
        if delta.copy_abs() < D("0.001"):
            delta = D("0")
        if delta <= D("0"):
            return True

        # 至少 1.1U 才对冲
        HEDGE_MIN_USD = D("1.1")
        op_px = _dec(od.get("price") or 0)
        if op_px <= D("0"):
            op_px = D("1")
        delta_usd = delta * op_px
        if delta_usd < HEDGE_MIN_USD:
            log.info(
                f"[SYNC][SELL][HOLD] ev={ev_key} oid={order_id} "
                f"delta={float(delta):.6f} (~${float(delta_usd):.4f}) < 1.10"
            )
            return True

        pm_token_id = str(od.get("pm_token_id") or "")
        pm_lim = float(od.get("pm_limit_price") or 0.0)
        if not pm_token_id or pm_lim <= 0:
            self.risk_mode[ev_key] = True
            log.error(f"[SYNC][SELL] ev={ev_key} missing pm_token/pm_lim -> risk_mode")
            return False

        # NEW: PM 对冲卖前，按真实 PM token 仓位 cap
        pm_av = self._pm_get_available_shares(token_id=pm_token_id)
        delta_cap = delta if delta <= pm_av else pm_av
        delta_cap = delta_cap.quantize(D("0.0001"), rounding=ROUND_DOWN)  # PM taker 4 decimals
        if delta_cap <= D("0"):
            log.error(
                f"[nonono] ev={ev_key} PM avail too low "
                f"(need={float(delta):.6f}, pm_av={float(pm_av):.6f}) -> risk_mode"
            )
            self.risk_mode[ev_key] = True
            try:
                self._safe_cancel_op(order_id)
            except Exception:
                pass
            return False

        ok_pm = self.pm_trader.safe_sell(
            token_id=pm_token_id,
            shares=float(delta_cap),
            limit_price=float(pm_lim),
            max_retries=3,
            retry_delay=0.2,
        )
        if not ok_pm:
            self.risk_mode[ev_key] = True
            log.error(f"[nonono] PM sell failed ev={ev_key} delta={float(delta):.6f} -> risk_mode")
            try:
                self._safe_cancel_op(order_id)
            except Exception:
                pass
            return False

        # 成功：水位对齐
        od["filled_total"] = float(op_filled)
        od["isNeedCheckReal"] = True
        self.active_sell_orders[ev_key] = od

        # 若满额 -> 清理 active_sell_orders + 清掉 positions
        target = _dec(od.get("target_shares") or 0)
        if status_enum == "Finished":
            self.active_sell_orders.pop(ev_key, None)
            self.positions.pop(ev_key, None)
            log.info(f"[DONE][SELL] ev={ev_key} sold={float(op_filled):.6f}/{float(target):.6f}")
            return False

        return True

    def process_active_sell_orders_hedge(self) -> None:
        for ev_key, od in list(self.active_sell_orders.items()):
            if self.risk_mode.get(ev_key):
                continue
            self._sync_and_hedge_sell_order_fill(ev_key, od)

    def try_execute_refined(
            self,
            ev_key: str,
            op_market: Dict[str, Any],
            *,
            op_side: str,  # "YES" or "NO"
            pm_side: str,  # "YES" or "NO"
            op_token_id: str,
            pm_token_id: str,
            op_bids: List[Tuple[Decimal, Decimal]],
            op_asks: List[Tuple[Decimal, Decimal]],
    ) -> None:
        """
        精筛+执行：
          - 拉 PM 完整 book
          - 计算 plan（①双市价 or ②PM市价+OP限价）
          - 执行 + 分支②填充就对冲
        """
        # 1) PM book
        # 1) PM book (prefer clob batch orderbooks)
        pm_ob = self.pm_data.get_order_book(pm_token_id, timeout=CFG.book_timeout_sec)
        if not pm_ob:
            return

        pm_bids, pm_asks = self.pm_data.parse_orderbook_summary_levels(pm_ob)
        if not pm_asks:
            return

        # 2) top3
        pm_asks3 = take_top_levels(pm_asks, CFG.max_book_levels)
        op_bids3 = take_top_levels(op_bids, CFG.max_book_levels)
        op_asks3 = take_top_levels(op_asks, CFG.max_book_levels)
        if not pm_asks3 or not op_bids3 or not op_asks3:
            return

        # 3) build plan
        plan = build_refined_plan(
            op_side=op_side,
            pm_side=pm_side,
            pm_asks=pm_asks3,
            op_bids=op_bids3,
            op_asks=op_asks3,
            topic_rate=D(str(CFG.topic_rate)),
            min_usd=D(str(CFG.min_order_usd)),
            target_usd=D(str(CFG.target_order_usd)),
            max_usd=D(str(CFG.max_order_usd)),
        )
        if not plan:
            return

        shares = float(plan["shares"])
        if shares <= 0:
            return

        # 4) execute
        if plan["branch"] == "MM":
            return
            op_px = float(plan["op_price"])
            pm_lim = float(plan["pm_limit_price"])
            target_sh = _dec(plan["shares"])

            log.info(
                f"[TRUE][MM] ev={ev_key} shares={float(target_sh):.6f} "
                f"sum={plan['est_sum_cost_per_share']:.6f} "
                f"PM_lim={pm_lim:.4f} OP_mkt≈{op_px:.4f} op_fee={plan['op_total_fee']:.4f}"
            )

            # OP “市价”-> marketable limit（你实现的 safe_place_market_buy_by_shares）
            oid = self.op_trader.safe_place_limit_buy_by_shares(
                market_id=int(op_market["marketId"]),
                token_id=str(op_token_id),
                price=(plan["op_price"] + float(0.001)),
                shares=float(target_sh),
                max_retries=3,
                retry_delay=0.2,
            )
            if not oid:
                return
            log.info(f"op market order ok! order_id={oid}")

            ok_pm = self.pm_trader.safe_buy(
                token_id=str(pm_token_id),
                shares=float(target_sh),
                limit_price=float(pm_lim),
                max_retries=3,
                retry_delay=0.2,
            )
            if not ok_pm:
                log.error(f"[ERROR][MM] ev={ev_key} PM hedge buy failed filled={float(target_sh):.6f} -> risk_mode")
                self.risk_mode[ev_key] = True
                return
            log.info(f"pm market order ok!")

            # ===== 只有“完全成交”才写入 positions =====
            od = {
                "market_id": int(op_market["marketId"]),
                "op_token_id": str(op_token_id),
                "pm_token_id": str(pm_token_id),
                "op_outcome": str(op_side).upper(),
            }

            if self._is_buy_order_fully_filled(op_filled=target_sh, target=target_sh, status_enum=status_enum):
                self._commit_position_full(ev_key, od, filled_shares=target_sh)
                log.info(
                    f"[DONE][MM->POS] ev={ev_key} oid={oid} filled={float(filled_sh):.6f}/{float(target_sh):.6f}"
                )
            else:
                # 这里你的要求是“不满不入仓”；但此时你已经对冲了部分 filled_sh
                # 最稳处理：标记 risk_mode 让你人工介入（否则你后续逻辑里会“看不到仓位”，但链上/账户里实际有持仓）
                log.warning(
                    f"[MM][PARTIAL] ev={ev_key} oid={oid} filled={float(filled_sh):.6f}/{float(target_sh):.6f} "
                    f"-> hedged only, NOT writing positions, enter risk_mode"
                )
                self.risk_mode[ev_key] = True

            return

        if plan["branch"] == "PM_OPLIM":
            my_px = float(plan["op_limit_price"])
            pm_lim = float(plan["pm_limit_price"])

            log.info(
                f"[EXEC][PM+OP_LIM] queue ev={ev_key} shares={shares:.6f} "
                f"sum={plan['est_sum_cost_per_share']:.6f} "
                f"OP={my_px:.4f} PM={pm_lim:.4f}"
            )

            # 只挂 OP maker（limit buy），不在这里轮询对冲
            order_id = self.op_trader.safe_place_limit_buy_by_shares(
                market_id=int(op_market["marketId"]),
                token_id=str(op_token_id),
                price=float(my_px),
                shares=float(shares),
                max_retries=3,
                retry_delay=0.2,
            )
            if not order_id:
                return

            log.info(f"op limit order ok! order_id={order_id}")

            # 记录计划，稍后统一扫 fills 再 PM 对冲
            # 记录计划，稍后统一扫 fills 再 PM 对冲
            self.active_orders[ev_key] = {
                "order_id": order_id,
                "market_id": int(op_market["marketId"]),
                "op_token_id": str(op_token_id),  # ✅ 新增：后面拉 orderbook 用
                "op_outcome": str(op_side).upper(),  # YES/NO
                "pm_token_id": str(pm_token_id),
                "pm_limit_price": float(pm_lim),
                "target_shares": float(shares),
                "filled_total": 0.0,
                "price": float(my_px),  # ✅ 新增：当前挂单价
                "created_ts": now_ts(),
                "last_reprice_ts": 0,  # ✅ 可选：防抖
            }

            return

    def run_once(self):
        log.info(f"risk_mode:{self.risk_mode}")
        # NEW: PM open orders housekeeping (every 8 min)
        self.manage_pm_live_orders()

        if (now_ts() - self.last_meta_refresh) > CFG.meta_refresh_sec or not self.event_map:
            self.refresh_meta()

        if not self.check_balances():
            time.sleep(5)
            return

        now = now_ts()
        do_scan = (now - self.last_scan_ts) >= self.scan_interval_sec
        if do_scan:
            self.last_scan_ts = now

            # 预扫描：收集 token_ids（去重）+ 缓存到本地
            pm_need_tids = set()
            pm_tid_cache = {}  # ev_key -> (pm_yes_tid, pm_no_tid)
            log.info(f"---------------【Pre_Scan】---------------")
            for ev_key, em in list(self.event_map.items()):
                if not em.get("valid"):
                    continue
                if self.risk_mode.get(ev_key):
                    continue

                # 新增：如果已存在该 ev_key，则跳过（避免覆盖/重复下单）
                if (ev_key in self.positions) or (ev_key in self.active_orders) or (ev_key in self.active_sell_orders):
                    continue

                pm = em["sources"]["polymarket"]
                pm_yes_tid, pm_no_tid = normalize_pm_token_ids(pm)
                if not pm_yes_tid or not pm_no_tid:
                    log.info(f"[NONONO] ev={ev_key} "
                             f"not pm_yes_tid or not pm_no_tid")
                    continue

                pm_yes_tid = str(pm_yes_tid)
                pm_no_tid = str(pm_no_tid)

                pm_tid_cache[ev_key] = (pm_yes_tid, pm_no_tid)
                pm_need_tids.add(pm_yes_tid)
                pm_need_tids.add(pm_no_tid)

            # 批量拉价
            pm_buy_price_map = {}
            if pm_need_tids:
                try:
                    pm_buy_price_map = self.pm_data.get_prices(
                        list(pm_need_tids),
                        side="SELL",
                        timeout=CFG.price_timeout_sec,
                        batch_size=200,
                        max_retries=3,
                    )
                except Exception:
                    pm_buy_price_map = {}
            log.info(f"---------------【Scan】---------------")
            start0 = time.time()
            # scan
            for ev_key, em in list(self.event_map.items()):
                if not em.get("valid"):
                    continue
                if self.risk_mode.get(ev_key):
                    continue

                # 新增：如果已存在该 ev_key，则跳过（避免覆盖/重复下单）
                if (ev_key in self.positions) or (ev_key in self.active_orders) or (ev_key in self.active_sell_orders):
                    continue

                pm = em["sources"]["polymarket"]
                op = em["sources"]["opinion"]

                # Opinion token ids：先判断 None，再 str
                yes_tid = op.get("yesTokenId")
                no_tid = op.get("noTokenId")
                if not yes_tid or not no_tid:
                    log.info(f"[NONONO] ev={ev_key} "
                             f"not op_yes_tid or not op_no_tid")
                    continue
                yes_tid = str(yes_tid)
                no_tid = str(no_tid)

                try:
                    start = time.time()
                    ob_yes = self.op_open.get_orderbook(yes_tid, timeout=CFG.book_timeout_sec)
                except Exception:
                    continue

                yes_bids, yes_asks = parse_op_orderbook(ob_yes)
                if not yes_bids or not yes_asks:
                    log.info(f"[NONONO] ev={ev_key} "
                             f"not yes_bids or yes_asks")
                    continue
                no_bids, no_asks = invert_yes_book_to_no(yes_bids, yes_asks)

                op_yes_bid, op_yes_ask = yes_bids[0][0], yes_asks[0][0]
                op_no_bid, op_no_ask = no_bids[0][0], no_asks[0][0]

                # PM token ids：用缓存（避免重复 normalize）
                tids = pm_tid_cache.get(ev_key)
                if not tids:
                    continue
                pm_yes_tid, pm_no_tid = tids

                # 从批量结果取
                pm_yes_ask = pm_buy_price_map.get(pm_yes_tid)
                pm_no_ask = pm_buy_price_map.get(pm_no_tid)

                # 兜底单拉
                if pm_yes_ask is None:
                    pm_yes_ask = self.pm_data.get_price(pm_yes_tid, "SELL", timeout=CFG.price_timeout_sec)
                if pm_no_ask is None:
                    pm_no_ask = self.pm_data.get_price(pm_no_tid, "SELL", timeout=CFG.price_timeout_sec)

                if pm_yes_ask is None or pm_no_ask is None:
                    log.info(f"[NONONO] ev={ev_key} "
                             f"not pm_yes_ask or pm_no_ask")
                    continue

                best = find_buy_opportunity(
                    buffer=CFG.buffer,
                    pm_yes_ask=float(pm_yes_ask),
                    pm_no_ask=float(pm_no_ask),
                    op_yes_bid=op_yes_bid, op_yes_ask=op_yes_ask,
                    op_no_bid=op_no_bid, op_no_ask=op_no_ask,
                )
                if not best:
                    continue

                if best["op_side"] == "YES":
                    best["op_token_id"] = yes_tid
                    best["pm_token_id"] = pm_no_tid
                else:
                    best["op_token_id"] = no_tid
                    best["pm_token_id"] = pm_yes_tid

                pm_show = float(pm_yes_ask) if best["pm_side"] == "YES" else float(pm_no_ask)
                log.info(
                    f"[SIGNAL] ev={ev_key} edge={best['est_edge']:.4f} "
                    f"OP[{best['op_side']}]:{best['op_limit_price']:.4f} "
                    f"PM[{best['pm_side']}]:{pm_show:.4f}"
                )

                # ===== refined execution =====
                if best["op_side"] == "YES":
                    ob_bids, ob_asks = yes_bids, yes_asks
                else:
                    ob_bids, ob_asks = no_bids, no_asks

                self.try_execute_refined(
                    ev_key=ev_key,
                    op_market=op,
                    op_side=str(best["op_side"]),
                    pm_side=str(best["pm_side"]),
                    op_token_id=str(best["op_token_id"]),
                    pm_token_id=str(best["pm_token_id"]),
                    op_bids=ob_bids,
                    op_asks=ob_asks,
                )
        else:
            log.info("[SCAN] skipped (cooldown)")
            start0 = time.time()

        log.info(f"---------------【step 1】---------------↑↑↑ {(time.time() - start0)} s")
        start1 = time.time()
        # 第二阶段前：重新评估 active_orders（无套利则撤，符合条件则改价）
        self.manage_active_orders_before_hedge()

        log.info(f"---------------【step 2】---------------↑↑↑ {(time.time() - start1)} s")
        start2 = time.time()
        # 第二阶段：扫 OP 限价单 fills -> PM 对冲
        self.process_active_orders_hedge()

        log.info(f"---------------【step 3】---------------↑↑↑ {(time.time() - start2)} s")
        start3 = time.time()
        # ===== NEW: 触发卖出逻辑（有仓位才会动）=====
        self.try_execute_sell_for_positions()

        log.info(f"---------------【step 4】---------------↑↑↑ {(time.time() - start3)} s")
        start4 = time.time()
        # 卖出：挂单管理 + 对冲同步
        self.manage_active_sell_orders_before_hedge()

        #log.info(f"---------------【step 5】---------------↑↑↑ {(time.time() - start4)} s")
        #start5 = time.time()
        #self.process_active_sell_orders_hedge()
        log.info(f"---------------【end】---------------↑↑↑ {(time.time() - start4)} s")

        if CFG.loop_sleep_sec > 0:
            time.sleep(CFG.loop_sleep_sec)

    def _cancel_all_op_orders_on_exit(self) -> None:
        """
        退出前：尽最大努力撤掉 OP 所有未成交挂单。
        - 优先用 OP 提供的 cancel_all_orders（全账户 open orders）
        - 若失败，再兜底撤本地追踪到的 active_orders / active_sell_orders
        """
        if not getattr(self, "op_trader", None) or not getattr(self.op_trader, "enabled", False):
            log.info("[EXIT] OP trading disabled or no op_trader; skip cancel_all.")
            return

        # 1) 全量撤单（推荐）
        try:
            res = self.op_trader.client.cancel_all_orders()  # 你贴的接口在 client 上
            # res 形如: {'total_orders':..., 'cancelled':..., 'failed':..., 'results':...}
            log.info(
                f"[EXIT][OP][CANCEL_ALL] total={res.get('total_orders')} "
                f"cancelled={res.get('cancelled')} failed={res.get('failed')}"
            )
            return
        except Exception as e:
            log.exception(f"[EXIT][OP][CANCEL_ALL] failed: {e}")

        # 2) 兜底：撤本地记录的限价单
        try:
            oids = []
            for od in (self.active_orders or {}).values():
                oid = str(od.get("order_id") or "")
                if oid:
                    oids.append(oid)
            for od in (self.active_sell_orders or {}).values():
                oid = str(od.get("order_id") or "")
                if oid:
                    oids.append(oid)

            seen = set()
            uniq = []
            for x in oids:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)

            if not uniq:
                log.info("[EXIT][OP][FALLBACK] no local orders to cancel.")
                return

            ok = 0
            for oid in uniq:
                try:
                    self._safe_cancel_op(oid)
                    ok += 1
                except Exception as e:
                    log.warning(f"[EXIT][OP][FALLBACK] cancel failed oid={oid} err={e}")

            log.info(f"[EXIT][OP][FALLBACK] canceled {ok}/{len(uniq)} local orders.")
        finally:
            try:
                self.active_orders.clear()
                self.active_sell_orders.clear()
            except Exception:
                pass

    def get_all_positions(self, user_addr: str, limit: int = 50):
        out = []
        offset = 0
        while True:
            r = requests.get(
                f"{BASE}/positions",
                params={
                    "user": user_addr,  # 有的文档示例叫 user/address，按文档参数名改一下即可
                    "limit": limit,
                    "offset": offset,
                    # 你也可以加：market=..., event=..., redeemable=..., mergeable=... 等
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            # 常见结构：{"data":[...], ...} 或直接 [...]
            items = data["data"] if isinstance(data, dict) and "data" in data else data
            out.extend(items)
            if len(items) < limit:
                break
            offset += limit
        return out

    def run(self):
        log.info(f"START | DRY_RUN={CFG.dry_run}")
        '''
        pm_clob.safe_sell(
            token_id=str("71478852790279095447182996049071040792010759617668969799049179229104800573786"),
            shares=float(1.5),
            limit_price=float(0.996),
            max_retries=3,
            retry_delay=0.2,
        )
        order_id = op_clob.safe_place_limit_buy_by_shares(
            market_id=int(1226),
            token_id=str(112796040680499184493768367425897772037344047715371428672472988730438104274752),
            price=float(0.99),
            shares=float(20),
            max_retries=3,
            retry_delay=0.2,
        )
        self._safe_cancel_op(order_id)

        trade = op_clob.get_my_trades(market_id=1229, page=1, limit=20)
        print(trade)
        order = op_clob.client.get_order_by_id("670243ce-e7ed-11f0-ae12-0a58a9feac02")
        print(order)

        info = self._op_extract_order_detail(order)
        print(info)
        if not info:
            return True

        pos = op_clob.client.get_my_positions()
        print(pos)

        params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_id="98527953404683968927351392535846134688575286140333769930915553620463004327686",
            signature_type=2)

        pm_pos = pm_clob.client.get_balance_allowance(params)
        print(pm_pos)

        self.get_all_positions(user_addr=CFG.poly_funder, limit=50)

        oid_yes = op_clob.safe_place_limit_buy_by_shares(
            market_id=int(1930),
            token_id=str(37032289872632189435235258956207035129250807483211860086565419451874754754057),
            price=float(0.444),
            shares=float(10),
            max_retries=3,
            retry_delay=0.2,
        )
        if not oid_yes:
            print("not ok")
            return
        else:
            print("ok")

        while 1:
            order = op_clob.get_order_by_id(oid_yes)
            log.info(order)
            time.sleep(1)
        
        res = pm_clob.client.get_orders()
        print(res)
        '''

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                log.info("[LOOP] KeyboardInterrupt -> cancel all OP open orders then exit.")
                try:
                    log.info(f"risk_mode:{self.risk_mode}")
                    self._cancel_all_op_orders_on_exit()
                except Exception:
                    log.exception("[EXIT] cancel on exit failed")
                log.info("bye.")
                return
            except Exception as e:
                log.exception(f"[LOOP] exception: {e}")
                time.sleep(1)


if __name__ == "__main__":
    # op_clob = OpinionTrader(CFG)
    # pm_clob = PolymarketTrader(CFG)
    Bot().run()