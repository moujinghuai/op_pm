# clients.py
import logging
import time
import concurrent.futures
from decimal import Decimal as D, Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple, Iterable

from torch._dynamo.polyfills import os

from arb import D0
from config import Config
from retry import with_retry
from net import HTTP

log = logging.getLogger(__name__)

# optional SDKs
_HAS_OPINION_SDK = False
try:
    from opinion_clob_sdk import Client as OpinionClient
    from opinion_clob_sdk.chain.py_order_utils.model.order import PlaceOrderDataInput
    from opinion_clob_sdk.chain.py_order_utils.model.sides import OrderSide
    from opinion_clob_sdk.chain.py_order_utils.model.order_type import LIMIT_ORDER
    _HAS_OPINION_SDK = True
except Exception:
    _HAS_OPINION_SDK = False

_HAS_POLY_CLOB = False
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, MarketOrderArgs, BookParams
    from py_clob_client.order_builder.constants import BUY, SELL
    _HAS_POLY_CLOB = True
except Exception:
    _HAS_POLY_CLOB = False

class FuturesTimeoutError(Exception):
    pass

PRICE_STEP = Decimal("0.001")
SHARE_STEP = Decimal("0.000001")
# 限价单向内让价的幅度（0.002 ≈ 0.2c）
MAX_SLIPPAGE = Decimal("0.005")

PM_TAKER_SIZE_STEP = Decimal("0.0001")  # 4 decimals
PM_MAKER_SIZE_STEP = Decimal("0.01")    # 2 decimals（如果你以后挂 GTC）

def _q_pm_taker_size(x: float) -> float:
    d = Decimal(str(x)).quantize(PM_TAKER_SIZE_STEP, rounding=ROUND_DOWN)
    return float(d)

def _fmt_decimal(x: Decimal) -> str:
    # 避免科学计数法
    return format(x, "f")

def _q_price(p: Decimal) -> Decimal:
    # Opinion 2 digits allowed
    return p.quantize(PRICE_STEP, rounding=ROUND_DOWN)

def _q_share(s: Decimal) -> Decimal:
    return s.quantize(SHARE_STEP, rounding=ROUND_DOWN)

def _call_with_timeout(fn, timeout_s: float, **kwargs):
    """
    用线程池给“同步阻塞函数”加超时。
    注意：超时后线程无法强制杀死，但至少主逻辑不会卡死，会进入拆批/兜底。
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as e:
            raise FuturesTimeoutError(f"timeout after {timeout_s}s") from e

def _extract_payload(res: Any) -> Any:
    # Opinion SDK 常见：res.result / res.data；也可能直接是 dict
    return getattr(res, "result", None) or getattr(res, "data", None) or res

def _extract_order_id(payload):
    # 1) dict 情况
    if isinstance(payload, dict):
        for k in ("orderId", "order_id", "id", "trans_no"):
            if k in payload and payload[k]:
                return str(payload[k])
        # 向下挖一层
        for k in ("order_data", "data", "result"):
            inner = payload.get(k)
            if inner is not None:
                oid = _extract_order_id(inner)
                if oid:
                    return oid

    # 2) object / pydantic 情况
    for name in ("orderId", "order_id", "id", "trans_no"):
        v = getattr(payload, name, None)
        if v:
            return str(v)

    # 3) SDK v2: result.order_data.order_id
    inner = getattr(payload, "order_data", None)
    if inner is not None:
        for name in ("order_id", "trans_no", "id"):
            v = getattr(inner, name, None)
            if v:
                return str(v)

    return None


def _extract_error_msg(payload: Any) -> Optional[str]:
    # dict errors
    if isinstance(payload, dict):
        # 常见错误字段
        for k in ("message", "error", "err", "reason", "detail"):
            v = payload.get(k)
            if v:
                return str(v)
        # success=false 的情况
        if payload.get("success") is False:
            return "success=false"
    # object errors
    for name in ("message", "error", "reason", "detail"):
        v = getattr(payload, name, None)
        if v:
            return str(v)
    return None


def _is_success_res(res: Any) -> bool:
    # SDK 返回通常在 res.errno / res.errmsg
    errno = getattr(res, "errno", None)
    errmsg = getattr(res, "errmsg", None)

    # errno==0 且 errmsg 为空/None -> 成功
    if errno is not None:
        return int(errno) == 0 and (errmsg in (None, ""))
    # 没 errno 字段时，无法强判断：保守返回 False，让上层依赖 oid 或 payload
    return False


def _extract_order_id_from_res(res: Any) -> Optional[str]:
    payload = _extract_payload(res)
    oid = _extract_order_id(payload)
    # 有些版本 oid 在 res 自己（不是 payload）
    if not oid:
        oid = _extract_order_id(res)
    return oid


class OpinionOpenAPI:
    def __init__(self, api_key: str, base: str):
        self.api_key = api_key
        self.base = base.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"apikey": self.api_key} if self.api_key else {}

    def list_markets(
            self,
            market_type: int = 2,
            page: int = 1,
            limit: int = 20,
            status: str = "activated",
            timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        url = f"{self.base}/market"
        params = {
            "marketType": market_type,
            "page": page,
            "limit": limit,
            "status": status,
        }
        j = HTTP.get(url, headers=self._headers(), params=params, timeout=timeout)
        # 1️⃣ 如果直接返回数组（极少数接口）
        if isinstance(j, list):
            return j
        if not isinstance(j, dict):
            return []
        # 2️⃣ 先定位“容器层”
        container = None
        if isinstance(j.get("result"), dict):
            container = j["result"]
        elif isinstance(j.get("data"), dict):
            container = j["data"]
        else:
            container = j
        # 3️⃣ 再从容器里找市场列表
        for key in ("list", "items", "markets", "records"):
            v = container.get(key)
            if isinstance(v, list):
                return v
        return []

    def get_orderbook(self, token_id: str, timeout: float = 5.0) -> Dict[str, Any]:
        url = f"{self.base}/token/orderbook"
        params = {"token_id": token_id}
        return HTTP.get(url, headers=self._headers(), params=params, timeout=timeout)


class OpinionTrader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = (not cfg.dry_run) and _HAS_OPINION_SDK and bool(cfg.opinion_private_key) and bool(cfg.opinion_api_key)
        self.client = None
        if self.enabled:
            self.client = OpinionClient(
                host=cfg.opinion_clob_host,
                apikey=cfg.opinion_api_key,
                chain_id=cfg.opinion_chain_id,
                rpc_url=cfg.opinion_rpc_url or "",  # 没用链上就保持空
                private_key=cfg.opinion_private_key or ("0x" + "0" * 64),
                conditional_tokens_addr=cfg.opinion_conditional_token,
                multi_sig_addr=cfg.opinion_multi_sig_address or ("0x" + "0" * 40),
            )

            log.info("[Opinion] trading enabled.")
        else:
            log.info("[Opinion] trading disabled (DRY_RUN or missing deps/keys).")

    def get_my_balances_usd(self) -> float:
        if not self.enabled:
            return 999999.0

        res = self.client.get_my_balances()

        # Opinion SDK 返回：res.result.balances
        payload = getattr(res, "result", None)
        if not payload:
            log.warning("[BAL] no result in get_my_balances()")
            return 0.0

        balances = getattr(payload, "balances", None)
        if not balances:
            log.warning("[BAL] result has no balances")
            return 0.0

        total = 0.0
        for b in balances:
            try:
                total += float(b.available_balance)
            except Exception:
                pass

        return total

    def safe_place_market_buy_by_shares(
            self, *, market_id: int, token_id: str, shares: Decimal,
            best_ask: Decimal,
            max_retries: int = 3, retry_delay: float = 0.2
    ) -> Optional[str]:
        """
        NOTE:
        Opinion MARKET_ORDER 不支持 makerAmountInBaseToken（shares）。
        所以这里“市价买”语义实现为：marketable limit（aggressive limit）。
        """
        out: Dict[str, Optional[str]] = {"oid": None}

        def _do():
            oid = self.place_aggressive_buy_by_shares(
                market_id=market_id, token_id=token_id, shares=shares, best_ask=best_ask
            )
            out["oid"] = oid
            return bool(oid)

        ok = with_retry(
            _do,
            what=f"[OP] AGGR_BUY market={market_id} token={str(token_id)[:12]} shares={shares} ask={best_ask}",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        return out["oid"] if ok else None

    def place_limit_buy_by_shares(
            self,
            *,
            market_id: int,
            token_id: str,
            price: Decimal,
            shares: Decimal,
    ) -> str:
        """
        Opinion 限价买：失败抛异常；成功返回 orderId（str）
        """
        if shares <= D0 or price <= D0:
            raise ValueError(f"invalid args price={price} shares={shares}")

        price = _q_price(price)
        shares = _q_share(shares)

        if not self.enabled:
            oid = f"DRY_OP_LIM_{market_id}_{token_id}"
            log.info(
                f"[Opinion][DRY] LIMIT BUY market={market_id} token={token_id} "
                f"price={_fmt_decimal(price)} shares={_fmt_decimal(shares)}"
            )
            return oid

        od = PlaceOrderDataInput(
            marketId=int(market_id),
            tokenId=str(token_id),
            side=OrderSide.BUY,
            orderType=LIMIT_ORDER,
            price=_fmt_decimal(price),
            makerAmountInBaseToken=_fmt_decimal(shares),
        )

        res = self.client.place_order(od)
        payload = _extract_payload(res)

        oid = _extract_order_id(payload)
        if oid:
            return oid

        # 关键：把失败原因抛出去，让 retry 日志能看到
        err = _extract_error_msg(payload) or "no orderId in response"
        raise RuntimeError(f"Opinion limit buy failed: {err} | payload={payload}")

    def safe_place_limit_buy_by_shares(
            self,
            *,
            market_id: int,
            token_id: str,
            price: float,
            shares: float,
            max_retries: int = 3,
            retry_delay: float = 0.2,
    ) -> Optional[str]:
        out: Dict[str, Optional[str]] = {"oid": None}

        px = Decimal(str(price))
        sh = Decimal(str(shares))

        def _do():
            oid = self.place_limit_buy_by_shares(
                market_id=market_id,
                token_id=token_id,
                price=px,
                shares=sh,
            )
            out["oid"] = oid
            return bool(oid)  # 成功就 True（失败会在 place_* 里抛异常）

        ok = with_retry(
            _do,
            what=f"[OP] LIM_BUY market={market_id} token={str(token_id)[:12]} "
                 f"px={float(px):.6f} sh={float(sh):.6f}",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        return out["oid"] if ok else None

    def place_limit_sell_by_shares(self, market_id: int, token_id: str, price: Decimal, shares: Decimal) -> Optional[str]:
        """
        限价卖：orderType = LIMIT_ORDER
        - 用 makerAmountInBaseToken 直接下份数
        """
        if shares <= D0 or price <= D0:
            return None

        shares = _q_share(shares)
        price = _q_price(price)

        if not self.enabled:
            log.info(
                f"[Opinion][DRY] LIMIT SELL market={market_id} token={token_id} "
                f"price={_fmt_decimal(price)} shares={_fmt_decimal(shares)}"
            )
            return f"DRY_OP_LIM_SELL_{market_id}_{token_id}"

        od = PlaceOrderDataInput(
            marketId=int(market_id),
            tokenId=str(token_id),
            side=OrderSide.SELL,
            orderType=LIMIT_ORDER,
            price=_fmt_decimal(price),
            makerAmountInBaseToken=_fmt_decimal(shares),
        )
        res = self.client.place_order(od)
        payload = _extract_payload(res)

        # 1) 成功先判 errno/errmsg
        if _is_success_res(res):
            oid = _extract_order_id_from_res(res)
            if oid:
                return oid
            # ✅ 成功但 oid 解析不到：不要让上层重试下单
            log.warning(f"[Opinion] place_order success but cannot parse oid. payload={payload}")
            return f"OP_OK_NO_OID_{int(time.time())}_{market_id}_{token_id}"

        # 2) 失败：抛出让 with_retry 去重试（这时重试才合理）
        err = _extract_error_msg(
            payload) or f"errno={getattr(res, 'errno', None)} errmsg={getattr(res, 'errmsg', None)}"
        raise RuntimeError(f"Opinion limit sell failed: {err} | payload={payload}")

    def safe_place_limit_sell_by_shares(
            self, *, market_id: int, token_id: str, price: float, shares: float,
            max_retries: int = 3, retry_delay: float = 0.2
    ) -> Optional[str]:
        out: Dict[str, Optional[str]] = {"oid": None}

        px = Decimal(str(price))
        sh = Decimal(str(shares))

        def _do():
            oid = self.place_limit_sell_by_shares(
                market_id=market_id, token_id=token_id, price=px, shares=sh
            )
            out["oid"] = oid
            return bool(oid)

        ok = with_retry(
            _do,
            what=f"[OP] LIM_SELL market={market_id} token={str(token_id)[:12]} px={float(px):.6f} sh={float(sh):.6f}",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        return out["oid"] if ok else None

    def place_market_buy_by_shares(self, market_id: int, token_id: str, shares: Decimal, *,
                                   cap_price: Decimal | None = None) -> Optional[str]:
        """
        真·市价买：orderType = MARKET_ORDER
        - 用 makerAmountInBaseToken 直接下份数
        - cap_price 可选：用于你自己做保护（有的平台 market 也允许带 price，当作保护价；不确定就传 None）
        """
        if shares <= D0:
            return None

        shares = _q_share(shares)

        if not self.enabled:
            log.info(
                f"[Opinion][DRY] MARKET BUY market={market_id} token={token_id} shares={_fmt_decimal(shares)} cap={cap_price}")
            return f"DRY_OP_MKT_{market_id}_{token_id}"

        od = PlaceOrderDataInput(
            marketId=int(market_id),
            tokenId=str(token_id),
            side=OrderSide.BUY,
            orderType=1,  # ✅ 真市价
            makerAmountInBaseToken=_fmt_decimal(shares),  # ✅ 按份数
        )

        # 如果你确认 Opinion 允许 market 也带 price 作为保护（有些撮合允许），再打开：
        if cap_price is not None and cap_price > D0:
            od.price = _fmt_decimal(_q_price(cap_price))

        res = self.client.place_order(od)
        payload = _extract_payload(res)

        if _is_success_res(res):
            oid = _extract_order_id_from_res(res)
            if oid:
                return oid
            log.warning(f"[Opinion] market buy success but cannot parse oid. payload={payload}")
            return f"OP_OK_NO_OID_{int(time.time())}_{market_id}_{token_id}"

        err = _extract_error_msg(
            payload) or f"errno={getattr(res, 'errno', None)} errmsg={getattr(res, 'errmsg', None)}"
        raise RuntimeError(f"Opinion market buy failed: {err} | payload={payload}")

    def place_limit_buy_by_shares_base(self, market_id: int, token_id: str, price: Decimal, shares: Decimal) -> \
    Optional[str]:
        """
        限价买（maker 或 taker 都可能）：orderType = LIMIT_ORDER
        - 用 makerAmountInBaseToken 直接下份数
        - price 按 0.01 精度
        """
        if shares <= D0 or price <= D0:
            return None

        shares = _q_share(shares)
        price = _q_price(price)

        if not self.enabled:
            log.info(
                f"[Opinion][DRY] LIMIT BUY market={market_id} token={token_id} price={_fmt_decimal(price)} shares={_fmt_decimal(shares)}")
            return f"DRY_OP_LIM_{market_id}_{token_id}"

        od = PlaceOrderDataInput(
            marketId=int(market_id),
            tokenId=str(token_id),
            side=OrderSide.BUY,
            orderType=LIMIT_ORDER,
            price=_fmt_decimal(price),
            makerAmountInBaseToken=_fmt_decimal(shares),  # ✅ 按份数
        )
        res = self.client.place_order(od)
        payload = _extract_payload(res)

        if _is_success_res(res):
            oid = _extract_order_id_from_res(res)
            if oid:
                return oid
            log.warning(f"[Opinion] limit buy(base) success but cannot parse oid. payload={payload}")
            return f"OP_OK_NO_OID_{int(time.time())}_{market_id}_{token_id}"

        err = _extract_error_msg(
            payload) or f"errno={getattr(res, 'errno', None)} errmsg={getattr(res, 'errmsg', None)}"
        raise RuntimeError(f"Opinion limit buy(base) failed: {err} | payload={payload}")

    def safe_place_market_sell_by_shares(
            self,
            *,
            market_id: int,
            token_id: str,
            shares: D,
            best_bid: D,
            max_retries: int = 3,
            retry_delay: float = 0.2,
    ) -> str:
        """
        市价卖（包装为 marketable LIMIT SELL）：
        - 使用 best_bid 作为可成交价格，保证立即吃掉 bid
        - shares 用 Decimal
        - 返回 order_id（失败返回 ""）
        """
        if shares is None or shares <= D("0"):
            return ""

        # Opinion tick 通常 0.01
        px = D(str(best_bid)).quantize(D("0.01"), rounding=ROUND_DOWN)
        if px <= D("0"):
            return ""

        return self.safe_place_limit_sell_by_shares(
            market_id=market_id,
            token_id=str(token_id),
            price=float(px),
            shares=float(shares),
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def place_aggressive_buy_by_shares(
            self, market_id: int, token_id: str, shares: Decimal, best_ask: Decimal
    ) -> Optional[str]:
        """
        “真·市价按 shares 买”的正确替代：marketable limit
        - 直接用 >= best_ask 的限价吃单（taker）
        - best_ask 必须由上层从 orderbook 传入（op_asks[0][0]）
        """
        if shares <= D0:
            return None
        if best_ask is None or best_ask <= D0:
            return None

        shares = _q_share(shares)

        # 穿一跳，避免刚好等于 best_ask 时盘口移动导致挂单不成交
        px = best_ask + PRICE_STEP
        px = _q_price(px)

        # ✅ 关键：place_limit_buy_by_shares 是 keyword-only（带 *），必须用关键字传参
        return self.place_limit_buy_by_shares(
            market_id=market_id,
            token_id=token_id,
            price=px,
            shares=shares,
        )

    def cancel_order(self, order_id: str) -> bool:
        if not self.enabled:
            log.info(f"[Opinion][DRY] cancel {order_id}")
            return True

        res = self.client.cancel_order(order_id=order_id)

        # 1️⃣ 直接取 SDK 的 result 对象
        result_obj = getattr(res, "result", None)

        # SDK 返回：OpenapiCancelOrderRespOpenAPI(result=True)
        if result_obj is not None:
            # 优先读 .result 字段
            ok = getattr(result_obj, "result", None)
            if ok is not None:
                return bool(ok)

        # 2️⃣ 兜底：dict 结构（防 SDK 版本变动）
        if isinstance(result_obj, dict):
            for k in ("result", "success", "ok"):
                if k in result_obj:
                    return bool(result_obj[k])

        # 3️⃣ 再兜底：data
        data = getattr(res, "data", None)
        if isinstance(data, dict):
            for k in ("result", "success", "ok"):
                if k in data:
                    return bool(data[k])

        log.warning(f"[Opinion] cancel_order unknown response: {res}")
        return False


    def get_order_by_id(self, order_id: str):
        """
        读订单详情（真相源）：filled_shares / status / side / outcome 等
        返回 SDK 原始对象（OpenapiOrderDetailRespOpenAPI）
        """
        if not self.enabled:
            return None
        return self.client.get_order_by_id(order_id)


    def get_my_trades(self, market_id: int, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        page = max(1, int(page))
        limit = max(1, min(int(limit), 20))  # SDK max 20

        res = self.client.get_my_trades(market_id=int(market_id), page=page, limit=limit)

        # 你的实际返回：errmsg/errno/result=OpenapiTradeListRespOpenAPI(list=[...], total=..)
        obj = getattr(res, "result", None) or getattr(res, "data", None) or res

        # 1) obj 可能直接就是 list
        if isinstance(obj, list):
            return obj

        # 2) obj 可能是 dict
        if isinstance(obj, dict):
            return obj.get("list") or obj.get("items") or []

        # 3) obj 是 SDK 对象：优先取 .list
        lst = getattr(obj, "list", None)
        if lst is None:
            return []

        # list 里的元素也是 SDK 对象：转成 dict（尽量）
        out: List[Dict[str, Any]] = []
        for x in lst:
            if isinstance(x, dict):
                out.append(x)
            elif hasattr(x, "model_dump"):  # pydantic v2
                out.append(x.model_dump())
            elif hasattr(x, "dict"):  # pydantic v1
                out.append(x.dict())
            elif hasattr(x, "__dict__"):  # 普通对象兜底
                out.append(dict(x.__dict__))
            else:
                out.append({"raw": str(x)})
        return out


class PolymarketData:
    def __init__(self, cfg: Config, clob_client: Optional["ClobClient"] = None):
        self.cfg = cfg
        self._clob = clob_client  # 复用 trader 的 client（可能为 None）
        if self._clob is not None:
            log.info("[PM][DATA] using shared ClobClient (batch enabled).")
        else:
            log.info("[PM][DATA] no ClobClient provided, fallback to /price loop.")

    def list_markets(self, offset: int, timeout: float) -> List[Dict[str, Any]]:
        url = f"{self.cfg.poly_gamma_base}/markets"
        params = {
            "limit": self.cfg.pm_page_size,
            "offset": offset,
            "closed": "false",
            "active": "true",
            "archived": "false",
        }
        return HTTP.get(url, params=params, timeout=timeout)

    def get_price(self, token_id: str, side: str, timeout: float) -> Optional[float]:
        url = f"{self.cfg.poly_clob_host}/price"
        params = {"token_id": token_id, "side": side}
        try:
            j = HTTP.get(url, params=params, timeout=timeout)
            p = j.get("price")
            return float(p) if p is not None else None
        except Exception:
            return None

    def get_prices(
        self,
        token_ids: List[str],
        side: str,                 # "BUY" or "SELL"
        timeout: float,
        batch_size: int = 100,
        max_retries: int = 3,
        enable_fallback_single: bool = True,  # 兜底单拉（建议开）
    ) -> Dict[str, float]:
        """
        真 batch（优先）：shared ClobClient.get_prices(params=[BookParams...])
          - 超时控制：_call_with_timeout
          - 失败处理：重试 + 二分拆批
        无 clob 时：fallback /price 循环
        返回：{ token_id: price(float) } 只返回指定 side 的价格
        """
        if not token_ids:
            return {}

        uniq = list(dict.fromkeys([str(t) for t in token_ids]))
        side_u = str(side).upper()

        # 1) 无 clob -> /price 循环
        if self._clob is None:
            out: Dict[str, float] = {}
            for tid in uniq:
                p = self.get_price(tid, side_u, timeout=timeout)
                if p is not None:
                    out[tid] = float(p)
            return out

        # 2) 有 clob -> 真 batch
        out: Dict[str, float] = {}
        for start in range(0, len(uniq), batch_size):
            batch = uniq[start:start + batch_size]
            out.update(self._fetch_prices_batch_only(
                token_batch=batch,
                side=side_u,
                timeout=timeout,
                max_retries=max_retries,
            ))

        # 3) 可选：对 batch 缺失项兜底单拉（强烈建议，避免偶发缺档）
        if enable_fallback_single:
            miss = [tid for tid in uniq if tid not in out]
            if miss:
                for tid in miss:
                    p = self.get_price(tid, side_u, timeout=timeout)
                    if p is not None:
                        out[tid] = float(p)

        return out

    def _fetch_prices_batch_only(
        self,
        token_batch: List[str],
        side: str,
        timeout: float,
        max_retries: int,
    ) -> Dict[str, float]:
        """
        只用 batch：self._clob.get_prices
        超时/失败：重试 max_retries；仍失败二分拆批。
        """
        if not token_batch:
            return {}

        # 统一 side 到 SDK 常量（避免 BookParams.side 兼容坑）
        side_norm = BUY if str(side).upper() == "BUY" else SELL
        params = [BookParams(token_id=tid, side=side_norm) for tid in token_batch]

        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                prices = _call_with_timeout(self._clob.get_prices, timeout_s=timeout, params=params)
                elapsed_ms = (time.time() - t0) * 1000.0

                if not isinstance(prices, dict):
                    raise TypeError(f"get_prices returned {type(prices)}, expect dict")

                out: Dict[str, float] = {}
                side_u = str(side).upper()
                side_l = str(side).lower()

                for tid in token_batch:
                    info = prices.get(tid)
                    if not isinstance(info, dict):
                        continue
                    p = info.get(side_u) or info.get(side_l)
                    if p is None:
                        continue
                    try:
                        out[tid] = float(p)
                    except Exception:
                        continue

                # 你想要可观察性可以开这行
                # log.info(f"[PM][DATA][BATCH] ok n={len(token_batch)} attempt={attempt}/{max_retries} {elapsed_ms:.1f}ms")

                return out

            except FuturesTimeoutError as e:
                last_err = e
                log.warning(f"[PM][DATA][BATCH][TIMEOUT] n={len(token_batch)} attempt={attempt}/{max_retries} timeout={timeout}s")
            except Exception as e:
                last_err = e
                log.debug(f"[PM][DATA][BATCH][FAIL] n={len(token_batch)} attempt={attempt}/{max_retries} err={e}")

            time.sleep(0.05)

        # 重试失败 -> 二分拆批
        if len(token_batch) == 1:
            tid = token_batch[0]
            log.warning(f"[PM][DATA][BATCH][GIVEUP] token={tid} last_err={last_err}")
            return {}

        mid = len(token_batch) // 2
        left = token_batch[:mid]
        right = token_batch[mid:]
        log.warning(f"[PM][DATA][BATCH][SPLIT] n={len(token_batch)} -> {len(left)}+{len(right)}")

        out: Dict[str, float] = {}
        out.update(self._fetch_prices_batch_only(left, side, timeout, max_retries))
        out.update(self._fetch_prices_batch_only(right, side, timeout, max_retries))
        return out

    def get_order_book(
            self,
            token_id: str,
            timeout: float,
    ) -> Optional[Any]:
        """
        单 token 取 orderbook：
        - 优先 ClobClient.get_order_book(token_id) -> OrderBookSummary
        - 没有 clob 或失败 -> fallback /book dict
        """
        tid = str(token_id)

        if self._clob is not None:
            try:
                return _call_with_timeout(
                    self._clob.get_order_book,
                    timeout_s=timeout,
                    token_id=tid,
                )
            except Exception as e:
                log.warning(f"[PM][DATA][BOOK] clob.get_order_book failed token={tid} err={e}")

        # fallback: raw /book
        return self.get_book(tid, timeout=timeout)

    def get_order_books(
            self,
            token_ids: List[str],
            timeout: float,
            batch_size: int = 50,
            max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        返回 {token_id: OrderBookSummary 或 dict(book)}。
        优先：ClobClient.get_order_books (batch)
        兜底：/book 单拉
        """
        uniq = list(dict.fromkeys([str(t) for t in (token_ids or []) if t]))
        if not uniq:
            return {}

        # 1) 无 clob -> /book 单拉
        if self._clob is None:
            out = {}
            for tid in uniq:
                ob = self.get_book(tid, timeout=timeout)
                if ob:
                    out[tid] = ob
            return out

        # 2) 有 clob -> batch
        out: Dict[str, Any] = {}

        for start in range(0, len(uniq), batch_size):
            batch = uniq[start:start + batch_size]
            out.update(self._fetch_books_batch_only(
                token_batch=batch,
                timeout=timeout,
                max_retries=max_retries,
            ))

        # 3) 缺失兜底
        miss = [tid for tid in uniq if tid not in out]
        for tid in miss:
            ob = self.get_book(tid, timeout=timeout)
            if ob:
                out[tid] = ob
        return out

    def _fetch_books_batch_only(
            self,
            token_batch: List[str],
            timeout: float,
            max_retries: int,
    ) -> Dict[str, Any]:
        if not token_batch:
            return {}

        params = [BookParams(token_id=tid) for tid in token_batch]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                obs = _call_with_timeout(self._clob.get_order_books, timeout_s=timeout, params=params)
                # SDK 返回 list[OrderBookSummary]，顺序一般跟 token_batch 对齐，但更稳妥是读 token_id 字段
                out: Dict[str, Any] = {}
                if isinstance(obs, list):
                    for ob in obs:
                        try:
                            tid = str(getattr(ob, "token_id", None) or getattr(ob, "tokenId", None) or "")
                            if tid:
                                out[tid] = ob
                        except Exception:
                            pass

                return out
            except Exception as e:
                last_err = e
                time.sleep(0.05)

        # 重试失败 -> 二分拆批
        if len(token_batch) == 1:
            log.warning(f"[PM][DATA][BOOKS][GIVEUP] token={token_batch[0]} last_err={last_err}")
            return {}

        mid = len(token_batch) // 2
        left = token_batch[:mid]
        right = token_batch[mid:]
        log.warning(f"[PM][DATA][BOOKS][SPLIT] n={len(token_batch)} -> {len(left)}+{len(right)}")

        out: Dict[str, Any] = {}
        out.update(self._fetch_books_batch_only(left, timeout, max_retries))
        out.update(self._fetch_books_batch_only(right, timeout, max_retries))
        return out

    @staticmethod
    def parse_orderbook_summary_levels(ob: Any) -> Tuple[List[Tuple[D, D]], List[Tuple[D, D]]]:
        """
        统一把 OrderBookSummary 或 /book dict 解析成 (bids_desc, asks_asc) 的 Decimal levels。
        """
        # case A: dict (/book)
        if isinstance(ob, dict):
            return PolymarketData._parse_pm_book_levels(ob)

        # case B: OrderBookSummary（字段名可能 bids/asks，也可能是 orderbook.bids/asks）
        bids_raw = getattr(ob, "bids", None)
        asks_raw = getattr(ob, "asks", None)

        # 有些 SDK 里 bids/asks 是对象数组，每个对象里有 price/size
        bids: List[Tuple[D, D]] = []
        asks: List[Tuple[D, D]] = []

        def _get_field(x, name: str):
            if x is None:
                return None
            if isinstance(x, dict):
                return x.get(name)
            return getattr(x, name, None)

        if isinstance(bids_raw, list):
            for lv in bids_raw:
                try:
                    bids.append((D(str(_get_field(lv, "price"))), D(str(_get_field(lv, "size")))))
                except Exception:
                    pass

        if isinstance(asks_raw, list):
            for lv in asks_raw:
                try:
                    asks.append((D(str(_get_field(lv, "price"))), D(str(_get_field(lv, "size")))))
                except Exception:
                    pass

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return bids, asks

    def _parse_pm_book_levels(j: Dict[str, Any]) -> Tuple[List[Tuple[D, D]], List[Tuple[D, D]]]:
        """
        解析 PM /book:
        兼容字段：bids/asks 可能是 [{"price":"0.12","size":"123"}, ...]
        返回 (bids_desc, asks_asc) Decimal
        """
        bids_raw = (j or {}).get("bids") or []
        asks_raw = (j or {}).get("asks") or []
        bids: List[Tuple[D, D]] = []
        asks: List[Tuple[D, D]] = []
        for lv in bids_raw:
            try:
                bids.append((D(str(lv.get("price"))), D(str(lv.get("size")))))
            except Exception:
                pass
        for lv in asks_raw:
            try:
                asks.append((D(str(lv.get("price"))), D(str(lv.get("size")))))
            except Exception:
                pass
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return bids, asks

    def get_book(self, token_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        url = f"{self.cfg.poly_clob_host}/book"
        params = {"token_id": token_id}
        try:
            return HTTP.get(url, params=params, timeout=timeout)
        except Exception:
            return None

class PolymarketTrader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = (not cfg.dry_run) and _HAS_POLY_CLOB and bool(cfg.poly_private_key)
        self.client = None
        if self.enabled:
            self.client = ClobClient(
                cfg.poly_clob_host,
                key=cfg.poly_private_key,
                chain_id=cfg.poly_chain_id,
                signature_type=2,
                funder=cfg.poly_funder or None,
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            log.info("[PM] trading enabled.")
        else:
            log.info("[PM] trading disabled (DRY_RUN or missing deps/keys).")

    def safe_buy(self, token_id: str, shares: float, limit_price: float, *, max_retries=3, retry_delay=0.2) -> bool:
        return with_retry(
            lambda: self.aggressive_buy(token_id=token_id, shares=shares, limit_price=limit_price),
            what=f"[PM] BUY token={token_id[:12]} shares={shares:.6f} lim={limit_price:.6f}",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def safe_sell(self, token_id: str, shares: float, limit_price: float, *, max_retries=3, retry_delay=0.2) -> bool:
        return with_retry(
            lambda: self.aggressive_sell(token_id=token_id, shares=shares, limit_price=limit_price),
            what=f"[PM] SELL token={token_id[:12]} shares={shares:.6f} lim={limit_price:.6f}",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def aggressive_buy(self, token_id: str, shares: float, limit_price: float) -> bool:
        """
        PM 对冲腿：
        - 可成交限价单（marketable limit）
        - FOK（要么全成，要么全撤）
        - 加滑点，避免盘口轻微移动导致失败
        """
        if not self.enabled:
            log.info(
                f"[PM][DRY] BUY(FOK) token={token_id} "
                f"shares={shares:.6f} limit={limit_price:.6f}"
            )
            return True

        try:
            base_px = Decimal(str(limit_price))
            px = base_px + MAX_SLIPPAGE

            size = _q_pm_taker_size(shares)  # ✅ 核心修复
            if size <= 0:
                return False

            args = OrderArgs(
                token_id=str(token_id),
                side=BUY,
                price=float(px),
                size=size,  # ✅ 用量化后的
            )

            signed = self.client.create_order(args)
            resp = self.client.post_order(signed)

            ok = isinstance(resp, dict) and resp.get("success", True)
            if not ok:
                log.error(f"[PM] FOK buy failed token={token_id} resp={resp}")

            return ok

        except Exception as e:
            log.exception(f"[PM] FOK buy exception: {e}")
            return False

    def aggressive_sell(self, token_id: str, shares: float, limit_price: float) -> bool:
        """
        PM 对冲腿：
        - 可成交限价单（marketable limit）
        - 卖出侧：减滑点（压低价格）提高成交确定性
        """
        if not self.enabled:
            log.info(
                f"[PM][DRY] SELL token={token_id} "
                f"shares={shares:.6f} limit={limit_price:.6f}"
            )
            return True

        try:
            base_px = Decimal(str(limit_price))
            px = base_px - MAX_SLIPPAGE  # ✅ 卖出侧：减滑点更易成交

            size = _q_pm_taker_size(shares)  # ✅ 核心修复：量化 size
            if size <= 0:
                return False

            args = OrderArgs(
                token_id=str(token_id),
                side=SELL,
                price=float(px),  # ✅ 必须是 float，别传 str
                size=size,  # ✅ 和 buy 一致：用量化后的 size
            )

            signed = self.client.create_order(args)
            resp = self.client.post_order(signed)

            ok = isinstance(resp, dict) and resp.get("success", True)
            if not ok:
                log.error(f"[PM] sell failed token={token_id} resp={resp}")

            return ok

        except Exception as e:
            log.exception(f"[PM] sell exception: {e}")
            return False