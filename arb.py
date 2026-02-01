# arb.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from config import CFG


# =========================
# Decimal helpers
# =========================

D0 = Decimal("0")
D1 = Decimal("1")


def _D(x) -> Decimal:
    """统一把 float/int/str/Decimal 转成 Decimal（避免 float 精度坑）"""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def clamp01d(x: Decimal) -> Decimal:
    if x < D0:
        return D0
    if x > D1:
        return D1
    return x


def make_ev_key(event_id: str, pm_side: str) -> str:
    """
    pm_side: "YES" or "NO"
    """
    return f"{event_id}::pm_{pm_side.lower()}"


# =========================
# VWAP / sizing helpers
# =========================

def vwap_from_levels(
    levels: List[Tuple[Decimal, Decimal]],
    target_shares: Decimal,
) -> Optional[Decimal]:
    """
    从订单薄 levels 计算买入/卖出 target_shares 的 VWAP（Decimal 版）。
    - levels: [(price, size), ...] 价格/数量均 Decimal
    - target_shares: Decimal
    返回：Decimal VWAP；若深度不足返回 None
    """
    need = target_shares
    if need <= D0:
        return None

    cost = D0
    got = D0

    for p, sz in levels:
        if sz <= D0:
            continue
        take = sz if sz < need else need
        cost += take * p
        got += take
        need -= take
        if need <= D0:
            break

    if got <= D0 or need > D0:
        return None
    return cost / got


def take_top_levels(levels: List[Tuple[Decimal, Decimal]], n: int) -> List[Tuple[Decimal, Decimal]]:
    out: List[Tuple[Decimal, Decimal]] = []
    for p, sz in levels[: max(0, n)]:
        if sz > D0:
            out.append((clamp01d(p), sz))
    return out


def max_shares_under_notional(
    asks: List[Tuple[Decimal, Decimal]],
    notional: Decimal
) -> Tuple[Decimal, Decimal, Optional[Decimal]]:
    """
    给定买盘 asks levels，最多吃到 notional 的情况下，最大可买 shares。
    返回 (shares, cost, vwap)；如果连1档都买不到则 shares=0。
    """
    if notional <= D0:
        return (D0, D0, None)

    remain = notional
    shares = D0
    cost = D0

    for p, sz in asks:
        if p <= D0 or sz <= D0:
            continue

        max_cost_lv = p * sz
        if max_cost_lv <= remain:
            shares += sz
            cost += max_cost_lv
            remain -= max_cost_lv
        else:
            take_sh = (remain / p).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
            if take_sh > sz:
                take_sh = sz
            if take_sh <= D0:
                break
            shares += take_sh
            cost += take_sh * p
            remain = D0
            break

        if remain <= D0:
            break

    if shares <= D0:
        return (D0, D0, None)

    return (shares, cost, cost / shares)


def opinion_taker_fee_rate(topic_rate: Decimal, price: Decimal) -> Decimal:
    """fee_rate = topic_rate × p × (1-p)"""
    p = clamp01d(price)
    return clamp01d(topic_rate * p * (D1 - p))


# =========================
# Orderbook cost/revenue (single source of truth)
# =========================

STEP = Decimal("0.000001")  # shares 粒度（二分用；最终下单你也会 format 到 6 位）

def _q(x: Decimal) -> Decimal:
    # 二分时用 ROUND_DOWN 更安全：避免 mid 变大导致不可行边界抖动
    return x.quantize(STEP, rounding=ROUND_DOWN)


def _cost_and_last_price_from_asks(
    asks: List[Tuple[Decimal, Decimal]],
    shares: Decimal,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    吃 asks 买 shares -> (cost, last_price)，深度不足 -> (None, None)
    asks: price asc
    """
    need = shares
    if need <= D0:
        return (None, None)

    cost = D0
    last: Optional[Decimal] = None

    for p, sz in asks:
        if p <= D0 or sz <= D0:
            continue
        take = sz if sz < need else need
        cost += take * p
        last = p
        need -= take
        if need <= D0:
            return (cost, last)

    return (None, None)


def _revenue_and_last_price_from_bids(
    bids: List[Tuple[Decimal, Decimal]],
    shares: Decimal,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    吃 bids 卖 shares -> (revenue, last_price)，深度不足 -> (None, None)
    bids: price desc
    """
    need = shares
    if need <= D0:
        return (None, None)

    rev = D0
    last: Optional[Decimal] = None

    for p, sz in bids:
        if p <= D0 or sz <= D0:
            continue
        take = sz if sz < need else need
        rev += take * p
        last = p
        need -= take
        if need <= D0:
            return (rev, last)

    return (None, None)


def _shares_at_notional_from_asks(
    asks: List[Tuple[Decimal, Decimal]],
    notional: Decimal,
) -> Tuple[Decimal, Decimal, Optional[Decimal], Optional[Decimal]]:
    """
    用 notional 去吃 asks，返回：
      (shares, cost, vwap, last_price)
    深度不够：会吃到尽可能多（cost <= notional），shares 可能>0。
    """
    if notional <= D0:
        return (D0, D0, None, None)

    remain = notional
    shares = D0
    cost = D0
    last: Optional[Decimal] = None

    for p, sz in asks:
        if p <= D0 or sz <= D0:
            continue
        max_cost_lv = p * sz
        if max_cost_lv <= remain:
            shares += sz
            cost += max_cost_lv
            remain -= max_cost_lv
            last = p
        else:
            take_sh = remain / p  # 不量化，精度留给二分/下单格式化
            if take_sh > sz:
                take_sh = sz
            if take_sh <= D0:
                break
            shares += take_sh
            cost += take_sh * p
            last = p
            remain = D0
            break

        if remain <= D0:
            break

    if shares <= D0:
        return (D0, D0, None, None)

    return (shares, cost, cost / shares, last)


# =========================
# Feasibility checks
# =========================

def _feasible_mm(
    sh: Decimal,
    pm_asks: List[Tuple[Decimal, Decimal]],
    op_asks: List[Tuple[Decimal, Decimal]],
    topic_rate: Decimal,
    min_usd: Decimal,
    max_usd: Decimal,
    *,
    exec_buffer: Decimal = Decimal(0.01),
) -> Optional[Dict[str, Any]]:
    """分支①：PM 吃单 + OP 吃单(+fee) 的可行性与指标"""
    if sh <= D0:
        return None

    pm_cost, pm_last = _cost_and_last_price_from_asks(pm_asks, sh)
    op_cost, op_last = _cost_and_last_price_from_asks(op_asks, sh)
    if pm_cost is None or op_cost is None or pm_last is None or op_last is None:
        return None

    if pm_cost < min_usd or op_cost < min_usd:
        return None
    if pm_cost > max_usd or op_cost > max_usd:
        return None

    op_avg = clamp01d(op_cost / sh)
    fee_r = opinion_taker_fee_rate(topic_rate, op_avg)

    fee_usd = op_cost * fee_r
    if fee_usd < CFG.OP_MIN_TAKER_FEE_USD:
        fee_usd = CFG.OP_MIN_TAKER_FEE_USD

    op_cost_eff = op_cost + fee_usd

    sum_per_share = (pm_cost + op_cost_eff) / sh
    if sum_per_share >= (D1 - exec_buffer):
        return None

    return {
        "pm_cost": pm_cost,
        "op_cost": op_cost,
        "op_cost_eff": op_cost_eff,
        "pm_last": pm_last,
        "op_last": op_last,
        "op_fee_rate": fee_r,
        "op_fee_usd": fee_usd,
        "sum_per_share": sum_per_share,
    }


def _feasible_pm_oplim(
    sh: Decimal,
    pm_asks: List[Tuple[Decimal, Decimal]],
    my_price: Decimal,
    op_best_ask: Optional[Decimal],
    topic_rate: Decimal,
    min_usd: Decimal,
    max_usd: Decimal,
    *,
    exec_buffer: Decimal = D0,
) -> Optional[Dict[str, Any]]:
    """分支②：PM 吃单 + OP 挂限价(可能taker保护) 的可行性与指标"""
    if sh <= D0 or my_price <= D0:
        return None

    pm_cost, pm_last = _cost_and_last_price_from_asks(pm_asks, sh)
    if pm_cost is None or pm_last is None:
        return None

    op_quote = sh * my_price
    if pm_cost < min_usd or op_quote < min_usd:
        return None
    if pm_cost > max_usd or op_quote > max_usd:
        return None

    op_fee_r = D0
    fee_usd = D0

    # 如果你的 maker 价贴到/跨过 ask：按 taker 估算手续费（含最低 0.5U）
    if op_best_ask is not None:
        eps = Decimal("0.000001")
        if my_price >= (op_best_ask - eps):
            op_fee_r = opinion_taker_fee_rate(topic_rate, clamp01d(op_best_ask))
            fee_usd = op_quote * op_fee_r
            if fee_usd < CFG.OP_MIN_TAKER_FEE_USD:
                fee_usd = CFG.OP_MIN_TAKER_FEE_USD

    op_cost_eff = op_quote + fee_usd
    sum_per_share = (pm_cost + op_cost_eff) / sh
    if sum_per_share >= (D1 - exec_buffer):
        return None

    return {
        "pm_cost": pm_cost,
        "op_quote": op_quote,
        "op_cost_eff": op_cost_eff,
        "pm_last": pm_last,
        "op_fee_rate": op_fee_r,
        "op_fee_usd": fee_usd,
        "sum_per_share": sum_per_share,
    }


def _binary_search_max_feasible(
    sh_hi: Decimal,
    check_fn: Callable[[Decimal], Optional[Dict[str, Any]]],
) -> Tuple[Decimal, Optional[Dict[str, Any]]]:
    """在 [0, sh_hi] 找最大可行 shares（近似单调），返回 (best_sh, best_info)"""
    hi = _q(sh_hi)
    if hi <= D0:
        return (D0, None)

    lo = D0
    best = D0
    best_info: Optional[Dict[str, Any]] = None

    info_hi = check_fn(hi)
    if info_hi is not None:
        return (hi, info_hi)

    for _ in range(6):
        mid = _q((lo + hi) / 2)
        if mid <= D0:
            hi = mid
            continue

        info = check_fn(mid)
        if info is not None:
            best = mid
            best_info = info
            lo = mid
        else:
            hi = hi - STEP if hi > STEP else D0

        if hi <= lo + STEP:
            break

    return (best, best_info)


# =========================
# Maker price helper (Decimal)
# =========================

def maker_price(best_bid: Decimal, best_ask: Decimal) -> Decimal:
    """
    生成“挂单买入”的 limit 价：
    - 默认挂在 best_bid（不抬价）
    - 若价差太大，可抬一点
    """
    spread = best_ask - best_bid
    if spread >= Decimal("0.002"):
        return clamp01d(best_bid + Decimal("0.001"))
    return clamp01d(best_bid)


# =========================
# build_refined_plan
# =========================

def build_refined_plan(
    *,
    op_side: str,
    pm_side: str,
    pm_asks,
    op_bids,
    op_asks,
    topic_rate: Decimal,
    min_usd: Decimal,
    target_usd: Decimal,
    max_usd: Decimal,
):
    if not pm_asks or not op_asks:
        return None

    exec_buffer = _D(0.01)

    bid0 = op_bids[0][0] if op_bids else None
    ask0 = op_asks[0][0] if op_asks else None
    my_price = maker_price(bid0, ask0) if (bid0 is not None and ask0 is not None) else None

    def _ladder_targets(t: Decimal):
        t = max(t, min_usd)
        half = max(min_usd, (t / 2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        fifth = max(min_usd, (t / 5).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return [t, half, fifth]

    GOOD_FILL_RATIO = Decimal("0.98")

    pm_sh_cap, _, _, _ = _shares_at_notional_from_asks(pm_asks, max_usd)
    op_sh_cap, _, _, _ = _shares_at_notional_from_asks(op_asks, max_usd)
    sh_cap_mm = pm_sh_cap if pm_sh_cap < op_sh_cap else op_sh_cap

    def _rank(c):
        return (
            c.get("_primary_notional", D0),
            c.get("_total_notional", D0),
            D1 - _D(c.get("est_sum_cost_per_share", 999999)),
        )

    def _strip_internal(c):
        return {k: v for k, v in c.items() if k not in ("_primary_notional", "_total_notional")}

    def _chk_mm(sh: Decimal):
        return _feasible_mm(sh, pm_asks, op_asks, topic_rate, min_usd, max_usd, exec_buffer=exec_buffer)

    # =======================
    # Pass 1: MM
    # =======================
    best_mm = None
    '''
    for tgt in _ladder_targets(target_usd):
        # ---- 主边=OP ----
        s_op_tgt, _, _, _ = _shares_at_notional_from_asks(op_asks, tgt)
        s0 = s_op_tgt if s_op_tgt < sh_cap_mm else sh_cap_mm

        s_best, info = _binary_search_max_feasible(s0, _chk_mm)
        if info is not None and s_best > D0:
            pm_cost = info["pm_cost"]
            op_cost_eff = info["op_cost_eff"]

            cand = {
                "branch": "MM",
                "op_side": op_side,
                "pm_side": pm_side,
                "shares": float(s_best),
                "pm_vwap": float(pm_cost / s_best),
                "op_vwap": float(info["op_cost"] / s_best),
                "op_fee_rate": float(info["op_fee_rate"]),
                "op_total_fee": float(info["op_fee_usd"]),
                "est_sum_cost_per_share": float(info["sum_per_share"]),
                "pm_limit_price": float(info["pm_last"]),
                "op_price": float(info["op_last"]),
                "_primary_notional": op_cost_eff,
                "_total_notional": pm_cost + op_cost_eff,
            }

            if op_cost_eff >= (tgt * GOOD_FILL_RATIO):
                return _strip_internal(cand)

            if best_mm is None or _rank(cand) > _rank(best_mm):
                best_mm = cand

        # ---- 主边=PM ----
        s_pm_tgt, _, _, _ = _shares_at_notional_from_asks(pm_asks, tgt)
        s0b = s_pm_tgt if s_pm_tgt < sh_cap_mm else sh_cap_mm

        s_best, info = _binary_search_max_feasible(s0b, _chk_mm)
        if info is not None and s_best > D0:
            pm_cost = info["pm_cost"]
            op_cost_eff = info["op_cost_eff"]

            cand = {
                "branch": "MM",
                "op_side": op_side,
                "pm_side": pm_side,
                "shares": float(s_best),
                "pm_vwap": float(pm_cost / s_best),
                "op_vwap": float(info["op_cost"] / s_best),
                "op_fee_rate": float(info["op_fee_rate"]),
                "op_total_fee": float(info["op_fee_usd"]),
                "est_sum_cost_per_share": float(info["sum_per_share"]),
                "pm_limit_price": float(info["pm_last"]),
                "op_price": float(info["op_last"]),
                "_primary_notional": pm_cost,
                "_total_notional": pm_cost + op_cost_eff,
            }

            if best_mm is None or _rank(cand) > _rank(best_mm):
                best_mm = cand

    if best_mm is not None:
        return _strip_internal(best_mm)
    '''
    # ==========================
    # Pass 2: PM + OP_LIM
    # ==========================
    best_oplim = None

    if my_price is not None and my_price > D0:
        op_sh_cap2 = (max_usd / my_price) if my_price > D0 else D0
        sh_cap2 = pm_sh_cap if pm_sh_cap < op_sh_cap2 else op_sh_cap2

        def _chk_oplim(sh: Decimal):
            return _feasible_pm_oplim(
                sh, pm_asks, my_price, ask0, topic_rate, min_usd, max_usd, exec_buffer=exec_buffer
            )

        for tgt in _ladder_targets(target_usd):
            # ---- 主边=OP ----
            s_op_tgt2 = (tgt / my_price) if my_price > D0 else D0
            s0 = s_op_tgt2 if s_op_tgt2 < sh_cap2 else sh_cap2

            s_best, info2 = _binary_search_max_feasible(s0, _chk_oplim)
            if info2 is not None and s_best > D0:
                pm_cost = info2["pm_cost"]
                op_quote = info2["op_quote"]

                cand = {
                    "branch": "PM_OPLIM",
                    "op_side": op_side,
                    "pm_side": pm_side,
                    "shares": float(s_best),
                    "pm_vwap": float(pm_cost / s_best),
                    "op_limit_price": float(my_price),
                    "op_bid0": float(bid0) if bid0 is not None else None,
                    "op_ask0": float(ask0) if ask0 is not None else None,
                    "op_fee_rate": float(info2.get("op_fee_rate", 0.0)),
                    "op_total_fee": float(0),
                    "est_sum_cost_per_share": float(info2["sum_per_share"]),
                    "pm_limit_price": float(info2["pm_last"]),
                    "_primary_notional": op_quote,
                    "_total_notional": pm_cost + op_quote,
                }

                if op_quote >= (tgt * GOOD_FILL_RATIO):
                    return _strip_internal(cand)

                if best_oplim is None or _rank(cand) > _rank(best_oplim):
                    best_oplim = cand

            # ---- 主边=PM ----
            s_pm_tgt2, _, _, _ = _shares_at_notional_from_asks(pm_asks, tgt)
            s0b = s_pm_tgt2 if s_pm_tgt2 < sh_cap2 else sh_cap2

            s_best, info2 = _binary_search_max_feasible(s0b, _chk_oplim)
            if info2 is not None and s_best > D0:
                pm_cost = info2["pm_cost"]
                op_quote = info2["op_quote"]

                cand = {
                    "branch": "PM_OPLIM",
                    "op_side": op_side,
                    "pm_side": pm_side,
                    "shares": float(s_best),
                    "pm_vwap": float(pm_cost / s_best),
                    "op_limit_price": float(my_price),
                    "op_bid0": float(bid0) if bid0 is not None else None,
                    "op_ask0": float(ask0) if ask0 is not None else None,
                    "op_fee_rate": float(info2.get("op_fee_rate", 0.0)),
                    "op_total_fee": float(0),
                    "est_sum_cost_per_share": float(info2["sum_per_share"]),
                    "pm_limit_price": float(info2["pm_last"]),
                    "_primary_notional": pm_cost,
                    "_total_notional": pm_cost + op_quote,
                }

                if best_oplim is None or _rank(cand) > _rank(best_oplim):
                    best_oplim = cand

    if best_oplim is None:
        return None
    return _strip_internal(best_oplim)


# =========================
# Orderbook parsing (Opinion)
# =========================

def parse_op_orderbook(ob: Dict[str, Any]) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
    """
    解析 Opinion orderbook，输出：
    - bids: price 从大到小
    - asks: price 从小到大
    """
    data = ob.get("data") or ob.get("result") or ob
    bids_raw = data.get("bids") or []
    asks_raw = data.get("asks") or []

    bids: List[Tuple[Decimal, Decimal]] = []
    asks: List[Tuple[Decimal, Decimal]] = []

    for lv in bids_raw:
        try:
            bids.append((_D(lv.get("price")), _D(lv.get("size"))))
        except Exception:
            pass

    for lv in asks_raw:
        try:
            asks.append((_D(lv.get("price")), _D(lv.get("size"))))
        except Exception:
            pass

    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    return bids, asks


# =========================
# YES <-> NO book inversion
# =========================

def invert_yes_book_to_no(
    yes_bids: List[Tuple[Decimal, Decimal]],
    yes_asks: List[Tuple[Decimal, Decimal]],
) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
    """
    二元市场 YES/NO 互补（理想情况下）：
    - NO_bid(price) 对应 YES_ask(1-price)
    - NO_ask(price) 对应 YES_bid(1-price)
    """
    no_bids = [(D1 - p, sz) for (p, sz) in yes_asks]
    no_asks = [(D1 - p, sz) for (p, sz) in yes_bids]

    no_bids.sort(key=lambda x: x[0], reverse=True)
    no_asks.sort(key=lambda x: x[0])
    return no_bids, no_asks


# =========================
# Opportunity finders (signal layer)
# =========================

def find_buy_opportunity(
    buffer: float,
    pm_yes_ask: float,
    pm_no_ask: float,
    op_yes_bid,
    op_yes_ask,
    op_no_bid,
    op_no_ask,
    *,
    use_maker_price: bool = False,
) -> Optional[Dict[str, Any]]:
    buf = _D(buffer)

    pm_yes_ask = _D(pm_yes_ask)
    pm_no_ask = _D(pm_no_ask)

    op_yes_bid = _D(op_yes_bid)
    op_yes_ask = _D(op_yes_ask)
    op_no_bid = _D(op_no_bid)
    op_no_ask = _D(op_no_ask)

    if use_maker_price:
        op_yes_limit = maker_price(op_yes_bid, op_yes_ask)
        op_no_limit = maker_price(op_no_bid, op_no_ask)
    else:
        op_yes_limit = op_yes_ask
        op_no_limit = op_no_ask

    # 新增：OP 价格过低（<0.2）则跳过该方向
    OP_MIN = _D("0.2")
    allow_a = op_no_ask >= OP_MIN and op_no_bid >= OP_MIN  # A: OP买NO + PM买YES
    allow_b = op_yes_ask >= OP_MIN and op_yes_bid >= OP_MIN  # B: OP买YES + PM买NO

    best_edge = Decimal("-1")
    best: Optional[Dict[str, Any]] = None

    if allow_a:
        edge_a = (D1 - buf) - (op_no_limit + pm_yes_ask)
        if edge_a > best_edge:
            best_edge = edge_a
            best = {
                "est_edge": float(best_edge),
                "op_side": "NO",
                "pm_side": "YES",
                "op_limit_price": float(op_no_limit),
            }

    if allow_b:
        edge_b = (D1 - buf) - (op_yes_limit + pm_no_ask)
        if edge_b > best_edge:
            best_edge = edge_b
            best = {
                "est_edge": float(best_edge),
                "op_side": "YES",
                "pm_side": "NO",
                "op_limit_price": float(op_yes_limit),
            }

    if best_edge <= D0:
        return None
    return best


def find_sell_opportunity(
    buffer: float,
    pm_yes_bid: float,
    pm_no_bid: float,
    op_yes_bid,
    op_yes_ask,
    op_no_bid,
    op_no_ask,
) -> Optional[Dict[str, Any]]:
    buf = _D(buffer)

    pm_yes_bid = _D(pm_yes_bid)
    pm_no_bid = _D(pm_no_bid)

    op_yes_bid = _D(op_yes_bid)
    op_no_bid = _D(op_no_bid)

    edge_yes = op_yes_bid - (D1 - pm_no_bid)
    edge_no = op_no_bid - (D1 - pm_yes_bid)

    if edge_yes > buf:
        return {"op_sell": "YES", "signal": float(edge_yes)}
    if edge_no > buf:
        return {"op_sell": "NO", "signal": float(edge_no)}
    return None