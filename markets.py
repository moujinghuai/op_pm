# markets.py
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, time


def iso_to_ts(s: Any) -> Optional[int]:
    """
    Polymarket endDate: "2026-01-28T00:00:00Z" (UTC ISO)
    也兼容：None / "" / 直接传 epoch(int/str)
    """
    if s is None or s == "":
        return None

    # 如果已经是 epoch 秒（int/float/纯数字字符串），直接转
    try:
        if isinstance(s, (int, float)):
            v = int(s)
            return v if v > 0 else None
        if isinstance(s, str) and s.strip().isdigit():
            v = int(s.strip())
            return v if v > 0 else None
    except Exception:
        pass

    # ISO 解析
    try:
        ss = str(s).strip()
        dt = datetime.fromisoformat(ss.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None

def _maybe_json_list(x: Any) -> Optional[List[Any]]:
    """Gamma markets 里很多字段是 JSON-string，做一层兼容。"""
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # 形如 '["a","b"]'
        if s[0] == "[" and s[-1] == "]":
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else None
            except Exception:
                return None
    return None

def normalize_pm_token_ids(pm: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    cids = _maybe_json_list(pm.get("clobTokenIds") or pm.get("clob_token_ids"))
    if cids and len(cids) >= 2:
        return str(cids[0]), str(cids[1])
    return None, None


def _parse_dt_to_ts(x: Any) -> Optional[int]:
    if not x:
        return None
    try:
        # 兼容: "2026-01-07T23:00:00Z"
        if isinstance(x, str):
            s = x.strip()
            # 兼容: "2026-01-07 23:00:00+00"
            if " " in s and "T" not in s and "+" in s:
                dt = datetime.fromisoformat(s)
                return int(dt.timestamp())
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(dt.timestamp())
    except Exception:
        return None
    return None

def get_pm_event_start_ts(m: Dict[str, Any]) -> Optional[int]:
    # 1) events[0].startTime 最像“比赛/事件开始”
    evs = m.get("events")
    if isinstance(evs, list) and evs:
        ts = _parse_dt_to_ts(evs[0].get("startTime") or evs[0].get("start_time"))
        if ts is not None:
            return ts

    # 2) gameStartTime / eventStartTime
    ts = _parse_dt_to_ts(m.get("gameStartTime"))
    if ts is not None:
        return ts

    ts = _parse_dt_to_ts(m.get("eventStartTime"))
    if ts is not None:
        return ts

    # 3) fallback：market.startDate（很多时候只是“市场创建/上架”）
    return _parse_dt_to_ts(m.get("startDate") or m.get("start_date"))


def filter_pm_markets(pm: List[Dict[str, Any]], now_ts: int, max_end_ts: int, min_vol_24h: float) -> List[Dict[str, Any]]:
    out = []
    for m in pm:
        if m.get("closed") is True:
            continue

        # ===== startDate 过滤=====
        start_ts = get_pm_event_start_ts(m)
        if start_ts is not None and start_ts > now_ts:
            continue

        # ===== endDate 过滤 =====
        end_ts = iso_to_ts(m.get("endDate") or m.get("end_date") or "")
        if end_ts is None:
            continue
        if end_ts < now_ts:
            continue
        if end_ts is None or end_ts > max_end_ts:
            continue

        # ===== volume 过滤 =====
        try:
            v = float(m.get("volume24hr") or m.get("volume_24hr") or m.get("volume24h") or 0)
            if v < min_vol_24h:
                continue
        except Exception:
            pass

        yes_id, no_id = normalize_pm_token_ids(m)
        if not yes_id or not no_id:
            continue
        out.append(m)

    return out


# ===================== Opinion flattening =====================

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None

def _full_question(parent_title: str, child_title: Optional[str]) -> str:
    pt = (parent_title or "").strip()
    ct = (child_title or "").strip() if child_title else ""
    return f"{pt} {ct}" if ct else pt

def flatten_opinion_markets(op_parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    输出扁平 MarketRow(dict) 列表 + full_question + event_id

    ① marketType==0 且 childMarkets 为空 => 二元事件：仅收集 statusEnum == "Activated" 的父 market
    ② marketType==1 且 childMarkets 不为空 => 多选项：仅收集 statusEnum == "Activated" 的子 market
    """
    out: List[Dict[str, Any]] = []

    for p in op_parents or []:
        parent_id = _safe_str(p.get("marketId"))
        if not parent_id:
            continue

        parent_type = int(p.get("marketType") or 0)
        parent_title = _safe_str(p.get("marketTitle"))
        parent_status = _safe_str(p.get("statusEnum"))

        children_raw = p.get("childMarkets")
        children: List[Dict[str, Any]] = children_raw if isinstance(children_raw, list) else []

        # ① Binary
        if parent_type == 0 and len(children) == 0:
            if parent_status != "Activated":
                continue
            row = dict(p)  # keep all original fields
            row["event_id"] = parent_id
            row["full_question"] = _full_question(parent_title, None)
            # 为了后面匹配更顺滑：把 marketTitle 也设成完整语义
            row["marketTitle"] = row["full_question"]
            row["_is_child"] = False
            out.append(row)
            continue

        # ② Multi
        if parent_type == 1 and len(children) > 0:
            for c in children:
                child_status = _safe_str(c.get("statusEnum"))
                if child_status != "Activated":
                    continue

                child_id = _safe_str(c.get("marketId"))
                if not child_id:
                    continue

                child_title = _safe_str(c.get("marketTitle"))

                row = dict(c)  # start from child fields (marketId/yesTokenId/noTokenId/cutoffAt...)
                # 关键：event_id + full_question
                row["event_id"] = parent_id
                row["full_question"] = _full_question(parent_title, child_title)
                row["marketTitle"] = row["full_question"]  # ✅ 让 mapping 直接吃完整语义
                row["_parentMarketId"] = parent_id
                row["_parentMarketTitle"] = parent_title
                row["_parent_volume24h"] = p.get("volume24h")
                row["_childMarketTitle"] = child_title
                row["_is_child"] = True

                # cutoffAt 若 child 缺失则继承 parent（更鲁棒）
                if not row.get("cutoffAt") and p.get("cutoffAt"):
                    row["cutoffAt"] = p.get("cutoffAt")

                out.append(row)
            continue

        # 其他异常结构：先跳过，避免把脏数据混进来
        continue

    return out


def filter_op_markets(op_rows: List[Dict[str, Any]], max_end_ts: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for m in op_rows or []:
        if _safe_str(m.get("statusEnum")) != "Activated":
            continue

        end_ts = _safe_int(m.get("cutoffAt"))
        if end_ts is None or end_ts > max_end_ts:
            continue

        if not m.get("yesTokenId") or not m.get("noTokenId"):
            continue

        if not m.get("marketId"):
            continue

        # ===== 24h volume 过滤 =====
        v24 = m.get("volume24h")
        if v24 is None:
            v24 = m.get("_parent_volume24h")

        try:
            if float(v24 or 0) <= 0:
                continue
        except Exception:
            continue

        out.append(m)

    return out


def pm_key(pm: Dict[str, Any]) -> str:
    return pm.get("slug")