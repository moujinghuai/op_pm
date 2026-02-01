# mapper.py
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from rapidfuzz import process, fuzz
import re

import logging

log = logging.getLogger(__name__)


class ManualEntry(TypedDict):
    op_marketId: str
    title: str  # 仅用于人眼识别


MANUAL_MAP: Dict[str, ManualEntry] = {
    "1139942": {"op_marketId": "3973", "title": "Will US or Israel strike Iran by January 31, 2026?"},
    "1198479": {"op_marketId": "4726", "title": "Will US or Israel strike Iran by February 28, 2026?"},
    "1198523": {"op_marketId": "4727", "title": "Will US or Israel strike Iran by March 31, 2026?"},
}

_HAS_ST = False


def token_count(text: str) -> int:
    s = normalize_text(text)
    if not s:
        return 0
    return len(s.split())


def token_count_gap_too_large(a: str, b: str, gap: int = 4) -> bool:
    """
    两个标题的词数差 >= gap 则认为“结构不一致”，用于剔除 token_set_ratio=100 的包含型假匹配
    """
    return abs(token_count(a) - token_count(b)) >= int(gap)


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()  # ① 全小写
    s = re.sub(r"[^\w\s]", " ", s)  # ② 去标点（保留字母/数字/空格）
    s = re.sub(r"\s+", " ", s).strip()  # 顺手压缩空白（不改变语义）
    return s


def has_time_bucket(text: str) -> bool:
    """
    判断文本是否包含“时间分桶/区间”语义。
    只要同时出现：
      - 区间/比较词（between/less than/more than/within/under/over/at least...）
      - 时间单位（minute/hour/day/week/month/year...）
    就认为是分桶。
    """
    s = normalize_text(text)

    range_tokens = [
        "between", "in between",
        "less than", "more than",
        "within", "under", "over",
        "at least", "at most",
        "no more than", "no less than",
        "from", "to",  # 有些标题会写 from X to Y
    ]
    time_tokens = [
        "second", "seconds",
        "minute", "minutes",
        "hour", "hours",
        "day", "days",
        "week", "weeks",
        "month", "months",
        "year", "years",
    ]

    has_range = any(t in s for t in range_tokens)
    has_time = any(t in s for t in time_tokens)

    return bool(has_range and has_time)


def pm_text(m: Dict[str, Any]) -> str:
    # return f"{m.get('question','')}\n{m.get('description','')}".strip()
    return f"{m.get('question', '')}".strip()


def op_text(m: Dict[str, Any]) -> str:
    # return f"{m.get('marketTitle') or m.get('title','')}\n{m.get('rules') or m.get('description','')}".strip()
    return f"{m.get('marketTitle') or m.get('title', '')}".strip()


class EventMapper:
    def __init__(self, topk_fuzz: int, topk_sbert: int, topk_rerank: int, min_conf: float):
        self.topk_fuzz = topk_fuzz
        self.topk_sbert = topk_sbert
        self.topk_rerank = topk_rerank
        self.min_conf = min_conf
        self.bi = None
        self.ce = None

    def build_map(
            self,
            pm_markets: List[Dict[str, Any]],
            op_markets: List[Dict[str, Any]],
            end_time_max_diff_hours: int,
            get_pm_end_ts,
            get_op_end_ts,
            pm_key_fn,
            now_ts: int,
    ) -> Dict[str, Dict[str, Any]]:
        # NEW: MANUAL_MAP 中已占用的 OP marketId 集合（用于 fuzz 结果去重/避让）
        manual_op_ids = {str(v["op_marketId"]) for v in MANUAL_MAP.values()}

        # Opinion: marketId -> index（用于 O(1) 查找手动映射）
        op_id_to_idx: Dict[str, int] = {}
        for idx, op in enumerate(op_markets):
            mid = op.get("marketId")
            if mid is not None:
                op_id_to_idx[str(mid)] = idx

        # Opinion 文本预处理（用于 fuzz）
        op_texts_raw = [op_text(x) for x in op_markets]
        op_texts_norm = [normalize_text(t) for t in op_texts_raw]

        out: Dict[str, Dict[str, Any]] = {}

        for pm in pm_markets:
            key = pm_key_fn(pm)

            pm_id = pm.get("id")
            pm_id_str = str(pm_id) if pm_id is not None else None

            # ===== 2) PM 不在手动表：走 fuzz（只认 100）=====
            q_raw = pm_text(pm)
            q_norm = normalize_text(q_raw)

            best = process.extractOne(
                q_norm,
                op_texts_norm,
                scorer=fuzz.token_set_ratio,
            )
            if not best:
                continue

            _choice, score, best_i = best

            # 只认 100，其余丢弃
            if int(score) != 100:
                continue

            op = op_markets[best_i]

            # 如果 fuzz 命中的 op_marketId 已经在 MANUAL_MAP 里被占用，则跳过
            op_mid = str(op.get("marketId") or "")
            if op_mid and op_mid in manual_op_ids:
                log.info(f"[fuzz_100][skip] pm_id={pm_id_str} -> op_mid={op_mid} already in MANUAL_MAP")
                continue

            # ===== 3) 结构一致性过滤（仅作用于 fuzz_100）=====
            # 3.1 时间分桶一致性：只有一边有分桶 => 不匹配
            pm_bucket = has_time_bucket(q_raw)
            op_bucket = has_time_bucket(op_text(op))
            if pm_bucket != op_bucket:
                continue

            # 3.2 token 数差过滤：短句被长句“包含”导致 token_set_ratio=100 的假匹配
            if token_count_gap_too_large(q_raw, op_text(op), gap=5):
                continue

            # ===== 4) 结束时间校验（保留你原逻辑）=====
            pm_end = get_pm_end_ts(pm)
            op_end = get_op_end_ts(op)
            if pm_end is None or op_end is None:
                continue

            # 24h 内严格一致
            pm_left = int(pm_end) - int(now_ts)
            op_left = int(op_end) - int(now_ts)
            if (pm_left < 24 * 3600) and (op_left < 24 * 3600):
                if int(pm_end) != int(op_end):
                    continue

            diff_sec = int(pm_end) - int(op_end)
            diff_hours = abs(diff_sec) / 3600.0

            out[key] = {
                "valid": True,
                "confidence": 1.0,
                "end_time_pm": pm_end,
                "end_time_op": op_end,
                "end_time_diff_sec": diff_sec,  # pm_end - op_end (可正可负)
                "end_time_diff_hours": diff_hours,  # 绝对值小时差
                "sources": {"polymarket": pm, "opinion": op},
                "reason": "fuzz_100",
            }

        return out