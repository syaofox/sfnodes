"""SF Wan Window Planner — 只算不跑的窗口映射表（SFWanWindowLoRA 伴侣工具）。

输入总帧 + 窗参 + 槽数 + cond 数，输出对照表（窗口 latent/实帧区间、
region/cond 段、slot）与错位警告。跑队列前看一眼，确认 N 与映射无误。

保真：窗口切分直接调原生 schedule 函数（`comfy/context_windows.py`，
`func(总长, handler, model_options)`，handler 仅需五个属性，此处用
`_StubHandler` 供给），region 数学复用原生 `IndexListContextWindow`；
原生改算法，planner 自动跟进。UNIFORM 系按 step 0 求值并强制漂移警告。

无 torch 依赖，无前端，单 STRING 输出。
"""

import json

from ...sf_utils.logger import get_logger

try:
    import comfy.context_windows as _cw
    _SCHEDULES = {
        "static": _cw.ContextSchedules.STATIC_STANDARD,
        "uniform": _cw.ContextSchedules.UNIFORM_STANDARD,
        "looped": _cw.ContextSchedules.UNIFORM_LOOPED,
        "batched": _cw.ContextSchedules.BATCHED,
    }
except Exception:
    _cw = None
    _SCHEDULES = {}

_CATEGORY = "sfnodes/model"

SCHEDULE_CHOICES = ["static", "uniform", "looped", "batched"]


class _StubHandler:
    """仅供给原生 schedule 函数所需的五个属性。"""

    def __init__(self, context_length, context_overlap, context_stride=1,
                 closed_loop=False, step=0):
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self._step = step


def _to_latent(real_frames):
    return max(((int(real_frames) - 1) // 4) + 1, 1)


def _real_span(start_latent, end_latent):
    """latent 闭区间 [s, e] → 实帧闭区间（Wan 4 倍时序）。"""
    return (int(start_latent) * 4 + 1, int(end_latent) * 4 + 1)


def _region_of(index_list, total_frames, n_conds):
    """region 数学走原生 IndexListContextWindow（单源头，防分叉）。"""
    try:
        w = _cw.IndexListContextWindow(list(index_list), dim=0,
                                       total_frames=total_frames)
        return w.get_region_index(n_conds)
    except Exception:
        if n_conds <= 1:
            return 0
        ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
        return min(max(int(ratio * n_conds), 0), n_conds - 1)


def plan_windows(total_frames, context_length, context_overlap, schedule,
                 n_slots, n_conds, context_stride=1, closed_loop=False):
    """纯计算：返回 (rows, warnings)。

    rows: [{order, latent:[s,e], real:[s,e], region, slot}]，order = 原生
      求值顺序（patch-swap 的 window_idx 与之同一时钟）。
    warnings: [str]，错位/回绕/漂移提示。
    """
    if _cw is None:
        raise RuntimeError(
            "[SFWanWindowPlanner] This ComfyUI build lacks comfy.context_windows. Update ComfyUI.")
    total_frames = max(int(total_frames), 1)
    context_length = max(int(context_length), 1)
    context_overlap = max(int(context_overlap), 0)
    n_slots = max(int(n_slots), 1)
    n_conds = max(int(n_conds), 1)
    if schedule not in _SCHEDULES:
        raise ValueError(f"Unknown schedule '{schedule}'.")
    handler = _StubHandler(context_length, context_overlap, context_stride,
                           closed_loop, step=0)
    func = _cw.get_matching_context_schedule(_SCHEDULES[schedule]).func
    windows = func(total_frames, handler, {})
    rows = []
    for order, idx in enumerate(windows):
        idx = list(idx)
        region = _region_of(idx, total_frames, n_conds)
        rows.append({
            "order": order,
            "latent": [min(idx), max(idx)],
            "real": list(_real_span(min(idx), max(idx))),
            "region": region,
            "slot": order % n_slots,
        })
    warnings = []
    n_windows = len(rows)
    if n_windows != n_slots:
        warnings.append(
            f"窗数({n_windows}) != 槽数({n_slots})："
            f"{'第 %d 窗起轮回复用槽' % n_slots if n_windows > n_slots else '后 %d 槽永不命中' % (n_slots - n_windows)}")
    used_regions = sorted({r["region"] for r in rows})
    idle = [c for c in range(n_conds) if c not in used_regions]
    if idle:
        warnings.append(f"cond {idle} 段全程未被任何窗口命中（共 {n_conds} 段）")
    # 注：多窗复用同一段（如 STATIC 尾窗收尾）属正常，不警告，表格自明。
    if schedule in ("uniform", "looped"):
        warnings.append("UNIFORM 系每步按 ordered_halving 漂移窗口划分：本表仅 step 0 有效，"
                        "严格分段请用 static；slot 按求值顺序轮换，逐 step 错位")
    if schedule == "batched":
        warnings.append("BATCHED 无重叠：窗间硬切，动作连续处可能见接缝")
    return rows, warnings


def flush_totals(length, overlap, k_max=8):
    """齐平总数序列：k 个窗恰好铺满（尾窗无前移）时的总数（latent/实帧）。

    T(k) = (k-1) * (L-O) + L；L=21/O=4 时得 latent 21/38/55/72…，
    即实帧 81/149/217/285…（`68a+13` 数列，此处算出而非硬编码）。
    """
    length = max(int(length), 1)
    stride = max(int(length) - int(overlap), 1)
    out = []
    for k in range(1, max(int(k_max), 1) + 1):
        total = (k - 1) * stride + length
        out.append({"windows": k, "total_latent": total,
                    "total_real": (total - 1) * 4 + 1})
    return out


def suggest_settings(total, length, overlap, schedule, n_slots, n_conds,
                     n_windows):
    """参数设置建议（纯函数）：每项给"当前值 → 建议 + 理由"。"""
    tips = []
    # 总实帧：就近两档齐平值
    seq = flush_totals(length, overlap)
    total_real = (total - 1) * 4 + 1
    below = [c for c in seq if c["total_real"] <= total_real]
    above = [c for c in seq if c["total_real"] > total_real]
    if below and below[-1]["total_real"] == total_real:
        tips.append(f"总实帧：{total_real}（当前 ✓，{below[-1]['windows']} 窗齐平）")
    else:
        opts = ([f"{below[-1]['total_real']}（{below[-1]['windows']} 窗齐平）"]
                if below else [])
        opts += [f"{above[0]['total_real']}（{above[0]['windows']} 窗齐平）"] if above else []
        tips.append(f"总实帧：当前 {total_real}（尾窗前移，不齐平）→ 建议 {' / '.join(opts)}")
    # 窗长 / 重叠：原则性建议，不动
    tips.append(f"窗长：保持 {length * 4 - 3} 实帧（Wan 原生长度，不建议动）"
                if length == 21 else f"窗长：当前 {(length - 1) * 4 + 1} 实帧（偏离 Wan 原生 81，注意单窗质量）")
    tips.append(f"重叠：保持 {overlap * 4} 实帧（融合余量；压到 0 即硬切）"
                if overlap > 0 else "重叠：当前 0（硬切，动作连续处可见接缝）")
    # schedule
    if schedule == "static":
        tips.append("schedule：static ✓（映射全 step 稳定）")
    else:
        tips.append(f"schedule：{schedule}（每步漂移，本表仅 step 0 有效；严格分段请用 static）")
    # 槽数 / cond 数
    if n_slots == n_windows:
        tips.append(f"槽数：{n_slots}（= 窗数 ✓）")
    else:
        tips.append(f"槽数：当前 {n_slots} → 建议 {n_windows}（= 窗数；否则轮回/闲置）")
    used_note = f"cond 数：{n_conds}（段数是剧情，不代设；应被窗口全覆盖且 ≤ 窗数）"
    tips.append(used_note)
    return tips


def render_report(rows, warnings, meta, tips):
    lines = []
    lines.append("## SF Wan Window Planner")
    lines.append(f"- 总实帧 {meta['total_real']} · 窗长 {meta['length_real']} · "
                 f"重叠 {meta['overlap_real']} · schedule {meta['schedule']} · "
                 f"槽数 {meta['n_slots']} · cond 数 {meta['n_conds']}")
    lines.append(f"- 换算：总数 {meta['total']} latent，窗 {meta['length']}，重叠 {meta['overlap']}")
    lines.append("")
    lines.append("## 映射表")
    lines.append("| # | latent | 实帧 | cond 段 | slot |")
    lines.append("|---|--------|------|---------|------|")
    for r in rows:
        lines.append(f"| {r['order']} | {r['latent'][0]}-{r['latent'][1]} "
                     f"| {r['real'][0]}-{r['real'][1]} | {r['region']} | {r['slot']} |")
    lines.append("")
    lines.append("## 警告")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- 无，窗/段/槽 1:1:1")
    lines.append("")
    lines.append("## 参数设置建议")
    for t in tips:
        lines.append(f"- {t}")
    return "\n".join(lines)


class SFWanWindowPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 241, "min": 1, "max": 100000,
                                        "tooltip": "总实帧数（和视频节点一致）。"}),
                "context_length": ("INT", {"default": 81, "min": 1, "max": 100000,
                                          "tooltip": "窗口长度（实帧，Wan 原生 81）。"}),
                "context_overlap": ("INT", {"default": 16, "min": 0, "max": 100000,
                                           "tooltip": "窗口重叠（实帧）。"}),
                "schedule": (SCHEDULE_CHOICES, {"default": "static",
                                               "tooltip": "uniform/looped 每步漂移，仅 step 0 表有效。"}),
                "n_slots": ("INT", {"default": 4, "min": 1, "max": 10,
                                    "tooltip": "SFWanWindowLoRA 的 window_N 槽总数（含空槽）。"}),
                "n_conds": ("INT", {"default": 3, "min": 1, "max": 20,
                                    "tooltip": "ConditionCombine 的 cond 路数（split_conds_to_windows 的段数）。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "plan"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("SF Wan Window Planner：只算不跑。输入总帧/窗长/重叠/schedule + "
                   "槽数 + cond 数，输出窗口 latent/实帧区间 × cond 段 × slot 对照表与错位警告。"
                   "排队前核对 N 与映射。")

    def plan(self, total_frames, context_length, context_overlap, schedule,
             n_slots, n_conds):
        total = _to_latent(total_frames)
        length = _to_latent(context_length)
        overlap = max(int(context_overlap) // 4, 0)
        rows, warnings = plan_windows(total, length, overlap, schedule,
                                      n_slots, n_conds)
        tips = suggest_settings(total, length, overlap, schedule,
                                int(n_slots), int(n_conds), len(rows))
        report = render_report(rows, warnings, {
            "total": total, "length": length, "overlap": overlap,
            "schedule": schedule, "n_slots": n_slots, "n_conds": n_conds,
            "total_real": int(total_frames), "length_real": int(context_length),
            "overlap_real": int(context_overlap),
        }, tips)
        logger = get_logger(__name__)
        logger.info("[SFWanWindowPlanner]\n%s", report)
        return (report,)


logger = get_logger(__name__)
