# SFWanWindowPlanner 模拟测试（Python 直接运行：python3 tests/test_wan_window_planner.py）
# mock comfy.context_windows（schedule 函数 + IndexListContextWindow），验证：
#   - 实帧→latent 换算（241/81/16 → 61/21/4）
#   - STATIC 真算法（经 _StubHandler 调 mock static，61/21/4 → 4 窗 + 实帧区间）
#   - region 走原生类（n_conds=3 → 0/1/2/2）、slot 取模
#   - cond 数警告：回绕/闲置段/复用段；schedule 警告：uniform 漂移/batched 硬切
#   - 非法 schedule 抛错；节点结构与 plan() 端到端
import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock comfy.context_windows ─────────────────────────────────────────────
cw_mod = types.ModuleType("comfy.context_windows")
CALLS = []


class Schedules:
    STATIC_STANDARD = "static"
    UNIFORM_STANDARD = "uniform"
    UNIFORM_LOOPED = "looped"
    BATCHED = "batched"


def fake_static(full, handler, opts):
    CALLS.append(("static", full, handler.context_length, handler.context_overlap,
                  handler.context_stride, handler.closed_loop, handler._step))
    L, d = handler.context_length, handler.context_length - handler.context_overlap
    out, s = [], 0
    while True:
        if s + L >= full:
            start = max(0, s - (s + L - full))
            out.append(list(range(start, start + L)))
            break
        out.append(list(range(s, s + L)))
        s += d
    return out


def _canned(name, windows):
    def fn(full, handler, opts):
        CALLS.append((name, full, handler._step))
        return [list(w) for w in windows]
    return fn


FUNCS = {
    "static": fake_static,
    "uniform": _canned("uniform", [[0, 1, 2], [2, 3, 4]]),
    "looped": _canned("looped", [[3, 4, 0]]),
    "batched": _canned("batched", [[0, 1], [2, 3]]),
}


class FakeSchedule:
    def __init__(self, func):
        self.func = func


cw_mod.ContextSchedules = Schedules
cw_mod.get_matching_context_schedule = lambda name: FakeSchedule(FUNCS[name])


class FakeWindow:
    def __init__(self, index_list, dim=0, total_frames=0):
        self.index_list = list(index_list)
        # 与原生同式：中心比（min+max)/(2*total)
        self.center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)

    def get_region_index(self, num_regions):
        region_idx = int(self.center_ratio * num_regions)
        return min(max(region_idx, 0), num_regions - 1)


cw_mod.IndexListContextWindow = FakeWindow

comfy_pkg = types.ModuleType("comfy")
comfy_pkg.context_windows = cw_mod
sys.modules["comfy"] = comfy_pkg
sys.modules["comfy.context_windows"] = cw_mod

for name, path in [
    ("sfnodes", root),
    ("sfnodes.nodes", os.path.join(root, "nodes")),
    ("sfnodes.nodes.model", os.path.join(root, "nodes", "model")),
]:
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(root, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


nmod = _load("sfnodes.nodes.model.wan_window_planner", "nodes/model/wan_window_planner.py")

# ── 换算 ───────────────────────────────────────────────────────────────────
check("to_latent: 241->61", nmod._to_latent(241) == 61)
check("to_latent: 81->21", nmod._to_latent(81) == 21)
check("to_latent: 下限 1", nmod._to_latent(0) == 1)
check("real_span: (0,20)->(1,81)", nmod._real_span(0, 20) == (1, 81))
check("real_span: (40,60)->(161,241)", nmod._real_span(40, 60) == (161, 241))

# ── STATIC 主路径（用户真实配置）────────────────────────────────────────────
rows, warnings = nmod.plan_windows(61, 21, 4, "static", 4, 3)
check("static: 4 窗", len(rows) == 4)
check("static: latent 区间",
      [r["latent"] for r in rows] == [[0, 20], [17, 37], [34, 54], [40, 60]])
check("static: 实帧区间",
      [r["real"] for r in rows] == [[1, 81], [69, 149], [137, 217], [161, 241]])
check("static: region 0/1/2/2", [r["region"] for r in rows] == [0, 1, 2, 2])
check("static: slot 0/1/2/3", [r["slot"] for r in rows] == [0, 1, 2, 3])
check("static: 调原生 static（step 0）",
      CALLS[-1][:2] == ("static", 61) and CALLS[-1][2:5] == (21, 4, 1))
check("static: 规范配置零警告", warnings == [])
check("static: 无回绕警告", not any("轮回" in w for w in warnings))

# ── N=3 回绕警告 ───────────────────────────────────────────────────────────
rows3, warn3 = nmod.plan_windows(61, 21, 4, "static", 3, 3)
check("N=3: slot 轮回 0/1/2/0", [r["slot"] for r in rows3] == [0, 1, 2, 0])
check("N=3: 回绕警告", any("轮回" in w for w in warn3))

# ── cond 数警告 ────────────────────────────────────────────────────────────
_, warn_c2 = nmod.plan_windows(61, 21, 4, "static", 4, 2)
check("n_conds=2: region 0/0/1/1",
      [r["region"] for r in nmod.plan_windows(61, 21, 4, "static", 4, 2)[0]] == [0, 0, 1, 1])
check("n_conds=2: 段全用上则零警告", warn_c2 == [])
_, warn_c5 = nmod.plan_windows(61, 21, 4, "static", 4, 5)
check("n_conds=5: 闲置段警告", any("未被任何窗口命中" in w for w in warn_c5))
_, warn_c1 = nmod.plan_windows(61, 21, 4, "static", 4, 1)
check("n_conds=1: region 全 0",
      [r["region"] for r in nmod.plan_windows(61, 21, 4, "static", 4, 1)[0]] == [0, 0, 0, 0])

# ── schedule 警告 ──────────────────────────────────────────────────────────
_, warn_u = nmod.plan_windows(61, 21, 4, "uniform", 2, 3)
check("uniform: 漂移警告", any("漂移" in w for w in warn_u))
check("uniform: 按 step 0 求值", CALLS[-1] == ("uniform", 61, 0))
_, warn_l = nmod.plan_windows(61, 21, 4, "looped", 1, 3)
check("looped: 漂移警告", any("漂移" in w for w in warn_l))
_, warn_b = nmod.plan_windows(61, 21, 4, "batched", 3, 3)
check("batched: 硬切警告", any("硬切" in w for w in warn_b))

# ── 非法输入 ───────────────────────────────────────────────────────────────
try:
    nmod.plan_windows(61, 21, 4, "nope", 4, 3)
    check("非法 schedule 抛错", False)
except ValueError:
    check("非法 schedule 抛错", True)

# ── 节点端到端 ─────────────────────────────────────────────────────────────
node = nmod.SFWanWindowPlanner()
it = node.INPUT_TYPES()
check("structure: 6 输入齐", set(it["required"]) == {
    "total_frames", "context_length", "context_overlap",
    "schedule", "n_slots", "n_conds"})
check("structure: schedule 选项", list(it["required"]["schedule"][0]) == [
    "static", "uniform", "looped", "batched"])
check("structure: RETURN STRING", node.RETURN_TYPES == ("STRING",)
      and node.RETURN_NAMES == ("report",))
check("structure: CATEGORY/DESCRIPTION",
      node.CATEGORY == "sfnodes/model" and bool(node.DESCRIPTION))
report = node.plan(241, 81, 16, "static", 4, 3)[0]
check("plan(): 实帧换算进表", "61 latent" in report and "1-81" in report)
check("plan(): 含 cond 列", "cond 段" in report)
check("plan(): Markdown 节", all(s in report for s in
      ("## SF Wan Window Planner", "## 映射表", "## 警告", "## 参数设置建议")))
check("plan(): 表行 Markdown", "| 0 | 0-20 | 1-81 | 0 | 0 |" in report)
check("plan(): 4 窗无回绕警告", "轮回" not in report)
report3 = node.plan(241, 81, 16, "static", 3, 3)[0]
check("plan(): N=3 报回绕", "轮回" in report3)
ok_rows = [{"order": 0, "latent": [0, 20], "real": [1, 81], "region": 0, "slot": 0},
           {"order": 1, "latent": [21, 41], "real": [85, 165], "region": 1, "slot": 1},
           {"order": 2, "latent": [42, 60], "real": [169, 241], "region": 2, "slot": 2}]
ok_rep = nmod.render_report(ok_rows, [], {"total": 61, "length": 21, "overlap": 0,
                                          "schedule": "static", "n_slots": 3, "n_conds": 3,
                                          "total_real": 241, "length_real": 81,
                                          "overlap_real": 0}, [])
check("render: 全对齐 ok 行", "1:1:1" in ok_rep)
check("render: 表头列齐", all(k in ok_rep for k in ("latent", "实帧", "cond 段", "slot")))

# ── 建议引擎 ─────────────────────────────────────────────────────────────────
seq = nmod.flush_totals(21, 4)
check("flush: 前四档 81/149/217/285",
      [c["total_real"] for c in seq[:4]] == [81, 149, 217, 285])
check("flush: 窗数标注", [c["windows"] for c in seq[:4]] == [1, 2, 3, 4])
tips = nmod.suggest_settings(61, 21, 4, "static", 4, 3, 4)
tips_text = "\n".join(tips)
check("tips: 非齐平总数给两档", "217" in tips_text and "285" in tips_text)
check("tips: 槽数已对", "槽数：4" in tips_text and "✓" in tips_text)
tips_ok = nmod.suggest_settings(55, 21, 4, "static", 3, 3, 3)
check("tips: 齐平总数打勾", any("217" in t and "✓" in t for t in tips_ok))
tips_n = nmod.suggest_settings(61, 21, 4, "static", 3, 3, 4)
check("tips: 槽数不对给值", any("建议 4" in t for t in tips_n))
tips_u = nmod.suggest_settings(61, 21, 4, "uniform", 4, 3, 4)
check("tips: uniform 给漂移", any("漂移" in t for t in tips_u))
check("tips: 每项一行", all(isinstance(t, str) and t for t in tips))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
