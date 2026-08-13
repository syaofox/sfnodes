# SFRegionalLoRA 纯逻辑引擎测试（Python 直接运行：python tests/test_regional_engine.py）
# 覆盖：
#   - normalize_key：kohya/diffusers/lora_unet_/transformer. 前缀剥除、kohya_xx/ 组织前缀、
#     去符号碰撞（双向匹配同一层）
#   - parse_lora_sd：down/up 配对、alpha 提取与 rank 兜底、缺因子跳过、无 base 跳过
#   - lora_scale：alpha/rank、rank 缺失兜底
#   - parse_regions：默认等分 box、越界 clamp、反向 box 拒绝、退化 box 回退、
#     enable 字符串防御、损坏 JSON 兜底
#   - 网格与 mask：token_grid（Krea2 f8+patch2 = latent//2）、羽化矩形行主序、
#     active_token_indices 稀疏选择 + [text|image] 尾部偏移、全序列回退分支
#   - plan_layer_map：sig 合并 + per-region 匹配计数（"region 2 匹配 0 层"诊断回归锁）
#   - render_preview：形状、区域色、重叠取 max
import importlib.util
import os
import sys

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


spec = importlib.util.spec_from_file_location(
    "sf_utils_regional_engine", os.path.join(root, "sf_utils", "regional_engine.py"))
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

# ── 1. normalize_key ──────────────────────────────────────────────────────
check("norm: kohya prefix", eng.normalize_key("lora_unet_blocks.0.attn.wq")
      == eng.normalize_key("blocks.0.attn.wq"))
check("norm: diffusers transformer.", eng.normalize_key("transformer.blocks.0.attn.wq")
      == "blocks0attnwq")
check("norm: model.diffusion_model. double strip",
      eng.normalize_key("model.diffusion_model.blocks.0.attn.wq") == "blocks0attnwq")
check("norm: kohya_xx/ org prefix",
      eng.normalize_key("kohya_xx/lora_unet_blocks.0.mlp.gate") == "blocks0mlpgate")
check("norm: lora_te_ stripped (may just not match model)",
      eng.normalize_key("lora_te_blocks.0.attn.wq") == "blocks0attnwq")
check("norm: digits survive", eng.normalize_key("blocks.10.attn.wq") != eng.normalize_key("blocks.1.attn.wq"))

# ── 2. parse_lora_sd ──────────────────────────────────────────────────────
def make_sd(seed=0, rank=4, inn=8, outd=8, prefix="lora_unet_", style="kohya"):
    rng = np.random.default_rng(seed)
    sd = {}
    for layer in ("blocks.0.attn.wq", "blocks.0.attn.wk", "blocks.0.mlp.gate"):
        base = prefix + layer
        if style == "kohya":
            sd[base + ".lora_down.weight"] = rng.normal(size=(rank, inn)).astype(np.float32)
            sd[base + ".lora_up.weight"] = rng.normal(size=(outd, rank)).astype(np.float32)
        else:
            sd[base + ".lora_A.weight"] = rng.normal(size=(rank, inn)).astype(np.float32)
            sd[base + ".lora_B.weight"] = rng.normal(size=(outd, rank)).astype(np.float32)
        sd[base + ".alpha"] = np.float32(8.0)
    return sd

parsed = eng.parse_lora_sd(make_sd())
check("parse: 3 layers classified", len(parsed) == 3)
check("parse: sig normalized", set(parsed) == {"blocks0attnwq", "blocks0attnwk", "blocks0mlpgate"})
check("parse: alpha read", all(abs(v["alpha"] - 8.0) < 1e-6 for v in parsed.values()))
check("parse: rank read", all(v["rank"] == 4 for v in parsed.values()))
check("parse: diffusers style", len(eng.parse_lora_sd(make_sd(style="diffusers"))) == 3)
check("parse: kohya_xx prefix", len(eng.parse_lora_sd(
    {("kohya_xx/" + k): v for k, v in make_sd().items()})) == 3)

# 缺 up 跳过 / 缺 base 跳过 / 无关键忽略
sd_missing_up = {"lora_unet_blocks.0.attn.wq.lora_down.weight": np.zeros((4, 8))}
check("parse: missing up skipped", len(eng.parse_lora_sd(sd_missing_up)) == 0)
sd_no_base = {"lora_down.weight": np.zeros((4, 8)),
              "lora_up.weight": np.zeros((8, 4))}
check("parse: empty base skipped", len(eng.parse_lora_sd(sd_no_base)) == 0)
check("parse: alpha fallback to rank",
      eng.parse_lora_sd({"lora_unet_b.lora_down.weight": np.zeros((4, 8)),
                         "lora_unet_b.lora_up.weight": np.zeros((8, 4))})["b"]["alpha"] == 4.0)

# ── 3. lora_scale ─────────────────────────────────────────────────────────
check("scale: alpha/rank", abs(eng.lora_scale({"alpha": 8.0, "rank": 4}) - 2.0) < 1e-9)
check("scale: rank zero guarded", eng.lora_scale({"alpha": 2.0, "rank": 0}) == 2.0)

# ── 4. parse_regions ──────────────────────────────────────────────────────
regs = eng.parse_regions(
    '[{"lora":"a.safetensors","strength":1.5,"enable":true,"x":0,"y":0,"w":0.5,"h":1},'
    ' {"lora":"b.safetensors","strength":0.8,"enable":false,"x":0.5,"y":0,"w":0.5,"h":1}]')
check("regions: parsed 2", len(regs) == 2)
check("regions: strength/enable", regs[0]["strength"] == 1.5 and regs[1]["enable"] is False)
check("regions: box clamp", regs[0]["box"] == (0.0, 0.0, 0.5, 1.0))
regs_out = eng.parse_regions('[{"lora":"a","x":0.8,"y":0.2,"w":0.5,"h":0.5}]')
check("regions: out-of-range clamped", regs_out[0]["box"] == (0.8, 0.2, 1.0, 0.7))
regs_def = eng.parse_regions('[{"lora":"a"}]')
check("regions: missing box -> default column", regs_def[0]["box"] == (0.0, 0.0, 1.0, 1.0))
regs_neg = eng.parse_regions('[{"lora":"a","x":0.5,"y":0,"w":-0.2,"h":1}]')
check("regions: negative w rejected", len(regs_neg) == 0)
regs_deg = eng.parse_regions('[{"lora":"a","x":0,"y":0,"w":0,"h":1}]')
check("regions: degenerate w -> default column", regs_deg[0]["box"] == (0.0, 0.0, 1.0, 1.0))
check("regions: enable string defense", not eng.parse_regions(
    '[{"lora":"a","enable":"false","x":0,"y":0,"w":0.5,"h":1}]')[0]["enable"])
check("regions: garbage json -> []", eng.parse_regions("not json {") == [])
check("regions: dict single -> list", len(eng.parse_regions('{"lora":"a"}')) == 1)
check("regions: non-dict rows skipped", len(eng.parse_regions('[1,2]')) == 0)
check("default_regions_json: 2 equal columns",
      len(eng.parse_regions(eng.default_regions_json(2))) == 2)
check("default_regions_json: valid boxes",
      eng.parse_regions(eng.default_regions_json(2))[1]["box"] == (0.5, 0.0, 1.0, 1.0))

# ── 5. 网格与 mask ────────────────────────────────────────────────────────
check("grid: krea2 f8+patch2", eng.token_grid(128, 128) == (64, 64))
# 左半/右半分离：feather 羽化下中心 token 强、box 外近零、边界平滑过渡
rows, cols = 2, 4
m_l = eng.rect_token_mask(rows, cols, (0.0, 0.0, 0.5, 1.0), 0.01)
check("mask: shape n_img", m_l.shape == (8,))
check("mask: left-center strong", m_l[5] > 0.9)
check("mask: right side zero", m_l[7] < 0.05)
m_r = eng.rect_token_mask(rows, cols, (0.5, 0.0, 1.0, 1.0), 0.01)
check("mask: right-center strong", m_r[7] > 0.9)
check("mask: left side zero", m_r[0] < 0.05)
# 大 box 中心高、四角近零（小 feather 下中心接近 1）
mc = eng.rect_token_mask(4, 8, (0.25, 0.25, 0.75, 0.75), 0.01)
check("mask: center token high", mc[2 * 8 + 4] > 0.95)
check("mask: corner token near zero", mc[0] < 0.05)
# 羽化增强 -> 边界扩散、不再与 feather=0 相同
m0 = eng.rect_token_mask(rows, cols, (0.0, 0.0, 0.5, 1.0), 0.0)
mf = eng.rect_token_mask(rows, cols, (0.0, 0.0, 0.5, 1.0), 0.5)
check("mask: feather spreads values", np.max(mf) < np.max(m0) and not np.allclose(mf, m0))
# 区域隔离回归锁（2026-08 实测教训）：羽化尾巴不得把 mask 扩散进对方区域深处。
# 旧公式 fc=feather*cols 在 64 宽网格上把左半 mask 拖到 55/64 列（87% 全图重叠），
# 两个角色 LoRA 全图混合导致区域隔离失效。过渡带必须 ≈ feather*网格尺寸。
iso_l = eng.rect_token_mask(64, 64, (0.0, 0.0, 0.5, 1.0), 0.08).reshape(64, 64)
check("mask: left mask does not reach right-deep cols", iso_l[:, 40:].max() < 0.05)
iso_r = eng.rect_token_mask(64, 64, (0.5, 0.0, 1.0, 1.0), 0.08).reshape(64, 64)
check("mask: right mask does not reach left-deep cols", iso_r[:, :24].max() < 0.05)
# 稀疏选择：[text | image] 布局，图像 token 在尾部 offset = seq - n_img
idx, w = eng.active_token_indices(m_l, 0.01, seq=12, n_img=8)
keep = np.nonzero(np.abs(m_l) > 0.01)[0]
check("tokens: tail offset", np.array_equal(idx, keep + 4))
check("tokens: weight matches mask", np.allclose(w, m_l[keep], atol=1e-6))
idx_all, w_all = eng.active_token_indices(m_l, 0.01, seq=4, n_img=8)
check("tokens: fallback whole-seq when n_img > seq", len(idx_all) == 4
      and np.allclose(w_all, m_l.mean(), atol=1e-6))
idx_empty, w_empty = eng.active_token_indices(np.zeros(8), 0.01, seq=12, n_img=8)
check("tokens: empty mask -> empty idx", idx_empty.size == 0 and w_empty.size == 0)

# ── 6. plan_layer_map（诊断回归锁）───────────────────────────────────────
sig_a = {"blocks0attnwq", "blocks0attnwk", "blocks0mlpgate"}
sig_b = {"blocks0attnwq", "blocks0attnwk", "blocks0mlpgate"}
plan, matched = eng.plan_layer_map([{s: {} for s in sig_a}, {s: {} for s in sig_b}], sig_a)
check("plan: both regions per sig", all(v == {0, 1} for v in plan.values()))
check("plan: both matched 3/3", matched == [3, 3])

# 核心诊断：region 2 的 LoRA 是不同架构（sig 匹配 0 层）
plan2, matched2 = eng.plan_layer_map(
    [{s: {} for s in sig_a}, {s: {} for s in {"blocks0attn1toq", "blocks0attn1tok"}}], sig_a)
check("plan: region2 0-layer diagnostic", matched2 == [3, 0])
check("plan: only region0 sigs planned", sorted(plan2) == sorted(sig_a)
      and all(v == {0} for v in plan2.values()))

# 部分重叠：region 2 只覆盖 1 个 sig
plan3, matched3 = eng.plan_layer_map(
    [{s: {} for s in sig_a}, {"blocks0attnwq": {}}], sig_a)
check("plan: partial overlap counts", matched3 == [3, 1])

# ── 7. render_preview ─────────────────────────────────────────────────────
pv = eng.render_preview([(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)], 64, 64)
check("preview: shape [1,h,w,3]", pv.shape == (1, 64, 64, 3))
check("preview: dtype float32", pv.dtype == np.float32)
check("preview: both halves colored",
      pv[0, 8, 8].sum() > 0 and pv[0, 8, 40].sum() > 0)
check("preview: values in 0-1", pv.min() >= 0.0 and pv.max() <= 1.0)
pv1 = eng.render_preview([(0.0, 0.0, 1.0, 1.0)], 32, 32)
check("preview: single box fills", pv1[0].sum() > 0)
pv_empty = eng.render_preview([], 32, 32)
check("preview: empty boxes -> black", pv_empty.max() == 0.0)

# ── 8. 重叠归一化（normalize_overlap）─────────────────────────────────────
m_full = eng.rect_token_mask(4, 8, (0.0, 0.0, 1.0, 1.0), 0.0)
# 完全重叠的两个框：内部各减半（总和封顶 1）；羽化边缘 total<=1 不归一
# （两个半值叠加成满幅）——恒等式 ov0+ov1 = min(2m, 1)
ov = eng.normalize_overlap([m_full, m_full])
check("overlap: fully overlapping sum caps at 1",
      np.allclose(ov[0] + ov[1], np.minimum(2 * m_full, 1.0), atol=1e-6))
check("overlap: interior halves", np.allclose(ov[0][20], m_full[20] / 2, atol=1e-6))
# 不重叠的左右两框：完全不变（非重叠区不受影响）
ml = eng.rect_token_mask(4, 8, (0.0, 0.0, 0.5, 1.0), 0.02)
mr = eng.rect_token_mask(4, 8, (0.5, 0.0, 1.0, 1.0), 0.02)
ov2 = eng.normalize_overlap([ml, mr])
check("overlap: disjoint boxes unchanged", np.allclose(ov2[0], ml, atol=1e-6)
      and np.allclose(ov2[1], mr, atol=1e-6))
# 部分重叠：共享区总和压到 1 且按 mask 比例分配；非重叠区不变
m_a = eng.rect_token_mask(4, 8, (0.0, 0.0, 0.75, 1.0), 0.02)
m_b = eng.rect_token_mask(4, 8, (0.25, 0.0, 1.0, 1.0), 0.02)
ov3 = eng.normalize_overlap([m_a, m_b])
total3 = ov3[0] + ov3[1]
check("overlap: partial sum never exceeds 1", total3.max() <= 1.0 + 1e-6)
check("overlap: left-only zone unchanged", np.allclose(ov3[0][:2], m_a[:2], atol=1e-6))
shared = (m_a > 0.5) & (m_b > 0.5)
check("overlap: shared zone sum capped at 1",
      np.allclose(total3[shared], 1.0, atol=1e-6) and shared.any())
check("overlap: shared zone splits proportionally",
      np.allclose(ov3[0][shared] / np.maximum(ov3[1][shared], 1e-9),
                  m_a[shared] / np.maximum(m_b[shared], 1e-9), atol=1e-5))
# 单框：恒不变
ov1 = eng.normalize_overlap([m_a])
check("overlap: single mask unchanged", np.allclose(ov1[0], m_a, atol=1e-6))
# 空列表
check("overlap: empty list passthrough", eng.normalize_overlap([]) == [])

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
