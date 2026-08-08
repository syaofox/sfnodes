// ==========================================================================
// sf_outpaint_core.js - SF Image Outpaint 纯函数库（数学 + 状态）
// ==========================================================================
//
// 无 app/DOM 依赖，供主扩展 sf_outpaint.js import，也供 tests/ 复制为 .mjs
// 直接测试。镜像 nodes/image/outpaint.py——Python 是权威：两边不一致时预览
// 就会对结果说谎。
//
// Python 侧最终尺寸经由 sf_utils/resize_engine 的 _apply_pad 与 _apply_max_mp
// 得到，所以下面的助手镜像这两个（外加 _apply_snap、_clamp_dims 与舍入）
// 而不只是 outpaint.py 本身。任何公式改动后必须重跑
// tests/test_outpaint_js.js（与 Python 交叉断言）。
//
// 状态存 node.properties.outpaintState（JSON 字符串），提交时由主扩展注入
// 隐藏输入 SFOutpaintState。
// ==========================================================================

export const STATE_PROP = "outpaintState";
export const STATE_VERSION = 1;
// 百万像素按钮的默认集（本版无设置面板，固定默认列表，0 = 关/不缩放）。
export const LIMITS = [0, 1, 1.5, 2];
export const MAX_MP = 64;      // _apply_max_mp 的上限，镜像 Python
export const SNAPS = [0, 8, 16, 32, 64];
export const DEFAULT_RATIOS = ["1:1", "4:5", "3:2", "16:9", "9:16"];

// 镜像 outpaint.py 的 _MAX_PAD（解析状态时每个单边值都被它夹紧）。导出以便
// UI 把自己的输入域夹到同一上限：允许输入超过它的字段会预览一个 Python
// 直接丢弃的 pad。
export const MAX_PAD = 8192;

export const DEFAULT_STATE = {
  version: STATE_VERSION,
  mode: "ratio",
  ratio: "3:2",
  anchor: "centre",
  top: 0, bottom: 0, left: 0, right: 0,
  limit: 0,
  // 中灰而非绿色。训练在绿色填充上的 LoRA 会把颜色连同形状一起学会，生成的
  // 整张图都带绿色色偏（2026-07-17 实际使用反馈）。中性灰没有色相可渗。
  // 必须与 nodes/image/outpaint.py 的 DEFAULT_STATE 一致——节点读 Python 的副本。
  color: "#808080",
  snap: 0,
  collapsed: false,
};

// Python 的 float() 严格而 JS 的 parseFloat() 宽松：float("2abc") 抛错而
// parseFloat("2abc") 返回 2。没有这道闸，"16:9:1" 和 "16:9abc" 会预览一个
// Python 拒绝生成的 pad。
const FINITE_NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function strictFloat(text) {
  const t = String(text).trim();
  return FINITE_NUMBER.test(t) ? parseFloat(t) : NaN;
}

export function parseRatio(text) {
  if (typeof text !== "string" || !text.includes(":")) return null;
  // 只按第一个冒号切分，镜像 Python 的 str.partition(":")，"3:2:5" 会把
  // "2:5" 交给数字闸门、两侧同时拒绝。
  const at = text.indexOf(":");
  const rw = strictFloat(text.slice(0, at));
  const rh = strictFloat(text.slice(at + 1));
  if (!isFinite(rw) || !isFinite(rh) || rw <= 0 || rh <= 0) return null;
  return [rw, rh];
}

// 所选比例会增长哪个轴。一次比例选择永远只长一个轴，所以 anchor 行是三枚
// 芯片而不是 3x3 网格。"h" = 更宽（左/中/右），"v" = 更高（上/中/下），
// null = 无增长，anchor 行无事可做。
export function anchorAxis(ratioText, srcW, srcH) {
  const r = parseRatio(ratioText);
  if (!r || !srcW || !srcH) return null;
  const target = r[0] / r[1];
  const cur = srcW / srcH;
  if (Math.abs(target - cur) < 1e-6) return null;
  return target > cur ? "h" : "v";
}

// 空原型，手改的 anchor 到不了 Object.prototype：普通字面量会让
// remapAnchor("constructor", "v") 返回一个函数。
const H_TO_V = Object.assign(Object.create(null), { left: "top", centre: "middle", right: "bottom" });
const V_TO_H = Object.assign(Object.create(null), { top: "left", middle: "centre", bottom: "right" });

// 活动轴翻转时保留用户的意图："贴远边"保持"贴远边"而非弹回中间。
export function remapAnchor(anchor, toAxis) {
  if (toAxis === "v") return H_TO_V[anchor] ?? (V_TO_H[anchor] ? anchor : "middle");
  if (toAxis === "h") return V_TO_H[anchor] ?? (H_TO_V[anchor] ? anchor : "centre");
  return anchor;
}

// 镜像 resize_engine._round_half_up，即 floor(x + 0.5)。Math.round 对除
// 0.49999999999999994 角落外的所有正值一致（那里 Math.round 更"正确"，但
// 不再镜像 Python）。outpaint.py 里每个尺寸都走 _round_half_up，绝不内建
// round()：内建是银行家舍入，999 高的源在 3:2（999*1.5 = 1498.5）下 Python
// 得 1498、JS 得 1499，预览会把绿色画得比真实输出窄一像素。
function roundHalfUp(x) {
  return Math.floor(x + 0.5);
}

// 镜像 resize_engine._apply_snap。FLOOR 而非就近舍入，吸附绝不会把尺寸推回
// 百万像素上限之上。下限 8（对齐 Python）而非 snap 步长：源比步长小时 Python
// 落在 8，这里 max(snap, ...) 会高报小图。
function snapTo(w, h, snap) {
  if (!snap || snap <= 0) return [w, h];
  return [Math.max(8, Math.floor(w / snap) * snap),
          Math.max(8, Math.floor(h / snap) * snap)];
}

// 镜像 resize_engine._clamp_dims：下限 8（极端吸附会把尺寸舍成零），上限
// 16384（Python 拒绝分配超过它的画布，预览承诺更多就是许一个兑现不了的诺）。
function clampDims(w, h) {
  return [Math.max(8, Math.min(Math.trunc(w), 16384)),
          Math.max(8, Math.min(Math.trunc(h), 16384))];
}

// 镜像 _apply_pad 里每边得到的 max(0, int(...))。用 Math.trunc 而非 |0，
// |0 会把大值绕回 32 位。
function padPx(v) {
  const n = Number(v);
  return isFinite(n) ? Math.max(0, Math.trunc(n)) : 0;
}

// anchor 的值命名"新空间去哪边"："right" 在右边填充。这与
// resize_engine._anchor_offsets 以及 Load Image 的裁剪 anchor（命名图片贴哪边）
// 刻意相反。原因："sides" 模式已经是每边绿色（right: 512 = 右边 512px 绿色），
// 同一个词在节点的两种模式下必须同义。UI 标签是"Add space"而非"Anchor"，
// 避免两个概念混淆。必须与 outpaint.py 的 _pads_for_ratio 完全一致——不一致
// 时实时预览把绿色画在错误的边上。不要把它"纠正"回 _anchor_offsets 的约定。
export function padsForRatio(srcW, srcH, ratioText, anchor) {
  const none = { top: 0, bottom: 0, left: 0, right: 0 };
  const axis = anchorAxis(ratioText, srcW, srcH);
  if (!axis) return none;
  const r = parseRatio(ratioText);
  const target = r[0] / r[1];

  // 两个轴的名字都接受，另一轴遗留的存储 anchor 仍按近/远读取而非静默居中。
  if (axis === "h") {
    const add = roundHalfUp(srcH * target) - srcW;
    if (add <= 0) return none;
    if (anchor === "left" || anchor === "top") return { ...none, left: add };
    if (anchor === "right" || anchor === "bottom") return { ...none, right: add };
    const half = Math.floor(add / 2);
    return { ...none, left: half, right: add - half };
  }
  const add = roundHalfUp(srcW / target) - srcH;
  if (add <= 0) return none;
  if (anchor === "top" || anchor === "left") return { ...none, top: add };
  if (anchor === "bottom" || anchor === "right") return { ...none, bottom: add };
  const half = Math.floor(add / 2);
  return { ...none, top: half, bottom: add - half };
}

// 镜像 _parse_state 的每边强制：max(0, min(int(v), _MAX_PAD))。这是与 padPx
// 分开的另一个夹紧，两者都需要：Python 在状态进入时夹紧（这里，到 _MAX_PAD），
// 在 pad 输出时夹紧（padPx，到 >= 0）。跳过这个的预览会高兴地画一条运行即将
// 裁到 8192 的 99999px 色带。
export function sidePad(v) {
  const n = Number(v);
  return isFinite(n) ? Math.max(0, Math.min(Math.trunc(n), MAX_PAD)) : 0;
}

// 一次运行将施加的四边 pad，无论哪个模式。镜像 outpaint() 自己的分派：
// ratio 模式按形状推导，By side 从状态读取。放这里而不是 UI 层，因为它是
// 决定绿色去哪里的东西——与 Python 漂移，预览就说谎。
export function padsForState(st, srcW, srcH) {
  if (st && st.mode === "ratio") return padsForRatio(srcW, srcH, st.ratio, st.anchor);
  return {
    top: sidePad(st && st.top), bottom: sidePad(st && st.bottom),
    left: sidePad(st && st.left), right: sidePad(st && st.right),
  };
}

// 镜像 outpaint()：先 pad，设了 limit 再封顶。二进制 MP（1024*1024），对齐
// ComfyUI 的 ImageScaleToTotalPixels 与 _apply_max_mp。
//
// Snap 恰好触发一次，与节点安排一致：开 limit 时 pad 过程不吸附、百万像素
// 过程吸附；否则 pad 过程吸附。clamp 夹在两次过程之间，因为 Python 在
// _apply_pad 内部夹紧，过大的 pad 在百万像素系数对着它测量之前就被封顶。
export function finalSize(srcW, srcH, pads, limit, snap) {
  // 与 Python _parse_state 完全相同的强制转换，预览绝不因怪类型值与运行
  // 分歧：limit 是 [0, MAX_MP] 内任意有限值否则关（Number() 让字符串 "0"
  // 读作 0 而非 truthy "开"）；snap 必须是固定步长之一否则无吸附。
  const lim = Number(limit);
  const okLim = isFinite(lim) && lim > 0 && lim <= MAX_MP;
  const sn = SNAPS.includes(Number(snap)) ? Number(snap) : 0;

  let w = srcW + padPx(pads?.left) + padPx(pads?.right);
  let h = srcH + padPx(pads?.top) + padPx(pads?.bottom);

  [w, h] = snapTo(w, h, okLim ? 0 : sn);
  [w, h] = clampDims(w, h);
  if (!okLim) return { w, h };

  const target = Math.max(0.01, Math.min(lim, MAX_MP));
  const targetPx = target * 1024 * 1024;
  const cur = w * h;
  let factor = cur > 0 ? Math.sqrt(targetPx / cur) : 1;
  factor = Math.min(factor, 8);
  w = roundHalfUp(w * factor);
  h = roundHalfUp(h * factor);
  [w, h] = snapTo(w, h, sn);
  [w, h] = clampDims(w, h);
  return { w, h };
}

export function readState(node) {
  let raw = node?.properties?.[STATE_PROP];
  if (typeof raw === "string") {
    try { raw = JSON.parse(raw); } catch { raw = null; }
  }
  return { ...DEFAULT_STATE, ...(raw && typeof raw === "object" ? raw : {}) };
}

export function writeState(node, patch) {
  const next = { ...readState(node), ...patch };
  if (!node.properties) node.properties = {};
  node.properties[STATE_PROP] = JSON.stringify(next);
  return next;
}
