// ==========================================================================
// sf_brush_poly.js - 画笔节点共享「多边形套索」（两个 brush 节点单源，§101）
// ==========================================================================
//
// 复刻 PS 多边形套索：多次左键落顶点，点首顶点 / 双击 / Enter 闭合后整体
// 涂抹——Brush 模式 = fill 添加，Eraser 模式 = fill_erase 打洞减去（与 AI
// 识别结果同一条「模式即运算」语义，见 §99）。合体节点 Crop 模式下入口
// 按钮由宿主自动切 Brush（Poly 是画笔工具）。
//
// 复用（禁止内联副本）：
//   - 顶点纯逻辑/笔触构造：sf_brush_mask_lib（polyCanClose / polyShouldAppend /
//     hitPolyFirst / polyStrokeForMode，可 .mjs 直测）
//   - 后端栅格化与前端预览：既有 fill / fill_erase 模式（sf_utils/brush_mask.py
//     + sf_brush_mask_lib.paintStrokeMask/drawStrokePath），零新增模式
//   - 宿主 cfg：与 sf_brush_ai 共用 AI_CFG（getState/patchState/addStrokes/
//     toImage/fromImage/inDisplay/displayOrigin，另需 toSource/cancelPoly）
//   - 线宽/顶点尺寸/提示：sf_common（getSfAccent/sfFrameWidth/sfPolyVertexSize/
//     sfToast；顶点标记尺寸走设置 sfnodes.Canvas.PolyVertexSize，默认 2）
//
// 会话 node._sfBrushPoly 为内存态（不随工作流保存；闭合前不影响输出）：
//   { points: [[x, y], ...]（源图像素 float，落点已按宿主规则钳制/拒绝）,
//     cursor: [lx, ly] | null }（光标局部坐标，画橡皮筋用）
// 键盘（Esc/Enter/Backspace/Delete）仅在 Poly 开启期由模块级 capture 监听
// 消费，与 sf_brush_ai 的 SAM 模式监听互斥（开一方关另一方：本模块开启时
// 由宿主 cancelSamMode；beginSamMode 经 cfg.cancelPoly 关本模块）。
// ==========================================================================

import { app } from "/scripts/app.js";
import { getSfAccent, sfToast, sfFrameWidth, sfPolyVertexSize } from "./sf_common.js";
import { polyStrokeForMode, polyCanClose, polyShouldAppend, hitPolyFirst } from "./sf_brush_mask_lib.js";

export const POLY_SNAP_PX = 10;   // 首顶点闭合命中半径（显示像素）
export const POLY_MIN_DIST = 1;   // 相邻顶点最小间距（源图像素，防双击重复落点）

let _activePolyNode = null;       // 键盘监听归属（仅一个节点的会话在跑）
let _polyKeyHandler = null;

function redraw() {
  if (app.graph) app.graph.setDirtyCanvas(true, true);
}

// 是否存在进行中的顶点会话（未闭合）
export function polyActive(node) {
  return !!(node && node._sfBrushPoly);
}

// 是否处于可落点状态：Poly 开 + 非 Crop 模式（合体节点）+ SAM 模式未激活
function polyReady(cfg, node) {
  const st = cfg.getState(node) || {};
  return !!st.brush_poly && st.brush_mode !== "crop" && !node._sfAiSam;
}

// 取消进行中的多边形（保留 Poly 开关状态）；返回是否有会话被取消
export function cancelPoly(node) {
  if (!node || !node._sfBrushPoly) return false;
  node._sfBrushPoly = null;
  redraw();
  return true;
}

function removeKeys() {
  if (_polyKeyHandler) {
    window.removeEventListener("keydown", _polyKeyHandler, true);
    _polyKeyHandler = null;
  }
}

// 卸载（宿主 onRemoved）：清会话 + 解绑键盘监听
export function disposePoly(node) {
  cancelPoly(node);
  if (_activePolyNode === node) {
    _activePolyNode = null;
    removeKeys();
  }
}

// 闭合提交：按当前模式生成 fill（添加）/ fill_erase（打洞）笔触并入统一列表
//（Undo/Clear/Invert/lean 注入全通用），清会话但保留开关（可连续套索）。
function commitPoly(cfg, node) {
  const ses = node._sfBrushPoly;
  if (!ses) return false;
  const stroke = polyStrokeForMode(cfg.getState(node).brush_mode, ses.points);
  node._sfBrushPoly = null;
  cfg.addStrokes(node, [stroke], {});
  sfToast({
    summary: cfg.toastTag,
    detail: `${stroke.mode === "fill_erase" ? "已减去" : "已添加"}多边形区域（${stroke.points.length} 顶点），可撤销`,
    severity: "success", fallbackTag: cfg.toastTag,
  });
  redraw();
  return true;
}

// 键盘监听：仅 Poly 开启期安装（宿主按钮开关触发；关闭/节点删除卸载）。
// Esc/Enter/Backspace 在模式激活期消费（否则会触发前端删节点等默认行为），
// 输入框内一律放行（与 sf_brush_ai SAM 模式同款守卫）。
function installKeys(cfg, node) {
  // 键盘归属只允许一个节点：抢占用旧节点的会话一并丢弃（避免其无键盘可用的半残状态）
  if (_activePolyNode && _activePolyNode !== node) cancelPoly(_activePolyNode);
  removeKeys();
  _activePolyNode = node;
  _polyKeyHandler = (e) => {
    if (_activePolyNode !== node || !polyReady(cfg, node)) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    if (e.key === "Enter") {
      const ses = node._sfBrushPoly;
      if (!ses) return; // 无会话不消费（放行前端 Enter 队列）
      e.preventDefault();
      e.stopPropagation();
      if (!polyCanClose(ses.points)) {
        sfToast({ summary: cfg.toastTag, detail: "多边形至少 3 个顶点才能闭合", severity: "warn", fallbackTag: cfg.toastTag });
        return;
      }
      commitPoly(cfg, node);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      if (!cancelPoly(node)) togglePoly(cfg, node); // 无进行中会话 → 关闭工具
    } else if (e.key === "Backspace" || e.key === "Delete") {
      const ses = node._sfBrushPoly;
      if (!ses || ses.points.length === 0) return;
      e.preventDefault();
      e.stopPropagation();
      ses.points.pop();
      redraw();
    }
  };
  window.addEventListener("keydown", _polyKeyHandler, true);
}

// togglePoly(cfg, node)：Poly 开关（面板按钮统一入口）。
// 关闭：丢弃未闭合会话 + 卸载键盘；开启：安装键盘（SAM 互斥由宿主 buttonAction
// 先 cancelSamMode，beginSamMode 反向经 cfg.cancelPoly 关本模块）。
export function togglePoly(cfg, node) {
  const st = cfg.getState(node) || {};
  const next = !st.brush_poly;
  if (!next) disposePoly(node);
  cfg.patchState(node, { brush_poly: next });
  if (next) installKeys(cfg, node);
  sfToast({
    summary: cfg.toastTag,
    detail: next
      ? "多边形套索已开启：左键落点，点首点/双击/Enter 闭合（Brush=添加 / Erase=减去），Esc 取消"
      : "多边形套索已关闭",
    severity: "info", life: 6000, fallbackTag: cfg.toastTag,
  });
}

// 节点鼠标处理插桩：phase = "down" | "move"；返回 true = 已消费。
// 与 sf_brush_ai.handleSamPointer 并列由宿主调用（SAM 优先）。
export function handlePolyPointer(cfg, node, phase, e, lp) {
  const st = cfg.getState(node) || {};
  if (!st.brush_poly || st.brush_mode === "crop" || node._sfAiSam) return false;
  if (phase === "down") {
    // 右键取消进行中会话（无会话时不消费，交还右键菜单）
    if (e && e.button === 2) return cancelPoly(node);
    if (e && e.button !== 0 && e.button !== undefined) return false;
    if (!cfg.inDisplay(node, lp[0], lp[1])) return false;
    const p = cfg.toSource ? cfg.toSource(node, lp[0], lp[1]) : cfg.toImage(node, lp[0], lp[1]);
    if (!p) return false; // 宿主判定落点无效（合体节点扩展区）
    if (!polyActive(node)) {
      node._sfBrushPoly = { points: [], cursor: [lp[0], lp[1]] };
      if (_activePolyNode !== node) installKeys(cfg, node); // 会话归属即键盘归属（多节点互斥）
    }
    const ses = node._sfBrushPoly;
    ses.cursor = [lp[0], lp[1]];
    // 点首顶点闭合（≥3 顶点；首点换算到局部坐标后用显示像素容差）
    if (polyCanClose(ses.points)) {
      const f = cfg.fromImage(node, ses.points[0][0], ses.points[0][1]);
      if (hitPolyFirst([[f.x, f.y]], lp[0], lp[1], POLY_SNAP_PX)) {
        commitPoly(cfg, node);
        return true;
      }
    }
    if (polyShouldAppend(ses.points, p.x, p.y, POLY_MIN_DIST)) {
      ses.points.push([p.x, p.y]);
    }
    redraw();
    return true;
  }
  if (phase === "move") {
    if (node._sfBrushPoly) node._sfBrushPoly.cursor = [lp[0], lp[1]];
    if (app.canvas?.canvas) app.canvas.canvas.style.cursor = "crosshair";
    redraw();
    return true;
  }
  return false;
}

// 双击闭合（宿主 onDblClick 调；返回 true = 已消费）。双击的第一击已作为
// 顶点落点（第二击被 polyShouldAppend 去重），<3 顶点则取消并提示。
export function handlePolyDblClick(cfg, node) {
  if (!polyReady(cfg, node)) return false;
  const ses = node._sfBrushPoly;
  if (!ses) return false;
  if (!polyCanClose(ses.points)) {
    cancelPoly(node);
    sfToast({ summary: cfg.toastTag, detail: "多边形至少 3 个顶点才能闭合（已取消）", severity: "warn", fallbackTag: cfg.toastTag });
    return true;
  }
  commitPoly(cfg, node);
  return true;
}

// 画布覆盖层：折线/橡皮筋/顶点标记/首点吸附高亮 + 顶部提示条。
// toLocal = 源图像素 → 局部坐标；anchor = 显示区左上角（提示条锚点，同
// drawSamOverlay）。Poly 开启但未落点时只画提示条。
export function drawPolyOverlay(cfg, node, ctx, toLocal, anchor) {
  const st = cfg.getState(node) || {};
  if (!st.brush_poly || st.brush_mode === "crop") return;
  const isErase = st.brush_mode === "erase";
  const color = isErase ? "rgba(255,255,255,0.95)" : (getSfAccent() || "rgba(100,150,255,0.9)");
  const ses = node._sfBrushPoly;

  if (ses && ses.points.length > 0) {
    const local = ses.points.map(([x, y]) => toLocal(x, y));
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = sfFrameWidth();
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    local.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    if (ses.cursor) ctx.lineTo(ses.cursor[0], ses.cursor[1]); // 橡皮筋
    ctx.stroke();
    ctx.setLineDash([]);
    // ≥3 顶点：末点→首点闭合提示边（淡虚线）
    if (local.length >= 3) {
      ctx.save();
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(local[local.length - 1].x, local[local.length - 1].y);
      ctx.lineTo(local[0].x, local[0].y);
      ctx.stroke();
      ctx.restore();
    }
    // 顶点（首点单独画：靠近可闭合时放大 + 变绿提示）。边长走设置
    // sfnodes.Canvas.PolyVertexSize（默认 2，原内联 4），首点圆半径 = 边长/2+1
    const vs = sfPolyVertexSize();
    for (let i = 1; i < local.length; i++) {
      ctx.fillStyle = color;
      ctx.fillRect(local[i].x - vs / 2, local[i].y - vs / 2, vs, vs);
    }
    const first = local[0];
    const nearFirst = ses.cursor && hitPolyFirst([[first.x, first.y]], ses.cursor[0], ses.cursor[1], POLY_SNAP_PX);
    const firstR = vs / 2 + 1;
    ctx.lineWidth = sfFrameWidth();
    ctx.strokeStyle = nearFirst ? "rgba(120,255,120,0.95)" : "rgba(0,0,0,0.8)";
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(first.x, first.y, nearFirst ? firstR + 2 : firstR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  if (anchor) {
    const n = ses ? ses.points.length : 0;
    const hint = `Poly: 左键落点（${n}）· 点首点/双击/Enter 闭合 · Backspace 删点 · Esc 取消`;
    ctx.save();
    ctx.font = "11px Arial";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const w = ctx.measureText(hint).width + 12;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(anchor.x - 2, anchor.y - 2, w, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(hint, anchor.x + 4, anchor.y + 7);
    ctx.restore();
  }
}
