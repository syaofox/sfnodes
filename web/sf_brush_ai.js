// ==========================================================================
// sf_brush_ai.js - 画笔节点共享「AI / 工具」右键菜单（两个 brush 节点单源）
// ==========================================================================
//
// 原 sf_brush_sam.js 泛化（SFImageBrushMask / SFImageCropExpandBrushMask 共用）：
//   菜单 8 项：
//     · SAM 蒙版：文本选择…（对话框：prompt/threshold/refine）
//     · SAM 点选分割（左键=正点 / Shift+左键=负点，Enter 执行、Esc 取消）
//     · SAM 框选分割（图上拖框，松开执行）
//     · 人物部位遮罩…（MediaPipe：脸/发/身体/衣服/背景）
//     · YOLO 检测/分割…（models/ultralytics/{bbox,segm} 权重，框/椭圆或掩码）
//     · 导入遮罩文件为笔触…
//     · ✓ 反选遮罩（invert 状态位）
//     · 卸载 AI 模型（SAM/人物/YOLO）
//
// 后端 nodes/image/brush_mask_sam.py + brush_mask_tools.py（按 src_path 通用）。
// 忙时熔断（§91）：工作流执行期间后端 409（aimdo 全局 file reader 与执行线程
// 并发模型加载会撞 slot）；本模块先 GET sam_status 预检 busy 并 warn，服务端
// 409 仍兜底。
//
// cfg: {
//   toastTag, logTag,
//   getState(node),                   宿主状态读取,
//   patchState(node, patch),          写状态 + 重绘（反选等）,
//   addStrokes(node, incoming, extra), 并入 fill 笔触 + extra 展开进 state
//                                     （菜单参数记忆；incoming 为空也写 extra）,
//   toImage(node, lx, ly) -> {x,y},   节点局部 → 源图像素坐标,
//   fromImage(node, x, y) -> {x,y},   源图像素 → 节点局部,
//   inDisplay(node, lx, ly) -> bool,  局部坐标是否落在显示区,
//   displayOrigin(node) -> {x,y},     显示区左上角（点/框模式提示条锚点）,
// }
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { sfToast, sfApiUrl } from "./sf_common.js";
import { CropAPI } from "./sf_crop_core.js";
import { readFileAsDataURL } from "./sf_crop_source.js";

const SAM_RUN = "/api/sfnodes/brush_mask/sam";
const SAM_STATUS = "/api/sfnodes/brush_mask/sam_status";
const PERSON_RUN = "/api/sfnodes/brush_mask/person_parts";
const YOLO_LIST = "/api/sfnodes/brush_mask/yolo_models";
const YOLO_CLASSES = "/api/sfnodes/brush_mask/yolo_classes";
const YOLO_RUN = "/api/sfnodes/brush_mask/yolo";
const IMPORT_MASK = "/api/sfnodes/brush_mask/import_mask";
const UNLOAD_ALL = "/api/sfnodes/brush_mask/unload_all";

const _BTN = "padding:5px 12px;border:none;border-radius:3px;cursor:pointer;font-size:12px;";
const _INPUT = "padding:5px;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;color:var(--sf-text);font-size:13px;box-sizing:border-box;";
const _LABEL = "color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;";
const _FIELDLABEL = "color:var(--sf-text-dim);font-size:10px;margin-bottom:3px;";

// ── 请求层 ────────────────────────────────────────────────────────────────

// POST 路由：失败时抛错并携带 status/payload（409 忙时熔断走 warn 提示）
export async function aiPost(path, body) {
  const res = await api.fetchApi(sfApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = null;
  try { data = await res.json(); } catch { /* 非 JSON 回退 */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || `HTTP ${res.status}`);
    err.status = res.status;
    err.payload = data;
    throw err;
  }
  return data || {};
}

async function aiGet(path) {
  const res = await api.fetchApi(sfApiUrl(path), { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

// 预检队列忙闲；查询失败视为不忙（服务端 409 兜底）
async function aiBusy() {
  try {
    const data = await aiGet(SAM_STATUS);
    return !!data.busy;
  } catch {
    return false;
  }
}

// 通用 AI 请求：预先检（缺源/忙）→ 推理中 toast → POST → onSuccess
async function runAiRequest(cfg, node, path, body, opts) {
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: cfg.toastTag, detail: "先加载源图再使用 AI 菜单", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  if (await aiBusy()) {
    sfToast({
      summary: cfg.toastTag,
      detail: "ComfyUI 正在执行工作流（SAM/模型加载中），为避免冲突，请等当前任务结束后再试",
      severity: "warn", life: 6000, fallbackTag: cfg.toastTag,
    });
    return;
  }
  sfToast({ summary: cfg.toastTag, detail: opts.running, severity: "info", life: 5000, fallbackTag: cfg.toastTag });
  try {
    const data = await aiPost(path, body);
    if (opts.onData) opts.onData(data);
    mergeStrokes(cfg, node, data, opts.extra || {}, opts);
  } catch (err) {
    console.error(`${cfg.logTag} ${opts.logKind || "ai"} failed:`, err);
    if (err && err.status === 409) {
      sfToast({ summary: cfg.toastTag, detail: (err && err.message) || "工作流运行中，请稍后再试", severity: "warn", life: 6000, fallbackTag: cfg.toastTag });
      return;
    }
    sfToast({ summary: cfg.toastTag, detail: `${opts.failPrefix || "AI 失败"}：${(err && err.message) || err}`, severity: "error", life: 6000, fallbackTag: cfg.toastTag });
  }
}

// 结果笔触并入 + 反馈（opts.mergedPrefix / emptyMsg 可定制）
function mergeStrokes(cfg, node, data, extra, opts) {
  const incoming = Array.isArray(data && data.strokes) ? data.strokes : [];
  if (!incoming.length) {
    cfg.addStrokes(node, [], extra || {});
    sfToast({ summary: cfg.toastTag, detail: opts.emptyMsg || "未检出目标（空结果，笔触不变）", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  cfg.addStrokes(node, incoming, extra || {});
  const cov = data.coverage != null ? `覆盖 ${Math.round(data.coverage * 100)}%，` : "";
  sfToast({
    summary: cfg.toastTag,
    detail: `${opts.mergedPrefix || "并入"} ${incoming.length} 个填充笔触（${cov}可擦除/撤销）`,
    severity: "success", fallbackTag: cfg.toastTag,
  });
}

// ── 通用弹窗（三关闭：外部点击 / Esc / Cancel；输入框 Enter 执行）──────────

function createModal({ id, title, bodyHtml, okLabel = "Run", width = 300 }) {
  if (document.getElementById(id)) return null;
  const overlay = document.createElement("div");
  overlay.id = id;
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;";
  const dialog = document.createElement("div");
  dialog.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
    `background:var(--sf-panel-bg);border:1px solid var(--sf-border-soft);border-radius:6px;padding:12px 14px;` +
    `box-shadow:0 4px 20px rgba(0,0,0,0.5);width:${width}px;box-sizing:border-box;`;
  dialog.innerHTML = `
    <div style="color:var(--sf-text);font-size:13px;margin-bottom:10px;font-weight:bold;">${title}</div>
    ${bodyHtml}
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px;">
      <button class="sf-ai-cancel" style="${_BTN}background:var(--sf-surface);color:var(--sf-text);">Cancel</button>
      <button class="sf-ai-ok" style="${_BTN}background:#4a90e2;color:white;">${okLabel}</button>
    </div>`;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    overlay.remove();
  };
  overlay.addEventListener("pointerdown", (e) => { if (e.target === overlay) close(); });
  window.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { close(); window.removeEventListener("keydown", esc); }
  });
  dialog.querySelector(".sf-ai-cancel").onclick = close;
  return { dialog, close };
}

// 输入框键位：Enter 执行 / Esc 关闭；放行 ctrl/meta/alt（Ctrl+S 不能漏成浏览器行为）
function wireDialogKeys(modal, inputs, apply) {
  for (const el of inputs) {
    el.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") {
        // 吞掉回车：否则 keydown 上浮到 window 触发全局键位（用户自绑的 Enter 队列）
        e.preventDefault();
        e.stopPropagation();
        apply();
      } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        modal.close();
      }
    };
  }
}

// ── SAM 文本对话框 ────────────────────────────────────────────────────────

export async function runSamMask(cfg, node, prompt, threshold, refine) {
  await runAiRequest(cfg, node, SAM_RUN, {
    src_path: cfg.getState(node).src_path,
    prompt, threshold, refine_iterations: refine,
  }, {
    running: `SAM 推理中：${prompt || "object"}…（首次需加载 1.7GB 模型）`,
    failPrefix: "SAM 失败",
    logKind: "sam",
    extra: { sam_prompt: prompt, sam_threshold: threshold, sam_refine: refine },
    emptyMsg: "SAM 未检出目标（空结果，笔触不变）",
  });
}

export function openSamDialog(cfg, node) {
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: cfg.toastTag, detail: "先加载源图再跑 SAM", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  const lastPrompt = st.sam_prompt || "";
  const lastThr = st.sam_threshold ?? 0.5;
  const lastRefine = st.sam_refine ?? 2;
  const modal = createModal({
    id: "sf-brush-mask-sam-overlay",
    title: "SAM 蒙版：文本选择",
    width: 300,
    okLabel: "Run SAM",
    bodyHtml: `
    <div style="margin-bottom:10px;">
      <label style="${_FIELDLABEL}">Prompt（英文，如 person / car，为空按 object）</label>
      <input type="text" id="sf-bm-sam-prompt" value="${String(lastPrompt).replace(/"/g, "&quot;")}" placeholder="person" style="width:100%;${_INPUT}">
      <div style="color:var(--sf-text-faint);font-size:10px;margin-top:3px;">多人用 person:3（:N = 每类最多 N 个），多类用逗号分隔</div>
    </div>
    <div style="margin-bottom:10px;">
      <label style="${_FIELDLABEL}">Threshold（0-1，越低越多）</label>
      <input type="number" id="sf-bm-sam-thr" value="${lastThr}" min="0" max="1" step="0.05" style="width:100%;${_INPUT}">
    </div>
    <div style="margin-bottom:10px;">
      <label style="${_FIELDLABEL}">Refine（0-5，SAM 解码精修轮数，0=用粗蒙版）</label>
      <input type="number" id="sf-bm-sam-refine" value="${lastRefine}" min="0" max="5" step="1" style="width:100%;${_INPUT}">
    </div>`,
  });
  if (!modal) return;

  const promptInput = modal.dialog.querySelector("#sf-bm-sam-prompt");
  const thrInput = modal.dialog.querySelector("#sf-bm-sam-thr");
  const refineInput = modal.dialog.querySelector("#sf-bm-sam-refine");
  setTimeout(() => promptInput.focus(), 100);

  const apply = () => {
    let thr = parseFloat(thrInput.value);
    if (!Number.isFinite(thr)) thr = 0.5;
    thr = Math.max(0, Math.min(1, thr));
    let refine = parseInt(refineInput.value, 10);
    if (!Number.isFinite(refine)) refine = 2;
    refine = Math.max(0, Math.min(5, refine));
    const p = promptInput.value || "";
    modal.close();
    runSamMask(cfg, node, p, thr, refine);
  };
  modal.dialog.querySelector(".sf-ai-ok").onclick = apply;
  wireDialogKeys(modal, [promptInput, thrInput, refineInput], apply);
}

// ── SAM 点选 / 框选（画布模式）────────────────────────────────────────────

let _activeSamNode = null;
let _modeKeyHandler = null;

function clampToSource(st, p) {
  return {
    x: Math.max(0, Math.min(st.src_w - 1, p.x)),
    y: Math.max(0, Math.min(st.src_h - 1, p.y)),
  };
}

export function samModeActive(node) {
  return !!(node && node._sfAiSam);
}

export function cancelSamMode(node) {
  if (!node || !node._sfAiSam) return;
  node._sfAiSam = null;
  if (_activeSamNode === node) _activeSamNode = null;
  if (_modeKeyHandler) {
    window.removeEventListener("keydown", _modeKeyHandler, true);
    _modeKeyHandler = null;
  }
  if (app.graph) app.graph.setDirtyCanvas(true, true);
}

function runSamPromptRequest(cfg, node, body, running) {
  runAiRequest(cfg, node, SAM_RUN, body, {
    running, failPrefix: "SAM 失败", logKind: "sam",
    emptyMsg: "SAM 未检出目标（空结果，笔触不变）",
  });
}

export function beginSamMode(cfg, node, kind) {
  if (!cfg.getState(node).src_path) {
    sfToast({ summary: cfg.toastTag, detail: "先加载源图再跑 SAM", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  if (_activeSamNode && _activeSamNode !== node) cancelSamMode(_activeSamNode);
  cancelSamMode(node);
  node._sfAiSam = { kind, pos: [], neg: [], box: null, dragging: false };
  _activeSamNode = node;
  _modeKeyHandler = (e) => {
    const mode = node._sfAiSam;
    if (!mode) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    if (kind === "point" && e.key === "Enter") {
      e.preventDefault();
      e.stopPropagation();
      if (!mode.pos.length) {
        sfToast({ summary: cfg.toastTag, detail: "至少点一个正点（左键）再执行", severity: "warn", fallbackTag: cfg.toastTag });
        return;
      }
      const st = cfg.getState(node);
      const body = {
        src_path: st.src_path,
        positive_coords: JSON.stringify(mode.pos),
        refine_iterations: st.sam_refine ?? 2,
      };
      if (mode.neg.length) body.negative_coords = JSON.stringify(mode.neg);
      const n = `${mode.pos.length} 正点${mode.neg.length ? ` / ${mode.neg.length} 负点` : ""}`;
      cancelSamMode(node);
      runSamPromptRequest(cfg, node, body, `SAM 点选推理中（${n}）…`);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      cancelSamMode(node);
    } else if (kind === "point" && (e.key === "Backspace" || e.key === "Delete")) {
      // 仅模式激活期消费（否则误吞前端删除选中节点快捷键）
      e.preventDefault();
      e.stopPropagation();
      if (mode.neg.length) mode.neg.pop();
      else mode.pos.pop();
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
  };
  window.addEventListener("keydown", _modeKeyHandler, true);
  sfToast({
    summary: cfg.toastTag,
    detail: kind === "point"
      ? "SAM 点选：左键=正点，Shift+左键=负点，Enter 执行，Esc 取消"
      : "SAM 框选：在图上拖出矩形，松开执行，Esc 取消",
    severity: "info", life: 6000, fallbackTag: cfg.toastTag,
  });
  if (app.graph) app.graph.setDirtyCanvas(true, true);
}

// 节点鼠标处理插桩：phase = "down" | "move" | "up"；返回 true = 已消费
export function handleSamPointer(cfg, node, phase, e, lp) {
  const mode = node._sfAiSam;
  if (!mode) return false;
  if (phase === "down") {
    if (e && e.button === 2) { cancelSamMode(node); return true; }
    if (e && e.button !== 0 && e.button !== undefined) return false;
    if (!cfg.inDisplay(node, lp[0], lp[1])) return false; // 左侧列/底行按钮交给控件逻辑
    if (mode.kind === "point") {
      const st = cfg.getState(node);
      const p = clampToSource(st, cfg.toImage(node, lp[0], lp[1]));
      // 负点用 Shift：Alt+点击被前端抢作"克隆节点"（canvas/Vue 两种渲染模式均在
      // 派发给 node.onMouseDown 之前处理），我们的 handler 拦不到
      (e && e.shiftKey ? mode.neg : mode.pos).push({ x: Math.round(p.x), y: Math.round(p.y) });
      if (app.graph) app.graph.setDirtyCanvas(true, true);
      return true;
    }
    const p = cfg.toImage(node, lp[0], lp[1]);
    mode.dragging = true;
    mode.box = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
    return true;
  }
  if (phase === "move") {
    // 释放丢失兜底（主键已松但橡皮筋仍在拖）：按当前框立即落定
    if (mode.kind === "box" && mode.dragging
        && e && typeof e.buttons === "number" && (e.buttons & 1) === 0) {
      return finalizeSamBox(cfg, node);
    }
    if (mode.kind === "box" && mode.dragging) {
      const p = cfg.toImage(node, lp[0], lp[1]);
      mode.box.x2 = p.x;
      mode.box.y2 = p.y;
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
    if (app.canvas?.canvas) app.canvas.canvas.style.cursor = "crosshair";
    return true;
  }
  if (phase === "up") {
    if (mode.kind === "box" && mode.dragging) {
      finalizeSamBox(cfg, node);
    }
    return true;
  }
  return false;
}

// 框选落定：归一矩形（钳制源图内，至少 2×2），过小取消；成功后发起推理
function finalizeSamBox(cfg, node) {
  const mode = node._sfAiSam;
  if (!mode || mode.kind !== "box" || !mode.dragging) return false;
  mode.dragging = false;
  const st = cfg.getState(node);
  const b = mode.box || { x1: 0, y1: 0, x2: 0, y2: 0 };
  const x1 = Math.max(0, Math.min(st.src_w, Math.min(b.x1, b.x2)));
  const y1 = Math.max(0, Math.min(st.src_h, Math.min(b.y1, b.y2)));
  const x2 = Math.max(0, Math.min(st.src_w, Math.max(b.x1, b.x2)));
  const y2 = Math.max(0, Math.min(st.src_h, Math.max(b.y1, b.y2)));
  if (x2 - x1 < 2 || y2 - y1 < 2) {
    cancelSamMode(node);
    sfToast({ summary: cfg.toastTag, detail: "框太小（至少 2×2 像素），已取消", severity: "warn", fallbackTag: cfg.toastTag });
    return true;
  }
  const body = { src_path: st.src_path, bbox: [x1, y1, x2, y2], refine_iterations: st.sam_refine ?? 2 };
  cancelSamMode(node);
  runSamPromptRequest(cfg, node, body, "SAM 框选推理中…");
  return true;
}

// 画布覆盖层（点在图上 + 顶部提示条）；toLocal = 源图像素 → 局部坐标
export function drawSamOverlay(node, ctx, toLocal, anchor) {
  const mode = node._sfAiSam;
  if (!mode) return;
  const drawDot = (pt, color, label) => {
    const p = toLocal(pt.x, pt.y);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.8)";
    ctx.lineWidth = 1;
    ctx.stroke();
    if (label) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px Arial";
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(label, p.x + 6, p.y - 6);
    }
    ctx.restore();
  };
  if (mode.kind === "point") {
    mode.pos.forEach((pt, i) => drawDot(pt, "rgba(60,200,90,0.95)", String(i + 1)));
    mode.neg.forEach((pt) => drawDot(pt, "rgba(230,70,70,0.95)", "−"));
  } else if (mode.box) {
    const p1 = toLocal(Math.min(mode.box.x1, mode.box.x2), Math.min(mode.box.y1, mode.box.y2));
    const p2 = toLocal(Math.max(mode.box.x1, mode.box.x2), Math.max(mode.box.y1, mode.box.y2));
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    ctx.restore();
  }
  if (anchor) {
    const hint = mode.kind === "point"
      ? `SAM 点选：左键正点 / Shift+左键负点（${mode.pos.length}+ / ${mode.neg.length}-）· Enter 执行 · Esc 取消`
      : "SAM 框选：拖出矩形后松开执行 · Esc 取消";
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

// ── 人物部位（MediaPipe）──────────────────────────────────────────────────

export async function runPersonParts(cfg, node, parts, confidence, refine) {
  await runAiRequest(cfg, node, PERSON_RUN, {
    src_path: cfg.getState(node).src_path, parts, confidence, refine,
  }, {
    running: "人物部位分割中…（首次需下载 tflite 模型）",
    failPrefix: "人物部位分割失败",
    logKind: "person",
    extra: { person_parts: parts, person_confidence: confidence, person_refine: refine },
    mergedPrefix: "人物部位",
    emptyMsg: "未检出所选部位（空结果，笔触不变）",
  });
}

const PERSON_PARTS = [
  ["face", "Face"],
  ["hair", "Hair"],
  ["body", "Body"],
  ["clothes", "Clothes"],
  ["background", "Background"],
];

export function openPersonDialog(cfg, node) {
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: cfg.toastTag, detail: "先加载源图再跑人物部位分割", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  const cur = Array.isArray(st.person_parts) && st.person_parts.length ? st.person_parts : ["face", "hair"];
  const curConf = st.person_confidence ?? 0.4;
  const curRefine = !!st.person_refine;
  const checks = PERSON_PARTS.map(([id, label]) =>
    `<label style="color:var(--sf-text);font-size:12px;display:flex;align-items:center;gap:4px;">
       <input type="checkbox" id="sf-ai-pm-${id}" ${cur.includes(id) ? "checked" : ""}>${label}
     </label>`).join("");
  const modal = createModal({
    id: "sf-brush-mask-person-overlay",
    title: "人物部位遮罩（MediaPipe）",
    okLabel: "Run",
    bodyHtml: `
    <div style="display:flex;flex-wrap:wrap;gap:8px 12px;margin-bottom:10px;">${checks}</div>
    <div style="margin-bottom:8px;">
      <label style="${_FIELDLABEL}">Confidence（0.01-1，阈值）</label>
      <input type="number" id="sf-ai-pm-conf" value="${curConf}" min="0.01" max="1" step="0.05" style="width:100%;${_INPUT}">
    </div>
    <label style="color:var(--sf-text);font-size:12px;display:flex;align-items:center;gap:4px;">
      <input type="checkbox" id="sf-ai-pm-refine" ${curRefine ? "checked" : ""}>Refine（二次分割提升边缘质量）
    </label>`,
  });
  if (!modal) return;

  const confInput = modal.dialog.querySelector("#sf-ai-pm-conf");
  const apply = () => {
    const parts = PERSON_PARTS.map(([id]) => id)
      .filter((id) => modal.dialog.querySelector(`#sf-ai-pm-${id}`).checked);
    if (!parts.length) {
      sfToast({ summary: cfg.toastTag, detail: "至少勾选一个部位", severity: "warn", fallbackTag: cfg.toastTag });
      return;
    }
    let conf = parseFloat(confInput.value);
    if (!Number.isFinite(conf)) conf = 0.4;
    conf = Math.max(0.01, Math.min(1, conf));
    const refine = modal.dialog.querySelector("#sf-ai-pm-refine").checked;
    modal.close();
    runPersonParts(cfg, node, parts, conf, refine);
  };
  modal.dialog.querySelector(".sf-ai-ok").onclick = apply;
  wireDialogKeys(modal, [confInput], apply);
}

// ── YOLO 检测/分割 ────────────────────────────────────────────────────────

export async function runYolo(cfg, node, kind, model, conf, boxShape, imgsz, classes) {
  const body = {
    src_path: cfg.getState(node).src_path, kind, model, conf,
    box_shape: boxShape, imgsz: imgsz || 640,
  };
  const ids = Array.isArray(classes) ? classes.filter((c) => Number.isFinite(Number(c))) : [];
  if (ids.length) body.classes = ids;
  await runAiRequest(cfg, node, YOLO_RUN, body, {
    running: `YOLO 推理中（${model}）…`,
    failPrefix: "YOLO 推理失败",
    logKind: "yolo",
    extra: {
      yolo_kind: kind, yolo_model: model, yolo_conf: conf, yolo_box_shape: boxShape,
      yolo_imgsz: imgsz || 640, yolo_classes: ids,
    },
    mergedPrefix: "YOLO 并入",
    emptyMsg: "YOLO 未检出目标（空结果，笔触不变）",
    onData: (data) => {
      if (data && data.warning) {
        sfToast({ summary: cfg.toastTag, detail: data.warning, severity: "warn", life: 6000, fallbackTag: cfg.toastTag });
      }
    },
  });
}

export async function openYoloDialog(cfg, node) {
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: cfg.toastTag, detail: "先加载源图再跑 YOLO", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  const curKind = st.yolo_kind === "segm" ? "segm" : "bbox";
  const curConf = st.yolo_conf ?? 0.25;
  const curShape = st.yolo_box_shape === "ellipse" ? "ellipse" : "rect";
  const curImgsz = [640, 960, 1280].includes(Number(st.yolo_imgsz)) ? Number(st.yolo_imgsz) : 640;
  const modal = createModal({
    id: "sf-brush-mask-yolo-overlay",
    title: "YOLO 检测 / 分割",
    okLabel: "Run",
    width: 340,
    bodyHtml: `
    <div style="display:flex;gap:8px;margin-bottom:8px;">
      <div style="flex:1;">
        <label style="${_FIELDLABEL}">类型</label>
        <select id="sf-ai-yo-kind" style="width:100%;${_INPUT}">
          <option value="bbox" ${curKind === "bbox" ? "selected" : ""}>bbox（检测框）</option>
          <option value="segm" ${curKind === "segm" ? "selected" : ""}>segm（分割掩码）</option>
        </select>
      </div>
      <div style="flex:1;">
        <label style="${_FIELDLABEL}">imgsz（小目标用 960/1280）</label>
        <select id="sf-ai-yo-imgsz" style="width:100%;${_INPUT}">
          ${[640, 960, 1280].map((v) => `<option value="${v}" ${curImgsz === v ? "selected" : ""}>${v}</option>`).join("")}
        </select>
      </div>
    </div>
    <div style="margin-bottom:8px;">
      <label style="${_FIELDLABEL}">模型（ultralytics/{bbox,segm} 或 yolo）</label>
      <select id="sf-ai-yo-model" style="width:100%;${_INPUT}"></select>
      <div id="sf-ai-yo-hint" style="color:var(--sf-text-faint);font-size:10px;margin-top:3px;"></div>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:8px;">
      <div style="flex:1;">
        <label style="${_FIELDLABEL}">框形状（仅 bbox）</label>
        <select id="sf-ai-yo-shape" style="width:100%;${_INPUT}">
          <option value="rect" ${curShape === "rect" ? "selected" : ""}>矩形</option>
          <option value="ellipse" ${curShape === "ellipse" ? "selected" : ""}>椭圆</option>
        </select>
      </div>
      <div style="flex:1;">
        <label style="${_FIELDLABEL}">Confidence（0.01-1）</label>
        <input type="number" id="sf-ai-yo-conf" value="${curConf}" min="0.01" max="1" step="0.05" style="width:100%;${_INPUT}">
      </div>
    </div>
    <div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
        <label style="${_FIELDLABEL};margin-bottom:0;">类别（默认全选）</label>
        <span>
          <a id="sf-ai-yo-all" style="color:var(--sf-text-dim);font-size:10px;cursor:pointer;text-decoration:underline;">全选</a>
          <a id="sf-ai-yo-none" style="color:var(--sf-text-dim);font-size:10px;cursor:pointer;text-decoration:underline;margin-left:6px;">全不选</a>
        </span>
      </div>
      <div id="sf-ai-yo-classes" style="max-height:132px;overflow-y:auto;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;padding:4px 6px;display:flex;flex-wrap:wrap;gap:3px 12px;">
        <span id="sf-ai-yo-cls-status" style="color:var(--sf-text-faint);font-size:11px;">选择模型后加载类别…</span>
      </div>
    </div>`,
  });
  if (!modal) return;

  const kindSel = modal.dialog.querySelector("#sf-ai-yo-kind");
  const imgszSel = modal.dialog.querySelector("#sf-ai-yo-imgsz");
  const shapeSel = modal.dialog.querySelector("#sf-ai-yo-shape");
  const modelSel = modal.dialog.querySelector("#sf-ai-yo-model");
  const confInput = modal.dialog.querySelector("#sf-ai-yo-conf");
  const hint = modal.dialog.querySelector("#sf-ai-yo-hint");
  const clsBox = modal.dialog.querySelector("#sf-ai-yo-classes");

  const syncShapeVisible = () => {
    const isBbox = kindSel.value === "bbox";
    shapeSel.disabled = !isBbox;
    shapeSel.parentElement.style.opacity = isBbox ? "1" : "0.5";
  };
  syncShapeVisible();

  let lists = { bbox: [], segm: [] };
  try {
    lists = await aiGet(YOLO_LIST);
  } catch (e) {
    console.warn(`${cfg.logTag} yolo list failed:`, e);
  }
  if (!modal.dialog.isConnected) return; // 拉列表期间已关闭

  // ── 类别清单（按 kind+model 拉取；只读元数据，后端不加熔断）──
  let classInfo = { names: {}, task: null };
  // null = 首次（默认全选）；Set = 用户显式选择（可空 → apply 时要求至少勾一个）
  let selectedIds = null;

  const renderClasses = () => {
    clsBox.replaceChildren();
    const entries = Object.entries(classInfo.names || {})
      .map(([id, label]) => [Number(id), String(label)])
      .filter(([id]) => Number.isFinite(id))
      .sort((a, b) => a[0] - b[0]);
    if (!entries.length) {
      const span = document.createElement("span");
      span.style.cssText = "color:var(--sf-text-faint);font-size:11px;";
      span.textContent = "（该模型无类别元数据，按全类运行）";
      clsBox.appendChild(span);
      return;
    }
    // 同名类用 (#id) 区分显示；值恒为 id（稳定，避免同名歧义）
    const dup = {};
    for (const [, label] of entries) dup[label] = (dup[label] || 0) + 1;
    const allIds = entries.map(([id]) => id);
    for (const [id, label] of entries) {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;align-items:center;gap:3px;font-size:11px;color:var(--sf-text);white-space:nowrap;";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.dataset.id = String(id);
      cb.checked = selectedIds === null ? true : selectedIds.has(id);
      cb.onchange = () => {
        if (selectedIds === null) selectedIds = new Set(allIds);
        if (cb.checked) selectedIds.add(id);
        else selectedIds.delete(id);
      };
      wrap.appendChild(cb);
      wrap.appendChild(document.createTextNode(dup[label] > 1 ? `${label} (#${id})` : label));
      clsBox.appendChild(wrap);
    }
  };

  const fetchClasses = async () => {
    const kind = kindSel.value === "segm" ? "segm" : "bbox";
    const model = modelSel.value || "";
    if (!model) { classInfo = { names: {}, task: null }; selectedIds = null; renderClasses(); return; }
    clsBox.replaceChildren();
    const loading = document.createElement("span");
    loading.id = "sf-ai-yo-cls-status";
    loading.style.cssText = "color:var(--sf-text-faint);font-size:11px;";
    loading.textContent = "加载类别中…（首次加载权重）";
    clsBox.appendChild(loading);
    try {
      const data = await aiGet(`${YOLO_CLASSES}?kind=${kind}&model=${encodeURIComponent(model)}`);
      if (!modal.dialog.isConnected) return;
      classInfo = { names: data.names || {}, task: data.task || null };
      const remembered = Array.isArray(st.yolo_classes) ? st.yolo_classes.map((v) => Number(v)) : [];
      const available = new Set(Object.keys(classInfo.names).map((v) => Number(v)));
      const keep = remembered.filter((v) => available.has(v));
      selectedIds = keep.length ? new Set(keep) : null;
    } catch (e) {
      console.warn(`${cfg.logTag} yolo classes failed:`, e);
      if (!modal.dialog.isConnected) return;
      classInfo = { names: {}, task: null };
      selectedIds = null;
    }
    renderClasses();
  };

  const fillModels = () => {
    const names = lists[kindSel.value] || [];
    modelSel.replaceChildren();
    if (!names.length) {
      modelSel.appendChild(new Option("（无可用权重）", ""));
      hint.textContent = "把 .pt 放入 models/ultralytics/bbox（检测）/ segm（分割）或 models/yolo 后重开此窗";
    } else {
      for (const n of names) modelSel.appendChild(new Option(n, n));
      const preferred = st.yolo_model && names.includes(st.yolo_model) ? st.yolo_model : names[0];
      modelSel.value = preferred;
      hint.textContent = `共 ${names.length} 个权重（需已安装 ultralytics，缺失时运行会报错）`;
    }
  };
  fillModels();
  fetchClasses();

  kindSel.onchange = () => {
    syncShapeVisible();
    fillModels();
    fetchClasses();
  };
  modelSel.onchange = () => fetchClasses();
  modal.dialog.querySelector("#sf-ai-yo-all").onclick = () => {
    selectedIds = new Set();
    for (const cb of clsBox.querySelectorAll("input[type=checkbox]")) {
      cb.checked = true;
      selectedIds.add(Number(cb.dataset.id));
    }
  };
  modal.dialog.querySelector("#sf-ai-yo-none").onclick = () => {
    selectedIds = new Set();
    for (const cb of clsBox.querySelectorAll("input[type=checkbox]")) cb.checked = false;
  };

  const apply = () => {
    const kind = kindSel.value === "segm" ? "segm" : "bbox";
    const model = modelSel.value || "";
    if (!model) {
      sfToast({ summary: cfg.toastTag, detail: "没有可用 YOLO 权重", severity: "warn", fallbackTag: cfg.toastTag });
      return;
    }
    if (kind === "segm" && classInfo.task && classInfo.task !== "segment") {
      sfToast({
        summary: cfg.toastTag,
        detail: `该权重是 ${classInfo.task} 模型，不能用于分割：请切换类型为 bbox 或换用 -seg 权重`,
        severity: "warn", life: 6000, fallbackTag: cfg.toastTag,
      });
      return;
    }
    let conf = parseFloat(confInput.value);
    if (!Number.isFinite(conf)) conf = 0.25;
    conf = Math.max(0.01, Math.min(1, conf));
    const shape = kind === "bbox" && shapeSel.value === "ellipse" ? "ellipse" : "rect";
    const imgsz = Number(imgszSel.value) || 640;
    // 勾选集合：全选/无类别元数据 → 不传过滤（后端等价全类）；部分勾选 → 传 id；
    // 全不选 → 拦下（避免"什么都不涂"的困惑）
    const boxes = [...clsBox.querySelectorAll("input[type=checkbox]")];
    const checked = boxes.filter((cb) => cb.checked).map((cb) => Number(cb.dataset.id));
    if (boxes.length > 0 && checked.length === 0) {
      sfToast({ summary: cfg.toastTag, detail: "至少勾选一个类别（或点“全选”）", severity: "warn", fallbackTag: cfg.toastTag });
      return;
    }
    const classes = boxes.length > 0 && checked.length < boxes.length ? checked : [];
    modal.close();
    runYolo(cfg, node, kind, model, conf, shape, imgsz, classes);
  };
  modal.dialog.querySelector(".sf-ai-ok").onclick = apply;
  wireDialogKeys(modal, [confInput], apply);
}

// ── 导入遮罩文件为笔触（纯本地）──────────────────────────────────────────

export function importMaskFile(cfg, node) {
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: cfg.toastTag, detail: "先加载源图（导入的遮罩会缩放到源图尺寸）", severity: "warn", fallbackTag: cfg.toastTag });
    return;
  }
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const dataURL = await readFileAsDataURL(file);
      const res = await CropAPI.uploadSrc("maskimport_" + Date.now(), dataURL);
      const srcPath = res?.path || "";
      if (!srcPath) {
        sfToast({ summary: cfg.toastTag, detail: "遮罩文件上传失败", severity: "error", fallbackTag: cfg.toastTag });
        return;
      }
      const data = await aiPost(IMPORT_MASK, { src_path: srcPath, src_w: st.src_w, src_h: st.src_h });
      mergeStrokes(cfg, node, data, {}, {
        mergedPrefix: "导入遮罩",
        emptyMsg: "遮罩里没有可追踪的白色区域（空结果，笔触不变）",
      });
    } catch (err) {
      console.error(`${cfg.logTag} import mask failed:`, err);
      sfToast({ summary: cfg.toastTag, detail: `导入失败：${(err && err.message) || err}`, severity: "error", life: 6000, fallbackTag: cfg.toastTag });
    }
  };
  input.click();
}

// ── 反选 / 卸载 ───────────────────────────────────────────────────────────

export function toggleInvert(cfg, node) {
  const st = cfg.getState(node);
  const next = !st.invert;
  cfg.patchState(node, { invert: next });
  sfToast({ summary: cfg.toastTag, detail: next ? "已开启反选（输出遮罩取反）" : "已关闭反选", severity: "info", fallbackTag: cfg.toastTag });
}

export async function unloadAiModels(cfg) {
  try {
    const data = await aiPost(UNLOAD_ALL, {});
    const parts = [];
    if (data.sam) parts.push("SAM");
    if (data.person) parts.push("人物");
    if (data.yolo) parts.push(`YOLO×${data.yolo}`);
    sfToast({
      summary: cfg.toastTag,
      detail: parts.length ? `已卸载：${parts.join(" / ")}，显存已释放` : "模型未在驻留，无需卸载",
      severity: "info", fallbackTag: cfg.toastTag,
    });
  } catch (err) {
    console.error(`${cfg.logTag} unload failed:`, err);
    sfToast({ summary: cfg.toastTag, detail: `卸载失败：${(err && err.message) || err}`, severity: "error", fallbackTag: cfg.toastTag });
  }
}

// ── 菜单安装（两个 brush 节点共用）────────────────────────────────────────

export function installBrushMenu(cfg, nodeType) {
  const origMenu = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
    if (origMenu) origMenu.apply(this, arguments);
    if (!Array.isArray(options)) return;
    const st = cfg.getState(this);
    options.push({
      content: "SAM 蒙版：文本选择…",
      callback: () => openSamDialog(cfg, this),
    });
    options.push({
      content: "SAM 点选分割（左键正点 / Shift 负点，Enter 执行）",
      callback: () => beginSamMode(cfg, this, "point"),
    });
    options.push({
      content: "SAM 框选分割（拖框后松开执行）",
      callback: () => beginSamMode(cfg, this, "box"),
    });
    options.push({
      content: "人物部位遮罩…（MediaPipe）",
      callback: () => openPersonDialog(cfg, this),
    });
    options.push({
      content: "YOLO 检测 / 分割…",
      callback: () => openYoloDialog(cfg, this),
    });
    options.push({
      content: "导入遮罩文件为笔触…",
      callback: () => importMaskFile(cfg, this),
    });
    options.push({
      content: (st.invert ? "✓ " : "") + "反选遮罩（Invert）",
      callback: () => toggleInvert(cfg, this),
    });
    options.push({
      content: "卸载 AI 模型（SAM/人物/YOLO）",
      callback: () => unloadAiModels(cfg),
    });
  };
}
