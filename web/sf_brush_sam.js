// ==========================================================================
// sf_brush_sam.js - SAM 文本分割共享 UI（SFImageBrushMask / SFImageCropExpandBrushMask）
// ==========================================================================
//
// 从 sf_brush_mask.js 提取（两节点单源，原 §45.9 实现）：
//   右键菜单 → prompt/threshold/refine 对话框 → POST /api/sfnodes/brush_mask/sam
//   （后端 nodes/image/brush_mask_sam.py，按 src_path 通用，路由名保留不改）
//   → 结果 fill 矢量笔触并入宿主列表统一管理（可擦除/撤销/清除）。
//
// 忙时熔断（§91）：工作流执行期间并发模型加载会撞 comfy-aimdo 的进程级全局
// file reader（native 崩），后端返回 409。本模块先 GET sam_status 预检 busy
// （warn toast，不发推理请求），服务端 409 仍兜底。
//
// cfg: {
//   toastTag: "SF Brush Mask" 等（sfToast summary 标签 + 文案基调）,
//   logTag:   "[SF Brush Mask]" 等（console 日志标签）,
//   getState(node): 读宿主状态（src_path / sam_prompt / sam_threshold / sam_refine）,
//   addStrokes(node, incoming, meta): 并入 fill 笔触 + 记忆字段（宿主 setState +
//     stateChanged）；incoming 为空时也应写入 meta（记忆上次参数），
//     meta = { prompt, threshold, refine },
// }
// ==========================================================================

import { api } from "/scripts/api.js";
import { sfToast, sfApiUrl } from "./sf_common.js";

const SAM_RUN = "/api/sfnodes/brush_mask/sam";
const SAM_UNLOAD = "/api/sfnodes/brush_mask/sam_unload";
const SAM_STATUS = "/api/sfnodes/brush_mask/sam_status";
const OVERLAY_ID = "sf-brush-mask-sam-overlay";

// POST 路由：失败时抛错并携带 status/payload（409 忙时熔断走 warn 提示）
export async function samPost(path, body) {
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

// 预检队列忙闲；查询失败视为不忙（服务端 409 兜底）
async function samBusy() {
  try {
    const res = await api.fetchApi(sfApiUrl(SAM_STATUS), { cache: "no-store" });
    if (!res.ok) return false;
    const data = await res.json();
    return !!data.busy;
  } catch {
    return false;
  }
}

export async function runSamMask(cfg, node, prompt, threshold, refine) {
  const toastTag = cfg.toastTag;
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: toastTag, detail: "先加载源图再跑 SAM", severity: "warn", fallbackTag: toastTag });
    return;
  }
  if (await samBusy()) {
    sfToast({
      summary: toastTag,
      detail: "ComfyUI 正在执行工作流（SAM/模型加载中），为避免冲突，请等当前任务结束后再试",
      severity: "warn", life: 6000, fallbackTag: toastTag,
    });
    return;
  }
  sfToast({ summary: toastTag, detail: `SAM 推理中：${prompt || "object"}…（首次需加载 1.7GB 模型）`, severity: "info", life: 5000, fallbackTag: toastTag });
  try {
    const data = await samPost(SAM_RUN, {
      src_path: st.src_path, prompt, threshold, refine_iterations: refine,
    });
    const incoming = Array.isArray(data && data.strokes) ? data.strokes : [];
    if (!incoming.length) {
      cfg.addStrokes(node, [], { prompt, threshold, refine });
      sfToast({ summary: toastTag, detail: "SAM 未检出目标（空结果，笔触不变）", severity: "warn", fallbackTag: toastTag });
      return;
    }
    cfg.addStrokes(node, incoming, { prompt, threshold, refine });
    const cov = data.coverage != null ? `覆盖 ${Math.round(data.coverage * 100)}%，` : "";
    sfToast({ summary: toastTag, detail: `SAM 并入 ${incoming.length} 个填充笔触（${cov}可擦除/撤销）`, severity: "success", fallbackTag: toastTag });
  } catch (err) {
    console.error(`${cfg.logTag} sam failed:`, err);
    if (err && err.status === 409) {
      sfToast({
        summary: toastTag,
        detail: (err && err.message) || "工作流运行中，请稍后再试",
        severity: "warn", life: 6000, fallbackTag: toastTag,
      });
      return;
    }
    sfToast({ summary: toastTag, detail: `SAM 失败：${(err && err.message) || err}`, severity: "error", life: 6000, fallbackTag: toastTag });
  }
}

export async function unloadSamModel(cfg) {
  const toastTag = cfg.toastTag;
  try {
    const data = await samPost(SAM_UNLOAD, {});
    sfToast({
      summary: toastTag,
      detail: data && data.unloaded ? "SAM 模型已卸载，显存已释放" : "SAM 模型未在驻留，无需卸载",
      severity: "info", fallbackTag: toastTag,
    });
  } catch (err) {
    console.error(`${cfg.logTag} sam unload failed:`, err);
    sfToast({ summary: toastTag, detail: `卸载失败：${(err && err.message) || err}`, severity: "error", fallbackTag: toastTag });
  }
}

// SAM prompt 对话框（prompt 文本 + threshold + refine；Enter 确认 / Esc 关闭；
// 放行 ctrl/meta/alt 组合键。样式同 CropExpand Custom 比例弹窗。）
export function openSamDialog(cfg, node) {
  const toastTag = cfg.toastTag;
  const st = cfg.getState(node);
  if (!st.src_path) {
    sfToast({ summary: toastTag, detail: "先加载源图再跑 SAM", severity: "warn", fallbackTag: toastTag });
    return;
  }
  if (document.getElementById(OVERLAY_ID)) return;

  const overlay = document.createElement("div");
  overlay.id = OVERLAY_ID;
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;";

  const dialog = document.createElement("div");
  dialog.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
    "background:var(--sf-panel-bg);border:1px solid var(--sf-border-soft);border-radius:6px;padding:12px 14px;" +
    "box-shadow:0 4px 20px rgba(0,0,0,0.5);width:300px;box-sizing:border-box;";
  const lastPrompt = st.sam_prompt || "";
  const lastThr = st.sam_threshold ?? 0.5;
  const lastRefine = st.sam_refine ?? 2;
  dialog.innerHTML = `
    <div style="color:var(--sf-text);font-size:13px;margin-bottom:10px;font-weight:bold;">SAM 蒙版：文本选择</div>
    <div style="margin-bottom:10px;">
      <label style="color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;">Prompt（英文，如 person / car，为空按 object）</label>
      <input type="text" id="sf-bm-sam-prompt" value="${String(lastPrompt).replace(/"/g, "&quot;")}" placeholder="person"
        style="width:100%;padding:5px;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;color:var(--sf-text);font-size:13px;box-sizing:border-box;">
      <div style="color:var(--sf-text-faint);font-size:10px;margin-top:3px;">多人用 person:3（:N = 每类最多 N 个），多类用逗号分隔</div>
    </div>
    <div style="margin-bottom:10px;">
      <label style="color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;">Threshold（0-1，越低越多）</label>
      <input type="number" id="sf-bm-sam-thr" value="${lastThr}" min="0" max="1" step="0.05"
        style="width:100%;padding:5px;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;color:var(--sf-text);font-size:13px;box-sizing:border-box;">
    </div>
    <div style="margin-bottom:10px;">
      <label style="color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;">Refine（0-5，SAM 解码精修轮数，0=用粗蒙版）</label>
      <input type="number" id="sf-bm-sam-refine" value="${lastRefine}" min="0" max="5" step="1"
        style="width:100%;padding:5px;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;color:var(--sf-text);font-size:13px;box-sizing:border-box;">
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;">
      <button id="sf-bm-sam-cancel" style="padding:5px 12px;background:var(--sf-surface);border:none;border-radius:3px;color:var(--sf-text);cursor:pointer;font-size:12px;">Cancel</button>
      <button id="sf-bm-sam-ok" style="padding:5px 12px;background:#4a90e2;border:none;border-radius:3px;color:white;cursor:pointer;font-size:12px;">Run SAM</button>
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

  const promptInput = dialog.querySelector("#sf-bm-sam-prompt");
  const thrInput = dialog.querySelector("#sf-bm-sam-thr");
  const refineInput = dialog.querySelector("#sf-bm-sam-refine");
  setTimeout(() => promptInput.focus(), 100);

  const apply = () => {
    let thr = parseFloat(thrInput.value);
    if (!Number.isFinite(thr)) thr = 0.5;
    thr = Math.max(0, Math.min(1, thr));
    let refine = parseInt(refineInput.value, 10);
    if (!Number.isFinite(refine)) refine = 2;
    refine = Math.max(0, Math.min(5, refine));
    const p = promptInput.value || "";
    close();
    runSamMask(cfg, node, p, thr, refine);
  };
  dialog.querySelector("#sf-bm-sam-ok").onclick = apply;
  dialog.querySelector("#sf-bm-sam-cancel").onclick = close;
  for (const el of [promptInput, thrInput, refineInput]) {
    el.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") {
        // 吞掉回车：否则 keydown 上浮到 window，会触发全局键位
        // （如用户自绑的 Enter 队列）导致与对话框无关的报错
        e.preventDefault();
        e.stopPropagation();
        apply();
      } else if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        close();
      }
    };
  }
}

// 右键两菜单（getExtraMenuOptions 原型包装，any_pack.js 同款）
export function installSamMenu(cfg, nodeType) {
  const origMenu = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
    if (origMenu) origMenu.apply(this, arguments);
    if (!Array.isArray(options)) return;
    options.push({
      content: "SAM 蒙版：文本选择…",
      callback: () => openSamDialog(cfg, this),
    });
    options.push({
      content: "卸载 SAM 模型",
      callback: () => unloadSamModel(cfg),
    });
  };
}
