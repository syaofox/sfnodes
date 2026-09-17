// ==========================================================================
// sf_crop_expand_ratios.js - 自定义比例预设库与 Custom 管理弹窗（节点无关）
// ==========================================================================
//
// SFImageCropExpand 与 SFImageCropExpandBrushMask 共用（原 crop_expand.js 内联
// 实现提升，仅把套用动作参数化为 cfg.applyCustom）：
//
// 真源为全局库（经 /api/sfnodes/crop_expand_presets 读写，后端
// sf_utils/crop_expand_presets.py）；点击已存定义即把该比例套用到宿主节点
// （写 properties 状态，随工作流保存）。后端不可用时列表降级为空、手输照常
// 可用。弹窗三关闭 / 输入框 keydown 放行组合键复用 sf_popup/sf_common。
// ==========================================================================

import { sfApiUrl, sfToast } from "./sf_common.js";
import { attachPopupDismiss } from "./sf_popup.js";
import { ASPECT_RATIOS, normalizeRatioPresets } from "./sf_crop_expand_lib.js";

export const RATIO_PRESETS_API = "/api/sfnodes/crop_expand_presets";

export function ratioLabel(key) {
  return ASPECT_RATIOS.find((r) => r.key === key)?.label || key;
}

export async function fetchRatioPresets() {
  try {
    const r = await fetch(sfApiUrl(RATIO_PRESETS_API), { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    return normalizeRatioPresets(j);
  } catch (e) {
    console.warn("[SF Crop Expand Ratios] 自定义比例预设库加载失败，手输仍可用:", e);
    return null; // null = 后端不可用
  }
}

export async function apiSaveRatioPreset(name, w, h) {
  try {
    const r = await fetch(sfApiUrl(RATIO_PRESETS_API), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, w, h }),
    });
    return r.ok;
  } catch (e) {
    console.warn("[SF Crop Expand Ratios]", e);
    return false;
  }
}

export async function apiDeleteRatioPreset(name) {
  try {
    const r = await fetch(sfApiUrl(`${RATIO_PRESETS_API}?name=${encodeURIComponent(name)}`), { method: "DELETE" });
    return r.ok;
  } catch (e) {
    console.warn("[SF Crop Expand Ratios]", e);
    return false;
  }
}

// ── Custom 比例管理弹窗（左已存定义列表 + 右编辑器；sf_popup 三关闭）──────

const _BTN = "padding:5px 12px;border:none;border-radius:3px;cursor:pointer;font-size:12px;";
const _INPUT = "padding:5px;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;color:var(--sf-text);font-size:13px;box-sizing:border-box;";

function markRatioFields(wInput, hInput, bad) {
  const color = bad ? "#e74c3c" : "var(--sf-border-soft)";
  wInput.style.borderColor = color;
  hInput.style.borderColor = color;
}

// openCustomRatioDialog(cfg)
// cfg: {
//   initialCustom: { w, h }  字段初值（宿主当前 custom_w/custom_h）,
//   applyCustom: (w, h) => void  套用比例（宿主写状态 + 改框 + 重绘；本函数
//                                成功后自动 close）,
//   toastTag: "SF Crop Expand" 等  sfToast summary 标签,
// }
export function openCustomRatioDialog(cfg) {
  if (document.getElementById("sf-crop-expand-ratio-overlay")) return;
  const toastTag = cfg.toastTag || "SF Crop Expand";
  const initW = cfg.initialCustom?.w ?? 1;
  const initH = cfg.initialCustom?.h ?? 1;

  const overlay = document.createElement("div");
  overlay.id = "sf-crop-expand-ratio-overlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;";

  const dialog = document.createElement("div");
  dialog.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);" +
    "background:var(--sf-panel-bg);border:1px solid var(--sf-border-soft);border-radius:6px;padding:12px 14px;" +
    "box-shadow:0 4px 20px rgba(0,0,0,0.5);width:460px;color:var(--sf-text);font-size:13px;box-sizing:border-box;";
  dialog.innerHTML = `
    <div style="font-weight:bold;margin-bottom:10px;">Custom Aspect Ratio</div>
    <div style="display:flex;gap:12px;align-items:stretch;">
      <div style="width:180px;display:flex;flex-direction:column;min-width:0;">
        <label style="color:var(--sf-text-dim);font-size:10px;margin-bottom:3px;">Saved Definitions</label>
        <div id="sf-ce-ratio-list" style="flex:1;min-height:150px;max-height:240px;overflow-y:auto;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;padding:3px;"></div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;gap:8px;min-width:0;">
        <div>
          <label style="color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;">Name</label>
          <input type="text" id="sf-ce-ratio-name" placeholder="definition name"
            style="width:100%;${_INPUT}">
        </div>
        <div style="display:flex;gap:10px;align-items:flex-end;">
          <div>
            <label style="color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;">Width</label>
            <input type="number" id="sf-ce-ratio-w" value="${initW}" min="0.1" step="0.1"
              style="width:100px;${_INPUT}">
          </div>
          <div style="color:var(--sf-text-faint);font-size:16px;padding-bottom:6px;">:</div>
          <div>
            <label style="color:var(--sf-text-dim);font-size:10px;display:block;margin-bottom:3px;">Height</label>
            <input type="number" id="sf-ce-ratio-h" value="${initH}" min="0.1" step="0.1"
              style="width:100px;${_INPUT}">
          </div>
        </div>
        <div style="display:flex;gap:8px;">
          <button id="sf-ce-ratio-save" style="${_BTN}background:#3a5f8a;color:#fff;">Save</button>
          <button id="sf-ce-ratio-del" style="${_BTN}background:#5a2f2f;color:#f0b0b0;">Delete</button>
        </div>
      </div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px;">
      <button id="sf-ce-ratio-cancel" style="${_BTN}background:var(--sf-surface);color:var(--sf-text);">Cancel</button>
      <button id="sf-ce-ratio-ok" style="${_BTN}background:#4a90e2;color:white;">Apply</button>
    </div>`;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    overlay.remove();
  };
  // 三关闭（外部点击 / Esc / 滚轮）复用公共库
  attachPopupDismiss(overlay, { onClose: close });

  const listEl = dialog.querySelector("#sf-ce-ratio-list");
  const nameInput = dialog.querySelector("#sf-ce-ratio-name");
  const wInput = dialog.querySelector("#sf-ce-ratio-w");
  const hInput = dialog.querySelector("#sf-ce-ratio-h");
  const saveBtn = dialog.querySelector("#sf-ce-ratio-save");
  const delBtn = dialog.querySelector("#sf-ce-ratio-del");

  let presets = [];
  let selectedIndex = -1;
  let routeOk = true;

  const parseFields = () => {
    const w = parseFloat(wInput.value);
    const h = parseFloat(hInput.value);
    const ok = !isNaN(w) && !isNaN(h) && w > 0 && h > 0;
    markRatioFields(wInput, hInput, !ok);
    return ok ? { w, h } : null;
  };

  // 套用当前字段比例（Apply / 双击列表项共用）
  const applyRatio = (w, h) => {
    cfg.applyCustom(w, h);
    close();
  };

  const apply = () => {
    const v = parseFields();
    if (!v) return;
    applyRatio(v.w, v.h);
  };

  const renderList = () => {
    listEl.replaceChildren();
    if (presets.length === 0) {
      const empty = document.createElement("div");
      empty.style.cssText = "color:var(--sf-text-faint);font-size:11px;padding:8px 4px;text-align:center;";
      empty.textContent = routeOk ? "No saved definitions" : "Preset store unavailable";
      listEl.appendChild(empty);
      return;
    }
    presets.forEach((p, idx) => {
      const fmt = (n) => (Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100));
      const item = document.createElement("div");
      item.style.cssText = "padding:4px 6px;border-radius:3px;cursor:pointer;margin-bottom:2px;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px;" +
        (idx === selectedIndex ? "background:#3a5f8a;" : "");
      item.textContent = `${p.name}  (${fmt(p.w)}:${fmt(p.h)})`;
      item.title = `${p.name} ${fmt(p.w)}:${fmt(p.h)}`;
      item.onmouseenter = () => { if (idx !== selectedIndex) item.style.background = "var(--sf-surface-hover)"; };
      item.onmouseleave = () => { item.style.background = idx === selectedIndex ? "#3a5f8a" : ""; };
      // 单击选中并回填字段；双击直接套用
      item.onclick = () => {
        selectedIndex = idx;
        nameInput.value = p.name;
        wInput.value = p.w;
        hInput.value = p.h;
        markRatioFields(wInput, hInput, false);
        renderList();
      };
      item.ondblclick = () => applyRatio(p.w, p.h);
      listEl.appendChild(item);
    });
  };

  const reload = async (selectName) => {
    const list = await fetchRatioPresets();
    if (list === null) {
      routeOk = false;
      renderList();
      return;
    }
    routeOk = true;
    presets = list;
    selectedIndex = selectName ? presets.findIndex((p) => p.name === selectName) : -1;
    renderList();
  };

  const save = async () => {
    const name = nameInput.value.trim();
    if (!name) {
      nameInput.style.borderColor = "#e74c3c";
      sfToast({ summary: toastTag, detail: "请填写定义名称", severity: "warn", fallbackTag: toastTag });
      return;
    }
    nameInput.style.borderColor = "var(--sf-border-soft)";
    const v = parseFields();
    if (!v) {
      sfToast({ summary: toastTag, detail: "宽度/高度必须为正数", severity: "warn", fallbackTag: toastTag });
      return;
    }
    if (!(await apiSaveRatioPreset(name, v.w, v.h))) {
      sfToast({ summary: toastTag, detail: "保存失败（后端路由不可用？重启 ComfyUI 后重试）", severity: "error", fallbackTag: toastTag });
      return;
    }
    sfToast({ summary: toastTag, detail: `已保存定义「${name}」`, fallbackTag: toastTag });
    await reload(name);
  };

  const remove = async () => {
    if (selectedIndex < 0 || selectedIndex >= presets.length) {
      sfToast({ summary: toastTag, detail: "请先在左侧选择要删除的定义", severity: "warn", fallbackTag: toastTag });
      return;
    }
    const name = presets[selectedIndex].name;
    if (!confirm(`删除定义「${name}」？`)) return;
    if (!(await apiDeleteRatioPreset(name))) {
      sfToast({ summary: toastTag, detail: "删除失败（后端路由不可用？）", severity: "error", fallbackTag: toastTag });
      return;
    }
    sfToast({ summary: toastTag, detail: `已删除定义「${name}」`, fallbackTag: toastTag });
    nameInput.value = "";
    await reload();
  };

  dialog.querySelector("#sf-ce-ratio-ok").onclick = apply;
  dialog.querySelector("#sf-ce-ratio-cancel").onclick = close;
  saveBtn.onclick = save;
  delBtn.onclick = remove;

  // Enter 应用 / Esc 关闭；放行 ctrl/meta/alt 组合键（Ctrl+S 不能漏成浏览器行为）
  for (const el of [wInput, hInput, nameInput]) {
    el.onkeydown = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") apply();
      else if (e.key === "Escape") close();
    };
    el.oninput = () => {
      wInput.style.borderColor = "var(--sf-border-soft)";
      hInput.style.borderColor = "var(--sf-border-soft)";
      nameInput.style.borderColor = "var(--sf-border-soft)";
    };
  }
  setTimeout(() => (presets.length ? wInput : nameInput).focus(), 100);

  reload();
}
