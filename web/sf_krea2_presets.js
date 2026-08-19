// ==========================================================================
// sf_krea2_presets.js — Krea2 预设管理共享模块
//   SFImageInterrogator（kind="interrogator"）与 SFKrea2SystemPrompt（kind="krea2"）
//   共用：预设 API 封装、combo 动态重建、节点"管理预设"按钮 + 管理 popup。
//
// 后端：sf_utils/krea2_presets.py（用户覆盖 + 墓碑删除 + 复位，GET/POST/DELETE/reset）。
// 弹层：复用 sf_popup.js 三件套（attachPopupDismiss / clampToViewport）。
// 数据形状（GET 返回）：{ presets: {名:文本}, builtin: {...}, user: {...}, deleted: [...] }
//   presets = 合并视图（combo 用），builtin = 内置默认，user = 用户 overrides。
// 事件：改动后派发 `sfnodes.<kind>-presets-changed`，各节点监听后重建 combo options。
// ==========================================================================

import { app } from "/scripts/app.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";

const KIND_LABEL = {
  interrogator: "反推预设",
  krea2: "系统指令预设",
};

function apiURL(kind) {
  return `/api/sfnodes/${kind}_presets`;
}

// ── API ────────────────────────────────────────────────────────────────
export async function fetchPresets(kind) {
  const resp = await fetch(apiURL(kind));
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  if (!data || typeof data.presets !== "object") throw new Error("bad payload");
  return data; // {presets, builtin, user, deleted}
}

export async function savePreset(kind, name, text) {
  const resp = await fetch(apiURL(kind), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, text }),
  });
  return resp.json();
}

export async function deletePreset(kind, name) {
  const resp = await fetch(`${apiURL(kind)}?name=${encodeURIComponent(name)}`, { method: "DELETE" });
  return resp.json();
}

export async function resetPreset(kind, name) {
  const resp = await fetch(`${apiURL(kind)}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return resp.json();
}

export async function resetAllPresets(kind) {
  const resp = await fetch(`${apiURL(kind)}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ all: true }),
  });
  return resp.json();
}

// ── combo options 重建 ────────────────────────────────────────────────
// 把某节点的 preset combo options 设为合并预设名列表（保留当前值——后端
// VALIDATE_INPUTS=True 兜底，值不在列表也不拦截）。
// 注意：ComfyUI combo 的 options 是对象 `{values: [...]}`（不是数组），须写
// `w.options.values`（对齐 sf_combo_selector / power_lora_preset 先例），直接
// 赋数组会破坏 combo 渲染。
export function setPresetOptions(node, presets) {
  const w = node?.widgets?.find((x) => x.name === "preset");
  if (!w) return;
  const keys = Object.keys(presets || {});
  if (!keys.length) keys.push("");
  w.options = { ...w.options, values: keys };
  node.setDirtyCanvas?.(true, true);
}

export function nodesOfClass(comfyClass) {
  return (app?.graph?._nodes ?? []).filter((n) => n?.comfyClass === comfyClass);
}

export function presetsChangedEvent(kind) {
  return `sfnodes.${kind}-presets-changed`;
}

export function broadcastPresetsChanged(kind) {
  document.dispatchEvent(new CustomEvent(presetsChangedEvent(kind), { detail: { kind } }));
}

// 重拉并重建所有同 class 节点 options（不广播——供事件监听回调使用，避免
// 监听到自身广播后再次广播造成无限循环）。
export async function reloadNodes(kind, comfyClass) {
  const data = await fetchPresets(kind);
  for (const n of nodesOfClass(comfyClass)) setPresetOptions(n, data.presets);
  return data;
}

// 改动后统一收尾：重拉 → 重建所有同 class 节点 options → 广播事件（供其他
// 窗口/节点同步）。仅由改动发起方（管理 popup）调用。
export async function refreshAllNodes(kind, comfyClass, onError) {
  try {
    const data = await reloadNodes(kind, comfyClass);
    broadcastPresetsChanged(kind);
    return data;
  } catch (e) {
    onError?.(e);
    return null;
  }
}

// ── 节点"管理预设"按钮（DOM widget，不存值 → 无值写入递归风险）────────
export function addManageButton(node, kind) {
  if (!node || node._sfK2PManageAdded) return;
  injectCSS();
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sf-k2p-manage";
  btn.textContent = "⚙ 管理预设";
  btn.title = "管理预设：新增 / 修改 / 删除 / 复位";
  btn.addEventListener("click", () => openPresetManager(kind, node, btn));
  node.addDOMWidget("sfK2PresetsManage", "sfK2PresetsManage", btn, {
    serialize: false,
    getValue: () => null,
    setValue: () => {},
    getMinHeight: () => 26,
    margin: 4,
  });
  node._sfK2PManageAdded = true;
}

// ── 管理 popup ─────────────────────────────────────────────────────────
export async function openPresetManager(kind, node, anchor) {
  const existing = document.getElementById("sf-k2p-overlay");
  if (existing) existing.remove();

  let data;
  try {
    data = await fetchPresets(kind);
  } catch (e) {
    const fail = el("div", "sf-k2p");
    fail.style.cssText = "position:fixed;z-index:10000;left:50%;top:40%;transform:translate(-50%,-50%);background:#222;border:1px solid #7a3a3a;color:#e57373;padding:14px 18px;border-radius:8px;font-size:13px";
    fail.textContent = "预设加载失败，请确认后端已重启容器（路由未生效）";
    document.body.appendChild(fail);
    setTimeout(() => fail.remove(), 3500);
    return;
  }
  const overlay = document.createElement("div");
  overlay.id = "sf-k2p-overlay";
  overlay.className = "sf-k2p";
  overlay.style.setProperty("--acc", "var(--sf-acc, #f66744)");
  document.body.appendChild(overlay);
  injectCSS();

  let editing = null; // null=列表，或 {name, isNew}
  let msg = "";

  function render() {
    if (editing) return renderEdit();
    renderList();
  }

  function rowLabel(name) {
    const isBuiltin = Object.prototype.hasOwnProperty.call(data.builtin, name);
    const modified = isBuiltin && Object.prototype.hasOwnProperty.call(data.user, name);
    if (!isBuiltin) return "用户";
    return modified ? "内置·已改" : "内置";
  }

  function renderList() {
    overlay.textContent = "";
    // 头部
    const head = el("div", "sf-k2p-head");
    head.append(el("span", "sf-k2p-title", `管理${KIND_LABEL[kind]}`));
    const close = el("span", "sf-k2p-x", "✕");
    close.addEventListener("click", () => overlay.remove());
    head.appendChild(close);
    overlay.appendChild(head);

    const bar = el("div", "sf-k2p-bar");
    const add = el("button", "sf-k2p-btn pri", "＋ 新增");
    add.addEventListener("click", () => { editing = { name: "", isNew: true }; render(); });
    const resetAll = el("button", "sf-k2p-btn", "复位全部内置");
    resetAll.addEventListener("click", async () => {
      if (!confirm("复位全部预设？用户对内置的修改/删除将被还原，用户新增的也会被删除。")) return;
      await resetAllPresets(kind);
      msg = "已复位全部";
      refreshAllNodes(kind, node?.comfyClass).then((d) => { if (d) data.presets = d.presets; render(); });
    });
    bar.append(add, resetAll);
    overlay.appendChild(bar);

    if (msg) overlay.appendChild(el("div", "sf-k2p-msg", msg));

    const list = el("div", "sf-k2p-list");
    const names = Object.keys(data.presets);
    if (!names.length) list.appendChild(el("div", "sf-k2p-empty", "(无预设)"));
    for (const nm of names) {
      list.appendChild(makeRow(nm, data.presets[nm]));
    }
    overlay.appendChild(list);
    clampToViewport(overlay, { scale: canvasScale() });
  }

  function makeRow(nm, text) {
    const row = el("div", "sf-k2p-row");
    const info = el("div", "sf-k2p-info");
    const top = el("div", "sf-k2p-name");
    top.appendChild(el("b", null, nm));
    const isBuiltin = Object.prototype.hasOwnProperty.call(data.builtin, nm);
    const isProtected = kind === "krea2" && nm === "none";
    const badge = el("span", "sf-k2p-badge " + (rowLabel(nm) === "用户" ? "u" : "b"), rowLabel(nm));
    if (isProtected) badge.textContent = "保护";
    top.appendChild(badge);
    info.appendChild(top);
    info.appendChild(el("div", "sf-k2p-text", String(text || "").slice(0, 120) + (String(text || "").length > 120 ? "…" : "")));

    const ops = el("div", "sf-k2p-ops");
    if (!isProtected) {
      const edit = el("button", "sf-k2p-btn", "编辑");
      edit.addEventListener("click", () => { editing = { name: nm, isNew: false }; render(); });
      ops.appendChild(edit);
      // 复位：仅内置被改，或用户新增（含内置被删后重新拉回列表前不存在——这里只覆盖可见项）
      const modified = isBuiltin && Object.prototype.hasOwnProperty.call(data.user, nm);
      if (modified || !isBuiltin) {
        const rst = el("button", "sf-k2p-btn", "复位");
        rst.title = isBuiltin ? "还原为内置默认" : "删除此用户预设";
        rst.addEventListener("click", async () => {
          if (!confirm(isBuiltin ? `还原 "${nm}" 为内置默认？` : `删除用户预设 "${nm}"？`)) return;
          if (isBuiltin) await resetPreset(kind, nm);
          else await deletePreset(kind, nm);
          msg = "";
          refreshAllNodes(kind, node?.comfyClass).then((d) => { if (d) data.presets = d.presets; render(); });
        });
        ops.appendChild(rst);
      }
      const del = el("button", "sf-k2p-btn danger", "删除");
      del.addEventListener("click", async () => {
        if (!confirm(`删除预设 "${nm}"？${isBuiltin ? "（内置，可用复位还原）" : ""}`)) return;
        await deletePreset(kind, nm);
        msg = "";
        refreshAllNodes(kind, node?.comfyClass).then((d) => { if (d) data.presets = d.presets; render(); });
      });
      ops.appendChild(del);
    }

    row.append(info, ops);
    return row;
  }

  function renderEdit() {
    overlay.textContent = "";
    const head = el("div", "sf-k2p-head");
    head.append(el("span", "sf-k2p-title", editing.isNew ? `新增${KIND_LABEL[kind]}` : `编辑：${editing.name}`));
    const back = el("span", "sf-k2p-x", "←");
    back.addEventListener("click", () => { editing = null; render(); });
    head.appendChild(back);
    overlay.appendChild(head);

    const form = el("div", "sf-k2p-form");
    const nameInput = el("input", "sf-k2p-input");
    nameInput.placeholder = "预设名称…";
    nameInput.maxLength = 64;
    nameInput.value = editing.isNew ? "" : editing.name;
    const textArea = el("textarea", "sf-k2p-textarea");
    textArea.placeholder = "预设指令文本…";
    textArea.value = editing.isNew ? "" : data.presets[editing.name] || "";
    form.append(el("label", null, "名称"), nameInput, el("label", null, "指令文本"), textArea);

    const ops = el("div", "sf-k2p-ops");
    const save = el("button", "sf-k2p-btn pri", "保存");
    save.addEventListener("click", async () => {
      const nm = nameInput.value.trim();
      if (!nm) { msg = "名称不能为空"; renderList(); return; }
      const r = await savePreset(kind, nm, textArea.value);
      if (!r?.ok) { msg = (r && r.error) || "保存失败"; renderList(); return; }
      msg = `已保存 "${nm}"`;
      editing = null;
      refreshAllNodes(kind, node?.comfyClass).then((d) => { if (d) data.presets = d.presets; render(); });
    });
    const cancel = el("button", "sf-k2p-btn", "取消");
    cancel.addEventListener("click", () => { editing = null; render(); });
    nameInput.addEventListener("keydown", (ev) => {
      ev.stopPropagation();
      if (ev.key === "Enter") { ev.preventDefault(); save.click(); }
      if (ev.key === "Escape") { ev.preventDefault(); editing = null; render(); }
    });
    textArea.addEventListener("keydown", (ev) => {
      ev.stopPropagation();
      if (ev.key === "Escape") { ev.preventDefault(); editing = null; render(); }
    });
    ops.append(save, cancel);
    form.appendChild(ops);
    overlay.appendChild(form);
    nameInput.focus();
    if (editing.isNew) nameInput.select();
    clampToViewport(overlay, { scale: canvasScale() });
  }

  function canvasScale() {
    // 各前端版本 scale 来源不一（Classic: canvas.ds.scale；Vue 等），尽力取，取不到按 1。
    try {
      return app.canvas?.ds?.scale ?? app.canvas?.ds?.scale ?? 1;
    } catch (e) {
      return 1;
    }
  }

  render();
  attachPopupDismiss(overlay, {
    exempt: (e) => overlay.contains(e.target),
    onClose: () => overlay.remove(),
  });
  // 定位到 anchor（管理按钮）附近，钳回视口
  if (anchor) {
    const r = anchor.getBoundingClientRect();
    overlay.style.left = `${r.left}px`;
    overlay.style.top = `${r.bottom + 4}px`;
  } else {
    overlay.style.left = "50%";
    overlay.style.top = "40%";
    overlay.style.transform = "translate(-50%, -50%)";
  }
  clampToViewport(overlay, { scale: canvasScale() });
  return overlay;
}

// ── DOM helpers ─────────────────────────────────────────────────────────
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function injectCSS() {
  if (document.getElementById("sf-k2p-css")) return;
  const css = document.createElement("style");
  css.id = "sf-k2p-css";
  css.textContent = `
.sf-k2p-manage{width:100%;cursor:pointer;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);color:var(--sf-acc,#f66744);border-radius:4px;padding:2px 6px;font-size:12px;text-align:center}
.sf-k2p-manage:hover{background:rgba(255,255,255,.12)}
.sf-k2p{position:fixed;z-index:10000;min-width:320px;max-width:420px;max-height:70vh;overflow:auto;background:#222;border:1px solid #444;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.5);color:#ddd;font-size:13px;display:flex;flex-direction:column}
.sf-k2p-head{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;border-bottom:1px solid #3a3a3a;font-weight:600}
.sf-k2p-title{color:var(--acc,var(--sf-acc,#f66744))}
.sf-k2p-x{cursor:pointer;color:#999;padding:0 4px}
.sf-k2p-x:hover{color:#fff}
.sf-k2p-bar{display:flex;gap:6px;padding:8px 10px}
.sf-k2p-btn{cursor:pointer;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.18);color:#ddd;border-radius:4px;padding:3px 8px;font-size:12px}
.sf-k2p-btn:hover{background:rgba(255,255,255,.16)}
.sf-k2p-btn.pri{background:var(--acc,var(--sf-acc,#f66744));color:#fff;border-color:transparent}
.sf-k2p-btn.danger{color:#e57373;border-color:#7a3a3a}
.sf-k2p-msg{padding:0 10px 6px;color:#7bd88f;font-size:12px}
.sf-k2p-list{padding:0 10px 10px;display:flex;flex-direction:column;gap:6px}
.sf-k2p-empty{color:#888;padding:6px;text-align:center}
.sf-k2p-row{border:1px solid #3a3a3a;border-radius:6px;padding:6px 8px;display:flex;justify-content:space-between;gap:8px;align-items:flex-start}
.sf-k2p-info{flex:1;min-width:0}
.sf-k2p-name{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.sf-k2p-badge{font-size:10px;padding:0 5px;border-radius:3px}
.sf-k2p-badge.b{background:rgba(255,255,255,.12);color:#aaa}
.sf-k2p-badge.u{background:rgba(123,216,143,.18);color:#7bd88f}
.sf-k2p-text{color:#aaa;margin-top:2px;word-break:break-word;white-space:pre-wrap;max-height:60px;overflow:hidden}
.sf-k2p-ops{display:flex;gap:4px;flex-shrink:0}
.sf-k2p-form{padding:10px;display:flex;flex-direction:column;gap:6px}
.sf-k2p-form label{font-size:11px;color:#999}
.sf-k2p-input{background:#1b1b1b;border:1px solid #444;color:#eee;border-radius:4px;padding:4px 6px}
.sf-k2p-textarea{background:#1b1b1b;border:1px solid #444;color:#eee;border-radius:4px;padding:6px;min-height:120px;resize:vertical;font-family:inherit;font-size:12px}
.sf-k2p-form .sf-k2p-ops{justify-content:flex-end;margin-top:4px}
`;
  document.head.appendChild(css);
}
