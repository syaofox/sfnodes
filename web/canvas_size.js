// ==========================================================================
// Canvas Size Preset - model -> resolution 动态联动 + 自定义分辨率库管理
// ==========================================================================
//
// 节点：SFCanvasSizePreset（nodes/utils/canvas_size.py）
// 数据源（均为 Python/磁盘唯一真源，本文件不内联副本）：
//  - 官方预设表：GET /api/sfnodes/canvas_size_presets（nodes/utils/canvas_size.py）
//  - 自定义库：GET/POST/DELETE /api/sfnodes/canvas_size_custom
//    （sf_utils/canvas_size_presets.py，user/sfnodes/canvas_size_presets.json）
// 纯逻辑（归一化/合并）在 web/sf_canvas_size_lib.js。任一接口失败降级：
// 官方表失败保持 INPUT_TYPES 静态选项（联动不可用但不破坏节点），自定义库
// 失败则仅隐藏自定义项。
//
// 自定义库归属：model 下拉的伪模型 "Custom Resolution"（后端 CUSTOM_MODEL，经
// payload.custom_model 下发），选中它时 resolution 只列自定义库；真实模型只列
// 各自官方档位（自定义项不混入，否则会被视为"归类到该模型"）。
//
// 联动时机：
//  - nodeCreated：初始渲染 + 挂管理按钮 + 触发两次数据加载
//  - model widget callback：切换模型就地按缓存重建（API 就绪时同步生效）
//  - onAfterGraphConfigured：加载工作流时 widget 值已恢复，combo options
//    不随工作流保存（text_preset 经验），按恢复的 model 值重建
//  - 自定义库增删：重拉后重建所有同类节点 options
// ==========================================================================

import { app } from "/scripts/app.js";
import { el, injectCSSOnce, sfApiUrl } from "./sf_common.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";
import {
  CUSTOM_HEADER,
  customOptionValue,
  firstSelectable,
  isTierHeader,
  mergeResolutionValues,
  normalizeCustomPresets,
  validCustomName,
  validDim,
} from "./sf_canvas_size_lib.js";

const CLASS = "SFCanvasSizePreset";
const OFFICIAL_API = "/api/sfnodes/canvas_size_presets";
const CUSTOM_API = "/api/sfnodes/canvas_size_custom";
const TOAST_TAG = "SF Canvas Size Preset";
const CUSTOM_OVERLAY_ID = "sf-cs-custom-overlay";

// ── 数据缓存 ────────────────────────────────────────────────────────────
let _officialPromise = null;
let _customPromise = null;
let _official = null; // {models, values}（官方表）
let _custom = [];     // [{name, w, h}]（自定义库，归一化）

function loadOfficial() {
  if (!_officialPromise) {
    _officialPromise = fetch(sfApiUrl(OFFICIAL_API))
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((d) => {
        if (d) _official = d;
        return d;
      });
  }
  return _officialPromise;
}

function loadCustom() {
  if (!_customPromise) {
    _customPromise = fetch(sfApiUrl(CUSTOM_API))
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((d) => {
        if (d) _custom = normalizeCustomPresets(d);
        return d; // null = 后端不可用（保留既有 _custom 供渲染）
      });
  }
  return _customPromise;
}

function refetchCustom() {
  _customPromise = null;
  return loadCustom();
}

// ── combo options 重建 ──────────────────────────────────────────────────
function nodesOfClass() {
  return (app?.graph?._nodes ?? []).filter((n) => n?.comfyClass === CLASS);
}

// 按当前缓存重建某节点的 resolution 选项。官方表未就绪时跳过——绝不用
// "当前静态选项"（只是默认模型的表）冒充其他 model 的表。
// 伪模型 CUSTOM_MODEL：官方表为空，options 由全局自定义库填充（空库给占位头）。
// preserveValue=true（数据加载/工作流恢复）：当前值不在列表时保留原值（自定义
// 库离线/删条目后旧值仍可执行），只重建选项；false（用户切换 model）：回退。
function syncNode(node, preserveValue = false) {
  const modelWidget = node.widgets?.find((w) => w.name === "model");
  const resWidget = node.widgets?.find((w) => w.name === "resolution");
  if (!modelWidget || !resWidget || !_official) return;

  const isCustom = !!_official.custom_model && modelWidget.value === _official.custom_model;
  const official = isCustom ? [] : _official.values?.[modelWidget.value];
  if (!Array.isArray(official)) return;
  if (!isCustom && official.length === 0) return;

  // 真实模型：只列官方档位（自定义库不混入——否则会被视为"归类到该模型"）。
  // 伪模型：官方表空，选项 = 自定义库；空库给占位头（保证 combo 非空）。
  let values;
  if (isCustom) {
    values = mergeResolutionValues([], _custom);
    if (values.length === 0) values = [CUSTOM_HEADER];
  } else {
    values = official.slice();
  }
  const keep = values.includes(resWidget.value);
  resWidget.options = Object.assign({}, resWidget.options, { values });
  if (resWidget.updateOptions) resWidget.updateOptions();
  if (!keep && !(preserveValue && !isTierHeader(resWidget.value))) {
    // 回退：真实模型取首个官方选项（不落到自定义项）；自定义模型取首个自定义项
    resWidget.value = isCustom
      ? (firstSelectable(values) ?? values[0])
      : (firstSelectable(official) ?? official[0]);
  }
  node.setDirtyCanvas?.(true, true);
}

function syncAll() {
  for (const n of nodesOfClass()) syncNode(n);
}

// ── 管理按钮 ────────────────────────────────────────────────────────────
function installManageButton(node) {
  if (!node || node._sfCsManageAdded || typeof node.addDOMWidget !== "function") return;
  injectCSS();
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sf-cs-manage";
  btn.textContent = "⚙ 自定义分辨率";
  btn.title = "管理全局自定义分辨率预设（跨工作流共享）";
  btn.addEventListener("click", () => openManager(node, btn));
  node.addDOMWidget("sfCsCustomManage", "sfCsCustomManage", btn, {
    serialize: false,
    getValue: () => null,
    setValue: () => {},
    getMinHeight: () => 26,
    getMaxHeight: () => 26,
    margin: 4,
  });
  node._sfCsManageAdded = true;
}

// ── 自定义库 API ────────────────────────────────────────────────────────
async function apiSave(name, w, h) {
  try {
    const r = await fetch(sfApiUrl(CUSTOM_API), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, w, h }),
    });
    return r.ok;
  } catch (e) {
    console.warn("[SF Canvas Size Preset]", e);
    return false;
  }
}

async function apiDelete(name) {
  try {
    const r = await fetch(sfApiUrl(`${CUSTOM_API}?name=${encodeURIComponent(name)}`), {
      method: "DELETE",
    });
    return r.ok;
  } catch (e) {
    console.warn("[SF Canvas Size Preset]", e);
    return false;
  }
}

// ── 管理弹窗（左已存定义列表 + 右编辑器；sf_popup 三关闭）────────────────
function canvasScale() {
  try {
    return app.canvas?.ds?.scale ?? 1;
  } catch (e) {
    return 1;
  }
}

function openManager(node, anchor) {
  if (document.getElementById(CUSTOM_OVERLAY_ID)) return;
  injectCSS();

  const overlay = el("div", "sf-cs-overlay");
  overlay.id = CUSTOM_OVERLAY_ID;
  document.body.appendChild(overlay);

  const dialog = el("div", "sf-cs-dialog");
  dialog.appendChild(el("div", "sf-cs-title", "自定义分辨率预设"));
  overlay.appendChild(dialog);

  const body = el("div", "sf-cs-body");
  const listWrap = el("div", "sf-cs-listwrap");
  listWrap.appendChild(el("label", "sf-cs-label", "已存定义"));
  const listEl = el("div", "sf-cs-list");
  listWrap.appendChild(listEl);
  body.appendChild(listWrap);

  const editor = el("div", "sf-cs-editor");
  editor.appendChild(el("label", "sf-cs-label", "名称"));
  const nameInput = el("input", "sf-cs-input");
  nameInput.placeholder = "定义名称（禁括号）";
  nameInput.maxLength = 200;
  editor.appendChild(nameInput);

  const dims = el("div", "sf-cs-dims");
  const wBox = el("div", "sf-cs-dim");
  wBox.appendChild(el("label", "sf-cs-label", "宽"));
  const wInput = el("input", "sf-cs-input");
  wInput.type = "number";
  wInput.min = "1";
  wInput.step = "1";
  wBox.appendChild(wInput);
  const hBox = el("div", "sf-cs-dim");
  hBox.appendChild(el("label", "sf-cs-label", "高"));
  const hInput = el("input", "sf-cs-input");
  hInput.type = "number";
  hInput.min = "1";
  hInput.step = "1";
  hBox.appendChild(hInput);
  dims.append(wBox, el("div", "sf-cs-colon", ":"), hBox);
  editor.appendChild(dims);

  const editOps = el("div", "sf-cs-editops");
  const saveBtn = el("button", "sf-cs-btn pri", "保存");
  const delBtn = el("button", "sf-cs-btn danger", "删除");
  editOps.append(saveBtn, delBtn);
  editor.appendChild(editOps);
  editor.appendChild(el("div", "sf-cs-hint", "点击左侧条目回填，双击直接套用到本节点。"));
  body.appendChild(editor);
  dialog.appendChild(body);

  const msgEl = el("div", "sf-cs-msg");
  dialog.appendChild(msgEl);

  const foot = el("div", "sf-cs-foot");
  const closeBtn = el("button", "sf-cs-btn", "关闭");
  foot.appendChild(closeBtn);
  dialog.appendChild(foot);

  let presets = [];
  let selectedIndex = -1;
  let routeOk = true;

  const setMsg = (text, bad) => {
    msgEl.textContent = text || "";
    msgEl.classList.toggle("bad", !!bad);
  };

  const markDims = (bad) => {
    const color = bad ? "#e74c3c" : "";
    wInput.style.borderColor = color;
    hInput.style.borderColor = color;
  };

  const readFields = () => {
    const w = Number(wInput.value);
    const h = Number(hInput.value);
    const ok = validDim(w) && validDim(h);
    markDims(!ok);
    return ok ? { w, h } : null;
  };

  const applyToNode = (name, w, h) => {
    const modelWidget = node?.widgets?.find((x) => x.name === "model");
    const resWidget = node?.widgets?.find((x) => x.name === "resolution");
    if (!modelWidget || !resWidget) return;
    // 自定义条目属于 Custom Resolution 伪模型：套用时一并切换 model
    if (_official?.custom_model) modelWidget.value = _official.custom_model;
    syncNode(node, true); // 保留当前值（避免重建瞬间回退），随后写入选中项
    resWidget.value = customOptionValue(name, w, h);
    node.setDirtyCanvas?.(true, true);
    close();
  };

  const renderList = () => {
    listEl.replaceChildren();
    if (presets.length === 0) {
      listEl.appendChild(el("div", "sf-cs-empty", routeOk ? "（暂无自定义）" : "预设库不可用"));
      return;
    }
    presets.forEach((p, idx) => {
      const item = el("div", "sf-cs-item" + (idx === selectedIndex ? " sel" : ""));
      item.textContent = `${p.name}  (${p.w}x${p.h})`;
      item.title = `${p.name} ${p.w}x${p.h}`;
      item.addEventListener("click", () => {
        selectedIndex = idx;
        nameInput.value = p.name;
        wInput.value = String(p.w);
        hInput.value = String(p.h);
        markDims(false);
        renderList();
      });
      item.addEventListener("dblclick", () => applyToNode(p.name, p.w, p.h));
      listEl.appendChild(item);
    });
  };

  const reload = async () => {
    const list = await refetchCustom();
    routeOk = list !== null;
    presets = _custom;
    selectedIndex = -1;
    renderList();
    syncAll(); // 重建所有同类节点 options（含新增/删除项）
  };

  const save = async () => {
    const name = nameInput.value.trim();
    if (!validCustomName(name)) {
      nameInput.style.borderColor = "#e74c3c";
      setMsg("名称非法（非空、无括号/路径分隔符、≤200 字）", true);
      return;
    }
    nameInput.style.borderColor = "";
    const v = readFields();
    if (!v) {
      setMsg("宽/高必须为 1..32768 的整数", true);
      return;
    }
    if (!(await apiSave(name, v.w, v.h))) {
      setMsg("保存失败（后端路由不可用？重启 ComfyUI 后重试）", true);
      return;
    }
    setMsg(`已保存「${name}」`);
    await reload();
  };

  const remove = async () => {
    if (selectedIndex < 0 || selectedIndex >= presets.length) {
      setMsg("请先在左侧选择要删除的定义", true);
      return;
    }
    const name = presets[selectedIndex].name;
    if (!confirm(`删除自定义分辨率「${name}」？`)) return;
    if (!(await apiDelete(name))) {
      setMsg("删除失败（后端路由不可用？）", true);
      return;
    }
    setMsg(`已删除「${name}」`);
    nameInput.value = "";
    await reload();
  };

  saveBtn.addEventListener("click", save);
  delBtn.addEventListener("click", remove);
  closeBtn.addEventListener("click", () => close());
  for (const block of [nameInput, wInput, hInput]) {
    block.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter") {
        e.preventDefault();
        save();
      } else if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    });
    block.addEventListener("input", () => {
      block.style.borderColor = "";
      setMsg("");
    });
  }

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    overlay.remove();
  };
  attachPopupDismiss(overlay, { onClose: close });

  // 定位到管理按钮附近，钳回视口；无 anchor 时居中。
  if (anchor) {
    const r = anchor.getBoundingClientRect();
    dialog.style.left = `${r.left}px`;
    dialog.style.top = `${r.bottom + 4}px`;
  }
  clampToViewport(dialog, { scale: canvasScale() });

  renderList();
  reload();
}

// ── 注册 ────────────────────────────────────────────────────────────────
app.registerExtension({
  name: "sfnodes.CanvasSizePreset",

  nodeCreated(node) {
    if (node.comfyClass !== CLASS) return;

    const modelWidget = node.widgets?.find((w) => w.name === "model");
    const resolutionWidget = node.widgets?.find((w) => w.name === "resolution");
    if (!modelWidget || !resolutionWidget) return;

    installManageButton(node);

    const originalCallback = modelWidget.callback;
    modelWidget.callback = function (value) {
      if (originalCallback) {
        originalCallback.call(this, value);
      }
      syncNode(node);
    };

    // 加载/恢复工作流：widget 值恢复发生在 onAfterGraphConfigured，combo
    // options 需按恢复的 model 值重建（值已恢复，不触发 callback）。等两份
    // 数据都就绪再同步，避免自定义库未到时把恢复的自定义值误回退；preserve
    // 保证库离线/条目已删时旧值仍保留（可执行）。
    node.onAfterGraphConfigured = () => {
      Promise.all([loadOfficial(), loadCustom()]).then(() => syncNode(node, true));
    };

    // 数据就绪后校准（默认模型静态选项已正确，此步幂等兜底）
    loadOfficial().then(() => syncNode(node, true));
    loadCustom().then(() => syncNode(node, true));
  },
});

// ── CSS ─────────────────────────────────────────────────────────────────
function injectCSS() {
  injectCSSOnce(
    "sf-cs-css",
    `
.sf-cs-manage{width:100%;height:auto;cursor:pointer;background:var(--sf-surface);border:1px solid var(--sf-border-soft);color:var(--sf-acc,#f66744);border-radius:4px;padding:2px 6px;font-size:12px;text-align:center;flex-shrink:0}
.sf-cs-manage:hover{background:var(--sf-surface-hover)}
.sf-cs-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:9999}
.sf-cs-dialog{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--sf-panel-bg);border:1px solid var(--sf-border-soft);border-radius:6px;padding:12px 14px;box-shadow:0 4px 20px rgba(0,0,0,.5);width:480px;max-width:92vw;color:var(--sf-text);font-size:13px;box-sizing:border-box}
.sf-cs-title{font-weight:600;margin-bottom:10px;color:var(--sf-acc,#f66744)}
.sf-cs-body{display:flex;gap:12px;align-items:stretch}
.sf-cs-listwrap{width:190px;display:flex;flex-direction:column;min-width:0}
.sf-cs-label{color:var(--sf-text-dim);font-size:10px;margin-bottom:3px}
.sf-cs-list{flex:1;min-height:160px;max-height:260px;overflow-y:auto;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;padding:3px}
.sf-cs-empty{color:var(--sf-text-faint);font-size:11px;padding:8px 4px;text-align:center}
.sf-cs-item{padding:4px 6px;border-radius:3px;cursor:pointer;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}
.sf-cs-item:hover{background:var(--sf-surface-hover)}
.sf-cs-item.sel{background:#3a5f8a;color:#fff}
.sf-cs-editor{flex:1;display:flex;flex-direction:column;gap:8px;min-width:0}
.sf-cs-input{width:100%;padding:5px;background:var(--sf-input-bg);border:1px solid var(--sf-border-soft);border-radius:3px;color:var(--sf-text);font-size:13px;box-sizing:border-box}
.sf-cs-dims{display:flex;gap:10px;align-items:flex-end}
.sf-cs-dim{display:flex;flex-direction:column;flex:1}
.sf-cs-colon{color:var(--sf-text-faint);font-size:16px;padding-bottom:6px}
.sf-cs-editops{display:flex;gap:8px}
.sf-cs-hint{color:var(--sf-text-faint);font-size:10px;line-height:1.4}
.sf-cs-btn{cursor:pointer;background:var(--sf-surface);border:1px solid var(--sf-border-soft);color:var(--sf-text);border-radius:4px;padding:4px 12px;font-size:12px}
.sf-cs-btn:hover{background:var(--sf-surface-hover)}
.sf-cs-btn.pri{background:var(--sf-acc,#f66744);color:#fff;border-color:transparent}
.sf-cs-btn.danger{color:#e57373;border-color:#7a3a3a}
.sf-cs-msg{color:var(--sf-positive);font-size:12px;min-height:16px;margin-top:8px}
.sf-cs-msg.bad{color:#e57373}
.sf-cs-foot{display:flex;justify-content:flex-end;margin-top:10px}
`
  );
}
