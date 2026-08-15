// SF Prompt Reader - read the positive prompt saved in a PNG's metadata.
//
// 复刻 Pixaroma Prompt Reader（node color 功能不复刻）。UX 与 Load Image SF 的
// 输入流一致（上传按钮、文件下拉、拖放），但渲染一个只读文本区而非图片预览。
// 提取的文本通过 /api/sfnodes/prompt_reader/extract 在每次文件变化时实时获取，
// 用户在运行前即可看到结果。
//
// 持久化：filename + 提取文本存 node.properties.promptReaderState，工作流
// 保存 / 重载 / Vue 标签页切换后读出版本原样恢复（Pattern #9 / Preview Pattern
// #4，同 SFPauseImage）。

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { sfApiUrl, isGraphLoading } from "./sf_common.js";

const STATE_PROP = "promptReaderState";

// Tracks the currently-selected Prompt Reader node so the global PageUp /
// PageDown keydown listener can route the step to the right one. Cleared
// on deselect or removal.
let _activePromptReaderNode = null;

// 加载路径守卫（wrap app.loadGraphData + 300ms 尾窗）由 sf_common.js 顶层
// 统一安装（幂等单例），isGraphLoading 从公共模块 import。

// ── State helpers ──────────────────────────────────────────────────────────

function readState(node) {
  return node.properties?.[STATE_PROP] || {};
}

function writeState(node, patch) {
  if (!node.properties) node.properties = {};
  const cur = node.properties[STATE_PROP] || {};
  node.properties[STATE_PROP] = { ...cur, ...patch };
}

// ── 目录切换（input/ ↔ output/）────────────────────────────────────────────

// Peel ComfyUI's type annotation off a combo value. A value can be
// "sub/out.png [output]" - the annotation names the folder the file lives
// in and must never reach the file-name display / split.
function splitTypeAnnotation(value) {
  const s = String(value ?? "");
  const m = s.match(/^(.*?)\s*\[(input|output|temp)\]\s*$/i);
  if (!m) return { name: s, type: "input" };
  return { name: m[1], type: m[2].toLowerCase() };
}

// 当前读取目录：state.folder === "output" → output/，否则 input/。
// （字段名避开 applyResult 写入的 source=comfyui/a1111 提取来源）
function currentSource(node) {
  return readState(node).folder === "output" ? "output" : "input";
}

function updateSourceButton(node, type) {
  const root = node._sfPrRoot;
  if (!root) return;
  const btn = root.querySelector('[data-role="source"]');
  if (!btn) return;
  const isOut = type === "output";
  btn.textContent = isOut ? "OUT" : "IN";
  btn.title = isOut
    ? "Reading from output/ · click to switch to input/"
    : "Reading from input/ · click to switch to output/";
  btn.classList.toggle("sf-pr-srcbtn-active", isOut);
}

// 拉取目录文件列表（纯相对路径，正斜杠）；失败返回 null。
async function fetchMediaList(type) {
  const url = sfApiUrl(`/api/sfnodes/prompt_reader/list?type=${encodeURIComponent(type)}`);
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const json = await resp.json();
    return Array.isArray(json) ? json : null;
  } catch (e) {
    return null;
  }
}

// 列表 → combo options 值：output 项拼 [output] 注解（get_annotated_filepath
// 原生解析，extract 路由的 allowed_roots 已含 output/）。
function buildSourceValues(list, type) {
  return (list || []).map((f) => (type === "output" ? f + " [output]" : f));
}

// 切换目录：拉列表 → 替换 options → 持久化 source → 选中 selectFile（若在新
// 列表中）否则第一项并提取；列表为空则清空值 + 提示。
async function switchSource(node, target, selectFile = null) {
  const w = node._sfPrImageWidget;
  if (!w) return;
  const list = await fetchMediaList(target);
  if (list == null) {
    applyResult(node, { found: false, message: "Could not load the file list." }, w.value || "");
    return;
  }
  const values = buildSourceValues(list, target);
  if (!w.options) w.options = {};
  w.options.values = values;
  writeState(node, { folder: target });
  updateSourceButton(node, target);
  let picked = null;
  if (selectFile && values.includes(selectFile)) {
    picked = selectFile;
  } else if (values.length) {
    picked = values[0];
  }
  if (picked) {
    w.value = picked;
    node._sfPrSelectedFilename = picked;
    onImageChanged(node);
  } else {
    w.value = "";
    node._sfPrSelectedFilename = "";
    applyResult(
      node,
      { found: false, message: `No media files in the ${target} folder.` },
      ""
    );
    refreshDropdown(node);
  }
}

// 上传/拖拽落盘的文件在 input/：若当前是 output 模式，切回 input 并选中新
// 文件（避免"选了但列表里没有"）。返回 true 表示已切回（调用方不再重复提取）。
async function ensureSourceIsInput(node, saved) {
  if (currentSource(node) !== "output") return false;
  await switchSource(node, "input", saved);
  return true;
}

// ── CSS injection ──────────────────────────────────────────────────────────

let _cssInjected = false;
function injectCSS() {
  if (_cssInjected) return;
  _cssInjected = true;
  const style = document.createElement("style");
  style.id = "sf-pr-css";
  style.textContent = `
    /* Nodes 2.0 renders its own .image-preview panel (fed by ComfyUI's
       internal node-preview state, NOT node.imgs which we lock to []). It
       goes stale on programmatic file changes and we don't want an image
       preview on this text-readout node anyway. Hide it, scoped to our node
       via :has() so no other node is affected. Legacy has no .lg-node /
       .image-preview, so this rule is a no-op there. */
    .lg-node:has(.sf-pr-root) .image-preview { display: none !important; }

    .sf-pr-root {
      width: 100%;
      box-sizing: border-box;
      padding: 8px;
      background: #2a2a2a;
      border-radius: 4px;
      color: #ddd;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 11px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sf-pr-upload-btn {
      width: 100%;
      background: var(--sf-acc, #f66744);
      border: none;
      border-radius: 4px;
      padding: 9px 8px;
      font-size: 11px;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      font-family: inherit;
      transition: background 0.08s;
    }
    .sf-pr-upload-btn:hover { background: #ff7e5a; }
    /* File row: [◀] [ dropdown ] [▶] - mirrors Load Image SF. */
    .sf-pr-filerow {
      display: flex;
      gap: 4px;
      align-items: stretch;
    }
    .sf-pr-filerow .sf-pr-dropdown { flex: 1; min-width: 0; }
    .sf-pr-srcbtn {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      color: #aaa;
      font-size: 9px;
      font-weight: 700;
      cursor: pointer;
      width: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      flex-shrink: 0;
      transition: background 0.08s, border-color 0.08s, color 0.08s;
    }
    .sf-pr-srcbtn:hover { border-color: var(--sf-acc, #f66744); color: var(--sf-acc, #f66744); }
    .sf-pr-srcbtn-active { background: var(--sf-acc, #f66744); border-color: var(--sf-acc, #f66744); color: #fff; }
    .sf-pr-srcbtn-active:hover { color: #fff; }
    .sf-pr-nav {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      color: #aaa;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      width: 26px;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      transition: background 0.08s, border-color 0.08s, color 0.08s;
      flex-shrink: 0;
    }
    .sf-pr-nav:hover:not(.disabled) { border-color: var(--sf-acc, #f66744); color: var(--sf-acc, #f66744); }
    .sf-pr-nav:active:not(.disabled) { background: var(--sf-acc, #f66744); color: #fff; }
    .sf-pr-nav.disabled { opacity: 0.3; cursor: default; }
    .sf-pr-dropdown .counter {
      color: #777;
      font-size: 9px;
      margin-left: 6px;
      flex-shrink: 0;
    }
    .sf-pr-popup-section {
      padding: 4px 10px 3px;
      font-size: 9px;
      color: #777;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      background: #161616;
      border-bottom: 1px solid #2a2a2a;
      user-select: none;
    }
    .sf-pr-popup-section:not(:first-child) { border-top: 1px solid #2a2a2a; }
    .sf-pr-hint {
      font-size: 9px;
      color: #777;
      text-align: center;
      letter-spacing: 0.3px;
      margin-top: -3px;
    }
    /* Wired state: a filename is connected, so the node reads that image and
       the picker is overridden. Dim the picker controls (still clickable - a
       click / drop takes over and disconnects the wire) and turn the hint
       orange, naming the connected image. */
    .sf-pr-wired .sf-pr-upload-btn,
    .sf-pr-wired .sf-pr-filerow { opacity: 0.45; }
    .sf-pr-hint.sf-pr-wired-hint { color: var(--sf-acc, #f66744); font-weight: 600; }
    .sf-pr-dropdown {
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      padding: 6px 8px;
      font-size: 11px;
      color: #ccc;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }
    .sf-pr-dropdown:hover { border-color: #666; }
    .sf-pr-dropdown .name {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .sf-pr-dropdown .arrow { color: var(--sf-acc, #f66744); font-size: 10px; margin-left: 6px; }
    .sf-pr-status {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      color: #888;
      padding: 0 2px;
    }
    .sf-pr-status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #555;
      flex-shrink: 0;
    }
    .sf-pr-status.found .sf-pr-status-dot { background: var(--sf-acc, #f66744); }
    .sf-pr-status.empty .sf-pr-status-dot { background: #555; }
    .sf-pr-status-label { flex: 1; }
    .sf-pr-copy {
      background: var(--sf-acc, #f66744);
      border: 1px solid var(--sf-acc, #f66744);
      color: #fff;
      font-weight: 600;
      border-radius: 3px;
      padding: 2px 10px;
      font-size: 10px;
      cursor: pointer;
      font-family: inherit;
      transition: background 0.08s;
    }
    .sf-pr-copy:hover { background: #ff7e5a; border-color: #ff7e5a; }
    .sf-pr-copy:disabled {
      opacity: 0.35; cursor: default;
    }
    .sf-pr-copy:disabled:hover { background: var(--sf-acc, #f66744); border-color: var(--sf-acc, #f66744); }
    .sf-pr-readout {
      width: 100%;
      box-sizing: border-box;
      background: #1d1d1d;
      border: 1px solid #333;
      border-radius: 4px;
      padding: 8px;
      color: #ddd;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.45;
      resize: none;
      min-height: 80px;
      flex: 1;
      outline: none;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .sf-pr-readout.empty {
      color: #777;
      font-style: italic;
      font-family: inherit;
    }
    .sf-pr-popup {
      position: fixed;
      z-index: 99999;
      background: #1d1d1d;
      border: 1px solid #444;
      border-radius: 4px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      max-height: 300px;
      overflow-y: auto;
      font-size: 11px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      color: #ccc;
      min-width: 200px;
    }
    .sf-pr-popup-item {
      padding: 6px 10px;
      cursor: pointer;
      border-bottom: 1px solid #2a2a2a;
    }
    .sf-pr-popup-item:hover { background: #2a2a2a; }
    .sf-pr-popup-item.active { color: var(--sf-acc, #f66744); font-weight: 600; }
    .sf-pr-popup-empty { padding: 8px; color: #666; }
  `;
  document.head.appendChild(style);
}

// ── DOM build ──────────────────────────────────────────────────────────────

function buildRoot() {
  const root = document.createElement("div");
  root.className = "sf-pr-root";

  const btnUpload = document.createElement("button");
  btnUpload.type = "button";
  btnUpload.className = "sf-pr-upload-btn";
  btnUpload.dataset.role = "upload";
  btnUpload.textContent = "Upload Image / Video";
  root.appendChild(btnUpload);

  const hint = document.createElement("div");
  hint.className = "sf-pr-hint";
  hint.textContent = "or drag a PNG / MP4 here";
  hint.dataset.role = "hint";
  root.appendChild(hint);

  // File row: [IN/OUT] [◀] [ dropdown ] [▶] - the source toggle switches the
  // file list between input/ and output/ (mirrors Load Image Browser).
  const fileRow = document.createElement("div");
  fileRow.className = "sf-pr-filerow";

  const srcBtn = document.createElement("button");
  srcBtn.type = "button";
  srcBtn.className = "sf-pr-srcbtn";
  srcBtn.dataset.role = "source";
  srcBtn.title = "Reading from input/ · click to switch to output/";
  srcBtn.textContent = "IN";

  const prev = document.createElement("button");
  prev.type = "button";
  prev.className = "sf-pr-nav";
  prev.dataset.role = "prev";
  prev.title = "Previous image (PageUp)";
  prev.textContent = "◀";

  const dd = document.createElement("div");
  dd.className = "sf-pr-dropdown";
  dd.dataset.role = "dropdown";
  dd.innerHTML = `<span class="name">— no image —</span><span class="counter" data-role="counter"></span><span class="arrow">▾</span>`;

  const next = document.createElement("button");
  next.type = "button";
  next.className = "sf-pr-nav";
  next.dataset.role = "next";
  next.title = "Next image (PageDown)";
  next.textContent = "▶";

  fileRow.append(srcBtn, prev, dd, next);
  root.appendChild(fileRow);

  // Order: dropdown → readout → status (info + Copy). The status pill is
  // placed AFTER the readout because (a) the user reads the prompt first
  // and the info chip below acts as a small caption, and (b) the Copy
  // button colocated with the status sits naturally underneath the text
  // it copies.
  const readout = document.createElement("textarea");
  readout.className = "sf-pr-readout empty";
  readout.readOnly = true;
  readout.value = "";
  readout.placeholder = "The positive prompt will appear here.";
  readout.dataset.role = "readout";
  root.appendChild(readout);

  const status = document.createElement("div");
  status.className = "sf-pr-status";
  status.dataset.role = "status";
  status.innerHTML = `
    <span class="sf-pr-status-dot"></span>
    <span class="sf-pr-status-label">Pick an image to read its prompt.</span>
    <button class="sf-pr-copy" data-role="copy" disabled>Copy</button>
  `;
  root.appendChild(status);

  return root;
}

// ── Native combo hiding ────────────────────────────────────────────────────

function hideNativeImageCombo(node) {
  let imageWidget = null;
  for (const w of (node.widgets || [])) {
    if (!w) continue;
    if (w.name === "image") imageWidget = w;
    w.hidden = true;
    w.computeSize = () => [0, -4];
    if (!w.options) w.options = {};
    w.options.canvasOnly = true;
    if (w.element) w.element.style.display = "none";
  }
  requestAnimationFrame(() => {
    for (const w of (node.widgets || [])) {
      if (!w || w.name === "sf_prompt_reader_ui") continue;
      const _el = w.element || w.inputEl; // prefer .element; .inputEl only on old builds (no deprecation warning)
      if (_el) _el.style.display = "none";
    }
  });
  return imageWidget;
}

// ── Backend calls ──────────────────────────────────────────────────────────

async function uploadImage(node, file, hintName = null) {
  const form = new FormData();
  if (file instanceof Blob && !(file instanceof File) && hintName) {
    form.append("image", file, hintName);
  } else {
    form.append("image", file);
  }
  const resp = await fetch("/upload/image", { method: "POST", body: form });
  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    throw new Error(`Upload failed (${resp.status}): ${t || resp.statusText}`);
  }
  const json = await resp.json();
  const saved = json?.name;
  if (!saved) throw new Error("Upload succeeded but no filename returned");

  const w = node._sfPrImageWidget || (node.widgets || []).find((x) => x.name === "image");
  if (w) {
    if (!w.options) w.options = {};
    const values = w.options.values || [];
    if (!values.includes(saved)) {
      values.push(saved);
      values.sort();
      w.options.values = values;
    }
    w.value = saved;
    // Defensive cache - same pattern used by every other pick path
    // (dropdown click, arrow nav, native drag-drop).
    node._sfPrSelectedFilename = saved;
  }
  node.graph?.setDirtyCanvas?.(true, true);
  return saved;
}

function pickAndUpload(node) {
  return new Promise((resolve, reject) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "image/*,video/*";
    inp.style.display = "none";
    inp.addEventListener("change", async () => {
      const file = inp.files?.[0];
      if (!file) { inp.remove(); resolve(null); return; }
      try {
        const saved = await uploadImage(node, file);
        resolve(saved);
      } catch (e) {
        reject(e);
      } finally {
        inp.remove();
      }
    });
    document.body.appendChild(inp);
    inp.click();
  });
}

async function extractPrompt(filename) {
  if (!filename) return { found: false, message: "No image selected." };
  const url = sfApiUrl(`/api/sfnodes/prompt_reader/extract?filename=${encodeURIComponent(filename)}`);
  try {
    const resp = await fetch(url);
    if (!resp.ok) return { found: false, message: `Server error (${resp.status})` };
    return await resp.json();
  } catch (e) {
    return { found: false, message: `Network error: ${e.message}` };
  }
}

// Per-node monotonic request id - rapid file-combo clicks would otherwise
// race, and an out-of-order response could stamp stale text on the readout.
// `onImageChanged` bumps node._sfPrReqId before fetching, then checks the
// id is still current before applying the result.
function nextReqId(node) {
  node._sfPrReqId = (node._sfPrReqId | 0) + 1;
  return node._sfPrReqId;
}

// ── Readout rendering ──────────────────────────────────────────────────────

function applyResult(node, result, storeFilename) {
  const root = node._sfPrRoot;
  // Guard: node may have been removed mid-fetch (onRemoved nulls _sfPrRoot).
  // Without this guard the in-flight response would still try to write to
  // detached DOM and persist state on a deleted node.
  if (!root || !root.isConnected) return;
  const readout = root.querySelector('[data-role="readout"]');
  const status = root.querySelector('[data-role="status"]');
  const statusLabel = status?.querySelector(".sf-pr-status-label");
  const copy = root.querySelector('[data-role="copy"]');
  if (!readout || !status || !statusLabel || !copy) return;

  status.classList.remove("found", "empty");
  if (result?.found) {
    readout.value = result.text || "";
    readout.classList.remove("empty");
    status.classList.add("found");
    const src = result.source === "a1111"
      ? "Found · A1111 / Forge metadata"
      : "Found · ComfyUI workflow";
    statusLabel.textContent = src;
    copy.disabled = !readout.value;
  } else {
    readout.value = "";
    readout.classList.add("empty");
    status.classList.add("empty");
    statusLabel.textContent = result?.message || "No prompt found in this image.";
    copy.disabled = true;
  }

  // Persist (Pattern #9 / Preview Pattern #4) so reload / Vue tab switching
  // brings the same readout back without re-hitting the server. storeFilename
  // is the source that produced this readout (the picker's value normally, or
  // the followed filename when a wire drives the read).
  writeState(node, {
    filename: storeFilename != null ? storeFilename : (node._sfPrImageWidget?.value || ""),
    found: !!result?.found,
    text: result?.text || "",
    message: result?.message || "",
    source: result?.source || null,
  });
}

function restoreFromState(node) {
  const s = readState(node);
  if (!s || !s.filename) return;
  if (s.found) {
    applyResult(node, { found: true, text: s.text, source: s.source }, s.filename);
  } else if (s.message) {
    applyResult(node, { found: false, message: s.message }, s.filename);
  }
}

function refreshDropdown(node) {
  const root = node._sfPrRoot;
  if (!root) return;
  const w = node._sfPrImageWidget;
  const nameEl = root.querySelector('[data-role="dropdown"] .name');
  const counter = root.querySelector('[data-role="counter"]');
  const value = w?.value || "";
  // 显示剥离 [output] 注解（"sub/out.png [output]" 只显示文件名路径）
  const display = value ? splitTypeAnnotation(value).name : "";
  if (nameEl) nameEl.textContent = display ? display : "— no image —";
  const values = w?.options?.values || [];
  if (counter) {
    if (value && values.length > 1) {
      const idx = values.indexOf(value);
      counter.textContent = idx >= 0 ? `${idx + 1} / ${values.length}` : "";
    } else {
      counter.textContent = "";
    }
  }
  const prev = root.querySelector('[data-role="prev"]');
  const next = root.querySelector('[data-role="next"]');
  const disabled = values.length < 2;
  if (prev) prev.classList.toggle("disabled", disabled);
  if (next) next.classList.toggle("disabled", disabled);
}

// Split "Studio1/cat.png" into {subfolder, filename}. Mirrors the same
// helper in sf_load_image_api.js so the two nodes share grouping behaviour.
function splitPath(path) {
  if (!path) return { subfolder: "", filename: "" };
  const norm = String(path).replace(/\\/g, "/");
  const idx = norm.lastIndexOf("/");
  if (idx < 0) return { subfolder: "", filename: norm };
  return { subfolder: norm.slice(0, idx), filename: norm.slice(idx + 1) };
}

// Step the selected image by `offset` (+1 / -1), wrapping. Routes through
// the same callback path as a manual pick so the extract refresh happens.
function pickByOffset(node, offset) {
  const w = node._sfPrImageWidget;
  if (!w) return;
  const values = w.options?.values || [];
  if (values.length === 0) return;
  const cur = values.indexOf(w.value);
  let next;
  if (cur < 0) next = offset > 0 ? 0 : values.length - 1;
  else next = ((cur + offset) % values.length + values.length) % values.length;
  w.value = values[next];
  node._sfPrSelectedFilename = values[next];
  // A manual step wins over any connected filename. If it severed a wire, the
  // disconnect's onConnectionsChange cascade already refreshed the readout, so
  // only extract here when nothing was severed (avoids a double fetch).
  if (!takeOverFromWire(node)) onImageChanged(node);
}

// Core extract-and-render for an explicit filename. Used by BOTH the picker
// path (onImageChanged) and the wired-follow path (pollFollowOnce). storeName
// is persisted as the readout's source (Pattern #9).
async function runExtract(node, filename, storeName) {
  refreshDropdown(node);
  if (!filename) {
    applyResult(node, { found: false, message: "Pick an image to read its prompt." }, storeName || "");
    return;
  }
  // Show a transient loading state.
  const statusLabel = node._sfPrRoot?.querySelector(".sf-pr-status-label");
  if (statusLabel) statusLabel.textContent = "Reading metadata...";
  const myId = nextReqId(node);
  const result = await extractPrompt(filename);
  // Bail if a newer request has been kicked off in the meantime - prevents
  // out-of-order responses from stamping stale text on the readout.
  if (node._sfPrReqId !== myId) return;
  applyResult(node, result, storeName != null ? storeName : filename);
}

// Picker path: read the node's own selected image.
async function onImageChanged(node) {
  const filename = node._sfPrImageWidget?.value || "";
  return runExtract(node, filename, filename);
}

// ── Wired filename input (Load Image SF → Prompt Reader) ──────────────────────
// When the optional `filename` input is connected, the node reads THAT image
// and ignores its own picker. The readout follows the upstream live (Vue Compat
// #1: no onDraw tick, so we poll). Any manual pick / upload / drop disconnects
// the wire and hands control back to the picker (Image Resize auto-swap idiom).

const INPUT_TYPE = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT != null) ? LiteGraph.INPUT : 1;

function isFilenameWired(node) {
  const inp = node.inputs?.find((i) => i && i.name === "filename");
  return !!(inp && inp.link != null);
}

// The CURRENT filename on the wire, read live from the upstream node's own
// state so scrubbing Load Image updates us before Run. Returns null when the
// upstream exposes no live filename (e.g. a plain String node computed only at
// run time) - the hint then says "loads on run" and the backend does the real
// read on execute.
function readWiredFilename(node) {
  const inp = node.inputs?.find((i) => i && i.name === "filename");
  if (!inp || inp.link == null) return null;
  const graph = node.graph;
  if (!graph) return null;
  let l = graph.links?.[inp.link];
  if (!l && typeof graph.links?.get === "function") l = graph.links.get(inp.link);
  if (!l) return null;
  const up = graph.getNodeById(l.origin_id);
  if (!up) return null;
  // Load Image SF exposes the selected filename WITH extension + subfolder.
  if (typeof up._sfLiSelectedFilename === "string" && up._sfLiSelectedFilename) {
    return up._sfLiSelectedFilename;
  }
  // A chained Prompt Reader.
  if (typeof up._sfPrSelectedFilename === "string" && up._sfPrSelectedFilename) {
    return up._sfPrSelectedFilename;
  }
  // Generic image-picker fallback: a widget literally named "image".
  const iw = (up.widgets || []).find((w) => w && w.name === "image");
  if (iw && typeof iw.value === "string" && iw.value) return iw.value;
  return null;
}

function disconnectInputByName(node, name) {
  const i = node.inputs?.findIndex((inp) => inp && inp.name === name);
  if (i != null && i >= 0 && node.inputs[i]?.link != null) {
    node.disconnectInput(i);
    return true;
  }
  return false;
}

// Called before every MANUAL pick (upload / drop / dropdown / arrow). If a
// filename wire is connected, drop it so the manual choice takes over. The
// disconnect fires onConnectionsChange → refreshWiredState, which removes the
// lock UI, stops the follow timer, and re-reads the picker.
function takeOverFromWire(node) {
  return disconnectInputByName(node, "filename");
}

function baseNameOf(path) {
  if (!path) return "";
  const n = String(path).replace(/\\/g, "/");
  const i = n.lastIndexOf("/");
  return i < 0 ? n : n.slice(i + 1);
}

// Toggle the "driven by a wire" look: dim the picker + orange hint naming the
// connected image.
function setWiredUI(node, wired, followed) {
  const root = node._sfPrRoot;
  if (!root) return;
  root.classList.toggle("sf-pr-wired", wired);
  const hint = root.querySelector('[data-role="hint"]');
  if (!hint) return;
  if (wired) {
    hint.classList.add("sf-pr-wired-hint");
    hint.textContent = followed
      ? `\u{1F517} Reading ${baseNameOf(followed)} · pick or drop to take over`
      : "\u{1F517} Connected · the prompt loads when you run";
  } else {
    hint.classList.remove("sf-pr-wired-hint");
    hint.textContent = "or drag a PNG / MP4 here";
  }
}

function stopFollow(node) {
  if (node._sfPrFollowTimer) {
    clearInterval(node._sfPrFollowTimer);
    node._sfPrFollowTimer = null;
  }
  node._sfPrFollowName = null;
}

function startFollow(node) {
  if (node._sfPrFollowTimer) return; // already running
  node._sfPrFollowTimer = setInterval(() => pollFollowOnce(node), 350);
}

// One follow tick: if the upstream's live filename changed, re-extract. Cheap
// no-op when unchanged.
function pollFollowOnce(node) {
  const root = node._sfPrRoot;
  // Skip a tick while the DOM widget is transiently detached (e.g. a Vue tab
  // switch) but KEEP the timer running so it resumes on re-attach. It is
  // cleared for real in onRemoved (permanent removal) and stopFollow (unwire),
  // so it never leaks for a node that is genuinely gone.
  if (!root || !root.isConnected) return;
  if (!isFilenameWired(node)) { stopFollow(node); return; }
  const name = readWiredFilename(node);
  if (name && name !== node._sfPrFollowName) {
    node._sfPrFollowName = name;
    setWiredUI(node, true, name);
    runExtract(node, name, name);
  } else if (!name && node._sfPrFollowName == null) {
    // Upstream exposes no live filename yet - show the "loads on run" hint
    // once, without clobbering an existing readout.
    setWiredUI(node, true, null);
  }
}

// Central wired-state refresh. UI + timer only (no serialized-state writes of
// its own), so it is safe during load (Vue Compat #17/#19). Called on
// connect/disconnect, on setup, and on configure.
function refreshWiredState(node) {
  if (isFilenameWired(node)) {
    // Do NOT reset _sfPrFollowName here: pollFollowOnce dedups on it, so a
    // wired node opened / configured (this runs up to 3x per load - the
    // link-restore replay plus both queueMicrotasks) extracts only ONCE. A
    // genuine disconnect->reconnect still re-extracts because stopFollow()
    // (the unwire branch) nulls it.
    setWiredUI(node, true, readWiredFilename(node));
    startFollow(node);
    pollFollowOnce(node);           // immediate first tick (extracts if changed)
  } else {
    stopFollow(node);
    setWiredUI(node, false, null);
    // Revert to the picker's own image, but never on the load path (the
    // configure replay fires connection events too; onConfigure's own
    // queueMicrotask handles the picker population there).
    if (!isGraphLoading()) onImageChanged(node);
  }
}

// ── Dropdown popup ─────────────────────────────────────────────────────────

function openDropdown(node, anchorEl) {
  const w = node._sfPrImageWidget;
  if (!w) return;
  const values = w.options?.values || [];

  document.querySelector(".sf-pr-popup")?.remove();
  const popup = document.createElement("div");
  popup.className = "sf-pr-popup";

  const rect = anchorEl.getBoundingClientRect();
  popup.style.left = `${rect.left}px`;
  popup.style.top = `${rect.bottom + 2}px`;
  popup.style.width = `${rect.width}px`;

  if (values.length === 0) {
    const empty = document.createElement("div");
    empty.className = "sf-pr-popup-empty";
    empty.textContent = "(no images uploaded yet)";
    popup.appendChild(empty);
  } else {
    // Group by subfolder: root first, then alphabetised folders. Each item
    // shows only the bare filename; the `title` attribute holds the full
    // path for hover discoverability. The [output] annotation is peeled
    // before splitting so it never lands in the subfolder / filename.
    const map = new Map();
    for (const v of values) {
      const { subfolder, filename } = splitPath(splitTypeAnnotation(v).name);
      if (!map.has(subfolder)) map.set(subfolder, []);
      map.get(subfolder).push({ full: v, name: filename });
    }
    for (const list of map.values()) list.sort((a, b) => a.name.localeCompare(b.name));
    const folders = [...map.keys()].sort((a, b) => {
      if (a === "" && b !== "") return -1;
      if (a !== "" && b === "") return 1;
      return a.localeCompare(b);
    });
    const showHeaders = folders.length > 1 || (folders.length === 1 && folders[0] !== "");
    let scrollTarget = null;
    for (const folder of folders) {
      if (showHeaders) {
        const head = document.createElement("div");
        head.className = "sf-pr-popup-section";
        head.textContent = folder === "" ? "root" : folder;
        popup.appendChild(head);
      }
      for (const entry of map.get(folder)) {
        const item = document.createElement("div");
        item.className = "sf-pr-popup-item" + (entry.full === w.value ? " active" : "");
        item.textContent = entry.name;
        item.title = entry.full;
        if (entry.full === w.value) scrollTarget = item;
        item.addEventListener("click", (e) => {
          e.stopPropagation();
          w.value = entry.full;
          node._sfPrSelectedFilename = entry.full;
          close();
          if (!takeOverFromWire(node)) onImageChanged(node);  // manual pick wins over a wire
        });
        popup.appendChild(item);
      }
    }
    if (scrollTarget) queueMicrotask(() => {
      try { scrollTarget.scrollIntoView({ block: "nearest" }); }
      catch (_e) { /* ignore */ }
    });
  }
  document.body.appendChild(popup);

  function close() {
    popup.remove();
    document.removeEventListener("mousedown", onDown, true);
    document.removeEventListener("pointerdown", onDown, true);
    document.removeEventListener("wheel", onWheel, true);
    document.removeEventListener("keydown", onKey, true);
  }
  const onDown = (e) => { if (!popup.contains(e.target)) close(); };
  const onWheel = (e) => { if (!popup.contains(e.target)) close(); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  setTimeout(() => {
    document.addEventListener("mousedown", onDown, true);
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("wheel", onWheel, true);
    document.addEventListener("keydown", onKey, true);
  }, 0);
}

// ── Setup ──────────────────────────────────────────────────────────────────

function setupNode(node) {
  injectCSS();
  const imageWidget = hideNativeImageCombo(node);
  node._sfPrImageWidget = imageWidget;

  // Suppress ComfyUI's native bottom-of-node image preview. `image_upload:
  // True` makes the framework fetch the selected file and assign it to
  // `node.imgs`, which LiteGraph then renders below the widgets. We don't
  // need that here - the readout is the whole point - so we lock `imgs` to
  // an empty array. Side effect: right-click menu items that read
  // `node.imgs[0]` (Save Image, Copy Clipspace, Open Image) become no-ops.
  // That's an acceptable tradeoff since the file is reachable directly via
  // ComfyUI's /view route from the input folder anyway.
  //
  // Probe the existing descriptor first: if a previous redefine made it
  // non-configurable or some earlier framework code assigned a value the
  // engine left non-configurable, defineProperty throws TypeError and the
  // previous try/catch was swallowing that silently. Log it once so a
  // future Vue-frontend change becomes visible in the console.
  const imgsDesc = Object.getOwnPropertyDescriptor(node, "imgs");
  if (imgsDesc && imgsDesc.configurable === false) {
    console.warn("[SFPromptReader] cannot suppress node.imgs - existing descriptor is non-configurable");
  } else {
    try {
      Object.defineProperty(node, "imgs", {
        configurable: true,
        get() { return []; },
        set(_v) { /* swallow */ },
      });
    } catch (e) {
      console.warn("[SFPromptReader] node.imgs suppression failed:", e.message);
    }
  }

  const root = buildRoot();
  node._sfPrRoot = root;

  // Load Image SF Pattern #4: measure each child's intrinsic
  // offsetHeight, but EXCLUDE the readout textarea (which has flex: 1 and
  // absorbs node-resize slack). Counting its grown offsetHeight here would
  // feed back into getMinHeight and the node would balloon every paint.
  // Treat the readout as a fixed minimum instead; the user can still drag
  // the node larger and the textarea fills the extra space, but the
  // measurement remains stable so the node can also be shrunk back down.
  const READOUT_MIN_H = 80;
  function measureHeight() {
    let total = 0;
    let visible = 0;
    for (const child of root.children) {
      const cs = window.getComputedStyle(child);
      if (cs.position === "absolute" || cs.position === "fixed") continue;
      if (cs.display === "none") continue;
      if (child.classList.contains("sf-pr-readout")) {
        total += READOUT_MIN_H;
      } else {
        total += child.offsetHeight;
      }
      visible += 1;
    }
    const padding = 16;
    const gaps = Math.max(0, visible - 1) * 8;
    return total + padding + gaps;
  }

  const _prWidget = node.addDOMWidget("sf_prompt_reader_ui", "custom", root, {
    // canvasOnly set adaptively below: true in legacy
    // (out of the Parameters tab), false in Nodes 2.0 (renders in Vue body).
    getValue: () => null,
    setValue: () => {},
    getMinHeight: measureHeight,
    margin: 4,
    serialize: false,
  });
  // Nodes 2.0 (Vue) 渲染器检测 + canvas 缩放兼容（同 sf_load_image.js）
  try {
    Object.defineProperty(_prWidget.options, "canvasOnly", {
      configurable: true,
      enumerable: true,
      get() {
        return !window.LiteGraph?.vueNodesMode;
      },
    });
  } catch (_e) {
    _prWidget.options.canvasOnly = !window.LiteGraph?.vueNodesMode;
  }

  // Default node size for fresh-on-canvas placements. LiteGraph's configure
  // (workflow restore) runs AFTER nodeCreated and overwrites node.size with
  // the saved value, so existing workflows keep whatever size the user had.
  // Only new drops get this size.
  node.size[0] = 400;
  node.size[1] = 300;

  // Wrap the image widget's callback so native drag-drop on the bottom of the
  // node, programmatic value sets, and our own picks all route through the
  // same extract refresh.
  if (imageWidget) {
    const orig = imageWidget.callback;
    imageWidget.callback = function () {
      const r = orig?.apply(this, arguments);
      if (imageWidget.value) node._sfPrSelectedFilename = imageWidget.value;
      const v = imageWidget.value;
      if (v && currentSource(node) === "output" && !v.includes("[output]")) {
        // 原生 drop 落盘的是 input 文件（无注解值）：切回 input 并选中它
        switchSource(node, "input", v);
      } else {
        // A genuine native drop / combo change is a manual pick and takes over
        // from a wire - but NEVER during load (a configure-time value restore
        // must not sever a saved wire). If the takeover severed a wire, its
        // cascade already refreshed the readout, so only extract here otherwise.
        if (isGraphLoading() || !takeOverFromWire(node)) onImageChanged(node);
      }
      return r;
    };
    // Seed the defensive cache from whatever value the widget has at setup
    // (covers saved-workflow restore where configure() landed before us).
    if (imageWidget.value) node._sfPrSelectedFilename = imageWidget.value;
  }

  // Upload button. Errors surface inline via the status pill (Note Pattern
  // #7: never use alert() inside our overlays / panels - it context-switches
  // away and can be blocked by Vue's modal layer).
  root.querySelector('[data-role="upload"]')?.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      const saved = await pickAndUpload(node);
      // 上传落盘在 input/：output 模式下自动切回 input 并选中新文件
      if (saved && !(await ensureSourceIsInput(node, saved))) {
        if (!takeOverFromWire(node)) onImageChanged(node);  // manual upload wins over a wire
      }
    } catch (err) {
      console.error("[SFPromptReader] upload failed", err);
      applyResult(node, { found: false, message: `Upload failed: ${err.message}` });
    }
  });

  // Source toggle: switch the file list between input/ and output/.
  root.querySelector('[data-role="source"]')?.addEventListener("click", (e) => {
    e.stopPropagation();
    const target = currentSource(node) === "output" ? "input" : "output";
    switchSource(node, target);
  });

  // Dropdown
  root.querySelector('[data-role="dropdown"]')?.addEventListener("click", (e) => {
    e.stopPropagation();
    openDropdown(node, e.currentTarget);
  });

  // Prev / Next arrows - flip through input/ images visually. PageUp/PageDown
  // when the node is selected do the same. Mirrors Load Image SF.
  root.querySelector('[data-role="prev"]')?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (e.currentTarget.classList.contains("disabled")) return;
    pickByOffset(node, -1);
  });
  root.querySelector('[data-role="next"]')?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (e.currentTarget.classList.contains("disabled")) return;
    pickByOffset(node, +1);
  });

  // Copy button
  root.querySelector('[data-role="copy"]')?.addEventListener("click", async (e) => {
    e.stopPropagation();
    const readout = root.querySelector('[data-role="readout"]');
    const text = readout?.value || "";
    if (!text) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        readout.select();
        document.execCommand("copy");
      }
      const btn = e.currentTarget;
      const orig = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = orig; }, 1200);
    } catch (err) {
      console.error("[SFPromptReader] copy failed", err);
    }
  });

  // Drag-drop on the DOM widget root. ComfyUI's native bottom-preview drop
  // handler also covers the node, so this is a safety net for drops landing
  // squarely over our panel.
  root.addEventListener("dragover", (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    e.stopPropagation();
  });
  root.addEventListener("drop", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    // 与 accept="image/*,video/*" 一致：图片与视频都收。type 为空（浏览器对
    // 未知扩展名如 .mkv 不给 MIME）时放行，交给后端上传决定。
    const t = file.type || "";
    if (t && !t.startsWith("image/") && !t.startsWith("video/")) return;
    try {
      const saved = await uploadImage(node, file);
      // 上传落盘在 input/：output 模式下自动切回 input 并选中新文件
      if (!(await ensureSourceIsInput(node, saved))) {
        if (!takeOverFromWire(node)) onImageChanged(node);  // dropping a file wins over a wire
      }
    } catch (err) {
      console.error("[SFPromptReader] drop upload failed", err);
      applyResult(node, { found: false, message: `Upload failed: ${err.message}` });
    }
  });

  // Initial population - defer past configure() so widget value is restored
  // (Vue Compat #8 - nodeCreated fires BEFORE configure resolves saved
  // values). We always re-extract on load rather than using the cached
  // state, so any message-text changes from an update propagate
  // to existing workflows without the user having to re-pick a file.
  queueMicrotask(() => initReadout(node, imageWidget));
}

// 初始/恢复统一入口（setupNode 与 onConfigure 共用）：
//   - filename 接线 → 跟随接线（忽略选择器）
//   - 值带 [output] 注解或保存的 source=output → 拉 output 列表恢复
//   - 否则按选择器值提取；无值 → 恢复缓存文本
function initReadout(node, imageWidget) {
  refreshDropdown(node);
  updateSourceButton(node, currentSource(node));
  // A connected filename input drives the read and overrides the picker.
  if (isFilenameWired(node)) {
    refreshWiredState(node);
    return;
  }
  const wval = imageWidget?.value || "";
  if (wval) {
    node._sfPrSelectedFilename = wval;
    if (currentSource(node) === "output" || /\[output\]\s*$/i.test(wval)) {
      // 保持当前值恢复 output 列表（值不在新列表时落到第一项）
      switchSource(node, "output", wval);
    } else {
      onImageChanged(node);
    }
  } else {
    // No image selected - restore at least the cached UI text so the
    // user sees the previous result on tab switch without a flash.
    restoreFromState(node);
  }
}

// ── Extension registration ─────────────────────────────────────────────────

app.registerExtension({
  name: "sfnodes.PromptReader",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SFPromptReader") return;
    const origCfg = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = origCfg?.apply(this, arguments);
      queueMicrotask(() => initReadout(this, this._sfPrImageWidget));
      return r;
    };

    // React to the optional `filename` input connecting / disconnecting. UI +
    // follow-timer only (no serialized-state writes here), so it is safe during
    // the configure-replay + link-restore load window (Vue Compat #17 / #19).
    // Gated to INPUT-type changes so wiring the `text` output doesn't trigger it.
    const origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, idx, connected, link, ioSlot) {
      const r = origConn?.apply(this, arguments);
      // Only react to the `filename` INPUT changing - not the `text` output,
      // and not any other input a future ComfyUI build might auto-expose.
      const slotName = ioSlot?.name || this.inputs?.[idx]?.name;
      if (type === INPUT_TYPE && slotName === "filename") {
        try { refreshWiredState(this); } catch (_e) { /* ignore */ }
      }
      return r;
    };

    // Track the active node so the global PageUp / PageDown handler knows
    // which Prompt Reader to step.
    const origSel = nodeType.prototype.onSelected;
    const origDes = nodeType.prototype.onDeselected;
    nodeType.prototype.onSelected = function () {
      _activePromptReaderNode = this;
      return origSel?.apply(this, arguments);
    };
    nodeType.prototype.onDeselected = function () {
      if (_activePromptReaderNode === this) _activePromptReaderNode = null;
      return origDes?.apply(this, arguments);
    };

    // Cleanup on node removal. The file-dropdown popup attaches FOUR
    // document-level capture-phase listeners on every open; without an
    // explicit close on removal those leak forever (closure pins the
    // popup + node alive). Bumping the per-node request id also
    // invalidates any in-flight extract response so it can't apply
    // results to a destroyed root.
    const origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      const r = origRemoved?.apply(this, arguments);
      try {
        // Trigger every open popup's close() path (their mousedown handler
        // closes when the click is outside, which fires here).
        document.querySelectorAll(".sf-pr-popup").forEach((p) => p.remove());
      } catch (_e) { /* ignore */ }
      // Stale in-flight requests after this point will all fail the
      // reqId match in onImageChanged.
      this._sfPrReqId = (this._sfPrReqId | 0) + 1;
      stopFollow(this);   // clear the wired-follow interval so it can't leak
      this._sfPrRoot = null;
      this._sfPrImageWidget = null;
      if (_activePromptReaderNode === this) _activePromptReaderNode = null;
      return r;
    };
  },

  nodeCreated(node) {
    if (node.comfyClass !== "SFPromptReader") return;
    setupNode(node);
  },
});

// Global PageUp / PageDown to step the active Prompt Reader node's image,
// matching the equivalent shortcut in Load Image SF.
window.addEventListener("keydown", (e) => {
  if (!_activePromptReaderNode) return;
  if (e.key !== "PageUp" && e.key !== "PageDown") return;
  const tag = (e.target?.tagName || "").toUpperCase();
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (e.target?.isContentEditable) return;
  e.preventDefault();
  e.stopPropagation();
  pickByOffset(_activePromptReaderNode, e.key === "PageUp" ? -1 : +1);
}, true);

// 加载守卫由 sf_common.js 顶层自动安装（幂等单例）。
