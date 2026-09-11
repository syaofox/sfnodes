// SFCharacterSelect 前端（角色多选图库，batch 输出）
// - 隐藏 SFCharacterState（{"role", "shots"} JSON）/SFCharacterPrompt STRING
//   widget 为值真源（Python "hidden" 声明，标准 widget 收集进 prompt，随
//   workflow 保存）；DOM widget 纯交互不承担值传输（规避 Vue DOMWidget value
//   setter 链，见 experience/nodes-image.md §11）
// - 单节点多角色切换：角色卡选中角色，shots 横条复选该角色不限数量图片，
//   batch 顺序按库内顺序；搜索过滤 / Reset 清空 / 选中置顶 / 悬停大预览
//   （CSS 类前缀 sf-ch- 隔离）
// - 提示词可手改：编辑框输入写入草稿 widget（空=回落分镜拼接；切换角色或改选
//   即清空，对齐 SFTextPreset text_override 语义）；后端草稿整体覆盖拼接路
// - 纯逻辑在 sf_character_lib.js

import { app } from "/scripts/app.js";
import { applyAdaptiveCanvasOnly, hideJsonWidget, injectCSSOnce, installWheelZoomPassthrough, isGraphLoading, isVueNodes, sfApiUrl } from "./sf_common.js";
import * as lib from "./sf_character_lib.js";

const NODE_TYPE = "SFCharacterSelect";
const LIST_H = 300; // 列表固定高度（滚动容器）
const TOOLS_H = 34; // 工具条 + 与列表的 gap
const PROMPT_H = 64; // 提示词编辑框固定高度
const SHOTS_H = 64; // shots 复选横条高度
const GAP = 6;
const ROOT_PAD = 8; // root padding 上下各 4
// 声称高度必须 ≥ 内容实际高度——低于内容时节点边框按声称高度绘制、
// 底部内容溢出被裁（对齐 sf_styles_selector.js §18.6 注释）
const WIDGET_H = LIST_H + TOOLS_H + PROMPT_H + SHOTS_H + GAP * 3 + ROOT_PAD + 4;
const MIN_W = 300;
const VIEW_PROP = "sfCharacterView"; // Grid/List 显示模式（随 workflow 保存，不注入 prompt）
const CARD_PREVIEW_MAX = 4; // 角色卡内预览总格数：超量时 3 图 + 1 个 "+N" 格，全正方形省空间
const POP_PREVIEW_MAX = 6; // hover 浮窗大图上限
const EMPTY_IMG =
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

function injectCSS() {
  injectCSSOnce("sf-character-css", `
.sf-ch-root{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;padding:4px 6px;position:relative;overflow:visible;}
.sf-ch-tools{display:flex;gap:6px;flex:0 0 auto;}
.sf-ch-search{flex:1;min-width:0;resize:none;font:12px sans-serif;color:#ddd;background:#1d1d1d;border:1px solid #333;border-radius:5px;padding:4px 6px;height:28px;box-sizing:border-box;outline:none;}
.sf-ch-reset{flex:0 0 auto;font:11px sans-serif;color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);border:1px solid color-mix(in srgb, var(--sf-acc, #f66744) 45%, transparent);border-radius:5px;padding:0 10px;cursor:pointer;height:28px;}
.sf-ch-reset:hover{background:color-mix(in srgb, var(--sf-acc, #f66744) 22%, transparent);}
.sf-ch-viewseg{flex:0 0 auto;display:flex;border:1px solid #444;border-radius:5px;overflow:hidden;height:28px;}
.sf-ch-viewseg button{font:11px sans-serif;color:#c8c8c8;background:#2a2a2a;border:none;padding:0 8px;cursor:pointer;}
.sf-ch-viewseg button:hover{background:#3a3a3a;}
.sf-ch-viewseg button.sf-ch-viewon{background:color-mix(in srgb, var(--sf-acc, #f66744) 25%, transparent);color:#fff;}
.sf-ch-prompt{flex:0 0 auto;width:100%;height:${PROMPT_H}px;min-height:${PROMPT_H}px;max-height:${PROMPT_H}px;resize:none;font:12px/1.5 sans-serif;color:#ddd;background:#1d1d1d;border:1px solid #333;border-radius:5px;padding:6px 8px;box-sizing:border-box;outline:none;overflow-y:auto;}
.sf-ch-shotsbar{flex:0 0 auto;display:flex;gap:6px;align-items:center;height:${SHOTS_H}px;min-height:${SHOTS_H}px;max-height:${SHOTS_H}px;overflow-x:auto;overflow-y:hidden;background:#1a1a1a;border:1px solid #333;border-radius:5px;padding:4px 6px;box-sizing:border-box;}
.sf-ch-shotpick{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:2px;cursor:pointer;padding:2px;border-radius:4px;border:1px solid transparent;}
.sf-ch-shotpick img{width:40px;height:40px;object-fit:cover;border-radius:4px;background:#111;display:block;}
.sf-ch-shotpick i{font:9px sans-serif;font-style:normal;color:#888;max-width:52px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-ch-shotpick:hover{border-color:#555;}
.sf-ch-shotpick.sf-ch-picked{border-color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);}
.sf-ch-shotpick.sf-ch-picked i{color:#fff;}
.sf-ch-shotsempty{font:11px sans-serif;color:#666;white-space:nowrap;}
.sf-ch-list{flex:0 0 auto;min-height:150px;height:calc(100% - 12px);overflow-y:auto;overflow-x:hidden;display:flex;flex-direction:column;gap:6px;padding:2px;box-sizing:border-box;}
.sf-ch-list.sf-ch-grid{display:grid !important;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));grid-auto-rows:max-content;gap:8px;align-content:start;padding:4px 0;}
.sf-ch-card{display:flex;flex-direction:column;gap:3px;padding:4px;border-radius:6px;cursor:pointer;background:#222;border:1px solid #3a3a3a;overflow:hidden;flex:0 0 auto;}
.sf-ch-card:hover{border-color:#555;}
.sf-ch-cardsel{border-color:var(--sf-acc, #f66744);background:color-mix(in srgb, var(--sf-acc, #f66744) 12%, transparent);}
.sf-ch-shots{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;}
.sf-ch-shot{display:flex;flex-direction:column;gap:2px;min-width:0;}
.sf-ch-shot img{width:100%;height:auto;aspect-ratio:1/1;object-fit:cover;border-radius:4px;background:#1a1a1a;flex:0 0 auto;display:block;}
.sf-ch-shot i{font:9px sans-serif;font-style:normal;color:#888;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-ch-card span.sf-ch-name{font:11px sans-serif;color:#ccc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center;user-select:none;}
.sf-ch-cardsel span.sf-ch-name{color:#fff;}
.sf-ch-more{display:flex;align-items:center;justify-content:center;aspect-ratio:1/1;background:#1a1a1a;border-radius:4px;font:11px sans-serif;color:#888;}
.sf-ch-tag{display:flex;align-items:center;gap:6px;padding:3px 6px;border-radius:4px;cursor:pointer;font:12px sans-serif;color:#ccc;user-select:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:0 0 auto;}
.sf-ch-tag input{flex:0 0 auto;accent-color:var(--sf-acc, #f66744);pointer-events:none;}
.sf-ch-tag span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sf-ch-tag:hover{background:#2c2c2c;}
.sf-ch-sel{background:color-mix(in srgb, var(--sf-acc, #f66744) 16%, transparent);color:#fff;}
.sf-ch-hide{display:none;}
.sf-ch-pop{position:absolute;display:none;pointer-events:none;width:330px;border-radius:8px;border:1px solid #4a4a4a;background:#202020;box-shadow:0 8px 22px rgba(0,0,0,.65);z-index:10;padding:6px;box-sizing:border-box;}
.sf-ch-popshots{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;}
.sf-ch-popshots img{width:100%;height:130px;object-fit:cover;background:#131313;border-radius:6px;display:block;}
.sf-ch-popname{display:block;font:12px sans-serif;color:#fff;margin:5px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-ch-poppos{font:10px/1.4 sans-serif;margin:2px 0;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;word-wrap:break-word;}
.sf-ch-poppos b{color:#7bd88f;}
.sf-ch-poppos span{color:#9ecfa8;}`);
}

// 角色库列表缓存（promise 级，避免加载期重复请求；失败缓存空列表会话内不重试）
const libraryCache = {};
function fetchLibrary(name) {
  if (!libraryCache[name]) {
    libraryCache[name] = (async () => {
      try {
        const resp = await fetch(sfApiUrl(`${lib.CHARACTERS_API}?name=${encodeURIComponent(name)}`));
        if (resp.ok) {
          const data = await resp.json();
          return Array.isArray(data) ? data : [];
        }
      } catch (e) {
        console.warn("sfnodes.CharacterSelect: 拉取角色库失败", name, e);
      }
      return [];
    })();
  }
  return libraryCache[name];
}

function isZh() {
  return (navigator.language || "en").toLowerCase().startsWith("zh");
}

function stateWidget(node) {
  return (node.widgets || []).find((w) => w.name === lib.STATE_WIDGET) || null;
}

function promptWidget(node) {
  return (node.widgets || []).find((w) => w.name === lib.PROMPT_WIDGET) || null;
}

function readRawState(node) {
  const w = stateWidget(node);
  return w ? w.value : "";
}

function writeState(node, role, shots) {
  const w = stateWidget(node);
  if (w) w.value = lib.serializeSelection(role, shots);
}

function readDraft(node) {
  const w = promptWidget(node);
  return w && w.value != null ? String(w.value) : "";
}

function writeDraft(node, text) {
  const w = promptWidget(node);
  if (w) w.value = String(text != null ? text : "");
}

function currentLibraryName(node) {
  const w = (node.widgets || []).find((x) => x.name === "library");
  return w && w.value ? String(w.value) : "";
}

// Grid/List 显示模式：存 node.properties 随 workflow 保存
function viewMode(node) {
  const v = node.properties && node.properties[VIEW_PROP];
  return v === "list" ? "list" : "grid";
}

function setViewMode(node, mode) {
  if (!node.properties) node.properties = {};
  node.properties[VIEW_PROP] = mode;
}

// 角色切换：点已选角色=取消整个选择；点新角色=选中其首图并清空草稿
// （默认首图语义，手动勾选更多走 shots 横条）。
// 改选分镜：复选框切换单图。
// 加载期门控点击防覆盖刚恢复的选择。
function toggleRole(node, ctx, name) {
  if (isGraphLoading()) return;
  const sel = lib.coerceSelection(ctx.roles || [], readRawState(node));
  if (sel.role === name) {
    writeState(node, "", []);
  } else {
    const entry = lib.entryOf(ctx.roles || [], name);
    writeState(node, name, lib.entryImages(entry).slice(0, 1).map((i) => i.label));
    writeDraft(node, "");
  }
  renderAll(ctx);
}

function toggleShot(node, ctx, label) {
  if (isGraphLoading()) return;
  const sel = lib.coerceSelection(ctx.roles || [], readRawState(node));
  if (!sel.role) return;
  const shots = sel.shots.slice();
  const i = shots.indexOf(label);
  if (i >= 0) shots.splice(i, 1);
  else {
    // 按库内顺序插入（batch 顺序确定性，不随点击顺序漂移）
    const order = lib.entryImages(lib.entryOf(ctx.roles || [], sel.role)).map((x) => x.label);
    shots.push(label);
    shots.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  }
  writeState(node, sel.role, shots);
  renderAll(ctx);
}

function thumbSrc(url) {
  return url ? (lib.isRemoteThumb(url) ? url : sfApiUrl(url)) : EMPTY_IMG;
}

// hover 信息浮窗（分镜大图 + 名称 + 提示词）
const POP_W = 334;
const POP_H = 240;

function placePop(pop, root, e) {
  const r = root.getBoundingClientRect();
  const scale = window.LiteGraph?.ds?.scale || 1;
  let x = (e.clientX - r.left + 14) / scale;
  let y = (e.clientY - r.top - 8) / scale;
  if (x + POP_W > r.width / scale - 4) x = r.width / scale - POP_W - 4;
  if (y + POP_H > r.height / scale - 4) y = r.height / scale - POP_H - 4;
  x = Math.max(4, x);
  y = Math.max(4, y);
  pop.style.left = `${x}px`;
  pop.style.top = `${y}px`;
}

function showPop(ctx, item, e) {
  const { popEl, root } = ctx;
  const shots = popEl.querySelector(".sf-ch-popshots");
  shots.innerHTML = "";
  const imgs = lib.entryImages(item.raw || {}).slice(0, POP_PREVIEW_MAX);
  for (const shot of imgs) {
    const img = document.createElement("img");
    img.src = thumbSrc(shot.url);
    img.onerror = () => {
      img.src = EMPTY_IMG;
    };
    shots.append(img);
  }
  if (!imgs.length) {
    const img = document.createElement("img");
    img.src = EMPTY_IMG;
    shots.append(img);
  }
  popEl.querySelector(".sf-ch-popname").textContent = item.label;
  const pos = popEl.querySelector(".sf-ch-poppos");
  const prompt = item.raw && item.raw.prompt ? String(item.raw.prompt) : "";
  pos.style.display = prompt ? "" : "none";
  if (prompt) pos.querySelector("span").textContent = prompt;
  placePop(popEl, root, e);
  popEl.style.display = "block";
}

function hidePop(ctx) {
  ctx.popEl.style.display = "none";
}

// 提示词编辑框显示值：草稿非空整体覆盖，否则选中分镜拼接（后端同语义镜像）
function renderPrompt(ctx) {
  const { node, promptEl, roles } = ctx;
  if (!promptEl) return;
  const text = lib.displayPrompt(roles || [], readRawState(node), readDraft(node));
  if (promptEl.value !== text) promptEl.value = text;
}

// shots 复选横条：当前角色的全部图片（库内顺序），勾选态 = 已选分镜
function renderShotsBar(ctx) {
  const { node, shotsBar, roles } = ctx;
  if (!shotsBar) return;
  shotsBar.innerHTML = "";
  const sel = lib.coerceSelection(roles || [], readRawState(node));
  const entry = lib.entryOf(roles || [], sel.role);
  const imgs = lib.entryImages(entry);
  if (!sel.role || !imgs.length) {
    const hint = document.createElement("span");
    hint.className = "sf-ch-shotsempty";
    hint.textContent = isZh() ? "未选择角色" : "No character selected";
    shotsBar.append(hint);
    return;
  }
  const picked = {};
  for (const s of sel.shots) picked[s] = true;
  for (const shot of imgs) {
    const cell = document.createElement("div");
    cell.className = "sf-ch-shotpick" + (picked[shot.label] ? " sf-ch-picked" : "");
    cell.title = shot.label + (shot.prompt ? `\n${shot.prompt}` : "");
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = thumbSrc(shot.url);
    img.onerror = () => {
      img.src = EMPTY_IMG;
    };
    const tag = document.createElement("i");
    tag.textContent = shot.label;
    cell.append(img, tag);
    cell.onclick = () => toggleShot(node, ctx, shot.label);
    shotsBar.append(cell);
  }
}

function renderAll(ctx) {
  renderShotsBar(ctx);
  renderList(ctx);
  renderPrompt(ctx);
}

function shotCell(url, label) {
  const cell = document.createElement("div");
  cell.className = "sf-ch-shot";
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = thumbSrc(url);
  img.onerror = () => {
    img.src = EMPTY_IMG;
  };
  const tag = document.createElement("i");
  tag.textContent = label;
  cell.append(img, tag);
  return cell;
}

function makeTag(item, ctx) {
  const { node } = ctx;
  const label = document.createElement("label");
  label.className = "sf-ch-tag" + (item.selected ? " sf-ch-sel" : "") + (item.hidden ? " sf-ch-hide" : "");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = item.selected;
  const span = document.createElement("span");
  span.textContent = item.label;
  label.append(cb, span);

  label.onclick = (e) => {
    // 阻止 label 默认激活 checkbox：默认行为会合成 input.click() 并冒泡回
    // label，造成 onclick 二次触发（选中又取消、表现"点不动"）
    e.preventDefault();
    toggleRole(node, ctx, item.name);
  };

  label.onmouseenter = (e) => showPop(ctx, item, e);
  label.onmousemove = (e) => {
    if (ctx.popEl.style.display !== "none") placePop(ctx.popEl, ctx.root, e);
  };
  label.onmouseleave = () => hidePop(ctx);
  return label;
}

// Grid 视图卡片：前 N 张缩略图 + 名字（loading="lazy" 避免大库一次性拉取）
function makeCard(item, ctx) {
  const { node } = ctx;
  const card = document.createElement("div");
  card.className = "sf-ch-card" + (item.selected ? " sf-ch-cardsel" : "") + (item.hidden ? " sf-ch-hide" : "");
  const shots = document.createElement("div");
  shots.className = "sf-ch-shots";
  const imgs = lib.entryImages(item.raw || {});
  const shown = imgs.length > CARD_PREVIEW_MAX ? imgs.slice(0, CARD_PREVIEW_MAX - 1) : imgs;
  for (const shot of shown) shots.append(shotCell(shot.url, shot.label));
  if (imgs.length === 0) shots.append(shotCell("", ""));
  if (imgs.length > CARD_PREVIEW_MAX) {
    const more = document.createElement("div");
    more.className = "sf-ch-more";
    more.textContent = `+${imgs.length - (CARD_PREVIEW_MAX - 1)}`;
    shots.append(more);
  }
  const span = document.createElement("span");
  span.className = "sf-ch-name";
  span.textContent = `${item.label} (${imgs.length})`;
  span.title = item.label;
  card.append(shots, span);
  card.onclick = () => {
    toggleRole(node, ctx, item.name);
  };
  card.onmouseenter = (e) => showPop(ctx, item, e);
  card.onmousemove = (e) => {
    if (ctx.popEl.style.display !== "none") placePop(ctx.popEl, ctx.root, e);
  };
  card.onmouseleave = () => hidePop(ctx);
  return card;
}

function renderList(ctx) {
  const { node, listEl, searchEl, roles } = ctx;
  if (!listEl) return;
  listEl.className = "sf-ch-list" + (viewMode(node) === "grid" ? " sf-ch-grid" : "");
  const sel = lib.coerceSelection(roles || [], readRawState(node));
  const items = lib.filterAndSort(roles || [], searchEl.value, sel.role, isZh());
  listEl.innerHTML = "";
  const make = viewMode(node) === "grid" ? makeCard : makeTag;
  for (const item of items) {
    listEl.append(make(item, ctx));
  }
}

// 库就绪后收敛选择：恢复值有效则保留，空/失效回落首角首图并清空草稿。
// configure 恢复发生在 widget 赋值之后，读到的是恢复后的 live 值，
// 故恢复的有效选择不会被覆盖。
function ensureSelection(ctx) {
  const raw = readRawState(ctx.node);
  const next = lib.coerceSelection(ctx.roles || [], raw);
  const cur = lib.parseState(raw);
  const curRole = cur.role || "";
  const curShots = cur._legacy ? null : cur.shots.slice().sort().join("\n");
  const nextShots = next.shots.slice().sort().join("\n");
  if (curRole !== next.role || curShots !== nextShots) {
    // 旧数组态恒迁移（shots: null → 首图），显式空在重载时回落首图（§47 同款语义）
    writeState(ctx.node, next.role, next.shots);
    if (curRole !== next.role) writeDraft(ctx.node, "");
  }
}

function ensureLoaded(ctx) {
  const name = currentLibraryName(ctx.node);
  if (ctx.name === name && ctx.roles) {
    ensureSelection(ctx);
    renderAll(ctx);
    return;
  }
  if (ctx.pending) {
    ctx.pending.then(() => {
      if (currentLibraryName(ctx.node) !== ctx.name) ensureLoaded(ctx); // 加载期间库被切换：重新加载
      else {
        ensureSelection(ctx);
        renderAll(ctx);
      }
    });
    return;
  }
  ctx.pending = fetchLibrary(name).then((data) => {
    ctx.pending = null;
    if (currentLibraryName(ctx.node) !== name) return; // 竞态：期间已切换库
    ctx.name = name;
    ctx.roles = data;
    ensureSelection(ctx);
    renderAll(ctx);
  });
}

function setupNode(node) {
  injectCSS();

  // ── 隐藏真源 widget（Python hidden 声明，自动存在；缺则补建）──
  let sw = stateWidget(node);
  if (!sw) {
    sw = node.addWidget("STRING", lib.STATE_WIDGET, '{"role": "", "shots": []}', () => {});
    sw.hidden = true;
    sw.computeSize = () => [0, -4];
    if (!sw.options) sw.options = {};
    sw.options.canvasOnly = true;
  } else {
    hideJsonWidget(node.widgets, lib.STATE_WIDGET);
  }
  let pw = promptWidget(node);
  if (!pw) {
    pw = node.addWidget("STRING", lib.PROMPT_WIDGET, "", () => {});
    pw.hidden = true;
    pw.computeSize = () => [0, -4];
    if (!pw.options) pw.options = {};
    pw.options.canvasOnly = true;
  } else {
    hideJsonWidget(node.widgets, lib.PROMPT_WIDGET);
  }

  const root = document.createElement("div");
  root.className = "sf-ch-root";

  const tools = document.createElement("div");
  tools.className = "sf-ch-tools";
  const resetBtn = document.createElement("button");
  resetBtn.className = "sf-ch-reset";
  resetBtn.textContent = isZh() ? "重置" : "Reset";
  resetBtn.title = isZh() ? "清空已选角色" : "Clear selected character";
  const searchEl = document.createElement("textarea");
  searchEl.className = "sf-ch-search";
  searchEl.rows = 1;
  searchEl.placeholder = isZh() ? "🔎 搜索角色 ..." : "🔎 Search characters ...";
  installWheelZoomPassthrough(searchEl);

  // Grid/List 显示模式切换
  const zh = isZh();
  const viewSeg = document.createElement("div");
  viewSeg.className = "sf-ch-viewseg";
  const syncViewBtns = () => {
    const cur = viewMode(node);
    for (const btn of viewSeg.children) {
      btn.classList.toggle("sf-ch-viewon", btn.dataset.mode === cur);
    }
  };
  const viewBtn = (mode, icon, label) => {
    const b = document.createElement("button");
    b.dataset.mode = mode;
    b.textContent = icon;
    b.title = label;
    b.onclick = () => {
      setViewMode(node, mode);
      syncViewBtns();
      renderList(ctx);
    };
    viewSeg.append(b);
    return b;
  };
  viewBtn("grid", "▦", zh ? "网格视图" : "Grid view");
  viewBtn("list", "☰", zh ? "列表视图" : "List view");
  syncViewBtns();

  tools.append(resetBtn, searchEl, viewSeg);

  // 提示词编辑框：手改写入草稿 widget（空=回落分镜拼接）
  const promptEl = document.createElement("textarea");
  promptEl.className = "sf-ch-prompt";
  promptEl.placeholder = isZh()
    ? "拼接提示词（选中后自动填入，可手改整体覆盖；清空=回落分镜拼接）..."
    : "Joined prompt (auto-filled on select; edit to override; clear to fall back) ...";
  promptEl.addEventListener("keydown", (e) => e.stopPropagation());
  installWheelZoomPassthrough(promptEl);

  // shots 复选横条：当前角色的全部图片
  const shotsBar = document.createElement("div");
  shotsBar.className = "sf-ch-shotsbar";

  const listEl = document.createElement("div");
  listEl.className = "sf-ch-list";

  // hover 信息浮窗（分镜大图 + 名称 + 提示词）
  const popEl = document.createElement("div");
  popEl.className = "sf-ch-pop";
  const popShots = document.createElement("div");
  popShots.className = "sf-ch-popshots";
  const popName = document.createElement("span");
  popName.className = "sf-ch-popname";
  const popPos = document.createElement("div");
  popPos.className = "sf-ch-poppos";
  const popPosB = document.createElement("b");
  popPosB.textContent = "Prompt: ";
  const popPosT = document.createElement("span");
  popPos.append(popPosB, popPosT);
  popEl.append(popShots, popName, popPos);

  root.append(tools, promptEl, shotsBar, listEl, popEl);

  const ctx = {
    node,
    root,
    listEl,
    searchEl,
    promptEl,
    shotsBar,
    popEl,
    roles: null,
    name: null,
    pending: null,
  };

  resetBtn.onclick = () => {
    if (isGraphLoading()) return;
    writeState(node, "", []);
    writeDraft(node, "");
    renderAll(ctx);
  };
  searchEl.oninput = () => renderList(ctx);
  promptEl.oninput = () => {
    writeDraft(node, promptEl.value);
  };

  // 列表高度显式管理：不用 CSS 百分比（父容器高度不确定时 calc 失效）
  const fitListHeight = () => {
    if (!root.clientHeight) return;
    const toolsH = tools.offsetHeight || TOOLS_H - 6;
    const h = Math.max(150, root.clientHeight - toolsH - PROMPT_H - SHOTS_H - GAP * 3 - ROOT_PAD);
    listEl.style.height = h + "px";
  };
  const listRo = new ResizeObserver(fitListHeight);
  listRo.observe(root);

  // 内容高度测量：工具条实测 + 提示词框 + shots 横条 + 列表固定高 + padding/gap
  const measureContentHeight = () => {
    const toolsH = tools.offsetHeight || TOOLS_H - 6;
    return ROOT_PAD + toolsH + GAP + PROMPT_H + GAP + SHOTS_H + GAP + LIST_H;
  };

  const widget = node.addDOMWidget(lib.DOM_WIDGET, lib.DOM_WIDGET, root, {
    serialize: false,
    getValue: () => null,
    setValue: () => {},
    getMinHeight: measureContentHeight,
    getMaxHeight: () => Math.max(measureContentHeight(), (node.size ? node.size[1] : WIDGET_H + 60) - 75),
    margin: 4,
  });
  applyAdaptiveCanvasOnly(widget);

  if (typeof node.setSize === "function") node.setSize([440, WIDGET_H + 60]);
  else { node.size[0] = 440; node.size[1] = WIDGET_H + 60; }

  // library 库切换 → 清空选择+草稿，重新拉取并渲染
  const libW = (node.widgets || []).find((x) => x.name === "library");
  if (libW) {
    const origCb = libW.callback;
    libW.callback = function () {
      const r = origCb?.apply(this, arguments);
      writeState(node, "", []);
      writeDraft(node, "");
      ctx.roles = null; // 强制重拉
      ctx.name = null;
      renderAll(ctx);
      ensureLoaded(ctx);
      return r;
    };
  }

  node._sfCharacterCtx = ctx;
  ensureLoaded(ctx);
  renderAll(ctx);
}

app.registerExtension({
  name: "sfnodes.CharacterSelect",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origCreated?.apply(this, arguments);
      setupNode(this);
      return r;
    };

    // 工作流加载：widget 值恢复发生在 configure，恢复后重渲染选中态
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origConfigure?.apply(this, arguments);
      const ctx = this._sfCharacterCtx;
      if (ctx) ensureLoaded(ctx);
      return r;
    };

    // 自愈最小尺寸。只抬升过小的尺寸，已保存（>= min）的尺寸永不变更 -> 不脏加载
    const origResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      // LEGACY ONLY：Nodes 2.0 的渲染尺寸在 Vue 布局 store 里而非 node.size
      if (!isVueNodes()) {
        if (size[0] < MIN_W) size[0] = MIN_W;
        if (size[1] < WIDGET_H) size[1] = WIDGET_H;
      }
      return origResize?.apply(this, arguments);
    };
  },
});
