// ==========================================================================
// sf_ignore_groups.js - SF Ignore Groups 前端（复刻孤海忽略多组）
// ==========================================================================
//
// 复刻 "nodes pass.js" 的画布编组开关面板（组行列表 + 开关拨钮、三种切换
// 模式、关键词/颜色筛选、排序、UI 缩放、主题色、设置弹窗、500ms 轮询同步
// 外部改动），差异（已确认范围）：
// - properties 键 guhai_ig_* → sf_ig_*（8 键，lib 读写）；CSS 前缀
//   guhai-ig- → sf-ig-；DOM widget 名 ig_custom → sf_ig_ui
// - 原型补丁改 nodeCreated 实例装配；import 改绝对路径 "/scripts/app.js"；
//   扩展名 sfnodes.IgnoreGroups
// - 几何/状态/切换纯逻辑收敛 lib（原版闭包内联）；滚轮转发复用
//   sf_common.installWheelZoomPassthrough（带可滚动容器检测，是原版手写
//   转发的超集）；设置弹窗关闭复用 sf_popup.attachPopupDismiss +
//   clampToViewport；静态样式经 sf_common.injectCSSOnce
// - 全局 app.graph.change 重复包装收敛为守卫单例（多节点共享 _live 集）；
//   定时器/document 监听按节点清理（onRemoved + 脱图兜底）
// - 切换/恢复后补 setDirtyCanvas（原版仅调 graph.change，面板自身状态条
//   靠重建刷新，画布节点灰化需显式 dirty）
//
// 后端权威：nodes/utils/ignore_groups.py::SFIgnoreGroups（空壳 OUTPUT_NODE）
// ==========================================================================

import { app } from "/scripts/app.js";
import { el, injectCSSOnce, installWheelZoomPassthrough } from "./sf_common.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";
import {
  MODE_ALWAYS,
  MODE_BYPASS,
  MODE_NEVER,
  SWITCH_ALWAYS_ONE,
  SWITCH_AT_MOST_ONE,
  SWITCH_DEFAULT,
  collectNodes,
  filterSortGroups,
  groupBounds,
  groupState,
  isNodeActive,
  nestedGroups,
  nodeBounds,
  normalizeColor,
  readState,
  stateSig,
  toggleTransition,
  writeState,
} from "./sf_ignore_groups_lib.js";

const CLASS = "SFIgnoreGroups";
const DOM_WIDGET = "sf_ig_ui";
const CSS_ID = "sf-ig-css";
const POLL_MS = 500;

// UI 缩放基数（原版 BASE_* 一致）
const BASE_ROW_H = 34;
const BASE_PAD_X = 16;
const BASE_ROW_PAD_L = 26;
const BASE_ROW_PAD_R = 16;
const BASE_TOGGLE_W = 67;
const BASE_TOGGLE_H = 26;
const BASE_KNOB_R = 8.8;
const BASE_FONT_SIZE = 20;
const BASE_BORDER_R = 13;
const HEADER_H = 14;
const ROW_GAP = 15;

let _uidCounter = 0;
// 存活实例集：全局 graph.change 单例包装经此集标记 dirty（原版逐节点重复包装）
const _live = new Set();

function ensureGraphPatched() {
  const g = app.graph;
  if (!g || g._sfIgPatched || typeof g.change !== "function") return;
  g._sfIgPatched = true;
  const orig = g.change;
  g.change = function (...args) {
    for (const inst of _live) {
      if (!inst.selfChanging) inst.dirty = true;
    }
    return orig.apply(this, args);
  };
}

function injectCSS() {
  injectCSSOnce(
    CSS_ID,
    `
.sf-ig{position:relative;width:100%;box-sizing:border-box;overflow:hidden;user-select:none;pointer-events:auto;margin-top:-10px;padding-bottom:10px}
.sf-ig-empty{color:#888;text-align:center;padding:20px 16px;font-size:13px}
.sf-ig-gear svg{width:13px;height:13px}
`
  );
}

function hexToRgba(hex, alpha) {
  if (!hex || hex.length < 7) return `rgba(120,120,120,${alpha})`;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const gearSVG = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="3" fill="rgba(120,120,120,0.3)" stroke="rgba(153,153,153,0.6)" stroke-width="1"/><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58z" fill="rgba(120,120,120,0.25)" stroke="rgba(120,120,120,0.4)" stroke-width="1.2"/></svg>`;

// ── 节点装配 ──
function setupIgnoreGroups(node) {
  injectCSS();
  const uid = "sf-ig-" + ++_uidCounter;
  const styleId = uid + "-dyn";

  const st = readState(node.properties);
  // 运行时字段（不进 properties）
  const rt = {
    selfChanging: false,
    dirty: true,
    prevVisibleTitles: new Set(),
    lastSig: "",
    lastStateSig: "",
    preserveActive: false,
    lastBuildSig: "",
    lastAppliedScale: -1,
    cleanups: [],
    dead: false,
  };
  node._sfIg = { st, rt };
  _live.add(rt);

  const S = (base) => Math.round(base * st.uiScale);

  function save() {
    node.properties = writeState(node.properties || {}, st);
  }

  // 快照：plain 数据（groups 保留 ref 供回写模式）
  function snapshot() {
    const g = app.graph;
    if (!g) return { groups: [], nodes: [] };
    const titleH =
      (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) || 30;
    const groups = (g._groups || []).map((gr) => ({
      title: (gr.title || "").trim() || "Unnamed",
      bounds: groupBounds(gr),
      color: normalizeColor(gr.color),
      ref: gr,
    }));
    const nodes = (g._nodes || []).map((n) => ({
      bounds: nodeBounds(n, titleH),
      mode: n.mode,
      disabled: !!(n.flags && n.flags.disabled),
      ref: n,
    }));
    return { groups, nodes };
  }

  function calcWidth() {
    const fS = S(BASE_FONT_SIZE);
    const w =
      S(BASE_PAD_X) * 2 +
      S(BASE_ROW_PAD_L) +
      S(BASE_ROW_PAD_R) +
      10 +
      S(BASE_TOGGLE_W) +
      2 +
      8 * fS;
    return Math.max(350, Math.round(w));
  }

  function calcHeight() {
    const { groups, nodes } = snapshot();
    const cnt = Math.max(
      filterSortGroups(groups, groups, nodes, st).length,
      1
    );
    return HEADER_H + cnt * (S(BASE_ROW_H) + ROW_GAP) - ROW_GAP + 10;
  }

  // 缩放相关动态样式（uiScale 变化才重写）
  function updateDynamicStyles() {
    if (Math.abs(rt.lastAppliedScale - st.uiScale) < 0.001) return;
    rt.lastAppliedScale = st.uiScale;
    const rH = S(BASE_ROW_H);
    const pX = S(BASE_PAD_X);
    const rPL = S(BASE_ROW_PAD_L);
    const rPR = S(BASE_ROW_PAD_R);
    const tW = S(BASE_TOGGLE_W);
    const tH = S(BASE_TOGGLE_H);
    const kR = S(BASE_KNOB_R);
    const kD = kR * 2;
    const fS = S(BASE_FONT_SIZE);
    const bR = S(BASE_BORDER_R);
    const knobPad = Math.max(0, Math.round((tH - kD) / 2));
    const knobOffL = knobPad + 3;
    const knobOnL = Math.max(knobOffL, tW - kD - knobPad - 3);
    const css = `
.${uid} .sf-ig-header{height:${HEADER_H}px;display:flex;justify-content:flex-end;align-items:center;padding:0 6px}
.${uid} .sf-ig-gear{width:16px;height:16px;cursor:pointer;opacity:.6;display:flex;align-items:center;justify-content:center;pointer-events:auto}
.${uid} .sf-ig-gear:hover{opacity:1}
.${uid} .sf-ig-row{display:flex;align-items:center;justify-content:space-between;height:${rH}px;margin:0 ${pX}px ${ROW_GAP}px;padding:0 ${rPR}px 0 ${rPL}px;background:#2B2F38;border:1px solid #6E7581;border-radius:${bR}px;cursor:pointer;box-sizing:border-box;pointer-events:auto}
.${uid} .sf-ig-row:hover{border-color:#8E95A1}
.${uid} .sf-ig-row:last-child{margin-bottom:5px}
.${uid} .sf-ig-label{font-weight:bold;font-size:${fS}px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:10px}
.${uid} .sf-ig-toggle{width:${tW}px;height:${tH}px;border-radius:${tH / 2}px;background:#606060;position:relative;flex-shrink:0}
.${uid} .sf-ig-knob{width:${kD}px;height:${kD}px;border-radius:50%;background:rgb(128,128,128);position:absolute;top:${knobPad}px;left:${knobOffL}px;box-shadow:0 1px 4px rgba(0,0,0,.3)}
.${uid} .sf-ig-toggle.on .sf-ig-knob{left:${knobOnL}px;background:rgb(230,230,230)}`;
    let s = document.getElementById(styleId);
    if (!s) {
      s = document.createElement("style");
      s.id = styleId;
      document.head.appendChild(s);
    }
    s.textContent = css;
  }

  const rootEl = el("div", "sf-ig " + uid);
  rootEl.style.visibility = "hidden";
  rt.cleanups.push(installWheelZoomPassthrough(rootEl));

  function buildStatefulSig(list) {
    const effectiveColor = st.nameColor || "#a89961";
    const entries = list.map((g) => ({
      title: g.title,
      state:
        st.mode === SWITCH_DEFAULT
          ? Array.isArray(st.activeSet) && st.activeSet.includes(g.title)
          : g.title === st.active,
    }));
    return (
      stateSig(entries) + "|" + effectiveColor + "|" + (st.mode || "") + "|" + st.uiScale
    );
  }

  function applyToGraph(list, nodes) {
    rt.selfChanging = true;
    try {
      for (const g of list) {
        const isOn =
          st.mode === SWITCH_DEFAULT
            ? st.activeSet.includes(g.title)
            : g.title === st.active;
        const members = collectNodes(g, list, nodes);
        for (const m of members) {
          if (isOn) {
            m.ref.mode = MODE_ALWAYS;
            if (m.ref.flags) m.ref.flags.disabled = false;
          } else {
            m.ref.mode = st.disable ? MODE_NEVER : MODE_BYPASS;
          }
        }
      }
      try {
        app.graph.change();
      } catch (_) {
        /* 忽略 */
      }
      node.setDirtyCanvas(true, true);
    } finally {
      rt.selfChanging = false;
    }
  }

  function refresh(forceApply) {
    const { groups, nodes } = snapshot();
    try {
      const list = filterSortGroups(groups, groups, nodes, st);
      const currentVisibleTitles = new Set(list.map((g) => g.title));
      const sig = list.map((g) => g.title).join("\x00");
      const listChanged = sig !== rt.lastSig;
      rt.lastSig = sig;

      let stateChanged = false;
      if (st.mode === SWITCH_DEFAULT) {
        if (!Array.isArray(st.activeSet)) {
          st.activeSet = [];
          for (const g of groups) {
            if (groupState(g, groups, nodes) !== false) st.activeSet.push(g.title);
          }
          stateChanged = true;
        } else {
          const allTitles = new Set(groups.map((g) => g.title));
          const before = st.activeSet.length;
          st.activeSet = st.activeSet.filter((t) => allTitles.has(t));
          if (st.activeSet.length !== before) stateChanged = true;
          if (listChanged) {
            for (const g of list) {
              if (!rt.prevVisibleTitles.has(g.title)) {
                const gs = groupState(g, groups, nodes);
                const idx = st.activeSet.indexOf(g.title);
                if (gs === true && idx < 0) {
                  st.activeSet.push(g.title);
                  stateChanged = true;
                } else if (gs === false && idx >= 0) {
                  st.activeSet.splice(idx, 1);
                  stateChanged = true;
                }
              }
            }
          }
        }
      } else {
        if (!rt.preserveActive) {
          const hasFilter = !!(st.filter.trim() || (st.colorFilter && st.colorFilter !== "none"));
          if (st.active && !groups.some((g) => g.title === st.active)) {
            st.active = st.mode === SWITCH_ALWAYS_ONE && list.length ? list[0].title : null;
            stateChanged = true;
          }
          if (st.active && hasFilter) {
            const filteredTitles = new Set(list.map((g) => g.title));
            if (!filteredTitles.has(st.active)) {
              if (st.mode === SWITCH_ALWAYS_ONE && list.length) st.active = list[0].title;
              else st.active = null;
              stateChanged = true;
            }
          }
          if (st.mode === SWITCH_ALWAYS_ONE && !st.active && list.length) {
            st.active = list[0].title;
            stateChanged = true;
          }
        }
      }
      rt.preserveActive = false;
      if (stateChanged) save();

      if (forceApply || stateChanged || listChanged) {
        applyToGraph(list, nodes);
      }
      rt.prevVisibleTitles = currentVisibleTitles;
      updateDynamicStyles();
      buildDom(forceApply || stateChanged || listChanged, list);
    } finally {
      /* 快照为同周期局部量，无需清理 */
    }
  }

  function buildDom(force, preList) {
    let list = preList;
    if (!list) {
      const { groups, nodes } = snapshot();
      list = filterSortGroups(groups, groups, nodes, st);
    }
    const sig = buildStatefulSig(list);
    if (sig === rt.lastBuildSig && !force) return;
    rt.lastBuildSig = sig;

    const effectiveColor = st.nameColor || "#a89961";
    rootEl.innerHTML = "";

    const header = el("div", "sf-ig-header");
    const gear = el("div", "sf-ig-gear");
    gear.innerHTML = gearSVG;
    gear.title = "设置";
    gear.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      showSettings(e.clientX, e.clientY);
    });
    gear.addEventListener("pointerdown", (e) => e.stopPropagation());
    rt.cleanups.push(installWheelZoomPassthrough(gear));
    header.appendChild(gear);
    rootEl.appendChild(header);

    if (!list.length) {
      const empty = el(
        "div",
        "sf-ig-empty",
        st.filter.trim() ? "无匹配的组" : "工作流中无编组或空的编组"
      );
      rootEl.appendChild(empty);
    } else {
      const h = calcHeight();
      rootEl.style.minHeight = h + "px";
      rootEl.style.height = h + "px";
      for (const g of list) {
        const isOn =
          st.mode === SWITCH_DEFAULT
            ? st.activeSet.includes(g.title)
            : g.title === st.active;
        const row = el("div", "sf-ig-row");
        const label = el("div", "sf-ig-label", g.title);
        label.style.color = effectiveColor;
        label.style.opacity = isOn ? "1" : "0.5";
        const toggle = el("div", "sf-ig-toggle" + (isOn ? " on" : ""));
        toggle.style.background = isOn ? effectiveColor : "#606060";
        toggle.style.boxShadow = isOn
          ? "0 0 8px " + hexToRgba(effectiveColor, 0.35)
          : "none";
        toggle.appendChild(el("div", "sf-ig-knob"));
        row.appendChild(label);
        row.appendChild(toggle);
        row.addEventListener("mousedown", (e) => {
          e.preventDefault();
          e.stopPropagation();
          handleToggle(g.title);
        });
        row.addEventListener("pointerdown", (e) => e.stopPropagation());
        rt.cleanups.push(installWheelZoomPassthrough(row));
        rootEl.appendChild(row);
      }
    }
  }

  function handleToggle(title) {
    const { groups } = snapshot();
    const grp = groups.find((g) => g.title === title);
    const nested = grp ? nestedGroups(grp, groups).map((g) => g.title) : [];
    const next = toggleTransition(st.mode, st.active, st.activeSet, title, nested);
    st.active = next.active;
    st.activeSet = next.activeSet;
    save();
    refresh(true);
  }

  // 外部改动同步（轮询命中时）：default 补 activeSet 增减，单选模式修 active 指向
  function syncExternalState() {
    if (!app.graph || !app.graph._nodes || app.graph._nodes.indexOf(node) < 0) return false;
    const { groups, nodes } = snapshot();
    let changed = false;
    if (st.mode === SWITCH_DEFAULT) {
      if (!Array.isArray(st.activeSet)) return false;
      for (const g of groups) {
        const gs = groupState(g, groups, nodes);
        const isActive = st.activeSet.includes(g.title);
        if (gs === true && !isActive) {
          st.activeSet.push(g.title);
          changed = true;
        } else if (gs === false && isActive) {
          st.activeSet = st.activeSet.filter((t) => t !== g.title);
          changed = true;
        }
      }
    } else {
      const list = filterSortGroups(groups, groups, nodes, st);
      if (st.mode === SWITCH_ALWAYS_ONE) {
        if (st.active) {
          const grp = list.find((g) => g.title === st.active);
          if (grp && groupState(grp, groups, nodes) === false) {
            const nextOn = list.find((g) => groupState(g, groups, nodes) === true);
            st.active = nextOn ? nextOn.title : list.length ? list[0].title : null;
            changed = true;
          } else if (!grp) {
            if (list.length) {
              const firstOn = list.find((g) => groupState(g, groups, nodes) === true);
              st.active = firstOn ? firstOn.title : list[0].title;
            } else {
              st.active = null;
            }
            changed = true;
          }
        }
        if (!st.active && list.length) {
          const firstOn = list.find((g) => groupState(g, groups, nodes) === true);
          if (firstOn) {
            st.active = firstOn.title;
            changed = true;
          }
        }
      } else if (st.mode === SWITCH_AT_MOST_ONE) {
        if (st.active) {
          const grp = list.find((g) => g.title === st.active);
          if (!grp || groupState(grp, groups, nodes) === false) {
            st.active = null;
            changed = true;
          }
        }
        if (!st.active) {
          for (const g of list) {
            if (groupState(g, groups, nodes) === true) {
              st.active = g.title;
              changed = true;
              break;
            }
          }
        }
      }
    }
    if (changed) save();
    return changed;
  }

  // ── 设置弹窗（原版 showSettings 全量移植，关闭走 sf_popup）──
  let settingsCleanup = null;
  function showSettings(x, y) {
    if (settingsCleanup) {
      settingsCleanup();
      settingsCleanup = null;
    }
    const initialScale = st.uiScale;
    let needResize = false;

    const overlay = el("div", "sf-ig-overlay");
    Object.assign(overlay.style, {
      position: "fixed",
      inset: "0",
      zIndex: "99998",
      background: "transparent",
      cursor: "default",
    });
    document.body.appendChild(overlay);

    const pop = el("div", "sf-ig-pop");
    Object.assign(pop.style, {
      position: "fixed",
      left: Math.min(x, (globalThis.innerWidth || 1280) - 280) + "px",
      top: Math.min(y, (globalThis.innerHeight || 800) - 620) + "px",
      background: "#2a2a2a",
      border: "1px solid #555",
      borderRadius: "8px",
      padding: "14px 18px",
      zIndex: "99999",
      minWidth: "250px",
      boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
      color: "#e0e0e0",
      fontFamily: "inherit",
    });

    let closed = false;
    function closePopup() {
      if (closed) return;
      closed = true;
      detachDismiss();
      document.removeEventListener("mousedown", closeColorPanel);
      overlay.remove();
      pop.remove();
      settingsCleanup = null;
      if (needResize) {
        needResize = false;
        node.size = [calcWidth(), calcHeight()];
        try {
          app.graph.change();
        } catch (_) {
          /* 忽略 */
        }
      }
    }
    // overlay 与 pop 同级挂载（原版结构）：helper 只认 overlay.contains，
    // pop 内点击必须经 exempt 豁免，否则设置里任意点击都会误判外部点击关窗
    const detachDismiss = attachPopupDismiss(overlay, {
      onClose: closePopup,
      exempt: (e) => pop.contains(e.target),
    });
    settingsCleanup = closePopup;
    overlay.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      closePopup();
    });
    // overlay 遮罩区滚轮透传画布缩放（原版同款；pop 自带透传，两者不重叠不双发）
    installWheelZoomPassthrough(overlay);
    rt.cleanups.push(installWheelZoomPassthrough(pop));

    function applyAll() {
      st.filter = fInput.value;
      st.colorFilter = colorSel;
      st.nameColor = cInput.value || null;
      st.disable = dRadioDisable.checked;
      st.sortOrder = sSelect.value;
      const newMode = mSelect.value;
      const newScale = parseFloat(scaleInput.value) || 1.0;
      st.uiScale = newScale;
      updateDynamicStyles();
      if (Math.abs(newScale - initialScale) > 0.001) needResize = true;
      if (newMode !== st.mode) {
        if (newMode === SWITCH_DEFAULT) {
          st.activeSet = filterSortGroups(
            snapshot().groups,
            snapshot().groups,
            snapshot().nodes,
            st
          ).map((g) => g.title);
        } else if (newMode === SWITCH_ALWAYS_ONE) {
          const snap = snapshot();
          const fl = filterSortGroups(snap.groups, snap.groups, snap.nodes, st);
          const ft = new Set(fl.map((g) => g.title));
          if (st.activeSet && st.activeSet.length) {
            const match = st.activeSet.find((t) => ft.has(t));
            st.active = match || (fl.length ? fl[0].title : null);
          } else {
            st.active = fl.length ? fl[0].title : null;
          }
          st.activeSet = null;
        } else {
          const snap = snapshot();
          const fl = filterSortGroups(snap.groups, snap.groups, snap.nodes, st);
          const ft = new Set(fl.map((g) => g.title));
          if (st.activeSet && st.activeSet.length) {
            const match = st.activeSet.find((t) => ft.has(t));
            st.active = match || null;
          } else if (!(st.mode === SWITCH_ALWAYS_ONE && st.active && ft.has(st.active))) {
            st.active = null;
          }
          st.activeSet = null;
        }
        st.mode = newMode;
      } else if (st.mode === SWITCH_ALWAYS_ONE && !st.active) {
        const snap = snapshot();
        const list = filterSortGroups(snap.groups, snap.groups, snap.nodes, st);
        if (list.length) st.active = list[0].title;
      }
      save();
      refresh(true);
    }

    const addLabel = (text) => {
      const d = el("div", null, text);
      Object.assign(d.style, { fontSize: "13px", fontWeight: "bold", marginBottom: "4px" });
      pop.appendChild(d);
      return d;
    };
    const styleInput = (inp) => {
      Object.assign(inp.style, {
        width: "100%",
        padding: "5px 8px",
        fontSize: "13px",
        background: "#1a1a1a",
        border: "1px solid #555",
        borderRadius: "4px",
        color: "#e0e0e0",
        outline: "none",
        boxSizing: "border-box",
        marginBottom: "14px",
      });
      return inp;
    };

    const titleEl = el("div", null, "SF 忽略多组 设置");
    Object.assign(titleEl.style, {
      fontSize: "15px",
      fontWeight: "bold",
      marginBottom: "14px",
      color: "#e0e0e0",
      borderBottom: "1px solid #444",
      paddingBottom: "8px",
    });
    pop.appendChild(titleEl);

    // 路由控制：旁路 / 禁用
    addLabel("路由控制");
    const dRow = el("div");
    Object.assign(dRow.style, {
      display: "flex",
      alignItems: "center",
      gap: "20px",
      marginBottom: "14px",
    });
    const mkRadio = (checked, text) => {
      const lbl = el("label");
      lbl.style.cursor = "pointer";
      lbl.style.fontSize = "13px";
      const r = el("input");
      r.type = "radio";
      r.name = "sf-ig-close-mode";
      r.checked = checked;
      r.style.cursor = "pointer";
      lbl.appendChild(r);
      lbl.appendChild(document.createTextNode(text));
      return [lbl, r];
    };
    const [dLblBypass, dRadioBypass] = mkRadio(!st.disable, " 绕过（ctrl+b）");
    const [dLblDisable, dRadioDisable] = mkRadio(!!st.disable, " 禁用（ctrl+m）");
    dRow.appendChild(dLblBypass);
    dRow.appendChild(dLblDisable);
    pop.appendChild(dRow);
    dRadioBypass.addEventListener("change", applyAll);
    dRadioDisable.addEventListener("change", applyAll);

    // 关键词筛选
    addLabel("关键词筛选");
    const fInput = el("input");
    fInput.type = "text";
    fInput.value = st.filter;
    fInput.placeholder = "留空 = 显示所有组";
    styleInput(fInput);
    pop.appendChild(fInput);
    let filterTimer = null;
    fInput.addEventListener("input", () => {
      if (filterTimer) clearTimeout(filterTimer);
      filterTimer = setTimeout(applyAll, 200);
    });

    // 颜色筛选
    addLabel("颜色筛选");
    let colorSel = st.colorFilter || "none";
    const cdContainer = el("div");
    Object.assign(cdContainer.style, { position: "relative", width: "100%", marginBottom: "14px" });
    const cdTrigger = el("div");
    Object.assign(cdTrigger.style, {
      width: "100%",
      padding: "5px 8px",
      fontSize: "13px",
      background: "#1a1a1a",
      border: "1px solid #555",
      borderRadius: "4px",
      color: "#e0e0e0",
      boxSizing: "border-box",
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      gap: "6px",
      userSelect: "none",
    });
    const cdRect = el("span");
    Object.assign(cdRect.style, {
      display: "inline-block",
      width: "44px",
      height: "14px",
      borderRadius: "2px",
      border: "1px solid #555",
      flexShrink: "0",
    });
    const cdText = el("span");
    const cdArrow = el("span", null, "▾");
    cdArrow.style.marginLeft = "auto";
    cdArrow.style.fontSize = "11px";
    cdArrow.style.color = "#888";
    cdTrigger.appendChild(cdRect);
    cdTrigger.appendChild(cdText);
    cdTrigger.appendChild(cdArrow);
    const updateColorPreview = () => {
      if (colorSel === "none") {
        cdRect.style.display = "none";
        cdText.textContent = "无";
      } else if (colorSel === "__transparent__") {
        cdRect.style.display = "inline-block";
        cdRect.style.background = "transparent";
        cdText.textContent = "透明色";
      } else {
        cdRect.style.display = "inline-block";
        cdRect.style.background = colorSel;
        cdText.textContent = colorSel.toUpperCase();
      }
    };
    updateColorPreview();
    cdContainer.appendChild(cdTrigger);
    let cdPanel = null;
    const buildColorOptions = () => {
      const { groups } = snapshot();
      const seen = new Map();
      for (const g of groups) {
        const c = g.color || "__transparent__";
        if (!seen.has(c)) seen.set(c, true);
      }
      if (cdPanel) {
        cdPanel.remove();
        cdPanel = null;
      }
      cdPanel = el("div");
      Object.assign(cdPanel.style, {
        position: "absolute",
        left: "0",
        right: "0",
        top: "calc(100% + 2px)",
        background: "#1a1a1a",
        border: "1px solid #555",
        borderRadius: "4px",
        zIndex: "100001",
        maxHeight: "200px",
        overflowY: "auto",
        boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
      });
      rt.cleanups.push(installWheelZoomPassthrough(cdPanel));
      const mkItem = (label, value, bg) => {
        const item = el("div");
        Object.assign(item.style, {
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "5px 8px",
          cursor: "pointer",
          fontSize: "13px",
        });
        if (bg !== null) {
          const rect = el("span");
          Object.assign(rect.style, {
            display: "inline-block",
            width: "44px",
            height: "16px",
            borderRadius: "2px",
            border: "1px solid #555",
            background: bg,
            flexShrink: "0",
          });
          item.appendChild(rect);
        }
        const txt = el("span", null, label);
        txt.style.color = "#fff";
        item.appendChild(txt);
        item.addEventListener("click", (e) => {
          e.stopPropagation();
          colorSel = value;
          updateColorPreview();
          if (cdPanel) {
            cdPanel.remove();
            cdPanel = null;
          }
          applyAll();
        });
        return item;
      };
      cdPanel.appendChild(mkItem("无", "none", null));
      for (const c of seen.keys()) {
        if (c === "__transparent__") cdPanel.appendChild(mkItem("透明色", c, "transparent"));
        else cdPanel.appendChild(mkItem(c.toUpperCase(), c, c));
      }
      cdContainer.appendChild(cdPanel);
    };
    cdTrigger.addEventListener("click", (e) => {
      e.stopPropagation();
      if (cdPanel) {
        cdPanel.remove();
        cdPanel = null;
      } else {
        buildColorOptions();
      }
    });
    pop.appendChild(cdContainer);
    const closeColorPanel = (e) => {
      if (cdPanel && !cdContainer.contains(e.target)) {
        cdPanel.remove();
        cdPanel = null;
      }
    };
    document.addEventListener("mousedown", closeColorPanel);

    // 切换模式
    addLabel("切换模式");
    const mSelect = el("select");
    for (const [val, txt] of [
      [SWITCH_DEFAULT, "默认"],
      [SWITCH_ALWAYS_ONE, "始终开启1个"],
      [SWITCH_AT_MOST_ONE, "最多开启1个"],
    ]) {
      const opt = el("option", null, txt);
      opt.value = val;
      mSelect.appendChild(opt);
    }
    mSelect.value = st.mode;
    styleInput(mSelect);
    pop.appendChild(mSelect);
    mSelect.addEventListener("change", applyAll);

    // 排序
    addLabel("排序");
    const sSelect = el("select");
    for (const [val, txt] of [
      ["position", "按位置"],
      ["alphabet", "按首字母"],
    ]) {
      const opt = el("option", null, txt);
      opt.value = val;
      sSelect.appendChild(opt);
    }
    sSelect.value = st.sortOrder;
    styleInput(sSelect);
    pop.appendChild(sSelect);
    sSelect.addEventListener("change", applyAll);

    // UI 组件大小
    addLabel("UI组件大小");
    const scaleRow = el("div");
    Object.assign(scaleRow.style, {
      display: "flex",
      alignItems: "center",
      gap: "10px",
      marginBottom: "14px",
    });
    const scaleInput = el("input");
    scaleInput.type = "range";
    scaleInput.min = "0.5";
    scaleInput.max = "5.0";
    scaleInput.step = "0.1";
    scaleInput.value = String(st.uiScale);
    Object.assign(scaleInput.style, {
      flex: "1",
      height: "4px",
      background: "#555",
      borderRadius: "2px",
      outline: "none",
      cursor: "pointer",
    });
    const scaleVal = el("span", null, st.uiScale.toFixed(1) + "x");
    Object.assign(scaleVal.style, {
      minWidth: "36px",
      fontSize: "12px",
      color: "#aaa",
      textAlign: "right",
    });
    scaleRow.appendChild(scaleInput);
    scaleRow.appendChild(scaleVal);
    pop.appendChild(scaleRow);
    scaleInput.addEventListener("input", () => {
      scaleVal.textContent = (parseFloat(scaleInput.value) || 1.0).toFixed(1) + "x";
      applyAll();
    });

    // 主题颜色
    addLabel("主题颜色");
    const colorRow = el("div");
    Object.assign(colorRow.style, {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      marginBottom: "16px",
    });
    const cInput = el("input");
    cInput.type = "color";
    cInput.value = st.nameColor || "#a89961";
    Object.assign(cInput.style, {
      width: "36px",
      height: "28px",
      padding: "0",
      border: "1px solid #555",
      borderRadius: "4px",
      background: "#1a1a1a",
      cursor: "pointer",
    });
    colorRow.appendChild(cInput);
    const cHex = el("input");
    cHex.type = "text";
    cHex.value = st.nameColor || "#a89961";
    Object.assign(cHex.style, {
      flex: "1",
      padding: "5px 8px",
      fontSize: "13px",
      background: "#1a1a1a",
      border: "1px solid #555",
      borderRadius: "4px",
      color: "#e0e0e0",
      outline: "none",
      boxSizing: "border-box",
    });
    colorRow.appendChild(cHex);
    cInput.addEventListener("input", () => {
      cHex.value = cInput.value;
      applyAll();
    });
    cHex.addEventListener("input", () => {
      if (/^#[0-9a-fA-F]{6}$/.test(cHex.value)) {
        cInput.value = cHex.value;
        applyAll();
      }
    });
    pop.appendChild(colorRow);

    document.body.appendChild(pop);
    clampToViewport(pop);
    if (typeof fInput.focus === "function") fInput.focus();
    pop.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  }

  node._sfIgShowSettings = showSettings;

  // ── DOM widget 装配 ──
  if (typeof node.addDOMWidget === "function") {
    const domWidget = node.addDOMWidget(DOM_WIDGET, DOM_WIDGET, rootEl, {
      serialize: false,
      hideOnZoom: false,
    });
    if (domWidget) {
      domWidget.computeSize = () => [calcWidth(), calcHeight()];
    }
  }
  const origComputeSize = node.computeSize;
  node.computeSize = function (...args) {
    if (node._sfIg) return [calcWidth(), calcHeight()];
    return origComputeSize ? origComputeSize.apply(this, args) : undefined;
  };

  // ── 右键菜单 ──
  const origMenu = node.getExtraMenuOptions;
  node.getExtraMenuOptions = function (canvas, options) {
    if (origMenu) {
      try {
        origMenu.apply(this, arguments);
      } catch (e) {
        console.warn("[SFIgnoreGroups] getExtraMenuOptions:", e);
      }
    }
    if (Array.isArray(options)) {
      options.splice(0, 0, null, {
        content: "SF 忽略多组 设置",
        callback: () =>
          showSettings(
            Math.round((globalThis.innerWidth || 1280) / 2 - 130),
            Math.round((globalThis.innerHeight || 800) / 2 - 200)
          ),
      });
    }
  };

  // ── 工作流恢复 ──
  const origConfigure = node.configure;
  node.configure = function (...args) {
    const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
    if (!this._sfIg) return r;
    const g = this._sfIg;
    Object.assign(g.st, readState(this.properties));
    g.rt.lastSig = "";
    g.rt.lastStateSig = "";
    g.rt.prevVisibleTitles = new Set();
    g.rt.preserveActive = true;
    g.rt.lastAppliedScale = -1;
    g.rt.dirty = true;
    refresh(true);
    return r;
  };

  // ── 清理 ──
  function cleanup() {
    if (rt.dead) return;
    rt.dead = true;
    _live.delete(rt);
    if (rt.timer) {
      clearInterval(rt.timer);
      rt.timer = null;
    }
    document.removeEventListener("keydown", onKeyDown);
    document.removeEventListener("visibilitychange", onVisibilityChange);
    for (const fn of rt.cleanups.splice(0)) {
      try {
        fn();
      } catch (_) {
        /* 忽略 */
      }
    }
    const s = document.getElementById(styleId);
    if (s) s.remove();
    if (settingsCleanup) {
      const fn = settingsCleanup;
      settingsCleanup = null;
      try {
        fn();
      } catch (_) {
        /* 忽略 */
      }
    }
  }
  const origRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    cleanup();
    if (origRemoved) origRemoved.apply(this, args);
  };

  function onKeyDown(e) {
    if (
      (e.ctrlKey || e.metaKey) &&
      (e.key === "m" || e.key === "b" || e.key === "M" || e.key === "B")
    ) {
      setTimeout(() => {
        rt.dirty = true;
      }, 100);
    }
  }
  document.addEventListener("keydown", onKeyDown);

  let pageVisible = typeof document.hidden === "boolean" ? !document.hidden : true;
  function onVisibilityChange() {
    const nowVisible = !document.hidden;
    if (nowVisible && !pageVisible) {
      pageVisible = true;
      rt.dirty = true;
    } else {
      pageVisible = nowVisible;
    }
  }
  document.addEventListener("visibilitychange", onVisibilityChange);

  // ── 轮询：dirty 才做签名计算（原版优化保留）──
  rt.timer = setInterval(() => {
    if (!node.graph) {
      cleanup();
      return;
    }
    if (!app.graph || !app.graph._nodes || app.graph._nodes.indexOf(node) < 0) return;
    if (!pageVisible) return;
    if (!rt.dirty) return;
    rt.dirty = false;
    ensureGraphPatched();
    const { groups, nodes } = snapshot();
    const list = filterSortGroups(groups, groups, nodes, st);
    const sig = list.map((g) => g.title).join("\x00");
    const listChanged = sig !== rt.lastSig;
    // 实态签名（画布节点真实旁路态，非面板期望态——外部 ctrl+b/m 改动靠此检出）
    const liveSig = stateSig(
      list.map((g) => ({ title: g.title, state: groupState(g, groups, nodes) }))
    );
    const stateChanged = liveSig !== rt.lastStateSig;
    if (listChanged || stateChanged) {
      rt.lastStateSig = liveSig;
      rt.selfChanging = true;
      try {
        const syncChanged = syncExternalState();
        updateDynamicStyles();
        if (listChanged || syncChanged) {
          refresh(true);
        } else {
          buildDom(true);
        }
        try {
          app.graph.change();
        } catch (_) {
          /* 忽略 */
        }
      } finally {
        rt.selfChanging = false;
      }
    }
  }, POLL_MS);

  // ── 初次渲染 ──
  ensureGraphPatched();
  updateDynamicStyles();
  refresh(false);
  {
    const { groups, nodes } = snapshot();
    const list = filterSortGroups(groups, groups, nodes, st);
    rt.lastStateSig = stateSig(
      list.map((g) => ({ title: g.title, state: groupState(g, groups, nodes) }))
    );
  }
  rt.dirty = false;
  node.size = [calcWidth(), calcHeight()];

  const raf = globalThis.requestAnimationFrame || ((fn) => fn());
  raf(() => {
    rootEl.style.visibility = "visible";
  });
}

app.registerExtension({
  name: "sfnodes.IgnoreGroups",

  async nodeCreated(node) {
    if (node.comfyClass !== CLASS) return;
    node.color = node.color || "#4E464A";
    node.bgcolor = node.bgcolor || "#4E464A";
    setupIgnoreGroups(node);
  },
});
