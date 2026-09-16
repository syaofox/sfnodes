// ==========================================================================
// sf_node_color.js — 任意节点颜色（📦 SF Menu ▶ SF Node Color…）
//
// 入口收敛到画布聚合菜单（web/sf_canvas_menu.js），本文件只 export 构建器
// buildNodeColorMenuItem（sf_canvas_align.js / sf_memory_menu.js 同款）。
// 选中 ≥1 节点时注入；点击弹出取色面板（原生 <input type="color"> + hex
// 输入 + 最近色），作用于全部选中节点的 node.color / node.bgcolor。
//
// 机制见 experience/platform.md §2.17：node.color/bgcolor 在序列化白名单内，
// 赋任意 hex 即随工作流保存。复用：
//   - sf_canvas_align_lib.js::getSelectedNodes（Object/Array/Map/Set 兼容）
//   - sf_popup.js::attachPopupDismiss（Esc/外部点击关闭）
//   - sf_common.js::el/injectCSSOnce/sfToast + --sf-* 主题令牌
// 纯逻辑收敛于 sf_node_color_lib.js（可 .mjs 直测）。
// ==========================================================================

import { app } from "/scripts/app.js";
import { getSelectedNodes } from "./sf_canvas_align_lib.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";
import { el, injectCSSOnce, sfToast } from "./sf_common.js";
import {
    normalizeHexColor,
    pushRecent,
    applyNodeColor,
    resetNodeColor,
    readNodeColor,
    RECENT_LIMIT,
} from "./sf_node_color_lib.js";

const TAG = "SF Node Color";
const CSS_ID = "sf-node-color-css";
const RECENT_KEY = "sfnodes.node_color.recent";

function ensureCSS() {
    injectCSSOnce(
        CSS_ID,
        `
.sf-nc-pop{position:fixed;box-sizing:border-box;min-width:260px;padding:14px 16px;
  background:var(--sf-panel-bg);border:1px solid var(--sf-border-soft);border-radius:8px;
  box-shadow:0 4px 24px rgba(0,0,0,.6);color:var(--sf-text);font-family:inherit;font-size:13px}
.sf-nc-title{font-weight:600;color:var(--sf-text-strong);margin-bottom:10px}
.sf-nc-row{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.sf-nc-color{width:40px;height:28px;padding:0;border:1px solid var(--sf-border);border-radius:4px;
  background:var(--sf-input-bg);cursor:pointer;flex:0 0 auto}
.sf-nc-hex{flex:1 1 auto;min-width:0;height:28px;box-sizing:border-box;padding:0 8px;
  background:var(--sf-input-bg);border:1px solid var(--sf-border);border-radius:4px;
  color:var(--sf-text);font-family:monospace;font-size:12px}
.sf-nc-label{color:var(--sf-text-dim);font-size:11px;margin-bottom:6px}
.sf-nc-recents{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;min-height:22px}
.sf-nc-swatch{width:22px;height:22px;border-radius:4px;border:1px solid var(--sf-border);cursor:pointer}
.sf-nc-swatch:hover{border-color:var(--sf-text-dim)}
.sf-nc-empty{color:var(--sf-text-faint);font-size:11px}
.sf-nc-actions{display:flex;gap:8px;justify-content:flex-end}
.sf-nc-btn{height:28px;padding:0 14px;border-radius:4px;border:1px solid var(--sf-border);
  background:var(--sf-surface);color:var(--sf-text);cursor:pointer;font-size:12px}
.sf-nc-btn:hover{background:var(--sf-surface-hover)}
.sf-nc-btn.primary{border-color:transparent;background:var(--sf-acc,#f66744);color:#fff}
.sf-nc-btn.primary:hover{filter:brightness(1.08)}
`
    );
}

function readRecent() {
    try {
        const raw = localStorage.getItem(RECENT_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return Array.isArray(arr) ? arr.map(normalizeHexColor).filter(Boolean) : [];
    } catch {
        return [];
    }
}

function writeRecent(list) {
    try {
        localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    } catch {
        /* 隐私模式等忽略 */
    }
}

let _activePanel = null;

export function buildNodeColorMenuItem() {
    const nodes = getSelectedNodes(app);
    if (!nodes.length) return null;
    return {
        content: "SF Node Color…",
        callback: () => openNodeColorPanel(nodes),
    };
}

function commit(mutate) {
    const g = app.graph;
    let result;
    try { g?.beforeChange?.(); } catch { /* ignore */ }
    try { result = mutate(); } finally { /* 确保 afterChange 总被调用 */ }
    try { g?.afterChange?.(); } catch { /* ignore */ }
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    return result;
}

export function openNodeColorPanel(nodes) {
    const targets = (nodes || []).filter(Boolean);
    if (!targets.length) return;
    if (_activePanel) _activePanel();

    ensureCSS();
    const overlay = el("div", "sf-nc-overlay");
    Object.assign(overlay.style, { position: "fixed", inset: "0", zIndex: "99998", background: "transparent" });
    const pop = el("div", "sf-nc-pop");
    overlay.appendChild(pop);
    document.body.appendChild(overlay);

    let closed = false;
    let recent = readRecent();
    const initial = readNodeColor(targets) || "#3f5159";

    pop.appendChild(el("div", "sf-nc-title", `SF Node Color · ${targets.length} node(s)`));
    const row = el("div", "sf-nc-row");
    const colorInput = el("input", "sf-nc-color");
    colorInput.type = "color";
    colorInput.value = initial;
    colorInput.setAttribute("aria-label", "Node color");
    const hexInput = el("input", "sf-nc-hex");
    hexInput.type = "text";
    hexInput.value = initial;
    hexInput.spellcheck = false;
    hexInput.setAttribute("aria-label", "Hex color");
    row.appendChild(colorInput);
    row.appendChild(hexInput);
    pop.appendChild(row);

    pop.appendChild(el("div", "sf-nc-label", "最近颜色"));
    const recents = el("div", "sf-nc-recents");
    pop.appendChild(recents);

    const actions = el("div", "sf-nc-actions");
    const clearBtn = el("button", "sf-nc-btn", "清除颜色");
    const applyBtn = el("button", "sf-nc-btn primary", "应用");
    actions.appendChild(clearBtn);
    actions.appendChild(applyBtn);
    pop.appendChild(actions);

    function renderRecents() {
        recents.innerHTML = "";
        if (!recent.length) {
            recents.appendChild(el("span", "sf-nc-empty", "暂无（应用后自动记录）"));
            return;
        }
        for (const c of recent) {
            const sw = el("div", "sf-nc-swatch");
            sw.style.background = c;
            sw.title = c;
            sw.addEventListener("click", () => {
                colorInput.value = c;
                hexInput.value = c;
                applyCurrent(c);
            });
            recents.appendChild(sw);
        }
    }

    function applyCurrent(hex) {
        const n = normalizeHexColor(hex);
        if (!n) {
            sfToast({ summary: TAG, detail: "无效的十六进制颜色", severity: "error", fallbackTag: TAG });
            return;
        }
        commit(() => applyNodeColor(targets, n));
        recent = pushRecent(recent, n, RECENT_LIMIT);
        writeRecent(recent);
        renderRecents();
        sfToast({ summary: TAG, detail: `已应用 ${n} 到 ${targets.length} 个节点`, severity: "success", fallbackTag: TAG });
    }

    function close() {
        if (closed) return;
        closed = true;
        detachDismiss();
        overlay.remove();
        if (_activePanel === close) _activePanel = null;
    }

    const detachDismiss = attachPopupDismiss(overlay, { onClose: close });
    _activePanel = close;

    overlay.addEventListener("mousedown", (e) => {
        if (e.target === overlay) {
            e.preventDefault();
            e.stopPropagation();
            close();
        }
    });

    colorInput.addEventListener("input", () => {
        hexInput.value = colorInput.value;
    });
    colorInput.addEventListener("change", () => applyCurrent(colorInput.value));
    hexInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const n = normalizeHexColor(hexInput.value);
            if (n) {
                colorInput.value = n;
                hexInput.value = n;
                applyCurrent(n);
            } else {
                sfToast({ summary: TAG, detail: "无效的十六进制颜色", severity: "error", fallbackTag: TAG });
            }
        }
    });
    applyBtn.addEventListener("click", () => {
        // 取色器 change 已即时应用；此按钮作用于 hex 文本框，要求合法输入。
        const n = normalizeHexColor(hexInput.value);
        if (!n) {
            sfToast({ summary: TAG, detail: "无效的十六进制颜色", severity: "error", fallbackTag: TAG });
            return;
        }
        colorInput.value = n;
        hexInput.value = n;
        applyCurrent(n);
    });
    clearBtn.addEventListener("click", () => {
        commit(() => resetNodeColor(targets));
        sfToast({ summary: TAG, detail: `已清除 ${targets.length} 个节点的颜色`, severity: "success", fallbackTag: TAG });
    });

    renderRecents();

    // 定位：视口中心偏上，再按画布缩放钳回可视区
    const scale = (() => {
        try { return app.canvas?.ds?.scale || 1; } catch { return 1; }
    })();
    pop.style.left = "0px";
    pop.style.top = "0px";
    const rect = pop.getBoundingClientRect ? pop.getBoundingClientRect() : { width: 280, height: 220 };
    pop.style.left = `${Math.max(8, (window.innerWidth - (rect.width || 280)) / 2)}px`;
    pop.style.top = `${Math.max(8, (window.innerHeight - (rect.height || 220)) / 2.5)}px`;
    clampToViewport(pop, { scale });
}
