// ==========================================================================
// SF Model Info - Shared LoRA/model metadata fetch utilities + loader widget
// Used by SFLoraLoader / SFLoraLoaderModelOnly (and future loader nodes).
// 历史对话框 showLoraInfoDialog 已收敛为浮动面板 shim（见 sf_lora_stack_info.js），
// 本文件保留 metadata 网关与 info-icon widget，旧 dialog 实现已删除。
// ==========================================================================
import { app } from "/scripts/app.js";
import { loadImageAsWorkflow } from "./sf_lora_shared_info.js";
import { openInfoPanelFor } from "./sf_lora_stack_info.js";
import { getNodeRect } from "./sf_lora_stack_settings.js";
export { loadImageAsWorkflow };

// ---------------------------------------------------------------------------
// Metadata fetch (merged custom notes + embedded safetensors metadata via
// the /api/sfnodes/lora_notes gateway endpoint)
// ---------------------------------------------------------------------------

export const loraMetadataCache = new Map();
const _loraMetadataPending = new Map();
// 世代号：invalidate 后在途旧响应不得写回缓存（与 sf_lora_stack_api 的 _infoGen 同理）
const _metaGen = new Map();
const _genOf = (n) => _metaGen.get(n) || 0;

// 2026-08 统一存储：元数据读写走 /api/sfnodes/lora_notes（后端网关，与
// SFLoraStack 同一 lora_triggers.json 真源）。`force` 跳过缓存与在途去重，
// 打开对话框时用（另一节点刚保存过，缓存可能陈旧）。
export async function getLoraMetadata(name, force = false) {
    if (!name || name === "None") return null;
    if (!force && loraMetadataCache.has(name)) return loraMetadataCache.get(name);
    // Join an in-flight request instead of firing a duplicate
    if (!force && _loraMetadataPending.has(name)) return _loraMetadataPending.get(name);
    const gen = _genOf(name);
    let p;
    p = (async () => {
        try {
            // force（打开对话框）= 必新：no-store 越过浏览器启发式缓存
            // （后端响应无 Cache-Control，默认模式可能命中陈旧副本）。
            const resp = await fetch(`/api/sfnodes/lora_notes?filename=${encodeURIComponent(name)}`,
                { cache: force ? "no-store" : "default" });
            if (!resp.ok) {
                if (_genOf(name) === gen) loraMetadataCache.set(name, null);
                return null;
            }
            const meta = await resp.json();
            // 仅当世代未变才写入缓存，否则该响应已 stale（期间有 invalidate）
            if (_genOf(name) === gen) loraMetadataCache.set(name, meta);
            else if (meta && typeof meta === "object") meta._stale = true;
            // stale 响应仍返回给调用方，由调用方决定是否丢弃（对话框已关闭则忽略）
            if (_genOf(name) !== gen && meta && typeof meta === "object") meta._stale = true;
            return meta;
        } catch {
            if (_genOf(name) === gen) loraMetadataCache.set(name, null);
            return null;
        } finally {
            if (_loraMetadataPending.get(name) === p) _loraMetadataPending.delete(name);
        }
    })();

    if (!force) {
        _loraMetadataPending.set(name, p);
        try { return await p; }
        finally { if (_loraMetadataPending.get(name) === p) _loraMetadataPending.delete(name); }
    }
    return p;
}

export function invalidateLoraMetadata(name) {
    if (name) {
        loraMetadataCache.delete(name);
        _metaGen.set(name, _genOf(name) + 1);
    }
}

// ── 跨节点缓存失效：任一节点（SFLoraLoader / SFLoraStack 面板）保存
// LoRA 用户数据后广播，两端各自清自己模块的缓存，下次打开即新数据。────
if (typeof document !== "undefined") {
    document.addEventListener("sfnodes.lora-data-changed", (e) => {
        const name = e?.detail?.name;
        if (name) {
            loraMetadataCache.delete(name);
            _metaGen.set(name, _genOf(name) + 1);
        }
    });
}

// ---------------------------------------------------------------------------
// Shim: 旧 showLoraInfoDialog -> 浮动面板
// ---------------------------------------------------------------------------
// 旧实现为 1400+ 行 <dialog>，现所有 loader 统一走 SFLoraStack 同款浮动面板
// （chip 形态、近节点）。保留函数签名供外部旧调用兼容；内部转调
// openInfoPanelFor，锚点优先用事件坐标，缺省回退视口中心（面板 place() 自钳制）。
const _shimRows = new Map();
export function showLoraInfoDialog(event, name, meta) {
    if (!name || name === "None") return;
    // 事件锚点：有 clientX/Y 时贴近点击位置，否则让面板 place() 居中
    const anchorRect = (event && typeof event.clientX === "number" && typeof event.clientY === "number")
        ? () => ({
            left: event.clientX, top: event.clientY,
            right: event.clientX + 10, bottom: event.clientY + 10,
            width: 10, height: 10,
        })
        : null;
    if (!_shimRows.has(name)) _shimRows.set(name, { id: name, name, triggers: [], custom: [] });
    // 若外部传入了 meta.trigger_words，同步到 shim 行的 custom 供面板初显
    // （面板随后会经 loraInfo 拉全量，hydrateCustom 会合并，以存储为准）
    if (meta && typeof meta.trigger_words === "string" && meta.trigger_words.trim()) {
        const cur = _shimRows.get(name);
        const words = meta.trigger_words.split(/[,，\n]+/).map((s) => s.trim()).filter(Boolean);
        if (words.length && (!cur.custom || !cur.custom.length)) cur.custom = words.slice(0, 64);
    }
    const ctx = {
        key: "shim:" + name,
        node: null,
        anchorRect,
        getRow: () => _shimRows.get(name) || { id: name, name, triggers: [], custom: [] },
        patchRow: (patch) => {
            const cur = _shimRows.get(name) || { id: name, name, triggers: [], custom: [] };
            Object.assign(cur, patch);
            cur.id = name; cur.name = name;
            _shimRows.set(name, cur);
        },
        accent: (() => { try { return app.ui.settings.getSettingValue("sfnodes.Accent") || "#f66744"; } catch { return "#f66744"; } })(),
        prefs: () => ({ civitai: true, thumbs: true }),
        refresh: () => {},
    };
    // 异步打开，不阻塞调用方；失败静默
    try { openInfoPanelFor(ctx, name); } catch {}
}

// ---------------------------------------------------------------------------
// Canvas event capture (shared single wrapper)
// ---------------------------------------------------------------------------
let _lastCanvasEvent = null;
let _eventHookInstalled = false;

export function ensureEventHook() {
    if (_eventHookInstalled) return;
    _eventHookInstalled = true;
    const origAdjust = LGraphCanvas.prototype.adjustMouseEvent;
    LGraphCanvas.prototype.adjustMouseEvent = function (e) {
        origAdjust.apply(this, arguments);
        _lastCanvasEvent = e;
    };
}

export function getLastCanvasEvent() {
    return _lastCanvasEvent;
}

// ---------------------------------------------------------------------------
// Standard-combo + info-icon mounting (shared by SFLoraLoader /
// SFLoraLoaderModelOnly and future loader nodes) - now uses floating panel
// ---------------------------------------------------------------------------
const INVALID_BOUNDS = [0, -1];

function getComboWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function getComboValue(node, name) {
    const v = getComboWidget(node, name)?.value;
    return typeof v === "string" ? v : null;
}

function createInfoWidget(comboName) {
    const w = {
        name: "_info",
        type: "custom",
        options: { serialize: false },
        value: {},
        y: 0,
        last_y: 0,
        _hit: INVALID_BOUNDS,
        computeSize(width) { return [width, 24]; },
        draw(ctx, n, width, posY, height) {
            this.last_y = posY;
            this._hit = INVALID_BOUNDS;
            const loraName = getComboValue(n, comboName);
            if (!loraName || loraName === "None") return;
            const cachedMeta = loraMetadataCache.get(loraName);
            const hasCustom = cachedMeta?._has_custom;
            const size = Math.max(14, height * 0.6);
            const posX = 10;
            const centerX = posX + size / 2;
            const midY = posY + height * 0.5;
            this._hit = [posX, size + 6];
            ctx.save();
            ctx.beginPath();
            ctx.arc(centerX, midY, size / 2 - 0.5, 0, Math.PI * 2);
            if (hasCustom) {
                ctx.fillStyle = "rgba(79,195,247,0.3)";
                ctx.strokeStyle = "rgba(79,195,247,0.7)";
            } else {
                ctx.fillStyle = "rgba(255,255,255,0.25)";
                ctx.strokeStyle = "rgba(255,255,255,0.4)";
            }
            ctx.lineWidth = 1;
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = hasCustom ? "rgba(79,195,247,0.9)" : "rgba(255,255,255,0.6)";
            ctx.font = `${Math.round(size * 0.6)}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("i", centerX, midY + 0.5);
            if ((app.canvas.ds?.scale || 1) > 0.5) {
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                ctx.textAlign = "left";
                ctx.fillText("Info", posX + size + 6, midY);
            }
            ctx.restore();
        },
        mouse(event, pos, n) {
            if (event.type !== "pointerdown") return false;
            const b = w._hit;
            if (b[1] < 0) return false;
            if (pos[0] >= b[0] && pos[0] <= b[0] + b[1]) {
                const loraName = getComboValue(n, comboName);
                if (loraName && loraName !== "None") {
                    // 统一为 Stack 同款浮动面板（chip 形态，近节点），不再使用 dialog
                    requestAnimationFrame(() => {
                        setTimeout(() => {
                            const ctx = loaderPanelCtx(n, loraName);
                            openInfoPanelFor(ctx, loraName);
                        }, 0);
                    });
                }
                return true;
            }
            return false;
        },
    };
    return w;
}

// Loader 专用浮动面板宿主（复用 Stack 面板 chip 形态）
const _loaderRows = new Map();
function loaderPanelCtx(node, loraName) {
    if (!_loaderRows.has(loraName)) {
        _loaderRows.set(loraName, { id: loraName, name: loraName, triggers: [], custom: [] });
    }
    return {
        key: "loader:" + node.id + ":" + loraName,
        node,
        anchorRect: () => {
            try {
                const w = node.widgets?.find((x) => x.name === "_info");
                const el = w?.element;
                if (el && el.getBoundingClientRect) {
                    const r = el.getBoundingClientRect();
                    if (r && r.width && r.height) return r;
                }
            } catch {}
            return getNodeRect(node);
        },
        getRow: () => _loaderRows.get(loraName) || { id: loraName, name: loraName, triggers: [], custom: [] },
        patchRow: (patch) => {
            const cur = _loaderRows.get(loraName) || { id: loraName, name: loraName, triggers: [], custom: [] };
            Object.assign(cur, patch);
            cur.id = loraName;
            cur.name = loraName;
            _loaderRows.set(loraName, cur);
            // 触发重绘以更新 i 图标高亮（_has_custom 来自 loraMetadataCache，由 panel 的 saveCustom 触发的全局事件已清缓存，下次 draw 即新）
            try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
        },
        accent: (() => { try { return app.ui.settings.getSettingValue("sfnodes.Accent") || "#f66744"; } catch { return "#f66744"; } })(),
        prefs: () => ({ civitai: true, thumbs: true }),
        refresh: () => { try { node.setDirtyCanvas?.(true, true); } catch {} },
    };
}

// Mounts a standard combo + info-icon widget pair onto a loader node:
// binds the combo callback to prefetch metadata, guards the positional
// restoration of widgets_values, and prefetches the restored value after
// configure (widget values are restored after onNodeCreated).
export function setupLoraInfoWidget(node, comboName = "lora_name") {
    const combo = getComboWidget(node, comboName);
    if (combo) {
        const origCallback = combo.callback;
        combo.callback = (value) => {
            if (origCallback) origCallback(value);
            if (value && value !== "None") getLoraMetadata(value);
        };
    }

    const _origConfigure = node.configure;
    node.configure = function (info) {
        const idx = this.widgets?.findIndex((w) => w.name === "_info") ?? -1;
        if (idx !== -1) this.widgets.splice(idx, 1);
        if (_origConfigure) _origConfigure.call(this, info);
        const loraName = getComboValue(this, comboName);
        if (loraName && loraName !== "None") getLoraMetadata(loraName);
        this.widgets.push(createInfoWidget(comboName));
    };

    node.widgets.push(createInfoWidget(comboName));
}
