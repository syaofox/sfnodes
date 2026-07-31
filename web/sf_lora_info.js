// ==========================================================================
// SF Model Info - Shared LoRA/model metadata dialog & fetch utilities
// Used by SFPowerLoraLoader and SFLoraLoaderModelOnly (and future nodes).
// ==========================================================================
import { app } from "/scripts/app.js";

// ---------------------------------------------------------------------------
// Metadata fetch (merged custom notes + embedded safetensors metadata via
// the /api/sfnodes/lora_notes endpoint, generic over folder type)
// ---------------------------------------------------------------------------

export const loraMetadataCache = new Map();
const _loraMetadataPending = new Map();

export async function getLoraMetadata(name, modelType = "loras") {
    if (!name || name === "None") return null;
    if (loraMetadataCache.has(name)) return loraMetadataCache.get(name);
    // Join an in-flight request instead of firing a duplicate
    if (_loraMetadataPending.has(name)) return _loraMetadataPending.get(name);

    const typeParam = modelType && modelType !== "loras" ? `&type=${encodeURIComponent(modelType)}` : "";
    const promise = (async () => {
        try {
            const resp = await fetch(`/api/sfnodes/lora_notes?filename=${encodeURIComponent(name)}${typeParam}`);
            if (!resp.ok) { loraMetadataCache.set(name, null); return null; }
            const meta = await resp.json();
            loraMetadataCache.set(name, meta);
            return meta;
        } catch {
            loraMetadataCache.set(name, null);
            return null;
        }
    })();

    _loraMetadataPending.set(name, promise);
    try { return await promise; }
    finally { _loraMetadataPending.delete(name); }
}

// ---------------------------------------------------------------------------
// Info dialog (native <dialog> modal, like rgthree)
// ---------------------------------------------------------------------------

export function showLoraInfoDialog(event, name, meta, modelType = "loras") {
    meta = meta || {};
    const state = {
        trigger_words: meta.trigger_words || "",
        description: meta.description || "",
    };

    // ---------- dialog (native modal, like rgthree) ----------
    if (!showLoraInfoDialog._cssInjected) {
        showLoraInfoDialog._cssInjected = true;
        const style = document.createElement("style");
        style.textContent = `
            dialog.sf-lora-info::backdrop { background: rgba(0,0,0,0.5); }
        `;
        document.head.appendChild(style);
    }

    const dialog = document.createElement("dialog");
    dialog.className = "sf-lora-info";
    dialog.style.cssText = `
        background: #2a2a2e; border: 1px solid #555; border-radius: 10px;
        min-width: 460px; max-width: 580px; max-height: 85vh;
        padding: 0; overflow: hidden; color: #ddd;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;

    const card = document.createElement("div");
    card.style.cssText = `
        display: flex; flex-direction: column; max-height: 85vh;
    `;

    // ---------- header ----------
    const header = document.createElement("div");
    header.style.cssText = `
        display: flex; align-items: center; justify-content: space-between;
        gap: 12px; padding: 14px 18px; border-bottom: 1px solid #444;
    `;
    const title = document.createElement("div");
    title.textContent = name;
    title.title = name;
    title.style.cssText = `
        font-size: 13px; font-weight: 600; color: #fff;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    `;
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.title = "Close";
    closeBtn.style.cssText = `
        flex: 0 0 auto; background: none; border: none; cursor: pointer;
        font-size: 14px; color: #aaa; padding: 2px 6px; border-radius: 4px;
    `;
    closeBtn.addEventListener("mouseenter", () => { closeBtn.style.color = "#fff"; });
    closeBtn.addEventListener("mouseleave", () => { closeBtn.style.color = "#aaa"; });
    closeBtn.addEventListener("click", () => closeDialog());
    header.appendChild(title);
    header.appendChild(closeBtn);

    // ---------- body ----------
    const body = document.createElement("div");
    body.style.cssText = "overflow-y: auto; padding: 6px 0;";

    // row factory: editable rows
    function createEditRow(displayLabel, key, isTextarea) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex; align-items: flex-start; gap: 10px;
            padding: 10px 18px; border-bottom: 1px solid #3a3a3e;
        `;
        const labelEl = document.createElement("div");
        labelEl.style.cssText = `
            flex: 0 0 100px; font-size: 12px; color: #aaa;
            padding-top: 5px; line-height: 1.4;
        `;
        labelEl.textContent = displayLabel;
        const valueEl = document.createElement("div");
        valueEl.style.cssText = `
            flex: 1; font-size: 13px; color: #eee; line-height: 1.5;
            white-space: pre-wrap; word-break: break-word; min-height: 20px;
        `;
        const actionEl = document.createElement("div");
        actionEl.style.cssText = "flex: 0 0 auto; display: flex; gap: 4px; align-items: center;";
        row.appendChild(labelEl);
        row.appendChild(valueEl);
        row.appendChild(actionEl);

        function renderValue() {
            valueEl.innerHTML = "";
            const v = state[key];
            if (!v) valueEl.innerHTML = '<span style="color:#666;">(empty)</span>';
            else valueEl.textContent = v;
            valueEl.title = v;
        }

        function renderActions() {
            actionEl.innerHTML = "";
            const btn = document.createElement("button");
            btn.textContent = "✏️";
            btn.title = "Edit " + displayLabel;
            btn.style.cssText = `
                background: none; border: 1px solid #555; border-radius: 4px;
                cursor: pointer; font-size: 12px; color: #bbb; padding: 2px 6px;
            `;
            btn.addEventListener("mouseenter", () => { btn.style.background = "#3a3a3e"; });
            btn.addEventListener("mouseleave", () => { btn.style.background = ""; });
            btn.addEventListener("click", () => startEdit());
            actionEl.appendChild(btn);
        }

        function startEdit() {
            const input = isTextarea ? document.createElement("textarea") : document.createElement("input");
            input.value = state[key];
            if (isTextarea) {
                input.rows = 4;
                input.style.resize = "vertical";
            }
            input.style.cssText = `
                width: 100%; box-sizing: border-box;
                background: #1a1a1e; color: #eee; border: 1px solid #6af;
                border-radius: 6px; padding: 6px 8px; font-size: 13px;
                font-family: inherit; outline: none;
            `;
            valueEl.innerHTML = "";
            valueEl.appendChild(input);
            actionEl.innerHTML = "";
            // save button
            const saveBtn = document.createElement("button");
            saveBtn.textContent = "💾";
            saveBtn.title = "Save (Enter)";
            saveBtn.style.cssText = `
                background: none; border: 1px solid #4f7cff; border-radius: 4px;
                cursor: pointer; font-size: 12px; color: #7aa2ff; padding: 2px 6px;
            `;
            saveBtn.addEventListener("click", () => saveEdit());
            // cancel button
            const cancelBtn = document.createElement("button");
            cancelBtn.textContent = "✕";
            cancelBtn.title = "Cancel (Esc)";
            cancelBtn.style.cssText = `
                background: none; border: 1px solid #555; border-radius: 4px;
                cursor: pointer; font-size: 12px; color: #aaa; padding: 2px 6px;
            `;
            cancelBtn.addEventListener("click", () => cancelEdit());
            actionEl.appendChild(saveBtn);
            actionEl.appendChild(cancelBtn);

            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !isTextarea) {
                    e.preventDefault();
                    saveEdit();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    cancelEdit();
                }
            });
            input.focus();
            input.select();
        }

        function saveEdit() {
            const input = valueEl.querySelector("input,textarea");
            if (!input) return;
            const newVal = input.value.trim();
            state[key] = newVal;
            renderValue();
            renderActions();
            saveNotes();
        }

        function cancelEdit() {
            renderValue();
            renderActions();
        }

        row.refresh = function () {
            renderValue();
            renderActions();
        };

        renderValue();
        renderActions();
        return row;
    }

    // read-only row factory
    function createReadonlyRow(displayLabel, value, linkUrl) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex; align-items: flex-start; gap: 10px;
            padding: 10px 18px; border-bottom: 1px solid #3a3a3e;
        `;
        const labelEl = document.createElement("div");
        labelEl.style.cssText = `
            flex: 0 0 100px; font-size: 12px; color: #aaa; padding-top: 5px;
        `;
        labelEl.textContent = displayLabel;
        const valueEl = document.createElement("div");
        valueEl.style.cssText = `
            flex: 1; font-size: 13px; color: #eee; line-height: 1.5;
            white-space: pre-wrap; word-break: break-word; min-height: 20px;
        `;
        if (linkUrl && value) {
            const a = document.createElement("a");
            a.href = linkUrl;
            a.target = "_blank";
            a.rel = "noopener";
            a.textContent = value;
            a.style.cssText = "color: #7aa2ff; text-decoration: none; word-break: break-all;";
            a.addEventListener("mouseenter", () => { a.style.textDecoration = "underline"; });
            a.addEventListener("mouseleave", () => { a.style.textDecoration = ""; });
            valueEl.appendChild(a);
        } else {
            valueEl.textContent = value || "";
            if (!value) valueEl.innerHTML = '<span style="color:#666;">(empty)</span>';
        }
        row.appendChild(labelEl);
        row.appendChild(valueEl);
        return row;
    }

    // ---------- build rows ----------
    const twRow = createEditRow("Trigger Words", "trigger_words", false);
    const descRow = createEditRow("Description", "description", true);
    body.appendChild(twRow);
    body.appendChild(descRow);
    if (meta.base_model) body.appendChild(createReadonlyRow("Base Model", meta.base_model));
    if (meta.source_url) body.appendChild(createReadonlyRow("Source URL", meta.source_url, meta.source_url));

    // ---------- footer ----------
    const footer = document.createElement("div");
    footer.style.cssText = `
        display: flex; align-items: center; gap: 8px;
        padding: 12px 18px; border-top: 1px solid #444;
    `;

    function makeFooterBtn(text, color, callback, title) {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.title = title || "";
        btn.style.cssText = `
            padding: 6px 14px; border: 1px solid ${color}; border-radius: 6px;
            font-size: 12px; cursor: pointer; color: ${color};
            background: transparent; transition: filter 0.15s;
        `;
        btn.addEventListener("mouseenter", () => { btn.style.filter = "brightness(1.3)"; });
        btn.addEventListener("mouseleave", () => { btn.style.filter = ""; });
        btn.addEventListener("click", callback);
        return btn;
    }

    const copyBtn = makeFooterBtn("📋 Copy Trigger Words", "#aaa", () => {
        if (state.trigger_words) {
            navigator.clipboard.writeText(state.trigger_words).catch(() => {});
        }
    }, "Copy trigger words to clipboard");
    const clearBtn = makeFooterBtn("🗑️ Clear Notes", "#e06c6c", () => {
        state.trigger_words = "";
        state.description = "";
        twRow.refresh();
        descRow.refresh();
        saveNotes();
    }, "Clear custom notes for this model");
    const spacer = document.createElement("div");
    spacer.style.cssText = "flex: 1;";
    const doneBtn = makeFooterBtn("Done", "#4f7cff", () => closeDialog());

    footer.appendChild(copyBtn);
    footer.appendChild(clearBtn);
    footer.appendChild(spacer);
    footer.appendChild(doneBtn);

    card.appendChild(header);
    card.appendChild(body);
    card.appendChild(footer);
    dialog.appendChild(card);

    // ---------- actions ----------
    function saveNotes() {
        const bodyData = {
            trigger_words: state.trigger_words,
            description: state.description,
        };
        const typeParam = modelType && modelType !== "loras" ? `&type=${encodeURIComponent(modelType)}` : "";
        fetch(`/api/sfnodes/lora_notes?filename=${encodeURIComponent(name)}${typeParam}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyData),
        })
            .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
            .then(updated => {
                loraMetadataCache.set(name, updated);
                app.graph.setDirtyCanvas(true, true);
                state.trigger_words = updated.trigger_words || "";
                state.description = updated.description || "";
                twRow.refresh();
                descRow.refresh();
            })
            .catch(e => console.warn("[SF Model Info] Failed to save notes:", e));
    }

    function closeDialog() {
        if (dialog.open) dialog.close();
    }

    // Native <dialog> modal: Esc triggers "cancel" (unless an input is being
    // edited, whose keydown handler stopPropagation's Escape first).
    dialog.addEventListener("cancel", (e) => {
        e.preventDefault();
        closeDialog();
    });
    dialog.addEventListener("close", () => {
        dialog.remove();
    });
    // Click on the backdrop (outside the dialog box) closes it.
    dialog.addEventListener("click", (e) => {
        const rect = dialog.getBoundingClientRect();
        if (
            e.clientX < rect.left || e.clientX > rect.right ||
            e.clientY < rect.top || e.clientY > rect.bottom
        ) {
            closeDialog();
        }
    });

    document.body.appendChild(dialog);
    dialog.showModal();
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
// SFLoraLoaderModelOnly and future loader nodes)
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
                    // 延迟到 pointerup 由 canvas 处理完成后再打开对话框，
                    // 避免 DOM 遮罩在点击过程中出现导致 LiteGraph widget 交互状态残留
                    getLoraMetadata(loraName).then((meta) => {
                        requestAnimationFrame(() => {
                            setTimeout(() => showLoraInfoDialog(event, loraName, meta, "loras"), 0);
                        });
                    });
                }
                return true;
            }
            return false;
        },
    };
    return w;
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
