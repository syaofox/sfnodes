// ==========================================================================
// SF Model Info - Shared LoRA/model metadata dialog & fetch utilities
// Used by SFPowerLoraLoader and SFLoraLoaderModelOnly (and future nodes).
// ==========================================================================
import { app } from "/scripts/app.js";
import { renderMarkdown } from "./sf_markdown.js";

// ---------------------------------------------------------------------------
// PNG 内嵌工作流解析（ComfyUI SaveImage 写入的 workflow/prompt chunk）
// 返回 { chunk: "workflow" | "prompt", data: string } 或 null
// ---------------------------------------------------------------------------
async function readPngWorkflowData(url) {
    let resp;
    try { resp = await fetch(url); } catch { return null; }
    if (!resp.ok) return null;
    const buf = await resp.arrayBuffer();
    const bytes = new Uint8Array(buf);
    // PNG 签名 89 50 4E 47 ...
    if (bytes.length < 24 || bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) return null;
    const dec = new TextDecoder();
    let off = 8;
    while (off + 12 <= bytes.length) {
        const len = ((bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3]) >>> 0;
        const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
        const dataStart = off + 8;
        const dataEnd = dataStart + len;
        if (dataEnd + 4 > bytes.length) break;
        if (type === "workflow" || type === "prompt") {
            return { chunk: type, data: dec.decode(bytes.slice(dataStart, dataEnd)) };
        }
        if (type === "tEXt") {
            const str = dec.decode(bytes.slice(dataStart, dataEnd));
            const nul = str.indexOf("\0");
            if (nul > 0) {
                const key = str.slice(0, nul);
                const value = str.slice(nul + 1);
                if (key === "workflow" || key === "prompt") return { chunk: key, data: value };
            }
        }
        off = dataEnd + 4;
    }
    return null;
}

// 将图片作为工作流载入：新建工作流标签页（不替换当前画布）
async function loadImageAsWorkflow(path, onError) {
    const url = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}`;
    const embedded = await readPngWorkflowData(url);
    if (!embedded) {
        onError("该图片未内嵌工作流数据，无法载入为工作流（可用 SaveImage 输出的 PNG 测试）。");
        return false;
    }
    try {
        const data = JSON.parse(embedded.data);
        const load = async () => {
            if (embedded.chunk === "prompt" && typeof app.loadApiJson === "function") {
                await app.loadApiJson(data);
            } else {
                await app.loadGraphData(data, true, true);
            }
        };
        // 首选：新建工作流标签页载入，保留当前画布
        const cmd = app.extensionManager?.command;
        if (cmd && typeof cmd.execute === "function") {
            await cmd.execute("Comfy.NewBlankWorkflow");
            await load();
            return true;
        }
        // 兜底（旧版环境无命令系统）：替换当前画布，需确认
        if (!confirm("当前 ComfyUI 不支持新建标签，载入将替换当前画布内容，继续吗？")) return false;
        await load();
        return true;
    } catch (e) {
        console.warn("[SF Model Info] load workflow failed:", e);
        onError("工作流载入失败：" + (e.message || e));
        return false;
    }
}

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
        min-width: 560px; max-width: 720px; max-height: 92vh;
        padding: 0; overflow: hidden; color: #ddd;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;

    const card = document.createElement("div");
    card.style.cssText = `
        display: flex; flex-direction: column; max-height: 92vh;
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
    function createEditRow(displayLabel, key, isTextarea, hint) {
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
        if (hint) labelEl.title = hint;
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
            if (!v) {
                valueEl.style.whiteSpace = "pre-wrap";
                valueEl.innerHTML = '<span style="color:#666;">(empty)</span>';
            } else if (key === "description") {
                // Description 支持 Markdown：查看态渲染，编辑态编辑源码
                valueEl.style.whiteSpace = "normal";
                valueEl.innerHTML = renderMarkdown(v);
            } else {
                valueEl.style.whiteSpace = "pre-wrap";
                valueEl.textContent = v;
            }
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
                input.rows = 12;
                input.style.resize = "vertical";
            }
            if (key === "description") {
                input.placeholder = "支持 Markdown：**加粗**、[链接](url)、列表、代码块；下方示例图点击即可插入";
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
            row._editing = true;
            row._baseValue = state[key];
            row._dirty = false;
            renderFooterActions();

            // 描述行：仅保留上传按钮；示例图面板默认展开；保存/取消移至底部按钮栏
            if (key === "description") {
                actionEl.style.flexDirection = "column";
                if (name && name !== "None") {
                    const uploadBtn = document.createElement("button");
                    uploadBtn.textContent = "📤";
                    uploadBtn.title = "上传图片到该 LoRA 的 sample 目录并插入";
                    uploadBtn.style.cssText = `
                        background: none; border: 1px solid #555; border-radius: 4px;
                        cursor: pointer; font-size: 12px; color: #bbb; padding: 2px 5px;
                    `;
                    uploadBtn.addEventListener("click", () => uploadInput.click());
                    actionEl.appendChild(uploadBtn);
                    openSamplePanel(input);
                }
            } else {
                actionEl.style.flexDirection = "";
            }

            // 内容改动追踪：变化时刷新底部按钮（Save 加 *）
            input.addEventListener("input", () => {
                const dirty = input.value !== row._baseValue;
                if (dirty !== row._dirty) {
                    row._dirty = dirty;
                    renderFooterActions();
                }
            });

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

        // 提交当前编辑（若处于编辑态），返回是否提交过
        row.commitEdit = function () {
            const input = valueEl.querySelector("input,textarea");
            if (!input) return false;
            state[key] = input.value.trim();
            row._editing = false;
            row._dirty = false;
            actionEl.style.flexDirection = "";
            closeSamplePanel();
            renderValue();
            renderActions();
            renderFooterActions();
            return true;
        };

        function saveEdit() {
            if (row.commitEdit()) saveNotes();
        }

        function cancelEdit() {
            if (row._editing) {
                // 放弃修改：恢复进入编辑前的基准值（防御其他路径中途写入 state）
                state[key] = row._baseValue;
            }
            row._editing = false;
            row._dirty = false;
            actionEl.style.flexDirection = "";
            closeSamplePanel();
            renderValue();
            renderActions();
            renderFooterActions();
        }

        row.cancelEdit = cancelEdit;

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
    const descRow = createEditRow(
        "Description",
        "description",
        true,
        "支持 Markdown 格式：![图片](url)、[链接](url)、**加粗**、列表、代码块等"
    );
    body.appendChild(twRow);
    body.appendChild(descRow);
    if (meta.base_model) body.appendChild(createReadonlyRow("Base Model", meta.base_model));
    if (meta.source_url) body.appendChild(createReadonlyRow("Source URL", meta.source_url, meta.source_url));

    // ---------- sample images（描述 Markdown 图片插入） ----------
    const samplePanel = document.createElement("div");
    samplePanel.style.cssText = `
        display: none; padding: 10px 18px; border-bottom: 1px solid #3a3a3e;
    `;
    const samplePanelHead = document.createElement("div");
    samplePanelHead.style.cssText = "display:flex;align-items:center;justify-content:space-between;font-size:12px;color:#aaa;";
    const samplePanelTitle = document.createElement("span");
    samplePanelTitle.textContent = "LoRA 示例图（点击插入）";
    const sampleCloseBtn = document.createElement("button");
    sampleCloseBtn.textContent = "✕";
    sampleCloseBtn.title = "关闭";
    sampleCloseBtn.style.cssText = "background:none;border:none;cursor:pointer;font-size:12px;color:#aaa;padding:0 4px;";
    sampleCloseBtn.addEventListener("click", () => closeSamplePanel());
    samplePanelHead.appendChild(samplePanelTitle);
    samplePanelHead.appendChild(sampleCloseBtn);
    const sampleGrid = document.createElement("div");
    sampleGrid.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;max-height:220px;overflow-y:auto;";
    const sampleHint = document.createElement("div");
    sampleHint.style.cssText = "font-size:12px;color:#888;margin-top:8px;line-height:1.5;";
    samplePanel.appendChild(samplePanelHead);
    samplePanel.appendChild(sampleGrid);
    samplePanel.appendChild(sampleHint);
    body.appendChild(samplePanel);

    let sampleOpen = false;
    let activeTextarea = null;

    const uploadInput = document.createElement("input");
    uploadInput.type = "file";
    uploadInput.accept = "image/*";
    uploadInput.style.display = "none";
    dialog.appendChild(uploadInput);

    function buildSampleMarkdown(path) {
        const base = path.split("/").pop() || "image";
        const alt = base.replace(/\.[^.]+$/, "");
        // encodeURIComponent 不编码括号，需手动转义，避免 markdown 解析截断 URL
        const url = encodeURIComponent(path).replace(/\(/g, "%28").replace(/\)/g, "%29");
        return `![${alt}](/api/sfnodes/lora_samples/image?path=${url})`;
    }

    function insertAtCursor(textarea, text) {
        const start = textarea.selectionStart ?? textarea.value.length;
        const end = textarea.selectionEnd ?? start;
        textarea.setRangeText(text, start, end, "end");
        // 不直接写 state：提交（Save）时统一从输入框读取，取消时保持原始内容
        textarea.focus();
        const pos = start + text.length;
        textarea.selectionStart = textarea.selectionEnd = pos;
    }

    async function refreshSamplePanel() {
        sampleGrid.innerHTML = "";
        sampleHint.textContent = "";
        if (!name || name === "None") {
            sampleHint.textContent = "当前未选择 LoRA。";
            return;
        }
        try {
            const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples?filename=${encodeURIComponent(name)}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (!Array.isArray(data.images) || !data.images.length) {
                sampleHint.textContent = `该 LoRA 没有示例图。请将图片放入 models/loras/${data.sample_dir || ""} 目录，或点击「📤 上传」。`;
                return;
            }
            for (const path of data.images) {
                const wrap = document.createElement("div");
                wrap.style.cssText = "position:relative;width:96px;height:96px;";
                const thumb = document.createElement("img");
                thumb.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}&w=256`;
                thumb.title = path.split("/").pop();
                thumb.loading = "lazy";
                thumb.style.cssText = `
                    width: 96px; height: 96px; object-fit: cover;
                    border-radius: 6px; border: 1px solid #3a3a3e; cursor: pointer;
                    display: block;
                `;
                thumb.addEventListener("mouseenter", () => { thumb.style.borderColor = "#6af"; });
                thumb.addEventListener("mouseleave", () => { thumb.style.borderColor = "#3a3a3e"; });
                thumb.addEventListener("click", () => {
                    if (activeTextarea) insertAtCursor(activeTextarea, buildSampleMarkdown(path));
                });
                // 删除按钮：悬停显示，右上角 ✕
                const delBtn = document.createElement("button");
                delBtn.textContent = "✕";
                delBtn.title = "删除该示例图";
                delBtn.style.cssText = `
                    position: absolute; top: 0; right: 0; display: none;
                    width: 18px; height: 18px; padding: 0; line-height: 1;
                    background: rgba(224, 108, 108, 0.9); color: #fff;
                    border: none; border-radius: 0 6px 0 6px; cursor: pointer;
                    font-size: 11px;
                `;
                wrap.addEventListener("mouseenter", () => {
                    delBtn.style.display = "block";
                    loadBtn.style.display = "block";
                });
                wrap.addEventListener("mouseleave", () => {
                    delBtn.style.display = "none";
                    loadBtn.style.display = "none";
                });
                // 载入工作流按钮：悬停显示，右下角 📂（解析 PNG 内嵌 workflow 数据）
                const loadBtn = document.createElement("button");
                loadBtn.textContent = "📂";
                loadBtn.title = "将该图片载入为工作流（需内嵌工作流数据）";
                loadBtn.style.cssText = `
                    position: absolute; bottom: 0; right: 0; display: none;
                    width: 18px; height: 18px; padding: 0; line-height: 1;
                    background: rgba(79, 124, 255, 0.9); color: #fff;
                    border: none; border-radius: 6px 0 6px 0; cursor: pointer;
                    font-size: 11px;
                `;
                loadBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    await loadImageAsWorkflow(path, (msg) => { sampleHint.textContent = msg; });
                });
                delBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const fileName = path.split("/").pop();
                    if (!confirm(`删除示例图「${fileName}」？此操作不可恢复。`)) return;
                    try {
                        const resp = await app.api.fetchApi(
                            `/api/sfnodes/lora_samples?path=${encodeURIComponent(path)}`,
                            { method: "DELETE" }
                        );
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        refreshSamplePanel();
                    } catch (err) {
                        console.warn("[SF Model Info] sample delete failed:", err);
                        sampleHint.textContent = "删除失败：" + (err.message || err);
                    }
                });
                wrap.appendChild(thumb);
                wrap.appendChild(delBtn);
                wrap.appendChild(loadBtn);
                sampleGrid.appendChild(wrap);
            }
        } catch (e) {
            console.warn("[SF Model Info] lora_samples list failed:", e);
            sampleHint.textContent = "获取示例图失败：" + (e.message || e);
        }
    }

    function openSamplePanel(textarea) {
        activeTextarea = textarea;
        sampleOpen = true;
        samplePanel.style.display = "block";
        refreshSamplePanel();
    }

    function closeSamplePanel() {
        sampleOpen = false;
        activeTextarea = null;
        samplePanel.style.display = "none";
    }

    uploadInput.addEventListener("change", async () => {
        const file = uploadInput.files?.[0];
        uploadInput.value = "";
        if (!file) return;
        if (!name || name === "None") return;
        const fd = new FormData();
        fd.append("image", file);
        fd.append("filename", name);
        try {
            const resp = await app.api.fetchApi("/api/sfnodes/lora_samples/upload", {
                method: "POST",
                body: fd,
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            if (activeTextarea) insertAtCursor(activeTextarea, buildSampleMarkdown(data.path));
            if (sampleOpen) refreshSamplePanel();
        } catch (e) {
            console.warn("[SF Model Info] sample upload failed:", e);
            sampleHint.textContent = "上传失败：" + (e.message || e);
        }
    });

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
    const footerRight = document.createElement("div");
    footerRight.style.cssText = "display: flex; gap: 8px;";

    // 底部右侧按钮随状态切换：编辑中 = 取消 + Save；浏览态 = Done（关闭）
    function renderFooterActions() {
        footerRight.innerHTML = "";
        const editing = twRow._editing || descRow._editing;
        if (editing) {
            const cancelBtn = makeFooterBtn("✕ Cancel", "#aaa", () => {
                twRow.cancelEdit();
                descRow.cancelEdit();
            }, "放弃修改返回浏览页");
            const dirty = twRow._dirty || descRow._dirty;
            const saveBtn = makeFooterBtn(dirty ? "Save*" : "Save", "#4f7cff", () => {
                twRow.commitEdit();
                descRow.commitEdit();
                saveNotes();
            }, "保存并返回浏览页");
            footerRight.appendChild(cancelBtn);
            footerRight.appendChild(saveBtn);
        } else {
            const doneBtn = makeFooterBtn("Done", "#4f7cff", () => closeDialog(), "关闭");
            footerRight.appendChild(doneBtn);
        }
    }

    footer.appendChild(copyBtn);
    footer.appendChild(clearBtn);
    footer.appendChild(spacer);
    footer.appendChild(footerRight);
    renderFooterActions();

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
        if (!dialog.open) return;
        // 有未保存的修改时确认，防止误关丢失内容
        if ((twRow._editing && twRow._dirty) || (descRow._editing && descRow._dirty)) {
            if (!confirm("有未保存的修改，确定要关闭吗？")) return;
        }
        dialog.close();
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
    // 以 mousedown 位置判定：编辑中布局变化（如 markdown 渲染内容切换为
    // textarea）会使 dialog 收缩/位移，click 事件坐标可能落到新矩形之外，
    // 误判为背景点击而关闭弹窗。mousedown 在框内则忽略该次 click。
    let mouseDownInside = false;
    dialog.addEventListener("mousedown", (e) => {
        const rect = dialog.getBoundingClientRect();
        mouseDownInside = (
            e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom
        );
    });
    dialog.addEventListener("click", (e) => {
        if (mouseDownInside) return;
        // 合成事件（element.click()）clientX/Y 为 0,0，按框内处理，避免误关
        if (!e.clientX && !e.clientY) return;
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
