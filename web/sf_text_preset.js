// SF Text Preset 前端扩展
// 预设真源为全局库 user/sfnodes/text_presets.json（经 /api/sfnodes/text_presets 读写，跨工作流共享）；
// 节点隐藏 widget presets_json 仅作旧工作流兼容回退（只读，combo 同时显示其残留项）。
// 编辑框为草稿语义：修改写本节点 text_override（随工作流保存，仅影响本节点输出），
// 切换预设即丢弃，「↧ 保存到预设」确认后才写入全局库。
// 前端重建下拉选项并提供弹窗管理（新增/编辑/删除），combo 与预览即时同步；
// 后端路由不可用时选项降级回工作流数据源（编辑仍走草稿，不写 presets_json）。

import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { injectCSSOnce, installWheelZoomPassthrough } from "./sf_common.js";

const API = "/api/sfnodes/text_presets";

let mgrEl = null;
const MGR_CSS = `
.sf-preset-mgr-overlay{position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,0.45);display:flex;align-items:center;justify-content:center;}
.sf-preset-mgr{background:#232323;border:1px solid #555;border-radius:8px;box-shadow:0 8px 30px rgba(0,0,0,0.6);width:min(600px,92vw);max-height:82vh;display:flex;flex-direction:column;font-size:12px;color:#ddd;}
.sf-preset-mgr-head{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-bottom:1px solid #3a3a3a;font-weight:600;font-size:13px;}
.sf-preset-mgr-close{background:none;border:none;color:#aaa;font-size:16px;cursor:pointer;padding:0 4px;}
.sf-preset-mgr-close:hover{color:#fff;}
.sf-preset-mgr-body{display:flex;min-height:0;flex:1;}
.sf-preset-mgr-list{width:42%;border-right:1px solid #3a3a3a;overflow-y:auto;padding:6px 8px;}
.sf-preset-mgr-empty{padding:20px;text-align:center;color:#888;}
.sf-preset-mgr-item{padding:5px 8px;border-radius:4px;cursor:pointer;margin-bottom:2px;border:1px solid transparent;}
.sf-preset-mgr-item:hover{background:#3a3a3a;}
.sf-preset-mgr-item.active{background:#3a5f8a;border-color:#4a7ab0;}
.sf-preset-mgr-item-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-preset-mgr-item-summary{font-size:11px;color:#aaa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sf-preset-mgr-editor{flex:1;display:flex;flex-direction:column;padding:8px 10px;gap:6px;min-width:0;}
.sf-preset-mgr-editor label{font-size:11px;color:#aaa;}
.sf-preset-mgr-input{width:100%;box-sizing:border-box;padding:5px 8px;background:#2c2c2c;border:1px solid #555;border-radius:4px;color:#ddd;font-size:12px;}
.sf-preset-mgr-textarea{flex:1;min-height:120px;resize:none;box-sizing:border-box;padding:5px 8px;background:#2c2c2c;border:1px solid #555;border-radius:4px;color:#ddd;font-size:12px;font-family:monospace;line-height:1.5;}
.sf-preset-mgr-btns{display:flex;gap:6px;}
.sf-preset-mgr-btn{padding:5px 12px;border-radius:4px;cursor:pointer;font-size:12px;border:1px solid #555;background:#2c2c2c;color:#ddd;}
.sf-preset-mgr-btn:hover{background:#3a3a3a;}
.sf-preset-mgr-btn.primary{background:#3a5f8a;border-color:#4a7ab0;color:#fff;}
.sf-preset-mgr-btn.primary:hover{background:#4a7ab0;}
.sf-preset-mgr-btn.danger{background:#5a2f2f;border-color:#7a4040;color:#f0b0b0;}
.sf-preset-mgr-btn.danger:hover{background:#7a4040;color:#fff;}
`;

function injectMgrStyle() {
    injectCSSOnce("sf-tp-mgr-css", MGR_CSS);
}

// ---------------- 数据解析 ----------------

function parsePresets(value) {
    try {
        const data = JSON.parse(value || "[]");
        if (!Array.isArray(data)) return [];
        return data
            .filter((x) => x && typeof x === "object")
            .map((x) => ({
                name: String(x.name ?? "").trim(),
                text: String(x.text ?? ""),
            }))
            .filter((x) => x.name);
    } catch (e) {
        return [];
    }
}

// ---------------- 全局库 API ----------------

async function fetchGlobalPresets() {
    try {
        const r = await fetch(API, { cache: "no-store" });
        if (!r.ok) throw new Error(`load presets failed: ${r.status}`);
        const j = await r.json();
        const arr = Array.isArray(j?.presets) ? j.presets : [];
        return arr
            .filter((x) => x && typeof x === "object" && String(x.name ?? "").trim())
            .map((x) => ({ name: String(x.name).trim(), text: String(x.text ?? "") }));
    } catch (e) {
        console.warn("[SFTextPreset] 全局预设库加载失败，回退工作流数据源:", e);
        return null;
    }
}

async function apiSave(name, text) {
    try {
        const r = await fetch(API, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, text }),
        });
        if (!r.ok) throw new Error(`save preset failed: ${r.status}`);
        return true;
    } catch (e) {
        console.warn("[SFTextPreset]", e);
        return false;
    }
}

async function apiDelete(name) {
    try {
        const r = await fetch(`${API}?name=${encodeURIComponent(name)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`delete preset failed: ${r.status}`);
        return true;
    } catch (e) {
        console.warn("[SFTextPreset]", e);
        return false;
    }
}

// ---------------- 节点状态（全局库 + 工作流残留合并） ----------------

function findWidget(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function workflowPresets(node) {
    return parsePresets(findWidget(node, "presets_json")?.value);
}

// 节点状态：{presets: 合并列表, globalNames: Set|null}
// globalNames 为 null 表示后端路由不可用（降级：工作流数据源，保持旧行为）
function stateOf(node) {
    if (!node?._sfTpState) {
        node._sfTpState = { presets: workflowPresets(node), globalNames: null };
    }
    return node._sfTpState;
}

async function rebuildState(node) {
    if (!node) return;
    // 先捕获当前选中值：configure 恢复的选中项是工作流级持久化真源，
    // 异步重建期间任何中间态都不得清掉它（选中预设仍存在则必须保留）
    const presetWidget = findWidget(node, "preset");
    const current = presetWidget ? presetWidget.value : undefined;
    const global = await fetchGlobalPresets();
    const wf = workflowPresets(node);
    if (global === null) {
        node._sfTpState = { presets: wf, globalNames: null };
    } else {
        const names = new Set(global.map((p) => p.name));
        node._sfTpState = {
            presets: global.concat(wf.filter((p) => !names.has(p.name))),
            globalNames: names,
        };
    }
    const state = node._sfTpState;
    if (presetWidget && current !== undefined && state.presets.some((p) => p.name === current)) {
        presetWidget.value = current;
    }
    syncFromJson(node);
}

function refreshAllNodes() {
    const nodes = app.graph?._nodes ?? [];
    nodes.forEach((n) => {
        if (n?.comfyClass === "SFTextPreset") rebuildState(n);
    });
}

function setPresetWidgetValues(node, presets) {
    const presetWidget = findWidget(node, "preset");
    if (!presetWidget) return;
    const names = presets.map((p) => p.name);
    presetWidget.options.values = names.length > 0 ? names : [""];
    if (names.includes(presetWidget.value)) return;
    presetWidget.value = names.length > 0 ? names[0] : "";
    if (typeof presetWidget.callback === "function") {
        presetWidget.callback(presetWidget.value);
    }
}

function refreshContentDisplay(node) {
    const display = findWidget(node, "content_display");
    if (!display) return;
    const state = stateOf(node);
    const name = findWidget(node, "preset")?.value ?? "";
    // 草稿优先：编辑过的文本显示草稿（仅本节点，未写全局库）
    const draft = findWidget(node, "text_override")?.value ?? "";
    const preset = state.presets.find((p) => p.name === name);
    display.value = draft ? draft : preset ? preset.text : "";
}

function saveEditedText(node, text) {
    const presetWidget = findWidget(node, "preset");
    if (!presetWidget) return;
    const name = presetWidget.value ?? "";
    if (!name) return;
    // 草稿语义：编辑只写本节点隐藏载体（随工作流保存），不写全局库；
    // 切换预设即丢弃，点「↧ 保存到预设」确认后才写入全局库
    const draft = findWidget(node, "text_override");
    if (!draft) return;
    if (draft.value === text) return;
    draft.value = text;
    node.setDirtyCanvas?.(true, true);
}

function updateDisplayEditable(node) {
    const display = findWidget(node, "content_display");
    if (!display?.inputEl) return;
    // 有选中预设即可编辑（含工作流残留项：编辑产生草稿，↧ 可将其晋升为全局预设）
    const name = findWidget(node, "preset")?.value ?? "";
    const state = stateOf(node);
    display.inputEl.readOnly = !state.presets.some((p) => p.name === name) && !findWidget(node, "text_override")?.value;
}

function syncFromJson(node) {
    if (!node) return;
    setPresetWidgetValues(node, stateOf(node).presets);
    refreshContentDisplay(node);
    updateDisplayEditable(node);
    node.setDirtyCanvas?.(true, true);
}

// ---------------- 管理弹窗（编辑全局库） ----------------

function closeMgr() {
    if (mgrEl) {
        mgrEl.remove();
        mgrEl = null;
    }
}

let mgrOpening = false;

async function openMgr(node) {
    if (mgrOpening) return; // 防双击重入：await 加载期间 mgrEl 尚为 null，二次进入会泄漏一个 Escape 关不掉的弹窗
    mgrOpening = true;
    injectMgrStyle();
    if (mgrEl) closeMgr();

    const global = await fetchGlobalPresets();
    mgrOpening = false;
    if (global === null) {
        alert("全局预设库加载失败（后端路由不可用？请重启 ComfyUI 后重试）");
        return;
    }
    let presets = global;
    let selectedIndex = -1;

    const overlay = document.createElement("div");
    overlay.className = "sf-preset-mgr-overlay";
    mgrEl = overlay;
    const panel = document.createElement("div");
    panel.className = "sf-preset-mgr";

    const head = document.createElement("div");
    head.className = "sf-preset-mgr-head";
    const title = document.createElement("span");
    title.textContent = "预设管理";
    const closeBtn = document.createElement("button");
    closeBtn.className = "sf-preset-mgr-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", closeMgr);
    head.appendChild(title);
    head.appendChild(closeBtn);
    panel.appendChild(head);

    const body = document.createElement("div");
    body.className = "sf-preset-mgr-body";

    const list = document.createElement("div");
    list.className = "sf-preset-mgr-list";
    body.appendChild(list);

    const editor = document.createElement("div");
    editor.className = "sf-preset-mgr-editor";
    const nameLabel = document.createElement("label");
    nameLabel.textContent = "名称";
    const nameInput = document.createElement("input");
    nameInput.className = "sf-preset-mgr-input";
    nameInput.type = "text";
    nameInput.placeholder = "预设名称（下拉框显示）";
    installWheelZoomPassthrough(nameInput); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
    const textLabel = document.createElement("label");
    textLabel.textContent = "文本内容";
    const textArea = document.createElement("textarea");
    textArea.className = "sf-preset-mgr-textarea";
    textArea.placeholder = "预设输出的文本";
    installWheelZoomPassthrough(textArea); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
    const btns = document.createElement("div");
    btns.className = "sf-preset-mgr-btns";

    const addBtn = document.createElement("button");
    addBtn.className = "sf-preset-mgr-btn primary";
    addBtn.textContent = "新增";
    const updateBtn = document.createElement("button");
    updateBtn.className = "sf-preset-mgr-btn";
    updateBtn.textContent = "更新";
    const delBtn = document.createElement("button");
    delBtn.className = "sf-preset-mgr-btn danger";
    delBtn.textContent = "删除";
    btns.appendChild(addBtn);
    btns.appendChild(updateBtn);
    btns.appendChild(delBtn);
    editor.appendChild(nameLabel);
    editor.appendChild(nameInput);
    editor.appendChild(textLabel);
    editor.appendChild(textArea);
    editor.appendChild(btns);
    body.appendChild(editor);
    panel.appendChild(body);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    function renderList() {
        list.replaceChildren();
        if (presets.length === 0) {
            const empty = document.createElement("div");
            empty.className = "sf-preset-mgr-empty";
            empty.textContent = "暂无预设，请在右侧填写后点击「新增」";
            list.appendChild(empty);
            return;
        }
        presets.forEach((p, idx) => {
            const item = document.createElement("div");
            item.className = "sf-preset-mgr-item" + (idx === selectedIndex ? " active" : "");
            const nameEl = document.createElement("div");
            nameEl.className = "sf-preset-mgr-item-name";
            nameEl.textContent = p.name;
            const summary = document.createElement("div");
            summary.className = "sf-preset-mgr-item-summary";
            summary.textContent = p.text.split("\n")[0];
            item.appendChild(nameEl);
            item.appendChild(summary);
            item.addEventListener("click", () => {
                selectedIndex = idx;
                nameInput.value = p.name;
                textArea.value = p.text;
                renderList();
            });
            list.appendChild(item);
        });
    }

    function nameConflict(name, ignoreIndex) {
        return presets.some((p, idx) => idx !== ignoreIndex && p.name === name);
    }

    function saveFailAlert() {
        alert("保存失败，请检查后端服务（需重启 ComfyUI 生效）");
    }

    async function handleAdd() {
        const name = nameInput.value.trim();
        if (!name) {
            alert("预设名称不能为空");
            return;
        }
        if (nameConflict(name, -1)) {
            alert(`已存在名为「${name}」的预设`);
            return;
        }
        if (!(await apiSave(name, textArea.value))) {
            saveFailAlert();
            return;
        }
        presets.push({ name, text: textArea.value });
        selectedIndex = presets.length - 1;
        renderList();
        textArea.focus();
        refreshAllNodes();
    }

    async function handleUpdate() {
        if (selectedIndex < 0 || selectedIndex >= presets.length) {
            alert("请先在左侧选择要更新的预设");
            return;
        }
        const oldName = presets[selectedIndex].name;
        const name = nameInput.value.trim();
        if (!name) {
            alert("预设名称不能为空");
            return;
        }
        if (nameConflict(name, selectedIndex)) {
            alert(`已存在名为「${name}」的预设`);
            return;
        }
        if (!(await apiSave(name, textArea.value))) {
            saveFailAlert();
            return;
        }
        if (name !== oldName && !(await apiDelete(oldName))) {
            console.warn(`[SFTextPreset] 旧名「${oldName}」清理失败（可能产生重名残留）`);
        }
        presets[selectedIndex] = { name, text: textArea.value };
        renderList();
        refreshAllNodes();
    }

    async function handleDelete() {
        if (selectedIndex < 0 || selectedIndex >= presets.length) {
            alert("请先在左侧选择要删除的预设");
            return;
        }
        const name = presets[selectedIndex].name;
        if (!(await apiDelete(name))) {
            saveFailAlert();
            return;
        }
        presets.splice(selectedIndex, 1);
        if (presets.length > 0) {
            selectedIndex = Math.min(selectedIndex, presets.length - 1);
            const p = presets[selectedIndex];
            nameInput.value = p.name;
            textArea.value = p.text;
        } else {
            selectedIndex = -1;
            nameInput.value = "";
            textArea.value = "";
        }
        renderList();
        refreshAllNodes();
    }

    addBtn.addEventListener("click", handleAdd);
    updateBtn.addEventListener("click", handleUpdate);
    delBtn.addEventListener("click", handleDelete);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeMgr();
    });

    renderList();
    if (presets.length > 0) {
        selectedIndex = 0;
        const p = presets[0];
        nameInput.value = p.name;
        textArea.value = p.text;
        renderList();
    }
}

function onKeyDownCapture(e) {
    if (mgrEl && e.key === "Escape") closeMgr();
}

// 「↧ 保存到预设」：把当前草稿（文本框内容）写入全局库并清草稿。
// 选中为工作流残留项时保存即把它新增为全局预设（晋升）。
async function saveDraftToPreset(node) {
    const presetWidget = findWidget(node, "preset");
    const display = findWidget(node, "content_display");
    if (!presetWidget || !display) return;
    const name = presetWidget.value ?? "";
    if (!name) {
        alert("未选中预设，无法保存");
        return;
    }
    if (!(await apiSave(name, display.value))) {
        alert("保存失败，请检查后端服务（需重启 ComfyUI 生效）");
        return;
    }
    const draft = findWidget(node, "text_override");
    if (draft) draft.value = "";
    refreshAllNodes(); // 本节点草稿已入库，同步所有节点的显示与选项
}

// ---------------- 节点挂载 ----------------

app.registerExtension({
    name: "sfnodes.text_preset",

    setup() {
        document.addEventListener("keydown", onKeyDownCapture, true);
    },

    nodeCreated(node) {
        if (node?.comfyClass !== "SFTextPreset") return;

        const presetsWidget = findWidget(node, "presets_json");
        const presetWidget = findWidget(node, "preset");
        if (!presetsWidget || !presetWidget) return;

        // 隐藏 JSON 数据载体（后端 display:hidden 在 Vue 新版前端不生效），
        // 保留在 widgets 数组中 → 值仍随 workflow 序列化/恢复（旧工作流兼容回退用）
        presetsWidget.hidden = true;
        presetsWidget.computeSize = () => [0, 0];
        presetsWidget.draw = () => {};

        // 草稿载体同样隐藏（值随 workflow 序列化/恢复 = 工作流级草稿）
        const draftWidget = findWidget(node, "text_override");
        if (draftWidget) {
            draftWidget.hidden = true;
            draftWidget.computeSize = () => [0, 0];
            draftWidget.draw = () => {};
        }

        const originalCallback = presetWidget.callback;
        presetWidget.callback = function (...args) {
            // 切换预设即丢弃草稿：草稿永远属于当前选中项
            if (draftWidget) draftWidget.value = "";
            refreshContentDisplay(node);
            updateDisplayEditable(node);
            if (typeof originalCallback === "function") {
                return originalCallback.apply(this, args);
            }
        };

        if (!findWidget(node, "content_display")) {
            const display = ComfyWidgets["STRING"](
                node,
                "content_display",
                ["STRING", { multiline: true }],
                app
            ).widget;
            display.serialize = false;
            // 编辑框 = 草稿：输入即写本节点 text_override（随工作流保存，不影响
            // 全局库与其他工作流）；「↧ 保存到预设」确认后才写入全局库
            display.inputEl.readOnly = false;
            display.inputEl.addEventListener("input", () => {
                saveEditedText(node, display.value);
                refreshContentDisplay(node);
            });
        }
        updateDisplayEditable(node);

        if (!node.widgets.some((w) => w.type === "button")) {
            node.addWidget("button", "⚙ 预设", null, () => openMgr(node));
            node.addWidget("button", "↧ 保存到预设", null, () => saveDraftToPreset(node));
        }

        syncFromJson(node);
        rebuildState(node);

        const originalOnAfterConfigured = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function (...args) {
            // 不调 syncFromJson：此时 _sfTpState 还是 nodeCreated 时的陈旧快照
            // （presets_json 尚未恢复时的默认值），套用会把 configure 刚恢复的
            // 选中值清掉 → rebuildState 回落第一项。选中值由 rebuildState
            // 的捕获-恢复逻辑保全，重建完成后的 sync 才是权威的。
            rebuildState(node);
            if (typeof originalOnAfterConfigured === "function") {
                return originalOnAfterConfigured.apply(this, args);
            }
        };
    },
});
