// ==========================================================================
// SF LoRA Stack - 主扩展。一个 DOM widget（Add / All / gear + 每行一个 LoRA），
// 固定 MODEL + CLIP 输入和 MODEL + CLIP + triggers 输出。Classic 与 Nodes 2.0
// 双渲染器可用。
//
// 架构镜像 Sizes / Resolution：状态在 node.properties.loraStackState，由下面
// graphToPrompt 钩子注入隐藏 LoraLoaderState 输入（Vue Compat #9）。信息面板、
// 齿轮面板、下拉和行菜单在姊妹模块。
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    applyAdaptiveCanvasOnly,
    applySfAccentVar,
    installCanvasZoomPassthrough,
    isGraphLoading,
    isVueNodes,
    LORA_DISPLAY_MODES,
    LORA_DISPLAY_SETTING,
} from "./sf_common.js";
import { listLoras, invalidateList, invalidateAllInfo } from "./sf_lora_stack_api.js";
import {
    HIDDEN_INPUT, DEFAULT_STATE,
    readState, loadDefaults, promptState,
} from "./sf_lora_stack_core.js";
import { injectCSS, renderNode, contentHeight, repaintAll } from "./sf_lora_stack_render.js";
import { attachInteractions, loadPresetInto, watchPresetUpstream } from "./sf_lora_stack_interaction.js";
import { openLoraPanel, closeLoraPanelFor } from "./sf_lora_stack_settings.js";
import { closeInfoPanelFor } from "./sf_lora_stack_info.js";
import { closeLoraDropdown } from "./sf_lora_stack_dropdown.js";
import { closeRowMenu } from "./sf_lora_stack_interaction.js";

const CLASS = "SFLoraStack";

const MIN_W = 300;
const CHROME = 66;      // legacy 回退：标题 + 输入槽 + 输出槽行（输入槽数不影响 widget 顶）
const VUE_CHROME = 96;  // Nodes 2.0 回退（槽带绝对定位，不占 widget 区高度）

// Python hidden 输入（LoraLoaderState）。多数环境不建 widget，此函数防御。
// Nodes 2.0 下 hidden + computeSize 单独不足以抑制 Vue 节点体里的 STRING
// widget——它会渲染成显示原始 JSON 的 textarea（Note 上见过）；canvasOnly
// 把它排除出 Vue 体（shouldRenderAsVue = !canvasOnly）并排除出 legacy
// Parameters 标签页。这是内部序列化 widget，两个渲染器都必须保持隐藏。
export function hideJsonWidget(node) {
    const w = node.widgets?.find((x) => x.name === HIDDEN_INPUT);
    if (!w) return;
    w.hidden = true;
    w.computeSize = () => [0, -4];
    if (!w.options) w.options = {};
    w.options.canvasOnly = true;
    // 优先现代 widget.element；旧构建只有 widget.inputEl 时回退。
    const hideEl = () => { const el = w.element || w.inputEl; if (el) el.style.display = "none"; };
    hideEl();
    requestAnimationFrame(hideEl);
}

function widgetH(node) { return contentHeight(readState(node)); }

// 无滚动条显示所有行所需的节点高度。chrome（标题 + 输入/输出槽行）委托给
// LiteGraph 的 computeSize；不可用时才退回常量估算。
function fitNodeH(node) {
    try {
        const cs = node.computeSize?.();
        if (cs && cs[1] > 0) return Math.round(cs[1]);
    } catch (_e) { /* 走回退 */ }
    return widgetH(node) + (isVueNodes() ? VUE_CHROME : CHROME);
}

// 节点高度贴合内容。仅用户动作（加载路径绝不执行，否则保存的尺寸被改写、
// 干净的工作流打开即 "modified"——Vue Compat #18）。宽度保留手调值。
function fitToContent(node) {
    if (isGraphLoading()) return;
    const w = Math.max(node.size?.[0] || MIN_W, MIN_W);
    const h = fitNodeH(node);
    if (node.setSize) node.setSize([w, h]);
    else node.size = [w, h];
}

function makeRefresh(node) {
    return (structural) => {
        renderNode(node);
        if (structural) fitToContent(node);
        node.setDirtyCanvas?.(true, true);
    };
}

function setupNode(node) {
    hideJsonWidget(node);

    const root = document.createElement("div");
    root.className = "sf-ls-root";
    const inner = document.createElement("div");
    inner.className = "sf-ls-inner";
    root.appendChild(inner);

    const widget = node.addDOMWidget("loras_ui", "sf_lora_stack", root, {
        getValue: () => readState(node),
        setValue: () => {},
        getMinHeight: () => widgetH(node),
        getMaxHeight: () => widgetH(node),
        margin: 4,
        serialize: false,
    });
    widget.computeLayoutSize = () => ({ minHeight: widgetH(node), minWidth: 1 });
    applyAdaptiveCanvasOnly(widget);
    // LoRA 列表上的滚轮仍要缩放画布（Classic；Nodes 2.0 无操作）。chips 列表
    // 保持自己的滚动——助手对仍有滚动余地的可滚区域让位。
    installCanvasZoomPassthrough(root);

    node._sfLsRoot = root;
    node._sfLsInner = inner;

    // 新节点的默认尺寸（configure() 会为加载的节点覆盖它，Vue Compat #8）。
    // 原地改而非替换数组（Vue 可能持有响应式代理）。
    if (!Array.isArray(node.size)) node.size = [336, 0];
    node.size[0] = Math.max(node.size[0] || 0, 336);
    node.size[1] = fitNodeH(node);

    attachInteractions(node, widget.element || root, makeRefresh(node));
    // 主扩展生命周期（onConnectionsChange / onAfterGraphConfigured）需要
    // refresh——原型方法里没有闭包可拿，存在节点上。
    node._sfLsRefresh = makeRefresh(node);

    // 首次渲染推迟到 configure() 之后，让恢复的工作流渲染已存行而非默认
    // （Vue Compat #8）。fitToContent 在加载路径上让位。
    queueMicrotask(() => { renderNode(node); fitToContent(node); });

    // 预热列表让 missing 标记无需打开选择器即可显示（磁盘上改名的工作流
    // 一加载就该说）。首个节点后缓存；重绘纯 DOM，不会弄脏刚加载的工作流
    // （Vue Compat #18）。
    listLoras().then(() => { if (node._sfLsRoot) renderNode(node); });
}

app.registerExtension({
    name: "sfnodes.LoraStack",

    // 全局强调色设置（sfnodes.Accent）：所有具备 accent 能力的 sf 节点统一
    // 读取（优先级链见 core.js accentOf；CSS 主题色走 document 根 --sf-acc
    // 变量）。ComfyUI Settings 页修改后：更新 CSS 变量（全 sf 节点响应式
    // 生效）+ 重渲染图中全部 SFLoraStack 节点（含子图）。
    init() {
        try {
            app.ui.settings.addSetting({
                id: "sfnodes.Accent",
                name: "SF nodes: default highlight colour (nodes without their own colour)",
                defaultValue: "#f66744",
                type: "combo",
                options: () => {
                    const cur = app.ui.settings.getSettingValue("sfnodes.Accent");
                    // 选项文本带颜色 emoji 图例（ComfyUI 设置 combo 只渲染
                    // 文字，无原生色块支持——emoji 是零风险的近似图例）。
                    const opts = [
                        { value: "#f66744", text: "🟠 Default (brand orange)", selected: cur === "#f66744" },
                        { value: "#4f7cff", text: "🔵 Blue", selected: cur === "#4f7cff" },
                        { value: "#3ec371", text: "🟢 Green", selected: cur === "#3ec371" },
                        { value: "#e9a53d", text: "🟡 Amber", selected: cur === "#e9a53d" },
                        { value: "#e2504a", text: "🔴 Red", selected: cur === "#e2504a" },
                        { value: "#a06ee0", text: "🟣 Purple", selected: cur === "#a06ee0" },
                        { value: "#3aa0b0", text: "🔷 Teal", selected: cur === "#3aa0b0" },
                        { value: "#e8e8e8", text: "⚪ Light grey", selected: cur === "#e8e8e8" },
                    ];
                    return opts;
                },
                onChange: (value) => {
                    // 注意：ComfyUI 在 store 更新前调用 onChange，且回调参数
                    // 是 (newValue, oldValue)——必须用传入的 value，此刻读
                    // getSettingValue 会拿到旧值（"设了 red 显示 teal"）。
                    applySfAccentVar(value);
                    // repaintAll 必须推迟到 store 更新后（applySettingLocally
                    // 在 onChange 返回后才写 e.value）：同步执行时 accentOf
                    // 读到的还是旧值，SFLoraStack 颜色不立即生效。
                    setTimeout(repaintAll, 0);
                },
            });
            // Civitai 批量下载原图样例开关（默认关，不压缩）
            try {
                app.ui.settings.addSetting({
                    id: "sfnodes.Civitai.DownloadSamples",
                    name: "SF: Civitai - download all sample images (original) to sample folder when fetching",
                    type: "boolean",
                    defaultValue: false,
                });
            } catch (_e2) { /* 设置系统不可用则忽略 */ }
            // 全局 LoRA 显示名设置（已从 power_lora_loader 彻底迁移至此，
            // 新键 sfnodes.Lora.DisplayName，旧键一次性搬运见 sf_common.js）
            try {
                const displayOptions = {
                    "Full path": LORA_DISPLAY_MODES.FULL,
                    "File name": LORA_DISPLAY_MODES.FILENAME,
                    "File name without extension": LORA_DISPLAY_MODES.BASENAME,
                    "Parent folder name": LORA_DISPLAY_MODES.FOLDER,
                    "Parent folder + name without ext": LORA_DISPLAY_MODES.PARENT_BASENAME,
                };
                app.ui.settings.addSetting({
                    id: LORA_DISPLAY_SETTING,
                    name: "SF LoRA: display name (full path / file name / no extension / parent folder / folder + name)",
                    defaultValue: LORA_DISPLAY_MODES.FULL,
                    type: "combo",
                    options: () => Object.entries(displayOptions).map(([text, value]) => ({
                        value, text, selected: app.ui.settings.getSettingValue(LORA_DISPLAY_SETTING) === value,
                    })),
                    onChange: () => {
                        app.graph.setDirtyCanvas(true);
                        document.dispatchEvent(new CustomEvent("sfnodes.lora-display-mode-changed"));
                    },
                });
                // 旧键一次性迁移（彻底迁移语义：读旧值后写入新键）
                try {
                    const legacy = app.ui.settings.getSettingValue("sfnodes.PowerLoraLoader.DisplayName");
                    const cur = app.ui.settings.getSettingValue(LORA_DISPLAY_SETTING);
                    if (legacy && !cur) {
                        app.ui.settings.setSettingValue(LORA_DISPLAY_SETTING, legacy);
                    }
                } catch {}
            } catch (_e3) { /* 设置系统不可用则忽略 */ }
            // 全局 LoRA 显示名设置变化时经事件桥通知——Stack/Plot
            // 行名随设置重渲染（DOM 重绘，setDirtyCanvas 管不到 widget DOM）。
            // setTimeout(0) 推迟到设置 store 更新后（同 Accent 时序教训）。
            document.addEventListener("sfnodes.lora-display-mode-changed", () => {
                setTimeout(repaintAll, 0);
            });
            // 任一节点保存 LoRA 用户数据（触发词/描述/封面）后广播——行 i 按钮
            // 的 _has_custom 高亮要跟着刷新。loraMetadataCache 已被
            // sf_lora_info.js 的同事件监听清掉，重渲染时每行重新查询得新值。
            document.addEventListener("sfnodes.lora-data-changed", () => {
                setTimeout(repaintAll, 0);
            });
            // 初始应用必须在 addSetting 之后：设置项未注册时 getSettingValue
            // 拿不到用户保存的值（返回默认 #f66744）——先 apply 会把 --sf-acc
            // 钉死在橙色，且注册后不再刷新（"Crop 品牌文字还是橙色"）。
            applySfAccentVar();
            // 设置值从服务器加载是异步的（comfy.settings.json 拉取可能晚于
            // 扩展 init）——轮询重试直到加载完成，否则 --sf-acc 会被钉死在
            // 默认色、CSS 变量类节点（Load Image Resize 等）不跟随（"硬刷新
            // 后其他节点恢复橙色"）。幂等且廉价；用户改设置由 onChange 处理。
            let tries = 12;
            const retryApply = () => {
                if (tries-- <= 0) return;
                setTimeout(() => {
                    applySfAccentVar();
                    retryApply();
                }, 500);
            };
            retryApply();
        } catch (_e) { /* 设置系统不可用则退化为仅节点级 accent */ }
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CLASS) return;
        if (nodeType.prototype._sfLsPatched) return;
        nodeType.prototype._sfLsPatched = true;

        injectCSS();

        // ComfyUI 按 R 时会对每个图节点调 node.refreshComboInNode(defs)——
        // 把它接进缓存失效与重渲染（power_lora_loader 同款先例）。
        const _origRefresh = nodeType.prototype.refreshComboInNode;
        nodeType.prototype.refreshComboInNode = function () {
            invalidateList();
            invalidateAllInfo();
            listLoras().then(() => {
                // 递归进子图：子图里嵌套的 LoRA Stack 的 missing 标记也要重画，
                // 不只是顶层。
                const walk = (g) => {
                    for (const n of (g?._nodes || [])) {
                        if ((n.comfyClass === CLASS || n.type === CLASS) && n._sfLsRoot) renderNode(n);
                        const sub = n.subgraph || n.graph || n._graph;
                        if (sub && sub !== g) walk(sub);
                    }
                };
                walk(app.graph);
            });
            if (_origRefresh) return _origRefresh.apply(this, arguments);
        };

        const _origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = _origConfigure?.apply(this, arguments);
            if (this._sfLsRoot) { renderNode(this); fitToContent(this); }
            return r;
        };

        const _origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            // Legacy ONLY：Nodes 2.0 的渲染尺寸活在 Vue 布局 store，
            // getMinHeight/computeLayoutSize 已锁定高度——这里钳 node.size
            // 会失步并在切换工作流标签时弹跳（Nodes 2.0 resize 规则）。
            if (!isVueNodes()) {
                if (this.size[0] < MIN_W) this.size[0] = MIN_W;
                this.size[1] = fitNodeH(this);
            }
            if (_origResize) return _origResize.call(this, size);
        };

        // 同一钳制的保险带（节点 UI 约定 #7）：onResize 不覆盖每条 legacy
        // resize 路径，先增后减的循环可能让节点停在 MIN_W 之下、行控件被
        // 右缘裁掉。Legacy only，理由同 onResize 上写的。
        const _origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            // 必须也用 isGraphLoading() 门控：这是唯一能在加载路径运行的宽度
            // 钳制（onConfigure 的 fitToContent 加载中已让位），而 node.size
            // 会被序列化——Nodes 2.0 保存的窄节点在 Classic 打开会在第一帧
            // 被改写，把未触碰的工作流标成 "modified"（Vue Compat #18）。
            if (!isVueNodes() && !isGraphLoading() && this.size[0] < MIN_W) this.size[0] = MIN_W;
            return _origDrawFg?.apply(this, arguments);
        };

        const _origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            closeLoraPanelFor(this);
            closeInfoPanelFor(this);
            closeLoraDropdown(); // 瞬态——删除节点的画布点击也会自动关
            closeRowMenu();
            return _origRemoved?.apply(this, arguments);
        };

        // ── preset 输入（SF_LORA_PRESET）：连接即加载配置、刷新列表 ──
        // 连接是用户交互（非加载路径）——直接加载（writeState 刷新 UI）。
        // 加载路径恢复连接（configure 直赋 links 不触发 onConnectionsChange，
        // 见 Vue Compat #18）由 onAfterGraphConfigured 补 watch，刻意不写
        // 状态：预设行已随工作流序列化，重写会把干净文件标成 modified。
        const _origConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const r = _origConn ? _origConn.apply(this, args) : undefined;
            if (this.comfyClass !== CLASS) return r;
            const [slot_type, , is_connected, , input] = args;
            if (slot_type !== LiteGraph.INPUT || input?.name !== "preset") return r;
            const refresh = this._sfLsRefresh;
            watchPresetUpstream(this, refresh);
            if (is_connected) loadPresetInto(this, refresh);
            return r;
        };

        const _origGraphCfg = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function (...args) {
            const r = _origGraphCfg ? _origGraphCfg.apply(this, args) : undefined;
            // 连接恢复发生在 onConfigure 之后（Vue Compat #18）——这里看
            // 到的连接是最终的。只 watch（上游 combo 变化仍跟随），不加载。
            if (this.comfyClass === CLASS) watchPresetUpstream(this, this._sfLsRefresh);
            return r;
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== CLASS) return;
        setupNode(node);
    },

    getNodeMenuItems(node) {
        if (node?.comfyClass !== CLASS) return [];
        return [
            { content: "⚙ LoRA Stack settings", callback: () => openLoraPanel(node, makeRefresh(node)) },
        ];
    },
});

// ── graphToPrompt：注入每节点状态（只注入，从不剪枝）────────────────────────
// buildIndex/findNode 参数化导出：SFLoraPlot 的注入钩子复用同一实现
// （AGENTS.md 规则 14——不内联副本）。classNames：类名数组或单个类名。
export function buildIndex(classNames) {
    const index = new Map();
    const visit = (graph, prefix) => {
        if (!graph) return;
        const names = Array.isArray(classNames) ? classNames : [classNames];
        for (const n of graph._nodes || graph.nodes || []) {
            if (!n) continue;
            // 复合 id（顶层 ""，子图内 "5:" 风格），让子图节点精确匹配它的
            // "5:3" prompt id，且不与碰巧共享裸 id 的顶层节点冲突。
            const cid = String(prefix) + n.id;
            if (names.includes(n.comfyClass) || names.includes(n.type)) {
                index.set(cid, n);
                // 裸 id，first-write-wins（顶层先访问），子图节点不覆盖顶层
                // 节点的精确 id 解析。
                if (!index.has(String(n.id))) index.set(String(n.id), n);
            }
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== graph) visit(inner, cid + ":");
        }
    };
    visit(app.graph, "");
    return index;
}
export function findNode(index, id) {
    const s = String(id);
    if (index.has(s)) return index.get(s);
    const tail = s.includes(":") ? s.slice(s.lastIndexOf(":") + 1) : null;
    return tail && index.has(tail) ? index.get(tail) : null;
}

const _origGraphToPrompt = app.graphToPrompt.bind(app);
app.graphToPrompt = async function (...args) {
    const result = await _origGraphToPrompt(...args);
    try {
        const out = result?.output;
        if (out) {
            let index = null;
            for (const id in out) {
                const entry = out[id];
                if (!entry || entry.class_type !== CLASS) continue;
                if (!index) index = buildIndex([CLASS]);
                entry.inputs = entry.inputs || {};
                const node = findNode(index, id);
                const st = node ? readState(node) : { ...DEFAULT_STATE, ...loadDefaults(), loras: [] };
                entry.inputs[HIDDEN_INPUT] = JSON.stringify(promptState(st));
            }
        }
    } catch (e) {
        console.warn("[SF LoRA Stack] could not inject state:", (e && e.message) || e);
    }
    return result;
};
