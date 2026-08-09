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
    installCanvasZoomPassthrough,
    isGraphLoading,
    isVueNodes,
} from "./sf_common.js";
import { listLoras, invalidateList, invalidateAllInfo } from "./sf_lora_stack_api.js";
import {
    HIDDEN_INPUT, DEFAULT_STATE,
    readState, loadDefaults, promptState,
} from "./sf_lora_stack_core.js";
import { injectCSS, renderNode, contentHeight } from "./sf_lora_stack_render.js";
import { attachInteractions } from "./sf_lora_stack_interaction.js";
import { openLoraPanel, closeLoraPanelFor } from "./sf_lora_stack_settings.js";
import { closeInfoPanelFor } from "./sf_lora_stack_info.js";
import { closeLoraDropdown } from "./sf_lora_stack_dropdown.js";
import { closeRowMenu } from "./sf_lora_stack_interaction.js";

const CLASS = "SFLoraStack";

const MIN_W = 300;
const CHROME = 66;      // legacy 回退：标题 + 2 输入 + 3 输出槽行
const VUE_CHROME = 96;  // Nodes 2.0 回退

// Python hidden 输入（LoraLoaderState）。多数环境不建 widget，此函数防御。
// Nodes 2.0 下 hidden + computeSize 单独不足以抑制 Vue 节点体里的 STRING
// widget——它会渲染成显示原始 JSON 的 textarea（Note 上见过）；canvasOnly
// 把它排除出 Vue 体（shouldRenderAsVue = !canvasOnly）并排除出 legacy
// Parameters 标签页。这是内部序列化 widget，两个渲染器都必须保持隐藏。
function hideJsonWidget(node) {
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
function buildIndex() {
    const index = new Map();
    const visit = (graph, prefix) => {
        if (!graph) return;
        for (const n of graph._nodes || graph.nodes || []) {
            if (!n) continue;
            // 复合 id（顶层 ""，子图内 "5:" 风格），让子图节点精确匹配它的
            // "5:3" prompt id，且不与碰巧共享裸 id 的顶层节点冲突。
            const cid = String(prefix) + n.id;
            if (n.comfyClass === CLASS || n.type === CLASS) {
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
function findNode(index, id) {
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
                if (!index) index = buildIndex();
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
