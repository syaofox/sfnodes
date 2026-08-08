// ==========================================================================
// sf_find_replace.js - SFTextFindReplace 主扩展
// ==========================================================================
//
// 复刻 Pixaroma Find & Replace（精简 shared 依赖版，与 sf_pause_text 同模式）：
//   - 节点体：全局开关 pill + 规则行（拖拽排序/ON-OFF/find→replace/删除）+
//     Add/Reset + 实时前后对比预览（web/sf_find_replace_ui.js）
//   - 状态存 node.properties.findReplaceState（随工作流保存，
//     见 web/sf_find_replace_lib.js）
//   - 注入走 Pattern #9：app.graphToPrompt hook 把规则状态（不含预览）打包进
//     隐藏 FindReplaceState 输入——它在缓存键里，规则变化自动失效下游
//   - executed 事件接收 Python 的 ui 预览样本（sf_find_replace 键）回填，
//     预览 = 上次运行输入 × 当前规则实时重算
//   - 子图安全：按复合路径 id 递归索引（"5:12"），子图内的节点也能注入
//
// 与原件差异（已确认范围）：无 accent 颜色设置（固定强调色）、无注册帮助面板
// 系统、无 resize floor / canvas zoom 穿透辅助；全局钩子加 guard（与 Pixaroma
// 共存时各自包装一次，链式组合安全）。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import {
    readState,
    restoreFromProperties,
    addRule,
    deleteRule,
    toggleRuleEnabled,
    setToggle,
    reorderRules,
    resetToDefault,
    setPreviewInput,
    STATE_PROP,
} from "./sf_find_replace_lib.js";
import {
    injectCSS,
    buildRoot,
    renderAll,
    renderPreview,
    refreshResetState,
    measureMinHeight,
    autoGrowAllFields,
    sfConfirm,
} from "./sf_find_replace_ui.js";

const CLASS = "SFTextFindReplace";
const STATE_INPUT = "FindReplaceState";
const WIDGET_TYPE = "sf_find_replace_ui";

const DEFAULT_W = 380;
const DEFAULT_H = 320;
const MIN_W = 340;
const MIN_H = 200;

// node.size 是整节点（标题栏 + 文本进/出槽行 + DOM widget）。measureMinHeight()
// 只返回 WIDGET 内容高度，所以给节点定尺寸时要加上 CHROME（标题 + 一行槽）。
// Legacy ComfyUI 通过 widget 的 getMinHeight 自纠正，但 Nodes 2.0 严格按
// node.size 执行——没有这层，添加规则时节点体会溢出到边框下方。
const CHROME = 60;

// 通过 setSize() 提交节点高度使其在两个渲染器里都生效（裸写 node.size[1] = h
// 可被 Nodes 2.0 的响应式布局在另一渲染器最后定尺寸时回退）。保留直接写作为
// 无 setSize 构建的回退。
function setNodeHeight(node, h) {
    node.size[1] = h;
    node.setSize?.([node.size[0], h]);
    node._sfFrAutoH = h; // 记住我们设定的高度，以区分自动适配与手动拖拽
}

// 让节点增高以容纳固定部分（开关 + 规则 + 操作）加最小预览。只增高，且只在
// 用户操作（添加/删除/重置/执行）时调用——加载路径永不调用。压缩地板纯由 CSS
// 处理（root 的自然 min-content 高度，见 ui 文件）。双向：内容增高则长高；
// 用户没有手动把节点拉高过（超出最近一次自动高度）时内容变矮则缩回——拖高
// 看大预览的人保留其尺寸，但清空多行字段或删除行会收回空间而非留下死区。
// 下限 DEFAULT_H。只从用户操作 handler 调用（add/delete/edit/drop）——加载路径
// 调用会重写 node.size 并误标工作流已修改。始终包含 CHROME，使 Nodes 2.0 中
// 边框包含 widget。
function refitNode(node) {
    const root = node._sfFrRoot;
    if (!root) return;
    const want = Math.max(measureMinHeight(root) + CHROME, DEFAULT_H);
    const cur = node.size[1];
    const autoH = node._sfFrAutoH;
    const userEnlarged = autoH != null && cur > autoH + 4;
    let target = cur;
    if (want > cur) target = want;                       // 总是增高以适配
    else if (!userEnlarged && want < cur) target = want; // 未手动拉高则缩回内容
    if (target !== cur) setNodeHeight(node, target);
}

// 强制节点回到舒适的默认高度（仅 Reset 时用——刻意丢弃任何手动拉高）。
function fitToDefault(node) {
    const root = node._sfFrRoot;
    if (!root) return;
    setNodeHeight(node, Math.max(measureMinHeight(root) + CHROME, DEFAULT_H));
}

function makeHandlers(node, root) {
    const rerender = () => {
        renderAll(node, root, handlers);
        requestAnimationFrame(() => {
            refitNode(node);
            node.setDirtyCanvas(true, true);
        });
    };
    const handlers = {
        onToggleGlobal: (key) => { setToggle(node, key); rerender(); },
        onToggleRule: (id) => { toggleRuleEnabled(node, id); rerender(); },
        onAdd: () => { addRule(node); rerender(); },
        // 即时删除——无确认（重建一条规则很便宜，实时预览也会显示效果）。
        // 只剩一条时删除按钮禁用，所以删除后至少还剩一条。Reset（清空一切）
        // 仍要确认。
        onDelete: (id) => {
            deleteRule(node, id);
            rerender();
        },
        onReset: async () => {
            const ok = await sfConfirm({
                title: "重置所有规则？",
                message: "这会清除所有规则并把开关恢复默认（Case 关、Whole word 关、Regex 关、Tidy 开）。",
                okText: "重置",
                cancelText: "取消",
            });
            if (!ok) return;
            resetToDefault(node);
            rerender();
            requestAnimationFrame(() => {
                fitToDefault(node);
                node.setDirtyCanvas(true, true);
            });
        },
        onDrop: (fromId, toId, above) => {
            const state = readState(node);
            const fromIdx = state.rules.findIndex((r) => r.id === fromId);
            const toIdxRaw = state.rules.findIndex((r) => r.id === toId);
            if (fromIdx < 0 || toIdxRaw < 0) return;
            let destIdx = above ? toIdxRaw : toIdxRaw + 1;
            if (fromIdx < destIdx) destIdx -= 1;
            if (destIdx === fromIdx) return;
            reorderRules(node, fromIdx, destIdx);
            rerender();
        },
    };
    return { handlers, rerender };
}

// ── Nodes 2.0（Vue）渲染器辅助（内联，见 sf_pause_text.js）─────────────
// 由设置 Comfy.VueNodes.Enabled 驱动；实时读取，运行时切换渲染器也尊重
function isVueNodes() {
    return !!window.LiteGraph?.vueNodesMode;
}
// adaptive canvasOnly：legacy 下 true（不进 Parameters tab），Nodes 2.0 下 false
// （否则 Vue 根本不渲染该 widget）。实时 getter，渲染时求值
function applyAdaptiveCanvasOnly(widget) {
    if (!widget || !widget.options) return widget;
    try {
        Object.defineProperty(widget.options, "canvasOnly", {
            configurable: true,
            enumerable: true,
            get() {
                return !isVueNodes();
            },
        });
    } catch { /* ignore */ }
    return widget;
}

app.registerExtension({
    name: "sfnodes.FindReplace",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CLASS) return;

        const origNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origNodeCreated) origNodeCreated.apply(this, arguments);
            const node = this;
            queueMicrotask(() => {
                injectCSS();
                restoreFromProperties(node);
                // 持有状态的一个私有深拷贝，使粘贴/克隆的节点不会按引用共享
                // 原节点的 rules 数组（否则编辑副本会改动原件）。克隆逐字节
                // 相同，因此不会把已加载工作流标脏。
                const _st = node.properties?.[STATE_PROP];
                if (_st) {
                    try { node.properties[STATE_PROP] = JSON.parse(JSON.stringify(_st)); } catch (_e) {}
                }

                const root = buildRoot();
                const { handlers, rerender } = makeHandlers(node, root);
                node._sfFrRoot = root;
                node._sfFrRerender = rerender;
                // 仅 DOM 渲染（不自动增高）——加载时使用，使保存的 node.size
                // 被信任、工作流不会被误标 "modified"。
                node._sfFrRenderOnly = () => renderAll(node, root, handlers);
                node._sfFrRefreshPreview = () => renderPreview(node, root);
                node._sfFrRefreshReset = () => refreshResetState(node, root);
                node._sfFrRefit = () => { refitNode(node); node.setDirtyCanvas(true, true); };

                const widget = node.addDOMWidget(WIDGET_TYPE, WIDGET_TYPE, root, {
                    serialize: false,
                    getMinHeight: () => measureMinHeight(root),
                });
                applyAdaptiveCanvasOnly(widget);
                // Nodes 2.0 通过 CSS grid 经 computeLayoutSize 尺寸化 widget，
                // 忽略上面 legacy 的 getMinHeight——没有它节点可被拖小到内容
                // 之下、节点体溢出到边框下方。给同样的固定部分 + 最小预览地板
                // （添加规则时增高）。minWidth:1 让保存的节点宽度在重载时仍能
                // 往返。
                widget.computeLayoutSize = () => ({ minHeight: measureMinHeight(root), minWidth: 1 });

                node._sfFrRenderOnly();

                // 节点宽度变化时重新测量 find/replace 字段，使窄宽度下换行增长
                // 的字段在节点加宽时缩回。宽度门控避免高度反馈回路。
                try {
                    let lastW = root.clientWidth;
                    const ro = new ResizeObserver(() => {
                        const w = root.clientWidth;
                        if (w !== lastW) { lastW = w; autoGrowAllFields(root); }
                    });
                    ro.observe(root);
                    node._sfFrFieldRO = ro;
                } catch (_e) {}

                // 仅全新放置时以舒适的默认尺寸打开。onConfigure 对加载的工作流
                // 设置 _sfFrConfigured（它在微任务前运行），所以已保存的尺寸——
                // 即使被用户缩到 DEFAULT 以下——原样保留，不会在切换工作流时
                // 跳回默认。
                if (!node._sfFrConfigured) {
                    const w = Math.max(node.size[0], DEFAULT_W);
                    const h = Math.max(node.size[1], DEFAULT_H);
                    if (w !== node.size[0] || h !== node.size[1]) {
                        node.size[0] = w;
                        node.size[1] = h;
                        node.setSize?.([w, h]);
                    }
                }
                node.setDirtyCanvas(true, true);
            });
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            // 标记此节点为从工作流加载，使 onNodeCreated 微任务保留已保存尺寸
            // 而非强制 DEFAULT_H。仅 DOM 渲染，不写尺寸——加载路径绝不触碰
            // node.size。
            this._sfFrConfigured = true;
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            restoreFromProperties(this);
            if (this._sfFrRenderOnly) this._sfFrRenderOnly();
            return r;
        };

        // 捕获执行过的输入/输出使实时预览能显示前后对比（并持久化，使分享的
        // 工作流打开即见）。预览从当前规则重算输出，所以这里只需要输入。
        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origExecuted) origExecuted.apply(this, arguments);
            try {
                const data = message?.sf_find_replace?.[0];
                if (data && typeof data.input === "string") {
                    setPreviewInput(this, data.input, !!data.truncated);
                    if (this._sfFrRefreshPreview) this._sfFrRefreshPreview();
                    // 不要在这里调整尺寸。Run 从不改变规则数，所以高度地板不变，
                    // 预览（flex 区域）吸收任何余量。每次 Run 调整尺寸会重写
                    // node.size，把普通 Run 误标为 "modified"。只重绘预览。
                    this.setDirtyCanvas(true, true);
                }
            } catch (err) {
                console.error("[sfnodes] Find Replace: onExecuted failed", err);
            }
        };

        const origOnResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            // LEGACY ONLY。Nodes 2.0 中渲染尺寸在 Vue 布局 store 里而非
            // node.size；在这里钳 node.size 会失步，且切换工作流时节点从（被
            // 钳大过的）node.size 重建并跳变到它。Nodes 2.0 通过 MIN_NODE_WIDTH
            // 与内容地板钳宽度/高度，那里不需要此钳制。
            if (!isVueNodes()) {
                if (size[0] < MIN_W) size[0] = MIN_W;
                if (size[1] < MIN_H) size[1] = MIN_H;
                if (this.size[0] < MIN_W) this.size[0] = MIN_W;
                if (this.size[1] < MIN_H) this.size[1] = MIN_H;
            }
            if (origOnResize) return origOnResize.apply(this, arguments);
        };

        const origDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDraw) origDraw.call(this, ctx);
            if (this.flags?.collapsed) return;
            if (isVueNodes()) return; // legacy-only 钳制（见 onResize）
            if (this.size[0] < MIN_W) this.size[0] = MIN_W;
            if (this.size[1] < MIN_H) this.size[1] = MIN_H;
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._sfFrFieldRO?.disconnect();
            this._sfFrFieldRO = null;
            this._sfFrRoot = null;
            this._sfFrRerender = null;
            this._sfFrRenderOnly = null;
            this._sfFrRefreshPreview = null;
            this._sfFrRefreshReset = null;
            this._sfFrRefit = null;
            if (origRemoved) return origRemoved.apply(this, arguments);
        };
    },
});

// ── graphToPrompt hook：提交时把规则状态（不含预览）注入隐藏输入（Pattern #9）──
//
// 子图安全：ComfyUI 把子图内节点扁平化进 API prompt 时使用复合 id（"5:12"），
// 而 app.graph.getNodeById 只暴露顶层节点——纯 parseInt(tail) + getNodeById 会
// 静默漏掉放在子图内的节点（规则永不注入 -> 节点近似空操作）。改为递归索引
// 每个嵌套子图。
function buildFindReplaceNodeIndex() {
    // 按复合路径 id（子图节点 5 内的节点为 "5:12"）为键，使两个各自包含 FR
    // 节点、内部 id 相同的子图（"5:12" 与 "7:12"）不会在裸尾号 "12" 上碰撞
    // （旧的 String(n.id) 键会互相覆盖、注入错误规则）。graphToPrompt 对嵌套
    // 节点的 prompt 键正是这个复合形式。
    const index = new Map(); // 复合 id（"5:12"）或裸 id -> node
    const visit = (graph, prefix) => {
        if (!graph) return;
        const nodes = graph._nodes || graph.nodes || [];
        for (const n of nodes) {
            if (!n) continue;
            const fullId = prefix + String(n.id);
            if (n.comfyClass === CLASS || n.type === CLASS) {
                index.set(fullId, n);
            }
            const inner = n.subgraph || n.graph || n._graph;
            if (inner && inner !== graph) visit(inner, fullId + ":");
        }
    };
    visit(app.graph, "");
    return index;
}

function findFindReplaceNode(index, promptId) {
    const sId = String(promptId);
    if (index.has(sId)) return index.get(sId);
    const tail = sId.includes(":") ? sId.slice(sId.lastIndexOf(":") + 1) : null;
    if (tail && index.has(tail)) return index.get(tail);
    return null;
}

if (!app._sfFindReplacePatched) {
    app._sfFindReplacePatched = true;
    const _origGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await _origGraphToPrompt(...args);
        // FAIL OPEN：这里抛错会拒绝 ComfyUI 自己的 graphToPrompt、弄坏整个
        // 工作流的 Run。绝不包住上面的 await；核心失败必须传播。
        try {
            const prompt = result?.output;
            if (prompt && typeof prompt === "object") {
                let index = null;
                for (const key of Object.keys(prompt)) {
                    const entry = prompt[key];
                    if (!entry || entry.class_type !== CLASS) continue;
                    if (!index) index = buildFindReplaceNodeIndex();
                    const node = findFindReplaceNode(index, key);
                    if (!node) continue;
                    // 经 readState（非裸 properties）读取，使注入的 payload 与
                    // 节点上预览计算的规范化方式一致——畸形/遗留的已保存状态
                    // 不能使真实运行偏离预览（如缺 `enabled` 的行经 !!r.enabled
                    // 注入为 OFF，而预览把它当 ON）。
                    const state = readState(node);
                    const payload = JSON.stringify({
                        version: 1,
                        caseSensitive: !!state.caseSensitive,
                        wholeWord: !!state.wholeWord,
                        regex: !!state.regex,
                        tidy: state.tidy !== false,
                        rules: state.rules.map((r) => ({
                            enabled: !!r.enabled,
                            find: r.find || "",
                            replace: r.replace || "",
                        })),
                    });
                    entry.inputs = entry.inputs || {};
                    entry.inputs[STATE_INPUT] = payload;
                }
            }
        } catch (err) {
            console.error("[sfnodes] Find Replace: prompt injection failed; prompt sent unchanged", err);
        }
        return result;
    };
}
