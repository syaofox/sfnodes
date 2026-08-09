// ==========================================================================
// sf_dropdown.js - SFValueDropdown 主扩展（接线）
// ==========================================================================
//
// 复刻 Pixaroma Dropdown（精简 shared 依赖版）：一个自己写的列表、一个输出，
// 值类型属于 NODE 而非它接的线。看 sf_dropdown_lib.js 了解状态，ui.mjs 看
// 节点面与输出点对齐，sf_dropdown_settings.js 看设置面板。
//
// 与原件差异（已确认范围）：无 accent 颜色设置、无 XY Plot sweep provider、
// 无帮助系统；右键菜单入口用 LiteGraph 原生 getExtraMenuOptions（any_pack.js
// 先例）而非 pixaroma 的 registerNodeSettings。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import {
    CLASS, HIDDEN_INPUT, MIN_W, DEFAULT_W, readState, writeState,
    syncOutput, injectedState, commitPick,
} from "./sf_dropdown_lib.js";
import {
    buildRow, renderRow, bodyHeight, alignOutputLegacy, scheduleAlign,
    watchAlign, unwatchAlign, closePopupFor, injectCSS, isVueNodes,
} from "./sf_dropdown_ui.js";
import { openDropdownPanel, closeDropdownPanelFor } from "./sf_dropdown_settings.js";

function openPanel(node) {
    openDropdownPanel(node, (n) => {
        syncOutput(n);
        renderRow(n);
        n.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    });
}

app.registerExtension({
    name: "sfnodes.ValueDropdown",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== CLASS) return;
        // 没有这个，热重载重复注册会把每个钩子双包装。
        if (nodeType.prototype._sfDropdownPatched) return;
        nodeType.prototype._sfDropdownPatched = true;

        injectCSS();

        // ── 创建 ─────────────────────────────────────────────────────
        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);

            // 把行钉在 body 顶部，在任何测量之前。没有它 _arrangeWidgets 把
            // widget 放在量得的槽界之下——而输出点停在行上，槽界于是依赖
            // widget.y，widget.y 依赖槽界。节点每帧长高。这是 litegraph 给
            // 自定义槽布局自己的字段，且不序列化。
            this.widgets_start_y = 2;

            buildRow(this, openPanel);
            syncOutput(this);
            renderRow(this);

            // Legacy 为每个输出预留 20px 槽行；我们的点住在行上，尺寸归我们
            // 管。MIN_W 与 NEVER this.size[0]：computeSize()[0] 也是拖拽最小
            // 值，返回活宽度会让下限每次加宽都垫高，节点从此只能长。
            if (!isVueNodes()) {
                this.computeSize = function () { return [MIN_W, bodyHeight()]; };
            }

            // 新尺寸，同步。configure() 在 onNodeCreated 之后立即运行并恢复
            // 已保存尺寸，这里延迟写会覆盖用户每次重载与每次复制的尺寸
            // （约定 #9）。
            if (!Array.isArray(this.size)) this.size = [DEFAULT_W, 60];
            this.size[0] = DEFAULT_W;
            this.size[1] = bodyHeight() + (isVueNodes() ? 52 : 0);

            queueMicrotask(() => {
                renderRow(this);
                syncOutput(this);
                watchAlign(this);
                scheduleAlign(this);
            });
        };

        // ── 加载 ─────────────────────────────────────────────────────
        const _configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = _configure?.apply(this, arguments);
            this.widgets_start_y = 2;
            // 只动 DOM + 槽。这里绝不允许写 node.size 或增删槽位，否则一个
            // 未触碰的工作流打开即标 "modified"（Vue Compat #18）。
            syncOutput(this);
            renderRow(this);
            queueMicrotask(() => {
                renderRow(this);
                watchAlign(this);
                scheduleAlign(this);
            });
            return r;
        };

        // ── Legacy：把点停在行上 ────────────────────────────────────
        // arrange() 计算 widget.y，定位点需要它；第二遍用已就位的位置重新
        // 测量槽。
        const _arrange = nodeType.prototype.arrange;
        nodeType.prototype.arrange = function () {
            const r = _arrange?.apply(this, arguments);
            if (!isVueNodes()) {
                alignOutputLegacy(this);
                _arrange?.apply(this, arguments);
            }
            return r;
        };

        // ── 几何不进已保存文件 ──────────────────────────────────────
        // Legacy 把 output.pos 写进工作流。它在 Nodes 2.0 毫无意义，所以一
        // 个渲染器保存的文件与另一个不同，干净的工作流打开即 "modified"。
        // 每次 arrange 都重建，剥掉它什么都不丢。
        const _serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const out = _serialize?.apply(this, arguments);
            try {
                for (const o of out?.outputs || []) delete o.pos;
            } catch {}
            return out;
        };

        // ── 尺寸钳制（Classic only）──────────────────────────────────
        // Nodes 2.0 中渲染尺寸活在 Vue 布局 store 而非 node.size，在那里钳
        // node.size 会失步：节点按拖拽尺寸渲染而 node.size 持有钳后值，切换
        // 工作流后按错误尺寸重建。
        const _resize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (!isVueNodes()) {
                if (size[0] < MIN_W) size[0] = MIN_W;
                size[1] = bodyHeight();   // 一行：高度归我们，不归拖拽
            }
            return _resize?.apply(this, arguments);
        };

        const _draw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            // 加载门要紧：draw 钩子在工作流加载的第一帧运行，早于任何其他
            // 钳制，所以未加门的写入是唯一能在干净打开时重写已保存 node.size
            // 的地方。
            if (!isVueNodes() && this.size[0] < MIN_W) this.size[0] = MIN_W;
            return _draw?.apply(this, arguments);
        };

        // ── 右键菜单入口（LiteGraph 原生，Classic/Vue 均支持）────────
        const _extra = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = _extra?.apply(this, arguments) || options || [];
            const node = this;
            r.push(null, {
                content: "⚙ Dropdown settings",
                callback: () => openPanel(node),
            });
            return r;
        };

        // ── 移除 ─────────────────────────────────────────────────────
        const _removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            unwatchAlign(this);
            closePopupFor(this);
            closeDropdownPanelFor(this);
            return _removed?.apply(this, arguments);
        };
    },
});

// ── graphToPrompt：注入选中的值 ─────────────────────────────────────────
// 只注入——绝不在这里剪（Export (API) 序列化同一个输出）。
function buildIndex() {
    const index = new Map();
    const seen = new Set();
    const visit = (graph) => {
        if (!graph || seen.has(graph)) return;   // 子图引用循环会栈溢出
        seen.add(graph);
        const nodes = graph._nodes || graph.nodes || [];
        for (const n of nodes) {
            if (!n) continue;
            if (n.comfyClass === CLASS || n.type === CLASS) index.set(String(n.id), n);
            const inner = n.subgraph || n.graph || n._graph;
            if (inner) visit(inner);
        }
    };
    visit(app.graph);
    return index;
}

function findNode(index, id) {
    const s = String(id);
    if (index.has(s)) return index.get(s);
    // 子图内的节点以复合 id（如 "5:12"）到达。
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
                const node = findNode(index, id);
                if (!node) continue;
                entry.inputs = entry.inputs || {};
                entry.inputs[HIDDEN_INPUT] = JSON.stringify(injectedState(node));
            }
        }
    } catch (e) {
        console.error("[SF Value Dropdown] inject failed", e);
    }
    return result;
};

// ── 花掉本次 run 的牌，只在 queue 真正被接受时 ─────────────────────────
// graphToPrompt 也跑在 Export、分享工作流、若干保存按钮、以及随后校验失败的
// queue 上。它们都不该推进 In-order 列表，所以牌 HOLD 到这次触发，在那之前
// 一直发同一条。
if (!app._sfDropdownQueuePatched && api && typeof api.queuePrompt === "function") {
    app._sfDropdownQueuePatched = true;
    const _origQueuePrompt = api.queuePrompt.bind(api);
    api.queuePrompt = async function (...args) {
        const res = await _origQueuePrompt(...args);   // 被拒的 queue 抛错 -> 牌保留
        try {
            const index = buildIndex();
            for (const node of index.values()) {
                commitPick(node);
                // 显示实际跑的。只动 DOM——绝不写序列化状态，否则每次 Run
                // 都会把工作流标 "modified"。
                renderRow(node);
            }
            app.graph?.setDirtyCanvas?.(true, false);
        } catch (err) {
            console.error("[SF Value Dropdown] commit failed", err);
        }
        return res;
    };
}
