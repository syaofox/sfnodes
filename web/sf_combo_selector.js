// ==========================================================================
// SF Combo Selector - 通用下拉选择器
// 输出连接到目标节点的 combo 输入（Convert to input 后）时，下拉选项自动
// 同步为目标 combo 的选项列表；断线/无连接时恢复占位。
// ==========================================================================
import { app } from "/scripts/app.js";

const NODE_TYPE = "SFComboSelector";
const PLACEHOLDER = [""];
const DBG = false;

function dbg(...args) {
    if (DBG) console.log("[SFComboSelector]", ...args);
}

// 按 link id 取 LLink，兼容对象表（旧版）与 Map（Vue 新版）
function getLink(graph, id) {
    if (!graph) return null;
    if (graph.links?.get) return graph.links.get(id) ?? null;
    return graph.links?.[id] ?? null;
}

// 按节点 id 找节点，兼容字符串/数字 id（Vue 新版节点 id 为字符串）
function findNodeById(graph, id) {
    if (!graph) return null;
    const byApi = graph.getNodeById?.(id);
    if (byApi) return byApi;
    const nodes = graph._nodes ?? [];
    return nodes.find((n) => String(n.id) === String(id)) ?? null;
}

// 把任意槽类型表示归一化为字符串数组
function normalizeTypeToList(t) {
    if (Array.isArray(t)) {
        const arr = t.filter((x) => x !== null && x !== undefined);
        return arr.length ? arr.map(String) : null;
    }
    if (typeof t === "string" && t.trim()) {
        const s = t.trim();
        if (s.startsWith("[")) {
            try {
                const arr = JSON.parse(s);
                if (Array.isArray(arr) && arr.length) return arr.map(String);
            } catch (e) { /* 非 JSON，继续按逗号解析 */ }
        }
        if (s.includes(",")) {
            const arr = s.split(",").map((x) => x.trim()).filter(Boolean);
            if (arr.length) return arr;
        }
    }
    return null;
}

// 从目标节点提取 combo 选项列表，兼容多种来源与版本：
// 1. 输入槽 type（Convert to input 时由原 widget 的 options.values 复制）
// 2. 残留的同名 widget 的 options.values（Vue 新版 convert 后 widget 保留）
// 3. nodeDef 中该输入的静态列表（兜底）
function extractComboOptions(targetNode, inputIndex, inputName) {
    const fromSlot = normalizeTypeToList(targetNode?.inputs?.[inputIndex]?.type);
    if (fromSlot) {
        dbg("options from slot.type:", fromSlot);
        return fromSlot;
    }
    const widget = targetNode?.widgets?.find((w) => w.name === inputName);
    if (widget && Array.isArray(widget.options?.values)) {
        const arr = widget.options.values.filter((x) => x !== null && x !== undefined).map(String);
        if (arr.length) {
            dbg("options from widget:", arr);
            return arr;
        }
    }
    const def =
        app.nodeDefsByType?.[targetNode?.type] ??
        (typeof app.registeredNodes === "object" && app.registeredNodes ? app.registeredNodes[targetNode?.type] : null);
    const spec = def?.input?.required?.[inputName] ?? def?.input?.optional?.[inputName];
    const fromDef = normalizeTypeToList(spec?.[0]);
    if (fromDef) {
        dbg("options from nodeDef:", fromDef);
        return fromDef;
    }
    return null;
}

app.registerExtension({
    name: "sfnodes.SFComboSelector",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;

        const widget = node.widgets?.find((w) => w.name === "value");
        if (!widget) return;

        function setOptions(values) {
            const opts = values && values.length ? values : PLACEHOLDER;
            const has = opts.includes(widget.value);
            // 整体替换 options 对象 + values 数组引用，兼容 Vue 渲染监听
            widget.options = { ...widget.options, values: [...opts] };
            widget.options.values = [...opts];
            if (!has) widget.value = opts[0];
            dbg("setOptions:", opts, "value ->", widget.value);
            node.setDirtyCanvas(true, true);
        }

        // 遍历输出链路上的目标 combo 输入，取第一个有效选项列表
        function syncOptions() {
            const out = node.outputs?.[0];
            const links = out?.links;
            if (!links || !links.length) {
                dbg("no links, reset placeholder");
                setOptions(PLACEHOLDER);
                return;
            }
            for (const linkId of links) {
                const link = getLink(node.graph, linkId);
                if (!link) continue;
                // 兼容旧版 target_node / Vue 新版 target_id 字段
                const targetNode = findNodeById(node.graph, link.target_node ?? link.target_id);
                const inputIndex = link.target_slot;
                const inputName = targetNode?.inputs?.[inputIndex]?.name ?? "";
                dbg("link ->", link.target_node ?? link.target_id, inputIndex, inputName);
                const opts = extractComboOptions(targetNode, inputIndex, inputName);
                if (opts) {
                    setOptions(opts);
                    return;
                }
            }
            dbg("no extractable options, reset placeholder");
            setOptions(PLACEHOLDER);
        }

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, slot, connected, linkInfo) {
            if (type === LiteGraph.OUTPUT && slot === 0) {
                // 连接刚建立时 links 可能尚未更新，延迟到事件完成后再同步
                setTimeout(syncOptions, 0);
            }
            return origOnConnectionsChange?.apply(this, arguments);
        };

        node.onGraphConfigured = syncOptions;
        node.onAfterGraphConfigured = syncOptions;

        // 诊断接口（真实环境排障用）
        node._sfComboSync = syncOptions;
        node._sfComboGetLinks = () =>
            (node.outputs?.[0]?.links ?? []).map((id) => getLink(node.graph, id)).filter(Boolean);
        node._sfComboFindTarget = (link) =>
            findNodeById(node.graph, link.target_node ?? link.target_id);

        syncOptions();
    },
});
