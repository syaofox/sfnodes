import { app } from "/scripts/app.js";

// "SF Text Encode (Krea2)" 的自动增长参考插槽：每张参考图是 (imageN, maskN) 一组。
// 始终保持恰好一个空尾对，最后一个对一旦被连接就追加一组新的。
const NODE_NAMES = new Set(["SFTextEncodeKrea2"]);
const IMAGE_RE = /^image(\d+)$/;

function pairNumbers(node) {
    const nums = [];
    for (const inp of node.inputs || []) {
        const m = IMAGE_RE.exec(inp.name);
        if (m) nums.push(parseInt(m[1], 10));
    }
    nums.sort((a, b) => a - b);
    return nums;
}

function inputIndex(node, name) {
    return (node.inputs || []).findIndex((i) => i.name === name);
}

function linked(node, name) {
    const idx = inputIndex(node, name);
    return idx >= 0 && node.inputs[idx].link != null;
}

function addPair(node, n) {
    node.addInput(`image${n}`, "IMAGE");
    node.addInput(`mask${n}`, "MASK");
}

function removePair(node, n) {
    // 先删 mask（位于 image 之后，删除后 image 的索引保持有效）。
    let idx = inputIndex(node, `mask${n}`);
    if (idx >= 0) node.removeInput(idx);
    idx = inputIndex(node, `image${n}`);
    if (idx >= 0) node.removeInput(idx);
}

function pairEmpty(node, n) {
    return !linked(node, `image${n}`) && !linked(node, `mask${n}`);
}

// 自愈：保证每个 imageN 都有配套的 maskN（例如遮罩功能出现前保存的工作流，
// 或过期的 Python schema 只定义了 image1）。
function ensureMasks(node) {
    for (const n of pairNumbers(node)) {
        if (inputIndex(node, `mask${n}`) < 0) {
            node.addInput(`mask${n}`, "MASK");
        }
    }
}

function syncPairs(node) {
    if (pairNumbers(node).length === 0) addPair(node, 1);
    ensureMasks(node);

    // 把尾部连续的空对收拢到只剩一个备用对。
    for (;;) {
        const nums = pairNumbers(node);
        if (nums.length <= 1) break;
        const last = nums[nums.length - 1];
        const prev = nums[nums.length - 2];
        if (pairEmpty(node, last) && pairEmpty(node, prev)) {
            removePair(node, last);
        } else {
            break;
        }
    }

    // 最后一个对已被使用，则追加一个新的备用对。
    const nums = pairNumbers(node);
    const last = nums[nums.length - 1];
    if (!pairEmpty(node, last)) {
        addPair(node, last + 1);
    }

    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "sfnodes.krea2.dynamicimages",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            syncPairs(this);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            // LiteGraph.INPUT === 1；只响应输入侧连接变化。
            if (slotType === 1) {
                syncPairs(this);
            }
            return r;
        };

        // 载入已保存的工作流后恢复备用对。
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            syncPairs(this);
            return r;
        };
    },
});
