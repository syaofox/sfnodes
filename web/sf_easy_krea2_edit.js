// SFEasyKrea2Edit 前端扩展：
//   - 动态参考图槽位 image1..imageN（复用 sf_dynamic_slots：全连追加/断开回收/加载恢复）
//   - 逐图 strength 的动态 number widget（显示），真源在 node.properties（见 lib）
//   - graphToPrompt 注入 hidden SFEasyKrea2EditState（subgraph-safe，fail-open）
import { app } from "/scripts/app.js";
import { installDynamicSlots, installConfiguredSlotRecovery } from "./sf_dynamic_slots.js";
import { HIDDEN_INPUT, stateJson, syncStrengthWidgets } from "./sf_easy_krea2_edit_lib.js";

const CLASS = "SFEasyKrea2Edit";
const SLOT_CFG = {
    inputPrefix: "image",
    inputStart: 1,
    inputCount: 20,
    inputType: "IMAGE",
    initialInputs: 1,
};

app.registerExtension({
    name: "sfnodes.EasyKrea2Edit",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CLASS) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            installDynamicSlots(this, SLOT_CFG);
            installConfiguredSlotRecovery(this, SLOT_CFG);
            const origConfigured = this.onAfterGraphConfigured;
            this.onAfterGraphConfigured = function () {
                origConfigured?.apply(this, arguments);
                syncStrengthWidgets(this);
            };
            const origConn = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                origConn?.apply(this, arguments);
                syncStrengthWidgets(this);
            };
            syncStrengthWidgets(this);
        };
    },
});

// ── graphToPrompt：注入逐图 strength 隐藏状态（subgraph-safe）──
const _origG2P = app.graphToPrompt.bind(app);
app.graphToPrompt = async function (...args) {
    const result = await _origG2P(...args);
    // FAIL OPEN — 这里抛错会让整个工作流的 Run 失败；不要包住上面的 await。
    try {
        const out = result?.output;
        if (out) {
            let index = null;
            const buildIndex = () => {
                const m = new Map();
                const visit = (g) => {
                    if (!g) return;
                    for (const n of (g._nodes || g.nodes || [])) {
                        if (!n) continue;
                        if (n.comfyClass === CLASS || n.type === CLASS) m.set(String(n.id), n);
                        const inner = n.subgraph || n.graph || n._graph;
                        if (inner && inner !== g) visit(inner);
                    }
                };
                visit(app.graph);
                return m;
            };
            for (const id in out) {
                if (out[id]?.class_type !== CLASS) continue;
                if (!index) index = buildIndex();
                const sId = String(id);
                let node = index.get(sId);
                if (!node && sId.includes(":")) node = index.get(sId.slice(sId.lastIndexOf(":") + 1));
                if (!node) continue;
                out[id].inputs = out[id].inputs || {};
                out[id].inputs[HIDDEN_INPUT] = stateJson(node);
            }
        }
    } catch (e) {
        console.error("[SFEasyKrea2Edit] prompt injection failed; prompt sent unchanged", e);
    }
    return result;
};
