// ==========================================================================
// sf_convert_anything.js - SFConvert Anything 输出槽改型扩展
// ==========================================================================
//
// 复刻 easy convertAnything 的前端半边（py/nodes/logic.py::SFConvertAnything
// 是后端权威）：output_type combo 变化时把输出槽 type 同步为大写类型名
// （STRING/INT/FLOAT/BOOLEAN），让画布连线校验与槽点颜色跟随所选类型。
// 后端始终声明 "*"（any_type），改型纯属前端渲染与连线校验。
//
// 与原件差异（已确认范围）：
// - 保留输出槽名 "output" 不随类型改名（easy 会把槽名改成类型值）
// - 补上 onAfterGraphConfigured 恢复：easy 无恢复逻辑，重载工作流后槽型
//   回退 "*"；这里按 widget 当前值恢复（platform §2.7：元素替换式改型）
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { setSlotType } from "./any_pack.js";

const CLASS = "SFConvertAnything";
const WIDGET_NAME = "output_type";
// combo 值 -> 槽位类型（与 py 侧 CONVERT_ANYTHING_CONVERTERS 键一致）
const SOCKET_TYPES = {
    string: "STRING",
    int: "INT",
    float: "FLOAT",
    boolean: "BOOLEAN",
};

function applyOutputType(node, value) {
    const type = SOCKET_TYPES[value] || "*";
    setSlotType(node, node.outputs, 0, type);
}

app.registerExtension({
    name: "sfnodes.ConvertAnything",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== CLASS) return;
        // 热重载防双包装（sf_dropdown.js 先例）
        if (nodeType.prototype._sfConvertAnythingPatched) return;
        nodeType.prototype._sfConvertAnythingPatched = true;

        // ── 创建：挂 combo callback（callback 参数即新值）────────────
        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.name === WIDGET_NAME);
            if (!widget) return;
            const original = widget.callback;
            widget.callback = (value) => {
                original?.(value);
                applyOutputType(this, value);
            };
            applyOutputType(this, widget.value);
        };

        // ── 加载恢复：nodeCreated 早于 widgets_values 恢复，挂这里 ────
        const _configured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = _configured?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.name === WIDGET_NAME);
            if (widget) applyOutputType(this, widget.value);
            return r;
        };
    },
});
