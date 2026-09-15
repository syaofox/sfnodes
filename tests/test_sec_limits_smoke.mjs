// sf_sec_limits.js 冒烟测试（Node 直接运行：node tests/test_sec_limits_smoke.mjs）
// 覆盖：beforeRegisterNodeDef 抬高 input spec 的 max；onNodeCreated/onAfterGraphConfigured
// 兜底抬高 widget.options.max；非目标节点不处理；已有更大 max 不降级。
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const WEB = path.resolve(__dirname, "..", "web");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_seclim_"));

let captured = null;
fs.writeFileSync(path.join(tmp, "stub_app.mjs"),
    "export const app = { registerExtension(cfg){ globalThis.__SF_CAPTURED__ = cfg; } };\n");

let src = fs.readFileSync(path.join(WEB, "sf_sec_limits.js"), "utf-8");
src = src.replace('import { app } from "/scripts/app.js";', 'import { app } from "./stub_app.mjs";');
fs.writeFileSync(path.join(tmp, "sf_sec_limits.mjs"), src);

const mod = await import(`file://${path.join(tmp, "sf_sec_limits.mjs")}`);
captured = globalThis.__SF_CAPTURED__;

check("扩展已注册", captured && captured.name === "sfnodes.SecLimits");
check("RAISE 常量", mod.RAISE.annotation_frame_idx === 1000000);

// spec 抬高
const spec = ["INT", { default: 0, min: 0 }];
mod.raiseSpec(spec, 1000000);
check("raiseSpec 设置 max", spec[1].max === 1000000);
const big = ["INT", { min: 0, max: 5000000 }];
mod.raiseSpec(big, 1000000);
check("已有更大 max 不降级", big[1].max === 5000000);

// 节点定义阶段
const proto = {};
const nodeType = { prototype: proto };
const nodeData = {
    name: "SeCVideoSegmentation",
    input: { optional: {
        annotation_frame_idx: ["INT", { default: 0, min: 0 }],
        object_id: ["INT", { default: 1, min: 1 }],
        max_frames_to_track: ["INT", { default: -1, min: -1 }],
    } },
};
await captured.beforeRegisterNodeDef(nodeType, nodeData);
check("spec annotation max", nodeData.input.optional.annotation_frame_idx[1].max === 1000000);
check("spec object_id max", nodeData.input.optional.object_id[1].max === 1000000);
check("spec max_frames max", nodeData.input.optional.max_frames_to_track[1].max === 1000000);

// onNodeCreated 兜底
const node = {
    comfyClass: "SeCVideoSegmentation",
    widgets: [
        { name: "annotation_frame_idx", options: { max: 2048 } },
        { name: "object_id", options: {} },
    ],
};
proto.onNodeCreated.call(node);
check("widget max 抬高", node.widgets[0].options.max === 1000000);
check("widget 无 max → 抬高", node.widgets[1].options.max === 1000000);

// 非目标节点不处理
const other = { comfyClass: "OtherNode", type: "OtherNode", widgets: [{ name: "annotation_frame_idx", options: { max: 2048 } }] };
proto.onAfterGraphConfigured.call(other);
check("非目标节点不动", other.widgets[0].options.max === 2048);

if (failures.length) {
    console.log("FAILED:", failures.length, failures);
    process.exit(1);
}
console.log("ALL PASS");
