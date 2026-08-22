// SFPauseMask 前端逻辑测试（Node 直接运行：node tests/test_pause_mask_js.js）
// 覆盖：state（getState/setGate）、prune 以 inputKey:"mask" 调用（pause 删下游 /
//       continue 删 mask 链接+菱形重路由+只删拉活上游的输出节点 / pass 不剪）
// prune 复用 sf_pause_text_lib.js（同一份 applyGateMode）
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pm_js_"));
// text_lib 是纯模块原样拷贝；kit 依赖 /scripts/app.js、/scripts/api.js，
// 改写为 globalThis 桩后拷贝
fs.copyFileSync(path.join(__dirname, "..", "web", "sf_pause_text_lib.js"), path.join(tmpDir, "sf_pause_text_lib.mjs"));
{
    const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_pause_kit.js"), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, "sf_pause_kit.mjs"), code);
}
// kit 依赖的 sf_common 桩（makeGateState 路径不会真正调用）
fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"),
    "export function applyAdaptiveCanvasOnly() {}\n"
    + "export function sfApiUrl(r) { return r; }\n"
    + "export function injectCSSOnce() {}\n");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const K = await import(path.join(tmpDir, "sf_pause_kit.mjs"));
    const P = await import(path.join(tmpDir, "sf_pause_text_lib.mjs"));
    const L = K.makeGateState("pauseMaskState");
    const { STATE_PROP, getState, setGate } = L;
    const { applyGateMode } = P;

    // ---- state ----
    const node = { properties: {} };
    let s = getState(node);
    check("初始状态", s.gate === "pause" && s.frame === null);
    check("STATE_PROP 键", node.properties[STATE_PROP] === s);
    setGate(node, "pass");
    check("setGate pass", getState(node).gate === "pass");
    setGate(node, "bogus");
    check("setGate 非法回退 pause", getState(node).gate === "pause");
    s.frame = { filename: "sf_pause_mask_1.png", subfolder: "", type: "temp" };
    check("frame 保留", getState(node).frame.filename === "sf_pause_mask_1.png");

    // ---- pause：删下游（inputKey: mask）----
    const outPause = {
        "0": { class_type: "MaskSource", inputs: {} },
        "1": { class_type: "SFPauseMask", inputs: { mask: ["0", 0] } },
        "2": { class_type: "MaskUpscale", inputs: { mask: ["1", 0] } },
        "3": { class_type: "SaveMask", inputs: { mask: ["2", 0] } },
        "4": { class_type: "OtherBranch", inputs: {} },
        "5": { class_type: "SaveOther", inputs: { x: ["4", 0] } },
    };
    const entryPause = outPause["1"];
    applyGateMode(outPause, "1", entryPause, "pause", (c) => c === "SaveMask" || c === "SaveOther" || c === "SFPauseMask", "PauseState", { inputKey: "mask" });
    check("pause 删除下游 2/3", !outPause["2"] && !outPause["3"]);
    check("pause 保留上游 0", outPause["0"] !== undefined);
    check("pause 保留无关分支 4/5", outPause["4"] && outPause["5"]);
    check("pause 注入 PauseState", JSON.parse(entryPause.inputs.PauseState).mode === "pause");

    // ---- pass：不剪 ----
    const outPass = {
        "0": { class_type: "MaskSource", inputs: {} },
        "1": { class_type: "SFPauseMask", inputs: { mask: ["0", 0] } },
        "2": { class_type: "MaskUpscale", inputs: { mask: ["1", 0] } },
    };
    const entryPass = outPass["1"];
    applyGateMode(outPass, "1", entryPass, "pass", null, "PauseState", { inputKey: "mask" });
    check("pass 不剪", outPass["0"] && outPass["1"] && outPass["2"]);
    check("pass 注入 PauseState", JSON.parse(entryPass.inputs.PauseState).mode === "pass");

    // ---- continue：删 mask 链接 + 菱形重路由 ----
    const outC = {
        "0": { class_type: "MaskSource", inputs: {} },
        "1": { class_type: "SFPauseMask", inputs: { mask: ["0", 0] } },
        "2": { class_type: "MaskUpscale", inputs: { mask: ["1", 0], ref: ["0", 0] } },  // 菱形：也读 MaskSource
        "3": { class_type: "SaveMask", inputs: { mask: ["2", 0] } },
        "4": { class_type: "SecondOut", inputs: { mask: ["0", 0] } },  // 平行输出
        "5": { class_type: "SaveOther", inputs: { x: ["6", 0] } },     // 无关分支
        "6": { class_type: "OtherSrc", inputs: {} },
    };
    const entryC = outC["1"];
    applyGateMode(outC, "1", entryC, "continue", (c) => c !== "MaskSource" && c !== "MaskUpscale" && c !== "OtherSrc" && c !== "SFPauseMask", "PauseState", { inputKey: "mask" });
    check("continue 删除 mask 链接", !("mask" in entryC.inputs));
    check("continue 注入 PauseState", JSON.parse(entryC.inputs.PauseState).mode === "continue");
    check("菱形重路由 ref 改指闸门", JSON.stringify(outC["2"].inputs.ref) === JSON.stringify(["1", 0]));
    check("菱形重路由 mask 保持指闸门", JSON.stringify(outC["2"].inputs.mask) === JSON.stringify(["1", 0]));
    check("continue 删拉活上游的平行输出 4", !outC["4"]);
    check("continue 保留下游 SaveMask 3", outC["3"] !== undefined);
    check("continue 保留无关分支 5/6", outC["5"] && outC["6"]);
    check("continue 保留上游为无害孤儿", outC["0"] !== undefined);

    // ---- continue 未接线闸门：不删任何输出 ----
    const outU = {
        "1": { class_type: "SFPauseMask", inputs: {} },
        "3": { class_type: "SaveMask", inputs: { mask: ["9", 0] } },
        "9": { class_type: "MaskSource", inputs: {} },
    };
    const entryU = outU["1"];
    applyGateMode(outU, "1", entryU, "continue", (c) => c === "SaveMask", "PauseState", { inputKey: "mask" });
    check("未接线 continue 不删无关输出", outU["3"] && outU["9"]);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
