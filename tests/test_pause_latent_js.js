// SFPauseLatent 前端逻辑测试（Node 直接运行：node tests/test_pause_latent_js.js）
// 覆盖：state（getState/setGate）、prune 以 inputKey:"latent" + extraInputKeys:
// ["image"] 调用（pause 删下游且保留 image 链接 / continue 删 latent+image 链接、
//       菱形重路由、只删拉活上游的输出节点 / pass 不剪）
// prune 实现复用 sf_pause_text_lib.js（同一份 applyGateMode）
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pl_js_"));
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
    const L = K.makeGateState("pauseLatentState");
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
    s.frame = { filename: "sf_pause_latent_1.png", subfolder: "", type: "temp" };
    check("frame 保留", getState(node).frame.filename === "sf_pause_latent_1.png");

    const isOutput = (c) => c === "SaveImage" || c === "SaveOther" || c === "SFPauseLatent";
    const H = "PauseState";

    // ---- pause：删下游（inputKey: latent），image 预览链接保留 ----
    const outPause = {
        "0": { class_type: "EmptyLatent", inputs: {} },
        "1": { class_type: "KSampler", inputs: { latent: ["0", 0] } },
        "2": { class_type: "VAEDecode", inputs: { samples: ["1", 0] } },
        "3": { class_type: "SFPauseLatent", inputs: { latent: ["1", 0], image: ["2", 0] } },
        "4": { class_type: "KSampler", inputs: { latent: ["3", 0] } },
        "5": { class_type: "SaveImage", inputs: { images: ["4", 0] } },
        "6": { class_type: "OtherBranch", inputs: {} },
        "7": { class_type: "SaveOther", inputs: { x: ["6", 0] } },
    };
    const entryPause = outPause["3"];
    applyGateMode(outPause, "3", entryPause, "pause", isOutput, H, { inputKey: "latent", extraInputKeys: ["image"] });
    check("pause 删除下游 4/5", !outPause["4"] && !outPause["5"]);
    check("pause 保留上游 0/1/2", outPause["0"] && outPause["1"] && outPause["2"]);
    check("pause 保留无关分支 6/7", outPause["6"] && outPause["7"]);
    check("pause 保留 image 预览链接", JSON.stringify(entryPause.inputs.image) === JSON.stringify(["2", 0]));
    check("pause 注入 PauseState", JSON.parse(entryPause.inputs.PauseState).mode === "pause");

    // ---- pass：不剪，image 预览链接保留 ----
    const outPass = {
        "2": { class_type: "VAEDecode", inputs: {} },
        "3": { class_type: "SFPauseLatent", inputs: { latent: ["1", 0], image: ["2", 0] } },
        "4": { class_type: "KSampler", inputs: { latent: ["3", 0] } },
    };
    const entryPass = outPass["3"];
    applyGateMode(outPass, "3", entryPass, "pass", null, H, { inputKey: "latent", extraInputKeys: ["image"] });
    check("pass 不剪", outPass["2"] && outPass["3"] && outPass["4"]);
    check("pass 保留 image 预览链接", JSON.stringify(entryPass.inputs.image) === JSON.stringify(["2", 0]));
    check("pass 注入 PauseState", JSON.parse(entryPass.inputs.PauseState).mode === "pass");

    // ---- continue：删 latent + image 链接、菱形重路由、保留第二段采样 ----
    const outC = {
        "0": { class_type: "EmptyLatent", inputs: {} },
        "1": { class_type: "KSampler", inputs: { latent: ["0", 0] } },       // 第一段采样（被跳过）
        "2": { class_type: "VAEDecode", inputs: { samples: ["1", 0] } },     // 预览解码（被跳过）
        "3": { class_type: "SFPauseLatent", inputs: { latent: ["1", 0], image: ["2", 0] } },
        "4": { class_type: "KSampler", inputs: { latent: ["3", 0], ref: ["1", 0] } },  // 第二段采样 + 菱形
        "5": { class_type: "SaveImage", inputs: { images: ["6", 0] } },
        "6": { class_type: "VAEDecode", inputs: { samples: ["4", 0] } },
        "7": { class_type: "SaveOther", inputs: { x: ["8", 0] } },           // 无关分支
        "8": { class_type: "OtherSrc", inputs: {} },
    };
    const entryC = outC["3"];
    applyGateMode(outC, "3", entryC, "continue", isOutput, H, { inputKey: "latent", extraInputKeys: ["image"] });
    check("continue 删除 latent 链接", !("latent" in entryC.inputs));
    check("continue 删除 image 预览链接（VAEDecode 失联）", !("image" in entryC.inputs));
    check("continue 注入 PauseState", JSON.parse(entryC.inputs.PauseState).mode === "continue");
    check("菱形重路由 ref 改指闸门", JSON.stringify(outC["4"].inputs.ref) === JSON.stringify(["3", 0]));
    check("菱形重路由 latent 保持指闸门", JSON.stringify(outC["4"].inputs.latent) === JSON.stringify(["3", 0]));
    check("continue 保留下游第二段采样 4", outC["4"] !== undefined);
    check("continue 保留下游 SaveImage 5 / VAE 6", outC["5"] !== undefined && outC["6"] !== undefined);
    check("continue 保留无关分支 7/8", outC["7"] && outC["8"]);
    check("continue 保留第一段为无害孤儿", outC["0"] !== undefined && outC["1"] !== undefined && outC["2"] !== undefined);
    // 关键：删 image 链接后 VAEDecode 输出不再被任何节点消费（不调度 → 不拉活 KSampler）
    const consumersOf2 = Object.entries(outC)
        .filter(([id, e]) => id !== "3" && e?.inputs && Object.values(e.inputs).some((v) => JSON.stringify(v) === JSON.stringify(["2", 0])));
    check("VAEDecode 输出不再被消费", consumersOf2.length === 0);

    // ---- continue 未接线闸门：不删任何输出 ----
    const outU = {
        "3": { class_type: "SFPauseLatent", inputs: {} },
        "5": { class_type: "SaveImage", inputs: { images: ["9", 0] } },
        "9": { class_type: "KSampler", inputs: {} },
    };
    const entryU = outU["3"];
    applyGateMode(outU, "3", entryU, "continue", isOutput, H, { inputKey: "latent", extraInputKeys: ["image"] });
    check("未接线 continue 不删无关输出", outU["5"] && outU["9"]);

    // ---- 回归：不传 extraInputKeys 时行为与 image/text/mask 版一致 ----
    const outLegacy = {
        "1": { class_type: "KSampler", inputs: {} },
        "3": { class_type: "SFPauseLatent", inputs: { latent: ["1", 0], image: ["2", 0] } },
        "4": { class_type: "KSampler", inputs: { latent: ["3", 0] } },
    };
    const entryLegacy = outLegacy["3"];
    applyGateMode(outLegacy, "3", entryLegacy, "continue", isOutput, H, { inputKey: "latent" });
    check("无 extraInputKeys 时 image 链接不被删（旧调用兼容）", JSON.stringify(entryLegacy.inputs.image) === JSON.stringify(["2", 0]));

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
