// SFPauseImage 前端逻辑测试（Node 直接运行：node tests/test_pause_image_js.js）
// 覆盖：state（getState/setGate）、prune 以 inputKey:"image" 调用（pause 删下游 /
//       continue 删 image 链接+菱形重路由+只删拉活上游的输出节点 / pass 不剪）
// prune 实现复用 sf_pause_text_lib.js（同一份 applyGateMode）
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_pi_js_"));
for (const n of ["sf_pause_text_lib.js", "sf_pause_image_lib.js"]) {
    fs.copyFileSync(path.join(__dirname, "..", "web", n), path.join(tmpDir, n.replace(/\.js$/, ".mjs")));
}

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(path.join(tmpDir, "sf_pause_image_lib.mjs"));
    const P = await import(path.join(tmpDir, "sf_pause_text_lib.mjs"));
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
    check("getState 容错非对象", getState({ properties: { [STATE_PROP]: "junk" } }).gate === "pause");
    s.frame = { filename: "sf_pause_1.png", subfolder: "", type: "temp" };
    check("frame 保留", getState(node).frame.filename === "sf_pause_1.png");

    // ---- pause：删下游（inputKey: image）----
    const outPause = {
        "0": { class_type: "EmptyImage", inputs: {} },
        "1": { class_type: "KSampler", inputs: { latent: ["0", 0] } },
        "2": { class_type: "VAEDecode", inputs: { samples: ["1", 0] } },
        "3": { class_type: "SFPauseImage", inputs: { image: ["2", 0] } },
        "4": { class_type: "Upscale", inputs: { image: ["3", 0] } },
        "5": { class_type: "SaveImage", inputs: { images: ["4", 0] } },
        "6": { class_type: "OtherBranch", inputs: {} },
        "7": { class_type: "SaveOther", inputs: { x: ["6", 0] } },
    };
    const entryPause = outPause["3"];
    applyGateMode(outPause, "3", entryPause, "pause", (c) => c === "SaveImage" || c === "SaveOther" || c === "SFPauseImage", "PauseState", { inputKey: "image" });
    check("pause 删除下游 4/5", !outPause["4"] && !outPause["5"]);
    check("pause 保留上游 0/1/2", outPause["0"] && outPause["1"] && outPause["2"]);
    check("pause 保留无关分支 6/7", outPause["6"] && outPause["7"]);
    check("pause 注入 PauseState", JSON.parse(entryPause.inputs.PauseState).mode === "pause");

    // ---- pass：不剪 ----
    const outPass = {
        "2": { class_type: "VAEDecode", inputs: {} },
        "3": { class_type: "SFPauseImage", inputs: { image: ["2", 0] } },
        "4": { class_type: "Upscale", inputs: { image: ["3", 0] } },
    };
    const entryPass = outPass["3"];
    applyGateMode(outPass, "3", entryPass, "pass", null, "PauseState", { inputKey: "image" });
    check("pass 不剪", outPass["2"] && outPass["3"] && outPass["4"]);
    check("pass 注入 PauseState", JSON.parse(entryPass.inputs.PauseState).mode === "pass");

    // ---- continue：删 image 链接 + 菱形重路由 ----
    const outC = {
        "0": { class_type: "EmptyImage", inputs: {} },
        "1": { class_type: "KSampler", inputs: { latent: ["0", 0] } },
        "2": { class_type: "VAEDecode", inputs: { samples: ["1", 0] } },
        "3": { class_type: "SFPauseImage", inputs: { image: ["2", 0] } },
        "4": { class_type: "Upscale", inputs: { image: ["3", 0], ref: ["2", 0] } },  // 菱形：也读 VAEDecode
        "5": { class_type: "SaveImage", inputs: { images: ["4", 0] } },
        "6": { class_type: "SecondOut", inputs: { images: ["2", 0] } },  // 平行输出：读 VAEDecode
        "7": { class_type: "SaveOther", inputs: { x: ["8", 0] } },       // 无关分支
        "8": { class_type: "OtherSrc", inputs: {} },
    };
    const entryC = outC["3"];
    applyGateMode(outC, "3", entryC, "continue", (c) => c !== "EmptyImage" && c !== "KSampler" && c !== "VAEDecode" && c !== "Upscale" && c !== "OtherSrc" && c !== "SFPauseImage", "PauseState", { inputKey: "image" });
    check("continue 删除 image 链接", !("image" in entryC.inputs));
    check("continue 注入 PauseState", JSON.parse(entryC.inputs.PauseState).mode === "continue");
    check("菱形重路由 ref 改指闸门", JSON.stringify(outC["4"].inputs.ref) === JSON.stringify(["3", 0]));
    check("菱形重路由 image 保持指闸门", JSON.stringify(outC["4"].inputs.image) === JSON.stringify(["3", 0]));
    check("continue 删拉活上游的平行输出 6", !outC["6"]);
    check("continue 保留下游 SaveImage 5", outC["5"] !== undefined);
    check("continue 保留无关分支 7/8", outC["7"] && outC["8"]);
    check("continue 保留上游为无害孤儿", outC["0"] !== undefined && outC["1"] !== undefined && outC["2"] !== undefined);

    // ---- continue 未接线闸门：不删任何输出 ----
    const outU = {
        "3": { class_type: "SFPauseImage", inputs: {} },
        "5": { class_type: "SaveImage", inputs: { images: ["9", 0] } },
        "9": { class_type: "KSampler", inputs: {} },
    };
    const entryU = outU["3"];
    applyGateMode(outU, "3", entryU, "continue", (c) => c === "SaveImage", "PauseState", { inputKey: "image" });
    check("未接线 continue 不删无关输出", outU["5"] && outU["9"]);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
