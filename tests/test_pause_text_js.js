// SFPauseText 纯函数库测试（Node 直接运行：node tests/test_pause_text_js.js）
// 覆盖：state 逻辑（getState/setGate/setText/setModelText/revertText/isEdited）、
//       prune 三模式（pause 删下游 / continue 跳上游模型链+菱形重路由+只删
//       拉活上游的输出节点 / pass 不剪）、isLink/buildConsumers/addAncestors
const fs = require("fs");
const os = require("os");
const path = require("path");

const tmpMjs = path.join(os.tmpdir(), "sf_pause_text_lib_test.mjs");
fs.copyFileSync(path.join(__dirname, "..", "web", "sf_pause_text_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

(async () => {
    const L = await import(tmpMjs);
    const {
        STATE_PROP, getState, setGate, setText, setModelText, revertText, isEdited,
        isLink, buildConsumers, addAncestors, applyGateMode,
    } = L;

    // ---- state ----
    const node = { properties: {} };
    let s = getState(node);
    check("初始状态", s.gate === "pause" && s.text === "" && s.original === "");
    check("STATE_PROP 键", node.properties[STATE_PROP] === s);
    setGate(node, "pass");
    check("setGate pass", getState(node).gate === "pass");
    setGate(node, "keep");
    check("setGate keep", getState(node).gate === "keep");
    setGate(node, "bogus");
    check("setGate 非法回退 pause", getState(node).gate === "pause");
    setText(node, "hello");
    check("setText", getState(node).text === "hello");
    check("isEdited（original 空）", isEdited(node) === true);
    setModelText(node, "model words");
    s = getState(node);
    check("setModelText 替换 text+original", s.text === "model words" && s.original === "model words");
    check("isEdited（与模型一致）", isEdited(node) === false);
    setText(node, "my edit");
    check("isEdited（编辑后）", isEdited(node) === true);
    revertText(node);
    check("revertText 回到模型原文", getState(node).text === "model words");
    check("getState 容错非对象", getState({ properties: { [STATE_PROP]: "junk" } }).gate === "pause");

    // ---- isLink / buildConsumers ----
    check("isLink", isLink([1, 0]) === true && isLink(["1", 2]) === true);
    check("isLink 拒绝", isLink([1]) === false && isLink("x") === false && isLink([1, "a"]) === false);
    const g = {
        "1": { inputs: { text: ["0", 0] } },
        "2": { inputs: { img: ["1", 0], other: "static" } },
        "3": { inputs: { t: ["1", 1] } },
    };
    const cons = buildConsumers(g);
    check("buildConsumers", cons.get("1").has("2") && cons.get("1").has("3") && cons.get("0").has("1"));
    const keep = new Set(["2"]);
    addAncestors(g, keep);
    check("addAncestors 并入图中存在的祖先", keep.has("1") && !keep.has("0"));  // "0" 不在 output 中

    // ---- pause：删下游，无关分支保留 ----
    const outPause = {
        "1": { class_type: "LLM", inputs: { seed: ["0", 0] } },
        "0": { class_type: "SFSeed", inputs: {} },
        "2": { class_type: "SFPauseText", inputs: { text: ["1", 0] } },
        "3": { class_type: "Process", inputs: { text: ["2", 0] } },
        "4": { class_type: "SaveImage", inputs: { img: ["3", 0] } },
        "5": { class_type: "OtherBranch", inputs: {} },
        "6": { class_type: "SaveOther", inputs: { x: ["5", 0] } },
    };
    const entryPause = outPause["2"];
    applyGateMode(outPause, "2", entryPause, "pause", (c) => c === "SaveImage" || c === "SaveOther" || c === "SFPauseText", "PauseState", { inputKey: "text", editedText: "box" });
    check("pause 删除下游 3/4", !outPause["3"] && !outPause["4"]);
    check("pause 保留闸门上游 0/1", outPause["0"] && outPause["1"]);
    check("pause 保留无关分支 5/6", outPause["5"] && outPause["6"]);
    check("pause 注入 PauseState", JSON.parse(entryPause.inputs.PauseState).mode === "pause" && JSON.parse(entryPause.inputs.PauseState).text === "box");

    // ---- pass：不剪 ----
    const outPass = {
        "1": { class_type: "LLM", inputs: {} },
        "2": { class_type: "SFPauseText", inputs: { text: ["1", 0] } },
        "3": { class_type: "Process", inputs: { text: ["2", 0] } },
    };
    const entryPass = outPass["2"];
    applyGateMode(outPass, "2", entryPass, "pass", null, "PauseState", { inputKey: "text", editedText: "box" });
    check("pass 不剪任何节点", outPass["1"] && outPass["2"] && outPass["3"]);
    check("pass 注入 PauseState", JSON.parse(entryPass.inputs.PauseState).mode === "pass");

    // ---- continue：跳上游模型链 ----
    const outC = {
        "0": { class_type: "SFSeed", inputs: {} },
        "1": { class_type: "LLM", inputs: { seed: ["0", 0] } },
        "2": { class_type: "SFPauseText", inputs: { text: ["1", 0] } },
        "3": { class_type: "Process", inputs: { text: ["2", 0] } },
        "4": { class_type: "SaveImage", inputs: { img: ["3", 0] } },
        "5": { class_type: "SecondOut", inputs: { text: ["1", 0] } },   // 也读 LLM 的输出节点
        "6": { class_type: "SaveOther", inputs: { x: ["7", 0] } },      // 无关分支
        "7": { class_type: "OtherSrc", inputs: {} },
    };
    const entryC = outC["2"];
    applyGateMode(outC, "2", entryC, "continue", (c) => c !== "SFSeed" && c !== "LLM" && c !== "Process" && c !== "OtherSrc" && c !== "SFPauseText", "PauseState", { inputKey: "text", editedText: "my edit" });
    check("continue 删除闸门 text 输入", !("text" in entryC.inputs));
    check("continue 注入 PauseState", JSON.parse(entryC.inputs.PauseState).mode === "continue" && JSON.parse(entryC.inputs.PauseState).text === "my edit");
    // 只删下游链之外、会拉活被跳过上游的输出节点："5"（SecondOut 读 LLM）删；
    // "4"（SaveImage）在闸门下游链上（消费闸门输出）保留
    check("continue 删下游链外拉活上游的输出节点 5", !outC["5"]);
    check("continue 保留下游链上的输出节点 4", outC["4"] !== undefined);
    check("continue 保留无关分支 6/7", outC["6"] && outC["7"]);
    check("continue 保留上游为无害孤儿（1/0）", outC["1"] !== undefined && outC["0"] !== undefined);
    check("continue 保留下游处理节点 3", outC["3"] !== undefined);
    check("continue 保留闸门自身", outC["2"] !== undefined);

    // ---- continue 菱形重路由：闸门之后直接读原文本源的链接改指闸门 ----
    const outD = {
        "1": { class_type: "LLM", inputs: {} },
        "2": { class_type: "SFPauseText", inputs: { text: ["1", 0] } },
        "3": { class_type: "Process", inputs: { text: ["2", 0], extra: ["1", 0] } },  // 菱形：也读 LLM
        "4": { class_type: "SaveImage", inputs: { img: ["3", 0] } },
    };
    const entryD = outD["2"];
    applyGateMode(outD, "2", entryD, "continue", (c) => c !== "LLM" && c !== "Process" && c !== "SFPauseText", "PauseState", { inputKey: "text", editedText: "e" });
    check("菱形重路由 extra 改指闸门", JSON.stringify(outD["3"].inputs.extra) === JSON.stringify(["2", 0]));
    check("菱形重路由 text 保持指闸门", JSON.stringify(outD["3"].inputs.text) === JSON.stringify(["2", 0]));
    check("菱形后 LLM 保留为无害孤儿（无输出读它则不删）", outD["1"] !== undefined && outD["3"] && outD["4"]);

    // ---- continue 未接线闸门（gateSrc null）：不删任何输出 ----
    const outU = {
        "2": { class_type: "SFPauseText", inputs: {} },
        "3": { class_type: "SaveImage", inputs: { img: ["9", 0] } },
        "9": { class_type: "KSampler", inputs: {} },
    };
    const entryU = outU["2"];
    applyGateMode(outU, "2", entryU, "continue", (c) => c === "SaveImage", "PauseState", { inputKey: "text", editedText: "u" });
    check("未接线 continue 不删无关输出", outU["3"] && outU["9"]);

    console.log("\nFAILURES:", failures.length);
    fs.unlinkSync(tmpMjs);
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("test crashed:", e);
    process.exit(1);
});
