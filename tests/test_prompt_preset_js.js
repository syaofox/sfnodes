// SFPromptPreset 前端逻辑测试（Node 直接运行：node tests/test_prompt_preset_js.js）
// 覆盖：pose/couple 互斥联动、description 动态 tooltip、分组选择器弹窗（按钮打开/tab 切换/分组渲染/选择写入/关闭）
const fs = require("fs");
const path = require("path");

const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "prompt_preset.js"), "utf8")
    .replace('import { app } from "/scripts/app.js";', "")
    .replace('import { api } from "/scripts/api.js";', "");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// DOM mock
globalThis.window = { addEventListener: () => {}, innerWidth: 1200, innerHeight: 800 };
const createdEls = [];
const makeEl = () => {
    const handlers = {};
    const el = {
        style: {}, textContent: "", value: "", className: "", children: [],
        offsetWidth: 220, offsetHeight: 300,
        appendChild(child) { this.children.push(child); },
        replaceChildren(...kids) { this.children = kids; },
        querySelector(sel) {
            const cls = sel.slice(1);
            const find = (nodes) => {
                for (const c of nodes) {
                    if (c.className === cls) return c;
                    const r = find(c.children ?? []);
                    if (r) return r;
                }
                return undefined;
            };
            return find(this.children);
        },
        querySelectorAll(sel) { return this.children.filter((c) => c.className.includes(sel.slice(1))); },
        classList: { add() {}, remove() {}, contains() { return false; } },
        addEventListener(t, fn) { (handlers[t] ??= []).push(fn); },
        click() { (handlers.click ?? []).forEach((fn) => fn()); },
        trigger(t, ...args) { (handlers[t] ?? []).forEach((fn) => fn(...args)); },
        remove() { this.removed = true; }, contains() { return false; }, focus() {},
    };
    createdEls.push(el);
    return el;
};
const docHandlers = {};
const headEl = { appendChild() {} };
globalThis.document = {
    createElement: makeEl,
    body: { appendChild() {} },
    head: headEl,
    addEventListener: (t, fn) => { docHandlers[t] = fn; },
};

const capturedExts = [];
const presetData = {
    Pose: {
        回眸: { description: "Looking back over the shoulder", group: "SFW" },
        裸卧张开腿: { description: "Nude lying pose", group: "NSFW" },
    },
    Outfit: {
        旗袍: { description: "Form-fitting qipao dress", group: "SFW" },
        全裸: { description: "Fully nude", group: "NSFW" },
    },
    Celebrity: {
        "Taylor Swift": { description: "American singer-songwriter", group: "歌手" },
        周杰伦: { description: "Taiwanese singer", group: "亚洲名人" },
    },
};
let dirtyCount = 0;
const app = {
    graph: { _nodes: [], setDirtyCanvas: () => { dirtyCount++; } },
    registerExtension: (ext) => capturedExts.push(ext),
};
const api = { fetchApi: async () => ({ ok: true, json: async () => presetData }) };
new Function("app", "api", code)(app, api);

const mainExt = capturedExts.find((e) => e.name === "sfnodes.prompt_preset");
check("存在扩展", mainExt !== undefined);

(async () => {
    await mainExt.setup();
    check("setup 挂载 keydown", typeof docHandlers.keydown === "function");

    // ---------- 互斥联动 + tooltip ----------
    const nodeType = { prototype: {} };
    mainExt.beforeRegisterNodeDef(nodeType, { name: "SFPromptPreset" });
    const otherType = { prototype: {} };
    mainExt.beforeRegisterNodeDef(otherType, { name: "OtherNode" });
    check("其他节点不受影响", otherType.prototype.onNodeCreated === undefined);

    const fakeWidget = (name, initial = "禁用") => ({ name, value: initial, callback: null, tooltip: "initial-tooltip" });
    let pickerCallback = null;
    const fakeNode = () => {
        const n = {
            type: "SFPromptPreset",
            widgets: [fakeWidget("pose_preset"), fakeWidget("couple_preset"), fakeWidget("outfit_preset"), fakeWidget("celebrity_preset"), fakeWidget("input_text")],
            addWidget(type, label, value, cb) { pickerCallback = cb; this.widgets.push({ type, name: label, value, callback: cb }); },
        };
        nodeType.prototype.onNodeCreated.call(n);
        return n;
    };

    const n0 = fakeNode();
    const pose0 = n0.widgets.find((w) => w.name === "pose_preset");
    const couple0 = n0.widgets.find((w) => w.name === "couple_preset");
    const outfit0 = n0.widgets.find((w) => w.name === "outfit_preset");
    const celeb0 = n0.widgets.find((w) => w.name === "celebrity_preset");
    const text0 = n0.widgets.find((w) => w.name === "input_text");
    check("按钮 widget 已添加", n0.widgets.some((w) => w.type === "button"));
    check("预设 widget 均有 callback 包装", ["pose_preset", "couple_preset", "outfit_preset", "celebrity_preset"]
        .every((n) => typeof n0.widgets.find((w) => w.name === n).callback === "function"));
    check("非预设 widget 不包装", text0.callback === null);

    pose0.value = "回眸";
    pose0.callback();
    check("pose tooltip 显示 description", pose0.tooltip === "Looking back over the shoulder");
    check("选 pose 后 couple 置禁用", couple0.value === "禁用");
    pose0.value = "禁用";
    pose0.callback();
    check("置禁用后 tooltip 清空", pose0.tooltip === null);

    // ---------- 分组选择器弹窗 ----------
    app.graph._nodes = [n0];
    const beforePicker = createdEls.length;
    pickerCallback();
    const overlay = createdEls.slice(beforePicker).find((e) => e.className === "sf-preset-picker-overlay");
    check("弹窗 overlay 已创建", overlay !== undefined && !overlay.removed);
    const panel = overlay.children[0];
    check("面板已创建", panel !== undefined);

    const tabs = panel.children[1];
    check("10 个分类 tab", tabs.children.length === 10);
    const poseTab = tabs.children.find((c) => c.textContent === "单人动作");
    check("tab 内容正确", poseTab !== undefined);

    // 默认 Celebrity tab：分组标题 + 选项
    const list = panel.children[3];
    const groups = list.children.filter((c) => c.textContent === "歌手" || c.textContent === "亚洲名人");
    check("Celebrity 分组标题渲染", groups.length === 2);
    const celebItems = list.children.filter((c) => c.textContent === "Taylor Swift" || c.textContent === "周杰伦");
    check("Celebrity 选项渲染", celebItems.length === 2);

    // 切换到 单人动作 tab → SFW/NSFW 分组
    poseTab.click();
    const poseList = panel.children[3];
    const poseGroups = poseList.children.filter((c) => c.textContent === "SFW" || c.textContent === "NSFW");
    check("Pose SFW/NSFW 分组", poseGroups.length === 2);

    // 搜索过滤
    const search = panel.children[2];
    search.value = "裸卧";
    search.trigger("input");
    const listAfterSearch = panel.children[3];
    const nsfwOnly = listAfterSearch.children.filter((c) => c.textContent === "裸卧张开腿");
    check("搜索后目标选项保留", nsfwOnly.length === 1);
    check("搜索过滤掉无关选项", !listAfterSearch.children.some((c) => c.textContent === "回眸"));

    // 选项点击写入 widget
    const item = listAfterSearch.children.find((c) => c.textContent === "裸卧张开腿");
    item.click();
    check("选项点击写入 pose widget", pose0.value === "裸卧张开腿");
    check("选择后弹窗关闭", overlay.removed === true);
    check("canvas 重绘触发", dirtyCount > 0);

    // 重新打开 → Escape 关闭
    const before2 = createdEls.length;
    pickerCallback();
    const overlay2 = createdEls.slice(before2).find((e) => e.className === "sf-preset-picker-overlay");
    docHandlers.keydown({ key: "Escape" });
    check("Escape 关闭弹窗", overlay2.removed === true);

    console.log("\nFAILURES:", failures.length);
    process.exit(failures.length ? 1 : 0);
})();
