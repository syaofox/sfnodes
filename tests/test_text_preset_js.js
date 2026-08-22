// SFTextPreset 前端逻辑测试（Node 直接运行：node tests/test_text_preset_js.js）
// 覆盖：扩展注册、nodeCreated 挂载（combo 重建/预览/按钮/callback 包装/onAfterGraphConfigured）、
//       管理弹窗（打开/列表渲染/新增/重名阻止/更新/删除/Escape 关闭）、presets_json 写回与 combo 同步
const fs = require("fs");
const path = require("path");

const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_text_preset.js"), "utf8")
    .replace('import { app } from "/scripts/app.js";', "")
    .replace('import { ComfyWidgets } from "/scripts/widgets.js";', "")
    .replace(/import\s*\{[^}]*\}\s*from\s*"\.\/sf_common\.js";/,
        "const installWheelZoomPassthrough = () => () => {};\nconst injectCSSOnce = () => {};");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// DOM mock
globalThis.window = { addEventListener: () => {}, innerWidth: 1200, innerHeight: 800 };
globalThis.alert = () => {};
const createdEls = [];
const makeEl = () => {
    const handlers = {};
    const el = {
        style: {}, textContent: "", value: "", className: "", children: [], type: "",
        placeholder: "", readOnly: false,
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
globalThis.document = {
    createElement: makeEl,
    body: { appendChild() {} },
    head: { appendChild() {} },
    addEventListener: (t, fn) => { docHandlers[t] = fn; },
};

const capturedExts = [];
const app = {
    graph: { _nodes: [] },
    registerExtension: (ext) => capturedExts.push(ext),
};
const fakeWidgetsApi = {
    STRING: (node, name, opts, appApi) => {
        const handlers = {};
        const w = { name, value: "", serialize: true, type: "string" };
        w.inputEl = {
            readOnly: false,
            addEventListener: (t, fn) => { (handlers[t] ??= []).push(fn); },
            trigger: (t, ...args) => { (handlers[t] ?? []).forEach((fn) => fn(...args)); },
        };
        node.widgets.push(w);
        return { widget: w };
    },
};
new Function("app", "ComfyWidgets", code)(app, fakeWidgetsApi);

const ext = capturedExts.find((e) => e.name === "sfnodes.text_preset");
check("存在扩展", ext !== undefined);
check("setup 挂载 keydown", typeof ext.setup === "function");
ext.setup();
check("keydown 已注册", typeof docHandlers.keydown === "function");

// ---------- nodeCreated 挂载 ----------
let mgrCallback = null;
const mkNode = (presetsJson = "[]", presetValue = "") => {
    const widgets = [
        { name: "preset", value: presetValue, type: "combo", options: { values: [""] }, callback: null, computeSize() {} },
        { name: "presets_json", value: presetsJson, type: "string" },
    ];
    const node = {
        comfyClass: "SFTextPreset",
        widgets,
        setDirtyCanvas: () => { node.dirty = (node.dirty ?? 0) + 1; },
        onAfterGraphConfigured: null,
        addWidget(type, label, value, cb) {
            if (type === "button") { mgrCallback = cb; this.mgrBtn = { type, name: label }; }
            return { type, name: label, value, callback: cb };
        },
    };
    ext.nodeCreated(node);
    return node;
};

const n1 = mkNode('[{"name": "A", "text": "hello"}, {"name": "B", "text": "world"}]', "A");
const presetW1 = n1.widgets.find((w) => w.name === "preset");
const jsonW1 = n1.widgets.find((w) => w.name === "presets_json");
const displayW1 = n1.widgets.find((w) => w.name === "content_display");
check("presets_json 已隐藏（hidden）", jsonW1.hidden === true);
check("presets_json 零尺寸", typeof jsonW1.computeSize === "function" && jsonW1.computeSize()[0] === 0);
check("presets_json 不绘制", typeof jsonW1.draw === "function");
check("presets_json 仍在数组中（可序列化）", n1.widgets.includes(jsonW1));
check("combo 选项重建为预设名", JSON.stringify(presetW1.options.values) === '["A","B"]');
check("当前选中保留", presetW1.value === "A");
check("预览 widget 已添加且不序列化", displayW1 !== undefined && displayW1.serialize === false);
check("预览显示选中文本", displayW1.value === "hello");
check("有选中预设时可编辑", displayW1.inputEl.readOnly === false);
check("⚙ 预设按钮已添加", n1.mgrBtn !== undefined);
check("combo callback 已包装", typeof presetW1.callback === "function");
check("onAfterGraphConfigured 已包装", typeof n1.onAfterGraphConfigured === "function");

presetW1.value = "B";
presetW1.callback("B");
check("切换 combo 同步预览", displayW1.value === "world");

// 编辑框直接修改：输入即写回 presets_json
displayW1.value = "edited world";
displayW1.inputEl.trigger("input");
check("编辑即时写回 presets_json", JSON.parse(jsonW1.value).find((p) => p.name === "B").text === "edited world");
check("编辑后显示保持", displayW1.value === "edited world");
presetW1.value = "A";
presetW1.callback("A");
check("切换 combo 不丢编辑", displayW1.value === "hello"
    && JSON.parse(jsonW1.value).find((p) => p.name === "B").text === "edited world");
presetW1.value = "B";
presetW1.callback("B");
check("切回显示编辑后文本", displayW1.value === "edited world");

const n2 = mkNode("[]", "");
check("空预设下拉保持空占位", JSON.stringify(n2.widgets.find((w) => w.name === "preset").options.values) === '[""]');
check("空预设时编辑框只读", n2.widgets.find((w) => w.name === "content_display").inputEl.readOnly === true);

const n3 = mkNode('[{"name": "A", "text": "x"}]', "Nonexistent");
check("选中值失效回落第一个", n3.widgets.find((w) => w.name === "preset").value === "A");

// 非法 JSON 容错
const n4 = mkNode("not json{{{", "");
check("非法 JSON 容错", JSON.stringify(n4.widgets.find((w) => w.name === "preset").options.values) === '[""]');

// onAfterGraphConfigured：加载 workflow 后重新同步
n1.widgets.find((w) => w.name === "presets_json").value = '[{"name": "C", "text": "ccc"}]';
n1.onAfterGraphConfigured();
check("configure 后 combo 同步新预设", JSON.stringify(presetW1.options.values) === '["C"]');
check("configure 后选中回落", presetW1.value === "C");
check("configure 后预览同步", displayW1.value === "ccc");

// ---------- 管理弹窗 ----------
const n5 = mkNode('[{"name": "A", "text": "hello"}, {"name": "B", "text": "world"}]', "A");
const jsonW5 = n5.widgets.find((w) => w.name === "presets_json");
const presetW5 = n5.widgets.find((w) => w.name === "preset");
const displayW5 = n5.widgets.find((w) => w.name === "content_display");

mgrCallback();
const overlay = createdEls.find((e) => e.className === "sf-preset-mgr-overlay");
check("弹窗已创建", overlay !== undefined && !overlay.removed);
const panel = overlay.children[0];
const body = panel.children[1];
const list = body.children[0];
check("列表渲染 2 项", list.children.length === 2);
check("列表项含名称与摘要", list.children[0].children[0].textContent === "A"
    && list.children[0].children[1].textContent === "hello");
const editor = body.children[1];
const nameInput = editor.children[1];
const textArea = editor.children[3];
check("初始载入第一个预设", nameInput.value === "A" && textArea.value === "hello");

// 新增
nameInput.value = "C";
textArea.value = "new text";
editor.children[4].children[0].click();
check("新增后列表 3 项", list.children.length === 3);
check("presets_json 已写回", JSON.parse(jsonW5.value).length === 3);
check("combo 同步新预设", JSON.stringify(presetW5.options.values) === '["A","B","C"]');
check("新增后选中新项", n5.dirty > 0 && JSON.parse(jsonW5.value)[2].name === "C");

// 重名阻止
nameInput.value = "A";
textArea.value = "dup";
editor.children[4].children[0].click();
check("重名新增被阻止", JSON.parse(jsonW5.value).length === 3);

// 更新
list.children[1].click();
check("点击选中载入编辑区", nameInput.value === "B" && textArea.value === "world");
nameInput.value = "B2";
textArea.value = "updated";
editor.children[4].children[1].click();
check("更新写回", JSON.parse(jsonW5.value)[1].name === "B2" && JSON.parse(jsonW5.value)[1].text === "updated");
check("combo 选项更新", JSON.stringify(presetW5.options.values) === '["A","B2","C"]');

// 重名更新阻止
nameInput.value = "A";
textArea.value = "x";
editor.children[4].children[1].click();
check("重名更新被阻止", JSON.parse(jsonW5.value)[1].name === "B2");

// 删除
list.children[1].click();
editor.children[4].children[2].click();
check("删除后列表 2 项", list.children.length === 2);
const afterDel = JSON.parse(jsonW5.value);
check("删除写回", afterDel.length === 2 && afterDel[1].name === "C");
check("combo 同步删除", JSON.stringify(presetW5.options.values) === '["A","C"]');
check("预览同步", displayW5.value === "hello");

// 预览随 combo 选中变化（选中 C，其文本为新增时写入的 "new text"）
presetW5.value = "C";
presetW5.callback("C");
check("选中 C 预览更新", displayW5.value === "new text");

// Escape 关闭
docHandlers.keydown({ key: "Escape" });
check("Escape 关闭弹窗", overlay.removed === true);

// 空预设列表的弹窗
const n6 = mkNode("[]", "");
mgrCallback();
const overlay6 = createdEls.find((e) => e.className === "sf-preset-mgr-overlay" && e !== overlay);
check("空列表弹窗渲染空提示", overlay6 !== undefined
    && overlay6.children[0].children[1].children[0].children[0].className.includes("sf-preset-mgr-empty"));
docHandlers.keydown({ key: "Escape" });

console.log("\nFAILURES:", failures.length);
process.exit(failures.length ? 1 : 0);
