// SFTextPreset 前端逻辑测试（Node 直接运行：node tests/test_text_preset_js.js）
// 覆盖：扩展注册、nodeCreated 挂载（combo 重建/预览/按钮/callback 包装/onAfterGraphConfigured）、
//       全局库合并渲染（全局优先 + 工作流残留）、草稿语义（编辑写 text_override 不碰全局库/
//       切预设清草稿/💾 保存到预设写库并清草稿/残留项晋升）、configure 保留选中值、
//       API 失败降级、管理弹窗（打开/列表/新增/重名/更新改名/删除/Escape/双击重入）
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

// 全局预设库（fetch mock 的数据源）
let store = [{ name: "G1", text: "g1 text" }, { name: "G2", text: "g2 text" }];
let apiFail = false;
const apiCalls = [];
globalThis.fetch = async (url, opts) => {
    const method = (opts && opts.method) || "GET";
    apiCalls.push({ method, url, body: opts && opts.body ? JSON.parse(opts.body) : null });
    if (apiFail) return { ok: false, status: 404, json: async () => ({}) };
    if (method === "GET") {
        return { ok: true, status: 200, json: async () => ({ presets: store.map((p) => ({ ...p })) }) };
    }
    if (method === "POST") {
        const { name, text } = JSON.parse(opts.body);
        const hit = store.find((p) => p.name === name);
        if (hit) hit.text = text;
        else store.push({ name, text });
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
    }
    if (method === "DELETE") {
        const name = decodeURIComponent(String(url).split("name=")[1] ?? "");
        const idx = store.findIndex((p) => p.name === name);
        if (idx < 0) return { ok: false, status: 404, json: async () => ({}) };
        store.splice(idx, 1);
        return { ok: true, status: 200, json: async () => ({ deleted: name }) };
    }
    return { ok: false, status: 500, json: async () => ({}) };
};

// DOM mock
globalThis.window = { addEventListener: () => {}, innerWidth: 1200, innerHeight: 800 };
const alerts = [];
globalThis.alert = (m) => alerts.push(m);
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

const flush = () => new Promise((r) => setImmediate(r));

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
(async () => {
const mkNode = (presetsJson = "[]", presetValue = "") => {
    const widgets = [
        { name: "preset", value: presetValue, type: "combo", options: { values: [""] }, callback: null, computeSize() {} },
        { name: "presets_json", value: presetsJson, type: "string" },
        { name: "text_override", value: "", type: "string" },
    ];
    const node = {
        comfyClass: "SFTextPreset",
        widgets,
        setDirtyCanvas: () => { node.dirty = (node.dirty ?? 0) + 1; },
        onAfterGraphConfigured: null,
        addWidget(type, label, value, cb) {
            if (type === "button") node.buttons[label] = cb;
            return { type, name: label, value, callback: cb };
        },
    };
    node.buttons = {};
    app.graph._nodes.push(node);
    ext.nodeCreated(node);
    return node;
};

const n1 = mkNode('[{"name": "A", "text": "hello"}, {"name": "B", "text": "world"}]', "A");
const presetW1 = n1.widgets.find((w) => w.name === "preset");
const jsonW1 = n1.widgets.find((w) => w.name === "presets_json");
const draftW1 = n1.widgets.find((w) => w.name === "text_override");
const displayW1 = n1.widgets.find((w) => w.name === "content_display");
check("presets_json 已隐藏（hidden）", jsonW1.hidden === true);
check("presets_json 零尺寸", typeof jsonW1.computeSize === "function" && jsonW1.computeSize()[0] === 0);
check("presets_json 不绘制", typeof jsonW1.draw === "function");
check("presets_json 仍在数组中（可序列化）", n1.widgets.includes(jsonW1));
check("text_override 已隐藏（零尺寸/不绘制）", draftW1.hidden === true && draftW1.computeSize()[0] === 0);
check("同步首绘 combo 为工作流预设", JSON.stringify(presetW1.options.values) === '["A","B"]');
check("当前选中保留", presetW1.value === "A");
check("预览 widget 已添加且不序列化", displayW1 !== undefined && displayW1.serialize === false);
check("预览显示选中文本", displayW1.value === "hello");
check("选中预设即可编辑（残留项也产生草稿）", displayW1.inputEl.readOnly === false);
check("⚙ 预设按钮已添加", n1.buttons["⚙ 预设"] !== undefined);
check("💾 保存到预设按钮已添加", n1.buttons["💾 保存到预设"] !== undefined);
check("combo callback 已包装", typeof presetW1.callback === "function");
check("onAfterGraphConfigured 已包装", typeof n1.onAfterGraphConfigured === "function");

// 异步全局库加载后：全局优先 + 工作流残留补尾
await flush();
check("全局库加载后 combo 合并", JSON.stringify(presetW1.options.values) === '["G1","G2","A","B"]');
check("选中 A（工作流残留项）保留", presetW1.value === "A");
check("工作流残留项可编辑（草稿语义）", displayW1.inputEl.readOnly === false);

// ---------- 草稿语义：编辑写 text_override，不碰全局库与 presets_json ----------
presetW1.value = "G2";
presetW1.callback("G2");
check("切换全局预设预览同步", displayW1.value === "g2 text");
check("全局预设可编辑", displayW1.inputEl.readOnly === false);

displayW1.value = "edited g2";
displayW1.inputEl.trigger("input");
const callsBefore = apiCalls.length;
check("编辑写 text_override（不 POST 全局库）", draftW1.value === "edited g2" && apiCalls.length === callsBefore);
check("presets_json 未被修改", JSON.parse(jsonW1.value).find((p) => p.name === "B").text === "world");
check("编辑后显示保持", displayW1.value === "edited g2");

// 切换预设 = 丢弃草稿，显示新选预设原文
presetW1.value = "G1";
presetW1.callback("G1");
check("切换预设清空草稿", draftW1.value === "");
check("切换后显示新预设原文", displayW1.value === "g1 text");
presetW1.value = "G2";
presetW1.callback("G2");
check("切回显示预设原文（草稿已丢）", displayW1.value === "g2 text");

// 空选中仅当无任何预设时出现（有全局库时自动回落第一项）

// ---------- 💾 保存到预设（草稿 → 全局库） ----------
presetW1.value = "G1";
presetW1.callback("G1");
displayW1.value = "g1 edited";
displayW1.inputEl.trigger("input");
check("草稿已写 text_override", draftW1.value === "g1 edited");
await n1.buttons["💾 保存到预设"]();
const postCall = apiCalls.filter((c) => c.method === "POST").pop();
check("保存 POST 全局库", postCall && postCall.body.name === "G1" && postCall.body.text === "g1 edited");
check("保存后草稿清空", draftW1.value === "");
check("保存后全局库已更新", store.find((p) => p.name === "G1").text === "g1 edited");
check("保存后显示为库中文本", displayW1.value === "g1 edited");

// 残留项晋升：选中工作流残留项 A，编辑后保存 → 新增为全局预设
presetW1.value = "A";
presetW1.callback("A");
check("选中残留项显示原文", displayW1.value === "hello");
displayW1.value = "hello promoted";
displayW1.inputEl.trigger("input");
await n1.buttons["💾 保存到预设"]();
check("残留项保存晋升为全局预设", store.find((p) => p.name === "A")?.text === "hello promoted");
check("晋升后草稿清空", draftW1.value === "");
await flush();
check("晋升后 combo 仍含 A", presetW1.options.values.includes("A"));

// ---------- configure 保留选中值（工作流切换/重载回归） ----------
// 重置全局库，隔离前序用例的写入
store = [{ name: "G1", text: "g1 text" }, { name: "G2", text: "g2 text" }];
const n3 = mkNode("[]", "");
await flush();
n3.widgets.find((w) => w.name === "presets_json").value = '[{"name": "C", "text": "ccc"}]';
n3.onAfterGraphConfigured();
await flush();
const presetW3 = n3.widgets.find((w) => w.name === "preset");
check("configure 后 combo 合并全局与工作流", JSON.stringify(presetW3.options.values) === '["G1","G2","C"]');
check("configure 后选中回落第一项", presetW3.value === "G1");

// 选中全局预设（不在本工作流 presets_json）必须保留
const n3b = mkNode("[]", "");
await flush();
const presetW3b = n3b.widgets.find((w) => w.name === "preset");
presetW3b.value = "G2"; // 模拟 configure 恢复的选中值
n3b.onAfterGraphConfigured();
await flush();
check("configure 保留选中的全局预设", presetW3b.value === "G2");
check("保留选中后预览同步", n3b.widgets.find((w) => w.name === "content_display").value === "g2 text");

// configure 恢复草稿（工作流级草稿生命周期：configure 直接赋值不触发 callback → 草稿不清）
const n3c = mkNode("[]", "");
await flush();
n3c.widgets.find((w) => w.name === "preset").value = "G2"; // 模拟 configure 恢复
n3c.widgets.find((w) => w.name === "text_override").value = "draft survives reload";
n3c.onAfterGraphConfigured();
await flush();
check("configure 保留选中值", n3c.widgets.find((w) => w.name === "preset").value === "G2");
check("草稿随工作流恢复且优先显示", n3c.widgets.find((w) => w.name === "content_display").value === "draft survives reload");

// ---------- API 失败降级：combo 为工作流数据源，编辑仍走草稿 ----------
apiFail = true;
const n4 = mkNode('[{"name": "A", "text": "hello"}]', "A");
await flush();
const presetW4 = n4.widgets.find((w) => w.name === "preset");
const jsonW4 = n4.widgets.find((w) => w.name === "presets_json");
const displayW4 = n4.widgets.find((w) => w.name === "content_display");
check("降级 combo 为工作流预设", JSON.stringify(presetW4.options.values) === '["A"]');
check("降级时预览同步", displayW4.value === "hello");
check("降级时有选中即可编辑", displayW4.inputEl.readOnly === false);
const callsBefore2 = apiCalls.length;
displayW4.value = "edited A";
displayW4.inputEl.trigger("input");
check("降级编辑写 text_override", n4.widgets.find((w) => w.name === "text_override").value === "edited A");
check("降级编辑不调 API", apiCalls.length === callsBefore2);
check("降级编辑不动 presets_json", JSON.parse(jsonW4.value).find((p) => p.name === "A").text === "hello");
// 降级保存失败 → 提示且草稿保留
const alertsBefore = alerts.length;
await n4.buttons["💾 保存到预设"]();
check("降级保存失败有提示且草稿保留", alerts.length > alertsBefore
    && n4.widgets.find((w) => w.name === "text_override").value === "edited A");
apiFail = false;

// ---------- 管理弹窗（编辑全局库） ----------
const n5 = mkNode('[{"name": "A", "text": "hello"}]', "A");
await flush();
const mgrCallback = n5.buttons["⚙ 预设"];
const presetW5 = n5.widgets.find((w) => w.name === "preset");
const displayW5 = n5.widgets.find((w) => w.name === "content_display");

// 双击重入防护：加载期间二次点击只创建一个弹窗
const p1 = mgrCallback();
const p2 = mgrCallback();
await p1;
await p2;
const overlays = createdEls.filter((e) => e.className === "sf-preset-mgr-overlay");
check("双击只创建一个弹窗", overlays.length === 1 && !overlays[0].removed);
const overlay = overlays[0];
const panel = overlay.children[0];
const body = panel.children[1];
const list = body.children[0];
check("列表渲染全局库项", list.children.length === store.length);
check("列表项含名称与摘要", list.children[0].children[0].textContent === "G1"
    && list.children[0].children[1].textContent === "g1 text");
const editor = body.children[1];
const nameInput = editor.children[1];
const textArea = editor.children[3];
check("初始载入第一个预设", nameInput.value === "G1" && textArea.value === "g1 text");

// 新增
nameInput.value = "C";
textArea.value = "new text";
editor.children[4].children[0].click();
await flush();
check("新增后列表 3 项", list.children.length === 3);
check("新增写全局库", store.find((p) => p.name === "C")?.text === "new text");
check("combo 同步新预设", JSON.stringify(presetW5.options.values).includes('"C"'));
check("新增后选中新项", list.children[2].className.includes("active"));

// 重名阻止
const storeLenBefore = store.length;
nameInput.value = "G1";
textArea.value = "dup";
editor.children[4].children[0].click();
await flush();
check("重名新增被阻止", store.length === storeLenBefore);

// 更新（改名 → POST 新名 + DELETE 旧名）
list.children[1].click();
check("点击选中载入编辑区", nameInput.value === "G2" && textArea.value === "g2 text");
nameInput.value = "G2x";
textArea.value = "updated";
editor.children[4].children[1].click();
await flush();
check("改名 POST 新名", store.find((p) => p.name === "G2x")?.text === "updated");
check("改名 DELETE 旧名", store.find((p) => p.name === "G2") === undefined);
check("combo 选项更新", JSON.stringify(presetW5.options.values).includes('"G2x"'));
check("其他节点同步刷新", n1.widgets.find((w) => w.name === "preset").options.values.includes("G2x"));

// 重名更新阻止
nameInput.value = "G1";
textArea.value = "x";
editor.children[4].children[1].click();
await flush();
check("重名更新被阻止", store.find((p) => p.name === "G2x")?.text === "updated");

// 删除
list.children[1].click();
editor.children[4].children[2].click();
await flush();
check("删除写全局库", store.find((p) => p.name === "G2x") === undefined);
check("删除后列表 2 项", list.children.length === 2);
check("combo 同步删除", !JSON.stringify(presetW5.options.values).includes('"G2x"'));
check("预览同步", displayW5.value === "hello");

// Escape 关闭
docHandlers.keydown({ key: "Escape" });
check("Escape 关闭弹窗", overlay.removed === true);

// 空预设列表的弹窗
store = [];
const n6 = mkNode("[]", "");
await flush();
await n6.buttons["⚙ 预设"]();
const overlay6 = createdEls.find((e) => e.className === "sf-preset-mgr-overlay" && e !== overlay);
check("空列表弹窗渲染空提示", overlay6 !== undefined
    && overlay6.children[0].children[1].children[0].children[0].className.includes("sf-preset-mgr-empty"));
docHandlers.keydown({ key: "Escape" });
// 无任何预设时 combo 空占位、编辑框只读
check("无预设时 combo 空占位", JSON.stringify(n6.widgets.find((w) => w.name === "preset").options.values) === '[""]');
check("无预设时编辑框只读", n6.widgets.find((w) => w.name === "content_display").inputEl.readOnly === true);

console.log("\nFAILURES:", failures.length);
process.exit(failures.length ? 1 : 0);
})();
