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
        站立肖像: { description: "Standing portrait", group: "SFW" },
        床上自慰: { description: "Masturbating", group: "NSFW" },
        侧卧: { description: "Side lying", group: "SFW" },
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
    check("11 个分类 tab", tabs.children.length === 11);
    check("表情 tab 存在", tabs.children.some((c) => c.textContent === "表情"));
    const poseTab = tabs.children.find((c) => c.textContent === "单人动作");
    check("tab 内容正确", poseTab !== undefined);

    // group 筛选行（默认 Celebrity tab）
    const groupBar = panel.children[2];
    check("group 筛选行渲染", groupBar.children.length === 4); // 全随机/全部/歌手/亚洲名人
    check("全随机 chip 存在", groupBar.children[0].textContent === "🎲 全随机");
    check("默认选中全部", groupBar.children[1].className.includes("active"));

    // 默认 Celebrity tab：分组标题 + 选项
    const list = panel.children[4];
    const groups = list.children.filter((c) => c.children?.[0]?.textContent === "歌手" || c.children?.[0]?.textContent === "亚洲名人");
    check("Celebrity 分组标题渲染", groups.length === 2);
    check("分组标题含随机按钮", groups[0].children.length === 2 && groups[0].children[1].textContent.includes("随机"));
    const celebItems = list.children.filter((c) => c.textContent === "Taylor Swift" || c.textContent === "周杰伦");
    check("Celebrity 选项渲染", celebItems.length === 2);

    // 选项 hover 显示 description 预览
    const preview = panel.children[5];
    check("预览条存在", preview.className.includes("sf-preset-picker-preview"));
    const tsItem = celebItems.find((c) => c.textContent === "Taylor Swift");
    tsItem.trigger("mouseenter");
    check("hover 显示 description", preview.textContent === "American singer-songwriter");
    tsItem.trigger("mouseleave");
    check("mouseleave 清空预览", preview.textContent === "");

    // group 筛选：点"歌手"只显示歌手
    const singerChip = groupBar.children.find((c) => c.textContent === "歌手");
    singerChip.click();
    const singerList = panel.children[4];
    check("group 筛选只显示歌手", singerList.children.some((c) => c.textContent === "Taylor Swift")
        && !singerList.children.some((c) => c.textContent === "周杰伦"));

    // 切换到 单人动作 tab → SFW/NSFW 分组 + group 筛选重置
    poseTab.click();
    const poseGroupBar = panel.children[2];
    check("tab 切换 group 重置为全部", poseGroupBar.children[1].className.includes("active"));
    const poseList = panel.children[4];
    const poseGroups = poseList.children.filter((c) => c.children?.[0]?.textContent === "SFW" || c.children?.[0]?.textContent === "NSFW");
    check("Pose SFW/NSFW 分组（交错数据仅 2 个标题）", poseGroups.length === 2);
    const sfwTitleIdx = poseList.children.findIndex((c) => c.children?.[0]?.textContent === "SFW");
    const nsfwTitleIdx = poseList.children.findIndex((c) => c.children?.[0]?.textContent === "NSFW");
    const sfwBlock = poseList.children.slice(sfwTitleIdx + 1, nsfwTitleIdx).filter((c) => c.className.includes("sf-preset-picker-item"));
    const nsfwBlock = poseList.children.slice(nsfwTitleIdx + 1).filter((c) => c.className.includes("sf-preset-picker-item"));
    check("SFW 组内选项完整", sfwBlock.map((c) => c.textContent).join(",") === "回眸,站立肖像,侧卧");
    check("NSFW 组内选项完整", nsfwBlock.map((c) => c.textContent).join(",") === "裸卧张开腿,床上自慰");

    // NSFW 筛选
    const nsfwChip = poseGroupBar.children.find((c) => c.textContent === "NSFW");
    nsfwChip.click();
    const nsfwList = panel.children[4];
    check("NSFW 筛选只显示 NSFW", nsfwList.children.some((c) => c.textContent === "裸卧张开腿")
        && !nsfwList.children.some((c) => c.textContent === "回眸"));

    // 搜索过滤（在 NSFW 筛选基础上再搜索）
    const search = panel.children[3];
    search.value = "裸卧";
    search.trigger("input");
    const listAfterSearch = panel.children[4];
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

    // 组标题 🎲 随机按钮写入 随机·组名
    const before3 = createdEls.length;
    pickerCallback();
    const overlay3 = createdEls.slice(before3).find((e) => e.className === "sf-preset-picker-overlay");
    const panel3 = overlay3.children[0];
    panel3.children[1].children.find((c) => c.textContent === "单人动作").click();
    const poseList3 = panel3.children[4];
    const nsfwTitle3 = poseList3.children.find((c) => c.children?.[0]?.textContent === "NSFW");
    nsfwTitle3.children[1].click();
    check("🎲 随机按钮写入组随机值", pose0.value === "随机·NSFW");
    check("🎲 后弹窗关闭", overlay3.removed === true);

    // 全随机 chip：当前分类整体随机
    const before4 = createdEls.length;
    pickerCallback();
    const overlay4 = createdEls.slice(before4).find((e) => e.className === "sf-preset-picker-overlay");
    const panel4 = overlay4.children[0];
    panel4.children[1].children.find((c) => c.textContent === "单人动作").click();
    const randomChip4 = panel4.children[2].children[0];
    check("全随机 chip 位于筛选行首", randomChip4.textContent === "🎲 全随机");
    randomChip4.click();
    check("全随机写入当前分类 widget", pose0.value === "随机");
    check("全随机后弹窗关闭", overlay4.removed === true);

    // 弹窗记忆：上次 tab 为 Pose（关闭时已保存），重新打开应保持
    const before5 = createdEls.length;
    pickerCallback();
    const overlay5 = createdEls.slice(before5).find((e) => e.className === "sf-preset-picker-overlay");
    const panel5 = overlay5.children[0];
    const tabs5 = panel5.children[1];
    const activeTab5 = tabs5.children.find((c) => c.className.includes("active"));
    check("弹窗记忆上次 tab", activeTab5.textContent === "单人动作");
    // 组筛选记忆：Pose tab 下选 NSFW 后关闭，再开应保持 NSFW
    const poseGroupBar5 = panel5.children[2];
    poseGroupBar5.children.find((c) => c.textContent === "NSFW").click();
    docHandlers.keydown({ key: "Escape" });
    const before6 = createdEls.length;
    pickerCallback();
    const overlay6 = createdEls.slice(before6).find((e) => e.className === "sf-preset-picker-overlay");
    const panel6 = overlay6.children[0];
    const groupBar6 = panel6.children[2];
    check("弹窗记忆 group 筛选", groupBar6.children.find((c) => c.textContent === "NSFW").className.includes("active"));
    docHandlers.keydown({ key: "Escape" });

    console.log("\nFAILURES:", failures.length);
    process.exit(failures.length ? 1 : 0);
})();
