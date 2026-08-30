// SFLoraStack 预设菜单端到端冒烟测试（Node 直接运行：node tests/test_lora_stack_presets_smoke.js）
// mock DOM/app/api/fetch 真实加载 interaction 模块，验证：
//   - Presets 按钮点击 -> 菜单渲染预设列表（来自 /api/sfnodes/lora_presets）
//   - 点击预设项 -> 确认框 -> Load -> 行写入 node.properties + refresh(true)
//   - 确认框 Cancel -> 状态不变
//   - 保存命名 -> POST 预设 -> 菜单刷新显示新名
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ── mock DOM（惰性元素 + 事件记录/分发；textContent 赋值清空子节点；
//    className 与 classList 双向同步——真实 DOM 语义，行 i 按钮的
//    classList.toggle 高亮依赖它）──
function makeEl() {
    const el = {
        style: { setProperty() {}, getPropertyValue() { return ""; } },
        dataset: {}, children: [], listeners: {}, _cls: new Set(),
        value: "", placeholder: "", type: "", title: "", rows: 1,
        disabled: false, isConnected: true, offsetWidth: 100, offsetHeight: 20,
        selectionStart: 0, selectionEnd: 0, _text: "",
        classList: {
            add(...c) { c.forEach((x) => el._cls.add(x)); sync(); },
            remove(...c) { c.forEach((x) => el._cls.delete(x)); sync(); },
            toggle(c, force) {
                if (force === undefined) { el._cls.has(c) ? el._cls.delete(c) : el._cls.add(c); }
                else { force ? el._cls.add(c) : el._cls.delete(c); }
                sync();
            },
            contains(c) { return el._cls.has(c); },
        },
        append(...kids) { this.children.push(...kids); },
        appendChild(c) { this.children.push(c); return c; },
        prepend(...kids) { this.children.unshift(...kids); },
        replaceWith() {}, replaceChildren(...kids) { this.children = kids; },
        remove() { this.removed = true; },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
        removeEventListener(type, fn) {
            const a = this.listeners[type];
            if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }
        },
        emit(type, evt) {
            const e = evt || { target: this, clientX: 10, clientY: 10 };
            e.stopPropagation ||= () => {};
            for (const fn of [...(this.listeners[type] || [])]) fn(e);
        },
        focus() {}, blur() {}, select() {}, click() { this.emit("click", { target: this, clientX: 10, clientY: 10 }); },
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
    };
    function sync() {
        el.className = [...el._cls].join(" ");
    }
    Object.defineProperty(el, "className", {
        get() { return [...el._cls].join(" "); },
        set(v) { el._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    });
    Object.defineProperty(el, "textContent", {
        get() { return el._text; },
        set(v) { el._text = v; el.children = []; },
    });
    // 真实 DOM：innerHTML = "" 清空所有子节点。renderNode/信息面板以此
    // 重建内容——mock 不实现会导致旧渲染残留、断言读到过期 DOM。
    Object.defineProperty(el, "innerHTML", {
        get() { return ""; },
        set() { el.children = []; },
    });
    return el;
}
const bodyChildren = [];
const headStyles = [];
globalThis.document = {
    createElement() { return makeEl(); },
    body: { appendChild(c) { bodyChildren.push(c); }, contains() { return false; } },
    head: { appendChild(c) { headStyles.push(c); } },
    addEventListener() {}, removeEventListener() {},
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    app: null,
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
};
globalThis.getComputedStyle = () => ({});
globalThis.api = { apiURL: (r) => r };

// ── app mock（registerExtension 捕获扩展）──
globalThis.app = {
    graph: { _nodes: [] },
    canvas: { setDirty() {} },
    ui: { settings: { getSettingValue: () => null, setSettingValueAsync: async () => {} } },
    registerExtension() {},
};
globalThis.window.app = globalThis.app;

// ── fetch mock：lora_presets 内存存储（POST 保存 / DELETE 删除 / GET 列表）──
const serverPresets = {
    "power-style": {
        normalize: true, normalize_weight: 1.0, separate: false,
        loras: [
            { lora: "dirA/x.safetensors", on: true, strength: 0.9, strengthTwo: 0.7 },
            { lora: "dirB/y.safetensors", on: false, strength: 1.5 },
        ],
    },
};
let lastPost = null;
globalThis.fetch = async (url, opts) => {
    const u = String(url);
    opts = opts || {};
    if (u.endsWith("/api/sfnodes/lora_presets") && (!opts.method || opts.method === "GET")) {
        return { ok: true, status: 200, json: async () => ({ presets: serverPresets }) };
    }
    if (u.endsWith("/api/sfnodes/lora_presets") && opts.method === "POST") {
        lastPost = JSON.parse(opts.body);
        serverPresets[lastPost.name] = lastPost.data;
        return { ok: true, status: 200, json: async () => ({ ok: true, name: lastPost.name }) };
    }
    if (u.includes("/api/sfnodes/lora_presets?name=") && opts.method === "DELETE") {
        const nm = decodeURIComponent(u.split("name=")[1]);
        delete serverPresets[nm];
        return { ok: true, status: 200, json: async () => ({ deleted: nm }) };
    }
    // 行 i 按钮 _has_custom 高亮判定（lora_notes 网关）：x.safetensors 有
    // 用户信息（_has_custom），y 无
    if (u.includes("/api/sfnodes/lora_notes?filename=")) {
        const nm = decodeURIComponent(u.split("filename=")[1]);
        return { ok: true, status: 200, json: async () => ({
            trigger_words: "", description: "", _has_custom: nm === "dirA/x.safetensors",
        }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
};

// ── 加载模块：/scripts/app|api.js -> globalThis；相对 import 改 .mjs ──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_lsp_"));
for (const n of ["sf_lora_stack_core.js", "sf_lora_stack_api.js",
    "sf_lora_stack_dropdown.js", "sf_lora_stack_info.js",
    "sf_lora_stack_settings.js", "sf_common.js", "sf_markdown.js",
    "sf_lora_shared_info.js", "sf_lora_info.js", "sf_lora_stack_render.js", "sf_lora_stack_interaction.js",
    "sf_workflows_ui.js", "sf_workflows_lib.js"]) {
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), code);
}

function hasClass(el, cls) {
    const want = cls.split(" ").filter(Boolean);
    const have = String(el.className || "").split(" ").filter(Boolean);
    return want.every((w) => have.includes(w));
}
function findByClass(root, cls) {
    if (!root) return null;
    for (const c of root.children || []) {
        if (hasClass(c, cls)) return c;
        const f = findByClass(c, cls);
        if (f) return f;
    }
    return null;
}
function findByText(root, text) {
    if (!root) return null;
    if (root._text === text) return root;
    for (const c of root.children || []) {
        const f = findByText(c, text);
        if (f) return f;
    }
    return null;
}
// 菜单项：label 的 click 在真实 DOM 冒泡到父 .it（监听所在）；mock 不冒泡，
// 因此直接找 .it 容器在其上触发。
function findItem(menu, text) {
    for (const c of menu.children || []) {
        if (String(c.className || "").includes("it") && findByText(c, text)) return c;
    }
    return null;
}
const menuChildren = () => bodyChildren.filter((c) => c.removed !== true);

(async () => {
    const I = await import(path.join(tmpDir, "sf_lora_stack_interaction.mjs"));
    const R = await import(path.join(tmpDir, "sf_lora_stack_render.mjs"));
    check("interaction 模块加载", typeof I.attachInteractions === "function");
    check("render 模块加载", typeof R.renderNode === "function");

    // FakeNode：预置两行（联动关闭，sm/sc 独立）。真实 renderNode 需要
    // _sfLsRoot（惰性元素，isConnected 恒 true）+ 隐藏 widget。
    const lorasRoot = makeEl();
    lorasRoot.className = "sf-ls-root";
    // mock 的 querySelector 惰性返回新元素会让 renderNode 拿到的 inner 与
    // root 失联（真实 DOM 中 inner 已挂载）。模拟真实结构：querySelector
    // 返回已挂载的 inner，appendChild 记录挂载。
    let innerEl = null;
    lorasRoot.querySelector = (sel) => {
        if (sel === ".sf-ls-inner") return innerEl;
        return makeEl();
    };
    lorasRoot.appendChild = function (c) {
        this.children.push(c);
        if (String(c.className || "").includes("sf-ls-inner")) innerEl = c;
        return c;
    };
    const node = {
        id: 1, comfyClass: "SFLoraStack", type: "SFLoraStack",
        properties: {
            loraStackState: JSON.stringify({
                version: 1, sep: ", ", step: 0.05, defStrength: 1.0,
                linkStrength: false, civitai: true, thumbs: true, hideExt: true, cacheMode: "last",
                loras: [
                    { id: "l1", name: "a.safetensors", on: true, sm: 1.2, sc: 0.8, triggers: ["t1"], custom: [] },
                    { id: "l2", name: "b.safetensors", on: false, sm: 0.5, sc: 2.0, triggers: [], custom: [] },
                ],
            }),
        },
        widgets: [{ name: "loras_ui", element: lorasRoot }],
        size: [336, 0],
        setDirtyCanvas() {}, computeSize() { return [336, 100]; }, setSize() {},
    };
    node._sfLsRoot = lorasRoot;

    const refreshCalls = [];
    const widgetEl = makeEl();
    // 真实 refresh 链路：renderNode（真实渲染）+ fitToContent 简化（高度数学
    // 由 core/render 覆盖，此处仅记录）
    I.attachInteractions(node, widgetEl, (structural) => {
        R.renderNode(node);
        refreshCalls.push(!!structural);
    });
    R.renderNode(node);
    check("初始渲染 2 行", node._sfLsRoot.children[0].children.length === 2);

    const readLoras = () => JSON.parse(node.properties.loraStackState).loras;

    // ── 打开预设菜单 ──
    const presetsBtn = makeEl();
    presetsBtn.className = "sf-ls-presets";
    presetsBtn.dataset.act = "presets";
    presetsBtn.closest = () => presetsBtn;
    widgetEl.emit("click", { target: presetsBtn, clientX: 40, clientY: 60 });
    await tick(); await tick();
    const menu = menuChildren().find((c) => c.className === "sf-ls-menu");
    check("预设菜单已打开", !!menu && menu._text !== "Loading presets…");
    check("菜单含 Save 项", !!findByText(menu, "Save current as preset…"));
    const powerItem = findItem(menu, "power-style");
    check("菜单列出预设", !!powerItem);

    // ── 载入预设：点预设名 -> 确认框 -> Load ──
    const beforeState = node.properties.loraStackState;
    powerItem.emit("click", { target: powerItem });
    await tick(); await tick();
    const mask = bodyChildren[bodyChildren.length - 1];
    check("确认框出现", mask.className === "sf-ls-confirm-mask");
    check("确认框标题", findByClass(mask, "sf-ls-confirm-t")._text === "Load preset?");
    check("确认框注入 CSS（不依赖信息面板已打开）", headStyles.some((s) => s.id === "sf-ls-info-css"));
    const loadBtn = findByClass(mask, "b pri");
    check("确认框 Load 按钮", !!loadBtn);
    loadBtn.emit("click", { target: loadBtn });
    await tick(); await tick();

    const loras = readLoras();
    check("载入后行数 = 2", loras.length === 2);
    check("载入后名称正确", loras[0].name === "dirA/x.safetensors" && loras[1].name === "dirB/y.safetensors");
    check("载入后强度正确", loras[0].sm === 0.9 && loras[0].sc === 0.7 && loras[1].sm === 1.5 && loras[1].sc === 1.5);
    check("载入后开关正确", loras[0].on === true && loras[1].on === false);
    check("载入后触发词清空", loras[0].triggers.length === 0);
    check("载入后新 id", loras[0].id !== "l1" && loras[0].id !== "l2");
    check("refresh(true) 已调用", refreshCalls[refreshCalls.length - 1] === true);
    check("状态已变化", node.properties.loraStackState !== beforeState);
    // 真实 renderNode 渲染新行：root > inner(.sf-ls-inner) > [band, rows]，
    // rows 下 2 个 .sf-ls-row
    const rowsWrap = node._sfLsRoot.children[0].children[1];
    check("界面渲染 2 行（真实 renderNode）", rowsWrap.children.length === 2
        && rowsWrap.children.every((r) => String(r.className).includes("sf-ls-row")));
    const nmEl = rowsWrap.children[0].children[1].children[0];
    check("行名渲染正确（hideExt 剥扩展名）", rowsWrap.children[0].children[1].children[0]._text === "x");

    // ── 行 i 按钮 _has_custom 高亮（lora_notes 网关判定，与 Power 系同源）──
    // 行结构（linkStrength=false 分离 model/clip）：[grip, name, wm, wm(c), info, sw]
    await tick(); // getLoraMetadata promise 落地
    const infoX = rowsWrap.children[0].children[4];
    const infoY = rowsWrap.children[1].children[4];
    check("有自定义信息的行 i 高亮", String(infoX.className).split(/\s+/).includes("net"));
    check("无自定义信息的行不高亮", !String(infoY.className).split(/\s+/).includes("net"));

    // ── Cancel 路径：再开菜单 -> 点预设 -> Cancel -> 状态不变 ──
    widgetEl.emit("click", { target: presetsBtn, clientX: 40, clientY: 60 });
    await tick(); await tick();
    const menu2 = menuChildren().find((c) => c.className === "sf-ls-menu");
    const powerItem2 = findItem(menu2, "power-style");
    const before2 = node.properties.loraStackState;
    powerItem2.emit("click", { target: powerItem2 });
    await tick(); await tick();
    const mask2 = bodyChildren[bodyChildren.length - 1];
    findByClass(mask2, "b gh").emit("click", { target: findByClass(mask2, "b gh") });
    await tick();
    check("Cancel 后状态不变", node.properties.loraStackState === before2);

    // ── 保存预设：菜单 -> Save current as preset… -> 输入名 -> Save ──
    widgetEl.emit("click", { target: presetsBtn, clientX: 40, clientY: 60 });
    await tick(); await tick();
    const menu3 = menuChildren().find((c) => c.className === "sf-ls-menu");
    findItem(menu3, "Save current as preset…").emit("click", { target: findItem(menu3, "Save current as preset…") });
    await tick();
    const inp = menu3.children[0].children[0];   // .in > input
    inp.value = "my-stack";
    findByClass(menu3, "ok pri").emit("click", { target: findByClass(menu3, "ok pri") });
    await tick(); await tick();
    check("POST 已发送", lastPost && lastPost.name === "my-stack");
    check("POST 数据形状兼容 Power", lastPost && lastPost.data.loras[0].lora === "dirA/x.safetensors"
        && lastPost.data.loras[0].strength === 0.9 && lastPost.data.loras[0].strengthTwo === 0.7);
    check("保存后菜单列出新预设", !!findByText(menu3, "my-stack"));

    // ── 删除预设：hover ✕ 点击 -> 二次确认 -> 删除 ──
    const myItem = findItem(menu3, "my-stack");
    const delBtn = findByClass(myItem, "del");
    check("预设项有删除 ✕", !!delBtn);
    delBtn.emit("click", { target: delBtn });
    await tick(); await tick();
    const delMask = bodyChildren[bodyChildren.length - 1];
    check("删除确认框出现", delMask.className === "sf-ls-confirm-mask");
    const delOk = findByClass(delMask, "b pri");
    check("删除确认框 Delete 按钮", !!delOk);
    delOk.emit("click", { target: delOk });
    await tick(); await tick();
    check("DELETE 后列表移除", !findByText(menu3, "my-stack") && !serverPresets["my-stack"]);
    // 取消路径：再次保存后点击 ✕ -> Cancel 不删除
    // 重新保存一个用于取消测试
    // 保存 my-stack-2
    findItem(menu3, "Save current as preset…").emit("click", { target: findItem(menu3, "Save current as preset…") });
    await tick();
    const inp2 = menu3.children[0].children[0];
    inp2.value = "my-stack-2";
    findByClass(menu3, "ok pri").emit("click", { target: findByClass(menu3, "ok pri") });
    await tick(); await tick();
    const myItem2 = findItem(menu3, "my-stack-2");
    const delBtn2 = findByClass(myItem2, "del");
    delBtn2.emit("click", { target: delBtn2 });
    await tick(); await tick();
    const delMask2 = bodyChildren[bodyChildren.length - 1];
    findByClass(delMask2, "b gh").emit("click", { target: findByClass(delMask2, "b gh") });
    await tick();
    check("Cancel 后保留", !!findByText(menu3, "my-stack-2") && !!serverPresets["my-stack-2"]);
    // 清理
    delete serverPresets["my-stack-2"];

    // ── preset 输入连接（SF_LORA_PRESET）：自动加载预设到行 ──
    const upstream = {
        id: 99, comfyClass: "SFLoraPreset", type: "SFLoraPreset",
        widgets: [{ name: "preset", value: "power-style", callback: null }],
    };
    const stack2 = {
        id: 3, comfyClass: "SFLoraStack", type: "SFLoraStack",
        properties: {
            loraStackState: JSON.stringify({
                version: 1, sep: ", ", linkStrength: false,
                loras: [{ id: "old", name: "old.safetensors", on: true, sm: 1, sc: 1, triggers: [], custom: [] }],
            }),
        },
        inputs: [{ name: "model", link: null }, { name: "clip", link: null }, { name: "preset", link: 7 }],
        graph: {
            links: { 7: { origin_id: 99 } },
            getNodeById(id) { return id === 99 ? upstream : null; },
        },
        widgets: [], size: [336, 0],
        setDirtyCanvas() {}, computeSize() { return [336, 100]; }, setSize() {},
    };
    const refresh2 = [];
    const loaded2 = await I.loadPresetInto(stack2, (structural) => refresh2.push(!!structural));
    const rows2 = JSON.parse(stack2.properties.loraStackState).loras;
    check("loadPresetInto 加载", loaded2 === true && rows2.length === 2);
    check("loadPresetInto 行内容", rows2[0].name === "dirA/x.safetensors"
        && rows2[0].sm === 0.9 && rows2[0].sc === 0.7 && rows2[1].on === false);
    check("loadPresetInto refresh(true)", refresh2[refresh2.length - 1] === true);
    check("loadPresetInto 未连接返回 false", (await I.loadPresetInto({ inputs: [] }, null)) === false);

    // 上游切换预设名 -> callback 包装 -> 自动重载
    const refresh3 = [];
    I.watchPresetUpstream(stack2, (structural) => refresh3.push(!!structural));
    check("watch 包装一次（幂等）", upstream.widgets[0]._sfLsPresetWatched === true);
    serverPresets["two"] = { loras: [{ lora: "d.safetensors", on: true, strength: 0.4, strengthTwo: 0.3 }] };
    upstream.widgets[0].value = "two";
    upstream.widgets[0].callback("two");
    await tick();
    const rows3 = JSON.parse(stack2.properties.loraStackState).loras;
    check("上游切换自动重载", rows3.length === 1 && rows3[0].name === "d.safetensors" && rows3[0].sm === 0.4);
    check("上游切换 refresh(true)", refresh3[refresh3.length - 1] === true);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    process.exit(1);
});
