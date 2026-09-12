// SF Ignore Groups 前端装配测试（Node 直接运行：node tests/test_ignore_groups_js.js）
// 覆盖（Function-eval 注入 lib 真源，test_boolean_switch_js.js 同款）：
// - 扩展注册 sfnodes.IgnoreGroups；非本类节点不受影响
// - nodeCreated：properties 惰性初始化（全开组进 activeSet，全关组排除）/
//   DOM widget 装配 / 行渲染与开关态 / computeSize 高度随组数
// - 行点击切换：关组恢复 mode=0，开组旁路 mode=4（默认 disable=false）
// - 右键菜单打开设置弹窗；overlay 点击关闭弹窗
// - graph.change 全局只包装一次（双节点共享）；onRemoved 清定时器/监听
const fs = require("fs");
const path = require("path");

// import 剥离须行首锚定（注释含 "import xxx.js" 字样，非锚定会吞代码，见 patterns §52）
const stripImports = (f) =>
    fs.readFileSync(path.join(__dirname, "..", "web", f), "utf8")
        .replace(/^import[^;]+;/gm, "")
        .replace(/export\s+(?=function|const|let|class|var)/g, "");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ---- DOM stubs ----
function makeEl(tag, cls, text) {
    const e = {
        tag, className: cls || "", textContent: text || "", value: "", type: "",
        checked: false, name: "", title: "", placeholder: "",
        children: [], _parent: null, _handlers: {}, _html: "",
        style: {},
        setAttribute() {},
        addEventListener(t, fn) { (this._handlers[t] = this._handlers[t] || []).push(fn); },
        removeEventListener() {},
        appendChild(c) { c._parent = this; this.children.push(c); return c; },
        append(...cs) { for (const c of cs) { c._parent = this; this.children.push(c); } },
        remove() {
            if (this._parent) {
                const i = this._parent.children.indexOf(this);
                if (i >= 0) this._parent.children.splice(i, 1);
                this._parent = null;
            }
            this._removed = true;
        },
        focus() {}, select() {}, click() {},
        contains(t) {
            if (t === this) return true;
            return (this.children || []).some((c) => c.contains ? c.contains(t) : c === t);
        },
    };
    Object.defineProperty(e, "innerHTML", {
        get() { return this._html; },
        set(v) { this._html = v; this.children = []; },
    });
    return e;
}
const el = (t, c, x) => makeEl(t, c, x);
const bodyStub = {
    children: [],
    appendChild(c) { c._parent = this; this.children.push(c); return c; },
};
const headStub = { children: [], appendChild(c) { this.children.push(c); return c; } };
const docListeners = { keydown: 0, mousedown: 0, visibilitychange: 0 };
const docHandlers = {};
globalThis.document = {
    createElement: (t) => makeEl(t),
    createTextNode: (t) => ({ text: t }),
    querySelectorAll: () => [],
    getElementById: () => null,
    body: bodyStub,
    head: headStub,
    hidden: false,
    addEventListener(t, fn) {
        docListeners[t] = (docListeners[t] || 0) + 1;
        (docHandlers[t] = docHandlers[t] || []).push(fn);
    },
    removeEventListener(t, fn) {
        docListeners[t] = Math.max(0, (docListeners[t] || 1) - 1);
        if (fn && docHandlers[t]) {
            const i = docHandlers[t].indexOf(fn);
            if (i >= 0) docHandlers[t].splice(i, 1);
        }
    },
};
// 在 document 上派发事件（测 sf_popup 真实关闭逻辑用）
function fireDocument(type, event) {
    for (const fn of (docHandlers[type] || []).slice()) fn(event);
}
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;

// ---- app/graph stubs ----
let ext = null;
let graphChangeCalls = 0;
const fakeGraph = {
    _groups: [
        { title: "A组", bounding: [0, 0, 200, 200], color: "#ff0000" },
        { title: "B组", bounding: [0, 300, 200, 200] },
        { title: "空组", bounding: [600, 0, 100, 100] },
    ],
    _nodes: [],
    change() { graphChangeCalls++; },
};
const memberA = { pos: [10, 10], size: [50, 50], mode: 0, flags: {} };
const memberB = { pos: [10, 310], size: [50, 50], mode: 4, flags: {} };
fakeGraph._nodes.push(memberA, memberB);
const app = {
    canvas: { ds: { scale: 1 }, setDirty: () => {} },
    graph: fakeGraph,
    registerExtension: (e) => { ext = e; },
};
const injectCSSOnce = () => {};
const installWheelZoomPassthrough = () => () => {};
// 用 sf_popup 真源（无 app 依赖）：锁定 overlay/pop 同级挂载时的 exempt 豁免逻辑；
// clampToViewport 真源需 window，这里 stub（不在测试范围内）
const popupSrc = stripImports("sf_popup.js");
const realPopup = new Function("document", popupSrc + "\nreturn { attachPopupDismiss };")(
    globalThis.document
);
const attachPopupDismiss = realPopup.attachPopupDismiss;
const clampToViewport = () => {};

const combined =
    stripImports("sf_ignore_groups_lib.js") + "\n" +
    stripImports("sf_ignore_groups.js");
new Function(
    "app", "el", "injectCSSOnce", "installWheelZoomPassthrough",
    "attachPopupDismiss", "clampToViewport", "document",
    combined
)(app, el, injectCSSOnce, installWheelZoomPassthrough,
    attachPopupDismiss, clampToViewport, globalThis.document);

check("扩展已注册 sfnodes.IgnoreGroups", ext && ext.name === "sfnodes.IgnoreGroups");

function makeNode() {
    const node = {
        comfyClass: "SFIgnoreGroups",
        title: "SF Ignore Groups",
        widgets: [],
        outputs: [],
        properties: {},
        size: [400, 100],
        graph: fakeGraph,
        setDirtyCanvas() { this._dirty = (this._dirty || 0) + 1; },
        addWidget(t, nm, v, cb, o) {
            const w = { name: nm, type: t, value: v, callback: cb, options: o || {} };
            this.widgets.push(w);
            return w;
        },
        addDOMWidget() { this._domWidget = {}; return this._domWidget; },
        configure() {},
    };
    fakeGraph._nodes.push(node);
    return node;
}
function rowsOf(node) {
    const root = node._domWidget && node._domWidget.el;
    if (!root) return [];
    const out = [];
    (function walk(e) {
        if (e.className === "sf-ig-row") out.push(e);
        for (const c of e.children || []) walk(c);
    })(root);
    return out;
}
function rowTitle(row) {
    return row.children[0] ? row.children[0].textContent : "";
}
function fireMouseDown(elm) {
    for (const fn of elm._handlers.mousedown || []) {
        fn({ preventDefault() {}, stopPropagation() {}, clientX: 0, clientY: 0 });
    }
}

// ---- 1. nodeCreated ----
// 劫持 addDOMWidget 捕获 root（addDOMWidget 第三个参数即 root 容器）
const n = makeNode();
const _addDOM = n.addDOMWidget.bind(n);
n.addDOMWidget = function (name, type, root, opts) {
    const w = _addDOM(name, type, root, opts);
    w.el = root;
    return w;
};
ext.nodeCreated(n);
check("DOM widget 已装配", !!n._domWidget);
// 空组按原版语义算全开（gs!==false），惰性初始化一并进集（应用时只走可见列表，无害）
check("activeSet 惰性初始化（A组进/B组出）", (() => {
    const s = n.properties.sf_ig_active_set || [];
    return s.includes("A组") && !s.includes("B组");
})());
check("渲染两行（空组过滤）", rowsOf(n).length === 2);
check("行开关态 A开B关", (() => {
    const rows = rowsOf(n);
    const a = rows.find((r) => rowTitle(r) === "A组");
    const b = rows.find((r) => rowTitle(r) === "B组");
    return a.children[1].className === "sf-ig-toggle on"
        && b.children[1].className === "sf-ig-toggle";
})());
check("computeSize 高度>表头", n.computeSize()[1] > 14);
check("节点尺寸已同步", Array.isArray(n.size) && n.size[1] > 14);

const other = { comfyClass: "SFOther" };
ext.nodeCreated(other);
check("非本类不装配", other._sfIg === undefined);

// ---- 2. 行点击切换 ----
// B组行：当前关 → 点开 → memberB.mode 回 0
{
    const row = rowsOf(n).find((r) => rowTitle(r) === "B组");
    fireMouseDown(row);
    check("B组点开：成员恢复 mode=0", memberB.mode === 0);
    check("B组点开：进 activeSet", n.properties.sf_ig_active_set.includes("B组"));
}
{
    // A组行：当前开 → 点关 → memberA.mode 置 4（旁路，disable=false）
    const row = rowsOf(n).find((r) => rowTitle(r) === "A组");
    fireMouseDown(row);
    check("A组点关：成员旁路 mode=4", memberA.mode === 4);
    check("A组点关：出 activeSet", !n.properties.sf_ig_active_set.includes("A组"));
}

// ---- 3. 设置弹窗开/关 ----
{
    const opts = [];
    n.getExtraMenuOptions({}, opts);
    const item = opts.find((o) => o && o.content === "SF 忽略多组 设置");
    check("右键菜单注入设置项", !!item);
    bodyStub.children.length = 0;
    item.callback();
    const pop = bodyStub.children.find((c) => c.className === "sf-ig-pop");
    const ov = bodyStub.children.find((c) => c.className === "sf-ig-overlay");
    check("设置弹窗已挂载", !!pop && !!ov);
    const hasTitle = [];
    (function walk(e) {
        if (e.textContent === "SF 忽略多组 设置") hasTitle.push(true);
        for (const c of e.children || []) walk(c);
    })(pop);
    check("弹窗标题正确", hasTitle.length > 0);
    for (const fn of ov._handlers.mousedown || []) {
        fn({ preventDefault() {}, stopPropagation() {} });
    }
    check("overlay 点击关闭弹窗",
        !bodyStub.children.includes(pop) && !bodyStub.children.includes(ov));
}

// ---- 4. 设置弹窗关闭逻辑（真实 sf_popup 行为锁定） ----
// overlay 与 pop 同级挂载：pop 内点击必须豁免，否则设置无法操作（§54 修法）
{
    const opts = [];
    n.getExtraMenuOptions({}, opts);
    const item = opts.find((o) => o && o.content === "SF 忽略多组 设置");
    const openPop = () => {
        bodyStub.children.length = 0;
        item.callback();
        return {
            pop: bodyStub.children.find((c) => c.className === "sf-ig-pop"),
            ov: bodyStub.children.find((c) => c.className === "sf-ig-overlay"),
        };
    };
    // 面板内点击（面板本身 + 内部控件）不关窗
    {
        const { pop, ov } = openPop();
        fireDocument("pointerdown", { target: pop });
        fireDocument("pointerdown", { target: pop.children[0] });
        check("面板内点击不关窗",
            bodyStub.children.includes(pop) && bodyStub.children.includes(ov));
        // 真正的外部点击关窗
        fireDocument("pointerdown", { target: bodyStub });
        check("外部点击关窗",
            !bodyStub.children.includes(pop) && !bodyStub.children.includes(ov));
    }
    // Esc 关窗
    {
        const { pop } = openPop();
        fireDocument("keydown", { key: "Escape" });
        check("Esc 关窗", !bodyStub.children.includes(pop));
    }
}

// ---- 5. graph.change 只包装一次 + onRemoved 清理 ----
{
    const wrappedOnce = fakeGraph.change;
    const n2 = makeNode();
    n2.addDOMWidget = function (name, type, root, opts) {
        const w = { el: root };
        n2._domWidget = w;
        return w;
    };
    ext.nodeCreated(n2);
    check("双节点共享同一次包装", fakeGraph.change === wrappedOnce);
    const t1 = n._sfIg.rt.timer;
    const t2 = n2._sfIg.rt.timer;
    check("两节点定时器独立", !!t1 && !!t2 && t1 !== t2);
    const kdBefore = docListeners.keydown;
    n.onRemoved();
    n2.onRemoved();
    check("onRemoved 清理完成", n._sfIg.rt.timer === null && n2._sfIg.rt.timer === null);
    check("document 监听成对移除", docListeners.keydown <= kdBefore);
    // 从 graph 摘除，避免轮询干扰后续
    fakeGraph._nodes = fakeGraph._nodes.filter((x) => x !== n && x !== n2);
}

if (failures.length) {
    console.log(`\n${failures.length} FAILED: ${failures}`);
    process.exit(1);
}
console.log("\nALL PASSED");
