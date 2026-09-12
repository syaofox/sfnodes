// SF Boolean Switch 前端装配测试（Node 直接运行：node tests/test_boolean_switch_js.js）
// 覆盖（Function-eval 注入 lib 真源，test_universal_slider_js.js 同款）：
// - 扩展注册 sfnodes.BooleanSwitch；非本类节点不受影响
// - nodeCreated：标签 properties 默认 / 原生 widget 隐藏 / 配色 /
//   自定义 widget 注册 / 开关初态跟随 widget 值
// - draw 不抛 + 标签落画；开关区单击切换值；标签区单击不切换
// - 双击打开标签编辑框；Enter 落盘 properties 并移除输入框
// - configure 重抓引用 + 同步开关态；onWidgetChanged 回填同步
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

// ---- mocks ----
let ext = null;
const app = {
    canvas: { ds: { scale: 1 }, setDirty: () => {} },
    registerExtension: (e) => { ext = e; },
};
const el = (tag, cls, text) => {
    const e = {
        tag, className: cls || "", textContent: text || "", value: "", type: "",
        checked: false, name: "", children: [], _parent: null, _handlers: {},
        style: {},
        setAttribute() {},
        addEventListener(t, fn) { (this._handlers[t] = this._handlers[t] || []).push(fn); },
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
    };
    return e;
};
const bodyStub = {
    children: [],
    appendChild(c) { c._parent = this; this.children.push(c); return c; },
};
globalThis.document = {
    createElement: (t) => el(t),
    createTextNode: (t) => ({ text: t }),
    querySelectorAll: () => [],
    body: bodyStub,
    addEventListener() {},
    removeEventListener() {},
};
globalThis.requestAnimationFrame = (fn) => fn();

const combined =
    stripImports("sf_boolean_switch_lib.js") + "\n" +
    stripImports("sf_boolean_switch.js");
new Function("app", "el", "document", combined)(app, el, globalThis.document);

check("扩展已注册 sfnodes.BooleanSwitch", ext && ext.name === "sfnodes.BooleanSwitch");

// ---- 假节点 ----
function makeNode(value) {
    return {
        comfyClass: "SFBooleanSwitch",
        title: "SF Boolean Switch",
        widgets: [{ name: "value", type: "boolean", value, options: {} }],
        outputs: [{ name: "value", type: "BOOLEAN", localized_name: "value" }],
        properties: {},
        size: [300, 100],
        setDirtyCanvas() { this._dirty = (this._dirty || 0) + 1; },
        addWidget(t, nm, v, cb, o) {
            const w = { name: nm, type: t, value: v, callback: cb, options: o || {} };
            this.widgets.push(w);
            return w;
        },
        addCustomWidget(w) { this._custom = w; return w; },
        configure() {},
    };
}

function ctxStub(captured) {
    return {
        save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {},
        arcTo() {}, closePath() {}, fill() {}, stroke() {}, arc() {},
        fillText(t) { if (captured) captured.texts.push(t); },
        measureText(t) { return { width: String(t).length * 12 }; },
    };
}

// ---- 1. nodeCreated ----
const n = makeNode(true);
ext.nodeCreated(n);
check("标签 properties 默认 value", n.properties.sfBoolLabel === "value");
check("原生 widget 隐藏", n.widgets[0].hidden === true);
check("节点配色", n.color === "#4F4047" && n.bgcolor === "#493C42");
check("自定义 widget 注册", n._custom && n._custom.type === "sf_boolean_switch");
check("开关初态跟随 True", n._sfBool && n._sfBool.isOn === true);

const other = { comfyClass: "SFOther" };
ext.nodeCreated(other);
check("非本类不装配", other._sfBool === undefined && other._custom === undefined);

// ---- 2. draw + 单击切换 ----
{
    const captured = { texts: [] };
    n._custom.draw(ctxStub(captured), n, 300, 0, 44);
    check("draw 不抛且标签落画", captured.texts.length > 0 && captured.texts[0] === "value");
}
{
    // 开关区：阈值 W-102=198，pos 250 命中
    const r = n._custom.mouse.call(n._custom, { type: "mousedown", clientX: 0, clientY: 0 }, [250, 20], n);
    check("开关区单击切换 True→False", r === true && n.widgets[0].value === false && n._sfBool.isOn === false);
    check("切换后 dirty", (n._dirty || 0) > 0);
}
{
    // 标签区 pos 0 不切换（上次双击计时已过 350ms 才会走单击分支——先冷却）
    n._sfBool.lastClickTime = 0;
    const v0 = n.widgets[0].value;
    const r = n._custom.mouse.call(n._custom, { type: "mousedown", clientX: 0, clientY: 0 }, [10, 20], n);
    check("标签区单击不切换", r === false && n.widgets[0].value === v0);
}

// ---- 3. 双击改标签 ----
{
    n._sfBool.lastClickTime = 0;
    bodyStub.children.length = 0;
    // 第一次单击（开关区外，避免翻转开关态干扰断言）
    n._custom.mouse.call(n._custom, { type: "mousedown", clientX: 50, clientY: 60 }, [10, 20], n);
    // 350ms 内第二次 → 双击开编辑框
    const r = n._custom.mouse.call(n._custom, { type: "mousedown", clientX: 50, clientY: 60 }, [10, 20], n);
    const input = bodyStub.children.find((c) => c.tag === "input");
    check("双击打开标签编辑框", r === true && !!input);
    input.value = "  主灯  ";
    for (const fn of input._handlers.keydown || []) {
        fn({ key: "Enter", preventDefault() {}, stopImmediatePropagation() {} });
    }
    check("Enter 落盘去空格标签", n.properties.sfBoolLabel === "主灯");
    check("编辑框已移除", !bodyStub.children.includes(input));
    const captured = { texts: [] };
    n._custom.draw(ctxStub(captured), n, 300, 0, 44);
    check("新标签落画", captured.texts[0] === "主灯");
}

// ---- 4. configure + onWidgetChanged ----
{
    n.widgets[0].value = true;
    n.properties.sfBoolLabel = "副灯";
    n.configure({});
    check("configure 同步开关态 True", n._sfBool.isOn === true);
    const captured = { texts: [] };
    n._custom.draw(ctxStub(captured), n, 300, 0, 44);
    check("configure 后标签恢复 副灯", captured.texts[0] === "副灯");

    n.widgets[0].value = false;
    const d0 = n._dirty || 0;
    n.onWidgetChanged("value", false, n.widgets[0]);
    check("onWidgetChanged 同步 False", n._sfBool.isOn === false);
    check("onWidgetChanged 回填 dirty", (n._dirty || 0) > d0);
}

if (failures.length) {
    console.log(`\n${failures.length} FAILED: ${failures}`);
    process.exit(1);
}
console.log("\nALL PASSED");
