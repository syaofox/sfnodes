// SF 注释纯前端节点测试（Node 直接运行：node tests/test_note_js.js）
// 覆盖（Function-eval，test_ignore_groups_js.js 同款 + LiteGraph 全套 mock）：
// - 扩展注册 sfnodes.Note；registerCustomNodes 注册 SF Note 类型
// - 节点默认属性/尺寸；serialize/configure 回合
// - drawMultilineText 链接识别（linkAreas）；onMouseDown 点链接新开窗口
// - onDblClick 开编辑器；Escape 关编辑器并清定时器
// - 全局补丁 once（drawNode 背景重绘/链接点击拦截）；二次加载不叠包
// - 画布菜单已收敛到聚合器（web/sf_canvas_menu.js 📦 SF Menu ▶ Add SF Note，
//   tests/test_canvas_menu_js.js 覆盖建节点落图；此处只回归本扩展不再注入口）
// - Vue dblclick 中继：本节点转发 onDblClick，交互元素/它节点忽略
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
function makeEl(tag) {
    const e = {
        tag, children: [], _parent: null, _handlers: {},
        textContent: "", value: "", type: "", checked: false,
        style: {},
        setAttribute() {},
        addEventListener(t, fn) { ((this._handlers[t] = this._handlers[t] || []).push(fn)); },
        removeEventListener() {},
        appendChild(c) { c._parent = this; this.children.push(c); return c; },
        append(...cs) { for (const c of cs) { c._parent = this; this.children.push(c); } },
        remove() {
            if (this._parent) {
                const i = this._parent.children.indexOf(this);
                if (i >= 0) this._parent.children.splice(i, 1);
                this._parent = null;
            }
        },
        focus() {}, select() {},
        contains(t) {
            if (t === this) return true;
            return (this.children || []).some((c) => (c.contains ? c.contains(t) : c === t));
        },
        getBoundingClientRect() { return { left: 0, top: 0 }; },
    };
    return e;
}
const docHandlers = {};
const bodyStub = {
    children: [],
    appendChild(c) { c._parent = this; this.children.push(c); return c; },
    removeChild(c) {
        const i = this.children.indexOf(c);
        if (i >= 0) this.children.splice(i, 1);
        c._parent = null;
        return c;
    },
};
globalThis.document = {
    createElement: () => makeEl(),
    createTextNode: (t) => ({ text: t }),
    querySelectorAll: () => [],
    getElementById: () => null,
    body: bodyStub,
    head: { appendChild() {} },
    hidden: false,
    addEventListener(t, fn) { ((docHandlers[t] = docHandlers[t] || []).push(fn)); },
    removeEventListener() {},
};
function fireDocument(type, event) {
    for (const fn of (docHandlers[type] || []).slice()) fn(event);
}
const openedUrls = [];
globalThis.window = { open: (url) => { openedUrls.push(url); } };
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;

// ---- LiteGraph stubs（模块顶层求值时即需存在） ----
const registeredTypes = {};
globalThis.LiteGraph = {
    NO_TITLE: 2,
    NODE_TITLE_HEIGHT: 30,
    getTime: () => Date.now(),
    registerNodeType(type, cls) { registeredTypes[type] = cls; },
    createNode(type) {
        const C = registeredTypes[type];
        return C ? new C() : null;
    },
};
globalThis.LGraphNode = class LGraphNode {
    constructor(title) {
        this.title = title;
        this.properties = {};
        this.flags = {};
        this.size = [100, 60];
        this.pos = [0, 0];
    }
    serialize() { return { type: this.type }; }
    setDirtyCanvas() {}
};
let drawNodeCalls = 0;
globalThis.LGraphCanvas = {
    active_canvas: null,
    prototype: {
        drawNode() { drawNodeCalls++; return "orig"; },
        processMouseDown() { return "orig-pm"; },
    },
};
globalThis.LGraph = {
    prototype: {
        getNodeOnPos() { return "orig-node"; },
    },
};

// ---- app stub ----
let ext = null;
const fakeGraph = {
    _nodes: [],
    add(n) { this._nodes.push(n); n.graph = this; },
    getNodeById(id) { return this._nodes.find((n) => n.id === id) || null; },
    setDirtyCanvas() {},
    change() {},
};
const app = {
    canvas: {
        ds: { offset: [0, 0], scale: 1 },
        canvas: { clientWidth: 800, clientHeight: 600 },
    },
    graph: fakeGraph,
    extensionManager: undefined, // 走 LiteGraph 兜底分支
    registerExtension: (e) => { ext = e; },
};

const combined =
    stripImports("sf_note_lib.js") + "\n" +
    stripImports("sf_note.js");
new Function("app", "document", "window", combined)(
    app, globalThis.document, globalThis.window
);

check("扩展已注册 sfnodes.Note", ext && ext.name === "sfnodes.Note");
check("registerCustomNodes 存在", typeof ext.registerCustomNodes === "function");
ext.registerCustomNodes();
check("SF Note 类型已注册", !!registeredTypes["SF Note"]);
check("原孤海注释类型未被占用", !registeredTypes["孤海注释"]);

// ---- 1. 节点默认 ----
const Cls = registeredTypes["SF Note"];
const n = new Cls();
n.id = 7;
check("默认文本", n.properties.text === "双击编辑文本内容...");
check("默认字号/对齐/描边", n.properties.fontSize === 24
    && n.properties.textAlign === "center" && n.properties.stroke === true);
check("默认尺寸", n.size[0] === 360 && n.size[1] === 100);
check("属性描述子齐全", !!Cls["@text"] && !!Cls["@fontSize"] && !!Cls["@stroke"]);

// ---- 2. serialize/configure 回合 ----
{
    n.properties.text = "回合文本";
    n.properties.fontSize = 30;
    const data = n.serialize();
    const n2 = new Cls();
    n2.configure(data);
    check("configure 恢复文本/字号", n2.properties.text === "回合文本" && n2.properties.fontSize === 30);
    check("configure 恢复尺寸", n2.size[0] === 360);
}

// ---- 3. 绘制与链接 ----
function ctxStub() {
    return {
        save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {},
        stroke() {}, fill() {}, fillText() {}, arc() {},
        roundRect() {},
        measureText: (s) => ({ width: String(s).length * 10 }),
    };
}
{
    n.properties.text = "看 https://x.com/a 好";
    n.draw(ctxStub());
    check("绘制识别链接区", n.linkAreas.length === 1 && n.linkAreas[0].url === "https://x.com/a");
}
{
    openedUrls.length = 0;
    const a = n.linkAreas[0];
    const r = n.onMouseDown({}, [n.pos[0] + a.x + 1, n.pos[1] + a.y + 1]);
    check("点链接新开窗口", r === true && openedUrls.length === 1 && openedUrls[0] === "https://x.com/a");
    const r2 = n.onMouseDown({}, [n.pos[0] + 5000, n.pos[1] + 5000]);
    check("点空白不打开", r2 === false && openedUrls.length === 1);
}

// ---- 4. 双击编辑器开/关 ----
{
    // 编辑器定位依赖活动画布（getBoundingClientRect + ds 缩放）
    globalThis.LGraphCanvas.active_canvas = {
        canvas: { getBoundingClientRect: () => ({ left: 10, top: 20 }) },
        ds: { offset: [0, 0], scale: 1 },
        onCanvasChanged: null,
    };
    n.onDblClick();
    check("编辑器已打开", !!n.editTextarea && !!n.editToolbar && n.isEditing === true);
    check("编辑框初值", n.editTextarea.value === "看 https://x.com/a 好");
    n.editTextarea.value = "改后";
    for (const fn of n.editTextarea._handlers.keydown || []) {
        fn({ key: "Enter", ctrlKey: true, stopPropagation() {} });
    }
    check("Ctrl+Enter 保存回写", n.properties.text === "改后" && n.isEditing === false);
    check("编辑器定时器已清", n.canvasUpdateInterval === null);
}
{
    // Escape 关闭不保存 tested via fresh open
    n.onDblClick();
    n.editTextarea.value = "丢弃";
    for (const fn of n.editTextarea._handlers.keydown || []) {
        fn({ key: "Escape", stopPropagation() {} });
    }
    check("Escape 关闭且不保存", n.properties.text === "改后" && n.isEditing === false);
    // 挂起的 blur/setTimeout 回调无害（isEditing 已 false）
}

// ---- 5. 全局补丁 once ----
{
    check("drawNode 已包装", !!globalThis.LGraphCanvas.prototype._sfNotePatched);
    check("getNodeOnPos 已包装", !!globalThis.LGraph.prototype._sfNotePatched);
    const before = drawNodeCalls;
    globalThis.LGraphCanvas.prototype.drawNode.call(
        { graph: fakeGraph }, n, ctxStub());
    check("便签节点走背景重绘", drawNodeCalls === before + 1 && n.bgcolor === "transparent");
}

// ---- 6. 画布菜单已移交聚合器（本扩展不再注入分散入口） ----
{
    check("画布菜单已移交聚合器", ext.getCanvasMenuItems === undefined);
}

// ---- 7. Vue dblclick 中继 ----
{
    const added = new Cls();
    added.id = 9;
    fakeGraph.add(added); // 中继经 graph.getNodeById(9) 定位节点，需先入图
    const bodyEl = {
        getAttribute: (k) => (k === "data-testid" ? "node-body-9" : null),
    };
    const otherBodyEl = {
        getAttribute: (k) => (k === "data-testid" ? "node-body-8" : null),
    };
    let dblOpened = false;
    added.onDblClick = () => {
        dblOpened = true;
    };
    const ev = (testId = "node-body-9", interactive = false) => ({
        target: {
            closest(sel) {
                if (sel.startsWith("[data-testid")) {
                    if (testId == null) return null;
                    return testId === "other" ? otherBodyEl : bodyEl;
                }
                return interactive ? {} : null;
            },
        },
        preventDefault() {},
        stopPropagation() {},
    });
    fireDocument("dblclick", ev());
    check("中继转发本节点 onDblClick", dblOpened === true);
    dblOpened = false;
    fireDocument("dblclick", ev(null, true));
    check("中继豁免交互元素", dblOpened === false);
    fireDocument("dblclick", ev("other", false));
    check("中继忽略它节点", dblOpened === false);

    finish();
}

function finish() {
    // 清理：未关的编辑器定时器（若有），避免挂起进程
    for (const nd of fakeGraph._nodes) {
        if (nd && nd.canvasUpdateInterval) {
            clearInterval(nd.canvasUpdateInterval);
            nd.canvasUpdateInterval = null;
        }
    }
    if (failures.length) {
        console.log(`\n${failures.length} FAILED: ${failures}`);
        process.exit(1);
    }
    console.log("\nALL PASSED");
}
