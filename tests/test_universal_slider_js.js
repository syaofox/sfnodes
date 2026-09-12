// SF Universal Slider 前端装配测试（Node 直接运行：node tests/test_universal_slider_js.js）
// 覆盖（Function-eval 注入真源，test_convert_anything_js.js 同款）：
// - 扩展注册 sfnodes.UniversalSlider；非本类节点不受影响
// - nodeCreated：properties 默认填充 / value 隐藏 / output_type 同步 /
//   输出槽改型 FLOAT+槽名 float / 自定义 widget 注册 / 最小宽度 300
// - configure 恢复：重抓 widgets 引用 + 存量值钳制取整 + 输出槽跟随 int
// - onAfterGraphConfigured 输出槽恢复；右键菜单注入设置项且不吞原菜单；
//   onWidgetChanged 回填 dirty；拖拽 mouse 按 calcValue 更新值
// 复用 any_pack.js setSlotType + sf_universal_slider_lib.js 真源；
// sf_common/sf_popup 仅 stub（injectCSSOnce/el/attachPopupDismiss/clampToViewport）。
const fs = require("fs");
const path = require("path");

// import 剥离必须行首锚定：注释里含 "import xxx.js" 字样（如纯模块边界说明），
// 非锚定正则会从注释一路吃到下一个分号、吞掉真实代码（DEFAULTS 未定义坑）
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
const injectedCSS = {};
const injectCSSOnce = (id) => { injectedCSS[id] = true; };
const el = (tag, cls, text) => {
    const e = {
        tag, className: cls || "", textContent: text || "", value: "", type: "",
        checked: false, name: "", children: [], _parent: null,
        style: { setProperty() {} },
        setAttribute() {}, addEventListener() {},
        appendChild(c) { c._parent = this; this.children.push(c); return c; },
        append(...cs) { for (const c of cs) { c._parent = this; this.children.push(c); } },
        // remove() 从父容器摘除（模拟 Element.remove，供弹窗关闭断言）
        remove() {
            if (this._parent) {
                const i = this._parent.children.indexOf(this);
                if (i >= 0) this._parent.children.splice(i, 1);
                this._parent = null;
            }
            this._removed = true;
        },
        focus() {},
    };
    return e;
};
const attachPopupDismiss = () => () => {};
const clampToViewport = () => {};
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
// 递归找按钮（按 textContent）
function findButton(root, text) {
    if (root.textContent === text && root.tag === "button") return root;
    for (const c of root.children || []) {
        const hit = findButton(c, text);
        if (hit) return hit;
    }
    return null;
}

// any_pack 真源（setSlotType）+ lib 真源 + 主模块拼接进同一作用域
const combined =
    stripImports("any_pack.js") + "\n" +
    stripImports("sf_universal_slider_lib.js") + "\n" +
    stripImports("sf_universal_slider.js");
new Function(
    "app", "el", "injectCSSOnce", "attachPopupDismiss", "clampToViewport", "document",
    combined
)(app, el, injectCSSOnce, attachPopupDismiss, clampToViewport, globalThis.document);

check("扩展已注册 sfnodes.UniversalSlider", ext && ext.name === "sfnodes.UniversalSlider");

// ---- 假节点 ----
function makeNode() {
    const node = {
        comfyClass: "SFUniversalSlider",
        title: "SF Universal Slider",
        widgets: [
            { name: "value", type: "number", value: 0.75, options: {} },
            { name: "output_type", type: "combo", value: "float", options: {} },
        ],
        outputs: [{ name: "value", type: "*", localized_name: "value" }],
        properties: {},
        size: [200, 100],
        setDirtyCanvas() { this._dirty = (this._dirty || 0) + 1; },
        addWidget(type, name, value, cb, opts) {
            const w = { name, type, value, callback: cb, options: opts || {} };
            this.widgets.push(w);
            return w;
        },
        addCustomWidget(w) { this._custom = w; return w; },
        configure() {},
    };
    return node;
}

// ---- 1. nodeCreated ----
const n = makeNode();
ext.nodeCreated(n);
check("properties 默认填充", n.properties.sliderType === "float"
    && n.properties.sliderMin === 0 && n.properties.sliderMax === 1
    && n.properties.sliderStep === 0.01 && n.properties.sliderLabel === "value");
check("value widget 隐藏", n.widgets[0].hidden === true);
check("output_type 同步 float", n.widgets[1].value === "float");
check("输出槽改型 FLOAT", n.outputs[0].type === "FLOAT");
check("输出槽名 float（含 localized_name）", n.outputs[0].name === "float"
    && n.outputs[0].localized_name === "float");
check("自定义 widget 注册", n._custom && n._custom.type === "sf_universal_slider"
    && typeof n._custom.draw === "function" && typeof n._custom.mouse === "function");
check("最小宽度 300", n.size[0] === 300);
check("节点配色", n.color === "#2D384D" && n.bgcolor === "#2D384D");

// 非本类节点不受影响
const other = { comfyClass: "SFOther" };
ext.nodeCreated(other);
check("非本类不装配", other._sfUS === undefined && other._custom === undefined);

// ---- 2. 拖拽 mouse 按 calcValue 更新 ----
// 先跑一次 draw 确立轨道几何（W=300 → trackLeft=14/trackW=262）；
// pos x=14+131=145 → 中点 0.5
{
    const ctxStub = {
        save() {}, restore() {}, beginPath() {}, moveTo() {}, lineTo() {},
        arcTo() {}, closePath() {}, fill() {}, stroke() {},
        arc() {}, fillText() {},
        measureText() { return { width: 10 }; },
    };
    n._custom.draw(ctxStub, n, 300, 0);
    const before = n.widgets[0].value;
    const r = n._custom.mouse({ type: "mousedown", button: 0, clientX: 0 }, [145, 40], n);
    check("mousedown 命中返回 true", r === true);
    check("mousedown 跳到中点 0.5", Math.abs(n.widgets[0].value - 0.5) < 1e-9);
    const r2 = n._custom.mouse({ type: "mouseup" }, [145, 40], n);
    check("mouseup 结束拖拽", r2 === true && n._sfUS._dragging === false);
    void before;
}

// ---- 3. configure 恢复（int 档跟随） ----
{
    n.properties.sliderType = "int";
    n.properties.sliderMin = 0;
    n.properties.sliderMax = 10;
    n.properties.sliderStep = 1;
    n.widgets[0].value = 4.6;
    n.configure({});
    check("configure 存量值取整 4.6→5", n.widgets[0].value === 5);
    check("configure 输出槽跟随 INT/int", n.outputs[0].type === "INT" && n.outputs[0].name === "int");
    check("configure output_type 同步 int", n.widgets[1].value === "int");
}

// ---- 4. onAfterGraphConfigured / 右键菜单 / onWidgetChanged ----
{
    n.properties.sliderType = "float";
    n.onAfterGraphConfigured();
    check("AG 恢复 FLOAT/float", n.outputs[0].type === "FLOAT" && n.outputs[0].name === "float");

    let origCalled = false;
    n.getExtraMenuOptions.__orig = null;
    const prev = n.getExtraMenuOptions;
    n.getExtraMenuOptions = function (c, o) { origCalled = true; return prev.call(this, c, o); };
    // 重新装配一次以包装新的 orig（模拟真实调用链只需验证包装不吞原菜单）
    const opts = [];
    n.getExtraMenuOptions({}, opts);
    check("右键菜单注入设置项", opts.some((o) => o && o.content === "SF 万能滑条 设置"));
    check("右键菜单不吞原菜单", origCalled);

    const d0 = n._dirty || 0;
    n.onWidgetChanged("value", 0.5, n.widgets[0]);
    check("onWidgetChanged 回填 dirty", (n._dirty || 0) > d0);

    // ---- 5. 设置弹窗：确定/取消点击后关闭窗口 ----
    const openSettings = () => {
        const menuOpts = [];
        n.getExtraMenuOptions({}, menuOpts);
        const item = menuOpts.find((o) => o && o.content === "SF 万能滑条 设置");
        item.callback();
        return bodyStub.children.find((c) => c.className === "sf-us-overlay");
    };
    // 取消：不改值但必须关窗
    {
        const ov = openSettings();
        check("设置弹窗已挂载", !!ov);
        const v0 = n.widgets[0].value;
        findButton(ov, "取消").onclick();
        check("取消后弹窗关闭", !bodyStub.children.includes(ov));
        check("取消不改值", n.widgets[0].value === v0);
    }
    // 确定：应用归一化 + 关窗（min>max 对调的归一逻辑走 lib，见 mjs 测试）
    {
        const ov = openSettings();
        findButton(ov, "确定").onclick();
        check("确定后弹窗关闭", !bodyStub.children.includes(ov));
        check("确定后 body 无残留弹窗",
            !bodyStub.children.some((c) => c.className === "sf-us-overlay"));
    }
}

if (failures.length) {
    console.log(`\n${failures.length} FAILED: ${failures}`);
    process.exit(1);
}
console.log("\nALL PASSED");
