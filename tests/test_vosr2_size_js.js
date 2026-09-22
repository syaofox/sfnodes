// sf_vosr2_size.js 目标尺寸模式显隐冒烟测试（Node 直接运行：node tests/test_vosr2_size_js.js）
// 验证：
//   - 扩展注册名 + 仅 SFVOSR2Upscale / SFVOSR2Video 生效
//   - MODE_WIDGETS 四模式映射
//   - nodeCreated 初始按 mode 显隐；callback 切换后重新显隐
//   - 无变化时不触发 refreshWidgetSnapshot；有变化时替换 widgets 引用并标脏
//   - 缺少 size_mode widget 的节点安全退出
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ---- 加载被测模块（去 import / export；显隐库剥壳注入）----
function loadStripped(file) {
    const raw = fs.readFileSync(path.join(__dirname, "..", "web", file), "utf8");
    return raw
        .replace(/import[^;]+;/g, "")
        .replace(/export\s*\{[^}]*\}\s*;?/g, "")
        .replace(/export\s+(?=function|const|let|class|var)/g, "");
}

const libCode = loadStripped("sf_widget_visibility_lib.js");
const lib = new Function(
    libCode + "\nreturn { setWidgetVisible, isWidgetVisible, refreshWidgetSnapshot };"
)();

let capturedExt = null;
const app = { registerExtension(ext) { capturedExt = ext; } };

const exported = new Function(
    "app", "setWidgetVisible", "refreshWidgetSnapshot", "isWidgetVisible",
    loadStripped("sf_vosr2_size.js") + "\nreturn { MODE_WIDGETS, SIZE_WIDGET_NAMES, applySizeModeVisibility };"
)(app, lib.setWidgetVisible, lib.refreshWidgetSnapshot, lib.isWidgetVisible);

check("扩展注册名", capturedExt && capturedExt.name === "sfnodes.SFVOSR2Size");
check("MODE_WIDGETS 四模式", Object.keys(exported.MODE_WIDGETS).length === 4);
check("scale 映射", JSON.stringify(exported.MODE_WIDGETS["scale"]) === JSON.stringify(["scale"]));
check("total pixels 映射", JSON.stringify(exported.MODE_WIDGETS["total pixels"]) === JSON.stringify(["total_pixels"]));
check("longer dimension 映射", JSON.stringify(exported.MODE_WIDGETS["longer dimension"]) === JSON.stringify(["longer_size"]));
check("shorter dimension 映射", JSON.stringify(exported.MODE_WIDGETS["shorter dimension"]) === JSON.stringify(["shorter_size"]));

function fakeWidget(name, value) {
    return { name, value, type: "number", hidden: false, options: {}, callback: null };
}

function makeNode(comfyClass, mode, withModeWidget = true) {
    const widgets = [];
    if (withModeWidget) widgets.push(fakeWidget("size_mode", mode));
    widgets.push(
        fakeWidget("scale", 4.0),
        fakeWidget("total_pixels", 1.0),
        fakeWidget("longer_size", 1024),
        fakeWidget("shorter_size", 1024),
        fakeWidget("seed", 42),
    );
    const node = { comfyClass, widgets, setDirtyCanvas() { node._dirty = true; } };
    return node;
}

const w = (n, name) => n.widgets.find((x) => x.name === name);
const vis = (n, name) => lib.isWidgetVisible(w(n, name));

// 1. 初始 scale 模式
{
    const n = makeNode("SFVOSR2Upscale", "scale");
    capturedExt.nodeCreated(n);
    check("scale 模式：scale 可见", vis(n, "scale"));
    check("scale 模式：total_pixels 隐藏", !vis(n, "total_pixels"));
    check("scale 模式：longer_size 隐藏", !vis(n, "longer_size"));
    check("scale 模式：shorter_size 隐藏", !vis(n, "shorter_size"));
    check("scale 模式：seed 不受影响", vis(n, "seed"));
    check("初始 toggle 标脏", n._dirty === true);
}

// 2. callback 切到 total pixels
{
    const n = makeNode("SFVOSR2Upscale", "scale");
    capturedExt.nodeCreated(n);
    const before = n.widgets;
    w(n, "size_mode").value = "total pixels";
    w(n, "size_mode").callback.call(w(n, "size_mode"));
    check("切换后 total_pixels 可见", vis(n, "total_pixels"));
    check("切换后 scale 隐藏", !vis(n, "scale"));
    check("切换后 widgets 引用替换", n.widgets !== before && n.widgets.length === before.length);
}

// 3. 长边 / 短边
{
    const n = makeNode("SFVOSR2Video", "longer dimension");
    capturedExt.nodeCreated(n);
    check("视频节点：longer_size 可见", vis(n, "longer_size"));
    check("视频节点：shorter_size 隐藏", !vis(n, "shorter_size"));
    w(n, "size_mode").value = "shorter dimension";
    w(n, "size_mode").callback.call(w(n, "size_mode"));
    check("切短边：shorter_size 可见", vis(n, "shorter_size"));
    check("切短边：longer_size 隐藏", !vis(n, "longer_size"));
}

// 4. 无关节点不处理
{
    const n = makeNode("SFOtherNode", "scale");
    capturedExt.nodeCreated(n);
    check("无关节点不显隐", vis(n, "total_pixels") === true && n._dirty === undefined);
}

// 5. 缺 size_mode 安全退出
{
    const n = makeNode("SFVOSR2Upscale", "scale", false);
    capturedExt.nodeCreated(n);
    check("缺 size_mode 不抛错", n._dirty === undefined);
}

// 6. applySizeModeVisibility 返回值（无变化 false）
{
    const n = makeNode("SFVOSR2Upscale", "scale");
    capturedExt.nodeCreated(n);
    check("重复应用无变化返回 false", exported.applySizeModeVisibility(n) === false);
    w(n, "size_mode").value = "total pixels";
    check("切模式返回 true", exported.applySizeModeVisibility(n) === true);
}

if (failures.length) {
    console.log(`\n${failures.length} 项失败: ${failures}`);
    process.exit(1);
}
console.log("\n全部通过");
