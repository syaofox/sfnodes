// sf_image_resize_plus.js size_mode 显隐冒烟测试（Node 直接运行：node tests/test_image_resize_plus_js.js）
// 验证：
//   - 扩展注册 + 仅 SFImageResizePlus 生效
//   - nodeCreated 初始：width & height 模式 → total_pixels 隐藏
//   - callback 切到 total pixels → width/height 隐藏
//   - configure / onAfterGraphConfigured 恢复后重新显隐（双钩子保恢复）
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

globalThis.document = { createElement() { return { style: {}, addEventListener() {}, click() {}, remove() {} }; }, body: { appendChild() {} } };
globalThis.app = { registerExtension(ext) { this._ext = ext; } };

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_resize_plus_"));
const file = path.join(tmp, "sf_image_resize_plus.js");
fs.writeFileSync(file, fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_image_resize_plus.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;"));
require(file);

const ext = globalThis.app._ext;
check("扩展注册名", ext && ext.name === "sfnodes.SFImageResizePlus");

function makeNode(sizeMode, dirty) {
    const node = {
        comfyClass: "SFImageResizePlus",
        widgets: [
            { name: "size_mode", value: sizeMode, callback: null },
            { name: "width", value: 1024, callback: null },
            { name: "height", value: 1024, callback: null },
            { name: "total_pixels", value: 1.0, callback: null },
            { name: "multiplier", value: 1.0, callback: null },
            { name: "longer_size", value: 512, callback: null },
            { name: "shorter_size", value: 512, callback: null },
            { name: "multiple", value: 8, callback: null },
            { name: "divisible_by", value: 8, callback: null },
            { name: "method", value: "keep proportion", callback: null },
            { name: "crop_position", value: "center", callback: null },
            { name: "pad_color", value: "#000000", callback: null },
        ],
    };
    if (dirty) node.setDirtyCanvas = () => { node._dirty = true; };
    return node;
}
const w = (n, name) => n.widgets.find((x) => x.name === name);

// 1. 初始（width & height）：total_pixels 隐藏
{
    const n = makeNode("width & height", true);
    ext.nodeCreated(n);
    check("初始 total_pixels 隐藏", w(n, "total_pixels").hidden === true);
    check("初始 width 可见", w(n, "width").hidden === false);
    check("初始 height 可见", w(n, "height").hidden === false);
    check("setDirtyCanvas 触发", n._dirty === true);
}

// 2. callback 切到 total pixels
{
    const n = makeNode("width & height");
    ext.nodeCreated(n);
    w(n, "size_mode").value = "total pixels";
    w(n, "size_mode").callback();
    check("切 total pixels: width/height 隐藏", w(n, "width").hidden === true && w(n, "height").hidden === true);
    check("切 total pixels: total_pixels 可见", w(n, "total_pixels").hidden === false);
}

// 3. configure 包装：恢复旧工作流值后重新显隐（setTimeout(0) 异步）
//    + 旧版 8 项 widgets_values remap（size_mode 置顶重排前的顺序）
{
    const n = makeNode("width & height");
    ext.nodeCreated(n);
    let captured = null;
    n.configure = function (data) {
        captured = data;
        // 模拟 LiteGraph 按位恢复（remap 后的顺序）
        if (Array.isArray(data.widgets_values)) {
            this.widgets[0].value = data.widgets_values[0];
        }
    };
    ext.nodeCreated(n);
    n.configure({
        widgets_values: [1024, 512, "lanczos", "pad", "always", 8, "center", "#000000"],
    });
    check("旧版 8 项 widgets_values remap 为 14 项", JSON.stringify(captured.widgets_values) === JSON.stringify([
        "width & height", 1024, 512, 1.0, 1.0, 512, 512, 8, "lanczos", "pad", "always", 8, "center", "#000000",
    ]));
    n.configure({ widgets_values: ["total pixels", 1024, 1024, 1.5, "lanczos", "pad", "always", 8, "center", "#000000"] });
    check("旧版 10 项 widgets_values remap 为 14 项",
        JSON.stringify(captured.widgets_values) === JSON.stringify([
            "total pixels", 1024, 1024, 1.5, 1.0, 512, 512, 8, "lanczos", "pad", "always", 8, "center", "#000000",
        ]));
    n.configure({ widgets_values: ["total pixels", 1024, 1024, 1.5, 1.0, 512, 512, 8, "lanczos", "pad", "always", 8, "center", "#000000"] });
    check("新版 14 项 widgets_values 不改写",
        captured.widgets_values[0] === "total pixels" && captured.widgets_values.length === 14);
    setTimeout(() => {
        check("configure 恢复 total pixels 后 width/height 隐藏",
            w(n, "width").hidden === true && w(n, "height").hidden === true && w(n, "total_pixels").hidden === false);

        // 3.5 method 联动：crop_position 仅 fill / crop，pad_color 仅 pad
        const n4 = makeNode("width & height");
        ext.nodeCreated(n4);
        check("默认 method=keep proportion: crop_position/pad_color 隐藏",
            w(n4, "crop_position").hidden === true && w(n4, "pad_color").hidden === true);
        w(n4, "method").value = "fill / crop";
        w(n4, "method").callback();
        check("fill / crop: crop_position 可显 pad_color 隐藏",
            w(n4, "crop_position").hidden === false && w(n4, "pad_color").hidden === true);
        w(n4, "method").value = "pad";
        w(n4, "method").callback();
        check("pad: pad_color 可显 crop_position 隐藏",
            w(n4, "pad_color").hidden === false && w(n4, "crop_position").hidden === true);
        w(n4, "method").value = "keep proportion";
        w(n4, "method").callback();
        check("keep proportion: 两者隐藏",
            w(n4, "crop_position").hidden === true && w(n4, "pad_color").hidden === true);

        // 3.6 新增 4 个 size_mode 的参数显隐
        const n5 = makeNode("width & height");
        ext.nodeCreated(n5);
        check("默认 width & height: 新模式参数全隐藏",
            w(n5, "multiplier").hidden === true && w(n5, "longer_size").hidden === true
            && w(n5, "shorter_size").hidden === true && w(n5, "multiple").hidden === true);
        const modes = [
            ["scale by multiplier", "multiplier"],
            ["longer dimension", "longer_size"],
            ["shorter dimension", "shorter_size"],
            ["scale to multiple", "multiple"],
        ];
        for (const [mode, active] of modes) {
            w(n5, "size_mode").value = mode;
            w(n5, "size_mode").callback();
            const all = ["multiplier", "longer_size", "shorter_size", "multiple"];
            const ok = all.every((name) => w(n5, name).hidden === (name !== active));
            check(`${mode}: 仅隐藏其余模式参数`, ok);
        }
        // scale to multiple 内部强制 cover → method 隐藏；倍数网格独占 → divisible_by 隐藏
        check("scale to multiple: method 隐藏",
            w(n5, "method").hidden === true);
        check("scale to multiple: divisible_by 隐藏",
            w(n5, "divisible_by").hidden === true);
        w(n5, "size_mode").value = "width & height";
        w(n5, "size_mode").callback();
        check("切回 width & height: method/divisible_by 恢复可见",
            w(n5, "method").hidden === false && w(n5, "divisible_by").hidden === false);

        // 4. onAfterGraphConfigured 同步恢复
        const n2 = makeNode("width & height");
        ext.nodeCreated(n2);
        w(n2, "size_mode").value = "total pixels";
        ext.nodeCreated(n2).onAfterGraphConfigured && null;
        (n2.onAfterGraphConfigured ? n2.onAfterGraphConfigured() : null);
        check("onAfterGraphConfigured 恢复显隐",
            w(n2, "width").hidden === true && w(n2, "total_pixels").hidden === false);

        // 5. 非 SFImageResizePlus 不动
        const n3 = { comfyClass: "SFOther", widgets: makeNode("width & height").widgets };
        ext.nodeCreated(n3);
        check("非目标类节点不动", w(n3, "total_pixels").hidden !== true);

        if (failures.length) {
            console.log(`\n${failures.length} FAILED`);
            process.exit(1);
        }
        console.log("\nall passed");
    }, 10);
}
