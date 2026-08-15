// sf_popup.js 公共弹层三件套冒烟测试（Node 直接运行：node tests/test_popup_smoke.js）
// 覆盖：三关闭（外部 pointerdown / Esc / wheel）、内部点击豁免、exempt 回调豁免、
// 关闭后监听清理幂等、detach 返回值幂等、clampToViewport 四向钳位与 scale 边距折算。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mocks ──
const listeners = new Map(); // type -> [{fn, capture}]
globalThis.document = {
    addEventListener(type, fn, capture) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push({ fn, capture });
    },
    removeEventListener(type, fn, capture) {
        const arr = listeners.get(type) || [];
        const i = arr.findIndex((l) => l.fn === fn && l.capture === capture);
        if (i >= 0) arr.splice(i, 1);
    },
    body: {},
};
globalThis.window = { innerWidth: 1000, innerHeight: 800 };

function makeOverlay(rect) {
    return {
        _rect: rect,
        style: {},
        addEventListener() {},   // attachPopupDismiss 只做 typeof 检查，监听都挂 document
        removeEventListener() {},
        contains(target) { return target === this; }, // 只有 overlay 自身算内部
        getBoundingClientRect() { return this._rect; },
    };
}
function fire(type, target, extra = {}) {
    for (const l of listeners.get(type) || []) l.fn({ target, ...extra });
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_popup_"));
    const code = fs.readFileSync(path.join(__dirname, "..", "web", "sf_popup.js"), "utf8");
    fs.writeFileSync(path.join(tmpDir, "sf_popup.mjs"), code);
    const mod = await import(path.join(tmpDir, "sf_popup.mjs"));

    // ── attachPopupDismiss：外部点击关闭 + 监听清理 ──
    let closed = 0;
    const overlay = makeOverlay({ left: 0, top: 0, width: 100, height: 50 });
    mod.attachPopupDismiss(overlay, { onClose: () => closed++ });

    fire("pointerdown", { fake: "outside" });           // 外部点击
    check("外部 pointerdown 关闭", closed === 1);
    fire("keydown", null, { key: "Escape" });           // 已关闭，监听已移除
    check("关闭后监听清理（不再重复触发）", closed === 1);

    // ── 内部点击豁免 + 非 Esc 键 + wheel ──
    closed = 0;
    const overlay2 = makeOverlay({ left: 0, top: 0, width: 100, height: 50 });
    mod.attachPopupDismiss(overlay2, { onClose: () => closed++ });

    fire("pointerdown", overlay2);                       // 内部点击
    check("内部点击不关闭", closed === 0);
    fire("keydown", null, { key: "Enter" });             // 非 Esc
    check("非 Esc 键不关闭", closed === 0);
    fire("wheel", { fake: "outside" });                  // 外部滚轮
    check("外部 wheel 关闭", closed === 1);

    // ── exempt 豁免 ──
    closed = 0;
    const overlay3 = makeOverlay({ left: 0, top: 0, width: 100, height: 50 });
    mod.attachPopupDismiss(overlay3, {
        onClose: () => closed++,
        exempt: (e) => e.skip === true,
    });
    fire("pointerdown", { fake: "outside" }, { skip: true });
    check("exempt 豁免外部点击", closed === 0);
    fire("pointerdown", { fake: "outside" });
    check("非豁免外部点击关闭", closed === 1);

    // ── detach 返回值幂等 ──
    closed = 0;
    const overlay4 = makeOverlay({ left: 0, top: 0, width: 100, height: 50 });
    const detach4 = mod.attachPopupDismiss(overlay4, { onClose: () => closed++ });
    detach4();
    detach4();
    fire("pointerdown", { fake: "outside" });
    check("detach 幂等且移除后不触发", closed === 0);

    // ── clampToViewport ──
    const c1 = makeOverlay({ left: 100, top: 100, width: 200, height: 50 });
    const r1 = mod.clampToViewport(c1);
    check("未越界不动", r1.left === 100 && r1.top === 100 && c1.style.left === undefined);

    const c2 = makeOverlay({ left: -20, top: -10, width: 200, height: 50 });
    const r2 = mod.clampToViewport(c2);
    check("左/上越界钳到 margin", r2.left === 8 && r2.top === 8
        && c2.style.left === "8px" && c2.style.top === "8px");

    const c3 = makeOverlay({ left: 900, top: 780, width: 200, height: 50 });
    const r3 = mod.clampToViewport(c3);
    check("右/下越界钳（1000-200-8=792）", r3.left === 792 && r3.top === 742);

    const c4 = makeOverlay({ left: -50, top: 100, width: 200, height: 50 });
    const r4 = mod.clampToViewport(c4, { scale: 2 });
    check("scale 折算边距（8*2=16）", r4.left === 16);

    if (failures.length) {
        console.error("\nFAILED:", failures.join(", "));
        process.exit(1);
    }
    console.log("\nALL PASS");
})();
