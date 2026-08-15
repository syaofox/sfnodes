// sf_common.js installWheelZoomPassthrough 冒烟测试（Node 直接运行）
// 覆盖三分支：
//   Ctrl/⌘+wheel → 总是转发 canvas 缩放
//   普通 wheel + 输入框可滚动 → 滚动文本（不转发）
//   普通 wheel + 输入框不可滚动 → 转发 canvas 缩放
//   cleanup 移除后不再转发
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mocks ──
const canvasEl = { dispatched: [], dispatchEvent(e) { this.dispatched.push(e); } };
globalThis.window = { addEventListener() {}, removeEventListener() {} };
globalThis.app = { canvas: { canvas: canvasEl } };
globalThis.api = { apiURL: (r) => r };

class MockWheelEvent {
    constructor(type, opts) { this.type = type; Object.assign(this, opts); }
}
globalThis.WheelEvent = MockWheelEvent;

function makeEl({ scrollHeight = 10, clientHeight = 10 } = {}) {
    return {
        scrollHeight, clientHeight, scrollWidth: 10, clientWidth: 10,
        listeners: {},
        addEventListener(type, fn, opts) { (this.listeners[type] ||= []).push(fn); },
        removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter(f => f !== fn); },
        dispatchEvent() {},
    };
}

function fakeWheel({ ctrlKey = false, metaKey = false } = {}) {
    return {
        clientX: 10, clientY: 20, deltaX: 0, deltaY: 100,
        ctrlKey, metaKey, shiftKey: false,
        preventDefault() {}, stopPropagation() {},
    };
}

(async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_wheel_"));
    const code = fs
        .readFileSync(path.join(__dirname, "..", "web", "sf_common.js"), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;");
    fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"), code);
    const C = await import(path.join(tmpDir, "sf_common.mjs"));

    // 1) Ctrl+wheel → 总是转发（即使可滚动）
    {
        const el = makeEl({ scrollHeight: 100, clientHeight: 10 }); // 可滚动
        const cleanup = C.installWheelZoomPassthrough(el);
        canvasEl.dispatched = [];
        el.listeners.wheel[0](fakeWheel({ ctrlKey: true }));
        check("Ctrl+wheel 可滚动也转发", canvasEl.dispatched.length === 1 && canvasEl.dispatched[0].deltaY === 100);
    }

    // 2) 普通 wheel + 可滚动 → 不转发（滚动文本）
    {
        const el = makeEl({ scrollHeight: 100, clientHeight: 10 });
        C.installWheelZoomPassthrough(el);
        canvasEl.dispatched = [];
        el.listeners.wheel[0](fakeWheel({}));
        check("普通 wheel + 可滚动 → 不转发", canvasEl.dispatched.length === 0);
    }

    // 3) 普通 wheel + 不可滚动 → 转发
    {
        const el = makeEl({ scrollHeight: 10, clientHeight: 10 });
        C.installWheelZoomPassthrough(el);
        canvasEl.dispatched = [];
        el.listeners.wheel[0](fakeWheel({}));
        check("普通 wheel + 不可滚动 → 转发", canvasEl.dispatched.length === 1);
    }

    // 4) 转发事件携带 ctrlKey（canvas 据此缩放）
    {
        const el = makeEl();
        C.installWheelZoomPassthrough(el);
        canvasEl.dispatched = [];
        el.listeners.wheel[0](fakeWheel({ ctrlKey: true }));
        check("转发事件携带 ctrlKey", canvasEl.dispatched[0]?.ctrlKey === true);
    }

    // 5) cleanup 移除监听
    {
        const el = makeEl();
        const cleanup = C.installWheelZoomPassthrough(el);
        check("cleanup 前有监听", (el.listeners.wheel || []).length === 1);
        cleanup();
        check("cleanup 后监听移除", (el.listeners.wheel || []).length === 0);
    }

    // 6) 非元素参数安全返回 noop
    {
        const cleanup = C.installWheelZoomPassthrough(null);
        check("null 参数安全", typeof cleanup === "function");
    }

    console.log();
    if (failures.length) { console.log(failures.length + " FAILURES:", failures); process.exit(1); }
    console.log("ALL PASS");
})().catch((e) => { console.error(e); process.exit(1); });
