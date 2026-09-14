// sf_common.js 画布节点拖拽释放兜底测试（Node 直接运行）
// 覆盖：primaryButtonReleased 语义（buttons 0/1/2/缺失/null）；
// installNodeReleaseGuard 幂等 + 四事件（mouseup/pointerup/pointercancel/blur）
// window capture 注册；removeNodeReleaseGuard 解绑并清 hook。
// 加载方式同 test_common_paste_js.js：剥掉 app/api import 后作 .mjs 直跑。
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// ── window 桩：按 type + capture 记录监听器 ──
const listeners = [];
globalThis.window = {
  addEventListener(type, fn, capture) { listeners.push({ type, fn, capture: !!capture }); },
  removeEventListener(type, fn, capture) {
    const i = listeners.findIndex(
      (l) => l.type === type && l.fn === fn && l.capture === !!capture);
    if (i >= 0) listeners.splice(i, 1);
  },
};
const fire = (type) => {
  for (const l of listeners.filter((l) => l.type === type)) l.fn({ type });
};

globalThis.app = { graph: { _nodes: [] }, canvas: {} };
globalThis.api = { apiURL: (r) => r };

(async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_common_guard_"));
  const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_common.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replaceAll('import { api } from "/scripts/api.js";', "const api = globalThis.api;");
  fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"), code);
  const mod = await import(path.join(tmpDir, "sf_common.mjs"));

  // ── primaryButtonReleased ──
  check("buttons=0 已松开", mod.primaryButtonReleased({ buttons: 0 }) === true);
  check("buttons=1 按住", mod.primaryButtonReleased({ buttons: 1 }) === false);
  check("buttons=1|2 仍按住主键", mod.primaryButtonReleased({ buttons: 3 }) === false);
  check("buttons=2 右键（无主键）", mod.primaryButtonReleased({ buttons: 2 }) === true);
  check("buttons 缺失放行", mod.primaryButtonReleased({}) === false);
  check("null 放行", mod.primaryButtonReleased(null) === false);
  check("undefined 放行", mod.primaryButtonReleased(undefined) === false);

  // ── installNodeReleaseGuard ──
  const node = {};
  let calls = 0;
  mod.installNodeReleaseGuard(node, () => { calls += 1; }, { hook: "_g" });
  check("安装后 hook 存在", !!node._g);
  const captured = listeners.filter((l) => l.type === "mouseup" || l.type === "pointerup"
    || l.type === "pointercancel" || l.type === "blur");
  check("四事件均已注册", captured.length === 4);
  check("全部 capture=true", captured.every((l) => l.capture === true));
  check("事件名齐全", ["mouseup", "pointerup", "pointercancel", "blur"]
    .every((t) => captured.some((l) => l.type === t)));

  // 幂等：重复安装同 hook 不叠加
  mod.installNodeReleaseGuard(node, () => { calls += 100; }, { hook: "_g" });
  check("幂等安装不叠加监听", listeners.filter((l) => l.type === "mouseup").length === 1);

  fire("mouseup");
  check("mouseup 触发回调", calls === 1);
  fire("pointerup");
  fire("pointercancel");
  fire("blur");
  check("四事件各自触发回调", calls === 4);

  // ── removeNodeReleaseGuard ──
  mod.removeNodeReleaseGuard(node, { hook: "_g" });
  check("卸载后 hook 清空", node._g === null);
  check("卸载后监听清空", listeners.filter((l) => l.type === "mouseup").length === 0);
  fire("mouseup");
  check("卸载后不再回调", calls === 4);

  // 卸载不存在的 hook 安全
  mod.removeNodeReleaseGuard(node, { hook: "_g" });
  check("重复卸载安全", true);

  // ── 回调抛错不影响其它（仅记录）──
  const node2 = {};
  mod.installNodeReleaseGuard(node2, () => { throw new Error("boom"); }, { hook: "_g2" });
  const errSpy = console.error;
  console.error = () => {};
  let threw = false;
  try { fire("mouseup"); } catch { threw = true; }
  console.error = errSpy;
  check("回调抛错被吞", threw === false);
  mod.removeNodeReleaseGuard(node2, { hook: "_g2" });

  console.log();
  if (failures.length) { console.log(`${failures.length} FAILED: ${failures}`); process.exit(1); }
  console.log("ALL PASS");
})().catch((e) => { console.error("TEST ERROR:", e); process.exit(1); });
