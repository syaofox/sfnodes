// sf_widget_width_lib 纯逻辑测试（Node 直接运行：node tests/test_sf_widget_width_lib.mjs）
// 覆盖：守卫读写语义（legacy 丢弃 / Vue 透传 / 动态切档）/ 幂等 /
//       全图扫描（_nodes·nodes·subgraph 去重防环）/ 原型工厂包装
//       （addWidget → addCustomWidget 两条路径、二次安装不叠包）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_widget_width_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_widget_width_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

class FakeLGraphNode {
  constructor() {
    this.widgets = [];
  }
  addWidget(type, name, value, callback, options) {
    const widget = { type, name, value, callback, options: options || {}, y: 0 };
    return this.addCustomWidget(widget);
  }
  addCustomWidget(widget) {
    this.widgets.push(widget);
    return widget;
  }
}

(async () => {
  const L = await import(pathToFileURL(tmpMjs).href);

  // ── 模式开关（可动态切换）──
  let vue = false;
  const isVue = () => vue;

  // ── 单 widget 守卫语义 ──
  {
    const w = { width: 200, name: "a" };
    check("guard 安装返回 true", L.guardWidgetWidth(w, isVue) === true);
    check("legacy 读 undefined", w.width === undefined);
    w.width = 333;
    check("legacy 写丢弃", w.width === undefined);
    vue = true;
    w.width = 333;
    check("Vue 写透传", w.width === 333);
    check("Vue 读透传", w.width === 333);
    vue = false;
    check("切回 legacy 读 undefined", w.width === undefined);
    check("切回 legacy 后回退 nodeWidth", (w.width || 300) === 300);
    vue = true;
    check("再切 Vue 保留上次值", w.width === 333);
    vue = false;
    check("重复守卫返回 false（幂等）", L.guardWidgetWidth(w, isVue) === false);
    const desc = Object.getOwnPropertyDescriptor(w, "width");
    check("属性可配置（上游可接管）", desc.configurable === true && desc.enumerable === true);
  }

  // ── 非对象/已守卫输入 ──
  check("null 返回 false", L.guardWidgetWidth(null, isVue) === false);
  check("字符串返回 false", L.guardWidgetWidth("x", isVue) === false);
  check("isWidgetWidthGuarded 未守卫 false", L.isWidgetWidthGuarded({}) === false);
  check("isWidgetWidthGuarded 已守卫 true", L.isWidgetWidthGuarded({ [L.GUARD_FLAG]: true }) === true);

  // ── 探测回调异常时保守透传 ──
  {
    const w = { width: 1 };
    L.guardWidgetWidth(w, () => { throw new Error("boom"); });
    w.width = 42;
    check("探测异常透传写入", w.width === 42);
  }

  // ── 节点扫描 ──
  {
    const node = { widgets: [{ name: "a" }, { name: "b" }, { name: "c" }] };
    check("扫描计数 3", L.guardNodeWidgets(node, isVue) === 3);
    check("扫描后全守卫", node.widgets.every((w) => L.isWidgetWidthGuarded(w)));
    check("重复扫描计数 0", L.guardNodeWidgets(node, isVue) === 0);
    check("无 widgets 不计", L.guardNodeWidgets({}, isVue) === 0);
    check("null 节点安全", L.guardNodeWidgets(null, isVue) === 0);
  }

  // ── 全图扫描：_nodes / nodes / subgraph 两形态 / 防环 ──
  {
    const mkWidget = () => ({ name: "w" });
    const n1 = { widgets: [mkWidget(), mkWidget()] };
    const n2 = { widgets: [mkWidget()] };
    const subNode = { widgets: [mkWidget()] };
    const sub = { _nodes: [subNode], subgraphs: new Map() };
    const embedded = { subgraph: { _nodes: [{ widgets: [mkWidget()] }], subgraphs: new Map() } };
    const child = { nodes: [{ widgets: [mkWidget()] }], subgraphs: new Map() };
    const root = {
      nodes: [n1, n2, embedded],
      subgraphs: new Map([["a", sub], ["b", child]]),
    };
    // 防环：子图互引
    sub.subgraphs.set("child", child);
    child.subgraphs.set("sub", sub);
    check("全图扫描计数 6", L.sweepGraphWidgets(root, isVue) === 6);
    check("node.subgraph 分支也扫到", L.isWidgetWidthGuarded(embedded.subgraph._nodes[0].widgets[0]));
    check("_nodes 分支也扫到", L.isWidgetWidthGuarded(subNode.widgets[0]));
    check("重复扫描计数 0", L.sweepGraphWidgets(root, isVue) === 0);
    check("空图安全", L.sweepGraphWidgets(null, isVue) === 0);
  }

  // ── 原型工厂包装 ──
  {
    check("包装返回 true", L.patchWidgetFactories(FakeLGraphNode, isVue) === true);
    const afterFirst = FakeLGraphNode.prototype.addWidget;
    check("二次包装返回 false（幂等）", L.patchWidgetFactories(FakeLGraphNode, isVue) === false);
    check("二次包装不叠包", FakeLGraphNode.prototype.addWidget === afterFirst);

    const node = new FakeLGraphNode();
    const viaAddWidget = node.addWidget("combo", "mode", "x", () => {}, { values: ["x"] });
    check("addWidget 路径即时守卫", L.isWidgetWidthGuarded(viaAddWidget));
    const viaCustom = node.addCustomWidget({ name: "custom" });
    check("addCustomWidget 路径即时守卫", L.isWidgetWidthGuarded(viaCustom));

    // legacy 下 DOM widget 写宽被丢弃 → LiteGraph 回退 nodeWidth
    viaAddWidget.width = 500;
    check("新 widget legacy 写丢弃", viaAddWidget.width === undefined);
    check("回退语义 width||nodeWidth", (viaAddWidget.width || 240) === 240);
    vue = true;
    viaAddWidget.width = 500;
    check("新 widget Vue 写透传", viaAddWidget.width === 500);
    vue = false;

    // 未实现工厂方法的类安全
    check("空原型安全", L.patchWidgetFactories({}, isVue) === false);
    check("null 类安全", L.patchWidgetFactories(null, isVue) === false);
  }

  if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
  }
  console.log("\nALL PASSED");
})();
