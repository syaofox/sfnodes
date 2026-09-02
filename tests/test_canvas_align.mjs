// sf_canvas_align_lib 纯逻辑冒烟（Node 直接运行：node tests/test_canvas_align.mjs）
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_canvas_align_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_canvas_align_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

function fakeNode(w, h, minW, minH) {
  return {
    size: [w, h],
    computeSize() { return [minW ?? w, minH ?? h]; },
    setSize(s) { this.size = s; },
  };
}

(async () => {
  const L = await import(tmpUrl);

  // ── calcTargetWidth ──
  const n1 = fakeNode(200, 100, 160);
  const n2 = fakeNode(320, 120, 160);
  const n3 = fakeNode(280, 90, 160);
  check("widest 三节点", L.calcTargetWidth([n1, n2, n3], "widest") === 320);
  check("narrowest 三节点", L.calcTargetWidth([n1, n2, n3], "narrowest") === 200);
  check("first 取首位", L.calcTargetWidth([n1, n2, n3], "first") === 200);
  check("first 单节点", L.calcTargetWidth([n2], "first") === 320);
  check("空数组 0", L.calcTargetWidth([], "widest") === 0);
  check("无 computeSize 回退 size", L.calcTargetWidth([{ size: [150, 80] }], "widest") === 150);
  check("默认 widest", L.calcTargetWidth([n1, n3]) === 280);
  check("narrowest 空数组 0", L.calcTargetWidth([], "narrowest") === 0);

  // ── calcTargetHeight ──
  check("tallest 三节点", L.calcTargetHeight([n1, n2, n3], "tallest") === 120);
  check("shortest 三节点", L.calcTargetHeight([n1, n2, n3], "shortest") === 90);
  check("height first 首位", L.calcTargetHeight([n1, n2, n3], "first") === 100);
  check("height 空数组 0", L.calcTargetHeight([], "tallest") === 0);
  check("height 默认 tallest", L.calcTargetHeight([n1, n3]) === 100);
  check("height shortest 空 0", L.calcTargetHeight([], "shortest") === 0);
  check("height 无 computeSize 回退", L.calcTargetHeight([{ size: [80, 150] }], "tallest") === 150);

  // ── alignNodesWidth ──
  {
    const a = fakeNode(200, 100, 160);
    const b = fakeNode(320, 120, 160);
    const c = fakeNode(280, 90, 160);
    const cnt = L.alignNodesWidth([a, b, c], 320);
    check("alignW 计数 3", cnt === 3);
    check("alignW 后同宽 320", a.size[0] === 320 && b.size[0] === 320 && c.size[0] === 320);
    check("alignW 高度不变", a.size[1] === 100 && b.size[1] === 120 && c.size[1] === 90);
  }
  {
    const small = fakeNode(200, 100, 300);
    L.alignNodesWidth([small], 150);
    check("alignW 钳制到 minW", small.size[0] === 300);
  }
  {
    check("alignW 无效 target 0", L.alignNodesWidth([n1], 0) === 0);
    check("alignW 空列表 0", L.alignNodesWidth([], 320) === 0);
  }
  {
    const bare = { size: [200, 100], computeSize() { return [100, 100]; } };
    L.alignNodesWidth([bare], 300);
    check("alignW 无 setSize 回退", bare.size[0] === 300);
  }

  // ── alignNodesHeight ──
  {
    const a = fakeNode(200, 100, 100, 80);
    const b = fakeNode(320, 120, 100, 80);
    const c = fakeNode(280, 90, 100, 80);
    const cnt = L.alignNodesHeight([a, b, c], 120);
    check("alignH 计数 3", cnt === 3);
    check("alignH 后同高 120", a.size[1] === 120 && b.size[1] === 120 && c.size[1] === 120);
    check("alignH 宽度不变", a.size[0] === 200 && b.size[0] === 320 && c.size[0] === 280);
  }
  {
    const small = fakeNode(200, 100, 100, 150);
    L.alignNodesHeight([small], 90);
    check("alignH 钳制到 minH", small.size[1] === 150);
  }
  {
    check("alignH 无效 0", L.alignNodesHeight([n1], 0) === 0);
    check("alignH 空列表 0", L.alignNodesHeight([], 120) === 0);
  }
  {
    const bare = { size: [200, 100], computeSize() { return [100, 100]; } };
    L.alignNodesHeight([bare], 200);
    check("alignH 无 setSize 回退", bare.size[1] === 200);
  }

  // ── alignNodesSize ──
  {
    const a = fakeNode(200, 100, 160, 80);
    const b = fakeNode(320, 120, 160, 80);
    const cnt = L.alignNodesSize([a, b], 320, 120);
    check("alignSize 计数 2", cnt === 2);
    check("alignSize a 320x120", a.size[0] === 320 && a.size[1] === 120);
    check("alignSize b 保持 320x120", b.size[0] === 320 && b.size[1] === 120);
  }
  {
    // 等大钳制：目标 200x90 但 a 最小 250x100
    const a = fakeNode(200, 100, 250, 100);
    const b = fakeNode(320, 120, 160, 80);
    L.alignNodesSize([a, b], 200, 90);
    check("alignSize 钳制 a 仍 250x100", a.size[0] === 250 && a.size[1] === 100);
    check("alignSize b 缩到 200x90", b.size[0] === 200 && b.size[1] === 90);
  }
  {
    check("alignSize 空列表 0", L.alignNodesSize([], 100, 100) === 0);
    check("alignSize 无效双 0", L.alignNodesSize([n1], 0, 0) === 0);
  }

  // ── getSelectedNodes ──
  {
    const a = fakeNode(100, 100, 100), b = fakeNode(100, 100, 100), c = fakeNode(100, 100, 100);
    let app = { canvas: { selected_nodes: { 1: a, 2: b } }, graph: { _nodes: [] } };
    check("selected_nodes Object 2 项", L.getSelectedNodes(app).length === 2);
    app = { canvas: { selected_nodes: [a, b, c] }, graph: { _nodes: [] } };
    check("selected_nodes Array 3 项", L.getSelectedNodes(app).length === 3);
    app = { canvas: { selected_nodes: new Map([[1, a], [2, b]]) }, graph: { _nodes: [] } };
    check("selected_nodes Map 2 项", L.getSelectedNodes(app).length === 2);
    app = { canvas: { selected_nodes: new Set([a, b]) }, graph: { _nodes: [] } };
    check("selected_nodes Set 2 项", L.getSelectedNodes(app).length === 2);
    a.is_selected = false; b.is_selected = false; c.is_selected = false;
    const flagged = { id: 1, is_selected: true, size: [100, 100], computeSize() { return [100, 100]; } };
    app = { canvas: { selected_nodes: {} }, graph: { _nodes: [flagged, { id: 2, size: [100, 100] }] } };
    check("回退扫描 is_selected", L.getSelectedNodes(app).length === 1 && L.getSelectedNodes(app)[0] === flagged);
    app = { canvas: { selected_nodes: { 1: a } }, graph: { _nodes: [flagged] } };
    check("单选返回 1 不回退", L.getSelectedNodes(app).length === 1);
    app = { canvas: { selected_nodes: {} }, graph: { _nodes: [] } };
    check("空图 0", L.getSelectedNodes(app).length === 0);
  }

  // ── 集成 ──
  {
    const first = fakeNode(250, 100, 100);
    const other = fakeNode(300, 120, 100);
    const tw = L.calcTargetWidth([first, other], "first");
    L.alignNodesWidth([first, other], tw);
    check("first 锚点后同宽 250", first.size[0] === 250 && other.size[0] === 250);
  }
  {
    const a = fakeNode(200, 100, 160);
    const b = fakeNode(320, 120, 160);
    const c = fakeNode(280, 90, 160);
    const tw = L.calcTargetWidth([a, b, c], "narrowest");
    check("narrowest 目标 200", tw === 200);
    L.alignNodesWidth([a, b, c], tw);
    check("narrowest 后同宽 200", a.size[0] === 200 && b.size[0] === 200 && c.size[0] === 200);
  }
  {
    const a = fakeNode(200, 100, 250);
    const b = fakeNode(320, 120, 160);
    const tw = L.calcTargetWidth([a, b], "narrowest");
    L.alignNodesWidth([a, b], tw);
    check("narrowest 钳制：a 保持 250", a.size[0] === 250);
    check("narrowest 钳制：b 缩到 200", b.size[0] === 200);
  }
  {
    // height first 锚点
    const a = fakeNode(200, 100, 100, 80);
    const b = fakeNode(200, 150, 100, 80);
    const th = L.calcTargetHeight([a, b], "first");
    L.alignNodesHeight([a, b], th);
    check("height first 锚点 100", a.size[1] === 100 && b.size[1] === 100);
  }
  {
    // size 集成：first 同时对齐
    const a = fakeNode(200, 100, 100, 100);
    const b = fakeNode(300, 150, 100, 100);
    const tw = L.calcTargetWidth([a, b], "first");
    const th = L.calcTargetHeight([a, b], "first");
    L.alignNodesSize([a, b], tw, th);
    check("size first 同步 200x100", a.size[0] === 200 && a.size[1] === 100 && b.size[0] === 200 && b.size[1] === 100);
  }

  if (failures.length) {
    console.log("FAILURES:", failures.join(", "));
    process.exit(1);
  }
  console.log("ALL PASS");
})();
