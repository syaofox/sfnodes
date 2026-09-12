// SF Ignore Groups lib 纯函数测试（Node 直接运行：node tests/test_ignore_groups_lib.mjs）
// 覆盖：readState/writeState（sf_ig_* 键）/ groupBounds / nodeBounds /
// normalizeColor / hit / inside / collectNodes / nestedGroups /
// isNodeActive / groupState / stateSig / filterSortGroups / toggleTransition。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_ignore_groups_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_ignore_groups_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

// 假数据：G1(0,0,200,200) 含 N1；G2(300,0,200,200) 含 N2；G3 空组；N3 游离
const G = (title, bounds, color = "") => ({ title, bounds, color });
const N = (bounds, mode = 0, disabled = false) => ({ bounds, mode, disabled });
const groups = [G("B组", [0, 300, 200, 200]), G("A组", [0, 0, 200, 200], "#ff0000"), G("空组", [600, 0, 100, 100])];
const nodes = [N([10, 10, 50, 50]), N([10, 310, 50, 50], 4), N([900, 900, 50, 50])];

(async () => {
  const L = await import(pathToFileURL(tmpMjs).href);

  // ── 常量 ──
  check("模式常量 0/2/4", L.MODE_ALWAYS === 0 && L.MODE_NEVER === 2 && L.MODE_BYPASS === 4);

  // ── readState/writeState ──
  {
    const st = L.readState({});
    check("空 properties 回默认", st.filter === "" && st.mode === "default" && st.activeSet === null
      && st.uiScale === 1.0 && st.disable === false);
    const st2 = L.readState({ sf_ig_filter: "ab", sf_ig_mode: "always_one", sf_ig_ui_scale: "bad" });
    check("读 sf_ig_* 键", st2.filter === "ab" && st2.mode === "always_one");
    check("非法 scale 回 1.0", st2.uiScale === 1.0);
    const p = L.writeState({}, { ...st, filter: "x", activeSet: ["A组"] });
    check("写 sf_ig_* 键", p.sf_ig_filter === "x" && JSON.stringify(p.sf_ig_active_set) === '["A组"]');
  }

  // ── 几何 ──
  check("groupBounds 取 _bounding", JSON.stringify(L.groupBounds({ _bounding: [1, 2, 3, 4] })) === "[1,2,3,4]");
  check("groupBounds 回退 pos/size", JSON.stringify(L.groupBounds({ pos: [1, 2], size: [3, 4] })) === "[1,2,3,4]");
  check("nodeBounds 普通", JSON.stringify(L.nodeBounds({ pos: [1, 2], size: [3, 4] })) === "[1,2,3,4]");
  check("nodeBounds 折叠估宽", L.nodeBounds({ pos: [0, 0], collapsed: true, title: "ab" })[2] >= 80);
  check("normalizeColor number", L.normalizeColor(255) === "#0000ff");
  check("normalizeColor #rgb", L.normalizeColor("#f00") === "#ff0000");
  check("normalizeColor #rrggbb", L.normalizeColor("#ff0000") === "#ff0000");
  check("normalizeColor 未知→空", L.normalizeColor("red") === "" && L.normalizeColor(null) === "");
  check("hit 相交", L.hit([0, 0, 10, 10], [5, 5, 10, 10]) === true);
  check("hit 相邻边不算交", L.hit([0, 0, 10, 10], [10, 0, 10, 10]) === false);
  check("inside 包含", L.inside([2, 2, 5, 5], [0, 0, 10, 10]) === true);
  check("inside 越界 false", L.inside([2, 2, 9, 9], [0, 0, 10, 10]) === false);

  // ── collectNodes / nestedGroups ──
  {
    const got = L.collectNodes(groups[1], groups, nodes);
    check("A组收到 N1", got.length === 1 && got[0] === nodes[0]);
    const sub = G("子组", [10, 10, 50, 50]);
    const all = [...groups, sub];
    const got2 = L.collectNodes(groups[1], all, nodes);
    check("子组矩形并入命中", got2.length === 1);
    check("嵌套组检出", L.nestedGroups(groups[1], all).length === 1);
    check("无嵌套为空", L.nestedGroups(groups[0], groups).length === 0);
  }

  // ── 状态 ──
  check("active 判定", L.isNodeActive(nodes[0]) === true);
  check("bypass 非 active", L.isNodeActive(nodes[1]) === false);
  check("disabled 非 active", L.isNodeActive(N([0, 0, 1, 1], 0, true)) === false);
  check("A组全开 true", L.groupState(groups[1], groups, nodes) === true);
  check("B组全关 false", L.groupState(groups[0], groups, nodes) === false);
  check("空组 true", L.groupState(groups[2], groups, nodes) === true);
  {
    const mixed = [N([10, 10, 10, 10], 0), N([20, 20, 10, 10], 4)];
    check("混合 null", L.groupState(groups[1], groups, mixed) === null);
  }
  check("stateSig 编码", L.stateSig([
    { title: "a", state: true }, { title: "b", state: false }, { title: "c", state: null },
  ]) === "a:1\x00b:0\x00c:m");

  // ── splitKeywords（`|` 多关键词）──
  check("单关键词", JSON.stringify(L.splitKeywords("A组")) === '["a组"]');
  check("多关键词 OR", JSON.stringify(L.splitKeywords("A|B")) === '["a","b"]');
  check("去空格", JSON.stringify(L.splitKeywords("  A组  |  B ")) === '["a组","b"]');
  check("空段丢弃", JSON.stringify(L.splitKeywords("|A||")) === '["a"]');
  check("全空→不过滤", JSON.stringify(L.splitKeywords("  |||  ")) === "[]");
  check("空输入→不过滤", JSON.stringify(L.splitKeywords("")) === "[]" && JSON.stringify(L.splitKeywords(null)) === "[]");

  // ── filterSortGroups ──
  {
    const list = L.filterSortGroups(groups, groups, nodes, { filter: "", colorFilter: "none", sortOrder: "position" });
    check("空组被过滤", list.length === 2);
    check("按位置排序（A 在 B 上）", list[0].title === "A组" && list[1].title === "B组");
    const alpha = L.filterSortGroups(groups, groups, nodes, { filter: "", colorFilter: "none", sortOrder: "alphabet" });
    check("按首字母排序", alpha[0].title === "A组");
    const kw = L.filterSortGroups(groups, groups, nodes, { filter: "B", colorFilter: "none", sortOrder: "position" });
    check("关键词过滤", kw.length === 1 && kw[0].title === "B组");
    const multi = L.filterSortGroups(groups, groups, nodes, { filter: "A|B", colorFilter: "none", sortOrder: "position" });
    check("`|` 多关键词 OR（A、B 进，空组照常被过滤）",
      multi.length === 2 && multi[0].title === "A组" && multi[1].title === "B组");
    const multiSpace = L.filterSortGroups(groups, groups, nodes, { filter: " A组 | 不存在 ", colorFilter: "none", sortOrder: "position" });
    check("多关键词去空格+部分命中", multiSpace.length === 1 && multiSpace[0].title === "A组");
    const pipesOnly = L.filterSortGroups(groups, groups, nodes, { filter: "|||", colorFilter: "none", sortOrder: "position" });
    check("纯 `|` 等同留空", pipesOnly.length === 2);
    const col = L.filterSortGroups(groups, groups, nodes, { filter: "", colorFilter: "#ff0000", sortOrder: "position" });
    check("颜色过滤", col.length === 1 && col[0].title === "A组");
    const tr = L.filterSortGroups(groups, groups, nodes, { filter: "", colorFilter: "__transparent__", sortOrder: "position" });
    check("透明色过滤", tr.length === 1 && tr[0].title === "B组");
  }

  // ── toggleTransition ──
  {
    let r = L.toggleTransition("default", null, ["A组"], "B组", []);
    check("default 开：加入", JSON.stringify(r.activeSet) === '["A组","B组"]');
    r = L.toggleTransition("default", null, ["A组", "B组"], "B组", []);
    check("default 关：移除", JSON.stringify(r.activeSet) === '["A组"]');
    r = L.toggleTransition("default", null, [], "A组", ["子组"]);
    check("default 开连带嵌套", JSON.stringify(r.activeSet) === '["A组","子组"]');
    r = L.toggleTransition("default", null, ["A组", "子组"], "A组", ["子组"]);
    check("default 关连带嵌套", JSON.stringify(r.activeSet) === "[]");
    r = L.toggleTransition("always_one", "A组", null, "A组", []);
    check("always_one 点已选项保持", r.active === "A组");
    r = L.toggleTransition("always_one", "A组", null, "B组", []);
    check("always_one 切换", r.active === "B组");
    r = L.toggleTransition("at_most_one", "A组", null, "A组", []);
    check("at_most_one 点已选项全关", r.active === null);
    r = L.toggleTransition("at_most_one", null, null, "B组", []);
    check("at_most_one 独占", r.active === "B组");
  }

  if (failures.length) {
    console.log(`\n${failures.length} FAILED`);
    process.exit(1);
  }
  console.log("\nALL PASSED");
})();
