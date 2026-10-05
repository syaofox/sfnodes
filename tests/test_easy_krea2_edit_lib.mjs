// SFEasyKrea2Edit 纯逻辑库测试（Node 直接运行：node tests/test_easy_krea2_edit_lib.mjs）
// 覆盖：imageNumbers 过滤/排序、strengthFor 回退、setStrength 写 properties、
// stateJson 只含当前槽位、syncStrengthWidgets 动态增减/回填/回调写回、serialize:false。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_easy_krea2_edit_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_easy_krea2_edit_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

function makeNode(slotNums = [1]) {
  return {
    inputs: slotNums.map((n) => ({ name: `image${n}`, link: null })),
    widgets: [],
    properties: {},
    addWidget(type, name, value, cb, options) {
      const w = { type, name, value, callback: cb, options };
      this.widgets.push(w);
      return w;
    },
  };
}

(async () => {
  const L = await import(pathToFileURL(tmpMjs).href);

  check("HIDDEN_INPUT 命名", L.HIDDEN_INPUT === "SFEasyKrea2EditState");
  check("IMAGE_RE 匹配/排除", L.IMAGE_RE.test("image2") && !L.IMAGE_RE.test("latent_image"));

  // imageNumbers：过滤非 imageN + 数字序排序（image10 在 image9 后）
  const node = makeNode([1, 10, 2]);
  check("imageNumbers 数字序", JSON.stringify(L.imageNumbers(node)) === "[1,2,10]");

  // strengthFor 缺省与非法回退
  check("strengthFor 缺省 1.0", L.strengthFor(node, 3) === 1.0);
  node.properties[L.STRENGTH_PROP] = { 1: 0.5, 2: -1, 10: "x" };
  check("strengthFor 有效值", L.strengthFor(node, 1) === 0.5);
  check("strengthFor 负数回退", L.strengthFor(node, 2) === 1.0);
  check("strengthFor 非数值回退", L.strengthFor(node, 10) === 1.0);

  // setStrength 写 properties（不影响其他槽）
  L.setStrength(node, 10, 0.25);
  check("setStrength 写入", L.strengthFor(node, 10) === 0.25 && L.strengthFor(node, 1) === 0.5);

  // stateJson 只含当前槽位
  const st = JSON.parse(L.stateJson(node));
  check("stateJson 键集合", JSON.stringify(Object.keys(st.strengths).sort()) === '["1","10","2"]');
  check("stateJson 值", st.strengths["2"] === 1.0 && st.strengths["10"] === 0.25);

  // syncStrengthWidgets：按槽位增删 + 值回填 + 非序列化
  const node2 = makeNode([1, 2]);
  L.setStrength(node2, 2, 0.4);
  L.syncStrengthWidgets(node2);
  check("sync 新增 2 个 widget", node2.widgets.length === 2);
  check("sync 标记槽位", node2.widgets[0]._sfStrengthN === 1 && node2.widgets[1]._sfStrengthN === 2);
  check("sync 回填 properties 值", node2.widgets[0].value === 1.0 && node2.widgets[1].value === 0.4);
  check("widget serialize:false", node2.widgets[0].options.serialize === false);
  check("widget 数值范围", node2.widgets[0].options.min === 0 && node2.widgets[0].options.max === 10);

  // 回调写回 properties
  node2.widgets[1].callback(0.7);
  check("回调写回 properties", L.strengthFor(node2, 2) === 0.7);

  // 槽位移除 → widget 回收；properties 保留
  node2.inputs = [{ name: "image1", link: null }];
  L.syncStrengthWidgets(node2);
  check("槽位移除回收 widget", node2.widgets.length === 1 && node2.widgets[0]._sfStrengthN === 1);
  check("properties 保留历史值", L.strengthFor(node2, 2) === 0.7);

  // 槽位追加 → 恢复历史值
  node2.inputs.push({ name: "image2", link: null });
  L.syncStrengthWidgets(node2);
  check("槽位恢复回填历史值", node2.widgets.length === 2 && node2.widgets[1].value === 0.7);

  // 同一槽位重复 sync 不重复添加
  L.syncStrengthWidgets(node2);
  check("重复 sync 幂等", node2.widgets.length === 2);

  console.log();
  if (failures.length) {
    console.log(`FAILED: ${failures.length} -> ${failures}`);
    process.exit(1);
  }
  console.log("ALL PASSED");
})();
