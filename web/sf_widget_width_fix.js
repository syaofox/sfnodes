// ==========================================================================
// sf_widget_width_fix.js — legacy（Classic）模式 widget 宽度冻结修复
//
// 内置版上游 bug 规避（等价 pekkAi-dev/ComfyUI-LegacyWidgetWidthFix，不引入
// 第三方包）：前端 1.53.6 实测 WidgetLegacy.vue 的 draw() 在 Classic 模式
// 仍无条件写 `widget.width`，覆盖 LiteGraph `widget.width || nodeWidth` 回退
// → 节点改宽后 widget 不跟随（溢出/悬空）。上游 issue #12443 open，修复 PR
// #12444 未合并；上游修好后本模块自动 no-op（写入只发生在 Vue 模式，守卫
// 透传）。根因与机制见 doc/experience/platform.md §121。
//
// 三路安装（缺一漏项）：
//   1. patchWidgetFactories —— 包装 LGraphNode.prototype.addWidget/
//      addCustomWidget（DOM widget 助手同走 addCustomWidget），之后创建的
//      widget 即时受控；
//   2. sweepAll —— 全图扫描已有节点（含 subgraph）；
//   3. nodeCreated/loadedGraphNode/afterConfigureGraph —— 兜底不经过
//      addWidget 自建 widget 的扩展与工作流载入恢复。
// 纯逻辑收敛于 sf_widget_width_lib.js（可 .mjs 直测），本文件只做接线。
// ==========================================================================

import { app } from "/scripts/app.js";
import {
  guardNodeWidgets,
  patchWidgetFactories,
  sweepGraphWidgets,
} from "./sf_widget_width_lib.js";

function litegraph() {
  return globalThis.LiteGraph || globalThis.window?.LiteGraph || null;
}

function isVueNodes() {
  return !!litegraph()?.vueNodesMode;
}

function install() {
  const LG = litegraph();
  const LGraphNode = globalThis.LGraphNode || LG?.LGraphNode || null;
  if (!LGraphNode) return false;
  return patchWidgetFactories(LGraphNode, isVueNodes);
}

function sweepAll() {
  try {
    sweepGraphWidgets(app.graph, isVueNodes);
  } catch {
    /* 图未就绪/异常时忽略，后续钩子会补扫 */
  }
}

function sweepNode(node) {
  try {
    guardNodeWidgets(node, isVueNodes);
  } catch {
    /* ignore */
  }
}

install();
sweepAll();

app.registerExtension({
  name: "sfnodes.WidgetWidthFix",

  init() {
    install();
    sweepAll();
  },

  setup() {
    install();
    sweepAll();
  },

  afterConfigureGraph() {
    install();
    sweepAll();
  },

  nodeCreated(node) {
    install();
    sweepNode(node);
  },

  loadedGraphNode(node) {
    install();
    sweepNode(node);
  },
});
