// ==========================================================================
// sf_widget_width_lib.js — legacy 模式 widget 宽度冻结修复纯逻辑
// （无 app/DOM 依赖，可拷 .mjs 用 Node 直测：test_sf_widget_width_lib.mjs）
//
// 背景（platform.md §121，前端 1.53.6 实测）：Vue 的 WidgetLegacy 组件在
// LiteGraph（Classic）模式仍挂载，其 draw() 每帧无条件执行
// `widgetInstance.width = DOM容器宽`（上游 #11574 引入的回归，issue
// Comfy-Org/ComfyUI_frontend#12443，修复 PR #12444 未合并）。而 LiteGraph
// drawWidgets 用 `widget.width || node.renderingSize[0]`、DomWidgets 用
// `posWidget.width ?? posNode.width` —— 一旦 width 是数字就覆盖节点宽度，
// 拖拽改宽不再联动（widget 溢出/悬空）。
//
// 修复形态：给 widget 实例的 `width` 换成受控访问器 ——
//   - Vue Nodes 2.0 模式：读写透传（保持原行为，上游修好后也兼容）
//   - Classic 模式：写入丢弃、读取 undefined（= widget 出厂未定义态，
//     让 `width || nodeWidth` 回退生效）
// 本模块只收敛可测逻辑，app/LiteGraph 接线留在 sf_widget_width_fix.js。
// ==========================================================================

// widget 上的一次性守卫标记（避免重复 defineProperty）。
export const GUARD_FLAG = "_sfWidgetWidthGuarded";
// LGraphNode.prototype 的一次性包装标记。
export const PATCH_FLAG = "_sfWidgetWidthPatched";

// 探测回调异常时保守透传（宁可保留原行为，也不误丢合法写入）。
function vueMode(isVueNodes) {
  try {
    return !!isVueNodes();
  } catch {
    return true;
  }
}

export function isWidgetWidthGuarded(widget) {
  return !!widget && widget[GUARD_FLAG] === true;
}

// 单 widget 幂等守卫：返回是否本次新安装。
export function guardWidgetWidth(widget, isVueNodes) {
  if (!widget || typeof widget !== "object" || isWidgetWidthGuarded(widget)) return false;
  let stored = widget.width;
  try {
    Object.defineProperty(widget, "width", {
      configurable: true,
      enumerable: true,
      get() {
        return vueMode(isVueNodes) ? stored : undefined;
      },
      set(value) {
        if (vueMode(isVueNodes)) stored = value;
      },
    });
    widget[GUARD_FLAG] = true;
  } catch {
    return false;
  }
  return true;
}

// 单节点全部 widget（含 DOM/custom widget，都在 node.widgets 里）。
export function guardNodeWidgets(node, isVueNodes) {
  const widgets = node?.widgets;
  if (!Array.isArray(widgets)) return 0;
  let count = 0;
  for (const widget of widgets) {
    if (guardWidgetWidth(widget, isVueNodes)) count += 1;
  }
  return count;
}

// 全图扫描（含 subgraph；graph.subgraphs 为 Map，节点上也挂 .subgraph）。
export function sweepGraphWidgets(graph, isVueNodes) {
  const seen = new Set();
  let count = 0;
  const visit = (g) => {
    if (!g || typeof g !== "object" || seen.has(g)) return;
    seen.add(g);
    const nodes = Array.isArray(g._nodes) ? g._nodes : g.nodes;
    if (Array.isArray(nodes)) {
      for (const node of nodes) {
        count += guardNodeWidgets(node, isVueNodes);
        if (node?.subgraph) visit(node.subgraph);
      }
    }
    const subs = g.subgraphs;
    if (subs) {
      if (typeof subs.forEach === "function" && !Array.isArray(subs)) subs.forEach(visit);
      else if (Array.isArray(subs)) for (const sg of subs) visit(sg);
    }
  };
  visit(graph);
  return count;
}

// 包装 LGraphNode.prototype 的 widget 工厂：addWidget 内部走
// addCustomWidget，DOM widget 助手也走 addCustomWidget —— 两个都包上，
// 覆盖之后创建的 widget（含其它扩展自建）。幂等，返回是否本次新安装。
export function patchWidgetFactories(LGraphNode, isVueNodes) {
  const proto = LGraphNode?.prototype;
  if (!proto || proto[PATCH_FLAG]) return false;
  let patched = false;
  for (const name of ["addWidget", "addCustomWidget"]) {
    const original = proto[name];
    if (typeof original !== "function") continue;
    proto[name] = function (...args) {
      const widget = original.apply(this, args);
      guardWidgetWidth(widget, isVueNodes);
      return widget;
    };
    patched = true;
  }
  if (patched) proto[PATCH_FLAG] = true;
  return patched;
}
