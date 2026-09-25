// ==========================================================================
// sf_widget_visibility_lib.js - widget 条件显隐公共库（自 sf_scail2.js 提取单源）
// ==========================================================================
//
// 供按 combo/开关条件显隐 widget 的节点复用（SFSCAIL2* / SFQwenImage21PromptEnhancer）：
//
//   import { setWidgetVisible, isWidgetVisible } from "./sf_widget_visibility_lib.js";
//   setWidgetVisible(widget, false);            // 隐藏（保留值，仍随工作流保存/提交）
//   if (!isWidgetVisible(widget)) setWidgetVisible(widget, true);
//
// 机制（双渲染器通用，见 experience/platform.md §121 相关）：
// - 隐藏 = widget.type 置 "hidden" + widget.hidden + options.hidden/canvasOnly，
//   并把 computeSize 换成 [0,-4]（Classic 布局收缩）；DOM widget 同步 display:none。
// - 原值/原 computeSize 首次隐藏时存入 WeakMap（storeWidgetDefaults），显示时恢复；
//   幂等，可反复切换。
// - 隐藏只影响渲染，不影响取值与序列化（隐藏 widget 仍进 widgets_values/请求体）。
//
// 纯逻辑（无 app 依赖），可拷贝 .mjs 直测。
// ==========================================================================

const WIDGET_DEFAULTS = new WeakMap();

const hiddenWidgetComputeSize = () => [0, -4];

function ensureOptions(widget) {
  widget.options ||= {};
  if (widget._state) {
    widget._state.options ||= {};
  }
}

function setOption(widget, key, value) {
  ensureOptions(widget);
  widget.options[key] = value;
  if (widget._state) {
    widget._state.options[key] = value;
  }
}

function deleteOption(widget, key) {
  widget.options ||= {};
  delete widget.options[key];
  if (widget._state?.options) {
    delete widget._state.options[key];
  }
}

export function storeWidgetDefaults(widget) {
  if (!widget || WIDGET_DEFAULTS.has(widget)) return;
  WIDGET_DEFAULTS.set(widget, {
    type: widget.type,
    computeSize: widget.computeSize,
    computeSizeDescriptor: Object.getOwnPropertyDescriptor(widget, "computeSize"),
    hidden: widget.hidden,
    optionsHidden: widget.options?.hidden,
    optionsCanvasOnly: widget.options?.canvasOnly,
  });
}

function restoreComputeSize(widget, defaults) {
  if (defaults.computeSizeDescriptor) {
    Object.defineProperty(widget, "computeSize", defaults.computeSizeDescriptor);
  } else {
    delete widget.computeSize;
  }
}

function hideComputeSize(widget) {
  Object.defineProperty(widget, "computeSize", {
    value: hiddenWidgetComputeSize,
    configurable: true,
    writable: true,
    enumerable: false,
  });
}

export function setWidgetVisible(widget, visible) {
  if (!widget) return false;
  storeWidgetDefaults(widget);
  const defaults = WIDGET_DEFAULTS.get(widget);
  const wasHidden = widget.type === "hidden" || widget.hidden === true || widget.options?.hidden === true;
  if (visible) {
    widget.hidden = defaults.hidden ?? false;
    widget.type = defaults.type;
    restoreComputeSize(widget, defaults);
    if (widget.inputEl) widget.inputEl.style.display = "";
    if (widget.element) widget.element.style.display = "";
    if (defaults.optionsHidden === undefined) {
      deleteOption(widget, "hidden");
    } else {
      setOption(widget, "hidden", defaults.optionsHidden);
    }
    if (defaults.optionsCanvasOnly === undefined) {
      deleteOption(widget, "canvasOnly");
    } else {
      setOption(widget, "canvasOnly", defaults.optionsCanvasOnly);
    }
  } else {
    widget.hidden = true;
    widget.type = "hidden";
    hideComputeSize(widget);
    if (widget.inputEl) widget.inputEl.style.display = "none";
    if (widget.element) widget.element.style.display = "none";
    setOption(widget, "hidden", true);
    setOption(widget, "canvasOnly", true);
  }
  if (widget._state) {
    widget._state.hidden = widget.hidden;
    widget._state.type = widget.type;
  }
  widget.triggerDraw?.();
  const isHidden = widget.type === "hidden" || widget.hidden === true || widget.options?.hidden === true;
  return wasHidden !== isHidden;
}

// 条件显隐后强制刷新 widgets 数组引用（Vue 渲染器响应式跟踪需要）
export function refreshWidgetSnapshot(node) {
  if (!Array.isArray(node?.widgets)) return;
  try {
    node.widgets = [...node.widgets];
  } catch {
    // Old ComfyUI builds expose widgets as a plain mutable field.
  }
}

export function isWidgetVisible(widget) {
  return widget && widget.type !== "hidden" && widget.hidden !== true && widget.options?.hidden !== true;
}

// 条件显隐后按内容自适应节点高度（隐藏 widget 的 computeSize 收缩为 [0,-4]，
// 显隐变化后调用一次可避免节点底部留空）。graph 可省略。
export function fitNodeToContent(node, graph) {
  if (!node || node.flags?.collapsed) return;
  const width = Math.max(node.size?.[0] || 300, 300);
  const size = node.computeSize?.([width, node.size?.[1] || 0]);
  if (!size) return;
  node.setSize?.([Math.max(width, size[0]), size[1]]);
  node.setDirtyCanvas?.(true, true);
  graph?.setDirtyCanvas?.(true, true);
}
