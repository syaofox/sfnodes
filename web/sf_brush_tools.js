// ==========================================================================
// sf_brush_tools.js - 画笔节点共享工具（步长设置 / [ ] 快捷键 / 步进器滚轮）
// ==========================================================================
//
// SFImageBrushMask 与 SFImageCropExpandBrushMask 共用（原 sf_brush_mask.js 内联
// 实现提升）：
//   - 步长设置读取/注册（sfnodes.BrushMask.SizeStep/OpacityStep，两节点共享
//     同一组用户设置键）；步进纯函数仍在 sf_brush_mask_lib.js。
//   - [ ] 画笔尺寸快捷键：双通道互补——① 官方通道（画布 processKey 分发选中
//     节点的 onKeyDown，宿主用返回的 keyStep）② window 冒泡监听兜底。同一
//     物理按键会走两遍，经 e.timeStamp 去重只执行一次（多选同理：首个节点的
//     applyAction 全量调整后，其余同戳调用直接跳过）。
//   - S±/O± 悬停滚轮快调：引擎无节点级 onMouseWheel 钩子，用 window capture
//     先手拦截（installPasteHandler 同款）；仅命中步进器且无 Ctrl/Meta 时
//     preventDefault+stopPropagation，其余放行画布缩放。
//
// 全局监听只装一次（模块级注册表）；多类共存时各自响应自己选中的节点。
// 宿主勿内联 stepBrushSize：action 一律经各自 buttonAction 统一入口，保证
// 步长设置三路同源（曾漏传设置值）。
// ==========================================================================

import { app } from "/scripts/app.js";
import { getSelectedNodes } from "./sf_canvas_align_lib.js";
import { hitStepper, wheelDir, wheelAction } from "./sf_brush_mask_lib.js";

// ── 步长设置（ComfyUI 设置页；init 幂等注册；读取失败回默认值）─────────────

export const SIZE_STEP_SETTING = "sfnodes.BrushMask.SizeStep";
export const OPA_STEP_SETTING = "sfnodes.BrushMask.OpacityStep";
export const SIZE_STEP_DEFAULT = 2;
export const OPA_STEP_DEFAULT = 5; // 整数百分比

export function brushSizeStep() {
  try {
    const v = Math.round(Number(app.ui.settings.getSettingValue(SIZE_STEP_SETTING)));
    return Number.isFinite(v) && v >= 1 && v <= 20 ? v : SIZE_STEP_DEFAULT;
  } catch {
    return SIZE_STEP_DEFAULT;
  }
}

export function brushOpacityStep() {
  try {
    const v = Math.round(Number(app.ui.settings.getSettingValue(OPA_STEP_SETTING)));
    return Number.isFinite(v) && v >= 1 && v <= 25 ? v : OPA_STEP_DEFAULT;
  } catch {
    return OPA_STEP_DEFAULT;
  }
}

let _brushStepSettingsRegistered = false;
export function registerBrushStepSettings() {
  if (_brushStepSettingsRegistered) return;
  _brushStepSettingsRegistered = true;
  try {
    app.ui.settings.addSetting({
      id: SIZE_STEP_SETTING,
      name: "SF Image Brush Mask: brush size step (Size± buttons, wheel, [ ] keys)",
      defaultValue: SIZE_STEP_DEFAULT,
      type: "slider",
      attrs: { min: 1, max: 20, step: 1 },
    });
    app.ui.settings.addSetting({
      id: OPA_STEP_SETTING,
      name: "SF Image Brush Mask: opacity step percent (Opa± buttons, wheel)",
      defaultValue: OPA_STEP_DEFAULT,
      type: "slider",
      attrs: { min: 1, max: 25, step: 1 },
    });
  } catch {
    // 设置系统不可用则退化为默认值
  }
}

// ── [ ] 快捷键 + 步进器滚轮（模块级注册表，单窗口监听）────────────────────

const _keyEntries = [];
let _keysInstalled = false;
const _lastKey = { t: -1, k: "" };

function _matches(node, cfg) {
  if (!node) return false;
  return cfg.classNames.includes(node.comfyClass) || cfg.classNames.includes(node.type);
}

function _keyStep(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  if (e.key !== "[" && e.key !== "]") return false;
  if (e.timeStamp === _lastKey.t && e.key === _lastKey.k) return true;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return false;
  if (typeof document !== "undefined" && document.querySelector && document.querySelector(".sf-px-overlay")) return false;
  _lastKey.t = e.timeStamp;
  _lastKey.k = e.key;
  // 经宿主 buttonAction 统一入口（步长设置三路同源，勿内联 stepBrushSize
  // ——曾漏传设置值）
  const action = e.key === "]" ? "sizePlus" : "sizeMinus";
  let hit = false;
  for (const cfg of _keyEntries) {
    for (const n of getSelectedNodes(app)) {
      if (!_matches(n, cfg) || !n.properties) continue;
      cfg.applyAction(n, action);
      hit = true;
    }
  }
  return hit;
}

// registerBrushSizeKeys({ classNames, controlsProp, applyAction })
//   classNames: 参与的节点类名数组
//   controlsProp: 宿主控件几何数组属性名（滚轮命中用，如 "_sfBrushCtrls"）
//   applyAction(node, action): action ∈ {sizeMinus, sizePlus}
// 返回 keyStep(e) → bool 供宿主 onKeyDown 使用（官方通道）。
export function registerBrushSizeKeys(cfg) {
  _keyEntries.push(cfg);
  if (!_keysInstalled) {
    _keysInstalled = true;
    window.addEventListener("keydown", (e) => {
      if (_keyStep(e)) {
        e.preventDefault();
        e.stopPropagation();
      }
    });
    window.addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) return;
      const dir = wheelDir(e.deltaY);
      if (!dir) return;
      const canvas = app.canvas;
      if (!canvas) return;
      const mx = canvas.graph_mouse?.[0], my = canvas.graph_mouse?.[1];
      if (mx == null || my == null) return;
      for (const entry of _keyEntries) {
        if (!entry.controlsProp) continue;
        for (const n of app.graph?._nodes || []) {
          if (!_matches(n, entry)) continue;
          if (n.flags?.collapsed || !n[entry.controlsProp]) continue;
          const lx = mx - n.pos[0], ly = my - n.pos[1];
          if (lx < 0 || ly < 0 || lx > n.size[0] || ly > n.size[1]) continue;
          const hovered = hitStepper(n[entry.controlsProp], lx, ly);
          if (!hovered) continue;
          const action = wheelAction(hovered, dir);
          if (!action) continue;
          e.preventDefault();
          e.stopPropagation();
          entry.applyAction(n, action);
          return;
        }
      }
    }, { passive: false, capture: true });
    // 折叠节点不可见由上方 flags.collapsed 跳过；输入框内按键在 _keyStep 判定
  }
  return (e) => _keyStep(e);
}
