// ==========================================================================
// Canvas Size Preset - model -> resolution 动态联动
// ==========================================================================
//
// 节点：SFCanvasSizePreset（nodes/utils/canvas_size.py）
// 数据源：GET /api/sfnodes/canvas_size_presets（Python 常量唯一真源，本文件
// 不内联副本——规则 14）。请求失败时降级为静态选项（INPUT_TYPES 默认模型
// 的完整列表，联动不可用但不破坏节点）。
//
// 联动时机：
//  - nodeCreated：初始渲染（静态选项已含默认模型列表，无需等待 API）
//  - model widget callback：切换模型触发重建（API 就绪后按新 model 精确重建）
//  - onAfterGraphConfigured：加载工作流时 widget 值已恢复，combo options
//    不随工作流保存（text_preset 经验），需按恢复的 model 值重建
// ==========================================================================

import { app } from "/scripts/app.js";
import { sfApiUrl } from "./sf_common.js";

const CLASS = "SFCanvasSizePreset";

// 分组头判定（ComfyUI combo 原生分组：--x-- 只显示不可选）
function isTierHeader(value) {
  return typeof value === "string" && value.startsWith("--") && value.endsWith("--");
}

// 预设表加载：模块级 promise 缓存，防重复请求；失败 resolves null（降级）
let _presetsPromise = null;
function loadPresets() {
  if (!_presetsPromise) {
    _presetsPromise = fetch(sfApiUrl("/api/sfnodes/canvas_size_presets"))
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
  }
  return _presetsPromise;
}

app.registerExtension({
  name: "sfnodes.CanvasSizePreset",

  nodeCreated(node) {
    if (node.comfyClass !== CLASS) return;

    const modelWidget = node.widgets.find((w) => w.name === "model");
    const resolutionWidget = node.widgets.find((w) => w.name === "resolution");
    if (!modelWidget || !resolutionWidget) return;

    // 重建 resolution 选项。data 为 null（API 失败）时跳过——保持静态选项。
    const applyValues = (data) => {
      const values = data?.values?.[modelWidget.value];
      if (!Array.isArray(values) || values.length === 0) return;
      const current = resolutionWidget.value;
      const keep = values.includes(current);
      resolutionWidget.options = Object.assign({}, resolutionWidget.options, { values });
      if (resolutionWidget.updateOptions) {
        resolutionWidget.updateOptions();
      }
      if (!keep) {
        // 当前选择不在新列表：回退第一个非分组头选项
        resolutionWidget.value = values.find((v) => !isTierHeader(v)) ?? values[0];
      }
      node.setDirtyCanvas(true, true);
    };

    // 数据就绪后按当前 model 重建（模型未切换时幂等；API 慢于 nodeCreated）。
    // 注意：绝不在这里用"当前静态选项"充当任意 model 的表——静态表只是
    // INPUT_TYPES 默认模型（Z-Image）的列表，套到其他 model 上会把 Z-Image
    // 的 33 项选项错误地显示给 Wan2.2（fetch 失败时该错误永久化）。
    // API 未就绪期间选项保持切换前的表，就绪后由 applyValues 精确重建。
    const syncFromData = () => loadPresets().then(applyValues);

    const originalCallback = modelWidget.callback;
    modelWidget.callback = function (value) {
      if (originalCallback) {
        originalCallback.call(this, value);
      }
      syncFromData();
    };

    // 加载/恢复工作流：widget 值恢复发生在 onAfterGraphConfigured，combo
    // options 需按恢复的 model 值重建（值已恢复，不触发 callback）
    node.onAfterGraphConfigured = () => {
      syncFromData();
    };

    // 初始：API 就绪后校准一次（默认模型静态选项已正确，此步幂等兜底）
    syncFromData();
  },
});
