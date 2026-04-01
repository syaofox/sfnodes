// ==========================================================================
// Empty Latent By Aspect Ratio - Dynamic Resolution List
// ==========================================================================
// 
// Description:
// JavaScript extension for EmptyLatentByAspectRatio node that enables
// dynamic updating of resolution options based on selected aspect ratio.
// 
// Features:
// - When aspect ratio changes, resolution dropdown updates automatically
// - Maintains selected resolution if it exists in new aspect ratio options
// - Falls back to first option if current selection is not available
// 
// ==========================================================================

import { app } from "/scripts/app.js";

// 定义宽高比和对应的分辨率（与Python代码保持一致）
const ASPECT_RATIOS = {
    "4:3": ["512x384", "640x480", "1024x768", "1280x960", "1600x1200", "2048x1536"],
    "3:4": ["384x512", "480x640", "768x1024", "960x1280", "1200x1600", "1536x2048"],
    "16:9": ["640x360", "960x540", "1280x720", "1920x1080", "2560x1440", "3840x2160"],
    "9:16": ["360x640", "540x960", "720x1280", "1080x1920", "1440x2560", "2160x3840"],
    "1:1": ["256x256", "384x384", "512x512", "768x768", "1024x1024", "1536x1536", "2048x2048"]
};

app.registerExtension({
    name: "sfnodes.EmptyLatentByAspectRatio",

    nodeCreated(node) {
        if (node.comfyClass === "SFEmptyLatentByAspectRatio") {
            // 查找控件
            const aspectRatioWidget = node.widgets.find(w => w.name === "aspect_ratio");
            const resolutionWidget = node.widgets.find(w => w.name === "resolution");
            const modelTypeWidget = node.widgets.find(w => w.name === "model_type");

            if (!aspectRatioWidget || !resolutionWidget) {
                return;
            }

            // 更新分辨率选项的函数
            const updateResolutionOptions = (selectedAspectRatio) => {
                const resolutions = ASPECT_RATIOS[selectedAspectRatio] || ASPECT_RATIOS["4:3"];
                const currentResolution = resolutionWidget.value;

                // 更新分辨率下拉列表的选项
                // ComfyUI的combo widget使用options属性存储选项
                if (resolutionWidget.options) {
                    resolutionWidget.options.values = resolutions;
                } else {
                    // 如果options不存在，尝试直接设置
                    resolutionWidget.options = { values: resolutions };
                }

                // 如果当前选择的分辨率在新列表中，保持选择
                // 否则选择第一个选项
                if (resolutions.includes(currentResolution)) {
                    resolutionWidget.value = currentResolution;
                } else {
                    resolutionWidget.value = resolutions[0];
                }

                // 如果widget有updateOptions方法，调用它来更新UI
                if (resolutionWidget.updateOptions) {
                    resolutionWidget.updateOptions();
                }

                // 触发节点更新
                node.setDirtyCanvas(true, true);
            };

            // 监听宽高比变化
            // 保存原始回调
            const originalCallback = aspectRatioWidget.callback;
            
            // 设置新的回调
            aspectRatioWidget.callback = function(value) {
                // 调用原始回调（如果有）
                if (originalCallback) {
                    originalCallback.call(this, value);
                }
                // 更新分辨率选项
                updateResolutionOptions(value);
            };

            // 监听模型类型变化，触发节点更新
            if (modelTypeWidget) {
                const originalModelTypeCallback = modelTypeWidget.callback;
                modelTypeWidget.callback = function(value) {
                    if (originalModelTypeCallback) {
                        originalModelTypeCallback.call(this, value);
                    }
                    node.setDirtyCanvas(true, true);
                };
            }

            // 初始化：确保分辨率选项与当前选择的宽高比匹配
            updateResolutionOptions(aspectRatioWidget.value);
        }
    },
});
