// SFVOSR2Upscale / SFVOSR2Video 目标尺寸模式联动显隐。
// 复用共享库 sf_widget_visibility_lib（模式切换时隐藏无关参数 widget，
// 保持节点高度紧凑）；两个节点共用同一份映射，避免内联副本。
import { app } from "/scripts/app.js";
import { refreshWidgetSnapshot, setWidgetVisible } from "./sf_widget_visibility_lib.js";

const NODE_CLASSES = new Set(["SFVOSR2Upscale", "SFVOSR2Video"]);

// size_mode -> 生效的参数 widget
export const MODE_WIDGETS = {
    "scale": ["scale"],
    "total pixels": ["total_pixels"],
    "longer dimension": ["longer_size"],
    "shorter dimension": ["shorter_size"],
};

export const SIZE_WIDGET_NAMES = ["scale", "total_pixels", "longer_size", "shorter_size"];

// 按当前 mode 显隐参数 widget；返回是否有可见性变化
export function applySizeModeVisibility(node) {
    if (!node?.widgets) return false;
    const modeWidget = node.widgets.find((w) => w.name === "size_mode");
    if (!modeWidget) return false;
    const visible = MODE_WIDGETS[modeWidget.value] || [];
    let changed = false;
    for (const name of SIZE_WIDGET_NAMES) {
        const widget = node.widgets.find((w) => w.name === name);
        if (!widget) continue;
        changed = setWidgetVisible(widget, visible.includes(name)) || changed;
    }
    return changed;
}

app.registerExtension({
    name: "sfnodes.SFVOSR2Size",
    async nodeCreated(node) {
        if (!NODE_CLASSES.has(node.comfyClass)) return;
        const modeWidget = node.widgets?.find((w) => w.name === "size_mode");
        if (!modeWidget) return;

        const toggle = () => {
            if (applySizeModeVisibility(node)) refreshWidgetSnapshot(node);
            node.setDirtyCanvas?.(true, true);
        };

        const orig = modeWidget.callback;
        modeWidget.callback = function (...args) {
            if (orig) orig.apply(this, args);
            toggle();
        };
        toggle();
    },
});
