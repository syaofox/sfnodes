import { app } from "/scripts/app.js";

function hexToRgb(hex) {
    if (!hex) return null;
    hex = hex.replace(/^#/, "");
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    const result = /^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
        ? {
              r: parseInt(result[1], 16),
              g: parseInt(result[2], 16),
              b: parseInt(result[3], 16),
          }
        : null;
}

function rgbToHex(r, g, b) {
    return "#" + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
    }).join("");
}

function getContrastColor(hexColor) {
    const rgb = hexToRgb(hexColor);
    if (!rgb) return "#ffffff";
    const brightness = (rgb.r * 299 + rgb.g * 587 + rgb.b * 114) / 1000;
    return brightness > 128 ? "#000000" : "#ffffff";
}

const SFColorPickerWidget = {
    COLOR: (key, val) => {
        const widget = {};
        widget.y = 0;
        widget.name = key;
        widget.type = "COLOR";

        let defaultColor = [255, 255, 255];
        if (Array.isArray(val) && val.length === 3) {
            defaultColor = val;
        } else if (typeof val === "string") {
            const rgb = hexToRgb(val);
            if (rgb) defaultColor = [rgb.r, rgb.g, rgb.b];
        }

        const defaultHex = rgbToHex(defaultColor[0], defaultColor[1], defaultColor[2]);
        widget.options = { default: defaultHex };
        widget.value = defaultHex;

        widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
            const hide = this.type !== "COLOR" && app.canvas.ds.scale > 0.5;
            if (hide) {
                return;
            }

            const border = 4;
            const H = height || 28;
            const margin = 10;

            ctx.fillStyle = "#1e1e1e";
            ctx.fillRect(0, widgetY, widgetWidth, H);

            ctx.fillStyle = this.value;
            ctx.beginPath();
            const x = margin;
            const y = widgetY + border;
            const w = widgetWidth - margin * 2 - 70;
            const h = H - border * 2;
            const radius = 4;
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + w - radius, y);
            ctx.arcTo(x + w, y, x + w, y + radius, radius);
            ctx.lineTo(x + w, y + h - radius);
            ctx.arcTo(x + w, y + h, x + w - radius, y + h, radius);
            ctx.lineTo(x + radius, y + h);
            ctx.arcTo(x, y + h, x, y + h - radius, radius);
            ctx.lineTo(x, y + radius);
            ctx.arcTo(x, y, x + radius, y, radius);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = "#444";
            ctx.lineWidth = 1;
            ctx.stroke();

            const rgb = hexToRgb(this.value);
            if (rgb) {
                ctx.fillStyle = getContrastColor(this.value);
            } else {
                ctx.fillStyle = "#fff";
            }
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            if (rgb) {
                ctx.fillText(`RGB(${rgb.r}, ${rgb.g}, ${rgb.b})`, x + 6, widgetY + H / 2 + 4);
            }
        };

        widget.mouse = function (e, pos, node) {
            if (e.type === "pointerdown") {
                const margin = 10;
                const widgetWidth = node.size[0];
                const colorAreaWidth = widgetWidth - margin * 2 - 70;

                if (pos[0] >= margin && pos[0] <= margin + colorAreaWidth) {
                    const picker = document.createElement("input");
                    picker.type = "color";
                    picker.value = this.value;

                    picker.style.position = "absolute";
                    picker.style.left = "-9999px";
                    picker.style.top = "-9999px";

                    document.body.appendChild(picker);

                    picker.addEventListener("input", () => {
                        this.value = picker.value;
                        node.setDirtyCanvas(true, true);
                    });

                    picker.addEventListener("change", () => {
                        this.value = picker.value;
                        node.graph._version++;
                        node.setDirtyCanvas(true, true);
                        picker.remove();
                    });

                    picker.addEventListener("blur", () => {
                        picker.remove();
                    });

                    picker.click();
                    return true;
                }
            }
            return false;
        };

        widget.computeSize = function (width) {
            return [width, 28];
        };

        return widget;
    }
};

// 新版 Vue 前端将 COLOR 收编为内置 widget（widgetStore 合并时 core 覆盖同名
// 自定义注册），实际渲染的是内置 ColorWidget：value 必须是 hex 字符串，数组
// 默认值会显示为 "0,0,0" 且无色块。后端 execute 对 hex 字符串与 [r,g,b] 数组
// 均兼容，这里只负责把 widget 值统一规整为 hex。
const COLOR_PICKER_CLASSES = new Set(["SFMaskFill", "SFImageResizePlus"]);

function toHexColor(val) {
    if (typeof val === "string") {
        const rgb = hexToRgb(val);
        return rgb ? rgbToHex(rgb.r, rgb.g, rgb.b) : null;
    }
    if (Array.isArray(val) && val.length === 3 && val.every(Number.isFinite)) {
        return rgbToHex(
            Math.round(val[0]),
            Math.round(val[1]),
            Math.round(val[2])
        );
    }
    return null;
}

function normalizeColorWidgets(node) {
    for (const w of node.widgets || []) {
        if (String(w.type).toLowerCase() !== "color") continue;
        const hex = toHexColor(w.value);
        if (hex) w.value = hex;
    }
}

app.registerExtension({
    name: "sfnodes.SFColorPicker",

    init() {
        console.log("SF Color Picker loaded");
    },

    getCustomWidgets() {
        return {
            COLOR: (node, inputName, inputData) => {
                let defaultValue = [255, 255, 255];
                const raw = inputData && inputData[1] && inputData[1].default;
                if (Array.isArray(raw) && raw.length === 3) {
                    defaultValue = raw;
                } else if (typeof raw === "string") {
                    const rgb = hexToRgb(raw);
                    if (rgb) defaultValue = [rgb.r, rgb.g, rgb.b];
                }
                return {
                    widget: node.addCustomWidget(
                        SFColorPickerWidget.COLOR(inputName, defaultValue)
                    ),
                    minWidth: 150,
                    minHeight: 30,
                };
            }
        };
    },

    async nodeCreated(node) {
        if (!COLOR_PICKER_CLASSES.has(node.comfyClass)) return;
        normalizeColorWidgets(node);
        const configure = node.configure;
        node.configure = function () {
            const result = configure.apply(this, arguments);
            normalizeColorWidgets(this);
            return result;
        };
    },

    loadedGraphNode(node) {
        if (COLOR_PICKER_CLASSES.has(node.comfyClass)) {
            normalizeColorWidgets(node);
        }
    }
});

export { SFColorPickerWidget };
