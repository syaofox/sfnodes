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

app.registerExtension({
    name: "sfnodes.SFColorPicker",

    init() {
        console.log("SF Color Picker loaded");
    },

    getCustomWidgets() {
        return {
            COLOR: (node, inputName, inputData) => {
                let defaultValue = [255, 255, 255];
                if (inputData && inputData[1] && inputData[1].default) {
                    const val = inputData[1].default;
                    if (Array.isArray(val) && val.length === 3) {
                        defaultValue = val;
                    }
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
        if (node.comfyClass === "SFMaskFillColor") {
            const colorWidget = node.widgets.find(w => w.name === "fill_color");
            if (colorWidget) {
                const serialize = node.serialize;
                node.serialize = function () {
                    const data = serialize.call(this);
                    if (this.widgets) {
                        const cw = this.widgets.find(w => w.name === "fill_color");
                        if (cw && cw.value) {
                            const rgb = hexToRgb(cw.value);
                            if (rgb) {
                                data.widgets_data = data.widgets_data || {};
                                const idx = this.widgets.indexOf(cw);
                                if (idx !== -1) {
                                    data.widgets_data[idx] = [rgb.r, rgb.g, rgb.b];
                                }
                            }
                        }
                    }
                    return data;
                };

                const configure = node.configure;
                node.configure = function (data) {
                    configure.apply(this, arguments);
                    if (data.widgets_data) {
                        for (let i = 0; i < this.widgets.length; i++) {
                            if (this.widgets[i].name === "fill_color" && Array.isArray(data.widgets_data[i])) {
                                const rgb = data.widgets_data[i];
                                this.widgets[i].value = rgbToHex(rgb[0], rgb[1], rgb[2]);
                            }
                        }
                    }
                };
            }
        }
    }
});

export { SFColorPickerWidget };
