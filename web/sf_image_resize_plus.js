import { app } from "/scripts/app.js";

app.registerExtension({
    name: "sfnodes.SFImageResizePlus",
    async nodeCreated(node) {
        if (node.comfyClass !== "SFImageResizePlus") return;
        const find = (n) => node.widgets.find((w) => w.name === n);
        const modeW = find("size_mode");
        const widthW = find("width");
        const heightW = find("height");
        const tpW = find("total_pixels");
        const methodW = find("method");
        const cropW = find("crop_position");
        const padW = find("pad_color");
        if (!modeW || !widthW || !heightW || !tpW || !methodW || !cropW || !padW)
            return;

        const toggle = () => {
            const byPixels = modeW.value === "total pixels";
            widthW.hidden = byPixels;
            heightW.hidden = byPixels;
            tpW.hidden = !byPixels;
            // method 联动：crop_position 仅 fill / crop 生效，pad_color 仅 pad 生效
            const method = methodW.value;
            cropW.hidden = method !== "fill / crop";
            padW.hidden = method !== "pad";
            if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        };

        const wrapCallback = (widget) => {
            const orig = widget.callback;
            widget.callback = function (...args) {
                if (orig) orig.apply(this, args);
                toggle();
            };
        };
        wrapCallback(modeW);
        wrapCallback(methodW);
        const origConfigure = node.configure;
        node.configure = function (data) {
            // size_mode 置顶重排（v2026-09-05）前旧工作流 widgets_values 为 8 项：
            // [width, height, interpolation, method, condition, divisible_by,
            //  crop_position, pad_color] —— 插入新前缀补齐新 10 项顺序
            if (
                data &&
                Array.isArray(data.widgets_values) &&
                data.widgets_values.length === 8
            ) {
                data.widgets_values = [
                    "width & height",
                    data.widgets_values[0],
                    data.widgets_values[1],
                    1.0,
                    ...data.widgets_values.slice(2),
                ];
            }
            const result = origConfigure
                ? origConfigure.apply(this, arguments)
                : undefined;
            setTimeout(toggle, 0);
            return result;
        };
        const origOnAG = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function (...args) {
            if (origOnAG) origOnAG.apply(this, args);
            toggle();
        };

        toggle();
    },
});
