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
        const multW = find("multiplier");
        const longW = find("longer_size");
        const shortW = find("shorter_size");
        const multipleW = find("multiple");
        const divW = find("divisible_by");
        const methodW = find("method");
        const cropW = find("crop_position");
        const padW = find("pad_color");
        if (
            !modeW || !widthW || !heightW || !tpW || !multW || !longW ||
            !shortW || !multipleW || !divW || !methodW || !cropW || !padW
        )
            return;

        const toggle = () => {
            const mode = modeW.value;
            const isWH = mode === "width & height";
            const isTP = mode === "total pixels";
            const isMul = mode === "scale by multiplier";
            const isLong = mode === "longer dimension";
            const isShort = mode === "shorter dimension";
            const isMultiple = mode === "scale to multiple";
            widthW.hidden = !isWH;
            heightW.hidden = !isWH;
            tpW.hidden = !isTP;
            multW.hidden = !isMul;
            longW.hidden = !isLong;
            shortW.hidden = !isShort;
            multipleW.hidden = !isMultiple;
            // scale to multiple 的倍数网格由 multiple 独占，divisible_by 不参与
            divW.hidden = isMultiple;
            // method 联动：scale to multiple 内部强制 cover，隐藏 method；
            // crop_position 仅 fill / crop 生效，pad_color 仅 pad 生效
            methodW.hidden = isMultiple;
            const method = methodW.value;
            cropW.hidden = isMultiple || method !== "fill / crop";
            padW.hidden = isMultiple || method !== "pad";
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
            // 新增 4 个尺寸模式参数（multiplier/longer_size/shorter_size/
            // multiple）前，10 项顺序为：[size_mode, width, height,
            //  total_pixels, interpolation, method, condition, divisible_by,
            //  crop_position, pad_color] —— 在 total_pixels 之后插入 4 项。
            if (
                data &&
                Array.isArray(data.widgets_values) &&
                data.widgets_values.length === 10
            ) {
                const v = data.widgets_values;
                data.widgets_values = [
                    ...v.slice(0, 4),
                    1.0,
                    512,
                    512,
                    8,
                    ...v.slice(4),
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
