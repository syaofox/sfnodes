import { app } from "/scripts/app.js";

app.registerExtension({
    name: "sfnodes.SFMaskFill",
    async nodeCreated(node) {
        if (node.comfyClass !== "SFMaskFill") return;
        const find = (n) => node.widgets.find((w) => w.name === n);
        const modeW = find("fill_mode");
        const colorW = find("fill_color");
        const opacityW = find("opacity");
        if (!modeW || !colorW || !opacityW) return;

        const toggle = () => {
            const show = modeW.value === "color";
            // LiteGraph widget hidden flag + DOM hidden handling
            colorW.hidden = !show;
            opacityW.hidden = !show;
            // also hide via type trick for nodes that check widget.type
            // keep original type but set hidden flag is sufficient for ComfyUI rendering
            if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
            // force node size recompute
            if (node.computeSize) {
                const sz = node.computeSize();
                if (sz && node.size) {
                    // do not shrink below computed, LiteGraph handles auto
                }
            }
        };

        // wrap callback
        const origCb = modeW.callback;
        modeW.callback = function (...args) {
            if (origCb) origCb.apply(this, args);
            toggle();
        };
        // also patch configure paths: after node configure values restored, re-apply
        const origConfigure = node.configure;
        node.configure = function (...args) {
            if (origConfigure) origConfigure.apply(this, args);
            // delay to ensure widgets values updated
            setTimeout(toggle, 0);
            return undefined;
        };
        // onAfterGraphConfigured ensures workflow load restores correctly
        const origOnAG = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function (...args) {
            if (origOnAG) origOnAG.apply(this, args);
            toggle();
        };

        // initial
        toggle();
    },
});
