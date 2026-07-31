// ==========================================================================
// SF LoRA Loader (Model Only) - Custom Node
// Standard widgets (lora_name combo + strength_model) plus an info icon that
// opens the shared metadata dialog (see sf_lora_info.js).
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    getLoraMetadata,
    loraMetadataCache,
    showLoraInfoDialog,
    ensureEventHook,
} from "./sf_lora_info.js";

const NODE_TYPE = "SFLoraLoaderModelOnly";
const INVALID_BOUNDS = [0, -1];

function getLoraCombo(node) {
    return node.widgets?.find((w) => w.name === "lora_name") || null;
}

function getLoraName(node) {
    const combo = getLoraCombo(node);
    const v = combo?.value;
    return typeof v === "string" ? v : null;
}

function isLowQuality() {
    return (app.canvas.ds?.scale || 1) <= 0.5;
}

function createInfoWidget() {
    const w = {
        name: "_info",
        type: "custom",
        options: { serialize: false },
        value: {},
        y: 0,
        last_y: 0,
        _hit: INVALID_BOUNDS,
        computeSize(width) { return [width, 24]; },
        draw(ctx, n, width, posY, height) {
            this.last_y = posY;
            this._hit = INVALID_BOUNDS;
            const loraName = getLoraName(n);
            if (!loraName || loraName === "None") return;
            const cachedMeta = loraMetadataCache.get(loraName);
            const hasCustom = cachedMeta?._has_custom;
            const size = Math.max(14, height * 0.6);
            const posX = 10;
            const centerX = posX + size / 2;
            const midY = posY + height * 0.5;
            this._hit = [posX, size + 6];
            ctx.save();
            ctx.beginPath();
            ctx.arc(centerX, midY, size / 2 - 0.5, 0, Math.PI * 2);
            if (hasCustom) {
                ctx.fillStyle = "rgba(79,195,247,0.3)";
                ctx.strokeStyle = "rgba(79,195,247,0.7)";
            } else {
                ctx.fillStyle = "rgba(255,255,255,0.25)";
                ctx.strokeStyle = "rgba(255,255,255,0.4)";
            }
            ctx.lineWidth = 1;
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = hasCustom ? "rgba(79,195,247,0.9)" : "rgba(255,255,255,0.6)";
            ctx.font = `${Math.round(size * 0.6)}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("i", centerX, midY + 0.5);
            if (!isLowQuality()) {
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                ctx.textAlign = "left";
                ctx.fillText("Info", posX + size + 6, midY);
            }
            ctx.restore();
        },
        mouse(event, pos, n) {
            if (event.type !== "pointerdown") return false;
            const b = w._hit;
            if (b[1] < 0) return false;
            if (pos[0] >= b[0] && pos[0] <= b[0] + b[1]) {
                const loraName = getLoraName(n);
                if (loraName && loraName !== "None") {
                    // 延迟到 pointerup 由 canvas 处理完成后再打开对话框，
                    // 避免 DOM 遮罩在点击过程中出现导致 LiteGraph widget 交互状态残留
                    getLoraMetadata(loraName).then((meta) => {
                        requestAnimationFrame(() => {
                            setTimeout(() => showLoraInfoDialog(event, loraName, meta, "loras"), 0);
                        });
                    });
                }
                return true;
            }
            return false;
        },
    };
    return w;
}

function setupNode(node) {
    // Bind combo callback: prefetch metadata on user change
    const combo = getLoraCombo(node);
    if (combo) {
        const origCallback = combo.callback;
        combo.callback = (value) => {
            if (origCallback) origCallback(value);
            if (value && value !== "None") getLoraMetadata(value);
        };
    }

    // Custom widgets must not interfere with the positional restoration of
    // widgets_values, so remove _info before the original configure runs and
    // rebuild it afterwards (same pattern as SFPowerLoraLoader).
    const _origConfigure = node.configure;
    node.configure = function (info) {
        const idx = this.widgets?.findIndex((w) => w.name === "_info") ?? -1;
        if (idx !== -1) this.widgets.splice(idx, 1);
        if (_origConfigure) _origConfigure.call(this, info);
        // On workflow load, widget values are restored in configure() after
        // onNodeCreated, so prefetch the restored value here.
        const loraName = getLoraName(this);
        if (loraName && loraName !== "None") getLoraMetadata(loraName);
        this.widgets.push(createInfoWidget());
    };

    node.widgets.push(createInfoWidget());
}

app.registerExtension({
    name: "sfnodes.SFLoraLoaderModelOnly",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;
        ensureEventHook();
        setupNode(node);
    },
});
