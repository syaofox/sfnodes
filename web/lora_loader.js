// ==========================================================================
// SF LoRA Loader (+ 官方 LoraLoader) - Custom Node
// Standard widgets (lora_name combo + strength_model + strength_clip) plus
// an info icon that opens the shared metadata dialog (see sf_lora_info.js).
// 官方节点仅前端挂件增强，不改 Python 行为。
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    setupLoraInfoWidget,
    ensureEventHook,
} from "./sf_lora_info.js";

const NODE_TYPES = ["SFLoraLoader", "LoraLoader"];

app.registerExtension({
    name: "sfnodes.SFLoraLoader",
    nodeCreated(node) {
        if (!NODE_TYPES.includes(node.comfyClass)) return;
        ensureEventHook();
        setupLoraInfoWidget(node);
    },
});
