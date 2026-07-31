// ==========================================================================
// SF LoRA Loader - Custom Node
// Standard widgets (lora_name combo + strength_model + strength_clip) plus
// an info icon that opens the shared metadata dialog (see sf_lora_info.js).
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    setupLoraInfoWidget,
    ensureEventHook,
} from "./sf_lora_info.js";

const NODE_TYPE = "SFLoraLoader";

app.registerExtension({
    name: "sfnodes.SFLoraLoader",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;
        ensureEventHook();
        setupLoraInfoWidget(node);
    },
});
