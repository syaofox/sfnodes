// ==========================================================================
// SF LoRA Loader (Model Only) (+ 官方 LoraLoaderModelOnly) - Custom Node
// Standard widgets (lora_name combo + strength_model) plus an info icon that
// opens the shared metadata dialog (see sf_lora_info.js).
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    setupLoraInfoWidget,
    ensureEventHook,
} from "./sf_lora_info.js";

const NODE_TYPES = ["SFLoraLoaderModelOnly", "LoraLoaderModelOnly"];

app.registerExtension({
    name: "sfnodes.SFLoraLoaderModelOnly",
    nodeCreated(node) {
        if (!NODE_TYPES.includes(node.comfyClass)) return;
        ensureEventHook();
        setupLoraInfoWidget(node);
    },
});
