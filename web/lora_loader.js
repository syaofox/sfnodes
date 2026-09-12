// ==========================================================================
// SF LoRA Loader (+ 官方 LoraLoader) - Custom Node
// Standard widgets (lora_name combo + strength_model + strength_clip) plus
// an info icon that opens the shared metadata dialog (see sf_lora_info.js).
// 官方节点仅前端挂件增强，不改 Python 行为。
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    setupLoaderInfoWidget,
    ensureEventHook,
    isOfficialInfoEnabled,
    registerOfficialInfoSettingOnce,
    registerOfficialInfoSpec,
} from "./sf_lora_info.js";

const NODE_TYPES = ["SFLoraLoader", "LoraLoader"];
const OFFICIAL_OPTS = { enabledOf: () => isOfficialInfoEnabled() };

app.registerExtension({
    name: "sfnodes.SFLoraLoader",
    init() {
        registerOfficialInfoSettingOnce();
        registerOfficialInfoSpec({ classes: ["LoraLoader"], comboName: "lora_name", opts: OFFICIAL_OPTS });
    },
    nodeCreated(node) {
        if (!NODE_TYPES.includes(node.comfyClass)) return;
        ensureEventHook();
        // SF 节点恒挂载；官方节点受 sfnodes.OfficialInfo.Enabled 门控。
        if (node.comfyClass === "LoraLoader") setupLoaderInfoWidget(node, "lora_name", OFFICIAL_OPTS);
        else setupLoaderInfoWidget(node);
    },
});
