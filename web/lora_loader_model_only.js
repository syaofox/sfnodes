// ==========================================================================
// SF LoRA Loader (Model Only) (+ 官方 LoraLoaderModelOnly) - Custom Node
// Standard widgets (lora_name combo + strength_model) plus an info icon that
// opens the shared metadata dialog (see sf_lora_info.js).
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    setupLoaderInfoWidget,
    ensureEventHook,
    isOfficialInfoEnabled,
    registerOfficialInfoSettingOnce,
    registerOfficialInfoSpec,
} from "./sf_lora_info.js";

const NODE_TYPES = ["SFLoraLoaderModelOnly", "LoraLoaderModelOnly"];
const OFFICIAL_OPTS = { enabledOf: () => isOfficialInfoEnabled() };

app.registerExtension({
    name: "sfnodes.SFLoraLoaderModelOnly",
    init() {
        registerOfficialInfoSettingOnce();
        registerOfficialInfoSpec({ classes: ["LoraLoaderModelOnly"], comboName: "lora_name", opts: OFFICIAL_OPTS });
    },
    nodeCreated(node) {
        if (!NODE_TYPES.includes(node.comfyClass)) return;
        ensureEventHook();
        // SF 节点恒挂载；官方节点受 sfnodes.OfficialInfo.Enabled 门控。
        if (node.comfyClass === "LoraLoaderModelOnly") setupLoaderInfoWidget(node, "lora_name", OFFICIAL_OPTS);
        else setupLoaderInfoWidget(node);
    },
});
