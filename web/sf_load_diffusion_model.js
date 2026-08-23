// ==========================================================================
// SF Load Diffusion Model - Custom Node
// 官方 UNETLoader 同款 combo + weight_dtype，外加 i 信息图标：点击打开
// SF LoRA Stack 同款浮动信息面板（openInfoPanelFor 宿主适配）。
//
// 与 LoRA 域的差异全部经 ctx 注入，面板本体零分支复用：
//   api:        sf_dmodel_api.js 的 dmodel 路由束
//   hideTriggers: diffusion 模型无触发词概念
//   samplesKind:  sample/ 目录路由按 kind=diffusion_models 解析
//   autoCivitai:  面板打开即自动匹配（LoRA 面板仍手动 ↻）
// ==========================================================================
import { app } from "/scripts/app.js";
import { setupLoaderInfoWidget, ensureEventHook } from "./sf_lora_info.js";
import { isGraphLoading } from "./sf_common.js";
import { getNodeRect } from "./sf_lora_stack_settings.js";
import { openInfoPanelFor } from "./sf_lora_stack_info.js";
import { dmodelApi } from "./sf_dmodel_api.js";

const NODE_TYPE = "SFLoadDiffusionModel";

// 有用户自定义数据的模型名集合（i 图标高亮）。数据以保存事件驱动：
// 任一面板/宿主广播 sfnodes.model-data-changed 即标记。会话级记忆，
// 初次加载时不预取（避免为高亮而逐个请求 info）。
const _hasData = new Set();
if (typeof document !== "undefined") {
    document.addEventListener("sfnodes.model-data-changed", (e) => {
        const name = e?.detail?.name;
        if (name) _hasData.add(name);
    });
}

// 面板宿主上下文（loaderPanelCtx 样板 + dmodel 域注入）。行状态存本模块
// 会话 Map（与 LoRA loader 同策略）：i 图标面板不承载值通道，工作流不携带。
const _rows = new Map();

function dmodelPanelCtx(node, modelName) {
    const rowKey = node.id + ":" + modelName;
    if (!_rows.has(rowKey)) {
        _rows.set(rowKey, { id: modelName, name: modelName, triggers: [], custom: [] });
    }
    return {
        key: "dmodel:" + rowKey,
        node,
        anchorRect: () => {
            try {
                const w = node.widgets?.find((x) => x.name === "_info");
                const el = w?.element;
                if (el && el.getBoundingClientRect) {
                    const r = el.getBoundingClientRect();
                    if (r && r.width && r.height) return r;
                }
            } catch {}
            return getNodeRect(node);
        },
        getRow: () => _rows.get(rowKey) || { id: modelName, name: modelName, triggers: [], custom: [] },
        patchRow: (patch) => {
            const cur = _rows.get(rowKey) || { id: modelName, name: modelName, triggers: [], custom: [] };
            Object.assign(cur, patch);
            cur.id = modelName;
            cur.name = modelName;
            _rows.set(rowKey, cur);
            try { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); } catch {}
        },
        accent: (() => { try { return app.ui.settings.getSettingValue("sfnodes.Accent") || "#f66744"; } catch { return "#f66744"; } })(),
        prefs: () => ({ civitai: true, thumbs: true }),
        refresh: () => { try { node.setDirtyCanvas?.(true, true); } catch {} },
        // dmodel 域四件套（见文件头注释）
        api: dmodelApi,
        hideTriggers: true,
        samplesKind: "diffusion_models",
        autoCivitai: true,
    };
}

function openDmodelPanel(node, modelName) {
    // 加载期门控：graphToPrompt/恢复期间 combo 值可能尚在还原途中，
    // 此时开面板会把旧行名钉进会话 Map（isGraphLoading 经验）。
    if (isGraphLoading()) return;
    openInfoPanelFor(dmodelPanelCtx(node, modelName), modelName);
}

app.registerExtension({
    name: "sfnodes.SFLoadDiffusionModel",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_TYPE) return;
        ensureEventHook();
        setupLoaderInfoWidget(node, "unet_name", {
            prefetch: null,                       // dmodel 无预取网关；info 在面板打开时取
            hasCustomOf: (name) => _hasData.has(name),
            onOpen: openDmodelPanel,
        });
    },
});
