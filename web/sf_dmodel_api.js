// ==========================================================================
// SF Load Diffusion Model - dmodel 域路由薄封装（与 sf_lora_stack_api.js
// 同形函数束）。信息面板经 ctx.api 注入整束替换，函数签名与返回形状与
// LoRA 版逐一同构：{ok}/{ok,info}/{ok,v}/reason 约定一致。
//
// 与 LoRA 域的差异只有 URL 前缀与事件名：
//   路由   /api/sfnodes/lora_*      -> /api/sfnodes/dmodel_*（lora_routes
//          别名注册 + diffusion_routes 的 dmodel_info）
//   事件   sfnodes.lora-data-changed -> sfnodes.model-data-changed（域隔离，
//          各自的监听者互不误伤）
// info 不做客户端缓存（每次打开面板都取服务端真源；Civitai 结果由后端侧车
// 缓存，第二次即时且离线），invalidate 为占位以满足面板调用契约。
// ==========================================================================
import { sfApiUrl } from "./sf_common.js";

function broadcastDataChanged(name) {
    try {
        document.dispatchEvent(new CustomEvent("sfnodes.model-data-changed", { detail: { name } }));
    } catch {}
}

export async function dmodelInfo(name) {
    if (!name) return { ok: false, message: "No model selected." };
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/dmodel_info?name=" + encodeURIComponent(name)),
            { cache: "no-store" });
        return await r.json();
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

// 占位：info 无客户端缓存，无需失效。保留导出让宿主代码与 LoRA 束同形。
export function invalidateInfo() {}

// `bust`（时间戳或计数器）越过浏览器一小时图片缓存（缩略图路由发 max-age=3600）。
export function thumbUrl(name, bust) {
    return sfApiUrl("/api/sfnodes/dmodel_thumb?name=" + encodeURIComponent(name) +
        (bust ? "&t=" + bust : ""));
}

// 与 lora 版同构：支持 {overwrite, modelId, versionId, civitaiUrl} 对象形式
// （Archive 回退），尊重 sfnodes.Civitai.DownloadSamples 开关。
export async function civitaiLookup(name, overwriteOrOpts) {
    try {
        let overwrite = false, modelId = null, versionId = null, civitaiUrl = null;
        if (overwriteOrOpts && typeof overwriteOrOpts === "object") {
            overwrite = !!overwriteOrOpts.overwrite;
            modelId = overwriteOrOpts.modelId ?? overwriteOrOpts.model_id ?? null;
            versionId = overwriteOrOpts.versionId ?? overwriteOrOpts.version_id ?? null;
            civitaiUrl = overwriteOrOpts.civitaiUrl ?? overwriteOrOpts.url ?? overwriteOrOpts.link ?? null;
        } else {
            overwrite = !!overwriteOrOpts;
        }
        const q = overwrite ? "&overwrite=1" : "";
        let dl = "";
        try {
            const v = globalThis.app?.ui?.settings?.getSettingValue?.("sfnodes.Civitai.DownloadSamples");
            if (v) dl = "&downloadSamples=1";
        } catch {}
        let extra = "";
        if (modelId) extra += "&modelId=" + encodeURIComponent(String(modelId));
        if (versionId) extra += "&versionId=" + encodeURIComponent(String(versionId));
        if (civitaiUrl) extra += "&civitaiUrl=" + encodeURIComponent(String(civitaiUrl));
        const r = await fetch(sfApiUrl("/api/sfnodes/dmodel/civitai?name=" + encodeURIComponent(name) + q + dl + extra));
        const j = await r.json();
        if (j?.ok) broadcastDataChanged(name);
        return j;
    } catch {
        return { ok: false, reason: "offline", message: "Could not reach Civitai." };
    }
}

async function _post(route, body, { invalidate = true, name = "" } = {}) {
    try {
        const r = await fetch(sfApiUrl(route), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const j = await r.json();
        if (j?.ok && invalidate) broadcastDataChanged(name);
        return j;
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

export function saveCustomDescription(name, description) {
    return _post("/api/sfnodes/dmodel/custom_description", { name, description }, { name });
}
export function saveLoraPreview(name, dataUrl) {
    return _post("/api/sfnodes/dmodel/preview", { name, dataUrl }, { name });
}
export function deleteLoraPreview(name) {
    return _post("/api/sfnodes/dmodel/preview_delete", { name }, { name });
}
export function saveCivitaiThumb(name) {
    return _post("/api/sfnodes/dmodel/civitai_thumb_save", { name }, { name });
}
export function deleteCivitai(name) {
    // 删侧车不产生用户数据变化，但缩略图/描述会回落——照常广播。
    return _post("/api/sfnodes/dmodel/civitai_delete", { name }, { name });
}
export function migrateLoraData(name, oldKey) {
    return _post("/api/sfnodes/dmodel/migrate", { name, old_key: oldKey || "" }, { name });
}
export function mergeLoraData(name, oldKey) {
    return _post("/api/sfnodes/dmodel/merge", { name, old_key: oldKey || "" }, { name });
}

// 面板注入用整束（键名与 sf_lora_stack_info.js 内 A.* 一致）。
export const dmodelApi = {
    info: dmodelInfo,
    thumbUrl,
    civitai: civitaiLookup,
    invalidate: invalidateInfo,
    delCivitai: deleteCivitai,
    saveDescription: saveCustomDescription,
    savePreview: saveLoraPreview,
    deletePreview: deleteLoraPreview,
    saveCivitaiThumb,
    migrate: migrateLoraData,
    merge: mergeLoraData,
};
