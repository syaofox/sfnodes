// ==========================================================================
// sf_llm_settings.js - 共享 LLM API 设置（翻译 / 图片反推共用）
// ==========================================================================
//
// 注册 ComfyUI Settings 项 sfnodes.LLM.{Provider,BaseUrl,Model,ApiKey}，供
// SFPauseText 翻译按钮与 SFImageInterrogatorAPI 共用同一套凭据。后端经
// sf_utils/llm_client.py 读服务器上的 comfy.settings.json，无需前端回传。
//
// 旧 id sfnodes.Translate.*（翻译功能首版）首次注册时把值迁移到新 id，用户
// 已填的 key 不丢。
//
// 纯工具模块（无扩展行为），由使用者 import（同 sf_common.js 惯例）。
// ==========================================================================

import { app } from "/scripts/app.js";

export const SETTING_PROVIDER = "sfnodes.LLM.Provider";
export const SETTING_BASE_URL = "sfnodes.LLM.BaseUrl";
export const SETTING_MODEL = "sfnodes.LLM.Model";
export const SETTING_API_KEY = "sfnodes.LLM.ApiKey";
export const SETTING_CACHE = "sfnodes.LLM.CacheEnabled";

export const DEFAULT_BASE_URL = "https://api.deepseek.com";
export const DEFAULT_MODEL = "deepseek-flash";

const LEGACY_IDS = {
    [SETTING_PROVIDER]: "sfnodes.Translate.Provider",
    [SETTING_BASE_URL]: "sfnodes.Translate.BaseUrl",
    [SETTING_MODEL]: "sfnodes.Translate.Model",
    [SETTING_API_KEY]: "sfnodes.Translate.ApiKey",
};

function getSetting(id) {
    try { return app.ui?.settings?.getSettingValue?.(id); } catch { return undefined; }
}

function setSetting(id, val) {
    try {
        const s = app.ui?.settings;
        if (typeof s?.setSettingValueAsync === "function") s.setSettingValueAsync(id, val);
        else s?.setSettingValue?.(id, val);
    } catch { /* 仅会话内生效 */ }
}

let _registered = false;

// 幂等注册。设置系统不可用时静默降级（调用方在缺失 key 时提示用户）。
export function registerLLMSettings() {
    if (_registered) return;
    _registered = true;
    try {
        const s = app.ui.settings;
        s.addSetting({
            id: SETTING_PROVIDER,
            name: "SF LLM: provider (DeepSeek / OpenAI-compatible)",
            defaultValue: "deepseek",
            type: "combo",
            options: () => [
                { value: "deepseek", text: "DeepSeek", selected: getSetting(SETTING_PROVIDER) !== "custom" },
                { value: "custom", text: "OpenAI-compatible (custom)", selected: getSetting(SETTING_PROVIDER) === "custom" },
            ],
            onChange: (value) => {
                // 切回 DeepSeek 时把 base/model 复位为官方默认（onChange 参数即新值）
                if (value === "deepseek") {
                    setSetting(SETTING_BASE_URL, DEFAULT_BASE_URL);
                    setSetting(SETTING_MODEL, DEFAULT_MODEL);
                }
            },
        });
        s.addSetting({
            id: SETTING_BASE_URL,
            name: "SF LLM: API base URL",
            defaultValue: DEFAULT_BASE_URL,
            type: "text",
        });
        s.addSetting({
            id: SETTING_MODEL,
            name: "SF LLM: model name",
            defaultValue: DEFAULT_MODEL,
            type: "text",
        });
        s.addSetting({
            id: SETTING_API_KEY,
            name: "SF LLM: API key (DeepSeek or OpenAI-compatible)",
            defaultValue: "",
            type: "text",
        });
        s.addSetting({
            id: SETTING_CACHE,
            name: "SF LLM: cache identical requests (LRU, avoids repeat API calls)",
            defaultValue: true,
            type: "boolean",
        });
        migrateLegacy();
    } catch (e) {
        console.warn("[sfnodes] LLM settings unavailable", e);
    }
}

// 旧 sfnodes.Translate.* → 新 sfnodes.LLM.*：仅当新值缺失时搬运一次。
function migrateLegacy() {
    for (const [newId, oldId] of Object.entries(LEGACY_IDS)) {
        const cur = getSetting(newId);
        if (cur !== undefined && cur !== null && cur !== "") continue;
        const old = getSetting(oldId);
        if (old !== undefined && old !== null && old !== "") setSetting(newId, old);
    }
}

// 当前配置（节点/路由调用）。返回 {provider, baseUrl, model, apiKey}，均带默认值。
export function getLLMConfig() {
    return {
        provider: getSetting(SETTING_PROVIDER) || "deepseek",
        baseUrl: String(getSetting(SETTING_BASE_URL) || DEFAULT_BASE_URL).trim() || DEFAULT_BASE_URL,
        model: String(getSetting(SETTING_MODEL) || DEFAULT_MODEL).trim() || DEFAULT_MODEL,
        apiKey: String(getSetting(SETTING_API_KEY) || "").trim(),
    };
}
