// ==========================================================================
// sf_prompt_tags_store.js - SFPromptTags 标签库存储层
// ==========================================================================
//
// 标签库是一个跨节点、跨工作流共享的数据：{ version, categories, listCats,
// tags, catModes }，作为一个 JSON 字符串存于 ComfyUI 的未注册设置
// "sfnodes.PromptTags.Library"（app.ui.settings，机器私有、随插件更新存活、
// 永不写入工作流）。
//
// 与 Pixaroma 语义一致：
//   - getLibrary / reloadLibrary：读（reload 强制重读，供跨标签页同步）
//   - setLibrary：整体替换并立即持久化 + 通知订阅者
//   - commitLibrary：工作副本变更，先更新缓存 + 通知，防抖持久化
//   - flushLibrary：仅当有未落盘的防抖写入时才写（不能覆盖他标签页的编辑）
//   - isSameAsStored：工作副本与存储是否一致（无变化不写回）
//   - subscribe：库变更时通知（节点重高亮 / 预览）
//
// 规范化、导入导出等纯函数见 sf_prompt_tags_lib.js。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { normalizeLibrary, applyImportData, sideOfCat, isListTag } from "./sf_prompt_tags_lib.js";
import { resetCursor, listKey } from "./sf_prompt_tags_cursors.js";

const LIBRARY_SETTING = "sfnodes.PromptTags.Library";
// 插件内置默认库（data/prompt_presets.json 转换产物，随插件分发）。web/
// 目录由 ComfyUI 以 /extensions/sfnodes/ 静态服务，可直接 fetch。
const DEFAULT_LIBRARY_URL = "/extensions/sfnodes/prompt_tags_default.json";

let _data = null;
let _persistTimer = null;
let _defaultPromise = null;
const _subs = new Set();

function settingsApi() {
    const s = app.ui?.settings;
    return s && typeof s.getSettingValue === "function" ? s : null;
}

// 内置默认库：fetch + 规范化。结果缓存（成功后编辑器"恢复默认库"复用同一
// 份，不重复请求）；失败 resolve null（调用方保持空库，本次会话不重试）。
export function fetchDefaultLibrary() {
    if (!_defaultPromise) {
        _defaultPromise = (async () => {
            try {
                const r = await fetch(DEFAULT_LIBRARY_URL);
                if (!r.ok) throw new Error("HTTP " + r.status);
                return normalizeLibrary(await r.json());
            } catch (err) {
                console.warn("[sfnodes.PromptTags] 内置默认库加载失败，使用空库:", err);
                return null;
            }
        })();
    }
    return _defaultPromise;
}

function persist(data) {
    const s = settingsApi();
    if (!s) return;
    const json = JSON.stringify(data);
    try {
        if (typeof s.setSettingValueAsync === "function") s.setSettingValueAsync(LIBRARY_SETTING, json);
        else if (typeof s.setSettingValue === "function") s.setSettingValue(LIBRARY_SETTING, json);
    } catch { /* 非致命：本次会话内存态仍然正确 */ }
}

function fanout() {
    for (const fn of _subs) { try { fn(_data); } catch { /* 单个监听出错不影响其余 */ } }
}

export function getLibrary() {
    if (_data) return _data;
    const s = settingsApi();
    if (!s) return normalizeLibrary({});
    const raw = s.getSettingValue(LIBRARY_SETTING);
    try {
        _data = normalizeLibrary(typeof raw === "string" ? JSON.parse(raw) : raw);
    } catch {
        _data = normalizeLibrary({});
    }
    // 新环境/设置被清：库里没有内容时异步载入插件内置默认库并落盘
    // （setLibrary 自动 persist + fanout，编辑器/节点打开时自动刷新）。
    // 仅在仍为空库时应用——fetch 完成前用户已动手建标签则不覆盖。
    if (!raw) {
        fetchDefaultLibrary().then((def) => {
            if (def && _data && !_data.tags.length && !_data.categories.length) setLibrary(def);
        });
    }
    return _data;
}

// 丢弃缓存，下次读取时从设置重读（跨标签页同步：另一个窗口可能已编辑）
export function reloadLibrary() {
    _data = null;
    return getLibrary();
}

// 工作副本与存储是否一致（两侧都规范化后比较）
export function isSameAsStored(data) {
    try { return JSON.stringify(normalizeLibrary(data)) === JSON.stringify(getLibrary()); }
    catch { return false; }
}

export function getTags() { return getLibrary().tags; }

// 有序分类名。`side`（"text" | "list"）过滤一侧；省略则双侧。该侧确有标签
// 坐在桶里时，把桶名追加在末尾。
export function getCategories(side) {
    const data = getLibrary();
    const out = data.categories.filter((c) => !side || sideOfCat(c, data) === side);
    if (side !== "list" && data.tags.some((t) => !t.cat && !isListTag(t))) out.push("Text");
    if (side !== "text" && data.tags.some((t) => !t.cat && isListTag(t))) out.push("List");
    return out;
}

export function findTag(name) {
    const k = String(name).toLowerCase();
    for (const t of getTags()) if (t.name.toLowerCase() === k) return t;
    return null;
}

export function tagsInCat(name) { return getTags().filter((t) => t.cat === name); }

// 整体替换（增删/导入/改名等结构性变更）：立即持久化 + 通知
export function setLibrary(data) {
    _data = normalizeLibrary(data);
    if (_persistTimer) { clearTimeout(_persistTimer); _persistTimer = null; }
    persist(_data);
    fanout();
    return _data;
}

// 工作副本实时编辑：先更新缓存 + 通知（节点随输入即时重高亮），防抖写设置
export function commitLibrary(data) {
    _data = normalizeLibrary(data);
    fanout();
    if (_persistTimer) clearTimeout(_persistTimer);
    _persistTimer = setTimeout(() => { persist(_data); _persistTimer = null; }, 350);
    return _data;
}

// 落盘任何未写入的防抖编辑；无待写则不动（不覆盖他标签页的编辑）
export function flushLibrary() {
    if (!_persistTimer) return;
    clearTimeout(_persistTimer);
    _persistTimer = null;
    if (_data) persist(_data);
}

export function subscribe(fn) { _subs.add(fn); return () => _subs.delete(fn); }

// 应用一次导入（parseImport 的结果）。mode: "both"/"replace"/"skip"。
// 被替换的标签其游标位置作废（内容全变了，续旧牌局没有意义）。
export function applyImport(parsed, mode) {
    const cur = getLibrary();
    const r = applyImportData(cur, parsed, mode);
    setLibrary(r.data);
    for (const name of r.replacedNames) {
        try { resetCursor(listKey(name)); } catch { /* ignore */ }
    }
    return r;
}
