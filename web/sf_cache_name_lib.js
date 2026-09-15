// 缓存节点前端公共库（纯工具，无 app 依赖，可拷 .mjs 单测）：
// 把节点的 `name` 下拉动态重建为磁盘上已有缓存名 + 「＋ 新建…」入口，
// 并提供刷新按钮；name_text 文本输入连接后把下拉置灰（后端 text 非空优先）。
// SFMaskCache / SFTrackDataCache 共用。
import { isSlotConnected } from "./sf_dynamic_slots.js";

export const NEW_LABEL = "＋ 新建缓存…";

// 纯逻辑：合并已有缓存名 + 当前值（保序去重）+ 新建入口，便于单测。
export function buildNameOptions(names, current, newLabel = NEW_LABEL) {
    const out = (names || []).map(String).filter((n) => n && n !== newLabel);
    if (current && current !== newLabel && !out.includes(current)) out.unshift(current);
    out.push(newLabel);
    return out;
}

export function findWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

// name_text 输入是否已连接（link/links 两形态兼容，复用 sf_dynamic_slots.isSlotConnected）
export function isNameTextLinked(node, inputName = "name_text") {
    const slot = node?.inputs?.find((i) => i?.name === inputName);
    return isSlotConnected(slot);
}

// 文本覆盖态应用：连接 name_text 时下拉 disabled=true（Vue 前端渲染半透明不可交互，
// 见 IBaseWidget.disabled/computedDisabled），断开恢复原值；状态未变时提前返回。
export function applyNameOverrideState(node, { inputName = "name_text", name = "name" } = {}) {
    const w = findWidget(node, name);
    if (!w) return false;
    const linked = isNameTextLinked(node, inputName);
    if (w._sfCacheNameTextOverridden === linked) return linked;
    const first = w._sfCacheNameTextOverridden === undefined;
    w._sfCacheNameTextOverridden = linked;
    if (first && !linked) return false;  // 初始未连接：不改动 widget
    if (w._sfCacheNameTextPrevDisabled === undefined) {
        w._sfCacheNameTextPrevDisabled = !!w.disabled;
    }
    w.disabled = linked ? true : !!w._sfCacheNameTextPrevDisabled;
    node.updateComputedDisabled?.();
    node.setDirtyCanvas?.(true, true);
    return linked;
}

export async function fetchNames(api) {
    try {
        const resp = await fetch(api, { cache: "no-store" });
        if (!resp.ok) return null;
        const data = await resp.json();
        return Array.isArray(data?.names) ? data.names.map(String) : null;
    } catch (e) {
        console.warn("[sf_cache_name_lib] 缓存列表加载失败:", e);
        return null;
    }
}

export function applyNames(node, names, newLabel = NEW_LABEL) {
    const w = findWidget(node, "name");
    if (!w) return;
    w.options = w.options || {};
    w.options.values = buildNameOptions(names, w.value, newLabel);
    node.setDirtyCanvas?.(true, true);
}

async function refresh(node, api, newLabel) {
    const names = await fetchNames(api);
    if (names) applyNames(node, names, newLabel);
}

// 安装 name 下拉：patch callback（新建入口）+ 刷新按钮 + 首次拉取 + 文本覆盖置灰。
// 幂等：同一 widget 只 patch 一次，按钮按标题去重（configure 重建后重装）。
export function installCacheNameList(node, { api, newLabel = NEW_LABEL, refreshLabel = "↻ 刷新缓存列表", textInput = "name_text" }) {
    if (!node || !api) return;
    const w = findWidget(node, "name");
    if (w && !w._sfCacheNamePatched) {
        w._sfCacheNamePatched = true;
        const orig = w.callback;
        w.callback = function (value, ...rest) {
            if (value === newLabel) {
                const entered = window.prompt("输入新的缓存名：", "");
                const nm = (entered || "").trim();
                const base = (w.options?.values || []).filter((v) => v !== newLabel);
                if (nm && !base.includes(nm)) base.push(nm);
                w.value = nm;
                w.options = w.options || {};
                w.options.values = base.concat([newLabel]);
                if (typeof orig === "function") orig(w.value, ...rest);
                return;
            }
            if (typeof orig === "function") orig(value, ...rest);
        };
    }
    if (!node.widgets?.some((x) => x?.name === refreshLabel)) {
        node.addWidget("button", refreshLabel, null, () => refresh(node, api, newLabel));
    }
    if (textInput) {
        if (!node._sfCacheNameTextHooked) {
            node._sfCacheNameTextHooked = true;
            const origConn = node.onConnectionsChange;
            node.onConnectionsChange = function (...args) {
                const r = origConn?.apply(this, args);
                applyNameOverrideState(this, { inputName: textInput });
                return r;
            };
        }
        applyNameOverrideState(node, { inputName: textInput });
    }
    refresh(node, api, newLabel);
}
