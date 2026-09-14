// 缓存节点前端公共库（纯工具，无 app 依赖，可拷 .mjs 单测）：
// 把节点的 `name` 下拉动态重建为磁盘上已有缓存名 + 「＋ 新建…」入口，
// 并提供刷新按钮。SFMaskCache / SFTrackDataCache 共用。
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

// 安装 name 下拉：patch callback（新建入口）+ 刷新按钮 + 首次拉取。
// 幂等：同一 widget 只 patch 一次，按钮按标题去重（configure 重建后重装）。
export function installCacheNameList(node, { api, newLabel = NEW_LABEL, refreshLabel = "↻ 刷新缓存列表" }) {
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
    refresh(node, api, newLabel);
}
