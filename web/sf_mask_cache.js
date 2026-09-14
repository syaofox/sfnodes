// SF Mask Cache 前端扩展：name 下拉动态重建为 user/sfnodes/mask_cache/ 下
// 已有缓存名 + 「＋ 新建缓存…」；通用逻辑在 sf_cache_name_lib。
import { app } from "/scripts/app.js";
import { installCacheNameList } from "./sf_cache_name_lib.js";

const NODE = "SFMaskCache";
const API = "/api/sfnodes/mask_cache/list";

app.registerExtension({
    name: "sfnodes.MaskCache",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            installCacheNameList(this, { api: API });
            return r;
        };

        // 值恢复/重载后重建下拉（configure 不触发 nodeCreated/onConnectionsChange）
        const onConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = onConfigured?.apply(this, arguments);
            installCacheNameList(this, { api: API });
            return r;
        };
    },
});
