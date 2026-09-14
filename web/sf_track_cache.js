// SF Track Data Cache 前端扩展：name 下拉动态重建为 user/sfnodes/track_cache/
// 下已有缓存名 + 「＋ 新建缓存…」；通用逻辑复用 sf_cache_name_lib。
import { app } from "/scripts/app.js";
import { installCacheNameList } from "./sf_cache_name_lib.js";

const NODE = "SFTrackDataCache";
const API = "/api/sfnodes/track_cache/list";

app.registerExtension({
    name: "sfnodes.TrackCache",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            installCacheNameList(this, { api: API });
            return r;
        };

        const onConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = onConfigured?.apply(this, arguments);
            installCacheNameList(this, { api: API });
            return r;
        };
    },
});
