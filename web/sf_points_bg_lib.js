// PointsEditor 底图上游解析纯逻辑（无 app 依赖，可拷 .mjs 单测）。
//
// KJNodes 自己的 resolveSourcePreview 只认"直接上游有 image/video widget 或
// videopreview.videoEl 或执行后 imgs"，对透传/变换链（ImageScale、SFImageBatch、
// Any Switch、SFImageCropExpand…）解析不到 → PointsEditor 需跑一次才有底图。
// 本模块沿 bg_image 链逐跳，优先返回执行后预览（尺寸最准），否则识别直接源
// （LoadImage/VHS 视频帧/裁剪源文件）。
//
// 返回 descriptor（不构造 URL，保持无依赖；URL 由调用方用 sf_common 构造）：
//   { kind: "preview", url }                 已执行预览（node.imgs）
//   { kind: "videoEl", videoEl }             VHS videopreview.videoEl（可直接抓帧）
//   { kind: "widget", widget, value }        名为 image/video 的 widget 值
//   { kind: "file", path, node }             SFImageCropExpand 的 src_path 源文件
//   null                                     解析不到

export const MAX_DEPTH = 12;
export const RANGE_TYPE = "SFImageBatchRange";
export const SEC_TYPE = "SeCVideoSegmentation";

function typeOf(node) {
    return node?.type || node?.comfyClass;
}

function widgetValue(node, name) {
    const w = (node?.widgets || []).find((x) => x?.name === name);
    return w ? w.value : undefined;
}

export function isRangeNode(node) {
    return typeOf(node) === RANGE_TYPE;
}

function linkById(graph, id) {
    if (!graph) return null;
    if (graph.links?.get) return graph.links.get(id) ?? null;
    return graph.links?.[id] ?? null;
}

// 从节点某个输入槽解析上游节点（Vue 版 links 可能是 Map 或对象表）。
export function linkedSource(graph, node, inp) {
    if (!node || !inp || inp.link == null) return null;
    const link = linkById(graph, inp.link);
    const srcId = link?.origin_id;
    if (srcId == null) return null;
    if (graph.getNodeById) return graph.getNodeById(srcId) ?? null;
    return (graph._nodes || []).find((n) => String(n.id) === String(srcId)) ?? null;
}

// 解析 SFImageCropExpand 的持久化状态（properties.sfCropExpandState，JSON 串或对象）。
export function parseCropExpandState(src) {
    const raw = src?.properties?.sfCropExpandState;
    if (!raw) return null;
    if (typeof raw === "object") return raw;
    try {
        const d = JSON.parse(raw);
        return d && typeof d === "object" ? d : null;
    } catch (e) {
        return null;
    }
}

// 单个节点能提供什么底图源（不向上递归）。
export function describeSource(src) {
    if (!src) return null;
    // 1. 已执行预览（尺寸最准，优先）
    const imgs = src.imgs;
    if (Array.isArray(imgs) && imgs.length) {
        const first = imgs[0];
        const url = typeof first === "string" ? first : first?.src;
        if (url) return { kind: "preview", url };
    }
    // 2. VHS 视频预览元素
    const vp = (src.widgets || []).find((w) => w?.name === "videopreview");
    if (vp?.videoEl?.src) return { kind: "videoEl", videoEl: vp.videoEl };
    // 3. 名为 image / video 的 widget（LoadImage / VHS 等）
    const w = (src.widgets || []).find((x) => x?.name === "image" || x?.name === "video");
    if (w?.value) return { kind: "widget", widget: w.name, value: String(w.value) };
    // 4. SFImageCropExpand 裁剪源
    const st = parseCropExpandState(src);
    if (st?.src_path) return { kind: "file", path: String(st.src_path), node: src };
    return null;
}

// graph API 适配（保持纯逻辑可测：注入 graph 即可）。
export function makeGraphApi(graph) {
    return {
        getInputSource(node, name) {
            const inp = (node?.inputs || []).find((i) => i.name === name);
            return linkedSource(graph, node, inp);
        },
        firstImageSource(node) {
            const inputs = node?.inputs || [];
            // 先显式 IMAGE 槽，再回退 any/`*` 槽（Any Switch / Convert 等）
            for (const want of ["IMAGE", "*"]) {
                for (const inp of inputs) {
                    if (inp?.link == null) continue;
                    if (String(inp.type || "") !== want) continue;
                    const s = linkedSource(graph, node, inp);
                    if (s) return s;
                }
            }
            return null;
        },
        linkTarget(linkId) {
            const link = linkById(graph, linkId);
            const tid = link?.target_id;
            if (tid == null) return null;
            if (graph.getNodeById) return graph.getNodeById(tid) ?? null;
            return (graph._nodes || []).find((n) => String(n.id) === String(tid)) ?? null;
        },
    };
}

// 从 PointsEditor 出发沿 bg_image 链找底图源；途经 SFImageBatchRange 时
// 累加 start_index 偏移（用于把"段内帧号"换算回源视频帧号）。
export function resolveBgSource(node, api, opts = {}) {
    const maxDepth = opts.maxDepth ?? MAX_DEPTH;
    const visited = new Set();
    let src = api.getInputSource(node, "bg_image");
    let depth = 0;
    let offset = 0;
    while (src && !visited.has(src) && depth < maxDepth) {
        visited.add(src);
        depth += 1;
        if (isRangeNode(src)) {
            const s = Number(widgetValue(src, "start_index"));
            if (Number.isFinite(s) && s > 0) offset += s;
            src = api.getInputSource(src, "images") || api.firstImageSource(src);
            continue;
        }
        const desc = describeSource(src);
        if (desc) return { ...desc, offset, node: src };
        src = api.firstImageSource(src);
    }
    return null;
}

// 该 PointsEditor 下游 SeC 的 annotation_frame_idx（找不到→0）。
export function resolveAnnotationFrame(node, api) {
    for (const out of (node?.outputs || [])) {
        for (const lid of (out?.links || [])) {
            const tgt = api.linkTarget(lid);
            if (!tgt || typeOf(tgt) !== SEC_TYPE) continue;
            const v = Number(widgetValue(tgt, "annotation_frame_idx"));
            if (Number.isFinite(v) && v >= 0) return v;
        }
    }
    return 0;
}
