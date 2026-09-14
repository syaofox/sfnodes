// SF Points Background —— 给 KJNodes PointsEditor 加「↻ 刷新底图」按钮。
//
// 背景：PointsEditor 的 bg_image 由 KJNodes 的 resolveSourcePreview 解析，
// 只认直接上游的 image/video widget、videopreview.videoEl 或执行后 imgs，
// 且工作流重载不走 onConnectionsChange → 不跑一次就没有底图；透传/变换链
// （ImageScale/SFImageBatch/Any Switch/SFImageCropExpand）更是完全解析不到。
//
// 本扩展不执行工作流：沿 bg_image 链解析（纯逻辑 sf_points_bg_lib.js），
// 抓到图/视频首帧后交给 KJNodes 的 editor.processImage()（其自身会持久化
// properties.imgData，点一次后随工作流保存、重载自动恢复）。
import { app } from "/scripts/app.js";
import { makeGraphApi, resolveBgSource, resolveAnnotationFrame } from "./sf_points_bg_lib.js";
import { buildSourceURL, parseAnnotatedImageValue, sfToast } from "./sf_common.js";

const NODE = "PointsEditor";
const BTN = "↻ 刷新底图";

function toast(summary, detail, severity = "info") {
    sfToast({ summary, detail, severity, fallbackTag: "SFPointsBg" });
}

function graphOf(node) {
    return node?.graph || app.graph;
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("图片加载失败"));
        img.src = url;
    });
}

function captureVideoFrame(videoEl) {
    if (!videoEl?.videoWidth || !videoEl?.videoHeight) throw new Error("视频尚未就绪");
    const c = document.createElement("canvas");
    c.width = videoEl.videoWidth;
    c.height = videoEl.videoHeight;
    c.getContext("2d").drawImage(videoEl, 0, 0);
    return c;
}

function captureVideoURL(url) {
    return new Promise((resolve, reject) => {
        const v = document.createElement("video");
        v.muted = true;
        v.preload = "auto";
        v.src = url;
        v.addEventListener("loadeddata", () => {
            try { resolve(captureVideoFrame(v)); } catch (e) { reject(e); }
        }, { once: true });
        v.addEventListener("error", () => reject(new Error("视频加载失败")), { once: true });
    });
}

// 把可 seek 的 videoEl 定位到指定秒并抓帧（VHS 高级预览已应用 skip/force_rate/cap，
// 故「输出帧 N」= currentTime N / (force_rate/select_every_nth)）。
function captureVideoElementAt(videoEl, seconds) {
    return new Promise((resolve) => {
        const grab = () => {
            clearTimeout(timer);
            videoEl.removeEventListener("seeked", grab);
            try { resolve(captureVideoFrame(videoEl)); } catch (e) { resolve(null); }
        };
        const timer = setTimeout(grab, 600);
        videoEl.addEventListener("seeked", grab, { once: true });
        try { videoEl.currentTime = Math.max(0, seconds); } catch (e) { grab(); }
    });
}

function videoElementFps(videoEl, params) {
    const fr = Number(params?.force_rate) || 0;
    if (fr <= 0) return 0;
    const nth = Number(params?.select_every_nth) || 1;
    return fr / nth;
}

async function applyBg(node, { silent }) {
    const editor = node.editor;
    if (!editor?.processImage) {
        if (!silent) toast("编辑器未就绪", "KJNodes PointsEditor 编辑器尚未创建", "warn");
        return false;
    }
    const api = makeGraphApi(graphOf(node));
    const source = resolveBgSource(node, api);
    if (!source) {
        if (!silent) toast("未找到底图源",
            "上游接 LoadImage / VHS / SFImageCropExpand，或先运行一次后刷新", "warn");
        return false;
    }
    // 目标帧 = 分段偏移（途经 SFImageBatchRange.start_index）+ 下游 SeC 的 annotation_frame_idx
    const frame = (source.offset || 0) + resolveAnnotationFrame(node, api);
    try {
        if (source.kind === "videoEl") {
            const vp = (source.node?.widgets || []).find((w) => w?.name === "videopreview");
            const fps = videoElementFps(source.videoEl, vp?.value?.params);
            const seekable = source.videoEl.seekable && source.videoEl.seekable.length > 0;
            let canvas = null;
            if (frame > 0 && fps > 0 && seekable) {
                canvas = await captureVideoElementAt(source.videoEl, frame / fps);
            }
            if (!canvas) canvas = captureVideoFrame(source.videoEl);
            editor.processImage(canvas, { resize: true });
            if (!silent) toast("底图已刷新", canvas && frame > 0 ? `取自上游视频第 ${frame} 帧` : "取自上游视频首帧");
            return true;
        }
        if (source.kind === "widget" && source.widget === "video") {
            const url = buildSourceURL(parseAnnotatedImageValue(source.value), true);
            editor.processImage(await captureVideoURL(url), { resize: true });
            if (!silent) toast("底图已刷新", "取自上游视频首帧");
            return true;
        }
        if (source.kind === "preview") {
            const imgs = source.node?.imgs;
            const pick = (Array.isArray(imgs) && imgs[frame]) ? imgs[frame] : null;
            const url = typeof pick === "string" ? pick : pick?.src;
            editor.processImage(await loadImage(url || source.url), { resize: true });
            if (!silent) toast("底图已刷新", "取自上游已执行预览");
            return true;
        }
        // widget=image 或 file（SFImageCropExpand 的 src_path）
        const isFile = source.kind === "file";
        const part = parseAnnotatedImageValue(isFile ? source.path : source.value);
        if (isFile) part.type = "input";
        const img = await loadImage(buildSourceURL(part, true));
        // 裁剪源尺寸可能不同于 SeC 帧尺寸：保留当前坐标空间，避免坐标错位
        editor.processImage(img, { resize: !isFile });
        if (!silent) {
            toast("底图已刷新", isFile
                ? "取自裁剪源图；若与 SeC 帧尺寸不符，请把 width/height 设为帧尺寸或跑一次后刷新"
                : null);
        }
        return true;
    } catch (e) {
        if (!silent) toast("刷新底图失败", String(e?.message || e), "warn");
        return false;
    }
}

// KJNodes 把 "Reset canvas"/"Align to image" 放在名为 editor_buttons 的 DOM
// widget 行里；用 addWidget 会叠到那行上。这里把刷新按钮追加进同一行 DOM
// （样式对齐），行尚未创建时短暂重试（扩展注册顺序不定）。
function findButtonRow(node) {
    const w = node.widgets?.find((x) => x?.name === "editor_buttons");
    const el = w?.element || w?.inputEl;
    return el && typeof el.appendChild === "function" ? el : null;
}

function ensureButton(node, tries = 0) {
    const row = findButtonRow(node);
    if (!row) {
        if (tries < 12) setTimeout(() => ensureButton(node, tries + 1), 60);
        return;
    }
    // 用行内标记去重（不能只看 isConnected：configure 期间 DOM 行会瞬时脱离
    // 又被复用，导致旧引用误判为"已移除"而重复追加）。多出来的旧残留一并清理。
    const existing = row.querySelectorAll("[data-sf-points-bg]");
    if (existing.length > 0) {
        for (let i = 1; i < existing.length; i++) existing[i].remove();
        node._sfPointsBgBtn = existing[0];
        return;
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = BTN;
    btn.title = "从上游解析底图（无需运行工作流）";
    btn.setAttribute("data-sf-points-bg", "1");
    btn.style.cssText = "flex:1;height:24px;padding:0 6px;font:12px sans-serif;cursor:pointer;";
    btn.addEventListener("click", () => applyBg(node, { silent: false }));
    row.appendChild(btn);
    node._sfPointsBgBtn = btn;
}

function install(node, { auto }) {
    if (node?.comfyClass !== NODE && node?.type !== NODE) return;
    ensureButton(node);
    if (!auto) return;
    if (node.properties?.imgData) return;  // 已有持久化底图，交给 KJNodes 恢复
    setTimeout(() => {
        if (!node.editor?.processImage || node.properties?.imgData) return;
        if (resolveBgSource(node, makeGraphApi(graphOf(node)))) applyBg(node, { silent: true });
    }, 700);
}

app.registerExtension({
    name: "sfnodes.PointsBackground",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            install(this, { auto: false });
            return r;
        };

        const onConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = onConfigured?.apply(this, arguments);
            install(this, { auto: true });
            return r;
        };

        // 新接线时静默尝试一次（重载不走本钩子，故 configure 分支另行兜底）
        const onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            const r = onConn?.apply(this, arguments);
            if (type === 1 && !this.properties?.imgData) {
                setTimeout(() => install(this, { auto: true }), 300);
            }
            return r;
        };
    },
});
