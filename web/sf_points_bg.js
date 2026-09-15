// SF Points Background —— 给 KJNodes PointsEditor 加「↻ 刷新底图」按钮。
//
// 背景：PointsEditor 的 bg_image 由 KJNodes 的 resolveSourcePreview 解析，
// 只认直接上游的 image/video widget、videopreview.videoEl 或执行后 imgs，
// 且工作流重载不走 onConnectionsChange → 不跑一次就没有底图；透传/变换链
// （ImageScale/SFImageBatch/Any Switch/SFImageCropExpand）更是完全解析不到。
//
// 本扩展不执行工作流：沿 bg_image 链解析（纯逻辑 sf_points_bg_lib.js），
// 抓到图/视频帧后交给 KJNodes 的 editor.processImage()（其自身会持久化
// properties.imgData，点一次后随工作流保存、重载自动恢复）。
//
// ⚠ 多个 PointsEditor 可能共用同一个 VHS videoEl：每帧都必须显式 seek 到目标
// 位置（含 frame=0），否则会互相"继承"当前帧；seek 失败时不得抓旧帧冒充成功
// （toast 必须如实报告）。seek 完成后等一帧（rVFC/120ms）再 drawImage，避免
// 画出 seek 前的旧帧。
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

// seek 完成后等一帧再抓：直接在 'seeked' 里 drawImage 可能仍画出旧帧。
function afterFrame(videoEl, done) {
    let settled = false;
    const finish = () => { if (settled) return; settled = true; done(); };
    if (typeof videoEl.requestVideoFrameCallback === "function") {
        try { videoEl.requestVideoFrameCallback(finish); } catch (e) { /* 回退定时器 */ }
    }
    setTimeout(finish, 120);
}

function seekVideoElement(videoEl, seconds, timeout) {
    return new Promise((resolve) => {
        const target = Math.max(0, seconds);
        // 已在目标位置：设置相同的 currentTime 不一定触发 seeked，直接取帧。
        if (videoEl.readyState >= 2 && Math.abs(videoEl.currentTime - target) < 0.04) {
            afterFrame(videoEl, () => resolve(true));
            return;
        }
        let settled = false;
        const finish = (ok) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            videoEl.removeEventListener("seeked", onSeeked);
            resolve(ok);
        };
        // ⚠ 非 seekable 流（VHS /vhs/viewvideo?deadline=realtime，seekable=[0,0]）会
        // 把 currentTime 钳回 0 却照样触发 seeked → 必须校验落点，否则误判成功。
        const onSeeked = () => afterFrame(videoEl,
            () => finish(Math.abs(videoEl.currentTime - target) < 0.05));
        const timer = setTimeout(() => finish(false), timeout);
        videoEl.addEventListener("seeked", onSeeked);
        try { videoEl.currentTime = target; } catch (e) { finish(false); }
    });
}

// 不可 seek 的流：从当前位置播放推进到目标秒再暂停抓帧（同一 videoEl 时间轴 =
// VHS 输出帧时间轴，故「帧 N」= 播放到 N/(force_rate/select_every_nth) 秒）。
function captureVideoByPlayback(videoEl, seconds, fps, timeout = 15000) {
    return new Promise((resolve) => {
        const target = Math.max(0, seconds);
        if (target <= 0.001) {
            try { videoEl.currentTime = 0; } catch (e) { /* 非 seekable 流忽略 */ }
            afterFrame(videoEl, () => {
                try { resolve(captureVideoFrame(videoEl)); } catch (e) { resolve(null); }
            });
            return;
        }
        let settled = false;
        let rafId = null;
        const cleanup = () => {
            if (rafId != null) cancelAnimationFrame(rafId);
            clearTimeout(timer);
        };
        const done = (canvas) => {
            if (settled) return;
            settled = true;
            cleanup();
            try { videoEl.pause(); } catch (e) { /* 忽略 */ }
            resolve(canvas);
        };
        const timer = setTimeout(() => done(null), timeout);
        const tick = () => {
            if (settled) return;
            if (videoEl.ended && videoEl.currentTime < target - 0.05) { done(null); return; }
            if (videoEl.currentTime >= target - 0.001 || videoEl.ended) {
                try { videoEl.pause(); } catch (e) { /* 忽略 */ }
                afterFrame(videoEl, () => {
                    try { done(captureVideoFrame(videoEl)); } catch (e) { done(null); }
                });
                return;
            }
            rafId = requestAnimationFrame(tick);
        };
        try { if (videoEl.currentTime > target) videoEl.currentTime = 0; } catch (e) { /* 忽略 */ }
        // 播放只能前进：回不到 target 之前（例如重置失败）就别抓一帧冒充。
        if (videoEl.currentTime > target + 0.05) { done(null); return; }
        const p = videoEl.play();
        if (p && typeof p.catch === "function") p.catch(() => done(null));
        rafId = requestAnimationFrame(tick);
    });
}

// 把 videoEl 定位到指定秒并抓帧；seek 未在超时内完成时返回 null（绝不抓旧帧冒充）。
// VHS 高级预览已应用 skip/force_rate/cap，故「输出帧 N」= currentTime N / (force_rate/select_every_nth)。
async function captureVideoElementAt(videoEl, seconds, timeout = 5000) {
    if (!(await seekVideoElement(videoEl, seconds, timeout))) return null;
    try { return captureVideoFrame(videoEl); } catch (e) { return null; }
}

// 用同一 URL 新建离屏 video 再 seek（绕开 VHS 预览元素未缓冲/不可 seek 的状态）。
function captureVideoURLAt(url, seconds, timeout = 8000) {
    return new Promise((resolve) => {
        const v = document.createElement("video");
        v.muted = true;
        v.preload = "auto";
        let settled = false;
        const finish = (ok) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            let out = null;
            if (ok) { try { out = captureVideoFrame(v); } catch (e) { out = null; } }
            try { v.removeAttribute("src"); v.load(); } catch (e) { /* 忽略 */ }
            resolve(out);
        };
        const timer = setTimeout(() => finish(false), timeout);
        v.addEventListener("error", () => finish(false), { once: true });
        v.addEventListener("loadeddata", () => {
            if (seconds <= 0) { afterFrame(v, () => finish(true)); return; }
            v.addEventListener("seeked", () => afterFrame(v, () => finish(true)), { once: true });
            try { v.currentTime = seconds; } catch (e) { finish(false); }
        }, { once: true });
        v.src = url;
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
            const seconds = fps > 0 ? frame / fps : 0;
            // 多个 PointsEditor 共用 VHS 的同一个 videoEl：每次都显式 seek 到目标帧，
            // 否则一个编辑器刷新会把另一个的当前帧当成自己的底图（frame=0 也是）。
            let canvas = null;
            let method = "current";
            if (fps > 0) {
                canvas = await captureVideoElementAt(source.videoEl, seconds);
                if (canvas) method = "seek";
            }
            if (!canvas && fps > 0) {
                // VHS 预览流不可 seek（seekable=[0,0]）：播放推进到目标帧再抓。
                canvas = await captureVideoByPlayback(source.videoEl, seconds, fps);
                if (canvas) method = "playback";
            }
            if (!canvas && fps > 0) {
                const url = source.videoEl.currentSrc || source.videoEl.src;
                if (url) {
                    canvas = await captureVideoURLAt(url, seconds);
                    if (canvas) method = "offscreen";
                }
            }
            if (!canvas) canvas = captureVideoFrame(source.videoEl);
            console.info(
                `[sf_points_bg] videoEl frame=${frame} fps=${fps} t=${seconds} method=${method}` +
                ` ready=${source.videoEl.readyState} dur=${source.videoEl.duration}` +
                ` cur=${source.videoEl.currentTime} seekable=${source.videoEl.seekable?.length ?? 0}` +
                ` seekEnd=${source.videoEl.seekable?.length ? source.videoEl.seekable.end(0) : "-"}` +
                ` src=${source.videoEl.currentSrc || source.videoEl.src}`);
            editor.processImage(canvas, { resize: true });
            if (!silent) {
                if (fps <= 0) {
                    toast("底图已刷新", "取自上游视频当前帧（无法从 force_rate 推算帧号）", "warn");
                } else if (method !== "current") {
                    toast("底图已刷新", `取自上游视频第 ${frame} 帧`);
                } else {
                    toast("底图已刷新（未定位到目标帧）",
                        `未能取到第 ${frame} 帧（视频未缓冲完），显示的是当前帧；稍后再点一次`, "warn");
                }
            }
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
