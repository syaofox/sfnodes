// ==========================================================================
// sf_video_compare.js — SFVideoCompare 前端：节点内双视频叠加对比播放
// ==========================================================================
// 复刻 TE_MAN「TE MAN 视频对比」的可见能力（干净室：代码自写，不抄混淆实现）：
// 两个 <video> 叠加（B 以 clip-path 裁到分界线右侧），鼠标移动调分界线，点击
// 视频区暂停/继续，底部进度条拖动 seek，悬停菜单切 静音/音频 A/音频 B 与
// 0.25x-2x 速度，同步帧（两视频总帧数相同才可用），全屏对比，同步播放（从头
// 播放、任一结束即自动重播）。视频信息存 node.properties.sfVideoCompareVideos，
// 随工作流保存，刷新/重开工作流后自动恢复。
//
// 复用：sf_common（el / injectCSSOnce / buildSourceURL / installCanvasZoomPassthrough /
// applyAdaptiveCanvasOnly）+ sf_video_compare_lib 纯逻辑（时间/帧换算/几何）。
// ==========================================================================

import { app } from "/scripts/app.js";
import {
    applyAdaptiveCanvasOnly,
    buildSourceURL,
    el,
    injectCSSOnce,
    installCanvasZoomPassthrough,
} from "./sf_common.js";
import {
    AUDIO_MODES,
    CONTROL_HEIGHT,
    DEFAULT_POSITION,
    INITIAL_NODE_HEIGHT,
    INITIAL_NODE_WIDTH,
    MIN_VIDEO_HEIGHT,
    NODE_MIN_H,
    NODE_MIN_W,
    PROGRESS_HEIGHT,
    SPEEDS,
    clamp01,
    formatTime,
    frameAtTime,
    normalizeMeta,
    placeHoverMenu,
    positionFromClientX,
    previewHeight,
    sameFrameCount,
    timeForFrame,
    widgetHeight,
} from "./sf_video_compare_lib.js";

const NODE_TYPE = "SFVideoCompare";
const EXT_NAME = "sfnodes.SFVideoCompare";
const STATE_PROP = "sfVideoCompareVideos";
const CSS_ID = "sf-video-compare-css";
const WIDGET_NAME = "sf_video_compare_preview";

// ── CSS（主题令牌跟随 Color Palette 明暗；:fullscreen 下预览区改为 fill）──
function injectCSS() {
    injectCSSOnce(CSS_ID, `
.sf-vc-root { display:flex; flex-direction:column; width:100%; height:100%;
  overflow:hidden; box-sizing:border-box; background:var(--sf-panel-bg);
  border:1px solid var(--sf-border-soft); border-radius:6px;
  font:12px sans-serif; color:var(--sf-text); }
.sf-vc-controls { display:flex; align-items:center; gap:5px; flex:0 0 auto;
  height:${CONTROL_HEIGHT}px; padding:0 6px; box-sizing:border-box; background:var(--sf-panel-bg-2); }
.sf-vc-btn { flex:1 1 0; min-width:0; height:26px; padding:0 5px; border-radius:4px;
  border:1px solid var(--sf-border-soft); background:var(--sf-surface); color:var(--sf-text);
  font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  cursor:pointer; user-select:none; }
.sf-vc-btn:hover:not(:disabled) { border-color:var(--sf-acc, #f66744); color:var(--sf-text-strong); }
.sf-vc-btn:disabled { opacity:0.45; cursor:default; }
.sf-vc-btn.on:not(:disabled) { border-color:var(--sf-acc, #f66744); background:rgba(246,103,68,0.20); }
.sf-vc-area { position:relative; flex:0 0 auto; width:100%; min-height:${MIN_VIDEO_HEIGHT}px;
  overflow:hidden; background:#050505; touch-action:none; }
.sf-vc-video { position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
  background:#050505; pointer-events:none; }
.sf-vc-divider { position:absolute; top:0; bottom:0; left:50%; width:1px; z-index:2;
  background:rgba(255,255,255,0.8); pointer-events:none; }
.sf-vc-progress { display:flex; align-items:center; gap:8px; flex:0 0 auto;
  height:${PROGRESS_HEIGHT}px; padding:0 8px; box-sizing:border-box; background:var(--sf-panel-bg-2); }
.sf-vc-range { flex:1 1 auto; min-width:0; }
.sf-vc-time { flex:0 0 auto; min-width:112px; text-align:right; font-size:11px;
  color:var(--sf-text-dim); font-variant-numeric:tabular-nums; }
.sf-vc-menu { display:none; position:fixed; z-index:100000; flex-direction:column; gap:4px;
  min-width:88px; padding:4px; box-sizing:border-box; border:1px solid var(--sf-border-soft);
  border-radius:3px; background:var(--sf-panel-bg-2); box-shadow:0 5px 16px rgba(0,0,0,0.45); }
.sf-vc-menu-item { width:100%; height:27px; padding:0 9px; border-radius:3px; text-align:left;
  border:1px solid var(--sf-border-soft); background:var(--sf-surface); color:var(--sf-text);
  font-size:12px; white-space:nowrap; cursor:pointer; }
.sf-vc-menu-item:hover { border-color:var(--sf-acc, #f66744); }
.sf-vc-menu-item.active { background:rgba(246,103,68,0.25); border-color:var(--sf-acc, #f66744); }
.sf-vc-root:fullscreen { width:100vw; height:100vh; border-radius:0; }
.sf-vc-root:fullscreen .sf-vc-area { flex:1 1 0; height:auto !important; min-height:0; }
`);
}

// ── 运行时状态（_sfVideoCompare 内存态；properties 只存 {a,b} 内容信息）───
function createState() {
    return {
        root: null, controls: null, area: null, progressBar: null,
        aVideo: null, bVideo: null, divider: null,
        progress: null, timeLabel: null, widget: null,
        playButton: null, speedButton: null, audioButton: null,
        frameButton: null, fullscreenButton: null,
        buttons: [], menus: [], speedMenu: null, audioMenu: null,
        aData: null, bData: null,
        aspectRatio: 0,
        position: DEFAULT_POSITION,
        speed: 1,
        audioMode: "muted",
        syncing: false,
        seeking: false,
        frameSync: false,
        frameSyncing: false,
        fullscreen: false,
        syncToken: 0,
        raf: 0,
        fullscreenHandler: null,
        cleanup: [],
    };
}

function hasAnyVideo(st) {
    return !!(st?.aData || st?.bData);
}

function bothVideos(st) {
    return !!(st?.aData && st?.bData);
}

function activeVideos(st) {
    return [st.aData ? st.aVideo : null, st.bData ? st.bVideo : null].filter(Boolean);
}

function masterVideo(st) {
    if (st.aData) return st.aVideo;
    if (st.bData) return st.bVideo;
    return null;
}

function elementDuration(video) {
    const d = Number(video?.duration);
    return Number.isFinite(d) && d > 0 ? d : 0;
}

function markDirty() {
    app?.canvas?.setDirtyCanvas?.(true, false);
}

// ── DOM 构建 ────────────────────────────────────────────────────────────
function makeButton(label, title, onClick) {
    const btn = el("button", "sf-vc-btn", label);
    btn.type = "button";
    btn.title = title;
    btn.addEventListener("pointerdown", (e) => e.stopPropagation());
    btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!btn.disabled) onClick();
        btn.blur();
    });
    return btn;
}

function makeVideoElement() {
    const video = el("video", "sf-vc-video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    return video;
}

// 悬停菜单：优先放按钮上方（全屏时挂进 fullscreen 元素，否则 body），
// 离开按钮/菜单 160ms 后隐藏（与 TE_MAN 同节奏）。
function makeHoverMenu(button, items, getValue, onPick) {
    const menu = el("div", "sf-vc-menu");
    let hideTimer = null;
    const rows = items.map(({ label, value }) => {
        const row = el("button", "sf-vc-menu-item", label);
        row.type = "button";
        row.addEventListener("pointerdown", (e) => e.stopPropagation());
        row.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            onPick(value);
            refresh();
            hide();
        });
        menu.append(row);
        return { row, value };
    });
    const refresh = () => {
        const current = getValue();
        for (const { row, value } of rows) row.classList.toggle("active", value === current);
        if (button.disabled) hide();
    };
    const hide = () => {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
        menu.style.display = "none";
    };
    const scheduleHide = () => {
        if (hideTimer) clearTimeout(hideTimer);
        hideTimer = setTimeout(hide, 160);
    };
    const show = () => {
        if (button.disabled) return;
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
        const host = document.fullscreenElement || document.body;
        if (menu.parentElement !== host) host.append(menu);
        refresh();
        menu.style.display = "flex";
        menu.style.visibility = "hidden";
        const buttonRect = button.getBoundingClientRect();
        const menuRect = menu.getBoundingClientRect();
        const pos = placeHoverMenu(buttonRect, menuRect.width, menuRect.height, window.innerWidth, window.innerHeight, 6);
        menu.style.left = `${pos.left}px`;
        menu.style.top = `${pos.top}px`;
        menu.style.visibility = "visible";
    };
    button.addEventListener("mouseenter", show);
    button.addEventListener("mouseleave", scheduleHide);
    menu.addEventListener("mouseenter", () => {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
    });
    menu.addEventListener("mouseleave", scheduleHide);
    menu.addEventListener("pointerdown", (e) => e.stopPropagation());
    return {
        show,
        hide,
        refresh,
        element: menu,
        destroy() {
            if (hideTimer) clearTimeout(hideTimer);
            menu.remove();
        },
    };
}

function buildWidgetDom(st) {
    st.root = el("div", "sf-vc-root");
    st.controls = el("div", "sf-vc-controls");
    st.area = el("div", "sf-vc-area");
    st.aVideo = makeVideoElement();
    st.bVideo = makeVideoElement();
    st.divider = el("div", "sf-vc-divider");
    st.area.append(st.aVideo, st.bVideo, st.divider);

    st.progressBar = el("div", "sf-vc-progress");
    st.progress = document.createElement("input");
    st.progress.type = "range";
    st.progress.min = "0";
    st.progress.max = "1000";
    st.progress.step = "1";
    st.progress.value = "0";
    st.progress.className = "sf-vc-range";
    st.progress.title = "播放进度";
    st.timeLabel = el("span", "sf-vc-time", "0:00.00 / 0:00.00");
    st.progressBar.append(st.progress, st.timeLabel);

    st.root.append(st.controls, st.area, st.progressBar);
}

// ── 媒体源与显示 ────────────────────────────────────────────────────────
function releaseVideo(video) {
    if (!video) return;
    try { video.pause(); } catch { /* 忽略 */ }
    video.removeAttribute("src");
    try { video.load(); } catch { /* 忽略 */ }
}

function setVideoSource(video, data) {
    releaseVideo(video);
    if (!data) return;
    const url = buildSourceURL(data, true);
    if (!url) return;
    video.src = url;
    video.load();
}

// 暂停态下把首帧画出来（preload=metadata 时部分浏览器停在黑帧）。
function renderPausedFrame(video) {
    if (!video || !video.src || video.readyState < 2) return;
    if (!video.paused) return;
    try { video.currentTime = 0; } catch { /* 忽略 */ }
}

function updateClip(node) {
    const st = node._sfVideoCompare;
    if (!st?.bVideo) return;
    const pct = `${(clamp01(st.position) * 100).toFixed(3)}%`;
    st.bVideo.style.clipPath = `inset(0 0 0 ${pct})`;
    if (st.divider) {
        st.divider.style.left = pct;
        st.divider.style.display = bothVideos(st) ? "" : "none";
    }
}

function updateAudio(node) {
    const st = node._sfVideoCompare;
    if (!st) return;
    if (!AUDIO_MODES.includes(st.audioMode)) st.audioMode = "muted";
    st.aVideo.muted = st.audioMode !== "a";
    st.bVideo.muted = st.audioMode !== "b";
    updateControls(node);
}

function applyPlaybackRates(node) {
    const st = node._sfVideoCompare;
    if (!st) return;
    const rate = Number(st.speed) || 1;
    st.aVideo.playbackRate = rate;
    st.bVideo.playbackRate = rate;
}

function masterDuration(st) {
    const duration = elementDuration(masterVideo(st));
    if (duration > 0) return duration;
    return st.aData?.duration || st.bData?.duration || 0;
}

function updateProgress(node) {
    const st = node._sfVideoCompare;
    if (!st || st.seeking) return;
    const duration = masterDuration(st);
    const video = masterVideo(st);
    const time = Math.min(Number(video?.currentTime) || 0, duration || Infinity);
    st.progress.value = duration > 0 ? String(Math.round((time / duration) * 1000)) : "0";
    st.timeLabel.textContent = `${formatTime(time)} / ${formatTime(duration)}`;
}

function updatePreviewSize(node) {
    const st = node._sfVideoCompare;
    if (!st?.area || st.fullscreen) return;
    const width = st.root?.clientWidth || node.size?.[0] || INITIAL_NODE_WIDTH;
    const height = previewHeight(width, st.aspectRatio);
    const cssHeight = `${height}px`;
    if (st.area.style.height !== cssHeight) st.area.style.height = cssHeight;
    const total = CONTROL_HEIGHT + height + PROGRESS_HEIGHT + 8;
    if ((node.size?.[1] || 0) < total) node.setSize?.([Math.max(node.size?.[0] || 0, NODE_MIN_W), total]);
    markDirty();
}

function updateControls(node) {
    const st = node._sfVideoCompare;
    if (!st?.playButton) return;
    const ready = hasAnyVideo(st);
    st.playButton.disabled = !ready;
    st.playButton.textContent = st.syncing ? "重新播放" : "同步播放";
    st.speedButton.disabled = !ready;
    st.speedButton.textContent = `播放速度 ${st.speed}x`;
    st.audioButton.disabled = !ready;
    st.audioButton.textContent = st.audioMode === "a" ? "音频 A" : st.audioMode === "b" ? "音频 B" : "静音";
    st.frameButton.disabled = !bothVideos(st) || !sameFrameCount(st.aData, st.bData) || st.frameSyncing;
    st.frameButton.textContent = "同步帧";
    st.frameButton.classList.toggle("on", st.frameSync);
    st.fullscreenButton.disabled = !ready;
    for (const menu of st.menus) menu.refresh();
}

// ── 播放控制 ────────────────────────────────────────────────────────────
function alignBToATime(node) {
    const st = node._sfVideoCompare;
    if (!bothVideos(st)) return;
    st.bVideo.currentTime = Math.min(st.aVideo.currentTime, st.bVideo.duration || Infinity);
}

function alignBToAFrame(node) {
    const st = node._sfVideoCompare;
    if (!bothVideos(st) || !sameFrameCount(st.aData, st.bData)) return false;
    const aFps = st.aData.frame_rate || 0;
    const bFps = st.bData.frame_rate || 0;
    if (aFps <= 0 || bFps <= 0) return false;
    const frame = frameAtTime(st.aVideo.currentTime, aFps, st.aData.frame_count);
    st.bVideo.currentTime = Math.min(timeForFrame(frame, bFps), st.bVideo.duration || Infinity);
    return true;
}

function seekVideo(video, time) {
    return new Promise((resolve) => {
        let done = false;
        let timer = 0;
        const finish = (ok) => {
            if (done) return;
            done = true;
            clearTimeout(timer);
            video.removeEventListener("seeked", onSeeked);
            video.removeEventListener("error", onError);
            resolve(ok);
        };
        const onSeeked = () => finish(true);
        const onError = () => finish(false);
        timer = setTimeout(() => finish(false), 700);
        if (Math.abs(video.currentTime - time) < 0.001) {
            finish(true);
            return;
        }
        video.addEventListener("seeked", onSeeked, { once: true });
        video.addEventListener("error", onError, { once: true });
        video.currentTime = time;
    });
}

function keepVideosSynchronized(node, token) {
    const st = node._sfVideoCompare;
    if (!st?.syncing || token !== st.syncToken) return;
    const videos = activeVideos(st);
    const anyEnded = videos.some((video) => {
        const duration = elementDuration(video) || (video === st.aVideo ? st.aData?.duration : st.bData?.duration) || 0;
        return video.ended || (duration > 0 && video.currentTime >= duration - 0.03);
    });
    if (anyEnded) {
        st.syncing = false;
        void startPlayback(node, true);
        return;
    }
    updateProgress(node);
    st.raf = requestAnimationFrame(() => keepVideosSynchronized(node, token));
}

async function startPlayback(node, fromStart) {
    const st = node._sfVideoCompare;
    if (!st) return;
    const videos = activeVideos(st);
    if (!videos.length) return;
    if (!fromStart && st.syncing) return;
    videos.forEach((video) => video.pause());
    if (fromStart || videos.some((video) => video.ended)) {
        videos.forEach((video) => {
            try { video.currentTime = 0; } catch { /* 忽略 */ }
        });
        st.frameSync = false;
    } else if (st.frameSync && bothVideos(st)) {
        alignBToAFrame(node);
    } else if (bothVideos(st)) {
        alignBToATime(node);
    }
    applyPlaybackRates(node);
    st.syncing = true;
    const token = ++st.syncToken;
    updateControls(node);
    await Promise.all(videos.map((video) => video.play().catch(() => {})));
    if (token !== st.syncToken) return;
    if (videos.every((video) => video.paused)) {
        st.syncing = false;
        updateControls(node);
        return;
    }
    keepVideosSynchronized(node, token);
}

async function restartPlayback(node) {
    const st = node._sfVideoCompare;
    if (!st) return;
    st.syncing = false;
    st.syncToken += 1;
    st.frameSync = false;
    applyPlaybackRates(node);
    await startPlayback(node, true);
}

async function togglePlayback(node) {
    const st = node._sfVideoCompare;
    if (!st) return;
    const videos = activeVideos(st);
    if (!videos.length) return;
    if (st.syncing || videos.some((video) => !video.paused)) {
        videos.forEach((video) => video.pause());
        st.syncing = false;
        st.syncToken += 1;
        updateControls(node);
        updateProgress(node);
    } else {
        await startPlayback(node, false);
    }
}

async function syncFrames(node) {
    const st = node._sfVideoCompare;
    if (!st || !bothVideos(st) || st.frameSyncing) return;
    if (!sameFrameCount(st.aData, st.bData)) return;
    const aFps = st.aData.frame_rate || 0;
    const bFps = st.bData.frame_rate || 0;
    if (aFps <= 0 || bFps <= 0) return;
    const wasPlaying = st.syncing || !st.aVideo.paused || !st.bVideo.paused;
    st.frameSyncing = true;
    st.syncing = false;
    st.syncToken += 1;
    st.aVideo.pause();
    st.bVideo.pause();
    updateControls(node);
    const frame = frameAtTime(st.aVideo.currentTime, aFps, st.aData.frame_count);
    const [aOk, bOk] = await Promise.all([
        seekVideo(st.aVideo, timeForFrame(frame, aFps)),
        seekVideo(st.bVideo, timeForFrame(frame, bFps)),
    ]);
    st.frameSyncing = false;
    if (!aOk || !bOk) {
        updateControls(node);
        return;
    }
    st.frameSync = true;
    applyPlaybackRates(node);
    updateControls(node);
    if (!wasPlaying) {
        updateProgress(node);
        return;
    }
    st.syncing = true;
    const token = ++st.syncToken;
    await Promise.all([
        st.aVideo.play().catch(() => {}),
        st.bVideo.play().catch(() => {}),
    ]);
    if (token !== st.syncToken) return;
    if (st.aVideo.paused || st.bVideo.paused) {
        st.syncing = false;
        updateControls(node);
        return;
    }
    keepVideosSynchronized(node, token);
}

function seekFromProgress(node) {
    const st = node._sfVideoCompare;
    const duration = masterDuration(st);
    const video = masterVideo(st);
    if (!duration || !video) return;
    const time = (Number(st.progress.value) / 1000) * duration;
    video.currentTime = Math.min(time, elementDuration(video) || time);
    if (bothVideos(st)) {
        if (!st.frameSync) alignBToATime(node);
        else alignBToAFrame(node);
    }
    st.timeLabel.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
}

function updatePosition(node, clientX) {
    const st = node._sfVideoCompare;
    if (!st?.area || !bothVideos(st)) return;
    const rect = st.area.getBoundingClientRect?.();
    if (!rect?.width) return;
    st.position = positionFromClientX(clientX, rect);
    updateClip(node);
    markDirty();
}

function toggleFullscreen(node) {
    const st = node._sfVideoCompare;
    if (!st?.root) return;
    if (document.fullscreenElement === st.root) document.exitFullscreen?.();
    else st.root.requestFullscreen?.();
}

function onFullscreenChange(node) {
    const st = node._sfVideoCompare;
    if (!st?.root) return;
    st.fullscreen = document.fullscreenElement === st.root;
    st.fullscreenButton.textContent = st.fullscreen ? "退出全屏" : "全屏";
    if (!st.fullscreen) updatePreviewSize(node);
}

// ── 内容装载（执行返回 ui / properties 恢复）─────────────────────────────
function writeState(node, st) {
    node.properties = node.properties || {};
    if (st.aData || st.bData) node.properties[STATE_PROP] = { a: st.aData, b: st.bData };
    else delete node.properties[STATE_PROP];
}

function setVideoPair(node, aRaw, bRaw) {
    const st = node._sfVideoCompare;
    if (!st) return;
    st.aData = normalizeMeta(aRaw);
    st.bData = normalizeMeta(bRaw);
    writeState(node, st);
    // 单视频自动占满：只接 B 时把分界线推到最左（修正原版只显示右半的行为）
    st.position = bothVideos(st) ? DEFAULT_POSITION : (st.bData ? 0 : DEFAULT_POSITION);
    st.syncing = false;
    st.syncToken += 1;
    st.frameSync = false;
    st.frameSyncing = false;
    for (const video of [st.aVideo, st.bVideo]) {
        if (video) video.pause();
    }
    setVideoSource(st.aVideo, st.aData);
    setVideoSource(st.bVideo, st.bData);
    applyPlaybackRates(node);
    updateAudio(node);
    updateClip(node);
    updateProgress(node);
    updatePreviewSize(node);
    updateControls(node);
    markDirty();
}

function restoreVideoPair(node) {
    const st = node._sfVideoCompare;
    const saved = node.properties?.[STATE_PROP];
    if (!st || !saved) return;
    setVideoPair(node, saved.a, saved.b);
}

function firstEntry(list) {
    return Array.isArray(list) && list.length ? list[0] : null;
}

// ── 装载 / 事件 / 清理 ──────────────────────────────────────────────────
function installEvents(node, st) {
    const onAreaMove = (e) => updatePosition(node, e.clientX);
    st.area.addEventListener("pointermove", onAreaMove);
    st.area.addEventListener("pointerdown", onAreaMove);
    st.area.addEventListener("click", () => void togglePlayback(node));
    st.aVideo.addEventListener("loadedmetadata", () => onVideoMetadata(node, st.aVideo, true));
    st.bVideo.addEventListener("loadedmetadata", () => onVideoMetadata(node, st.bVideo, false));
    st.aVideo.addEventListener("loadeddata", () => renderPausedFrame(st.aVideo));
    st.bVideo.addEventListener("loadeddata", () => renderPausedFrame(st.bVideo));
    st.progress.addEventListener("pointerdown", () => { st.seeking = true; });
    st.progress.addEventListener("input", () => seekFromProgress(node));
    st.progress.addEventListener("change", () => {
        seekFromProgress(node);
        st.seeking = false;
        updateProgress(node);
    });
    st.root.addEventListener("contextmenu", (e) => e.preventDefault());
    st.cleanup.push(installCanvasZoomPassthrough(st.root));
    st.fullscreenHandler = () => onFullscreenChange(node);
    document.addEventListener("fullscreenchange", st.fullscreenHandler);
    // Vue 模式 onResize 不触发（§44）：宽度变化经 ResizeObserver 兜住
    if (typeof ResizeObserver === "function") {
        const observer = new ResizeObserver(() => updatePreviewSize(node));
        observer.observe(st.root);
        st.cleanup.push(() => observer.disconnect());
    }
}

function onVideoMetadata(node, video, isA) {
    const st = node._sfVideoCompare;
    if (!st) return;
    if (video.videoWidth && video.videoHeight) {
        const ratio = video.videoWidth / video.videoHeight;
        if (isA || !st.aData) st.aspectRatio = ratio;
    }
    renderPausedFrame(video);
    updateProgress(node);
    updatePreviewSize(node);
}

function installVideoCompare(node) {
    if (node._sfVideoCompare?.root) return;
    injectCSS();
    const st = createState();
    buildWidgetDom(st);
    node._sfVideoCompare = st;

    st.playButton = makeButton("同步播放", "从头同步播放两个视频；播放中点击视频区域暂停/继续", () => void restartPlayback(node));
    st.speedButton = makeButton(`播放速度 ${st.speed}x`, "选择播放速度", () => st.speedMenu.show());
    st.audioButton = makeButton("静音", "选择静音、音频 A 或音频 B", () => st.audioMenu.show());
    st.frameButton = makeButton("同步帧", "两个视频总帧数相同时，把 B 对齐到 A 的当前帧", () => void syncFrames(node));
    st.fullscreenButton = makeButton("全屏", "全屏对比", () => toggleFullscreen(node));
    st.buttons = [st.playButton, st.speedButton, st.audioButton, st.frameButton, st.fullscreenButton];
    st.controls.append(...st.buttons);

    st.speedMenu = makeHoverMenu(
        st.speedButton,
        SPEEDS.map((value) => ({ label: `${value}x`, value })),
        () => st.speed,
        (value) => {
            st.speed = value;
            applyPlaybackRates(node);
            updateControls(node);
        },
    );
    st.audioMenu = makeHoverMenu(
        st.audioButton,
        [
            { label: "静音", value: "muted" },
            { label: "音频 A", value: "a" },
            { label: "音频 B", value: "b" },
        ],
        () => st.audioMode,
        (value) => {
            st.audioMode = value;
            updateAudio(node);
        },
    );
    st.menus = [st.speedMenu, st.audioMenu];

    st.widget = node.addDOMWidget(WIDGET_NAME, WIDGET_NAME, st.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => NODE_MIN_H,
    });
    applyAdaptiveCanvasOnly(st.widget);
    st.widget.computeSize = (width) => [width, widgetHeight(width, st.aspectRatio)];

    // 拖拽 resize 双端都经 node.computeSize（onResize 是 legacy-only，§44）
    const origComputeSize = node.computeSize;
    node.computeSize = function (...args) {
        const base = origComputeSize ? origComputeSize.apply(this, args) : [0, 0];
        const width = Math.max(Number(base?.[0]) || 0, NODE_MIN_W);
        return [
            width,
            Math.max(Number(base?.[1]) || 0, widgetHeight(width, st.aspectRatio), NODE_MIN_H),
        ];
    };

    installEvents(node, st);
    updateClip(node);
    updateAudio(node);
    updateControls(node);
    updateProgress(node);
    updatePreviewSize(node);
}

function cleanupVideoCompare(node) {
    const st = node._sfVideoCompare;
    if (!st) return;
    st.syncing = false;
    st.syncToken += 1;
    if (st.raf) cancelAnimationFrame(st.raf);
    releaseVideo(st.aVideo);
    releaseVideo(st.bVideo);
    for (const menu of st.menus) menu.destroy();
    for (const dispose of st.cleanup || []) {
        try { dispose(); } catch { /* 忽略 */ }
    }
    if (st.fullscreenHandler) document.removeEventListener("fullscreenchange", st.fullscreenHandler);
    node._sfVideoCompare = null;
}

app.registerExtension({
    name: EXT_NAME,
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            installVideoCompare(this);
            const width = Math.max(this.size?.[0] || 0, INITIAL_NODE_WIDTH);
            const height = Math.max(this.size?.[1] || 0, INITIAL_NODE_HEIGHT);
            this.setSize?.([width, height]);
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = origConfigure?.apply(this, arguments);
            installVideoCompare(this);
            restoreVideoPair(this);
            return result;
        };

        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origExecuted?.apply(this, arguments);
            installVideoCompare(this);
            setVideoPair(this, firstEntry(message?.a_videos), firstEntry(message?.b_videos));
        };

        const origResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = origResize?.apply(this, arguments);
            updatePreviewSize(this);
            return result;
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            cleanupVideoCompare(this);
            return origRemoved?.apply(this, arguments);
        };
    },
});
