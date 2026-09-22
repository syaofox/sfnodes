// ==========================================================================
// sf_video_compare_lib.js — SFVideoCompare 纯逻辑（无 app / DOM 依赖，可 .mjs 直测）
// ==========================================================================
// 时间/帧换算、后端 ui 元数据归一、分界线与悬停菜单几何、预览高度估算。
// DOM 与播放控制主模块在 sf_video_compare.js；本文件不得 import sf_common.js
// （纯模块边界，见 experience/patterns.md §26）。
// ==========================================================================

export const SPEEDS = [0.25, 0.5, 0.75, 1, 1.2, 1.5, 2];
export const AUDIO_MODES = ["muted", "a", "b"];

export const DEFAULT_ASPECT = 16 / 9;
export const DEFAULT_POSITION = 0.5;

export const CONTROL_HEIGHT = 38;
export const PROGRESS_HEIGHT = 42;
export const MIN_VIDEO_HEIGHT = 180;
export const MIN_VIDEO_WIDTH = 160;

export const INITIAL_NODE_WIDTH = 460;
export const INITIAL_NODE_HEIGHT = 390;
export const NODE_MIN_W = 360;
export const NODE_MIN_H = 280;

// ── 时间 ────────────────────────────────────────────────────────────────
// "M:SS.ss"（与 TE_MAN 进度条时间标签同形）。
export function formatTime(seconds) {
    const s = Math.max(0, Number(seconds) || 0);
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
}

// ── 后端 ui 元数据归一 ───────────────────────────────────────────────────
// 形如 {filename, subfolder, type, frame_count, frame_rate, duration}；
// 无 filename 视为无效（null）。duration 缺失时用 帧数/帧率 兜底。
export function normalizeMeta(data) {
    if (!data || typeof data !== "object" || !data.filename) return null;
    const frameCount = Math.max(0, Math.round(Number(data.frame_count) || 0));
    const frameRate = Math.max(0, Number(data.frame_rate) || 0);
    let duration = Math.max(0, Number(data.duration) || 0);
    if (duration <= 0 && frameRate > 0 && frameCount > 0) duration = frameCount / frameRate;
    return {
        filename: String(data.filename),
        subfolder: String(data.subfolder || ""),
        type: String(data.type || "temp"),
        frame_count: frameCount,
        frame_rate: frameRate,
        duration,
    };
}

// ── 帧/时间换算（「同步帧」用）───────────────────────────────────────────
export function sameFrameCount(aData, bData) {
    const ca = Number(aData?.frame_count) || 0;
    const cb = Number(bData?.frame_count) || 0;
    return ca > 0 && ca === cb;
}

export function frameAtTime(time, fps, frameCount) {
    const rate = Number(fps) || 0;
    const total = Math.max(0, Math.round(Number(frameCount) || 0));
    if (rate <= 0) return 0;
    const raw = Math.round((Number(time) || 0) * rate);
    const max = total > 0 ? total - 1 : Number.MAX_SAFE_INTEGER;
    return Math.max(0, Math.min(raw, max));
}

export function timeForFrame(frame, fps) {
    const rate = Number(fps) || 0;
    if (rate <= 0) return 0;
    return Math.max(0, Math.round(Number(frame) || 0)) / rate;
}

// ── 分界线几何 ──────────────────────────────────────────────────────────
export function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
}

export function positionFromClientX(clientX, rect) {
    const width = Number(rect?.width) || 0;
    if (width <= 0) return DEFAULT_POSITION;
    return clamp01((Number(clientX) - Number(rect.left || 0)) / width);
}

// ── 预览高度估算（widget computeSize / 最小节点高）──────────────────────
export function previewHeight(nodeWidth, aspect) {
    const width = Math.max((Number(nodeWidth) || INITIAL_NODE_WIDTH) - 12, MIN_VIDEO_WIDTH);
    const ar = Number(aspect) > 0 ? Number(aspect) : DEFAULT_ASPECT;
    return Math.max(MIN_VIDEO_HEIGHT, Math.round(width / ar));
}

export function widgetHeight(nodeWidth, aspect) {
    return CONTROL_HEIGHT + previewHeight(nodeWidth, aspect) + PROGRESS_HEIGHT;
}

// ── 悬停菜单定位（fixed 坐标）───────────────────────────────────────────
// 优先放在按钮上方；上方空间不足时改放下方，最后再钳回视口内。
export function placeHoverMenu(buttonRect, menuWidth, menuHeight, viewportWidth, viewportHeight, margin = 6) {
    const btn = buttonRect || {};
    const w = Number(menuWidth) || 0;
    const h = Number(menuHeight) || 0;
    const vw = Number(viewportWidth) || 0;
    const vh = Number(viewportHeight) || 0;
    const maxLeft = Math.max(margin, vw - w - margin);
    const left = Math.min(Math.max(margin, (Number(btn.left) || 0) + ((Number(btn.width) || 0) - w) / 2), maxLeft);
    const above = (Number(btn.top) || 0) - h - 5;
    const top = above >= margin
        ? above
        : Math.min((Number(btn.bottom) || 0) + 5, Math.max(margin, vh - h - margin));
    return { left, top };
}
