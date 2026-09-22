// sf_video_compare_lib 纯逻辑冒烟（Node 直接运行：node tests/test_video_compare_lib.mjs）
// 覆盖：时间格式、后端 ui 元数据归一、帧/时间换算与夹紧、分界线几何、
// 预览高度估算、悬停菜单定位（上方优先 / 下方回退 / 四向钳位）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_video_compare_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_video_compare_lib.js"), tmpMjs);
const tmpUrl = pathToFileURL(tmpMjs).href;

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}

const L = await import(tmpUrl);

// ── formatTime ──
check("formatTime 0", L.formatTime(0) === "0:00.00");
check("formatTime 5.2", L.formatTime(5.2) === "0:05.20");
check("formatTime 65.5", L.formatTime(65.5) === "1:05.50");
check("formatTime 负数归零", L.formatTime(-3) === "0:00.00");
check("formatTime 非数字归零", L.formatTime(undefined) === "0:00.00");

// ── normalizeMeta ──
const meta = L.normalizeMeta({ filename: "a.mp4", subfolder: "s", type: "temp", frame_count: "48", frame_rate: "24", duration: "2" });
check("normalizeMeta 字符串数字归一",
  meta.filename === "a.mp4" && meta.subfolder === "s" && meta.type === "temp"
  && meta.frame_count === 48 && meta.frame_rate === 24 && meta.duration === 2);
check("normalizeMeta 缺 filename 无效", L.normalizeMeta({ frame_count: 1 }) === null);
check("normalizeMeta 空值无效", L.normalizeMeta(null) === null && L.normalizeMeta(undefined) === null);
const meta2 = L.normalizeMeta({ filename: "b.mp4", frame_count: 90, frame_rate: 30 });
check("normalizeMeta 缺时长用 帧数/帧率 兜底", meta2.duration === 3 && meta2.subfolder === "" && meta2.type === "temp");
check("normalizeMeta 非法数字归零",
  L.normalizeMeta({ filename: "c.mp4", frame_count: "x", frame_rate: -1, duration: "y" }).frame_count === 0);

// ── sameFrameCount ──
check("sameFrameCount 相等", L.sameFrameCount({ frame_count: 48 }, { frame_count: 48 }) === true);
check("sameFrameCount 不等", L.sameFrameCount({ frame_count: 48 }, { frame_count: 49 }) === false);
check("sameFrameCount 0 不可用", L.sameFrameCount({ frame_count: 0 }, { frame_count: 0 }) === false);
check("sameFrameCount 缺失不可用", L.sameFrameCount({}, { frame_count: 48 }) === false);

// ── frameAtTime / timeForFrame ──
check("frameAtTime 常规", L.frameAtTime(1.0, 24, 48) === 24);
check("frameAtTime 上限夹紧", L.frameAtTime(99, 24, 48) === 47);
check("frameAtTime 负时间归零", L.frameAtTime(-1, 24, 48) === 0);
check("frameAtTime fps 缺失归零", L.frameAtTime(1, 0, 48) === 0);
check("frameAtTime 帧数未知不设上限", L.frameAtTime(10, 24, 0) === 240);
check("timeForFrame 常规", L.timeForFrame(24, 24) === 1);
check("timeForFrame 负帧归零", L.timeForFrame(-5, 24) === 0);
check("timeForFrame fps 缺失归零", L.timeForFrame(24, 0) === 0);

// ── clamp01 / positionFromClientX ──
check("clamp01 上下夹紧", L.clamp01(-1) === 0 && L.clamp01(2) === 1 && L.clamp01(0.25) === 0.25);
check("positionFromClientX 居中", L.positionFromClientX(150, { left: 100, width: 100 }) === 0.5);
check("positionFromClientX 左外归零", L.positionFromClientX(0, { left: 100, width: 100 }) === 0);
check("positionFromClientX 右外归一", L.positionFromClientX(999, { left: 100, width: 100 }) === 1);
check("positionFromClientX 零宽回退 0.5", L.positionFromClientX(10, { left: 0, width: 0 }) === 0.5);

// ── previewHeight / widgetHeight ──
check("previewHeight 按宽高比", L.previewHeight(372, 2) === 180);
check("previewHeight 下限", L.previewHeight(100, 16 / 9) === L.MIN_VIDEO_HEIGHT);
check("previewHeight 非法宽高比回退默认", L.previewHeight(372, 0) === Math.round(360 / (16 / 9)));
check("widgetHeight 三段相加",
  L.widgetHeight(372, 2) === L.CONTROL_HEIGHT + 180 + L.PROGRESS_HEIGHT);

// ── placeHoverMenu ──
const above = L.placeHoverMenu({ left: 200, top: 300, bottom: 326, width: 60 }, 88, 120, 1200, 800);
check("placeHoverMenu 上方优先", above.top === 175 && above.left === 186);
const below = L.placeHoverMenu({ left: 200, top: 60, bottom: 86, width: 60 }, 88, 120, 1200, 800);
check("placeHoverMenu 上方不足转下方", below.top === 91);
const leftClamp = L.placeHoverMenu({ left: 0, top: 300, bottom: 326, width: 10 }, 88, 120, 1200, 800);
check("placeHoverMenu 左钳位", leftClamp.left === 6);
const rightClamp = L.placeHoverMenu({ left: 1190, top: 300, bottom: 326, width: 10 }, 88, 120, 1200, 800);
check("placeHoverMenu 右钳位", rightClamp.left === 1200 - 88 - 6);
const bottomClamp = L.placeHoverMenu({ left: 100, top: 60, bottom: 86, width: 60 }, 88, 120, 1200, 100);
check("placeHoverMenu 上下都放不下时钳回视口", bottomClamp.top === 6);

console.log();
if (failures.length) {
  console.log(`${failures.length} FAILED: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("test_video_compare_lib: all assertions passed");
