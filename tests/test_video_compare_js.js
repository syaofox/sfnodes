// SFVideoCompare 前端逻辑测试（Node 直接运行：node tests/test_video_compare_js.js）
// 覆盖：扩展注册名与节点类型门控；DOM widget 安装与初始尺寸；onExecuted 装载
// （src/properties/分界线/按钮状态）；单视频自动占满；onConfigure 恢复；
// 播放/暂停；同步帧（帧号对齐 + 帧数不同禁用）；速度/音频悬停菜单；分界线拖动；
// computeSize 最小尺寸；onRemoved 清理。
const fs = require("fs");
const path = require("path");

const failures = [];
function check(name, cond) {
  if (cond) console.log("PASS:", name);
  else { failures.push(name); console.log("FAIL:", name); }
}
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

// ---- 假 DOM ----
class FakeClassList {
  constructor() { this.set = new Set(); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : !!force;
    if (on) this.set.add(name); else this.set.delete(name);
    return on;
  }
  contains(name) { return this.set.has(name); }
}

class FakeElement {
  constructor(tag) {
    this.tagName = String(tag || "div").toUpperCase();
    this.children = [];
    this.style = {};
    this.classList = new FakeClassList();
    this.dataset = {};
    this.listeners = {};
    this.parentElement = null;
    this.textContent = "";
    this.className = "";
    this.disabled = false;
    this.value = "0";
    this.muted = false;
    this.paused = true;
    this.ended = false;
    this.duration = NaN;
    this.readyState = 0;
    this.videoWidth = 0;
    this.videoHeight = 0;
    this.src = "";
    this.clientWidth = 0;
    this._currentTime = 0;
  }
  get currentTime() { return this._currentTime; }
  set currentTime(value) {
    this._currentTime = value;
    queueMicrotask(() => this.dispatch("seeked"));
  }
  append(...nodes) {
    for (const node of nodes) { node.parentElement = this; this.children.push(node); }
  }
  appendChild(node) { this.append(node); }
  remove() {
    const parent = this.parentElement;
    if (parent) {
      const index = parent.children.indexOf(this);
      if (index >= 0) parent.children.splice(index, 1);
    }
    this.parentElement = null;
  }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) {
    const arr = this.listeners[type] || [];
    const index = arr.indexOf(fn);
    if (index >= 0) arr.splice(index, 1);
  }
  dispatch(type, event = {}) {
    for (const fn of [...(this.listeners[type] || [])]) {
      fn({ preventDefault() {}, stopPropagation() {}, target: this, ...event });
    }
  }
  setAttribute() {}
  removeAttribute(name) { if (name === "src") this.src = ""; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 200, height: 100, right: 200, bottom: 100 }; }
  load() { this.readyState = 1; }
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
  blur() {}
}

globalThis.document = {
  createElement: (tag) => new FakeElement(tag),
  body: new FakeElement("body"),
  fullscreenElement: null,
  listeners: {},
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
  removeEventListener(type, fn) {
    const arr = this.listeners[type] || [];
    const index = arr.indexOf(fn);
    if (index >= 0) arr.splice(index, 1);
  },
};
globalThis.window = { innerWidth: 1200, innerHeight: 800 };
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 5);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.ResizeObserver = class {
  observe() {}
  disconnect() {}
};

// ---- 假 app / sf_common / 被测模块 ----
const capturedExts = [];
const app = {
  canvas: { setDirtyCanvas() {} },
  registerExtension: (ext) => capturedExts.push(ext),
};

const sfCommon = {
  el: (tag, cls, text) => {
    const element = document.createElement(tag);
    if (cls) element.className = cls;
    if (text != null) element.textContent = text;
    return element;
  },
  injectCSSOnce: () => {},
  applyAdaptiveCanvasOnly: (widget) => widget,
  installCanvasZoomPassthrough: () => () => {},
  buildSourceURL: (part) => (part?.filename ? `/view?filename=${part.filename}&type=${part.type || "temp"}` : null),
};

function loadStripped(file) {
  const raw = fs.readFileSync(path.join(__dirname, "..", "web", file), "utf8");
  // 只剥行首 import（注释里出现的 "import" 字样不能被误吞——裸 /import[^;]+;/ 会跨行删代码）
  return raw
    .replace(/^import[^;]+;/gm, "")
    .replace(/export\s*\{[^}]*\}\s*;?/g, "")
    .replace(/export\s+(?=function|const|let|class|var)/g, "");
}

const libNames = [
  "AUDIO_MODES", "CONTROL_HEIGHT", "DEFAULT_ASPECT", "DEFAULT_POSITION",
  "INITIAL_NODE_HEIGHT", "INITIAL_NODE_WIDTH", "MIN_VIDEO_HEIGHT", "MIN_VIDEO_WIDTH",
  "NODE_MIN_H", "NODE_MIN_W", "PROGRESS_HEIGHT", "SPEEDS", "clamp01", "formatTime",
  "frameAtTime", "normalizeMeta", "placeHoverMenu", "positionFromClientX",
  "previewHeight", "sameFrameCount", "timeForFrame", "widgetHeight",
];
const lib = new Function(loadStripped("sf_video_compare_lib.js") + "\nreturn {" + libNames.join(",") + "};")();

const mainFactory = new Function(
  "app", "sfCommon", "lib",
  `
  const { applyAdaptiveCanvasOnly, buildSourceURL, el, injectCSSOnce, installCanvasZoomPassthrough } = sfCommon;
  const { ${libNames.join(", ")} } = lib;
  ${loadStripped("sf_video_compare.js")}
  return { installVideoCompare, setVideoPair, cleanupVideoCompare };
  `
);
mainFactory(app, sfCommon, lib);

// ---- 假节点 ----
class FakeNode {
  constructor() {
    this.size = [140, 80];
    this.properties = {};
    this.widgets = [];
    this.inputs = [];
    this.outputs = [];
    this.comfyClass = "SFVideoCompare";
  }
  addDOMWidget(name, type, element, options) {
    const widget = { name, type, element, options: { ...options }, computeSize: () => [0, 0] };
    this.widgets.push(widget);
    return widget;
  }
  setSize(size) { this.size = size; }
  setDirtyCanvas() {}
}

function makeNode() {
  const nodeType = { prototype: {} };
  capturedExts[0].beforeRegisterNodeDef(nodeType, { name: "SFVideoCompare" });
  class TestNode extends FakeNode {}
  Object.assign(TestNode.prototype, nodeType.prototype);
  const node = new TestNode();
  node.onNodeCreated();
  return node;
}

const META_A = { filename: "a.mp4", type: "temp", frame_count: 48, frame_rate: 24, duration: 2 };
const META_B = { filename: "b.mp4", type: "temp", frame_count: 48, frame_rate: 24, duration: 2 };

(async () => {
  // ---- 1. 扩展注册与类型门控 ----
  check("扩展注册名", capturedExts.length === 1 && capturedExts[0].name === "sfnodes.SFVideoCompare");
  const otherType = { prototype: {} };
  capturedExts[0].beforeRegisterNodeDef(otherType, { name: "OtherNode" });
  check("非本节点不包装", otherType.prototype.onNodeCreated === undefined);

  // ---- 2. onNodeCreated：DOM widget 与初始状态 ----
  let node = makeNode();
  let st = node._sfVideoCompare;
  check("DOM widget 安装", node.widgets.length === 1 && node.widgets[0].name === "sf_video_compare_preview");
  check("widget 关闭 hideOnZoom/serialize", node.widgets[0].options.hideOnZoom === false && node.widgets[0].options.serialize === false);
  check("初始尺寸不小于默认", node.size[0] >= 460 && node.size[1] >= 390);
  check("五个控制按钮", st.buttons.length === 5 && st.controls.children.length === 5);
  check("无视频时控件禁用",
    st.playButton.disabled && st.speedButton.disabled && st.audioButton.disabled
    && st.frameButton.disabled && st.fullscreenButton.disabled);
  check("computeSize 钳最小尺寸", node.computeSize()[0] >= 360 && node.computeSize()[1] >= 280);
  check("widget computeSize 用宽高比", node.widgets[0].computeSize(400)[1] === lib.widgetHeight(400, 0));

  // ---- 3. onExecuted：装载双视频 ----
  node.onExecuted({ a_videos: [META_A], b_videos: [META_B] });
  st = node._sfVideoCompare;
  check("A/B src 设置", st.aVideo.src.includes("a.mp4") && st.bVideo.src.includes("b.mp4"));
  check("properties 写入 a/b", node.properties.sfVideoCompareVideos.a.filename === "a.mp4"
    && node.properties.sfVideoCompareVideos.b.frame_rate === 24);
  check("分界线默认 50%", st.bVideo.style.clipPath === "inset(0 0 0 50.000%)" && st.divider.style.display === "");
  check("双视频控件可用", !st.playButton.disabled && !st.frameButton.disabled && !st.fullscreenButton.disabled);
  check("按钮初始文案", st.playButton.textContent === "同步播放"
    && st.speedButton.textContent === "播放速度 1x" && st.audioButton.textContent === "静音");
  check("时间标签初始化", st.timeLabel.textContent.includes("/"));

  // ---- 4. 播放/暂停 ----
  st.area.dispatch("click");
  await tick();
  check("点击视频区开始播放", st.syncing === true && st.aVideo.paused === false && st.bVideo.paused === false);
  st.area.dispatch("click");
  await tick();
  check("再次点击暂停", st.syncing === false && st.aVideo.paused === true);
  check("暂停后按钮回到同步播放", st.playButton.textContent === "同步播放");

  // ---- 5. 同步帧 ----
  st.aVideo.currentTime = 0.52;
  st.bVideo.currentTime = 0.1;
  await tick();
  st.frameButton.dispatch("click");
  await tick();
  await tick();
  check("同步帧：A 对齐到整数帧", Math.abs(st.aVideo.currentTime - 0.5) < 1e-6);
  check("同步帧：B 同帧号", Math.abs(st.bVideo.currentTime - 0.5) < 1e-6);
  check("frameSync 开启且按钮高亮", st.frameSync === true && st.frameButton.classList.contains("on"));
  node.onExecuted({ a_videos: [META_A], b_videos: [{ ...META_B, frame_count: 24 }] });
  check("帧数不同禁用同步帧", st.frameButton.disabled === true && st.frameSync === false);

  // ---- 6. 速度/音频悬停菜单 ----
  node.onExecuted({ a_videos: [META_A], b_videos: [META_B] });
  st.speedMenu.show();
  check("速度菜单挂到 body", st.speedMenu.element.parentElement === document.body);
  st.speedMenu.element.children[6].dispatch("click");
  check("选择 2x：播放速率与文案", st.speed === 2 && st.aVideo.playbackRate === 2
    && st.speedButton.textContent === "播放速度 2x");
  st.audioMenu.show();
  st.audioMenu.element.children[1].dispatch("click");
  check("选择音频 A：A 不静音 B 静音", st.aVideo.muted === false && st.bVideo.muted === true
    && st.audioButton.textContent === "音频 A");

  // ---- 7. 分界线拖动 ----
  st.area.getBoundingClientRect = () => ({ left: 100, top: 0, width: 200, height: 100 });
  st.area.dispatch("pointermove", { clientX: 150 });
  check("pointermove 更新分界线", st.bVideo.style.clipPath === "inset(0 0 0 25.000%)" && st.divider.style.left === "25.000%");

  // ---- 8. 单视频自动占满 ----
  node.onExecuted({ a_videos: [], b_videos: [META_B] });
  check("只接 B：位置 0 且分界线隐藏",
    st.position === 0 && st.bVideo.style.clipPath === "inset(0 0 0 0.000%)" && st.divider.style.display === "none");
  check("只接 B：播放可用、同步帧禁用", !st.playButton.disabled && st.frameButton.disabled);
  node.onExecuted({ a_videos: [META_A], b_videos: [] });
  check("只接 A：B 空源", st.bVideo.src === "" && st.divider.style.display === "none");
  node.onExecuted({});
  check("空 ui：清空内容与 properties", st.aVideo.src === "" && node.properties.sfVideoCompareVideos === undefined);

  // ---- 9. onConfigure 恢复 ----
  const node2 = makeNode();
  node2.properties.sfVideoCompareVideos = { a: META_A, b: META_B };
  node2.onConfigure({});
  check("configure 恢复播放内容", node2._sfVideoCompare.aVideo.src.includes("a.mp4")
    && node2._sfVideoCompare.bVideo.src.includes("b.mp4"));
  check("configure 恢复后分界线 50%", node2._sfVideoCompare.bVideo.style.clipPath === "inset(0 0 0 50.000%)");

  // ---- 10. onRemoved 清理 ----
  const st2 = node2._sfVideoCompare;
  const fullscreenListeners = (document.listeners.fullscreenchange || []).length;
  node2.onRemoved();
  check("onRemoved 释放视频源", st2.aVideo.src === "" && st2.bVideo.src === "");
  check("onRemoved 清空状态", node2._sfVideoCompare === null);
  check("onRemoved 摘除菜单", st2.speedMenu.element.parentElement === null);
  check("onRemoved 摘除 fullscreen 监听",
    (document.listeners.fullscreenchange || []).length === fullscreenListeners - 1);

  console.log();
  if (failures.length) {
    console.log(`${failures.length} FAILED: ${failures.join(", ")}`);
    process.exit(1);
  }
  console.log("test_video_compare_js: all assertions passed");
})();
