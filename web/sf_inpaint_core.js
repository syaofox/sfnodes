// ============================================================
// SF Inpaint Crop — 编辑器核心（构造、open/close、UI 构建）
// 移植自 comfyui-pixaroma js/inpaint_crop/core.mjs
// ============================================================
import {
  BRAND,
  createEditorLayout,
  createPanel,
  createButton,
  createPillGrid,
  createSliderRow,
  createInfo,
} from "./sf_crop_framework.js";
import { installGraphUndoGuard } from "./sf_crop_undo_guard.js";
import { api } from "/scripts/api.js";

export { BRAND };

// 绝对安全的 URL：api.apiURL 处理托管部署基址，失败降级原样返回
function pixApiUrl(route) {
  try {
    if (typeof api?.apiURL === "function") return api.apiURL(route);
  } catch {
    /* 降级 */
  }
  return route;
}

// ── 图标（内联 data URI，项目惯例见 sf_crop_framework.js）──────────────
const ICONS = {
  swap: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M13.622,36.23L.925,20.673c-.339-.416-.509-.922-.509-1.429s.17-1.013.509-1.429L13.622,2.259c1.344-1.646,4.01-.696,4.01,1.429v10.319h41.824c1.027,0,1.86.833,1.86,1.86v6.756c0,1.027-.833,1.86-1.86,1.86H17.632v10.319c0,2.125-2.666,3.075-4.01,1.429ZM63.491,43.327l-12.697-15.557c-1.344-1.646-4.01-.696-4.01,1.429v10.319H4.96c-1.027,0-1.86.833-1.86,1.86v6.756c0,1.027.833,1.86,1.86,1.86h41.824v10.319c0,2.125,2.667,3.075,4.01,1.429l12.697-15.557c.339-.416.509-.922.509-1.429s-.17-1.013-.509-1.429Z"/></svg>'),
  delete: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M11.381,21.396h41.237l-5.143,38.175c-.281,1.924-1.931,3.35-3.876,3.35h-23.2c-1.944,0-3.594-1.426-3.876-3.35l-5.143-38.175ZM50.148,6.863h-12.997v-2.935c0-1.176-.953-2.13-2.13-2.13h-6.043c-1.176,0-2.13.953-2.13,2.13v2.935h-12.997c-3.934.235-7.003,3.493-7.003,7.434v2.984h50.302v-2.984c0-3.941-3.07-7.199-7.003-7.434Z"/></svg>'),
  eraser: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M49.328,60.913h-31.466c-1.313,0-2.416-1.115-3.263-1.919l-8.617-8.202c-.99-.949-2.089-1.841-2.871-2.909-2.276-3.109-1.91-6.961.552-9.809L33.565,3.556c2.432-2.805,6.407-2.828,9.053-.3l16.847,16.054c2.565,2.438,2.971,6.453.543,9.237l-24.506,28.138h14.007c1.067.005,1.661,1.316,1.644,2.189-.015.784-.7,2.04-1.825,2.04ZM29.48,56.666c.65-.311,1.036-.759,1.48-1.268l5.89-6.745-20.339-19.059-9.914,11.531c-1.384,1.61-.89,3.617.512,4.952l9.892,9.42c.394.375.945,1.167,1.53,1.169h10.949Z"/></svg>'),
  image: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M63.18,52.412c-.004,3.614-2.766,7.013-6.442,7.014H7.039c-3.249,0-6.211-3.26-6.219-6.459V10.835c.586-3.382,2.813-5.753,6.206-6.261h49.968c3.801.005,6.321,3.865,6.061,7.231-.082,1.062.127,1.823.126,2.78v37.828ZM35.44,41.668l7.523-7.113c1.843-1.979,4.642-2.324,6.674-.466l7.08,5.615V11.017H7.239v33.764l7.227-6.84c2.076-2.077,4.091-4.217,6.475-5.902,1.87-1.322,3.965-.17,5.418,1.181l9.082,8.448ZM46.33,15.667c-3.14,0-5.685,2.545-5.685,5.685s2.545,5.685,5.685,5.685,5.685-2.545,5.685-5.685-2.545-5.685-5.685-5.685Z"/></svg>'),
  reset: "data:image/svg+xml," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M5.076,36.174h7.951c-.055,7.957,5.073,14.981,12.165,17.655,7.796,2.94,16.386.61,21.503-5.823,3.292-4.139,4.582-9.187,3.971-14.41-1.006-8.613-7.761-15.308-16.394-16.422v6.522c-.005.556-.627,1.25-1.062,1.41-.492.182-1.518.167-1.943-.194l-11.966-10.184c-.593-.505-.816-1.117-.82-1.857-.003-.68.401-1.278,1.043-1.825L31.164,1.139c.559-.476,1.425-.561,2.095-.312.476.177,1.01.956,1.01,1.62v6.426c4.626.52,9.001,1.929,12.753,4.544,6.46,4.503,10.607,11.215,11.617,18.995.345,2.661.41,4.965.007,7.63-.945,6.242-3.878,11.989-8.439,16.154-12.151,11.097-30.942,8.938-40.35-4.563-3.056-4.386-4.798-9.678-4.781-15.459ZM38.682,41.707v-9.247c.003-1.069-.713-1.902-1.696-2.175h-10.107c-1.019.222-1.738,1.087-1.738,2.147v9.334c0,1.196.921,2.138,2.127,2.138h9.073c1.213,0,2.339-.973,2.342-2.197Z"/></svg>'),
};

// Brush default size (used by the "Reset to default" button in the Brush panel).
const DEFAULT_BRUSH_SIZE = 80;   // px 直径

// Mask/seam 预览色调选项（仅显示）。选橙色时裁剪框改画白色。
export const INPAINT_PREVIEW_COLORS = {
  Red: "#f6303a", Green: "#25d366", Blue: "#3a9bff", Yellow: "#ffd21a", Orange: "#ff8c1a",
};

export const InpaintAPI = {
  async uploadSrc(projectId, dataURL) {
    const res = await api.fetchApi(pixApiUrl("/api/sfnodes/inpaint/upload_src"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId, image: dataURL }),
    });
    return await res.json();
  },
  async saveMask(projectId, dataURL) {
    const res = await api.fetchApi(pixApiUrl("/api/sfnodes/inpaint/save_mask"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId, mask: dataURL }),
    });
    return await res.json();
  },
};

export class InpaintCropEditor {
  constructor() {
    this.onSave = null;          // (stateJsonStr, { context_px }, previewDataURL)
    this.onSaveToDisk = null;    // (previewDataURL)
    this.onClose = null;
    this.onLoadImage = null;     // 宿主断开上游连线
    this.onPreviewColor = null;  // 宿主持久化预览色调设置
    this.el = {};
    this.layout = null;
    this.img = null;
    this.imgW = 0;
    this.imgH = 0;
    this.projectId = null;
    this._scale = 1;        // 实际 源->显示 px = _baseScale * _zoom
    this._baseScale = 1;    // 适配窗口缩放（zoom 1）
    this._zoom = 1;         // 1 = 适配；滚轮朝光标缩放
    this._panX = 0;         // 图像原点显示 px 偏移（适配时为 0）
    this._panY = 0;
    this._dispW = 0;   // 逻辑显示（视口）尺寸，_fitCanvas 在任何绘制前设置
    this._dispH = 0;
    this._srcPath = "";
    this._maskPath = "";
    this._pendingSrcDataURL = null;

    // brush / mask 状态
    this.tool = "add";           // "add" | "erase"
    this.brushSize = DEFAULT_BRUSH_SIZE;   // 显示 px（直径）；跨打开按节点持久化
    this.softness = 0;                     // 锐利笔刷；接缝 Softness 滑块掌管混合
    this.maskOpacity = 0.5;
    this.maskVisible = true;
    this.previewColor = "#f6303a";   // mask + seam 预览色调（仅显示）
    this._cropBoxColor = null;       // 裁剪框描边覆盖色（橙色色调时用白色）
    this._painting = false;
    this._lastPt = null;

    // geometry 参数（节点旋钮的快照）
    this.params = {};
    this._bbox = null;           // 原始画过的 bbox（源 px）
    this._region = null;         // computeRegion() 结果

    // undo
    this._undo = [];
    this._redo = [];
  }

  // upstreamUrl: 接入 IMAGE 输入的实时源（优先于保存的 src）。
  // params: { size_mode, target, multiple, context_px, mask_grow }（内部键）
  open(jsonStr, upstreamUrl, params, prefs) {
    let data = {};
    try { data = jsonStr && jsonStr !== "{}" ? JSON.parse(jsonStr) : {}; } catch (e) {}

    this.projectId = data.project_id || "inpaint_" + Date.now() + "_" + Math.random().toString(36).slice(2, 9);
    this._srcPath = data.src_path || "";
    this._maskPath = data.mask_path || "";
    this.params = { ...(params || {}) };
    if (this.params.blend == null) this.params.blend = 16;
    this._fromUpstream = !!upstreamUrl;

    // 恢复上次打开此节点时的笔刷偏好（size / opacity 跨打开持久化；
    // 缺失 -> 使用构造器默认）。
    if (prefs && typeof prefs === "object") {
      if (prefs.brushSize != null) this.brushSize = prefs.brushSize;
      if (prefs.maskOpacity != null) this.maskOpacity = prefs.maskOpacity;
    }

    this._buildUI();
    this.layout.mount();
    this._undoGuardOff = installGraphUndoGuard(() => !!this.el.overlay?.isConnected);

    let sourceURL = null;
    if (upstreamUrl) sourceURL = upstreamUrl;
    else if (this._srcPath) {
      const fn = this._srcPath.split(/[\\/]/).pop();
      sourceURL = pixApiUrl(`/view?filename=${encodeURIComponent(fn)}&type=input&subfolder=sfnodes_inpaint&t=${Date.now()}`);
    }

    if (sourceURL) {
      this._loadImageFromURL(sourceURL, () => {
        // 图像尺寸确定后恢复保存的画过遮罩（尽力而为）
        if (this._maskPath) {
          const mfn = this._maskPath.split(/[\\/]/).pop();
          const murl = pixApiUrl(`/view?filename=${encodeURIComponent(mfn)}&type=input&subfolder=sfnodes_inpaint&t=${Date.now()}`);
          this._loadMaskFromURL(murl);
        }
      });
    } else {
      this._setStatus("没有源图。接入 IMAGE 输入并运行一次工作流，或点 Load Image——也可以直接把图片粘贴/拖进来。");
    }
    this._bindKeys();
    this._bindDropPaste();   // 把图片粘贴/拖进编辑器
  }

  _close() { this.layout?.unmount(); }

  _buildUI() {
    const layout = createEditorLayout({
      editorName: "SF Inpaint Crop",
      editorId: "sf-inpaint-editor",
      showUndoRedo: true,
      showZoomBar: false,
      showStatusBar: true,
      onSave: () => this._save(),
      onClose: () => this._close(),
      onUndo: () => this._doUndo(),
      onRedo: () => this._doRedo(),
      helpTitle: "SF Inpaint Crop — 使用说明",
      helpContent: `
        <b style="color:#f66744;">这是做什么的</b><br>
        在想让模型重画的区域上涂抹。节点会找到涂抹的外接框、加上边距，裁出一块干净
        的图去修复。橙色框就是实际发给模型的区域。<br><br>

        <b style="color:#f66744;">涂抹</b><br>
        <b>画笔 / 橡皮：</b>选工具或按 <kbd>B</kbd> / <kbd>E</kbd>（按住 <kbd>X</kbd> 临时翻转）<br>
        <b>笔刷大小：</b><kbd>[</kbd> / <kbd>]</kbd> 或 Size 滑块<br>
        <b>缩放 / 平移：</b>滚轮朝光标缩放；按住 <kbd>Space</kbd> 拖动（或中键拖动）平移；滚到最小即适配<br>
        <b>显示 / 隐藏遮罩：</b><kbd>H</kbd><br>
        <b>反选 / 清空：</b>侧栏按钮（反选 = 已涂与未涂互换）<br>
        <b>换一张图：</b>Load Image 按钮，或直接把图片<b>粘贴</b> / <b>拖</b>进来<br><br>

        <b style="color:#f66744;">裁剪尺寸</b>（侧栏 + 节点）<br>
        <b>Keep shape：</b>区域长边缩放到 Target，不拉伸（质量最佳）<br>
        <b>Force square：</b>恒为 Target&times;Target 方块<br>
        <b>Free：</b>自然尺寸，仅对齐到 Multiple<br>
        <b>Context margin：</b>涂抹周围包含多少上下文<br><br>

        <b style="color:#f66744;">接缝 — 贴回时的混合</b><br>
        <b>Softness：</b>贴回边缘淡化多远（实时橙色色调预览）<br>
        <b>Mask grow：</b>裁剪前把涂抹区域向外扩一点<br>
        <b>Blend mode — Mask：</b>只替换涂抹过的区域，裁剪图其余部分保留原图（常规修复）<br>
        <b>Blend mode — Whole crop：</b>整个框都被模型版本替换——模型也改了周围光照/环境，
        或对整块做 img2img 式处理时使用<br><br>

        <b style="color:#f66744;">快捷键</b><br>
        <b>撤销 / 重做：</b><kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Shift+Z</kbd> &middot;
        <b>保存：</b><kbd>Ctrl+S</kbd> &middot; <b>关闭：</b><kbd>Escape</kbd>`,
    });
    this.layout = layout;
    layout.onSaveToDisk = () => { this._diskSavePending = true; this._save(); };
    layout.onCleanup = () => {
      if (this._undoGuardOff) { this._undoGuardOff(); this._undoGuardOff = null; }
      this._unbindKeys();
      if (this.onClose) this.onClose();
    };
    this.el.overlay = layout.overlay;
    this.el.workspace = layout.workspace;

    this._buildLeftSidebar(layout.leftSidebar);
    this._buildRightSidebar(layout.rightSidebar, layout.sidebarFooter);

    // canvas 栈：主图 + 遮罩 + 叠加层，顶部再加一个光标 canvas
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;display:inline-block;line-height:0;";
    this.el.canvasWrap = wrap;
    const cvs = document.createElement("canvas");
    cvs.width = 100; cvs.height = 100;
    this.el.canvas = cvs;
    this.el.ctx = cvs.getContext("2d");
    const cur = document.createElement("canvas");
    cur.width = 100; cur.height = 100;
    cur.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;";
    this.el.cursor = cur;
    this.el.curCtx = cur.getContext("2d");
    wrap.append(cvs, cur);
    layout.workspace.appendChild(wrap);

    this._bindMouse(cvs);
    layout.setUndoState({ canUndo: false, canRedo: false });
  }

  _buildLeftSidebar(sidebar) {
    // Mask 工具
    const secTools = createPanel("Mask");
    this._toolGrid = createPillGrid(
      [{ label: "Brush (B)", value: "add" }, { label: "Erase (E)", value: "erase" }],
      2, (v) => this._setTool(v), { activeValue: "add" },
    );
    secTools.content.appendChild(this._toolGrid.el);
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;margin-top:8px;";
    row.append(
      createButton("Invert", { variant: "standard", iconSrc: ICONS.swap, onClick: () => this._invertMask() }),
      createButton("Clear", { variant: "standard", iconSrc: ICONS.delete, onClick: () => this._clearMask() }),
    );
    for (const b of row.children) b.style.flex = "1";
    secTools.content.appendChild(row);
    sidebar.appendChild(secTools.el);

    // Brush
    const secBrush = createPanel("Brush");
    this.el.sizeSlider = createSliderRow("Size", 2, 300, this.brushSize, () => {
      this.brushSize = parseInt(this.el.sizeSlider.numInput.value) || this.brushSize;
    });
    const sizeHint = document.createElement("div");
    sizeHint.innerHTML = "[ smaller  ·  ] bigger<br>scroll = zoom  ·  space-drag = pan";
    sizeHint.style.cssText = "font-size:10px;color:#888;margin-top:5px;line-height:1.5;";
    secBrush.content.append(this.el.sizeSlider.el, sizeHint);
    sidebar.appendChild(secBrush.el);

    // Seam（缝合混合；canvas 实时预览）
    const secSeam = createPanel("Seam — how it blends");
    this.el.blendSlider = createSliderRow("Softness", 0, 150, this.params.blend ?? 16, () => {
      this.params.blend = parseInt(this.el.blendSlider.numInput.value) || 0;
      // softness 超过 context_px 会扩张裁剪图（Option B），裁剪框 + 尺寸徽标
      // 必须重算——不止 seam 色调（与 Mask grow 镜像）。
      this._recomputeRegion();
      this._draw();
    });
    this.el.growSlider = createSliderRow("Mask grow", 0, 256, this.params.mask_grow ?? 4, () => {
      this.params.mask_grow = parseInt(this.el.growSlider.numInput.value) || 0;
      this._recomputeRegion();
      this._draw();
    });
    const bmLabel = document.createElement("div");
    bmLabel.textContent = "Blend mode";
    bmLabel.style.cssText = "font-size:11px;color:#aaa;margin:10px 0 5px;";
    this._blendModeGrid = createPillGrid(
      [{ label: "Mask", value: "mask" }, { label: "Whole crop", value: "whole_crop" }],
      2, (v) => { this.params.blend_mode = v; this._draw(); },
      { activeValue: this.params.blend_mode || "mask" },
    );
    secSeam.content.append(this.el.blendSlider.el, this.el.growSlider.el,
      bmLabel, this._blendModeGrid.el);
    sidebar.appendChild(secSeam.el);

    // View
    const secView = createPanel("Mask overlay");
    this._visBtn = createButton("Toggle mask (H)", { variant: "full", iconSrc: ICONS.eraser, onClick: () => this._toggleMaskVisible() });
    this.el.opacitySlider = createSliderRow("Opacity", 10, 100, Math.round(this.maskOpacity * 100), () => {
      this.maskOpacity = (parseInt(this.el.opacitySlider.numInput.value) || 50) / 100;
      this._draw();
    });
    secView.content.append(this._visBtn, this.el.opacitySlider.el);
    // 预览色调色块（仅显示；选橙色时裁剪框改画白色）
    const swatchRow = document.createElement("div");
    swatchRow.style.cssText = "display:flex;gap:8px;align-items:center;margin-top:8px;";
    const swatchLabel = document.createElement("span");
    swatchLabel.textContent = "Color";
    swatchLabel.style.cssText = "font-size:11px;color:#aaa;margin-right:2px;";
    swatchRow.appendChild(swatchLabel);
    this._colorSwatches = [];
    for (const [name, hex] of Object.entries(INPAINT_PREVIEW_COLORS)) {
      const dot = document.createElement("span");
      dot.title = name;
      dot.style.cssText = `width:20px;height:20px;border-radius:50%;background:${hex};cursor:pointer;box-sizing:border-box;border:2px solid ${this.previewColor === hex ? "#fff" : "transparent"};`;
      dot.addEventListener("click", () => this._setPreviewColor(name, hex));
      this._colorSwatches.push({ dot, hex });
      swatchRow.appendChild(dot);
    }
    secView.content.appendChild(swatchRow);
    sidebar.appendChild(secView.el);

    // Context margin（镜像节点 context_px，实时更新预览）
    const secCtx = createPanel("Context margin");
    const ctxStart = this.params.context_px != null ? this.params.context_px : 24;
    this.el.ctxSlider = createSliderRow("Pixels", 0, 512, ctxStart, () => {
      this.params.context_px = parseInt(this.el.ctxSlider.numInput.value) || 0;
      this._recomputeRegion();
      this._draw();
    });
    secCtx.content.appendChild(this.el.ctxSlider.el);
    sidebar.appendChild(secCtx.el);

    // Crop size（镜像节点旋钮；实时预览）
    const secCrop = createPanel("Crop size");
    this._sizeModeGrid = createPillGrid(
      [{ label: "Keep shape", value: "keep" }, { label: "Force square", value: "force" }, { label: "Free", value: "free" }],
      3, (v) => { this.params.size_mode = v; this._recomputeRegion(); this._draw(); },
      { activeValue: this.params.size_mode || "keep" },
    );
    this.el.targetSlider = createSliderRow("Target", 64, 8192, this.params.target ?? 1024, () => {
      const m = this.params.multiple || 8;
      const raw = parseInt(this.el.targetSlider.numInput.value) || 1024;
      const snapped = Math.max(m, Math.round(raw / m) * m);   // 落到 multiple 上
      this.params.target = snapped;
      if (snapped !== raw) this.el.targetSlider.setValue(snapped);
      this._recomputeRegion(); this._draw();
    });
    this._multipleGrid = createPillGrid(
      [{ label: "8", value: 8 }, { label: "16", value: 16 }, { label: "32", value: 32 }, { label: "64", value: 64 }],
      4, (v) => {
        this.params.multiple = v;
        const snapped = Math.max(v, Math.round((this.params.target || 1024) / v) * v);
        this.params.target = snapped;
        this.el.targetSlider?.setValue(snapped);   // target 重新对齐到新 multiple
        this._recomputeRegion(); this._draw();
      },
      { activeValue: this.params.multiple || 8 },
    );
    secCrop.content.append(this._sizeModeGrid.el, this.el.targetSlider.el, this._multipleGrid.el);
    sidebar.appendChild(secCrop.el);

    // 重置全部设置到默认
    const resetAll = createButton("Reset all to default", {
      variant: "standard", iconSrc: ICONS.reset, onClick: () => this._resetAll(),
    });
    resetAll.style.marginTop = "10px";
    sidebar.appendChild(resetAll);

    // Load Image
    const fileInput = document.createElement("input");
    fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
    fileInput.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f) {
        const reader = new FileReader();
        reader.onload = (ev) => { this._loadImageFromDataURL(ev.target.result); this.onLoadImage?.(); };
        reader.readAsDataURL(f);
      }
      fileInput.value = "";
    });
    const loadBtn = createButton("Load Image", { variant: "full", iconSrc: ICONS.image, onClick: () => fileInput.click() });
    loadBtn.style.marginTop = "10px";
    sidebar.append(loadBtn, fileInput);
  }

  _buildRightSidebar(sidebar, footer) {
    const sec = createPanel("Output crop");
    this._infoBlock = createInfo("Paint a mask to begin");
    sidebar.insertBefore(sec.el, footer);
    sec.content.appendChild(this._infoBlock.el);
  }

  _setTool(v) { this.tool = v; }

  _resetAll() {
    this._setTool("add");        // 回到 Brush（重置前可能停在 Erase）
    this.brushSize = DEFAULT_BRUSH_SIZE;
    this.maskOpacity = 0.5;
    this.params.blend = 16;
    this.params.mask_grow = 4;
    this.params.mask_blur = 4;
    this.params.blend_mode = "mask";
    this.params.context_px = 24;
    this.params.size_mode = "keep";
    this.params.target = 1024;
    this.params.multiple = 8;
    this.el.sizeSlider?.setValue(this.brushSize);
    this.el.opacitySlider?.setValue(50);
    this.el.blendSlider?.setValue(16);
    this.el.growSlider?.setValue(4);
    this.el.ctxSlider?.setValue(24);
    this.el.targetSlider?.setValue(1024);
    this._toolGrid?.setActive?.("add");
    this._blendModeGrid?.setActive?.("mask");
    this._sizeModeGrid?.setActive?.("keep");
    this._multipleGrid?.setActive?.(8);
    this._recomputeRegion();
    this._draw();
    if (this._lastCursorPos) this._drawCursor(this._lastCursorPos);
  }

  _setPreviewColor(name, hex) {
    this.previewColor = hex;
    this._cropBoxColor = (hex === INPAINT_PREVIEW_COLORS.Orange) ? "#ffffff" : null;
    for (const s of this._colorSwatches || [])
      s.dot.style.borderColor = (s.hex === hex) ? "#fff" : "transparent";
    this.onPreviewColor?.(name);
    this._draw();
  }

  _toggleMaskVisible() {
    this.maskVisible = !this.maskVisible;
    this._visBtn.classList.toggle("active", !this.maskVisible);
    this._draw();
  }

  _setStatus(msg) { this.layout?.setStatus(msg); }
}
