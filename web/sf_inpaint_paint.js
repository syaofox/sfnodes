// ============================================================
// SF Inpaint Crop — 遮罩绘制（画笔、橡皮、光标、撤销、快捷键）
// 移植自 comfyui-pixaroma js/inpaint_crop/paint.mjs
// 遮罩 canvas = 透明底上的白色 RGB + 逐像素 alpha（强度），红色叠加层经
// destination-in 着色，导出时在黑底上合成烘焙软边。自包含。
// ============================================================
import { InpaintCropEditor, BRAND } from "./sf_inpaint_core.js";

const proto = InpaintCropEditor.prototype;
const MAX_UNDO = 15;

proto._ensureMaskCanvas = function () {
  if (this._mask && this._mask.width === this.imgW && this._mask.height === this.imgH) return;
  const m = document.createElement("canvas");
  m.width = this.imgW; m.height = this.imgH;
  this._mask = m;
  this._mctx = m.getContext("2d");
  // 单笔缓冲：印章在此累积，烘焙到遮罩时带模糊软边（重叠印章不会叠出硬边）。
  const sc = document.createElement("canvas");
  sc.width = this.imgW; sc.height = this.imgH;
  this._stroke = sc;
  this._sctx = sc.getContext("2d");
  this._strokeHasContent = false;
  this._undo = []; this._redo = [];
  this.layout?.setUndoState({ canUndo: false, canRedo: false });
};

// ── pointer ──────────────────────────────────────────────────────────────
proto._displayPos = function (e) {
  const r = this.el.canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
};

proto._bindMouse = function (cvs) {
  // POINTER 事件（非 mouse）+ touch-action:none 让数位笔/触屏可绘制——否则浏览器
  // 吃掉笔/触控手势，且许多笔根本不发 mouse 事件，遮罩笔刷根本起不了笔。
  cvs.style.touchAction = "none";
  if (this.el.canvasWrap) this.el.canvasWrap.style.touchAction = "none";

  let winMove = null, winUp = null;
  const detach = () => {
    if (winMove) window.removeEventListener("pointermove", winMove), (winMove = null);
    if (winUp) {
      window.removeEventListener("pointerup", winUp);
      window.removeEventListener("pointercancel", winUp);
      winUp = null;
    }
  };
  // 暴露出来：中途关闭（pointerup 前编辑器被拆除）时解除 window 监听，而不是
  // 泄漏到页面整个生命周期。
  this._detachStroke = detach;

  let panMove = null, panUp = null;
  const detachPan = () => {
    if (panMove) window.removeEventListener("pointermove", panMove), (panMove = null);
    if (panUp) {
      window.removeEventListener("pointerup", panUp);
      window.removeEventListener("pointercancel", panUp);
      panUp = null;
    }
  };
  this._detachPan = detachPan;

  cvs.addEventListener("pointerdown", (e) => {
    // 平移缩放视图：中键，或 Space 按住 + 左键/笔尖。
    if (e.button === 1 || (e.button === 0 && this._spaceHeld)) {
      e.preventDefault();
      try { cvs.setPointerCapture(e.pointerId); } catch {}
      const x0 = e.clientX, y0 = e.clientY, px0 = this._panX, py0 = this._panY;
      this._panning = true;
      cvs.style.cursor = "grabbing";
      if (this._dispW) this.el.curCtx?.clearRect(0, 0, this._dispW, this._dispH);  // 平移时隐藏笔刷环
      this._lastCursorPos = null;
      detachPan();
      panMove = (ev) => {
        this._panX = px0 + (ev.clientX - x0);
        this._panY = py0 + (ev.clientY - y0);
        this._clampPan();
        this._requestRedraw();
      };
      panUp = () => { this._panning = false; detachPan(); cvs.style.cursor = this._spaceHeld ? "grab" : "none"; };
      window.addEventListener("pointermove", panMove);
      window.addEventListener("pointerup", panUp);
      window.addEventListener("pointercancel", panUp);
      return;
    }
    if (!this.img || e.button !== 0) return;
    e.preventDefault();
    try { cvs.setPointerCapture(e.pointerId); } catch {}
    this._beginStroke(e);
    detach();
    winMove = (ev) => {
      // coalesced 事件 = 快速笔画（尤其笔）的更高分辨率输入
      if (ev.getCoalescedEvents) {
        const co = ev.getCoalescedEvents();
        if (co.length > 1) { for (const ce of co) this._strokeMove(ce); return; }
      }
      this._strokeMove(ev);
    };
    winUp = () => { this._endStroke(); detach(); };
    window.addEventListener("pointermove", winMove);
    window.addEventListener("pointerup", winUp);
    window.addEventListener("pointercancel", winUp);
  });
  cvs.addEventListener("pointermove", (e) => { if (!this._painting && !this._panning && !this._spaceHeld) this._drawCursor(this._displayPos(e)); });
  cvs.addEventListener("pointerleave", () => { if (!this._painting && this._dispW) { this.el.curCtx?.clearRect(0, 0, this._dispW, this._dispH); this._lastCursorPos = null; } });
  // 滚轮朝光标缩放（笔刷大小改由 [ / ] 和 Size 滑块控制）
  cvs.addEventListener("wheel", (e) => {
    e.preventDefault();
    const p = this._displayPos(e);
    this._applyZoom(e.deltaY < 0 ? 1.15 : 1 / 1.15, p.x, p.y);
  }, { passive: false });
};

proto._effectiveTool = function () {
  // 按住 X 临时翻转到橡皮
  return this._xHeld ? (this.tool === "erase" ? "add" : "erase") : this.tool;
};

proto._beginStroke = function (e) {
  this._pushUndo();
  this._painting = true;
  this._lastPt = null;
  // 整笔锁定 add/erase 选择（拖拽中途松开 X 不影响本笔）
  this._strokeTool = this._effectiveTool();
  this._sctx.clearRect(0, 0, this.imgW, this.imgH);
  this._strokeHasContent = false;
  this._strokeMove(e);
};

proto._strokeMove = function (e) {
  const p = this._displayPos(e);
  const s = this._scale || 1;
  const sx = (p.x - this._panX) / s, sy = (p.y - this._panY) / s;   // pan/zoom -> 源 px
  if (this._lastPt) this._stampLine(this._lastPt.x, this._lastPt.y, sx, sy);
  else this._stampDab(sx, sy);
  this._lastPt = { x: sx, y: sy };
  this._requestRedraw();   // 合并（每帧一次重绘）——原来是每次 move 同步 _draw = 卡顿
  this._drawCursor(p);     // 光标保持即时（便宜，独立 canvas）
};

// 烘焙到笔画上的软边羽化宽度（源 px）。0 = 锐利。
proto._bakeBlurPx = function () {
  const rSrc = (this.brushSize / 2) / (this._scale || 1);
  return Math.round(this.softness * rSrc);
};

// 用锁定的工具把（模糊后的）笔画缓冲合成到目标 ctx
proto._compositeStroke = function (ctx) {
  const blur = this._bakeBlurPx();
  ctx.save();
  if (blur > 0) ctx.filter = `blur(${blur}px)`;
  ctx.globalCompositeOperation = this._strokeTool === "erase" ? "destination-out" : "source-over";
  ctx.drawImage(this._stroke, 0, 0);
  ctx.restore();
};

proto._endStroke = function () {
  if (this._drawRaf) { cancelAnimationFrame(this._drawRaf); this._drawRaf = null; }
  this._painting = false;
  this._lastPt = null;
  if (this._strokeHasContent) {
    this._compositeStroke(this._mctx);   // 烘焙进遮罩
    this._sctx.clearRect(0, 0, this.imgW, this.imgH);
    this._strokeHasContent = false;
  }
  this._rescanBBox();
  this._recomputeRegion();
  this._draw();
};

// 丢弃进行中的笔画缓冲，避免在随后的 mouseup 上被重新烘焙。
// 拖拽中途 undo/redo 时使用（否则撤销掉的笔画会在松开鼠标时重新合成）。
proto._abandonStroke = function () {
  if (this._sctx) this._sctx.clearRect(0, 0, this.imgW, this.imgH);
  this._strokeHasContent = false;
  this._lastPt = null;
};

// 当前应显示的遮罩（拖拽期间的实时笔画 + 遮罩）
proto._effectiveMaskCanvas = function () {
  if (!this._painting || !this._strokeHasContent) return this._mask;
  if (!this._effMask) this._effMask = document.createElement("canvas");
  const c = this._effMask;
  if (c.width !== this.imgW || c.height !== this.imgH) { c.width = this.imgW; c.height = this.imgH; }
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, this.imgW, this.imgH);
  ctx.drawImage(this._mask, 0, 0);
  this._compositeStroke(ctx);
  return c;
};

// ── 笔刷印章：笔画缓冲上画一个抗锯齿的清晰圆盘。软边是烘焙时的模糊
//    （_bakeBlurPx），重叠印章永不叠出硬边。
proto._stampDab = function (sx, sy) {
  const ctx = this._sctx;
  const r = Math.max(0.5, (this.brushSize / 2) / (this._scale || 1));
  const g = ctx.createRadialGradient(sx, sy, Math.max(0, r - 1.2), sx, sy, r);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  // "lighten" = max 混合，单笔内重叠印章不会叠加 alpha（快速笔画不会变暗）；
  // 软边来自笔末的烘焙模糊。
  ctx.globalCompositeOperation = "lighten";
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fill();
  ctx.globalCompositeOperation = "source-over";
  this._strokeHasContent = true;
};

proto._stampLine = function (x0, y0, x1, y1) {
  const r = Math.max(0.5, (this.brushSize / 2) / (this._scale || 1));
  const dx = x1 - x0, dy = y1 - y0;
  const dist = Math.hypot(dx, dy);
  const step = Math.max(1, r * 0.12);  // 足够密，快速缩小视图的笔画也不出空隙
  const n = Math.ceil(dist / step);
  for (let i = 0; i <= n; i++) {
    const t = n === 0 ? 0 : i / n;
    this._stampDab(x0 + dx * t, y0 + dy * t);
  }
};

// ── 光标环 ────────────────────────────────────────────────────────────
proto._drawCursor = function (p) {
  const ctx = this.el.curCtx;
  if (!ctx || !this._dispW) return;   // canvas 尚未适配（无图）
  this._lastCursorPos = p;
  ctx.clearRect(0, 0, this._dispW, this._dispH);
  const r = this.brushSize / 2;
  const erase = this._effectiveTool() === "erase";
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = erase ? "#ffffff" : BRAND;
  if (erase) ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = erase ? "#ffffff" : BRAND;
  ctx.beginPath(); ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2); ctx.fill();
};

// ── 遮罩操作 ────────────────────────────────────────────────────────────────
proto._clearMask = function () {
  if (!this._mask) return;
  this._pushUndo();
  this._mctx.clearRect(0, 0, this.imgW, this.imgH);
  this._endStroke();
};

proto._invertMask = function () {
  if (!this._mask) return;
  this._pushUndo();
  const id = this._mctx.getImageData(0, 0, this.imgW, this.imgH);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) { d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = 255 - d[i + 3]; }
  this._mctx.putImageData(id, 0, 0);
  this._endStroke();
};

proto._loadMaskFromURL = function (url) {
  const token = this._loadToken;   // 与请求它的图像加载代次绑定
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    // 有更新的 open/load 覆盖了它，或用户已开始绘制（画着 / 有撤销历史 /
    // 有画的 bbox）——不要覆盖新笔画。
    if (token !== this._loadToken) return;
    if (this._painting || this._undo?.length || this._bbox) {
      this._setStatus("保留了你的新笔画——保存的遮罩未重载。");
      return;
    }
    this._ensureMaskCanvas();
    const tmp = document.createElement("canvas");
    tmp.width = this.imgW; tmp.height = this.imgH;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(img, 0, 0, this.imgW, this.imgH);
    const id = tctx.getImageData(0, 0, this.imgW, this.imgH);
    const d = id.data;
    // 灰度（白 = 遮罩）-> 白色 RGB + alpha = 亮度
    for (let i = 0; i < d.length; i += 4) {
      const a = d[i]; // grayscale, r==g==b
      d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = a;
    }
    this._mctx.putImageData(id, 0, 0);
    this._undo = []; this._redo = [];
    this.layout?.setUndoState({ canUndo: false, canRedo: false });
    this._rescanBBox();
    this._recomputeRegion();
    this._draw();
  };
  img.onerror = () => {};
  img.src = url;
};

// 遮罩在黑底上合成 -> 灰度 dataURL（白 = 遮罩）
proto._exportMaskDataURL = function () {
  const out = document.createElement("canvas");
  out.width = this.imgW; out.height = this.imgH;
  const o = out.getContext("2d");
  o.fillStyle = "#000"; o.fillRect(0, 0, this.imgW, this.imgH);
  // 包含任何进行中的笔画，拖拽中 Ctrl+S 保存的是屏幕所见
  const m = (this._effectiveMaskCanvas && this._effectiveMaskCanvas()) || this._mask;
  if (m) o.drawImage(m, 0, 0);
  return out.toDataURL("image/png");
};

// ── 撤销 / 重做 ──────────────────────────────────────────────────────────
proto._pushUndo = function () {
  if (!this._mask) return;
  try { this._undo.push(this._mctx.getImageData(0, 0, this.imgW, this.imgH)); } catch (e) { return; }
  if (this._undo.length > MAX_UNDO) this._undo.shift();
  this._redo = [];
  this.layout?.setUndoState({ canUndo: this._undo.length > 0, canRedo: false });
};

proto._doUndo = function () {
  if (!this._mask || !this._undo.length) return;
  this._redo.push(this._mctx.getImageData(0, 0, this.imgW, this.imgH));
  this._mctx.putImageData(this._undo.pop(), 0, 0);
  this._abandonStroke();
  this._rescanBBox(); this._recomputeRegion(); this._draw();
  this.layout?.setUndoState({ canUndo: this._undo.length > 0, canRedo: this._redo.length > 0 });
};

proto._doRedo = function () {
  if (!this._mask || !this._redo.length) return;
  this._undo.push(this._mctx.getImageData(0, 0, this.imgW, this.imgH));
  this._mctx.putImageData(this._redo.pop(), 0, 0);
  this._abandonStroke();
  this._rescanBBox(); this._recomputeRegion(); this._draw();
  this.layout?.setUndoState({ canUndo: this._undo.length > 0, canRedo: this._redo.length > 0 });
};

// ── 快捷键 ────────────────────────────────────────────────────────────────────
proto._bindKeys = function () {
  this._keyHandler = (e) => {
    const ae = document.activeElement;
    if ((ae?.tagName === "INPUT" || ae?.tagName === "TEXTAREA" || ae?.tagName === "SELECT") && !ae?.dataset?.sfPaintTrap) return;
    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    if (key === "escape") { e.preventDefault(); e.stopImmediatePropagation(); this._close(); return; }
    if (ctrl && key === "s") { e.preventDefault(); this._save(); return; }
    if (ctrl && key === "z" && !e.shiftKey) { e.preventDefault(); this._doUndo(); return; }
    if (ctrl && (key === "y" || (key === "z" && e.shiftKey))) { e.preventDefault(); this._doRedo(); return; }
    if (ctrl) return;
    if (e.code === "Space") {   // 按住 Space 平移缩放视图（拖动）
      e.preventDefault();
      if (!this._spaceHeld) {
        this._spaceHeld = true;
        if (this.el.canvas) this.el.canvas.style.cursor = "grab";
        if (this._dispW) this.el.curCtx?.clearRect(0, 0, this._dispW, this._dispH);  // 隐藏笔刷环
        this._lastCursorPos = null;
      }
      return;
    }
    if (key === "b") { e.preventDefault(); this._setTool("add"); this._toolGrid?.setActive?.("add"); return; }
    if (key === "e") { e.preventDefault(); this._setTool("erase"); this._toolGrid?.setActive?.("erase"); return; }
    if (key === "h") { e.preventDefault(); this._toggleMaskVisible(); return; }
    if (key === "x") { this._xHeld = true; return; }
    if (key === "[" || key === "]") {
      e.preventDefault();
      const dir = key === "[" ? -1 : 1;
      // 忽略系统自动重复（卡顿）——按住时自己驱动平滑加速
      if (!e.repeat) { this._adjustBrush(dir * 4); this._startBrushHold(dir); }
      return;
    }
  };
  this._keyUpHandler = (e) => {
    const k = e.key.toLowerCase();
    if (k === "x") this._xHeld = false;
    if (k === "[" || k === "]") this._stopBrushHold();
    if (e.code === "Space") { this._spaceHeld = false; if (this.el.canvas && !this._panning) this.el.canvas.style.cursor = "none"; }
  };
  window.addEventListener("keydown", this._keyHandler, { capture: true });
  window.addEventListener("keyup", this._keyUpHandler, { capture: true });
};

// 笔刷缩放：点按即时，按住平滑加速（无 OS 重复延迟）
proto._adjustBrush = function (delta) {
  this.brushSize = Math.max(2, Math.min(300, this.brushSize + delta));
  this.el.sizeSlider?.setValue(this.brushSize);
  if (this._lastCursorPos) this._drawCursor(this._lastCursorPos);
};

proto._startBrushHold = function (dir) {
  this._stopBrushHold();
  this._holdDir = dir;
  let rate = 0.5, accum = 0;
  const tick = () => {
    if (!this.el.overlay?.isConnected) { this._stopBrushHold(); return; }
    rate = Math.min(3.4, rate + 0.12);
    accum += this._holdDir * rate;
    const step = Math.trunc(accum);
    if (step !== 0) { accum -= step; this._adjustBrush(step); }
    this._holdRaf = requestAnimationFrame(tick);
  };
  this._holdRaf = requestAnimationFrame(tick);
};

proto._stopBrushHold = function () {
  if (this._holdRaf) { cancelAnimationFrame(this._holdRaf); this._holdRaf = null; }
};

// ── 把源图粘贴/拖进打开的编辑器 ──────────────────────────────────────────
// 把丢入/粘贴的图加载为新源（同 Load Image 按钮：_loadImageFromDataURL +
// onLoadImage 断开上游连线）。三个监听都在 window 捕获层，以 overlay 存活
// 为门，preventDefault + stop 防止 ComfyUI 在背后自建 Load Image 节点。
proto._bindDropPaste = function () {
  const alive = () => !!this.el.overlay?.isConnected;
  const loadFile = (file) => {
    if (!file || !file.type?.startsWith("image/")) return false;
    const reader = new FileReader();
    reader.onload = (ev) => { this._loadImageFromDataURL(ev.target.result); this.onLoadImage?.(); };
    reader.readAsDataURL(file);
    return true;
  };
  this._pasteHandler = (e) => {
    if (!alive()) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    const it = Array.from(e.clipboardData?.items || []).find((x) => x.type?.startsWith("image/"));
    if (!it) return;
    const blob = it.getAsFile();
    if (!blob) return;
    e.preventDefault(); e.stopImmediatePropagation();
    loadFile(blob);
  };
  this._dragOverHandler = (e) => {
    if (!alive()) return;
    if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); e.stopImmediatePropagation(); }
  };
  this._dropHandler = (e) => {
    if (!alive()) return;
    const file = e.dataTransfer?.files?.[0];
    if (!file || !file.type?.startsWith("image/")) return;
    e.preventDefault(); e.stopImmediatePropagation();
    loadFile(file);
  };
  window.addEventListener("paste", this._pasteHandler, true);
  window.addEventListener("dragover", this._dragOverHandler, true);
  window.addEventListener("drop", this._dropHandler, true);
};

proto._unbindDropPaste = function () {
  if (this._pasteHandler) window.removeEventListener("paste", this._pasteHandler, true);
  if (this._dragOverHandler) window.removeEventListener("dragover", this._dragOverHandler, true);
  if (this._dropHandler) window.removeEventListener("drop", this._dropHandler, true);
  this._pasteHandler = this._dragOverHandler = this._dropHandler = null;
};

proto._unbindKeys = function () {
  this._stopBrushHold();
  if (this._drawRaf) { cancelAnimationFrame(this._drawRaf); this._drawRaf = null; }
  this._detachStroke?.();   // 解除存活的笔画 window 监听（拖拽中关闭）
  this._detachPan?.();      // 解除存活的平移 window 监听
  this._unbindDropPaste();
  if (this._keyHandler) window.removeEventListener("keydown", this._keyHandler, { capture: true });
  if (this._keyUpHandler) window.removeEventListener("keyup", this._keyUpHandler, { capture: true });
};
