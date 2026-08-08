// ============================================================
// SF Inpaint Crop — canvas 渲染 + 区域预览 + 保存
// 移植自 comfyui-pixaroma js/inpaint_crop/render.mjs
// ============================================================
import { InpaintCropEditor, BRAND, InpaintAPI } from "./sf_inpaint_core.js";
import { computeRegion, maskBBoxFromImageData, growBBox, seamAlphaFromAlpha } from "./sf_inpaint_geometry.js";

const proto = InpaintCropEditor.prototype;

// ── 图像加载 ──────────────────────────────────────────────────────────────
proto._loadImageFromURL = function (url, onDone) {
  const token = (this._loadToken = (this._loadToken || 0) + 1);   // 最新加载胜出
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    if (token !== this._loadToken) return;   // 有更新的 open/load 覆盖了它
    this.img = img;
    this.imgW = img.naturalWidth;
    this.imgH = img.naturalHeight;
    this._ensureMaskCanvas();
    this._fitCanvas();
    this._rescanBBox();
    this._recomputeRegion();
    this._draw();
    this._setStatus(`已加载: ${this.imgW}×${this.imgH}`);
    onDone?.();
  };
  img.onerror = () => this._setStatus("源图加载失败。");
  img.src = url;
};

proto._loadImageFromDataURL = function (dataURL) {
  const token = (this._loadToken = (this._loadToken || 0) + 1);   // 最新加载胜出
  const img = new Image();
  img.onload = () => {
    if (token !== this._loadToken) return;   // 有更新的 open/load 覆盖了它
    this.img = img;
    this.imgW = img.naturalWidth;
    this.imgH = img.naturalHeight;
    this._pendingSrcDataURL = dataURL;
    this._maskPath = "";
    this._mask = null;
    this._ensureMaskCanvas();
    this._fitCanvas();
    this._rescanBBox();
    this._recomputeRegion();
    this._draw();
    this._setStatus(`已加载: ${this.imgW}×${this.imgH}`);
  };
  img.src = dataURL;
};

// ── fit + draw ────────────────────────────────────────────────────────────
proto._fitCanvas = function () {
  if (!this.img) return;
  const ws = this.el.workspace, pad = 40;
  const maxW = ws.clientWidth - pad * 2, maxH = ws.clientHeight - pad * 2;
  if (maxW <= 0 || maxH <= 0) return;
  const asp = this.imgW / this.imgH;
  let dw, dh;
  if (maxW / maxH > asp) { dh = maxH; dw = dh * asp; } else { dw = maxW; dh = dw / asp; }
  this._baseScale = dw / this.imgW;
  this._zoom = 1; this._panX = 0; this._panY = 0;   // （重新）适配重置视图
  this._scale = this._baseScale;
  this._dispW = Math.round(dw); this._dispH = Math.round(dh);
  // 后备存储按 devicePixelRatio，高分屏上绘制 canvas 保持清晰；所有绘制都
  // 在逻辑 px（ctx 被 dpr 缩放）。
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  for (const c of [this.el.canvas, this.el.cursor]) {
    c.width = Math.round(this._dispW * dpr); c.height = Math.round(this._dispH * dpr);
    c.style.width = this._dispW + "px"; c.style.height = this._dispH + "px";
  }
  this.el.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  this.el.curCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  this.el.canvas.style.cursor = "none";
};

// 保持图像铺满视口（不留空隙）；zoom 1 时强制 pan 0
proto._clampPan = function () {
  const iw = this.imgW * this._scale, ih = this.imgH * this._scale;
  const minX = Math.min(0, this._dispW - iw), minY = Math.min(0, this._dispH - ih);
  this._panX = Math.min(0, Math.max(minX, this._panX));
  this._panY = Math.min(0, Math.max(minY, this._panY));
};

// 按 `factor` 缩放，保持 (ancX, ancY 显示 px) 下的源点不动
proto._applyZoom = function (factor, ancX, ancY) {
  const nz = Math.max(1, Math.min(8, this._zoom * factor));
  if (nz === this._zoom) return;
  const sx = (ancX - this._panX) / this._scale;   // 光标下的源点
  const sy = (ancY - this._panY) / this._scale;
  this._zoom = nz;
  this._scale = this._baseScale * nz;
  this._panX = ancX - sx * this._scale;
  this._panY = ancY - sy * this._scale;
  this._clampPan();
  this._draw();
  if (this._lastCursorPos) this._drawCursor(this._lastCursorPos);
};

// 显示分辨率的羽化接缝 alpha：遮罩在 (blend * scale) px 上的向外有符号距离
// smoothstep——Python _blur_alpha 的 scipy 路径的 canvas 镜像，编辑器预览
// 因此与真实缝合接缝一致（预览 == 结果）。
// 在缩小缓冲上计算（接缝是软的，chamfer DT + 放大不可见）以保持每次 _draw
// 都快——笔刷笔画 + softness 拖动。空闲鼠标移动不触发 _draw（光标单独绘制），
// 无需缓存。
proto._seamAlphaCanvas = function () {
  const dpr = Math.max(1, window.devicePixelRatio || 1);   // DPR 后备，HiDPI 清晰
  const W = Math.round(this._dispW * dpr), H = Math.round(this._dispH * dpr);
  if (!this._seamCv) this._seamCv = document.createElement("canvas");
  const c = this._seamCv;
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const ctx = c.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const src = this._effectiveMaskCanvas();
  if (!src) return c;

  // WHOLE CROP 模式：缝合淡化整个裁剪**矩形**（Python _feather_alpha）——
  // 从区域边缘向内的线性斜坡，不是遮罩边缘。展示它而非遮罩羽化，预览与
  // 结果一致。
  if (this.params.blend_mode === "whole_crop" && this._region) {
    // _baseScale（不是 _scale）：seam canvas 先适配再按缩放视图矩形绘制，
    // 缩放随后应用——用 _scale 会双重计数。
    const blendD = Math.max(0, (this.params.blend ?? 16) * (this._baseScale || 1) * dpr);
    const CAP = 480;
    const sc = Math.min(1, CAP / Math.max(W, H));
    const bw = Math.max(1, Math.round(W * sc)), bh = Math.max(1, Math.round(H * sc));
    const r = this._region;
    const rx = r.rx / this.imgW * bw, ry = r.ry / this.imgH * bh;
    const rw = Math.max(1, r.rw / this.imgW * bw), rh = Math.max(1, r.rh / this.imgH * bh);
    const kBuf = blendD * (bw / W);
    const kEff = Math.min(kBuf, Math.max(0.5, Math.min(rw, rh) / 2 - 0.5));  // 内部保持不透明
    const b = this._seamBuf || (this._seamBuf = document.createElement("canvas"));
    if (b.width !== bw || b.height !== bh) { b.width = bw; b.height = bh; }
    const bctx = b.getContext("2d");
    bctx.setTransform(1, 0, 0, 1, 0, 0);
    const id = bctx.createImageData(bw, bh);
    const d = id.data;
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        let a = 0;
        if (x >= rx && x < rx + rw && y >= ry && y < ry + rh) {
          const dist = Math.min(x - rx, rx + rw - 1 - x, y - ry, ry + rh - 1 - y);
          a = kEff <= 0 ? 1 : Math.max(0, Math.min(1, dist / kEff));
        }
        const p = (y * bw + x) * 4;
        d[p] = 255; d[p + 1] = 255; d[p + 2] = 255; d[p + 3] = Math.round(a * 255);
      }
    }
    bctx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(b, 0, 0, bw, bh, 0, 0, W, H);
    return c;
  }

  // _baseScale（不是 _scale）：适配分辨率的 seam canvas，缩放随后在视图
  // 绘制时应用。
  const blendDisp = Math.max(0, (this.params.blend ?? 16) * (this._baseScale || 1) * dpr);
  if (blendDisp < 0.5) { ctx.drawImage(src, 0, 0, W, H); return c; }   // 锐利接缝

  // 把遮罩画进小缓冲、对 alpha 做距离变换、写回羽化 alpha，再放大到显示
  // 分辨率的 seam canvas。
  const CAP = 480;
  const sc = Math.min(1, CAP / Math.max(W, H));
  const bw = Math.max(1, Math.round(W * sc)), bh = Math.max(1, Math.round(H * sc));
  const kBuf = blendDisp * (bw / W);          // 缓冲 px 的羽化宽度
  const b = this._seamBuf || (this._seamBuf = document.createElement("canvas"));
  if (b.width !== bw || b.height !== bh) { b.width = bw; b.height = bh; }
  const bctx = b.getContext("2d");
  bctx.setTransform(1, 0, 0, 1, 0, 0);
  bctx.clearRect(0, 0, bw, bh);
  bctx.drawImage(src, 0, 0, bw, bh);
  const id = bctx.getImageData(0, 0, bw, bh);
  const alpha = seamAlphaFromAlpha(id.data, bw, bh, kBuf);
  const dpx = id.data;
  for (let i = 0, p = 0; i < alpha.length; i++, p += 4) {
    dpx[p] = 255; dpx[p + 1] = 255; dpx[p + 2] = 255;
    dpx[p + 3] = Math.round(alpha[i] * 255);
  }
  bctx.putImageData(id, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(b, 0, 0, bw, bh, 0, 0, W, H);   // 放大软 alpha
  return c;
};

// 合并高频重绘（笔刷笔画）到每动画帧一次，快速鼠标移动不堆积同步全量重绘
// （消除卡顿）。
proto._requestRedraw = function () {
  if (this._drawRaf) return;
  this._drawRaf = requestAnimationFrame(() => { this._drawRaf = null; this._draw(); });
};

proto._draw = function () {
  if (!this.img || !this._dispW) return;  // _fitCanvas 在任何绘制前设置 _dispW
  const ctx = this.el.ctx, s = this._scale;
  const W = this._dispW, H = this._dispH;
  ctx.clearRect(0, 0, W, H);
  // pan/zoom：图像按 _scale（= base*zoom）绘制在 (_panX,_panY)；
  // 适配（zoom 1）时为 (0,0,W,H)。以下所有内容按同一 pan 偏移。
  ctx.drawImage(this.img, this._panX, this._panY, this.imgW * s, this.imgH * s);

  // 接缝预览：用所选颜色给羽化接缝 alpha（Softness）染色，裁剪区域裁剪。
  // DPR 后备（HiDPI 清晰）。镜像 Python _blur_alpha 的 scipy smoothstep，
  // 色调与真实缝合接缝一致。
  if (this.maskVisible && this._mask) {
    if (!this._tint) this._tint = document.createElement("canvas");
    const t = this._tint;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const tw = Math.round(W * dpr), th = Math.round(H * dpr);
    if (t.width !== tw || t.height !== th) { t.width = tw; t.height = th; }
    const tc = t.getContext("2d");
    tc.setTransform(1, 0, 0, 1, 0, 0);
    tc.clearRect(0, 0, tw, th);
    // 始终显示羽化接缝预览（哪怕绘制中途），遮罩不会"画时锐利、松开变糊"。
    // rAF 合并（_requestRedraw）保证流畅；超大图在笔画中变卡时，应节流
    // _seamAlphaCanvas 而不是跳过它。
    const alphaSrc = this._seamAlphaCanvas();
    // 遮罩按与图像相同的 pan/zoom 矩形绘制（后备 px），保证对齐
    tc.drawImage(alphaSrc, this._panX * dpr, this._panY * dpr, this.imgW * s * dpr, this.imgH * s * dpr);
    tc.globalCompositeOperation = "source-in";
    tc.fillStyle = this.previewColor || "#f6303a";
    tc.fillRect(0, 0, tw, th);
    tc.globalCompositeOperation = "source-over";
    ctx.save();
    // 把接缝色调裁剪到裁剪框内——但绘制**中途**不裁：裁剪框只在笔画结束时
    // 长大，中途裁剪会藏起画在现有框外的遮罩（"被上下文边距打断"直到松开）。
    // 笔画期间全图显示遮罩，松手后裁剪接缝回归。
    if (this._region && !this._painting) {
      const r = this._region;
      ctx.beginPath();
      ctx.rect(r.rx * s + this._panX, r.ry * s + this._panY, r.rw * s, r.rh * s);
      ctx.clip();
    }
    ctx.globalAlpha = this.maskOpacity;
    ctx.drawImage(t, 0, 0, W, H);             // 逻辑尺寸的后备分辨率色调 = 清晰
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  // 画过的紧致 bbox（白色虚线）
  if (this._bbox) {
    const [x0, y0, x1, y1] = this._bbox;
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.strokeRect(x0 * s + this._panX + 0.5, y0 * s + this._panY + 0.5, (x1 - x0) * s, (y1 - y0) * s);
    ctx.setLineDash([]);
  }

  // 裁剪区域（橙色虚线）+ 手柄 + 尺寸徽标
  if (this._region) {
    const r = this._region;
    const rx = r.rx * s + this._panX, ry = r.ry * s + this._panY, rw = r.rw * s, rh = r.rh * s;
    const boxColor = this._cropBoxColor || BRAND;   // 橙色色调激活时用白色
    ctx.strokeStyle = boxColor; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
    ctx.fillStyle = boxColor;
    for (const [hx, hy] of [[rx, ry], [rx + rw, ry], [rx, ry + rh], [rx + rw, ry + rh]])
      ctx.fillRect(hx - 4, hy - 4, 8, 8);
    const label = `${r.out_w} × ${r.out_h}`;
    ctx.font = "bold 12px 'Segoe UI', sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const tw = ctx.measureText(label).width + 12;
    const by = Math.max(10, ry - 11);
    ctx.fillStyle = BRAND;
    if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(rx, by - 9, tw, 18, 3); ctx.fill(); }
    else ctx.fillRect(rx, by - 9, tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, rx + 6, by);
  }
};

// ── 几何预览 ──────────────────────────────────────────────────────────────
// 扫描遮罩找画过的外接框。这是昂贵部分（全图 getImageData + 逐像素扫描），
// 只在遮罩实际变化时跑（笔画结束 / 清空 / 反选 / 撤销 / 加载），不是每次
// 上下文滑块移动都跑。
proto._rescanBBox = function () {
  if (!this.img || !this._mask) { this._bbox = null; return; }
  const id = this._mctx.getImageData(0, 0, this.imgW, this.imgH);
  this._bbox = maskBBoxFromImageData(id.data, this.imgW, this.imgH);
};

// 从**缓存的** bbox + 当前旋钮计算裁剪区域。便宜，每次上下文滑块移动都可
// 安全调用，无需重扫遮罩。
proto._recomputeRegion = function () {
  if (!this.img) { this._region = null; this._bbox = null; return; }
  const grow = this.params.mask_grow != null ? this.params.mask_grow : 0;
  const grown = growBBox(this._bbox, grow, this.imgW, this.imgH);
  // force 模式是 `target` 的方形（节点只有一个 Target 旋钮），镜像 Python
  // _params：target_w = target_h = target，否则预览会错误地显示默认 1024。
  const p = { ...this.params, target_w: this.params.target, target_h: this.params.target };
  this._region = computeRegion(grown, this.imgW, this.imgH, p);
  this._updateInfo(this._bbox);
};

proto._updateInfo = function (bbox) {
  if (!this._infoBlock) return;
  if (!bbox) { this._infoBlock.setHTML("Paint a mask to begin.<br>The whole image will be used until you do."); return; }
  const r = this._region;
  this._infoBlock.setHTML(
    `<b>Crop out:</b> ${r.out_w} × ${r.out_h}<br>` +
    `<b>Region:</b> ${r.rw} × ${r.rh}px<br>` +
    `<b>Context:</b> ${this.params.context_px ?? 0}px`,
  );
};

// ── 保存 ────────────────────────────────────────────────────────────────────
proto._buildPreview = function () {
  if (!this.img || !this._region) return null;
  const r = this._region;
  const c = document.createElement("canvas");
  c.width = r.out_w; c.height = r.out_h;
  c.getContext("2d").drawImage(this.img, r.rx, r.ry, r.rw, r.rh, 0, 0, r.out_w, r.out_h);
  return c.toDataURL("image/png");
};

proto._save = async function () {
  if (!this.img) { this._setStatus("没有可保存的图"); return; }
  this.layout.setSaving();
  try {
    if (this._pendingSrcDataURL) {
      try {
        const d = await InpaintAPI.uploadSrc(this.projectId, this._pendingSrcDataURL);
        this._srcPath = d.path || this._srcPath;
      } catch (e) { console.warn("[InpaintCrop] src upload failed:", e); }
      this._pendingSrcDataURL = null;
    }
    try {
      const d = await InpaintAPI.saveMask(this.projectId, this._exportMaskDataURL());
      this._maskPath = d.path || this._maskPath;
    } catch (e) { console.warn("[InpaintCrop] mask save failed:", e); }

    const state = {
      project_id: this.projectId,
      src_path: this._srcPath,
      mask_path: this._maskPath,
      doc_w: this.imgW,
      doc_h: this.imgH,
      blend_mode: this.params.blend_mode || "mask",
    };
    const extra = {
      context_px: this.params.context_px,
      mask_grow: this.params.mask_grow,
      mask_blur: this.params.mask_blur,
      softness: this.params.blend,
      size_mode: this.params.size_mode,
      target: this.params.target,
      multiple: this.params.multiple,
      blend_mode: this.params.blend_mode,   // 把编辑器胶囊镜像回节点 widget
    };
    const preview = this._buildPreview();
    if (this.onSave) this.onSave(JSON.stringify(state), extra, preview);
    if (this._diskSavePending) {
      this._diskSavePending = false;
      if (this.onSaveToDisk && preview) this.onSaveToDisk(preview);
    }
    this.layout.setSaved();
  } catch (err) {
    console.error("[InpaintCrop] Save error:", err);
    this.layout.setSaveError("Save failed: " + err.message);
  }
};
