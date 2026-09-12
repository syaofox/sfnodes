// ==========================================================================
// sf_universal_slider.js - SF Universal Slider 前端（复刻孤海万能滑条）
// ==========================================================================
//
// 复刻 goohai_universal_slider.js 的 Canvas 大滑条交互（居中标签+数值、
// 轨道/填充/旋钮绘制、点击跳转+拖拽、document mousemove/mouseup 保底、
// 右键设置面板、节点标题重绘），差异（已确认范围）：
// - 去掉全局 LGraphCanvas.prototype.drawNode 圆角补丁（污染所有节点），
//   仅保留本节点级 onDrawForeground 标题重绘（用户确认保留）
// - import 改绝对路径 "/scripts/app.js"（check_web_imports.py B2）
// - CSS 前缀 ghs- → sf-us-，样式经 sf_common.injectCSSOnce 注入
// - 设置弹窗关闭经 sf_popup.attachPopupDismiss + clampToViewport
// - 输出槽名动态 int/float（静态 RETURN_NAMES value + 前端改名，
//   any_pack.setSlotType 元素替换，patterns §41 同款；用户已确认）
// - 值数学（pct/clamp/snap/calcValue/归一化）收敛 lib 纯模块
//
// 后端权威：nodes/utils/universal_slider.py::SFUniversalSlider
// ==========================================================================

import { app } from "/scripts/app.js";
import { el, injectCSSOnce } from "./sf_common.js";
import { attachPopupDismiss, clampToViewport } from "./sf_popup.js";
import { setSlotType } from "./any_pack.js";
import {
  DEFAULTS,
  calcValue,
  clamp,
  fmtVal,
  outputSlotForType,
  normalizeSliderSettings,
  pct,
} from "./sf_universal_slider_lib.js";

const CLASS = "SFUniversalSlider";
const CSS_ID = "sf-us-css";
const WIDGET_NAME = "value";
const OUTPUT_TYPE_NAME = "output_type";

function injectCSS() {
  injectCSSOnce(
    CSS_ID,
    `
.sf-us-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);backdrop-filter:blur(3px);z-index:100000;display:flex;justify-content:center;align-items:center}
.sf-us-panel{background:#1c1c1e;border:1px solid #333;border-radius:14px;padding:28px 32px;min-width:380px;box-shadow:0 24px 80px rgba(0,0,0,.6);font-family:'Segoe UI',system-ui,-apple-system,sans-serif;--sf-us-c:#e8c547}
.sf-us-ptitle{font-size:16px;font-weight:700;color:#eee;margin-bottom:22px}
.sf-us-row{display:flex;align-items:center;margin-bottom:14px}
.sf-us-rlbl{width:72px;font-size:12.5px;color:#999;flex-shrink:0}
.sf-us-inp{flex:1;background:#2a2a2c;border:1px solid #3a3a3c;border-radius:8px;padding:8px 12px;color:#eee;font-size:13px;outline:none;font-family:inherit}
.sf-us-inp:focus{border-color:var(--sf-us-c)}
.sf-us-clr{width:48px;height:34px;padding:2px;border-radius:8px;border:1px solid #3a3a3c;background:#2a2a2c;cursor:pointer}
.sf-us-btns{display:flex;justify-content:flex-end;gap:10px;margin-top:22px}
.sf-us-btn{padding:8px 22px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;font-family:inherit}
.sf-us-bx{background:#2a2a2c;color:#aaa;border:1px solid #3a3a3c}
.sf-us-bx:hover{background:#333;color:#ccc}
.sf-us-bok{background:var(--sf-us-c);color:#111;font-weight:600}
.sf-us-bok:hover{filter:brightness(1.12)}
.sf-us-radio-wrap{flex:1;display:flex;gap:20px;align-items:center}
.sf-us-radio-label{display:flex;align-items:center;gap:6px;cursor:pointer;color:#eee;font-size:13px;padding:6px 12px;border-radius:6px;background:#2a2a2c;border:1px solid #3a3a3c}
.sf-us-radio-label:hover{border-color:var(--sf-us-c)}
.sf-us-radio-label input[type="radio"]{appearance:none;-webkit-appearance:none;width:16px;height:16px;border:2px solid #555;border-radius:50%;cursor:pointer;position:relative}
.sf-us-radio-label input[type="radio"]:checked{border-color:var(--sf-us-c)}
.sf-us-radio-label input[type="radio"]:checked::after{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:8px;height:8px;border-radius:50%;background:var(--sf-us-c)}
    `
  );
}

// ── Canvas 圆角矩形 ──
function rrect(ctx, x, y, w, h, r) {
  if (w < 0) w = 0;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function updateVis(node) {
  node.setDirtyCanvas(true, true);
}

function syncWidgetType(node) {
  const g = node._sfUS;
  if (!g || !g.widget) return;
  const p = node.properties;
  const isInt = p.sliderType === "int";
  let v = g.widget.value;
  v = clamp(v, p.sliderMin, p.sliderMax);
  v = isInt ? Math.round(v) : parseFloat(Number(v).toFixed(2));
  g.widget.value = v;
}

// 输出槽：类型 + 槽名动态 int/float（元素替换，Vue 渲染生效）
function syncOutputType(node) {
  const g = node._sfUS;
  const isInt = node.properties.sliderType === "int";
  if (g?.outputTypeWidget) g.outputTypeWidget.value = isInt ? "int" : "float";
  if (node.outputs?.length) {
    const slot = outputSlotForType(node.properties.sliderType);
    setSlotType(node, node.outputs, 0, slot.type, {
      name: slot.name,
      localized_name: slot.name,
    });
  }
}

function cleanupDrag(node) {
  const g = node._sfUS;
  if (!g) return;
  g._dragging = false;
  if (g._docCleanup) {
    g._docCleanup();
    g._docCleanup = null;
  }
}

// ── 设置弹窗 ──
function showSettings(node) {
  cleanupDrag(node);
  injectCSS();
  document.querySelectorAll(".sf-us-overlay").forEach((e) => e.remove());
  const p = node.properties;

  const ov = el("div", "sf-us-overlay");
  ov.setAttribute("tabindex", "-1");
  const pl = el("div", "sf-us-panel");
  pl.style.setProperty("--sf-us-c", p.sliderColor);
  pl.appendChild(el("div", "sf-us-ptitle", "SF 万能滑条 设置"));

  function addRow(labelText, input) {
    const r = el("div", "sf-us-row");
    r.appendChild(el("label", "sf-us-rlbl", labelText));
    r.appendChild(input);
    pl.appendChild(r);
  }
  function mkInp(type, value, attrs) {
    const i = el("input", "sf-us-inp");
    i.type = type;
    i.value = value;
    if (attrs) Object.entries(attrs).forEach(([k, v]) => i.setAttribute(k, v));
    return i;
  }

  const clrI = el("input", "sf-us-clr");
  clrI.type = "color";
  clrI.value = p.sliderColor;
  addRow("滑条颜色", clrI);

  const radioWrap = el("div", "sf-us-radio-wrap");
  let selectedType = p.sliderType;
  for (const opt of [
    { v: "float", t: "浮点 (Float)" },
    { v: "int", t: "整数 (Integer)" },
  ]) {
    const label = el("label", "sf-us-radio-label");
    const radio = el("input");
    radio.type = "radio";
    radio.name = "sf-us-slider-type";
    radio.value = opt.v;
    radio.checked = p.sliderType === opt.v;
    radio.addEventListener("change", () => {
      if (radio.checked) selectedType = opt.v;
    });
    label.appendChild(radio);
    label.appendChild(document.createTextNode(opt.t));
    radioWrap.appendChild(label);
  }
  addRow("类型", radioWrap);

  const minI = mkInp("number", p.sliderMin, { step: "any" });
  addRow("最小值", minI);
  const maxI = mkInp("number", p.sliderMax, { step: "any" });
  addRow("最大值", maxI);
  const stepI = mkInp("number", p.sliderStep, { step: "any", min: "0.0001" });
  addRow("步长", stepI);
  const lblI = mkInp("text", p.sliderLabel);
  addRow("显示名称", lblI);

  const btns = el("div", "sf-us-btns");
  const bCancel = el("button", "sf-us-btn sf-us-bx", "取消");
  // 必须调包装后的 ov.remove()（解绑监听 + 移出 DOM）；裸 detach() 只解绑监听，
  // 弹窗会留在页面上关不掉
  bCancel.onclick = () => ov.remove();
  const bOk = el("button", "sf-us-btn sf-us-bok", "确定");
  bOk.onclick = () => {
    const n = normalizeSliderSettings({
      type: selectedType,
      min: minI.value,
      max: maxI.value,
      step: stepI.value,
      label: lblI.value,
    });
    p.sliderType = n.type;
    p.sliderMin = n.min;
    p.sliderMax = n.max;
    p.sliderStep = n.step;
    p.sliderLabel = n.label;
    p.sliderColor = clrI.value;

    const w = node._sfUS.widget;
    if (w) w.value = calcValue(w.value, n.min, n.max, n.step, n.type === "int");
    syncWidgetType(node);
    syncOutputType(node);
    updateVis(node);
    ov.remove();
  };
  btns.append(bCancel, bOk);
  pl.appendChild(btns);
  ov.appendChild(pl);

  const detach = attachPopupDismiss(ov, { onClose: () => ov.remove() });
  const _remove = ov.remove.bind(ov);
  ov.remove = () => {
    detach();
    _remove();
  };
  ov.addEventListener("click", (e) => {
    if (e.target === ov) ov.remove();
  });
  ov.addEventListener("keydown", (e) => {
    if (e.key === "Escape") ov.remove();
    if (e.key === "Enter") bOk.click();
  });

  document.body.appendChild(ov);
  clampToViewport(pl);
  ov.focus();
}

// ── 节点装配 ──
function setupSlider(node) {
  if (!node.properties) node.properties = {};
  for (const [k, v] of Object.entries(DEFAULTS)) {
    if (node.properties[k] === undefined) node.properties[k] = v;
  }
  const p = node.properties;
  const isInt = p.sliderType === "int";

  // 隐藏 PY 自带 value 滑条（标准隐藏 widget，值仍走 prompt 收集）
  const dw = node.widgets ? node.widgets.find((w) => w.name === WIDGET_NAME) : null;
  if (dw) {
    dw.hidden = true;
    dw.computeSize = () => [0, 0];
  }

  // output_type 隐藏 widget（后端 hidden 声明，随 workflow 保存）
  let outputTypeWidget = node.widgets
    ? node.widgets.find((w) => w.name === OUTPUT_TYPE_NAME)
    : null;
  if (!outputTypeWidget && typeof node.addWidget === "function") {
    node.addWidget("combo", OUTPUT_TYPE_NAME, isInt ? "int" : "float", function () {}, {
      values: ["float", "int"],
    });
    outputTypeWidget = node.widgets ? node.widgets.find((w) => w.name === OUTPUT_TYPE_NAME) : null;
  }
  if (outputTypeWidget) {
    outputTypeWidget.value = isInt ? "int" : "float";
    outputTypeWidget.hidden = true;
    outputTypeWidget.computeSize = () => [0, 0];
    outputTypeWidget.draw = function () {};
    outputTypeWidget.mouse = function () {};
  }

  // 节点外观（原版配色）
  node.color = "#2D384D";
  node.bgcolor = "#2D384D";

  // 标题重绘（用户确认保留；节点级实现，不碰全局 drawNode）
  const origFG = node.onDrawForeground;
  node.onDrawForeground = function (ctx) {
    const th = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) || 30;
    const r = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_ROUND_RADIUS) || 8;
    const w = this.size[0];
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(r, -th);
    ctx.lineTo(w - r, -th);
    ctx.arcTo(w, -th, w, -th + r, r);
    ctx.lineTo(w, 2);
    ctx.lineTo(0, 2);
    ctx.lineTo(0, -th + r);
    ctx.arcTo(0, -th, r, -th, r);
    ctx.closePath();
    ctx.fillStyle = this.color || "#2D384D";
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.font = "20px 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif";
    ctx.fillStyle = "#E3E3E3";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(this.title || "", w / 2, -th + 10);
    ctx.restore();
    if (origFG) origFG.call(this, ctx);
  };

  // 绘制状态缓存
  const ds = { trackLeft: 14, trackW: 192 };
  node._sfUS = {
    widget: dw,
    outputTypeWidget,
    _dragging: false,
    _docCleanup: null,
    _ds: ds,
  };

  syncWidgetType(node);
  syncOutputType(node);

  // Canvas 自定义滑条（原版绘制 1:1，CSS 前缀除外）
  if (typeof node.addCustomWidget === "function") {
    node.addCustomWidget({
      name: "sf_us_ui",
      type: "sf_universal_slider",
      draw(ctx, n, W, y) {
        const g = n._sfUS;
        if (!g) return;
        const pp = n.properties;
        const v = g.widget ? g.widget.value : 0;
        const vIsInt = pp.sliderType === "int";
        const ratio = clamp(pct(v, pp.sliderMin, pp.sliderMax), 0, 100);
        const color = pp.sliderColor;

        const ml = 14;
        const mr = 24;
        ds.trackLeft = ml;
        ds.trackW = W - ml - mr;

        // 居中标签（名称 16px + 数值 20px bold）
        ctx.save();
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        const nameFont = "16px 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif";
        const valFont = "bold 24px 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif";
        ctx.font = nameFont;
        const nameText = pp.sliderLabel;
        const nameW = ctx.measureText(nameText).width;
        ctx.font = valFont;
        const valText = " " + fmtVal(v, vIsInt);
        const valW = ctx.measureText(valText).width;
        const lx = (W - nameW - valW) / 2;
        const ly = y + 6;
        ctx.shadowColor = "rgba(0,0,0,0.6)";
        ctx.shadowBlur = 3;
        ctx.shadowOffsetY = 1;
        ctx.font = nameFont;
        ctx.fillStyle = "#B2B7BD";
        ctx.fillText(nameText, lx, ly);
        ctx.font = valFont;
        ctx.fillStyle = color;
        ctx.fillText(valText, lx + nameW, ly);
        ctx.restore();

        // 轨道背景
        const trackY = y + 34;
        const trackH = 10;
        const trackR = 5;
        ctx.save();
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 2;
        ctx.shadowOffsetY = 1;
        rrect(ctx, ml, trackY, ds.trackW, trackH, trackR);
        ctx.fillStyle = "#1a1a1a";
        ctx.fill();
        ctx.restore();

        const fillW = (ds.trackW * ratio) / 100;
        if (fillW > 0) {
          ctx.save();
          ctx.globalAlpha = 0.2;
          ctx.shadowColor = color;
          ctx.shadowBlur = 10;
          rrect(ctx, ml, trackY + 2, fillW, trackH - 4, 5);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.restore();

          ctx.save();
          rrect(ctx, ml, trackY, fillW, trackH, trackR);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.restore();
        }

        // 旋钮
        const thumbX = ml + fillW;
        const thumbY = trackY + trackH / 2;
        ctx.save();
        if (g._dragging) {
          ctx.shadowColor = "rgba(232,197,71,0.12)";
          ctx.shadowBlur = 5;
        } else {
          ctx.shadowColor = "rgba(0,0,0,0.4)";
          ctx.shadowBlur = 4;
        }
        ctx.fillStyle = "#f5f0e8";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(thumbX, thumbY, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      },
      mouse(event, pos, n) {
        const g = n._sfUS;
        if (!g) return;
        const pp = n.properties;
        if (event.type === "mouseup" || event.type === "pointerup") {
          if (g._dragging) {
            g._dragging = false;
            if (g._docCleanup) {
              g._docCleanup();
              g._docCleanup = null;
            }
            n.setDirtyCanvas(true, true);
            return true;
          }
          return;
        }
        if ((event.type === "mousemove" || event.type === "pointermove") && g._dragging) {
          const vIsInt = pp.sliderType === "int";
          const ratio = clamp((pos[0] - ds.trackLeft) / ds.trackW, 0, 1);
          let v = pp.sliderMin + ratio * (pp.sliderMax - pp.sliderMin);
          v = calcValue(v, pp.sliderMin, pp.sliderMax, pp.sliderStep, vIsInt);
          if (g.widget) g.widget.value = v;
          n.setDirtyCanvas(true, false);
          return true;
        }
        if (event.type !== "mousedown" && event.type !== "pointerdown") return;
        if (event.button !== 0) return;
        if (g._dragging) {
          g._dragging = false;
          if (g._docCleanup) {
            g._docCleanup();
            g._docCleanup = null;
          }
        }
        const vIsInt = pp.sliderType === "int";
        const ratio = clamp((pos[0] - ds.trackLeft) / ds.trackW, 0, 1);
        let v = pp.sliderMin + ratio * (pp.sliderMax - pp.sliderMin);
        v = calcValue(v, pp.sliderMin, pp.sliderMax, pp.sliderStep, vIsInt);
        if (g.widget) g.widget.value = v;
        n.setDirtyCanvas(true, false);
        g._dragging = true;

        const startCX = event.clientX;
        const startVal = v;
        function onDocMove(e2) {
          if (!g._dragging) return;
          const dx = e2.clientX - startCX;
          const scale = app.canvas?.ds?.scale || 1;
          const ratioD = dx / scale / ds.trackW;
          let nv = startVal + ratioD * (pp.sliderMax - pp.sliderMin);
          nv = calcValue(nv, pp.sliderMin, pp.sliderMax, pp.sliderStep, pp.sliderType === "int");
          if (g.widget) g.widget.value = nv;
          n.setDirtyCanvas(true, false);
        }
        function onDocUp() {
          if (!g._dragging) return;
          g._dragging = false;
          g._docCleanup = null;
          document.removeEventListener("mousemove", onDocMove);
          document.removeEventListener("mouseup", onDocUp);
          n.setDirtyCanvas(true, true);
        }
        g._docCleanup = function () {
          document.removeEventListener("mousemove", onDocMove);
          document.removeEventListener("mouseup", onDocUp);
        };
        document.addEventListener("mousemove", onDocMove);
        document.addEventListener("mouseup", onDocUp);
        return true;
      },
      computeSize(width) {
        return [width, 60];
      },
    });
  }

  // 外部值变化回填
  const origCB = node.onWidgetChanged;
  node.onWidgetChanged = function (name, value, widget) {
    if (origCB) origCB.call(this, name, value, widget);
    if (name === WIDGET_NAME) updateVis(this);
  };

  // 工作流恢复：configure 值还原后按保存的 properties 重应用（不换算存量值）
  const origConfigure = node.configure;
  node.configure = function (...args) {
    const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
    if (!this._sfUS) return r;
    // configure 可能重建 widgets 数组，重新抓取引用
    this._sfUS.widget = this.widgets?.find((w) => w.name === WIDGET_NAME) || null;
    this._sfUS.outputTypeWidget =
      this.widgets?.find((w) => w.name === OUTPUT_TYPE_NAME) || null;
    const w = this._sfUS.widget;
    if (w) {
      w.value = calcValue(
        w.value,
        this.properties.sliderMin,
        this.properties.sliderMax,
        this.properties.sliderStep,
        this.properties.sliderType === "int"
      );
    }
    syncWidgetType(this);
    syncOutputType(this);
    updateVis(this);
    return r;
  };
  const origAG = node.onAfterGraphConfigured;
  node.onAfterGraphConfigured = function (...args) {
    if (origAG) origAG.apply(this, args);
    syncOutputType(this);
  };

  // 右键设置菜单（any_pack.js 同款 per-node 包装）
  const origMenu = node.getExtraMenuOptions;
  node.getExtraMenuOptions = function (canvas, options) {
    if (origMenu) {
      try {
        origMenu.apply(this, arguments);
      } catch (e) {
        console.warn("[SFUniversalSlider] getExtraMenuOptions:", e);
      }
    }
    if (Array.isArray(options)) {
      options.splice(0, 0, null, {
        content: "SF 万能滑条 设置",
        callback: () => showSettings(this),
      });
    }
  };

  node.size[0] = Math.max(node.size?.[0] || 0, 300);
}

app.registerExtension({
  name: "sfnodes.UniversalSlider",

  async nodeCreated(node) {
    if (node.comfyClass !== CLASS) return;
    injectCSS();
    setupSlider(node);
  },
});
