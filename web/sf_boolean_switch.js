// ==========================================================================
// sf_boolean_switch.js - SF Boolean Switch 前端（复刻孤海布尔开关）
// ==========================================================================
//
// 复刻 BOOLEAN.js 的 Canvas 大开关交互（标签左 + 开关轨道右、单击切换、
// 双击改标签、节点配色），差异（已确认范围）：
// - widget 名中文 `开关`→英文 `value`（default True 不变）；标签 properties
//   键 `guhai_label`→`sfBoolLabel`，默认标签 `开关`→`value`
// - 原型补丁（beforeRegisterNodeDef）改 nodeCreated 实例装配
//  （simple_math.js 先例）；import 改绝对路径 "/scripts/app.js"
//  （check_web_imports.py B2）；扩展名 `sfnodes.BooleanSwitch`（B3）
// - 自定义 widget type `toggle_custom`→`sf_boolean_switch`、
//   name `guhai_toggle`→`sf_bool_ui`（与原插件共存不冲突）
// - 切换/改名后补 setDirtyCanvas（原版缺失，靠画布偶然重绘刷新）
// - 绘制/命中魔法数字收敛 lib TOGGLE 常量；标签归一/截断走 lib 纯函数
//
// 后端权威：nodes/utils/boolean_switch.py::SFBooleanSwitch
// ==========================================================================

import { app } from "/scripts/app.js";
import { el } from "./sf_common.js";
import {
  DEFAULT_LABEL,
  TOGGLE,
  ellipsisText,
  normalizeLabel,
  toggleHit,
  trackX,
} from "./sf_boolean_switch_lib.js";

const CLASS = "SFBooleanSwitch";
const WIDGET_NAME = "value";
const LABEL_PROP = "sfBoolLabel";
const DBL_MS = 350;

// ── Canvas 圆角矩形 ──
function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function getLabel(node) {
  return (node.properties && node.properties[LABEL_PROP]) || DEFAULT_LABEL;
}

function setLabel(node, text) {
  node.properties = node.properties || {};
  node.properties[LABEL_PROP] = normalizeLabel(text);
}

// ── 双击标签编辑（DOM 浮层输入框，原版 startLabelEdit/finishEdit 一致）──
function startLabelEdit(node, st, clientX, clientY) {
  if (st.activeInput) return;
  const scale = app.canvas?.ds?.scale ?? 1;
  const input = el("input");
  input.type = "text";
  input.value = getLabel(node);
  Object.assign(input.style, {
    position: "fixed",
    left: clientX + "px",
    top: clientY - 14 * scale + "px",
    fontSize: Math.max(12, Math.round(20 * scale)) + "px",
    fontWeight: "bold",
    color: "#e0e0e0",
    background: "#2a2a2a",
    border: "1px solid #555",
    borderRadius: "4px",
    padding: "2px 6px",
    outline: "none",
    zIndex: "99999",
    minWidth: "80px",
  });
  document.body.appendChild(input);
  st.activeInput = input;

  const finish = () => {
    if (!st.activeInput) return;
    setLabel(node, st.activeInput.value);
    st.activeInput.remove();
    st.activeInput = null;
    node.setDirtyCanvas(true, true);
  };
  st._finishEdit = finish;

  const raf = globalThis.requestAnimationFrame || ((fn) => fn());
  raf(() => {
    input.focus();
    input.select();
  });
  input.addEventListener(
    "keydown",
    (ev) => {
      if (ev.key === "Enter" || ev.key === "Escape") {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        finish();
      }
    },
    true
  );
  input.addEventListener("blur", () => {
    setTimeout(finish, 50);
  });
}

// ── 节点装配 ──
function setupSwitch(node) {
  if (!node.properties) node.properties = {};
  if (node.properties[LABEL_PROP] === undefined) node.properties[LABEL_PROP] = DEFAULT_LABEL;

  const boolWidget = node.widgets ? node.widgets.find((w) => w.name === WIDGET_NAME) : null;
  if (!boolWidget) return;
  boolWidget.hidden = true;

  // 节点配色（原版，仅首次创建时；configure 恢复不覆盖用户改色——
  // 原版同样只在 onNodeCreated 设色，工作流存色优先）
  node.color = "#4F4047";
  node.bgcolor = "#493C42";

  const st = {
    widget: boolWidget,
    isOn: !!boolWidget.value,
    lastClickTime: 0,
    activeInput: null,
    _finishEdit: null,
  };
  node._sfBool = st;

  const syncFromWidget = () => {
    st.isOn = !!st.widget.value;
  };

  if (typeof node.addCustomWidget === "function") {
    node.addCustomWidget({
      name: "sf_bool_ui",
      type: "sf_boolean_switch",
      draw(ctx, n, widgetWidth, y, H) {
        const labelText = getLabel(n);
        const { tw, th, m } = TOGGLE;
        const tx = trackX(widgetWidth);
        const ty = y + 6 + (H - th) / 2;

        // 标签：在按钮左侧区域内水平居中
        const textAreaRight = tx - m;
        const textAreaCenter = (m + textAreaRight) / 2;
        const maxTextW = textAreaRight - m;
        ctx.font = "bold 24px sans-serif";
        ctx.fillStyle = "#e0e0e0";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          ellipsisText(labelText, maxTextW, (t) => ctx.measureText(t)),
          textAreaCenter,
          y + 6 + H / 2
        );

        // 开关轨道
        ctx.save();
        if (st.isOn) {
          ctx.shadowColor = "rgba(76,175,80,0.4)";
          ctx.shadowBlur = 10;
          ctx.fillStyle = "#4CAF50";
        } else {
          ctx.fillStyle = "#606060";
        }
        rrect(ctx, tx, ty, tw, th, th / 2);
        ctx.fill();
        ctx.restore();

        // 圆形旋钮
        const kr = 11;
        const kx = st.isOn ? tx + tw - kr - 3 : tx + kr + 3;
        const ky = ty + th / 2;
        ctx.save();
        ctx.shadowColor = "rgba(0,0,0,0.3)";
        ctx.shadowBlur = 4;
        ctx.fillStyle = st.isOn ? "#ffffff" : "#999999";
        ctx.beginPath();
        ctx.arc(kx, ky, kr, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
        const now = Date.now();
        const dbl = now - st.lastClickTime < DBL_MS;
        st.lastClickTime = now;
        // 双击 → 编辑标签
        if (dbl) {
          startLabelEdit(n, st, event.clientX, event.clientY);
          return true;
        }
        // 单击落点开关区 → 切换（宽度直接读节点 size，不依赖 draw 先跑）
        if (toggleHit(pos[0], n.size?.[0] || 200)) {
          st.isOn = !st.isOn;
          st.widget.value = st.isOn;
          this.value = st.isOn;
          n.setDirtyCanvas(true, false);
          return true;
        }
        return false;
      },
      computeSize(width) {
        return [width, 44];
      },
    });
  }

  // 外部值变化回填（含工作流恢复后联动）
  const origCB = node.onWidgetChanged;
  node.onWidgetChanged = function (name, value, widget) {
    if (origCB) origCB.call(this, name, value, widget);
    if (name === WIDGET_NAME) {
      syncFromWidget();
      updateVis(this);
    }
  };

  // 工作流恢复：configure 值还原后同步开关态（编辑中输入框先落盘）
  const origConfigure = node.configure;
  node.configure = function (...args) {
    const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
    if (!this._sfBool) return r;
    const g = this._sfBool;
    g.widget = this.widgets?.find((w) => w.name === WIDGET_NAME) || null;
    if (g.widget) g.isOn = !!g.widget.value;
    if (g.activeInput && g._finishEdit) g._finishEdit();
    updateVis(this);
    return r;
  };

  function updateVis(nd) {
    nd.setDirtyCanvas(true, true);
  }
}

app.registerExtension({
  name: "sfnodes.BooleanSwitch",

  async nodeCreated(node) {
    if (node.comfyClass !== CLASS) return;
    setupSwitch(node);
  },
});
