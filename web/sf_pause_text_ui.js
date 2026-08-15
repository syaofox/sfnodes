// ==========================================================================
// sf_pause_text_ui.js - SFPauseText 节点体 UI（单个 DOM widget）
// ==========================================================================
//
// 布局：状态条（Pause/Continue 提示）在顶部作为普通 DOM 行（Classic 与 Vue
// 渲染器统一显示，不做浮动/偏移的渲染器双路径）→ 可编辑文本框（占满剩余高度，
// 头部带字段名、Pause/Pass/Keep 切换与 Copy/Revert 图标）→ 计数 + Regenerate /
// Continue 按钮行。
//
// 与 Pixaroma 的差异（已确认范围）：状态条不做 canvas 绘制 / slot 行浮动
// （简化为普通行）；无 accent 颜色设置（固定强调色）；无 resize floor /
// canvas zoom 穿透辅助。
//
// ==========================================================================

import { getState, isEdited } from "./sf_pause_text_lib.js";
import { installWheelZoomPassthrough } from "./sf_common.js";


// 非填充行的固定垂直预算 -> getMinHeight 是常量（不随内容抖动）
const PAD = 6;
const HDR_H = 24;
const BODY_MIN_H = 120;
const BOT_H = 28;
const CORE_H = PAD + HDR_H + BODY_MIN_H + PAD + BOT_H + PAD;
// 计数 + 两个按钮舒适容纳的最小宽度；保留得较紧凑
export const NODE_MIN_W = 400;
export function nodeMinH() { return CORE_H; }

function injectCSS() {
    if (document.getElementById("sf-ptx-css")) return;
    const s = document.createElement("style");
    s.id = "sf-ptx-css";
    s.textContent = `
.sf-ptx-root { position:relative; display:flex; flex-direction:column; flex:1 1 0;
  min-height:0; box-sizing:border-box; padding:${PAD}px; gap:${PAD}px;
  font:12px sans-serif; color:#ddd; overflow:hidden; background:transparent; }
.sf-ptx-band { flex:0 0 auto; font:11px sans-serif; color:rgba(255,255,255,0.72);
  text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  padding:0 8px; box-sizing:border-box; height:16px; line-height:16px; }
.sf-ptx-box { flex:1 1 0; min-height:0; display:flex; flex-direction:column;
  background:#1d1d1d; border:1px solid #333; border-radius:5px; overflow:hidden; }
.sf-ptx-box.pt-focus { border-color:${"var(--sf-acc, #f66744)"}; }
.sf-ptx-box.pt-off { opacity:0.55; }
.sf-ptx-hdr { flex:0 0 auto; display:flex; align-items:center; gap:6px;
  padding:3px 6px 3px 9px; border-bottom:1px solid #2c2c2c; background:rgba(255,255,255,0.02); }
.sf-ptx-hlbl { font:10px 'Segoe UI',-apple-system,sans-serif; color:#8f8f8f; flex:1 1 0;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-ptx-toggle { display:flex; background:rgba(0,0,0,0.25); border-radius:5px; padding:1px; gap:2px; flex:0 0 auto; }
.sf-ptx-seg { text-align:center; padding:2px 9px; border-radius:4px; cursor:pointer;
  color:rgba(255,255,255,0.6); user-select:none; border:1px solid transparent; font-size:10px; }
.sf-ptx-seg.active { background:${"var(--sf-acc, #f66744)"}; color:#fff; border-color:${"var(--sf-acc, #f66744)"}; }
.sf-ptx-seg:not(.active):hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
.sf-ptx-hic { width:19px; height:18px; border-radius:4px; display:flex; align-items:center;
  justify-content:center; cursor:pointer; background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.14); color:rgba(255,255,255,0.72); flex:0 0 auto; }
.sf-ptx-hic:hover:not(.off) { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-ptx-hic.ok, .sf-ptx-hic.ok:hover { background:#3ec371; border-color:#3ec371; color:#fff; }
.sf-ptx-hic.off { opacity:0.35; cursor:default; }
.sf-ptx-ta { flex:1 1 0; min-height:0; width:100%; box-sizing:border-box;
  background:transparent; color:#e0e0e0; border:0; outline:none; resize:none;
  font:12px monospace; line-height:1.4; padding:6px 8px; }
.sf-ptx-ta::placeholder { color:#5c5c5c; font-style:italic; }
.sf-ptx-ta:disabled { color:#9a9a9a; }
.sf-ptx-bot { display:flex; align-items:center; gap:6px; flex:0 0 auto; flex-wrap:wrap; justify-content:flex-end; }
.sf-ptx-count { flex:1 1 0; min-width:0; font-size:10px; color:#aaa;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sf-ptx-btn { height:26px; padding:0 12px; border-radius:4px;
  border:1px solid rgba(255,255,255,0.18); background:rgba(255,255,255,0.05);
  color:rgba(255,255,255,0.85); font:12px sans-serif; cursor:pointer;
  box-sizing:border-box; white-space:nowrap; user-select:none; flex:0 0 auto; }
.sf-ptx-btn:hover:not(:disabled) { border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-ptx-btn.primary:not(:disabled) { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-ptx-btn.primary:hover:not(:disabled) { background:#ff8a5e; border-color:#ff8a5e; }
.sf-ptx-btn:disabled { opacity:0.45; cursor:default; }
`;
    document.head.appendChild(s);
}

const COPY_SVG =
    '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="9" y="9" width="11" height="11" rx="2"/>' +
    '<path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
const REVERT_SVG =
    '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-1"/></svg>';

// 当前状态串（busy > flash > 由 gate 推导）
export function statusText(node) {
    const s = getState(node);
    if (node._sfPauseTextBusy) return node._sfPauseTextBusy;
    if (node._sfPauseTextFlash) return node._sfPauseTextFlash;
    if (s.gate === "pass") return "Passing through: whole workflow runs";
    if (s.gate === "keep") return "Keeping this text: each Run makes an image";
    return isEdited(node) ? "Edited. Continue when ready." : "Paused. Edit and press Continue.";
}

// 构建 DOM widget。callbacks: { onGate, onContinue, onRegenerate, onCopy, onRevert, onInput }
export function buildPauseTextWidget(node, callbacks) {
    injectCSS();
    const root = document.createElement("div");
    root.className = "sf-ptx-root";

    // 状态条（顶部普通行）
    const band = document.createElement("div");
    band.className = "sf-ptx-band";

    // 可编辑盒子；头部带字段名 + Pause/Pass 切换 + 图标
    const box = document.createElement("div");
    box.className = "sf-ptx-box";
    const hdr = document.createElement("div");
    hdr.className = "sf-ptx-hdr";
    const hlbl = document.createElement("span");
    hlbl.className = "sf-ptx-hlbl";
    hlbl.textContent = "text";
    const toggle = document.createElement("div");
    toggle.className = "sf-ptx-toggle";
    const segPause = document.createElement("div");
    segPause.className = "sf-ptx-seg";
    segPause.textContent = "Pause";
    segPause.title = "Run 时停在此处，编辑后再继续";
    const segPass = document.createElement("div");
    segPass.className = "sf-ptx-seg";
    segPass.textContent = "Pass";
    segPass.title = "直接放行；整条工作流一次跑完";
    const segKeep = document.createElement("div");
    segKeep.className = "sf-ptx-seg";
    segKeep.textContent = "Keep";
    segKeep.title = "保留这段文本；每次 Run 用它出图（模型被跳过）";
    toggle.append(segPause, segPass, segKeep);
    const copyBtn = document.createElement("span");
    copyBtn.className = "sf-ptx-hic";
    copyBtn.innerHTML = COPY_SVG;
    copyBtn.title = "复制这段文本";
    const revertBtn = document.createElement("span");
    revertBtn.className = "sf-ptx-hic";
    revertBtn.innerHTML = REVERT_SVG;
    revertBtn.title = "恢复模型的原始文本";
    hdr.append(hlbl, toggle, copyBtn, revertBtn);
    const ta = document.createElement("textarea");
    ta.className = "sf-ptx-ta";
    ta.spellcheck = false;
    ta.placeholder = "The model's text will appear here on Run";
    box.append(hdr, ta);
    installWheelZoomPassthrough(ta); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)

    // 底部行：计数 + Regenerate / Continue
    const bot = document.createElement("div");
    bot.className = "sf-ptx-bot";
    const count = document.createElement("span");
    count.className = "sf-ptx-count";
    const btnRegen = document.createElement("button");
    btnRegen.className = "sf-ptx-btn";
    btnRegen.textContent = "⟳ Regenerate";
    btnRegen.title = "获取新文本：滚动上游生成节点的种子";
    const btnContinue = document.createElement("button");
    btnContinue.className = "sf-ptx-btn primary";
    btnContinue.textContent = "▶ Continue";
    btnContinue.title = "只用你的编辑文本运行工作流其余部分";
    bot.append(count, btnRegen, btnContinue);

    root.append(band, box, bot);

    // 事件。stopPropagation 防止 canvas 拖拽/取消选中/快捷键触发
    ta.addEventListener("input", () => callbacks.onInput(ta.value));
    ta.addEventListener("keydown", (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;  // 放行所有修饰键组合(保存/复制/运行等)
        e.stopPropagation();
    });
    ta.addEventListener("pointerdown", (e) => e.stopPropagation());
    ta.addEventListener("mousedown", (e) => e.stopPropagation());
    ta.addEventListener("focus", () => box.classList.add("pt-focus"));
    ta.addEventListener("blur", () => box.classList.remove("pt-focus"));

    segPause.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pause"); });
    segPass.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pass"); });
    segKeep.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("keep"); });
    copyBtn.addEventListener("click", (e) => { e.stopPropagation(); if (!copyBtn.classList.contains("off")) callbacks.onCopy(); });
    revertBtn.addEventListener("click", (e) => { e.stopPropagation(); if (!revertBtn.classList.contains("off")) callbacks.onRevert(); });
    for (const b of [segPause, segPass, segKeep, copyBtn, revertBtn]) {
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        b.addEventListener("mousedown", (e) => e.stopPropagation());
    }
    btnRegen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onRegenerate(); });
    btnContinue.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onContinue(); });

    node._sfPauseTextEls = {
        root, band, box, hlbl, segPause, segPass, segKeep, copyBtn, revertBtn, ta, count, btnRegen, btnContinue,
    };
    return root;
}

function countLabel(text) {
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    return `${chars} char${chars === 1 ? "" : "s"} · ${words} word${words === 1 ? "" : "s"}`;
}

export function flashIcon(iconEl) {
    if (!iconEl) return;
    iconEl.classList.add("ok");
    setTimeout(() => iconEl.classList.remove("ok"), 700);
}

// 把存储文本推进 textarea。只在不同时写入，绝不打断用户打字中的光标
export function syncText(node) {
    const els = node._sfPauseTextEls;
    if (!els) return;
    const s = getState(node);
    if (els.ta.value !== s.text) els.ta.value = s.text;
}

// 从 state 重渲染控件。纯 DOM 操作，加载路径上安全
export function renderPause(node) {
    const els = node._sfPauseTextEls;
    if (!els) return;
    const s = getState(node);
    const gate = s.gate;
    const pass = gate === "pass";
    const keep = gate === "keep";
    const edited = isEdited(node);
    // 编辑与操作按钮在 Pause 和 Keep 可用，Pass 关闭（pass 直接跑模型、忽略盒子）
    const editable = !pass;

    els.segPause.classList.toggle("active", gate === "pause");
    els.segPass.classList.toggle("active", pass);
    els.segKeep.classList.toggle("active", keep);

    els.ta.disabled = !editable;
    els.box.classList.toggle("pt-off", pass);
    els.ta.placeholder = editable
        ? "The model's text will appear here on Run"
        : "Passing through - the model's text is sent as-is";

    els.hlbl.innerHTML = edited
        ? `text · <span style="color:${"var(--sf-acc, #f66744)"}">edited</span>`
        : "text";

    const hasText = !!s.text;
    els.copyBtn.classList.toggle("off", !hasText);
    els.revertBtn.classList.toggle("off", !edited);
    // Keep 模式下 Regenerate 变灰：Keep 复用当前文本，从模型取新提示词不属于这里——
    // 想取新文本切回 Pause
    els.btnRegen.disabled = !editable || keep || !!node._sfPauseTextBusy;
    els.btnRegen.title = keep
        ? "切回 Pause 才能从模型获取新文本"
        : "获取新文本：滚动上游生成节点的种子";
    els.btnContinue.disabled = !editable || !!node._sfPauseTextBusy;
    // Keep 下按钮只是出图（同顶部 Run），叫 Run；Pause 下提交你的编辑，保持 Continue
    els.btnContinue.textContent = keep ? "▶ Run" : "▶ Continue";
    els.btnContinue.title = keep
        ? "用这段文本出图（同按 Run）"
        : "只用你的编辑文本运行工作流其余部分";

    els.count.textContent = countLabel(s.text);
    els.band.textContent = statusText(node);
    node.setDirtyCanvas?.(true, false);
}
