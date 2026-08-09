// ==========================================================================
// sf_pause_mask_ui.js - SFPauseMask 节点体 UI（单个 DOM widget）
// ==========================================================================
//
// 与 sf_pause_image_ui.js 同构（类前缀 sf-pm-）：Pause/Pass 切换、状态行、
// 两行按钮（Regenerate/Continue + Copy/Save Disk/Save Output/Open）、遮罩
// 灰度预览（object-fit:contain）、尺寸行。快照是 L 模式灰度 PNG，/view 预览。
//
// ==========================================================================

import { getState } from "./sf_pause_mask_lib.js";
import { sfApiUrl } from "./sf_common.js";


const HEADER_H = 130;       // 切换 + 状态 + 两行按钮
const PREVIEW_MIN_H = 150;  // 预览区最小高度
const DIMS_H = 16;          // 预览下的尺寸行
export const NODE_MIN_W = 300;
export const NODE_MIN_H = HEADER_H + PREVIEW_MIN_H + DIMS_H;

function injectCSS() {
    if (document.getElementById("sf-pm-css")) return;
    const s = document.createElement("style");
    s.id = "sf-pm-css";
    s.textContent = `
.sf-pm-root { display:flex; flex-direction:column; flex:1 1 0; min-height:0;
  box-sizing:border-box; padding:6px; gap:6px; font:12px sans-serif; color:#ddd;
  overflow:hidden; }
.sf-pm-toggle { display:flex; background:rgba(0,0,0,0.25); border-radius:6px; padding:2px; gap:2px; flex:0 0 auto; }
.sf-pm-seg { flex:1 1 0; text-align:center; padding:4px 0; border-radius:5px; cursor:pointer;
  color:rgba(255,255,255,0.6); user-select:none; border:1px solid transparent; }
.sf-pm-seg.active { background:${"var(--sf-acc, #f66744)"}; color:#fff; border-color:${"var(--sf-acc, #f66744)"}; }
.sf-pm-seg:not(.active):hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
.sf-pm-status { flex:0 0 auto; font-size:11px; color:rgba(255,255,255,0.7); min-height:14px; text-align:center; }
.sf-pm-btns { display:flex; gap:6px; flex:0 0 auto; }
.sf-pm-btn { flex:1 1 0; min-width:0; height:26px; line-height:24px; border-radius:4px;
  border:1px solid rgba(255,255,255,0.18); background:rgba(255,255,255,0.05);
  color:rgba(255,255,255,0.85); font:12px sans-serif; cursor:pointer; padding:0 6px;
  box-sizing:border-box; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; user-select:none; }
.sf-pm-btn:hover:not(:disabled) { border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-pm-btn.primary:not(:disabled) { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-pm-btn.primary:hover:not(:disabled) { background:#ff8a5e; border-color:#ff8a5e; }
.sf-pm-btn:disabled { opacity:0.45; cursor:default; }
.sf-pm-preview { flex:1 1 0; min-height:0; position:relative; background:#1d1d1d;
  border:1px solid #333; border-radius:4px; overflow:hidden; }
.sf-pm-img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:none; }
.sf-pm-empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  text-align:center; color:#777; font-size:11px; padding:8px; box-sizing:border-box; }
.sf-pm-dims { flex:0 0 auto; text-align:center; font-size:10px; color:#aaa;
  min-height:13px; line-height:13px; }
`;
    document.head.appendChild(s);
}

// 构建 DOM widget。callbacks: { onGate(gate), onContinue(), onRegenerate(),
// onCopy(), onSaveDisk(), onSaveOutput(), onOpen() }。元素引用缓存在
// node._sfPauseMaskEls。
export function buildPauseWidget(node, callbacks) {
    injectCSS();
    const root = document.createElement("div");
    root.className = "sf-pm-root";

    const toggle = document.createElement("div");
    toggle.className = "sf-pm-toggle";
    const segPause = document.createElement("div");
    segPause.className = "sf-pm-seg";
    segPause.textContent = "Pause";
    segPause.title = "Run 时停在此处，预览后再继续";
    const segPass = document.createElement("div");
    segPass.className = "sf-pm-seg";
    segPass.textContent = "Pass";
    segPass.title = "直接放行；整条工作流一次跑完";
    toggle.append(segPause, segPass);
    segPause.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pause"); });
    segPass.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pass"); });

    const status = document.createElement("div");
    status.className = "sf-pm-status";

    // 第一行：工作流决策
    const btns = document.createElement("div");
    btns.className = "sf-pm-btns";
    const btnContinue = document.createElement("button");
    btnContinue.className = "sf-pm-btn primary";
    btnContinue.textContent = "▶ Continue";
    btnContinue.title = "从快照运行工作流其余部分";
    const btnRegen = document.createElement("button");
    btnRegen.className = "sf-pm-btn";
    btnRegen.textContent = "⟳ Regenerate";
    btnRegen.title = "在此处掷一张新遮罩（尊重你的种子）";
    btns.append(btnRegen, btnContinue);

    // 第二行：作用于预览图的工具
    const btns2 = document.createElement("div");
    btns2.className = "sf-pm-btns";
    const btnCopy = document.createElement("button");
    btnCopy.className = "sf-pm-btn";
    btnCopy.textContent = "Copy";
    btnCopy.title = "把预览的遮罩复制到剪贴板";
    const btnSaveDisk = document.createElement("button");
    btnSaveDisk.className = "sf-pm-btn";
    btnSaveDisk.textContent = "Save Disk";
    btnSaveDisk.title = "把预览的遮罩保存到电脑上的文件夹";
    const btnSaveOut = document.createElement("button");
    btnSaveOut.className = "sf-pm-btn";
    btnSaveOut.textContent = "Save Output";
    btnSaveOut.title = "把预览的遮罩保存到 ComfyUI 的 output 目录";
    const btnOpen = document.createElement("button");
    btnOpen.className = "sf-pm-btn";
    btnOpen.textContent = "Open";
    btnOpen.title = "在新标签页打开预览的遮罩";
    btns2.append(btnCopy, btnSaveDisk, btnSaveOut, btnOpen);

    btnContinue.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onContinue(); });
    btnRegen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onRegenerate(); });
    btnCopy.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onCopy(); });
    btnSaveDisk.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onSaveDisk(); });
    btnSaveOut.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onSaveOutput(); });
    btnOpen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onOpen(); });

    const preview = document.createElement("div");
    preview.className = "sf-pm-preview";
    const img = document.createElement("img");
    img.className = "sf-pm-img";
    const empty = document.createElement("div");
    empty.className = "sf-pm-empty";
    empty.textContent = "Press Run to preview the mask here";
    preview.append(img, empty);

    const dims = document.createElement("div");
    dims.className = "sf-pm-dims";

    root.append(toggle, status, btns, btns2, preview, dims);

    node._sfPauseMaskEls = {
        segPause, segPass, status,
        btnContinue, btnRegen, btnCopy, btnSaveDisk, btnSaveOut, btnOpen,
        img, empty, dims,
    };
    return root;
}

// 从当前状态重渲染控件。纯 DOM（绝不碰 node.size / node.properties 值）
export function renderPause(node) {
    const els = node._sfPauseMaskEls;
    if (!els) return;
    const s = getState(node);
    const paused = s.gate === "pause";
    const hasSnap = !!node._sfPauseMaskHasSnapshot;  // 仅运行时
    els.segPause.classList.toggle("active", paused);
    els.segPass.classList.toggle("active", !paused);

    els.btnRegen.disabled = !paused;
    els.btnContinue.disabled = !paused || !hasSnap;
    els.btnCopy.disabled = !hasSnap;
    els.btnSaveDisk.disabled = !hasSnap;
    els.btnSaveOut.disabled = !hasSnap;
    els.btnOpen.disabled = !hasSnap;

    if (node._sfPauseMaskFlash) {
        els.status.textContent = node._sfPauseMaskFlash;
    } else if (node._sfPauseMaskBusy) {
        els.status.textContent = node._sfPauseMaskBusy;
    } else if (!paused) {
        els.status.textContent = "Passing through: whole workflow runs";
    } else if (hasSnap) {
        els.status.textContent = "Paused and ready. Continue to run the rest.";
    } else {
        els.status.textContent = "Paused. Press Run to preview.";
    }
}

// 构建快照 frame 的 /view URL。带缓存戳：快照按节点确定性覆盖写
export function frameViewUrl(frame) {
    const params = new URLSearchParams({
        filename: frame.filename,
        subfolder: frame.subfolder || "",
        type: frame.type || "temp",
        t: String(Date.now()),
    });
    return sfApiUrl(`/view?${params.toString()}`);
}

// 在预览中加载 + 显示快照 frame。成功启用 Continue；失败显示 expired 并禁用
export function showFrame(node, frame) {
    const els = node._sfPauseMaskEls;
    if (!els || !frame || !frame.filename) return;
    const url = frameViewUrl(frame);
    const { img, empty, dims } = els;
    img.onload = () => {
        img.style.display = "block";
        empty.style.display = "none";
        dims.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
        node._sfPauseMaskHasSnapshot = true;
        renderPause(node);
    };
    img.onerror = () => {
        img.style.display = "none";
        empty.style.display = "flex";
        empty.textContent = "Preview expired. Press Run to pause again.";
        dims.textContent = "";
        node._sfPauseMaskHasSnapshot = false;
        renderPause(node);
    };
    img.src = url;
}
