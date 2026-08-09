// ==========================================================================
// sf_pause_image_ui.js - SFPauseImage 节点体 UI（单个 DOM widget）
// ==========================================================================
//
// 布局（自上而下）：Pause/Pass 切换 pill、状态行、两行按钮（Regenerate/
// Continue + Copy/Save Disk/Save Output/Open）、图片预览（object-fit:contain
// 填满剩余高度）、尺寸行。
//
// 预览 /view 是 ComfyUI 原生路由，URL 经 api.apiURL() 构建（兼容托管部署）；
// 与 Pixaroma 的差异：无 accent 颜色设置（固定强调色）、无 canvas zoom 辅助。
//
// ==========================================================================

import { api } from "/scripts/api.js";
import { getState } from "./sf_pause_image_lib.js";
import { sfApiUrl } from "./sf_common.js";


const HEADER_H = 130;       // 切换 + 状态 + 两行按钮
const PREVIEW_MIN_H = 150;  // 预览区最小高度
const DIMS_H = 16;          // 预览下的尺寸行
export const NODE_MIN_W = 300;  // 4 按钮工具行容纳所需
// 常量 getMinHeight：固定数字每次保存/加载字节一致，node.size 不抖动，
// 工作流不会被误标"已修改"
export const NODE_MIN_H = HEADER_H + PREVIEW_MIN_H + DIMS_H;

function injectCSS() {
    if (document.getElementById("sf-pi-css")) return;
    const s = document.createElement("style");
    s.id = "sf-pi-css";
    s.textContent = `
.sf-pi-root { display:flex; flex-direction:column; flex:1 1 0; min-height:0;
  box-sizing:border-box; padding:6px; gap:6px; font:12px sans-serif; color:#ddd;
  overflow:hidden; }
.sf-pi-toggle { display:flex; background:rgba(0,0,0,0.25); border-radius:6px; padding:2px; gap:2px; flex:0 0 auto; }
.sf-pi-seg { flex:1 1 0; text-align:center; padding:4px 0; border-radius:5px; cursor:pointer;
  color:rgba(255,255,255,0.6); user-select:none; border:1px solid transparent; }
.sf-pi-seg.active { background:${"var(--sf-acc, #f66744)"}; color:#fff; border-color:${"var(--sf-acc, #f66744)"}; }
.sf-pi-seg:not(.active):hover { border-color:${"var(--sf-acc, #f66744)"}; color:#ddd; }
.sf-pi-status { flex:0 0 auto; font-size:11px; color:rgba(255,255,255,0.7); min-height:14px; text-align:center; }
.sf-pi-btns { display:flex; gap:6px; flex:0 0 auto; }
.sf-pi-btn { flex:1 1 0; min-width:0; height:26px; line-height:24px; border-radius:4px;
  border:1px solid rgba(255,255,255,0.18); background:rgba(255,255,255,0.05);
  color:rgba(255,255,255,0.85); font:12px sans-serif; cursor:pointer; padding:0 6px;
  box-sizing:border-box; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; user-select:none; }
.sf-pi-btn:hover:not(:disabled) { border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-pi-btn.primary:not(:disabled) { background:${"var(--sf-acc, #f66744)"}; border-color:${"var(--sf-acc, #f66744)"}; color:#fff; }
.sf-pi-btn.primary:hover:not(:disabled) { background:#ff8a5e; border-color:#ff8a5e; }
.sf-pi-btn:disabled { opacity:0.45; cursor:default; }
.sf-pi-preview { flex:1 1 0; min-height:0; position:relative; background:#1d1d1d;
  border:1px solid #333; border-radius:4px; overflow:hidden; }
.sf-pi-img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:none; }
.sf-pi-empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  text-align:center; color:#777; font-size:11px; padding:8px; box-sizing:border-box; }
.sf-pi-dims { flex:0 0 auto; text-align:center; font-size:10px; color:#aaa;
  min-height:13px; line-height:13px; }
`;
    document.head.appendChild(s);
}

// 构建 DOM widget。callbacks: { onGate(gate), onContinue(), onRegenerate(),
// onCopy(), onSaveDisk(), onSaveOutput(), onOpen() }。元素引用缓存在
// node._sfPauseImageEls，供 renderPause / showFrame 使用。
export function buildPauseWidget(node, callbacks) {
    injectCSS();
    const root = document.createElement("div");
    root.className = "sf-pi-root";

    const toggle = document.createElement("div");
    toggle.className = "sf-pi-toggle";
    const segPause = document.createElement("div");
    segPause.className = "sf-pi-seg";
    segPause.textContent = "Pause";
    segPause.title = "Run 时停在此处，预览后再继续";
    const segPass = document.createElement("div");
    segPass.className = "sf-pi-seg";
    segPass.textContent = "Pass";
    segPass.title = "直接放行；整条工作流一次跑完";
    toggle.append(segPause, segPass);
    segPause.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pause"); });
    segPass.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onGate("pass"); });

    const status = document.createElement("div");
    status.className = "sf-pi-status";

    // 第一行：工作流决策
    const btns = document.createElement("div");
    btns.className = "sf-pi-btns";
    const btnContinue = document.createElement("button");
    btnContinue.className = "sf-pi-btn primary";
    btnContinue.textContent = "▶ Continue";
    btnContinue.title = "从快照运行工作流其余部分";
    const btnRegen = document.createElement("button");
    btnRegen.className = "sf-pi-btn";
    btnRegen.textContent = "⟳ Regenerate";
    btnRegen.title = "在此处掷一张新图（尊重你的种子）";
    // Regenerate 在左、Continue 在右（Continue 是主要"提交"动作）
    btns.append(btnRegen, btnContinue);

    // 第二行：作用于预览图的工具
    const btns2 = document.createElement("div");
    btns2.className = "sf-pi-btns";
    const btnCopy = document.createElement("button");
    btnCopy.className = "sf-pi-btn";
    btnCopy.textContent = "Copy";
    btnCopy.title = "把预览的图片复制到剪贴板";
    const btnSaveDisk = document.createElement("button");
    btnSaveDisk.className = "sf-pi-btn";
    btnSaveDisk.textContent = "Save Disk";
    btnSaveDisk.title = "把预览的图片保存到电脑上的文件夹";
    const btnSaveOut = document.createElement("button");
    btnSaveOut.className = "sf-pi-btn";
    btnSaveOut.textContent = "Save Output";
    btnSaveOut.title = "把预览的图片保存到 ComfyUI 的 output 目录";
    const btnOpen = document.createElement("button");
    btnOpen.className = "sf-pi-btn";
    btnOpen.textContent = "Open";
    btnOpen.title = "在新标签页打开预览的图片";
    btns2.append(btnCopy, btnSaveDisk, btnSaveOut, btnOpen);

    // stopPropagation 防止点击到达 canvas（取消选中/拖拽）
    btnContinue.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onContinue(); });
    btnRegen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onRegenerate(); });
    btnCopy.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onCopy(); });
    btnSaveDisk.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onSaveDisk(); });
    btnSaveOut.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onSaveOutput(); });
    btnOpen.addEventListener("click", (e) => { e.stopPropagation(); callbacks.onOpen(); });

    const preview = document.createElement("div");
    preview.className = "sf-pi-preview";
    const img = document.createElement("img");
    img.className = "sf-pi-img";
    const empty = document.createElement("div");
    empty.className = "sf-pi-empty";
    empty.textContent = "Press Run to preview the image here";
    preview.append(img, empty);

    // 尺寸行在预览下方（自己的行），不压图
    const dims = document.createElement("div");
    dims.className = "sf-pi-dims";

    root.append(toggle, status, btns, btns2, preview, dims);

    node._sfPauseImageEls = {
        segPause, segPass, status,
        btnContinue, btnRegen, btnCopy, btnSaveDisk, btnSaveOut, btnOpen,
        img, empty, dims,
    };
    return root;
}

// 从当前状态重渲染控件。纯 DOM（绝不碰 node.size / node.properties 值），
// 加载路径上调用安全。
export function renderPause(node) {
    const els = node._sfPauseImageEls;
    if (!els) return;
    const s = getState(node);
    const paused = s.gate === "pause";
    const hasSnap = !!node._sfPauseImageHasSnapshot;  // 仅运行时（见 lib 注释）
    els.segPause.classList.toggle("active", paused);
    els.segPass.classList.toggle("active", !paused);

    // Continue / Regenerate 只在 Pause 模式有意义；Copy / Open 只要有图就可用
    els.btnRegen.disabled = !paused;
    els.btnContinue.disabled = !paused || !hasSnap;
    els.btnCopy.disabled = !hasSnap;
    els.btnSaveDisk.disabled = !hasSnap;
    els.btnSaveOut.disabled = !hasSnap;
    els.btnOpen.disabled = !hasSnap;

    if (node._sfPauseImageFlash) {
        els.status.textContent = node._sfPauseImageFlash;
    } else if (node._sfPauseImageBusy) {
        els.status.textContent = node._sfPauseImageBusy;
    } else if (!paused) {
        els.status.textContent = "Passing through: whole workflow runs";
    } else if (hasSnap) {
        els.status.textContent = "Paused and ready. Continue to run the rest.";
    } else {
        els.status.textContent = "Paused. Press Run to preview.";
    }
}

// 构建快照 frame 的 /view URL。带缓存戳：快照文件名按节点确定性（每次 pause
// run 覆盖写），不戳会命中浏览器缓存
export function frameViewUrl(frame) {
    const params = new URLSearchParams({
        filename: frame.filename,
        subfolder: frame.subfolder || "",
        type: frame.type || "temp",
        t: String(Date.now()),
    });
    return sfApiUrl(`/view?${params.toString()}`);
}

// 在预览中加载 + 显示快照 frame。frame = {filename, subfolder, type}。
// 成功启用 Continue；失败（重启后 temp PNG 被清）显示"expired"并禁用 Continue。
export function showFrame(node, frame) {
    const els = node._sfPauseImageEls;
    if (!els || !frame || !frame.filename) return;
    const url = frameViewUrl(frame);
    const { img, empty, dims } = els;
    img.onload = () => {
        img.style.display = "block";
        empty.style.display = "none";
        dims.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
        // 仅运行时标志（非 node.properties）：加载时的图片解析绝不重写序列化
        // 状态、不弄脏工作流
        node._sfPauseImageHasSnapshot = true;
        renderPause(node);
    };
    img.onerror = () => {
        img.style.display = "none";
        empty.style.display = "flex";
        empty.textContent = "Preview expired. Press Run to pause again.";
        dims.textContent = "";
        node._sfPauseImageHasSnapshot = false;
        renderPause(node);
    };
    img.src = url;
}
