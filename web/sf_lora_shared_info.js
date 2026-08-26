// ==========================================================================
// SF LoRA Shared Info - 样例图/预览/hover/markdown 复用内核
// 由 sf_lora_stack_info.js（浮动面板）与 sf_lora_info.js（对话框）共享，
// 消除两者 800L+ 重复（sample 网格/预览/标题悬停/缓存/markdown 插入）。
// 纯 DOM + app 依赖，无节点状态，可直接 import。
// ==========================================================================
import { app } from "/scripts/app.js";
import { copyText, injectCSSOnce } from "./sf_common.js";

// ── 图标（mask-image data URI，与两面板历史统一样式）───────────────────────
export const SAMPLE_ICON_TRASH = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 3h6l1 2h4v2H4V5h5l1-2zm-2 6h10l-1 9a1 1 0 01-1 1H8a1 1 0 01-1-1L6 9zM10 11v6M14 11v6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";
export const SAMPLE_ICON_LOAD = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z' fill='black'/%3E%3C/svg%3E";
export const SAMPLE_ICON_PROMPT = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M9 5a2 2 0 002 2h6a2 2 0 002-2M9 5a2 2 0 012-2h4a2 2 0 012 2M9 12h6M9 16h6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";

export function makeSampleIcon(url) {
    const s = document.createElement("span");
    s.style.cssText = "width:10px;height:10px;background-color:#fff;display:block;-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;";
    s.style.webkitMaskImage = `url("${url}")`;
    s.style.maskImage = `url("${url}")`;
    return s;
}

// ── 样例 markdown 插入 ───────────────────────────────────────────────────
export function buildSampleMarkdown(path) {
    const base = String(path || "").split("/").pop() || "image";
    const alt = base.replace(/\.[^.]+$/, "");
    const rel = `sample/${encodeURIComponent(base)}`;
    return `![${alt}](${rel})`;
}

export function insertAtCursor(textarea, text) {
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? start;
    textarea.setRangeText(text, start, end, "end");
    textarea.focus();
    const pos = start + text.length;
    textarea.selectionStart = textarea.selectionEnd = pos;
}

// 相对 sample 路径 -> 图片 URL（基于当前 lora 目录，改名后自动跟随）
export function resolveSampleUrl(rel, loraName) {
    let r = rel;
    try { r = decodeURIComponent(rel); } catch { /* 保留原样 */ }
    const idx = loraName.lastIndexOf("/");
    const dir = idx === -1 ? "" : loraName.slice(0, idx + 1);
    return `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(dir + r)}`;
}

// 统一视频扩展判定：与 sf_lora_stack_info / lora_routes 保持一致
const _VIDEO_RE = /\.(mp4|m4v|mov|webm|mkv)$/i;
export function isVideoPath(p) {
    return typeof p === "string" && _VIDEO_RE.test(p);
}

// ── 样例列表短期缓存（2s 去重，避免同面板三处并发各发一次）────────────────
const _sampleCache = new Map();
// 缓存键含 kind（loras 缺省省略前缀，旧调用行为不变）。
function _sampleCacheKey(loraName, kind) {
    return kind ? `${kind}:${loraName}` : loraName;
}
export function fetchSamplesCached(loraName, kind) {
    if (!loraName || loraName === "None") return Promise.resolve({ images: [], sample_dir: "" });
    const key = _sampleCacheKey(loraName, kind);
    if (_sampleCache.has(key)) return _sampleCache.get(key);
    const kq = kind ? `&kind=${encodeURIComponent(kind)}` : "";
    const p = app.api.fetchApi(`/api/sfnodes/lora_samples?filename=${encodeURIComponent(loraName)}${kq}`)
        .then((r) => r.ok ? r.json() : { images: [] })
        .catch(() => ({ images: [] }))
        .finally(() => setTimeout(() => _sampleCache.delete(key), 2000));
    _sampleCache.set(key, p);
    return p;
}

export function invalidateSamplesCache(loraName, kind) {
    if (loraName) _sampleCache.delete(_sampleCacheKey(loraName, kind));
}

// ── 大图预览（图片/视频，支持方向键切换）──────────────────────────────────
export function openSamplePreview(path, allPaths) {
    const list = Array.isArray(allPaths) && allPaths.length ? allPaths : [path];
    let idx = list.indexOf(path);
    if (idx < 0) idx = 0;
    const overlay = document.createElement("div");
    overlay.className = "sf-lora-sample-preview";
    overlay.style.cssText = "position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.75);display:flex;align-items:center;justify-content:center;cursor:pointer;";
    let media = null;
    const render = (i) => {
        if (i < 0 || i >= list.length) return;
        idx = i;
        const p = list[idx];
        const isVideo = isVideoPath(p);
        if (media) media.remove();
        if (isVideo) {
            media = document.createElement("video");
            media.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(p)}`;
            media.controls = true;
            media.autoplay = true;
            media.style.cssText = "max-width:90vw;max-height:90vh;border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,0.6);";
        } else {
            media = document.createElement("img");
            media.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(p)}`;
            media.alt = p.split("/").pop();
            media.style.cssText = "max-width:90vw;max-height:90vh;border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,0.6);";
        }
        media.addEventListener("click", close);
        overlay.appendChild(media);
    };
    const close = () => {
        overlay.remove();
        document.removeEventListener("keydown", onKey, true);
        // 标记由预览消费了 Esc，避免外层 dialog 的 cancel 误关（与 sf_lora_info 对话框协同）
        try { document.body.dataset.sfPreviewEsc = "1"; setTimeout(() => { try { delete document.body.dataset.sfPreviewEsc; } catch {} }, 50); } catch {}
    };
    const onKey = (e) => {
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); close(); }
        else if (e.key === "ArrowLeft") { if (idx > 0) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); render(idx - 1); } }
        else if (e.key === "ArrowRight") { if (idx < list.length - 1) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); render(idx + 1); } }
    };
    render(idx);
    overlay.addEventListener("click", close);
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(overlay);
}

// ── PNG 内嵌工作流解析（复用 sf_lora_info 原实现）──────────────────────
async function readPngWorkflowData(url) {
    let resp;
    try { resp = await fetch(url); } catch { return null; }
    if (!resp.ok) return null;
    const buf = await resp.arrayBuffer();
    const bytes = new Uint8Array(buf);
    if (bytes.length < 24 || bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47) return null;
    const dec = new TextDecoder();
    let off = 8;
    while (off + 12 <= bytes.length) {
        const len = ((bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3]) >>> 0;
        const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
        const dataStart = off + 8;
        const dataEnd = dataStart + len;
        if (dataEnd + 4 > bytes.length) break;
        if (type === "workflow" || type === "prompt") {
            return { chunk: type, data: dec.decode(bytes.slice(dataStart, dataEnd)) };
        }
        if (type === "tEXt") {
            const str = dec.decode(bytes.slice(dataStart, dataEnd));
            const nul = str.indexOf("\0");
            if (nul > 0) {
                const key = str.slice(0, nul);
                const value = str.slice(nul + 1);
                if (key === "workflow" || key === "prompt") return { chunk: key, data: value };
            }
        }
        off = dataEnd + 4;
    }
    return null;
}

// URL 参数化的通用入口：url 指向任意可 fetch 的图片原始字节（PNG 需含
// workflow/prompt chunk）。image_browser 等外部模块经 /view 原始字节复用此路径。
export async function loadWorkflowFromImageUrl(url, onError) {
    const embedded = await readPngWorkflowData(url);
    if (!embedded) {
        onError("该图片未内嵌工作流数据，无法载入为工作流（可用 SaveImage 输出的 PNG 测试）。");
        return false;
    }
    try {
        const data = JSON.parse(embedded.data);
        const load = async () => {
            if (embedded.chunk === "prompt" && typeof app.loadApiJson === "function") {
                await app.loadApiJson(data);
            } else {
                await app.loadGraphData(data, true, true);
            }
        };
        const cmd = app.extensionManager?.command;
        if (cmd && typeof cmd.execute === "function") {
            await cmd.execute("Comfy.NewBlankWorkflow");
            await load();
            return true;
        }
        if (!confirm("当前 ComfyUI 不支持新建标签，载入将替换当前画布内容，继续吗？")) return false;
        await load();
        return true;
    } catch (e) {
        console.warn("[SF Model Info] load workflow failed:", e);
        onError("工作流载入失败：" + (e.message || e));
        return false;
    }
}

export async function loadImageAsWorkflow(path, onError) {
    const url = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}`;
    return loadWorkflowFromImageUrl(url, onError);
}

// ── 标题悬停预览（civitai_00_xxx 标题 -> 对应 sample 原图）──────────────
export function attachSampleTitleHover(container, loraName) {
    if (!container || !loraName) return;
    const links = container.querySelectorAll("h3 a");
    if (!links.length) return;
    let sampleMap = null;
    let hoverEl = null;
    let hoverTimer = null;
    const show = async (a) => {
        const text = a.textContent || "";
        const m = text.match(/civitai_\d+_([0-9a-f]{8})/i);
        const hash = m ? m[1] : "";
        if (!hash) return;
        if (!sampleMap) {
            try {
                const data = await fetchSamplesCached(loraName);
                const imgs = Array.isArray(data.images) ? data.images : [];
                sampleMap = new Map();
                for (const p of imgs) {
                    const hm = p.match(/_([0-9a-f]{8})\./i);
                    if (hm) sampleMap.set(hm[1].toLowerCase(), p);
                }
            } catch { return; }
        }
        const rel = sampleMap.get(hash.toLowerCase());
        if (!rel) return;
        const isVideo = isVideoPath(rel);
        hoverEl = document.createElement("div");
        hoverEl.className = "sf-lora-sample-hover";
        hoverEl.style.cssText = "position:fixed;z-index:10060;background:#1e1e1e;border:1px solid #444;border-radius:8px;padding:6px;box-shadow:0 8px 24px rgba(0,0,0,0.6);pointer-events:none;";
        let media;
        if (isVideo) {
            media = document.createElement("video");
            media.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(rel)}`;
            media.autoplay = true;
            media.muted = true;
            media.loop = true;
        } else {
            media = document.createElement("img");
            media.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(rel)}&w=512`;
        }
        media.style.cssText = "max-width:320px;max-height:320px;border-radius:6px;display:block;";
        hoverEl.appendChild(media);
        document.body.appendChild(hoverEl);
        const rect = a.getBoundingClientRect();
        const vw = window.innerWidth, vh = window.innerHeight;
        let left = rect.right + 12;
        let top = rect.top;
        const hr = hoverEl.getBoundingClientRect();
        if (left + hr.width > vw - 8) left = Math.max(8, rect.left - hr.width - 12);
        if (top + hr.height > vh - 8) top = Math.max(8, vh - hr.height - 8);
        hoverEl.style.left = left + "px";
        hoverEl.style.top = top + "px";
    };
    const hide = () => {
        if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
        if (hoverEl) { hoverEl.remove(); hoverEl = null; }
    };
    for (const a of links) {
        a.addEventListener("mouseenter", () => { hoverTimer = setTimeout(() => show(a), 220); });
        a.addEventListener("mouseleave", hide);
        a.addEventListener("click", hide);
    }
}

// ── 描述内样例 prompt 复制按钮（civitai 标题紧邻的代码块右上角）────────────
// 后端 _format_sample_prompts 固定结构：### [civitai_NN_hash …] 后紧跟 ```
// 围栏代码块，renderMarkdown 渲染为相邻的 h3 + pre。复制内容取渲染时的
// pre.textContent（围栏原文，换行保留）。notify(msg) 由宿主接面板消息条。
const _PROMPT_COPY_CSS = `
.sf-ls-desc-copybtn { position:absolute; top:4px; right:4px; width:18px; height:18px;
  display:flex; align-items:center; justify-content:center; padding:0;
  border:1px solid #444; border-radius:4px; background:#242428;
  cursor:pointer; opacity:0.55; z-index:2; }
.sf-ls-desc-copybtn:hover { opacity:1; border-color:var(--acc, var(--sf-acc, #f66744)); background:#32302e; }
`;
export function attachSamplePromptCopyButtons(container, notify) {
    if (!container || !container.children) return;
    injectCSSOnce("sf-lora-desc-copybtn", _PROMPT_COPY_CSS);
    let prev = null;
    for (const child of container.children) {
        if (prev && prev.tagName === "H3" && child.tagName === "PRE"
            && /civitai_\d+_[0-9a-f]{8}/i.test(prev.textContent || "")) {
            _wireSamplePromptCopy(child, notify);
        }
        prev = child;
    }
}

function _wireSamplePromptCopy(pre, notify) {
    if (typeof pre.querySelector === "function" && pre.querySelector(".sf-ls-desc-copybtn")) return;
    const text = pre.textContent ?? "";
    pre.style.position = "relative";
    const btn = document.createElement("button");
    btn.className = "sf-ls-desc-copybtn";
    btn.title = "Copy this prompt to clipboard";
    btn.appendChild(makeSampleIcon(SAMPLE_ICON_PROMPT));
    btn.addEventListener("click", async (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        btn.style.opacity = "0.3";
        try {
            const ok = await copyText(text);
            notify?.(ok ? "Prompt copied to clipboard." : "Could not copy to clipboard.");
        } finally {
            btn.style.opacity = "";
        }
    });
    pre.appendChild(btn);
}
