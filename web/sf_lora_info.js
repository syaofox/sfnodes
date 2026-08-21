// ==========================================================================
// SF Model Info - Shared LoRA/model metadata dialog & fetch utilities
// Used by SFPowerLoraLoader and SFLoraLoaderModelOnly (and future nodes).
// ==========================================================================
import { app } from "/scripts/app.js";
import { renderMarkdown } from "./sf_markdown.js";
import { copyText } from "./sf_common.js";
// Civitai 查询/账户封装复用 SFLoraStack 同一套（同一 civitai.json 配置，
// 机器级共享）。该模块只依赖 sf_common.js，无 Stack 节点依赖。
import { loraInfo, civitaiLookup, deleteCivitai, saveCivitaiThumb,
    getCivitaiAccount, setCivitaiAccount, migrateLoraData } from "./sf_lora_stack_api.js";

// 样例图悬浮按钮 SVG（mask-image，与 Stack 面板统一样式）
const _SAMPLE_ICON_TRASH = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 3h6l1 2h4v2H4V5h5l1-2zm-2 6h10l-1 9a1 1 0 01-1 1H8a1 1 0 01-1-1L6 9zM10 11v6M14 11v6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";
const _SAMPLE_ICON_LOAD = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z' fill='black'/%3E%3C/svg%3E";
const _SAMPLE_ICON_PROMPT = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M9 5a2 2 0 002 2h6a2 2 0 002-2M9 5a2 2 0 012-2h4a2 2 0 012 2M9 12h6M9 16h6' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";

function _makeSampleIcon(url) {
    const s = document.createElement("span");
    s.style.cssText = "width:10px;height:10px;background-color:#fff;display:block;-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;";
    s.style.webkitMaskImage = `url("${url}")`;
    s.style.maskImage = `url("${url}")`;
    return s;
}

function _openSamplePreview(path, allPaths) {
    const list = Array.isArray(allPaths) && allPaths.length ? allPaths : [path];
    let idx = list.indexOf(path);
    if (idx < 0) idx = 0;
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.75);display:flex;align-items:center;justify-content:center;cursor:pointer;";
    let media = null;
    const render = (i) => {
        if (i < 0 || i >= list.length) return;
        idx = i;
        const p = list[idx];
        const isVideo = /\.(mp4|m4v|mov|webm|mkv)$/i.test(p);
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
    };
    const onKey = (e) => {
        if (e.key === "Escape") {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            // 标记由预览消费了 Esc，避免随后 dialog 的 cancel 关闭对话框
            try { document.body.dataset.sfPreviewEsc = "1"; setTimeout(() => { try { delete document.body.dataset.sfPreviewEsc; } catch {} }, 50); } catch {}
            close();
        }
        else if (e.key === "ArrowLeft") { if (idx > 0) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); render(idx - 1); } }
        else if (e.key === "ArrowRight") { if (idx < list.length - 1) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); render(idx + 1); } }
    };
    render(idx);
    overlay.addEventListener("click", close);
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(overlay);
}

function _attachSampleTitleHover(container, loraName) {
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
                const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples?filename=${encodeURIComponent(loraName)}`);
                if (!resp.ok) return;
                const data = await resp.json();
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
        const isVideo = /\.(mp4|m4v|mov|webm|mkv)$/i.test(rel);
        hoverEl = document.createElement("div");
        hoverEl.className = "sf-li-desc-hover";
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

// ---------------------------------------------------------------------------
// PNG 内嵌工作流解析（ComfyUI SaveImage 写入的 workflow/prompt chunk）
// 返回 { chunk: "workflow" | "prompt", data: string } 或 null
// ---------------------------------------------------------------------------
async function readPngWorkflowData(url) {
    let resp;
    try { resp = await fetch(url); } catch { return null; }
    if (!resp.ok) return null;
    const buf = await resp.arrayBuffer();
    const bytes = new Uint8Array(buf);
    // PNG 签名 89 50 4E 47 ...
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

// 将图片作为工作流载入：新建工作流标签页（不替换当前画布）
export async function loadImageAsWorkflow(path, onError) {
    const url = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}`;
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
        // 首选：新建工作流标签页载入，保留当前画布
        const cmd = app.extensionManager?.command;
        if (cmd && typeof cmd.execute === "function") {
            await cmd.execute("Comfy.NewBlankWorkflow");
            await load();
            return true;
        }
        // 兜底（旧版环境无命令系统）：替换当前画布，需确认
        if (!confirm("当前 ComfyUI 不支持新建标签，载入将替换当前画布内容，继续吗？")) return false;
        await load();
        return true;
    } catch (e) {
        console.warn("[SF Model Info] load workflow failed:", e);
        onError("工作流载入失败：" + (e.message || e));
        return false;
    }
}

// ---------------------------------------------------------------------------
// Metadata fetch (merged custom notes + embedded safetensors metadata via
// the /api/sfnodes/lora_notes gateway endpoint)
// ---------------------------------------------------------------------------

export const loraMetadataCache = new Map();
const _loraMetadataPending = new Map();

// 2026-08 统一存储：元数据读写走 /api/sfnodes/lora_notes（后端网关，与
// SFLoraStack 同一 lora_triggers.json 真源）。`force` 跳过缓存与在途去重，
// 打开对话框时用（另一节点刚保存过，缓存可能陈旧）。
export async function getLoraMetadata(name, force = false) {
    if (!name || name === "None") return null;
    if (!force && loraMetadataCache.has(name)) return loraMetadataCache.get(name);
    // Join an in-flight request instead of firing a duplicate
    if (!force && _loraMetadataPending.has(name)) return _loraMetadataPending.get(name);

    const promise = (async () => {
        try {
            // force（打开对话框）= 必新：no-store 越过浏览器启发式缓存
            // （后端响应无 Cache-Control，默认模式可能命中陈旧副本）。
            const resp = await fetch(`/api/sfnodes/lora_notes?filename=${encodeURIComponent(name)}`,
                { cache: force ? "no-store" : "default" });
            if (!resp.ok) { loraMetadataCache.set(name, null); return null; }
            const meta = await resp.json();
            loraMetadataCache.set(name, meta);
            return meta;
        } catch {
            loraMetadataCache.set(name, null);
            return null;
        }
    })();

    if (!force) {
        _loraMetadataPending.set(name, promise);
        try { return await promise; }
        finally { _loraMetadataPending.delete(name); }
    }
    return promise;
}

// ── 跨节点缓存失效：任一节点（Power 系对话框 / SFLoraStack 面板）保存
// LoRA 用户数据后广播，两端各自清自己模块的缓存，下次打开即新数据。────
if (typeof document !== "undefined") {
    document.addEventListener("sfnodes.lora-data-changed", (e) => {
        const name = e?.detail?.name;
        if (name) loraMetadataCache.delete(name);
    });
}

// ---------------------------------------------------------------------------
// Info dialog (native <dialog> modal, like rgthree)
// ---------------------------------------------------------------------------

export function showLoraInfoDialog(event, name, meta) {
    meta = meta || {};
    const state = {
        trigger_words: meta.trigger_words || "",
        description: meta.description || "",
    };

    // ── Civitai 查询状态（与 SFLoraStack 面板同语义，见 sf_lora_stack_info.js）──
    let civ = null;            // { state:"searching"|"found"|"nofind"|"offline", info?, message?, note? }
    let hasSidecar = false;    // .civitai.info 侧车存在（fire-and-forget 探测，控制 🗑 按钮）
    let _thumbBust = 0;        // 封面 bust：越过缩略图路由的一小时缓存
    let _acc = null;           // Civitai 账户公开形状 {configured,hint,host,adultThumbs}（key 永不回页）
    let _accBusy = false;      // 账户保存防重入

    // ---------- dialog (native modal, like rgthree) ----------
    if (!showLoraInfoDialog._cssInjected) {
        showLoraInfoDialog._cssInjected = true;
        const style = document.createElement("style");
        style.textContent = `
            dialog.sf-lora-info::backdrop { background: rgba(0,0,0,0.5); }
            /* ── Civitai 查询状态条（与 SFLoraStack 面板四态同语义） ── */
            .sf-li-civstrip { display:flex; align-items:flex-start; gap:8px; margin:10px 18px 2px;
                padding:8px 10px; border-radius:6px; font-size:11px; line-height:1.5; }
            .sf-li-civstrip .ic { flex:0 0 auto; }
            .sf-li-civstrip.searching { background:rgba(121,170,255,0.10); border:1px solid rgba(121,170,255,0.35); color:#9db8e8; }
            .sf-li-civstrip.found { background:rgba(62,195,113,0.10); border:1px solid rgba(62,195,113,0.4); color:#8fce9f; }
            .sf-li-civstrip.nofind { background:rgba(255,193,7,0.10); border:1px solid rgba(255,193,7,0.35); color:#e0c27a; }
            .sf-li-civstrip.archive { background:rgba(255,193,7,0.10); border:1px solid rgba(200,160,30,0.5); color:#e8c877; }
            .sf-li-civstrip.offline { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.18); color:#b0b0b0; }
            .sf-li-archive-row { display:flex; gap:6px; margin-top:8px; align-items:center; }
            .sf-li-archive-row input { flex:1; min-width:0; background:#1a1a1e; border:1px solid #444; border-radius:5px; color:#ddd; font:11px 'Segoe UI'; padding:5px 8px; outline:none; }
            .sf-li-archive-row input:focus { border-color:#6af; }
            .sf-li-archive-row button { background:#4f7cff; border:1px solid #4f7cff; color:#fff; border-radius:5px; padding:5px 10px; font:11px 'Segoe UI'; cursor:pointer; font-weight:600; }
            .sf-li-civstrip .civlink { color:#8fc0ff; cursor:pointer; }
            .sf-li-civstrip .civlink:hover { color:#b8d8ff; text-decoration:underline; }
            .sf-li-spin { display:inline-block; width:10px; height:10px; border:2px solid rgba(255,255,255,0.25);
                border-top-color:#9db8e8; border-radius:50%; animation:sf-li-spin .7s linear infinite; vertical-align:-1px; }
            @keyframes sf-li-spin { to { transform:rotate(360deg); } }
            /* ── Civitai 账户设置区（同一份 civitai.json，与 Stack 共享） ── */
            .sf-li-acc { margin:0 18px 10px; border:1px solid #444; border-radius:8px; padding:10px 12px; }
            .sf-li-acc-head { font-size:12px; font-weight:600; color:#eee; }
            .sf-li-acc-sub { font-size:10.5px; color:#8a8a8a; line-height:1.5; margin-top:3px; }
            .sf-li-acc-row { display:flex; align-items:center; gap:8px; margin-top:9px; font-size:11.5px; color:#ccc; }
            .sf-li-acc-row .lab { flex:1; min-width:0; }
            .sf-li-acc-mini { flex:0 0 auto; font-size:11px; color:#8fc0ff; cursor:pointer;
                border:1px solid #3a5a80; border-radius:4px; padding:2px 8px; }
            .sf-li-acc-mini:hover { color:#b8d8ff; border-color:#5a7ab0; }
            .sf-li-acc-mini.rm { color:#c9736a; border-color:#6a4038; }
            .sf-li-acc-mini.rm:hover { color:#e0604a; border-color:#8a4a40; }
            .sf-li-acc-key { flex:1; min-width:0; background:#1a1a1e; border:1px solid #6af; border-radius:5px;
                color:#eee; font-size:12px; padding:5px 8px; outline:none; }
            .sf-li-acc-seg { display:flex; gap:4px; }
            .sf-li-acc-segb { font-size:11px; color:#aaa; border:1px solid #555; border-radius:4px; padding:3px 10px; cursor:pointer; }
            .sf-li-acc-segb:hover { color:#ddd; }
            .sf-li-acc-segb.on { color:#fff; border-color:var(--sf-acc, #f66744); background:rgba(246,103,68,0.12); }
            .sf-li-acc-sw { width:30px; height:16px; border-radius:9px; background:#555; position:relative; cursor:pointer; flex:0 0 auto; }
            .sf-li-acc-sw::after { content:""; position:absolute; top:2px; left:2px; width:12px; height:12px;
                border-radius:50%; background:#ccc; transition:left .12s; }
            .sf-li-acc-sw.on { background:rgba(246,103,68,0.7); }
            .sf-li-acc-sw.on::after { left:16px; background:#fff; }
            .sf-li-acc-msg { font-size:10.5px; margin-top:6px; display:none; }
            .sf-li-acc-msg.ok { color:#3ec371; }
            .sf-li-desc-hover { position:fixed; z-index:10060; background:#1e1e1e; border:1px solid #444; border-radius:8px; padding:6px; box-shadow:0 8px 24px rgba(0,0,0,0.6); pointer-events:none; }
            .sf-li-desc-hover img, .sf-li-desc-hover video { max-width:320px; max-height:320px; border-radius:6px; display:block; }
        `;
        document.head.appendChild(style);
    }

    const dialog = document.createElement("dialog");
    dialog.className = "sf-lora-info";
    dialog.style.cssText = `
        background: #2a2a2e; border: 1px solid #555; border-radius: 10px;
        min-width: 560px; max-width: 720px; max-height: 92vh;
        padding: 0; overflow: hidden; color: #ddd;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;

    const card = document.createElement("div");
    card.style.cssText = `
        display: flex; flex-direction: column; max-height: 92vh;
    `;

    // ---------- header ----------
    const header = document.createElement("div");
    header.style.cssText = `
        display: flex; align-items: center; justify-content: space-between;
        gap: 12px; padding: 14px 18px; border-bottom: 1px solid #444;
    `;
    const title = document.createElement("div");
    title.textContent = name;
    title.title = name;
    title.style.cssText = `
        flex: 1 1 auto; min-width: 0;
        font-size: 13px; font-weight: 600; color: #fff;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    `;
    // ---------- 封面缩略图（只读展示） ----------
    // 与 SFLoraStack 共用 /api/sfnodes/lora_thumb（用户自定义预览 > 模型旁
    // .preview 图）。Stack 面板换封面后本对话框打开即新图：URL 带时间戳
    // bust 越过缩略图路由的一小时缓存（Stack 面板的 thumbUrl 同款机制）。
    // 无图时 404 -> onerror 隐藏，不占布局。
    const thumbEl = document.createElement("img");
    thumbEl.alt = "";
    thumbEl.style.cssText = `
        width: 44px; height: 44px; border-radius: 6px; object-fit: cover;
        border: 1px solid #444; flex: 0 0 auto; display: none;
    `;
    if (name && name !== "None") {
        thumbEl.onload = () => { thumbEl.style.display = "block"; };
        thumbEl.onerror = () => { thumbEl.style.display = "none"; };
        thumbEl.src = `/api/sfnodes/lora_thumb?name=${encodeURIComponent(name)}&t=${Date.now()}`;
    }
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.title = "Close";
    closeBtn.style.cssText = `
        flex: 0 0 auto; background: none; border: none; cursor: pointer;
        font-size: 14px; color: #aaa; padding: 2px 6px; border-radius: 4px;
    `;
    closeBtn.addEventListener("mouseenter", () => { closeBtn.style.color = "#fff"; });
    closeBtn.addEventListener("mouseleave", () => { closeBtn.style.color = "#aaa"; });
    closeBtn.addEventListener("click", () => closeDialog());
    header.appendChild(thumbEl);
    header.appendChild(title);
    header.appendChild(closeBtn);

    // ---------- body ----------
    const body = document.createElement("div");
    body.style.cssText = "overflow-y: auto; padding: 6px 0;";

    // ---------- 孤儿/文件缺失提示（改名/移动后数据在旧路径 key 下） ----------
    // 文件不存在（旧路径行）：数据在旧 key 下，无法迁移（迁移端点需文件
    // 存在）——提示用户重新选择 LoRA 路径。
    if (meta._file_missing && meta.orphan_key && name) {
        const strip = document.createElement("div");
        strip.style.cssText = `
            margin: 8px 18px 4px; padding: 8px 10px; border-radius: 6px;
            background: rgba(255, 193, 7, 0.12); border: 1px solid rgba(255, 193, 7, 0.4);
            font-size: 11px; color: #e8c877; line-height: 1.5;
        `;
        strip.textContent = `该 LoRA 文件已被移动或改名，数据仍保存在旧路径下（${meta.orphan_key}）。请在节点上重新选择该 LoRA 以读取。`;
        body.appendChild(strip);
    }
    // 孤儿数据迁移提示（文件被移动/改名后旧键数据仍在，本文件可读取）：
    // 与 SFLoraStack 面板同机制——Migrate 调 /api/sfnodes/lora/migrate 把
    // 旧键下的词/描述/预览图迁到当前文件；Dismiss 本次打开隐藏。明细
    // （orphan_triggers/orphan_description/orphan_preview）由后端
    // get_merged_metadata 孤儿命中时附带。
    let orphanStrip = null;
    if (meta.orphan_key && !meta._file_missing && name) {
        orphanStrip = document.createElement("div");
        orphanStrip.style.cssText = `
            display: flex; align-items: flex-start; gap: 10px;
            margin: 8px 18px 4px; padding: 8px 10px; border-radius: 6px;
            background: rgba(255, 193, 7, 0.12); border: 1px solid rgba(255, 193, 7, 0.4);
            font-size: 11px; color: #e8c877; line-height: 1.5;
        `;
        const parts = [];
        if ((meta.orphan_triggers?.length || 0) > 0) parts.push(meta.orphan_triggers.length + " 个触发词");
        if (meta.orphan_description) parts.push("描述");
        if (meta.orphan_preview) parts.push("预览图");
        const txt = document.createElement("div");
        txt.style.cssText = "flex: 1; min-width: 0;";
        txt.textContent = `检测到该 LoRA 在旧路径（${meta.orphan_key}）下保存的数据（${parts.join("、") || "自定义数据"}）。迁移到当前文件？`;
        const acts = document.createElement("div");
        acts.style.cssText = "flex: 0 0 auto; display: flex; gap: 6px;";
        const mig = document.createElement("button");
        mig.textContent = "迁移";
        mig.title = "把旧路径下的触发词、描述和预览图迁到当前文件";
        mig.style.cssText = "padding: 3px 10px; border: 1px solid #4f7cff; border-radius: 5px; font-size: 11px; cursor: pointer; color: #4f7cff; background: transparent;";
        mig.addEventListener("mouseenter", () => { mig.style.background = "rgba(79,124,255,0.15)"; });
        mig.addEventListener("mouseleave", () => { mig.style.background = ""; });
        mig.addEventListener("click", () => runMigrate());
        const dis = document.createElement("button");
        dis.textContent = "忽略";
        dis.title = "本次打开不提示（数据保留在旧路径下）";
        dis.style.cssText = "padding: 3px 10px; border: 1px solid #777; border-radius: 5px; font-size: 11px; cursor: pointer; color: #aaa; background: transparent;";
        dis.addEventListener("mouseenter", () => { dis.style.background = "#3a3a3e"; });
        dis.addEventListener("mouseleave", () => { dis.style.background = ""; });
        dis.addEventListener("click", () => orphanStrip.remove());
        acts.append(mig, dis);
        orphanStrip.append(txt, acts);
        body.appendChild(orphanStrip);
    }

    // 把旧路径键下的自定义数据（词/描述/预览图）迁移到当前 LoRA 名。
    // 成功后移除提示条 + force 重取合并元数据（applyMetaRefresh 更新词/描述
    // 行，orphan 字段消失）；失败就地替换条内文本（保留条，可关闭重来）。
    async function runMigrate() {
        const res = await migrateLoraData(name, meta.orphan_key);
        if (!dialog.isConnected) return;
        if (!res?.ok) {
            if (orphanStrip) orphanStrip.textContent = "迁移失败：" + ((res && res.message) || "未知错误");
            return;
        }
        orphanStrip?.remove();
        const meta2 = await getLoraMetadata(name, true);
        if (!dialog.isConnected) return;
        if (meta2 && !meta2._not_found) applyMetaRefresh(meta2);
        app.graph.setDirtyCanvas(true, true);
    }

    // row factory: editable rows
    function createEditRow(displayLabel, key, isTextarea, hint) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex; align-items: flex-start; gap: 10px;
            padding: 10px 18px; border-bottom: 1px solid #3a3a3e;
        `;
        const labelEl = document.createElement("div");
        labelEl.style.cssText = `
            flex: 0 0 100px; font-size: 12px; color: #aaa;
            padding-top: 5px; line-height: 1.4;
        `;
        labelEl.textContent = displayLabel;
        if (hint) labelEl.title = hint;
        const valueEl = document.createElement("div");
        valueEl.style.cssText = `
            flex: 1; font-size: 13px; color: #eee; line-height: 1.5;
            white-space: pre-wrap; word-break: break-word; min-height: 20px;
        `;
        const actionEl = document.createElement("div");
        actionEl.style.cssText = "flex: 0 0 auto; display: flex; gap: 4px; align-items: center;";
        row.appendChild(labelEl);
        row.appendChild(valueEl);
        row.appendChild(actionEl);

        function renderValue() {
            valueEl.innerHTML = "";
            const v = state[key];
            if (!v) {
                valueEl.style.whiteSpace = "pre-wrap";
                valueEl.innerHTML = '<span style="color:#666;">(empty)</span>';
            } else if (key === "description") {
                // Description 支持 Markdown：查看态渲染，编辑态编辑源码
                // 相对路径（sample/xxx.png）按当前 lora 目录解析，目录改名后自动跟随
                valueEl.style.whiteSpace = "normal";
                valueEl.innerHTML = renderMarkdown(v, { resolveRelative: resolveNoteRelativeUrl });
                // 标题悬停预览对应 sample 图（与下方网格同源）
                queueMicrotask(() => _attachSampleTitleHover(valueEl, name));
            } else {
                valueEl.style.whiteSpace = "pre-wrap";
                valueEl.textContent = v;
            }
            if (key === "description") valueEl.removeAttribute("title");
            else valueEl.title = v;
        }

        function renderActions() {
            actionEl.innerHTML = "";
            // 触发词行增加复制按钮（SVG，与样例图按钮统一）
            if (key === "trigger_words") {
                const cbtn = document.createElement("button");
                cbtn.title = "复制全部触发词到剪贴板";
                cbtn.style.cssText = `
                    background: none; border: 1px solid #555; border-radius: 4px;
                    cursor: pointer; color: #bbb; padding: 3px 6px; display: flex; align-items: center; justify-content: center;
                `;
                const cic = _makeSampleIcon(_SAMPLE_ICON_PROMPT);
                cic.style.backgroundColor = "#bbb";
                cbtn.appendChild(cic);
                cbtn.addEventListener("mouseenter", () => { cbtn.style.background = "#3a3a3e"; cic.style.backgroundColor = "#fff"; cbtn.style.borderColor = "#6af"; });
                cbtn.addEventListener("mouseleave", () => { cbtn.style.background = ""; cic.style.backgroundColor = "#bbb"; cbtn.style.borderColor = "#555"; });
                cbtn.addEventListener("click", async () => {
                    const v = state[key];
                    if (!v || !String(v).trim()) {
                        // 空状态给提示（复用 sampleHint 区域或临时标题）
                        const hint = document.createElement("div");
                        hint.textContent = "没有可复制的触发词。";
                        hint.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);background:#2a2a2e;color:#ddd;border:1px solid #555;border-radius:6px;padding:6px 10px;font-size:12px;z-index:99999;";
                        document.body.appendChild(hint);
                        setTimeout(() => hint.remove(), 1500);
                        return;
                    }
                    const ok = await copyText(String(v));
                    const hint = document.createElement("div");
                    hint.textContent = ok ? "已复制触发词到剪贴板。" : "复制失败。";
                    hint.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);background:#2a2a2e;color:#ddd;border:1px solid #555;border-radius:6px;padding:6px 10px;font-size:12px;z-index:99999;";
                    document.body.appendChild(hint);
                    setTimeout(() => hint.remove(), 1500);
                });
                actionEl.appendChild(cbtn);
            }
            const btn = document.createElement("button");
            btn.title = "Edit " + displayLabel;
            btn.style.cssText = `
                background: none; border: 1px solid #555; border-radius: 4px;
                cursor: pointer; color: #bbb; padding: 3px 6px; display: flex; align-items: center; justify-content: center;
            `;
            const _editIconUrl = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M11 4H4a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-5M18 2l3 3-9 9H9v-3l9-9z' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";
            const eic = document.createElement("span");
            eic.style.cssText = "width:12px;height:12px;background-color:#bbb;display:block;-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;";
            eic.style.webkitMaskImage = `url("${_editIconUrl}")`;
            eic.style.maskImage = `url("${_editIconUrl}")`;
            btn.appendChild(eic);
            btn.addEventListener("mouseenter", () => { btn.style.background = "#3a3a3e"; eic.style.backgroundColor = "#fff"; btn.style.borderColor = "#6af"; });
            btn.addEventListener("mouseleave", () => { btn.style.background = ""; eic.style.backgroundColor = "#bbb"; btn.style.borderColor = "#555"; });
            btn.addEventListener("click", () => startEdit());
            actionEl.appendChild(btn);
        }

        function startEdit() {
            const input = isTextarea ? document.createElement("textarea") : document.createElement("input");
            input.value = state[key];
            if (isTextarea) {
                input.rows = 12;
                input.style.resize = "vertical";
            }
            if (key === "description") {
                input.placeholder = "支持 Markdown：**加粗**、[链接](url)、列表、代码块；下方示例图点击即可插入";
            }
            input.style.cssText = `
                width: 100%; box-sizing: border-box;
                background: #1a1a1e; color: #eee; border: 1px solid #6af;
                border-radius: 6px; padding: 6px 8px; font-size: 13px;
                font-family: inherit; outline: none;
            `;
            valueEl.innerHTML = "";
            valueEl.appendChild(input);
            actionEl.innerHTML = "";
            row._editing = true;
            row._baseValue = state[key];
            row._dirty = false;
            renderFooterActions();

            // 描述行：仅保留上传按钮；示例图面板默认展开；保存/取消移至底部按钮栏
            if (key === "description") {
                actionEl.style.flexDirection = "column";
                if (name && name !== "None") {
                    const uploadBtn = document.createElement("button");
                    uploadBtn.title = "上传图片到该 LoRA 的 sample 目录并插入";
                    uploadBtn.style.cssText = `
                        background: none; border: 1px solid #555; border-radius: 4px;
                        cursor: pointer; color: #bbb; padding: 3px 6px; display: flex; align-items: center; justify-content: center;
                    `;
                    const _upIconUrl = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 16V3M8 7l4-4 4 4M3 17v4a2 2 0 002 2h14a2 2 0 002-2v-4' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E";
                    const _upIc = document.createElement("span");
                    _upIc.style.cssText = "width:12px;height:12px;background-color:#bbb;display:block;-webkit-mask-size:contain;mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-position:center;mask-position:center;";
                    _upIc.style.webkitMaskImage = `url("${_upIconUrl}")`;
                    _upIc.style.maskImage = `url("${_upIconUrl}")`;
                    uploadBtn.appendChild(_upIc);
                    uploadBtn.addEventListener("mouseenter", () => { _upIc.style.backgroundColor = "#fff"; uploadBtn.style.borderColor = "#6af"; });
                    uploadBtn.addEventListener("mouseleave", () => { _upIc.style.backgroundColor = "#bbb"; uploadBtn.style.borderColor = "#555"; });
                    uploadBtn.addEventListener("click", () => uploadInput.click());
                    actionEl.appendChild(uploadBtn);
                    openSamplePanel(input);
                }
            } else {
                actionEl.style.flexDirection = "";
            }

            // 内容改动追踪：变化时刷新底部按钮（Save 加 *）
            input.addEventListener("input", () => {
                const dirty = input.value !== row._baseValue;
                if (dirty !== row._dirty) {
                    row._dirty = dirty;
                    renderFooterActions();
                }
            });

            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !isTextarea) {
                    e.preventDefault();
                    saveEdit();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    cancelEdit();
                }
            });
            input.focus();
            input.select();
        }

        // 提交当前编辑（若处于编辑态），返回是否提交过
        row.commitEdit = function () {
            const input = valueEl.querySelector("input,textarea");
            if (!input) return false;
            state[key] = input.value.trim();
            row._editing = false;
            row._dirty = false;
            actionEl.style.flexDirection = "";
            closeSamplePanel();
            renderValue();
            renderActions();
            renderFooterActions();
            return true;
        };

        function saveEdit() {
            if (row.commitEdit()) saveNotes();
        }

        function cancelEdit() {
            if (row._editing) {
                // 放弃修改：恢复进入编辑前的基准值（防御其他路径中途写入 state）
                state[key] = row._baseValue;
            }
            row._editing = false;
            row._dirty = false;
            actionEl.style.flexDirection = "";
            closeSamplePanel();
            renderValue();
            renderActions();
            renderFooterActions();
        }

        row.cancelEdit = cancelEdit;

        row.refresh = function () {
            renderValue();
            renderActions();
        };

        renderValue();
        renderActions();
        return row;
    }

    // read-only row factory（row.valueEl 供 Civitai 查询后刷新复用）
    function renderReadonlyValue(row, value, linkUrl) {
        row.valueEl.innerHTML = "";
        if (linkUrl && value) {
            const a = document.createElement("a");
            a.href = linkUrl;
            a.target = "_blank";
            a.rel = "noopener";
            a.textContent = value;
            a.style.cssText = "color: #7aa2ff; text-decoration: none; word-break: break-all;";
            a.addEventListener("mouseenter", () => { a.style.textDecoration = "underline"; });
            a.addEventListener("mouseleave", () => { a.style.textDecoration = ""; });
            row.valueEl.appendChild(a);
        } else {
            row.valueEl.textContent = value || "";
            if (!value) row.valueEl.innerHTML = '<span style="color:#666;">(empty)</span>';
        }
    }

    function createReadonlyRow(displayLabel, value, linkUrl) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex; align-items: flex-start; gap: 10px;
            padding: 10px 18px; border-bottom: 1px solid #3a3a3e;
        `;
        const labelEl = document.createElement("div");
        labelEl.style.cssText = `
            flex: 0 0 100px; font-size: 12px; color: #aaa; padding-top: 5px;
        `;
        labelEl.textContent = displayLabel;
        const valueEl = document.createElement("div");
        valueEl.style.cssText = `
            flex: 1; font-size: 13px; color: #eee; line-height: 1.5;
            white-space: pre-wrap; word-break: break-word; min-height: 20px;
        `;
        row.appendChild(labelEl);
        row.appendChild(valueEl);
        row.valueEl = valueEl;
        renderReadonlyValue(row, value, linkUrl);
        return row;
    }

    // ---------- build rows ----------
    const twRow = createEditRow("Trigger Words", "trigger_words", false);
    const descRow = createEditRow(
        "Description",
        "description",
        true,
        "支持 Markdown 格式：![图片](url)、[链接](url)、**加粗**、列表、代码块等"
    );
    // Civitai 查询后 base_model/source_url 可能变化（侧车信息）：保存行引用供刷新
    let bmRow = null, urlRow = null;
    body.appendChild(twRow);
    body.appendChild(descRow);
    if (meta.base_model) { bmRow = createReadonlyRow("Base Model", meta.base_model); body.appendChild(bmRow); }
    if (meta.source_url) { urlRow = createReadonlyRow("Source URL", meta.source_url, meta.source_url); body.appendChild(urlRow); }

    // ---------- sample images（描述 Markdown 图片插入） ----------
    const samplePanel = document.createElement("div");
    samplePanel.style.cssText = `
        display: none; padding: 10px 18px; border-bottom: 1px solid #3a3a3e;
    `;
    const samplePanelHead = document.createElement("div");
    samplePanelHead.style.cssText = "display:flex;align-items:center;justify-content:space-between;font-size:12px;color:#aaa;";
    const samplePanelTitle = document.createElement("span");
    samplePanelTitle.textContent = "LoRA 示例图（点击插入）";
    const sampleCloseBtn = document.createElement("button");
    sampleCloseBtn.textContent = "✕";
    sampleCloseBtn.title = "关闭";
    sampleCloseBtn.style.cssText = "background:none;border:none;cursor:pointer;font-size:12px;color:#aaa;padding:0 4px;";
    sampleCloseBtn.addEventListener("click", () => closeSamplePanel());
    samplePanelHead.appendChild(samplePanelTitle);
    samplePanelHead.appendChild(sampleCloseBtn);
    const sampleGrid = document.createElement("div");
    sampleGrid.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;max-height:220px;overflow-y:auto;";
    const sampleHint = document.createElement("div");
    sampleHint.style.cssText = "font-size:12px;color:#888;margin-top:8px;line-height:1.5;";
    samplePanel.appendChild(samplePanelHead);
    samplePanel.appendChild(sampleGrid);
    samplePanel.appendChild(sampleHint);
    body.appendChild(samplePanel);

    // ---------- sample images 浏览区（常驻底部，空则隐藏，预览大图） ----------
    const browsePanel = document.createElement("div");
    browsePanel.style.cssText = "display:none; padding:10px 18px; border-top:1px solid #3a3a3e;";
    const browseHead = document.createElement("div");
    browseHead.style.cssText = "font:600 9.5px 'Segoe UI'; text-transform:uppercase; letter-spacing:.7px; color:var(--sf-acc, #f66744); margin-bottom:8px;";
    browseHead.textContent = "Sample images";
    const browseGrid = document.createElement("div");
    browseGrid.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;max-height:220px;overflow-y:auto;";
    const browseHint = document.createElement("div");
    browseHint.style.cssText = "font-size:12px;color:#888;margin-top:8px;line-height:1.5;";
    browsePanel.appendChild(browseHead);
    browsePanel.appendChild(browseGrid);
    browsePanel.appendChild(browseHint);
    body.appendChild(browsePanel);

    async function refreshBrowseSamplePanel() {
        browseGrid.innerHTML = "";
        browseHint.textContent = "";
        if (!name || name === "None") { browsePanel.style.display = "none"; return; }
        try {
            const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples?filename=${encodeURIComponent(name)}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const imgs = Array.isArray(data.images) ? data.images : [];
            if (!imgs.length) { browsePanel.style.display = "none"; return; }
            browsePanel.style.display = "";
            for (const path of imgs) {
                const wrap = document.createElement("div");
                wrap.style.cssText = "position:relative;width:96px;height:96px;";
                const isVideo = /\.(mp4|m4v|mov|webm|mkv)$/i.test(path);
                let thumb;
                if (isVideo) {
                    thumb = document.createElement("img");
                    thumb.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}&w=256`;
                    thumb.title = path.split("/").pop() + " (video) — 点击预览";
                    thumb.loading = "lazy";
                    thumb.style.cssText = "width:96px;height:96px;object-fit:cover;border-radius:6px;border:1px solid #3a3a3e;cursor:pointer;display:block;";
                    thumb.addEventListener("mouseenter", () => { thumb.style.borderColor = "#6af"; });
                    thumb.addEventListener("mouseleave", () => { thumb.style.borderColor = "#3a3a3e"; });
                    thumb.addEventListener("error", () => {
                        if (thumb.dataset.fallback) return;
                        thumb.dataset.fallback = "1";
                        const fb = document.createElement("div");
                        fb.textContent = path.split("/").pop().slice(0, 14);
                        fb.title = path.split("/").pop() + " (video) — 点击预览";
                        fb.style.cssText = "width:96px;height:96px;border-radius:6px;border:1px solid #3a3a3e;display:flex;align-items:center;justify-content:center;text-align:center;background:#1c1c1e;color:#888;font-size:9px;word-break:break-all;cursor:pointer;padding:4px;box-sizing:border-box;";
                        fb.addEventListener("click", () => _openSamplePreview(path, imgs));
                        thumb.replaceWith(fb);
                        thumb = fb;
                    });
                } else {
                    thumb = document.createElement("img");
                    thumb.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}&w=256`;
                    thumb.title = path.split("/").pop() + " — 点击预览";
                    thumb.loading = "lazy";
                    thumb.style.cssText = "width:96px;height:96px;object-fit:cover;border-radius:6px;border:1px solid #3a3a3e;cursor:pointer;display:block;";
                    thumb.addEventListener("mouseenter", () => { thumb.style.borderColor = "#6af"; });
                    thumb.addEventListener("mouseleave", () => { thumb.style.borderColor = "#3a3a3e"; });
                }
                if (!thumb.dataset.fallback) thumb.addEventListener("click", () => _openSamplePreview(path, imgs));
                const delBtn = document.createElement("button");
                delBtn.title = "删除该示例图";
                delBtn.style.cssText = "position:absolute;top:0;right:0;display:none;align-items:center;justify-content:center;width:18px;height:18px;padding:0;line-height:1;background:rgba(224,108,108,0.9);color:#fff;border:none;border-radius:0 6px 0 6px;cursor:pointer;";
                delBtn.appendChild(_makeSampleIcon(_SAMPLE_ICON_TRASH));
                const loadBtn = document.createElement("button");
                loadBtn.title = "将该图片载入为工作流（需内嵌工作流数据）";
                loadBtn.style.cssText = "position:absolute;bottom:0;right:0;display:none;align-items:center;justify-content:center;width:18px;height:18px;padding:0;line-height:1;background:rgba(79,124,255,0.9);color:#fff;border:none;border-radius:6px 0 6px 0;cursor:pointer;";
                loadBtn.appendChild(_makeSampleIcon(_SAMPLE_ICON_LOAD));
                const promptBtn = document.createElement("button");
                promptBtn.title = "复制该图片的 prompt 到剪贴板";
                promptBtn.style.cssText = "position:absolute;bottom:0;left:0;display:none;align-items:center;justify-content:center;width:18px;height:18px;padding:0;line-height:1;background:rgba(46,160,90,0.92);color:#fff;border:none;border-radius:0 6px 0 5px;cursor:pointer;";
                promptBtn.appendChild(_makeSampleIcon(_SAMPLE_ICON_PROMPT));
                wrap.addEventListener("mouseenter", () => { delBtn.style.display = "flex"; loadBtn.style.display = "flex"; promptBtn.style.display = "flex"; });
                wrap.addEventListener("mouseleave", () => { delBtn.style.display = "none"; loadBtn.style.display = "none"; promptBtn.style.display = "none"; });
                loadBtn.addEventListener("click", async (e) => { e.stopPropagation(); await loadImageAsWorkflow(path, (msg) => { browseHint.textContent = msg; }); });
                promptBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    promptBtn.style.opacity = "0.5"; promptBtn.style.pointerEvents = "none";
                    try {
                        const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples/prompt?path=${encodeURIComponent(path)}`);
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);
                        if (!data.found || !data.text) { browseHint.textContent = data.message || "该图片未包含 prompt。"; return; }
                        const ok = await copyText(data.text);
                        browseHint.textContent = ok ? "已复制 prompt 到剪贴板。" : "复制失败。";
                        if (ok) setTimeout(() => { if (browseHint.textContent === "已复制 prompt 到剪贴板。") browseHint.textContent = ""; }, 2000);
                    } catch (err) { browseHint.textContent = "读取失败：" + (err.message || err); } finally { promptBtn.style.opacity = ""; promptBtn.style.pointerEvents = ""; }
                });
                delBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const fileName = path.split("/").pop();
                    if (!confirm(`删除示例图「${fileName}」？此操作不可恢复。`)) return;
                    try {
                        const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples?path=${encodeURIComponent(path)}`, { method: "DELETE" });
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        refreshSamplePanel();
                        refreshBrowseSamplePanel();
                    } catch (err) { browseHint.textContent = "删除失败：" + (err.message || err); }
                });
                wrap.appendChild(thumb);
                wrap.appendChild(delBtn);
                wrap.appendChild(loadBtn);
                wrap.appendChild(promptBtn);
                browseGrid.appendChild(wrap);
            }
        } catch (e) { browseHint.textContent = "获取示例图失败：" + (e.message || e); }
    }
    // 初次打开即加载浏览区
    refreshBrowseSamplePanel();

    let sampleOpen = false;
    let activeTextarea = null;

    const uploadInput = document.createElement("input");
    uploadInput.type = "file";
    uploadInput.accept = "image/*,video/*";
    uploadInput.style.display = "none";
    dialog.appendChild(uploadInput);

    function buildSampleMarkdown(path) {
        const base = path.split("/").pop() || "image";
        const alt = base.replace(/\.[^.]+$/, "");
        // 相对 lora 目录的路径：目录改名/移动后，渲染时按当前 lora 路径解析，无需修复
        const rel = `sample/${encodeURIComponent(base)}`;
        return `![${alt}](${rel})`;
    }

    // 把描述中的相对路径解析为 sample 图片绝对 URL（基于当前 lora 路径）
    function resolveNoteRelativeUrl(rel) {
        let r = rel;
        try { r = decodeURIComponent(rel); } catch { /* 保留原样 */ }
        const idx = name.lastIndexOf("/");
        const dir = idx === -1 ? "" : name.slice(0, idx + 1);
        return `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(dir + r)}`;
    }

    function insertAtCursor(textarea, text) {
        const start = textarea.selectionStart ?? textarea.value.length;
        const end = textarea.selectionEnd ?? start;
        textarea.setRangeText(text, start, end, "end");
        // 不直接写 state：提交（Save）时统一从输入框读取，取消时保持原始内容
        textarea.focus();
        const pos = start + text.length;
        textarea.selectionStart = textarea.selectionEnd = pos;
    }

    async function refreshSamplePanel() {
        sampleGrid.innerHTML = "";
        sampleHint.textContent = "";
        if (!name || name === "None") {
            sampleHint.textContent = "当前未选择 LoRA。";
            return;
        }
        try {
            const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples?filename=${encodeURIComponent(name)}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (!Array.isArray(data.images) || !data.images.length) {
                sampleHint.textContent = `该 LoRA 没有示例图。请将图片放入 models/loras/${data.sample_dir || ""} 目录，或点击“上传”。`;
                return;
            }
            for (const path of data.images) {
                const wrap = document.createElement("div");
                wrap.style.cssText = "position:relative;width:96px;height:96px;";
                const isVideo = /\.(mp4|m4v|mov|webm|mkv)$/i.test(path);
                let thumb;
                if (isVideo) {
                    thumb = document.createElement("img");
                    thumb.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}&w=256`;
                    thumb.title = path.split("/").pop() + " (video) — 点击插入引用";
                    thumb.loading = "lazy";
                    thumb.style.cssText = `
                        width: 96px; height: 96px; object-fit: cover;
                        border-radius: 6px; border: 1px solid #3a3a3e; cursor: pointer;
                        display: block;
                    `;
                    thumb.addEventListener("mouseenter", () => { thumb.style.borderColor = "#6af"; });
                    thumb.addEventListener("mouseleave", () => { thumb.style.borderColor = "#3a3a3e"; });
                    thumb.addEventListener("error", () => {
                        if (thumb.dataset.fallback) return;
                        thumb.dataset.fallback = "1";
                        const fb = document.createElement("div");
                        fb.textContent = path.split("/").pop().slice(0, 14);
                        fb.title = path.split("/").pop() + " (video) — 点击插入引用";
                        fb.style.cssText = `
                            width: 96px; height: 96px; border-radius: 6px; border: 1px solid #3a3a3e;
                            display: flex; align-items: center; justify-content: center; text-align: center;
                            background: #1c1c1e; color: #888; font-size: 9px; word-break: break-all;
                            cursor: pointer; padding: 4px; box-sizing: border-box;
                        `;
                        fb.addEventListener("click", () => {
                            if (activeTextarea) insertAtCursor(activeTextarea, buildSampleMarkdown(path));
                        });
                        thumb.replaceWith(fb);
                        thumb = fb;
                    });
                    thumb.addEventListener("click", () => {
                        if (activeTextarea) insertAtCursor(activeTextarea, buildSampleMarkdown(path));
                    });
                } else {
                    thumb = document.createElement("img");
                    thumb.src = `/api/sfnodes/lora_samples/image?path=${encodeURIComponent(path)}&w=256`;
                    thumb.title = path.split("/").pop();
                    thumb.loading = "lazy";
                    thumb.style.cssText = `
                        width: 96px; height: 96px; object-fit: cover;
                        border-radius: 6px; border: 1px solid #3a3a3e; cursor: pointer;
                        display: block;
                    `;
                    thumb.addEventListener("mouseenter", () => { thumb.style.borderColor = "#6af"; });
                    thumb.addEventListener("mouseleave", () => { thumb.style.borderColor = "#3a3a3e"; });
                    thumb.addEventListener("click", () => {
                        if (activeTextarea) insertAtCursor(activeTextarea, buildSampleMarkdown(path));
                    });
                }
                // 删除按钮：悬停显示，右上角（SVG）
                const delBtn = document.createElement("button");
                delBtn.title = "删除该示例图";
                delBtn.style.cssText = `
                    position: absolute; top: 0; right: 0; display: none; align-items: center; justify-content: center;
                    width: 18px; height: 18px; padding: 0; line-height: 1;
                    background: rgba(224, 108, 108, 0.9); color: #fff;
                    border: none; border-radius: 0 6px 0 6px; cursor: pointer;
                `;
                delBtn.appendChild(_makeSampleIcon(_SAMPLE_ICON_TRASH));
                // 载入工作流按钮：悬停显示，右下角（SVG）
                const loadBtn = document.createElement("button");
                loadBtn.title = "将该图片载入为工作流（需内嵌工作流数据）";
                loadBtn.style.cssText = `
                    position: absolute; bottom: 0; right: 0; display: none; align-items: center; justify-content: center;
                    width: 18px; height: 18px; padding: 0; line-height: 1;
                    background: rgba(79, 124, 255, 0.9); color: #fff;
                    border: none; border-radius: 6px 0 6px 0; cursor: pointer;
                `;
                loadBtn.appendChild(_makeSampleIcon(_SAMPLE_ICON_LOAD));
                // 复制 prompt 按钮：悬停显示，左下角（SVG）
                const promptBtn = document.createElement("button");
                promptBtn.title = "复制该图片的 prompt 到剪贴板";
                promptBtn.style.cssText = `
                    position: absolute; bottom: 0; left: 0; display: none; align-items: center; justify-content: center;
                    width: 18px; height: 18px; padding: 0; line-height: 1;
                    background: rgba(46,160,90,0.92); color: #fff;
                    border: none; border-radius: 0 6px 0 5px; cursor: pointer;
                `;
                promptBtn.appendChild(_makeSampleIcon(_SAMPLE_ICON_PROMPT));
                wrap.addEventListener("mouseenter", () => {
                    delBtn.style.display = "flex";
                    loadBtn.style.display = "flex";
                    promptBtn.style.display = "flex";
                });
                wrap.addEventListener("mouseleave", () => {
                    delBtn.style.display = "none";
                    loadBtn.style.display = "none";
                    promptBtn.style.display = "none";
                });
                loadBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    await loadImageAsWorkflow(path, (msg) => { sampleHint.textContent = msg; });
                });
                promptBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    promptBtn.style.opacity = "0.5";
                    promptBtn.style.pointerEvents = "none";
                    try {
                        const resp = await app.api.fetchApi(`/api/sfnodes/lora_samples/prompt?path=${encodeURIComponent(path)}`);
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);
                        if (!data.found || !data.text) {
                            sampleHint.textContent = data.message || "该图片未包含 prompt。";
                            return;
                        }
                        const ok = await copyText(data.text);
                        sampleHint.textContent = ok ? "已复制 prompt 到剪贴板。" : "复制失败。";
                        if (ok) setTimeout(() => { if (sampleHint.textContent === "已复制 prompt 到剪贴板。") sampleHint.textContent = ""; }, 2000);
                    } catch (err) {
                        sampleHint.textContent = "读取失败：" + (err.message || err);
                    } finally {
                        promptBtn.style.opacity = "";
                        promptBtn.style.pointerEvents = "";
                    }
                });
                delBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const fileName = path.split("/").pop();
                    if (!confirm(`删除示例图「${fileName}」？此操作不可恢复。`)) return;
                    try {
                        const resp = await app.api.fetchApi(
                            `/api/sfnodes/lora_samples?path=${encodeURIComponent(path)}`,
                            { method: "DELETE" }
                        );
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        refreshSamplePanel();
                        refreshBrowseSamplePanel();
                    } catch (err) {
                        console.warn("[SF Model Info] sample delete failed:", err);
                        sampleHint.textContent = "删除失败：" + (err.message || err);
                    }
                });
                wrap.appendChild(thumb);
                wrap.appendChild(delBtn);
                wrap.appendChild(loadBtn);
                wrap.appendChild(promptBtn);
                sampleGrid.appendChild(wrap);
            }
        } catch (e) {
            console.warn("[SF Model Info] lora_samples list failed:", e);
            sampleHint.textContent = "获取示例图失败：" + (e.message || e);
        }
    }

    function openSamplePanel(textarea) {
        activeTextarea = textarea;
        sampleOpen = true;
        samplePanel.style.display = "block";
        refreshSamplePanel();
    }

    function closeSamplePanel() {
        sampleOpen = false;
        activeTextarea = null;
        samplePanel.style.display = "none";
    }

    uploadInput.addEventListener("change", async () => {
        const file = uploadInput.files?.[0];
        uploadInput.value = "";
        if (!file) return;
        if (!name || name === "None") return;
        const fd = new FormData();
        fd.append("image", file);
        fd.append("filename", name);
        try {
            const resp = await app.api.fetchApi("/api/sfnodes/lora_samples/upload", {
                method: "POST",
                body: fd,
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            if (activeTextarea) insertAtCursor(activeTextarea, buildSampleMarkdown(data.path));
            if (sampleOpen) refreshSamplePanel();
            refreshBrowseSamplePanel();
        } catch (e) {
            console.warn("[SF Model Info] sample upload failed:", e);
            sampleHint.textContent = "上传失败：" + (e.message || e);
        }
    });

    // ---------- footer ----------
    const footer = document.createElement("div");
    footer.style.cssText = `
        display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
        padding: 12px 18px; border-top: 1px solid #444;
    `;

    function makeFooterBtn(text, color, callback, title) {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.title = title || "";
        btn.style.cssText = `
            padding: 6px 14px; border: 1px solid ${color}; border-radius: 6px;
            font-size: 12px; cursor: pointer; color: ${color};
            background: transparent; transition: filter 0.15s;
        `;
        btn.addEventListener("mouseenter", () => { btn.style.filter = "brightness(1.3)"; });
        btn.addEventListener("mouseleave", () => { btn.style.filter = ""; });
        btn.addEventListener("click", callback);
        return btn;
    }

    const copyBtn = makeFooterBtn("📋 Copy Trigger Words", "#aaa", () => {
        if (state.trigger_words) {
            navigator.clipboard.writeText(state.trigger_words).catch(() => {});
        }
    }, "Copy trigger words to clipboard");
    const clearBtn = makeFooterBtn("🗑️ Clear Notes", "#e06c6c", () => {
        state.trigger_words = "";
        state.description = "";
        twRow.refresh();
        descRow.refresh();
        saveNotes();
    }, "Clear custom notes for this model");
    const spacer = document.createElement("div");
    spacer.style.cssText = "flex: 1;";
    const footerRight = document.createElement("div");
    footerRight.style.cssText = "display: flex; gap: 8px;";

    // 底部右侧按钮随状态切换：编辑中 = 取消 + Save；浏览态 = Done（关闭）
    function renderFooterActions() {
        footerRight.innerHTML = "";
        const editing = twRow._editing || descRow._editing;
        if (editing) {
            const cancelBtn = makeFooterBtn("✕ Cancel", "#aaa", () => {
                twRow.cancelEdit();
                descRow.cancelEdit();
            }, "放弃修改返回浏览页");
            const dirty = twRow._dirty || descRow._dirty;
            const saveBtn = makeFooterBtn(dirty ? "Save*" : "Save", "#4f7cff", () => {
                twRow.commitEdit();
                descRow.commitEdit();
                saveNotes();
            }, "保存并返回浏览页");
            if (dirty) {
                // 未保存修改高亮：实底主色 + 白字粗体（替代纯文本星号）
                saveBtn.style.background = "#4f7cff";
                saveBtn.style.color = "#fff";
                saveBtn.style.fontWeight = "600";
                saveBtn.style.borderColor = "#4f7cff";
            }
            footerRight.appendChild(cancelBtn);
            footerRight.appendChild(saveBtn);
        } else {
            const doneBtn = makeFooterBtn("Done", "#4f7cff", () => closeDialog(), "关闭");
            footerRight.appendChild(doneBtn);
        }
    }

    // ── Civitai 按钮组（常驻 footer，不随编辑态切换）───────────────────────
    // 账户展开区先声明：refreshCivButtons 的 Account 按钮切换它（const TDZ）
    let accOpen = false;
    const accPanel = document.createElement("div");
    accPanel.className = "sf-li-acc";
    accPanel.style.display = "none";
    const civBtns = document.createElement("div");
    civBtns.style.cssText = "display:flex; align-items:center; gap:8px;";

    function refreshCivButtons() {
        civBtns.innerHTML = "";
        // ↻ Civitai：查询入口（文件缺失时无意义，隐藏）
        if (name && name !== "None" && !meta._file_missing) {
            const searching = civ?.state === "searching";
            const b = makeFooterBtn(searching ? "Looking up…" : "↻ Civitai", "#aaa", searching ? null : runCivitai,
                "按内容指纹在 Civitai 上查找该文件，并把信息保存到文件旁（之后离线即得）");
            if (searching) b.style.opacity = "0.6";
            civBtns.appendChild(b);
        }
        // 🗑 删除已存 Civitai 侧车——回到文件自己的词（仅当侧车存在）
        if (hasSidecar) {
            const b = makeFooterBtn("🗑", "#c9736a", runDeleteCivitai,
                "Delete the saved Civitai info (back to the file's own words)");
            b.title = "Delete the saved Civitai info (back to the file's own words)";
            civBtns.appendChild(b);
        }
        // Civitai account：与 SFLoraStack 同一份配置（civitai.json，机器级）
        const ab = makeFooterBtn(accOpen ? "▾ Account" : "▸ Account", "#aaa", toggleAccount,
            "API key & lookup preferences — saved on this computer, shared with the LoRA Stack node");
        civBtns.appendChild(ab);
    }

    footer.appendChild(copyBtn);
    footer.appendChild(clearBtn);
    footer.appendChild(spacer);
    footer.appendChild(civBtns);
    footer.appendChild(footerRight);
    renderFooterActions();
    refreshCivButtons();

    card.appendChild(header);
    card.appendChild(body);
    card.appendChild(accPanel);
    card.appendChild(footer);
    dialog.appendChild(card);

    // ---------- actions ----------
    function saveNotes() {
        const bodyData = {
            trigger_words: state.trigger_words,
            description: state.description,
        };
        fetch(`/api/sfnodes/lora_notes?filename=${encodeURIComponent(name)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyData),
        })
            .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
            .then(updated => {
                // 先广播失效再写回自己的缓存：dispatchEvent 同步触发事件桥监听器
                // （含本模块的 loraMetadataCache.delete），顺序反了会把自己刚写入的
                // 新值一并删掉——保存后 i 图标变灰，重开对话框 force 重取才恢复。
                document.dispatchEvent(new CustomEvent("sfnodes.lora-data-changed", { detail: { name } }));
                loraMetadataCache.set(name, updated);
                // 另一端（SFLoraStack 面板等）的缓存已被广播清掉，下次打开即新数据
                app.graph.setDirtyCanvas(true, true);
                state.trigger_words = updated.trigger_words || "";
                state.description = updated.description || "";
                twRow.refresh();
                descRow.refresh();
            })
            .catch(e => console.warn("[SF Model Info] Failed to save notes:", e));
    }

    // ── Civitai 查询（复用 SFLoraStack 同一套路由/配置）────────────────────
    // 状态条：body 顶部动态插入/移除；只更新自身元素，不重建行（编辑中的
    // 行由 applyMetaRefresh 守卫跳过）。
    let civStripEl = null;
    function refreshCivStrip() {
        if (!civ) { civStripEl?.remove(); civStripEl = null; return; }
        if (!civStripEl) {
            civStripEl = document.createElement("div");
            body.insertBefore(civStripEl, body.firstChild);
        }
        const st = civ.state;
        civStripEl.className = "sf-li-civstrip " + (st === "searching" ? "searching"
            : st === "found" ? "found" : st === "archive" ? "archive" : st === "offline" ? "offline" : "nofind");
        civStripEl.innerHTML = "";
        const ic = document.createElement("span");
        ic.className = "ic";
        if (st === "searching") ic.innerHTML = '<span class="sf-li-spin"></span>';
        else ic.textContent = st === "found" ? "✓" : st === "archive" ? "📦" : st === "offline" ? "!" : "?";
        const stripBody = document.createElement("div");
        if (st === "searching") {
            stripBody.textContent = "Looking up on Civitai… matching this file's fingerprint.";
        } else if (st === "found") {
            stripBody.textContent = "Found on Civitai. Saved next to the file, so it's instant and offline next time.";
            if (civ.note) stripBody.appendChild(document.createTextNode(" " + civ.note));
            if (civ.info?.model_id != null) {
                // 按账户主机偏好选网页域（red 用户看 civitai.red）
                const host = _acc?.host === "red" ? "civitai.red" : "civitai.com";
                const link = document.createElement("a");
                link.className = "civlink";
                link.textContent = " View on Civitai ↗";
                link.href = `https://${host}/models/${civ.info.model_id}`
                    + (civ.info.version_id != null ? `?modelVersionId=${civ.info.version_id}` : "");
                link.target = "_blank";
                link.rel = "noopener";
                stripBody.appendChild(link);
            }
        } else if (st === "archive") {
            stripBody.textContent = civ.hint || "This version is a .zip archive containing multiple files — the local file's fingerprint doesn't match the archive.";
            const note = document.createElement("div");
            note.style.cssText = "color:#9a9a9a;font-size:10px;margin-top:2px;";
            note.textContent = "Paste Civitai link or Version ID:";
            stripBody.appendChild(note);
            const row = document.createElement("div");
            row.className = "sf-li-archive-row";
            const inp = document.createElement("input");
            inp.type = "text";
            inp.placeholder = "https://civitai.com/models/1874153?modelVersionId=2121297  or  2121297";
            inp.value = civ.prefill || "";
            const btn = document.createElement("button");
            btn.textContent = "Fetch";
            btn.addEventListener("click", () => {
                const v = inp.value.trim();
                if (!v) { stripBody.appendChild(document.createTextNode(" Paste a link first.")); return; }
                runCivitai({ civitaiUrl: v });
            });
            inp.addEventListener("keydown", (e) => {
                if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); btn.click(); }
                else e.stopPropagation();
            });
            inp.addEventListener("mousedown", (e) => e.stopPropagation());
            row.append(inp, btn);
            stripBody.appendChild(row);
        } else if (st === "nofind") {
            stripBody.textContent = "Not on Civitai. This exact file isn't in their database (it may be private, renamed, or custom-trained). The words read from the file are still shown.";
        } else {
            stripBody.textContent = civ.message || "Couldn't reach Civitai. No connection, or it's busy. Use the file's own words, or try again.";
        }
        civStripEl.append(ic, stripBody);
    }

    // 查询成功/删除侧车后用新合并元数据刷新展示：非编辑中的行直接更新
    // （不写存储——用户点 Save 才落盘自定义词）；编辑中的行保持草稿。
    function applyMetaRefresh(meta2) {
        if (!twRow._editing) { state.trigger_words = meta2.trigger_words || ""; twRow.refresh(); }
        if (!descRow._editing) { state.description = meta2.description || ""; descRow.refresh(); }
        if (meta2.base_model !== undefined) {
            if (meta2.base_model) {
                if (!bmRow) { bmRow = createReadonlyRow("Base Model", meta2.base_model); body.insertBefore(bmRow, urlRow || samplePanel); }
                else { bmRow.style.display = ""; renderReadonlyValue(bmRow, meta2.base_model); }
            } else if (bmRow) bmRow.style.display = "none";
        }
        if (meta2.source_url !== undefined) {
            if (meta2.source_url) {
                if (!urlRow) { urlRow = createReadonlyRow("Source URL", meta2.source_url, meta2.source_url); body.insertBefore(urlRow, samplePanel); }
                else { urlRow.style.display = ""; renderReadonlyValue(urlRow, meta2.source_url, meta2.source_url); }
            } else if (urlRow) urlRow.style.display = "none";
        }
    }

    function updateThumb() {
        if (name && name !== "None") {
            thumbEl.src = `/api/sfnodes/lora_thumb?name=${encodeURIComponent(name)}&t=${_thumbBust || Date.now()}`;
        }
    }

    async function runCivitai(opts) {
        if (!name || name === "None" || civ?.state === "searching") return;
        civ = { state: "searching" };
        refreshCivStrip();
        refreshCivButtons();
        const res = await civitaiLookup(name, opts);
        if (!dialog.isConnected) return;
        if (res.ok && res.found) {
            civ = { state: "found", info: res.info || {}, note: "" };
            // 封面保存结果附在状态条上：成功静默（本地图经 /lora_thumb 刷新后
            // 自动显示）；被跳过（已有自定义预览）稍后用确认框询问；失败则提示。
            if (res.thumb_v) _thumbBust = res.thumb_v;
            else if (res.thumb_skipped) civ.note = "Your own preview picture was kept.";
            else if (res.thumb_error) civ.note = "Couldn't save the preview: " + res.thumb_error;
            // 样例原图批量结果（开关 sfnodes.Civitai.DownloadSamples）
            if (res.samples_downloaded) {
                const n = res.samples_downloaded;
                civ.note = (civ.note ? civ.note + " " : "") + `Downloaded ${n} sample image${n > 1 ? "s" : ""} to sample/.`;
            } else if (res.samples_note) {
                civ.note = (civ.note ? civ.note + " " : "") + res.samples_note;
            } else if (res.samples_error) {
                civ.note = (civ.note ? civ.note + " " : "") + res.samples_error;
            }
            hasSidecar = true;
            refreshCivStrip();
            // 若样例面板正打开，刷新以显示新下载的图；浏览区常驻，始终刷新
            try { if (typeof refreshSamplePanel === "function" && samplePanel.style.display !== "none") refreshSamplePanel(); } catch {}
            try { if (typeof refreshBrowseSamplePanel === "function") refreshBrowseSamplePanel(); } catch {}
            // 侧车已写入：force 重取合并元数据刷新展示（编辑中的行跳过）
            const meta2 = await getLoraMetadata(name, true);
            if (!dialog.isConnected) return;
            if (meta2 && !meta2._not_found) applyMetaRefresh(meta2);
            updateThumb();
            app.graph.setDirtyCanvas(true, true);
            // 已有用户自定义预览时查询不覆盖保存（thumb_skipped）——确认后走
            // 独立保存端点（读侧车同一张图下载，无需重新查询）
            if (res.thumb_skipped) {
                if (confirm("This LoRA already has a preview picture you set.\nReplace it with the one found on Civitai?")) {
                    const sv = await saveCivitaiThumb(name);
                    if (!dialog.isConnected) return;
                    if (!sv?.ok) civ.note = "Couldn't save the preview: " + ((sv && sv.message) || "unknown error");
                    else { _thumbBust = sv.v || Date.now(); updateThumb(); }
                    refreshCivStrip();
                }
            }
        } else if (res.reason === "archive") {
            civ = { state: "archive", hint: res.hint || res.message, prefill: opts?.civitaiUrl || "" };
        } else if (res.reason === "notfound") {
            civ = { state: "nofind" };
        } else {
            civ = { state: "offline", message: res.message || "Couldn't reach Civitai." };
        }
        if (!dialog.isConnected) return;
        refreshCivStrip();
        refreshCivButtons();
    }

    async function runDeleteCivitai() {
        if (!name || name === "None") return;
        await deleteCivitai(name);
        if (!dialog.isConnected) return;
        civ = null;
        hasSidecar = false;
        _thumbBust = Date.now();              // 侧车（因此预览）变了
        const meta2 = await getLoraMetadata(name, true);
        if (!dialog.isConnected) return;
        if (meta2 && !meta2._not_found) applyMetaRefresh(meta2);
        updateThumb();
        app.graph.setDirtyCanvas(true, true);
        refreshCivStrip();
        refreshCivButtons();
    }

    // ── Civitai 账户（与 SFLoraStack 同一份 civitai.json，机器级共享）────────
    function buildAccountPanel() {
        accPanel.innerHTML = "";
        const head = document.createElement("div");
        head.className = "sf-li-acc-head";
        head.textContent = "Civitai account";
        const sub = document.createElement("div");
        sub.className = "sf-li-acc-sub";
        sub.textContent = "Saved on this computer, shared with the LoRA Stack node — never in your workflows. A key lets the lookup see models that Civitai hides from anonymous requests.";
        accPanel.append(head, sub);

        let editing = false;
        const msg = document.createElement("div");
        msg.className = "sf-li-acc-msg";
        const say = (t, ok) => {
            msg.textContent = t || "";
            msg.style.display = t ? "block" : "none";
            msg.className = "sf-li-acc-msg" + (ok ? " ok" : "");
        };

        // key 行（显示 <-> 编辑切换；编辑中绝不整行重建——见 paintKeyRow）
        const keyRow = document.createElement("div");
        keyRow.className = "sf-li-acc-row";
        const paintKeyRow = () => {
            editing = false;
            keyRow.textContent = "";
            const st = document.createElement("span");
            st.className = "lab";
            st.textContent = _acc?.configured ? "✓ Key saved  " + (_acc.hint || "") : "No key — anonymous lookups";
            const edit = document.createElement("span");
            edit.className = "sf-li-acc-mini";
            edit.textContent = _acc?.configured ? "Change" : "Add key";
            edit.title = "Paste a key from civitai.com > Account settings > API Keys";
            edit.addEventListener("click", showEditor);
            keyRow.append(st, edit);
            if (_acc?.configured) {
                const rm = document.createElement("span");
                rm.className = "sf-li-acc-mini rm";
                rm.textContent = "Remove";
                rm.title = "Forget the key. Lookups go back to anonymous.";
                rm.addEventListener("click", () => save({ key: "" }, "Key removed."));
                keyRow.appendChild(rm);
            }
        };
        function showEditor() {
            editing = true;
            say("");
            keyRow.textContent = "";
            const inp = document.createElement("input");
            inp.className = "sf-li-acc-key";
            inp.type = "password";
            inp.placeholder = "Paste your API key";
            inp.autocomplete = "off";
            inp.spellcheck = false;
            inp.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") { e.preventDefault(); commit(); }
            });
            const ok = document.createElement("span");
            ok.className = "sf-li-acc-mini";
            ok.textContent = "Save";
            const no = document.createElement("span");
            no.className = "sf-li-acc-mini";
            no.textContent = "Cancel";
            const commit = () => {
                const v = inp.value.trim();
                if (!v) { say("Nothing to save — paste a key first."); return; }
                // 唯一一个应把编辑器换回状态行的保存，且只在服务器确认后
                save({ key: v }, "Key saved.", true);
            };
            ok.addEventListener("click", commit);
            no.addEventListener("click", () => { say(""); paintKeyRow(); });
            keyRow.append(inp, ok, no);
            inp.focus();
        }

        // host 行
        const hostRow = document.createElement("div");
        hostRow.className = "sf-li-acc-row";
        const hostLab = document.createElement("span");
        hostLab.className = "lab";
        hostLab.textContent = "Ask this site first";
        const hostSeg = document.createElement("div");
        hostSeg.className = "sf-li-acc-seg";
        const HOSTS = [
            { v: "com", label: "Standard", hint: "civitai.com, then civitai.red as a backup", title: "The usual choice" },
            { v: "red", label: "Unrestricted", hint: "civitai.red first, for adult-rated models", title: "Civitai's unrestricted domain. Use this if your LoRAs are not found." },
        ];
        for (const o of HOSTS) {
            const b = document.createElement("div");
            b.className = "sf-li-acc-segb";
            b.textContent = o.label;
            b.dataset.v = o.v;
            b.title = o.title;
            b.addEventListener("click", () => save({ host: o.v }, ""));
            hostSeg.appendChild(b);
        }
        hostRow.append(hostLab, hostSeg);

        // adult 行
        const adultRow = document.createElement("div");
        adultRow.className = "sf-li-acc-row";
        const adultLab = document.createElement("span");
        adultLab.className = "lab";
        adultLab.textContent = "Allow adult preview images";
        const adultSw = document.createElement("div");
        adultSw.className = "sf-li-acc-sw";
        adultSw.addEventListener("click", () => save({ adultThumbs: !_acc?.adultThumbs }, ""));
        adultRow.append(adultLab, adultSw);

        const paintRest = () => {
            for (const b of hostSeg.children) b.classList.toggle("on", b.dataset.v === _acc?.host);
            adultSw.classList.toggle("on", !!_acc?.adultThumbs);
        };
        const paint = () => { if (!editing) paintKeyRow(); paintRest(); };

        // 按服务器实际存储的应答重绘，绝不按我们以为它收下的。_accDirty：
        // 用户已保存过，打开面板时发出的 GET 应答可能迟到，落地会把面板从
        // 刚设的值跳回旧值（"设了 red 它显示 com"）。
        let _accDirty = false;
        async function save(patch, okNote, closeEditor) {
            if (_accBusy) return;
            _accBusy = true;
            try {
                const res = await setCivitaiAccount(patch);
                if (!dialog.isConnected) return;
                if (!res || !res.ok) {
                    say((res && res.message) || "Could not save.");
                    // 不整面板重绘：paintKeyRow 从零重建会丢掉编辑器和打好的 key
                    paintRest();
                    return;
                }
                _accDirty = true;
                _acc = res;
                if (closeEditor) editing = false;
                say(okNote || "", true);
                paint();
                app.graph.setDirtyCanvas(true, true);
            } finally {
                _accBusy = false;
            }
        }

        accPanel.append(keyRow, msg, hostRow, adultRow);
        paint();
        // 对话框尾部预取失败时（_acc 为 null）展开面板再读一次
        if (!_acc) {
            getCivitaiAccount().then((res) => {
                if (!dialog.isConnected || !res || !res.ok || _accDirty) return;
                _acc = res;
                paint();
            });
        }
    }

    function toggleAccount() {
        accOpen = !accOpen;
        accPanel.style.display = accOpen ? "block" : "none";
        if (accOpen && !accPanel.firstChild) buildAccountPanel();
        refreshCivButtons();
    }

    function closeDialog() {
        if (!dialog.open) return;
        // 有未保存的修改时确认，防止误关丢失内容
        if ((twRow._editing && twRow._dirty) || (descRow._editing && descRow._dirty)) {
            if (!confirm("有未保存的修改，确定要关闭吗？")) return;
        }
        dialog.close();
    }

    // Native <dialog> modal: Esc triggers "cancel" (unless an input is being
    // edited, whose keydown handler stopPropagation's Escape first).
    dialog.addEventListener("cancel", (e) => {
        if (document.body.dataset.sfPreviewEsc === "1" || document.querySelector(".sf-lora-sample-preview, .sf-li-sample-preview, .sf-li-desc-hover, .sf-ls-sample-preview, .sf-ls-desc-hover")) {
            e.preventDefault();
            try { delete document.body.dataset.sfPreviewEsc; } catch {}
            // 若预览仍在（keydown 未消费），则关闭预览而非对话框
            const preview = document.querySelector(".sf-lora-sample-preview, .sf-li-sample-preview");
            if (preview) {
                try { preview.remove(); } catch {}
                // 移除预览的 keydown 监听（由预览自身 close 清理，此处兜底）
                try { document.body.dataset.sfPreviewEsc = "1"; setTimeout(() => { try { delete document.body.dataset.sfPreviewEsc; } catch {} }, 50); } catch {}
            }
            return;
        }
        e.preventDefault();
        closeDialog();
    });
    dialog.addEventListener("close", () => {
        dialog.remove();
    });
    // Click on the backdrop (outside the dialog box) closes it.
    // 以 mousedown 位置判定：编辑中布局变化（如 markdown 渲染内容切换为
    // textarea）会使 dialog 收缩/位移，click 事件坐标可能落到新矩形之外，
    // 误判为背景点击而关闭弹窗。mousedown 在框内则忽略该次 click。
    let mouseDownInside = false;
    dialog.addEventListener("mousedown", (e) => {
        const rect = dialog.getBoundingClientRect();
        mouseDownInside = (
            e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom
        );
    });
    dialog.addEventListener("click", (e) => {
        if (mouseDownInside) return;
        // 合成事件（element.click()）clientX/Y 为 0,0，按框内处理，避免误关
        if (!e.clientX && !e.clientY) return;
        const rect = dialog.getBoundingClientRect();
        if (
            e.clientX < rect.left || e.clientX > rect.right ||
            e.clientY < rect.top || e.clientY > rect.bottom
        ) {
            closeDialog();
        }
    });

    document.body.appendChild(dialog);
    dialog.showModal();

    // ── 打开即预取（fire-and-forget，dialog 关闭后落地无副作用）───────────
    // 账户公开状态：View on Civitai 链接域选择 + Account 展开区显示。
    getCivitaiAccount().then((res) => {
        if (dialog.isConnected && res && res.ok) _acc = res;
    });
    // 已存 Civitai 侧车探测：决定 🗑 按钮（对话框打开时静默，不打扰）。
    if (name && name !== "None" && !meta._file_missing) {
        loraInfo(name).then((res) => {
            if (!dialog.isConnected || !res?.ok || !res.info) return;
            const had = res.info.source === "sidecar" || (res.info.sidecar_triggers?.length || 0) > 0;
            if (had && !hasSidecar) {
                hasSidecar = true;
                refreshCivButtons();
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Canvas event capture (shared single wrapper)
// ---------------------------------------------------------------------------
let _lastCanvasEvent = null;
let _eventHookInstalled = false;

export function ensureEventHook() {
    if (_eventHookInstalled) return;
    _eventHookInstalled = true;
    const origAdjust = LGraphCanvas.prototype.adjustMouseEvent;
    LGraphCanvas.prototype.adjustMouseEvent = function (e) {
        origAdjust.apply(this, arguments);
        _lastCanvasEvent = e;
    };
}

export function getLastCanvasEvent() {
    return _lastCanvasEvent;
}

// ---------------------------------------------------------------------------
// Standard-combo + info-icon mounting (shared by SFLoraLoader /
// SFLoraLoaderModelOnly and future loader nodes)
// ---------------------------------------------------------------------------
const INVALID_BOUNDS = [0, -1];

function getComboWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function getComboValue(node, name) {
    const v = getComboWidget(node, name)?.value;
    return typeof v === "string" ? v : null;
}

function createInfoWidget(comboName) {
    const w = {
        name: "_info",
        type: "custom",
        options: { serialize: false },
        value: {},
        y: 0,
        last_y: 0,
        _hit: INVALID_BOUNDS,
        computeSize(width) { return [width, 24]; },
        draw(ctx, n, width, posY, height) {
            this.last_y = posY;
            this._hit = INVALID_BOUNDS;
            const loraName = getComboValue(n, comboName);
            if (!loraName || loraName === "None") return;
            const cachedMeta = loraMetadataCache.get(loraName);
            const hasCustom = cachedMeta?._has_custom;
            const size = Math.max(14, height * 0.6);
            const posX = 10;
            const centerX = posX + size / 2;
            const midY = posY + height * 0.5;
            this._hit = [posX, size + 6];
            ctx.save();
            ctx.beginPath();
            ctx.arc(centerX, midY, size / 2 - 0.5, 0, Math.PI * 2);
            if (hasCustom) {
                ctx.fillStyle = "rgba(79,195,247,0.3)";
                ctx.strokeStyle = "rgba(79,195,247,0.7)";
            } else {
                ctx.fillStyle = "rgba(255,255,255,0.25)";
                ctx.strokeStyle = "rgba(255,255,255,0.4)";
            }
            ctx.lineWidth = 1;
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = hasCustom ? "rgba(79,195,247,0.9)" : "rgba(255,255,255,0.6)";
            ctx.font = `${Math.round(size * 0.6)}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("i", centerX, midY + 0.5);
            if ((app.canvas.ds?.scale || 1) > 0.5) {
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                ctx.textAlign = "left";
                ctx.fillText("Info", posX + size + 6, midY);
            }
            ctx.restore();
        },
        mouse(event, pos, n) {
            if (event.type !== "pointerdown") return false;
            const b = w._hit;
            if (b[1] < 0) return false;
            if (pos[0] >= b[0] && pos[0] <= b[0] + b[1]) {
                const loraName = getComboValue(n, comboName);
                if (loraName && loraName !== "None") {
                    // 延迟到 pointerup 由 canvas 处理完成后再打开对话框，
                    // 避免 DOM 遮罩在点击过程中出现导致 LiteGraph widget 交互状态残留
                    // force：打开必新（SFLoraStack 面板等另一端可能刚保存过）
                    getLoraMetadata(loraName, true).then((meta) => {
                        requestAnimationFrame(() => {
                            setTimeout(() => showLoraInfoDialog(event, loraName, meta), 0);
                        });
                    });
                }
                return true;
            }
            return false;
        },
    };
    return w;
}

// Mounts a standard combo + info-icon widget pair onto a loader node:
// binds the combo callback to prefetch metadata, guards the positional
// restoration of widgets_values, and prefetches the restored value after
// configure (widget values are restored after onNodeCreated).
export function setupLoraInfoWidget(node, comboName = "lora_name") {
    const combo = getComboWidget(node, comboName);
    if (combo) {
        const origCallback = combo.callback;
        combo.callback = (value) => {
            if (origCallback) origCallback(value);
            if (value && value !== "None") getLoraMetadata(value);
        };
    }

    const _origConfigure = node.configure;
    node.configure = function (info) {
        const idx = this.widgets?.findIndex((w) => w.name === "_info") ?? -1;
        if (idx !== -1) this.widgets.splice(idx, 1);
        if (_origConfigure) _origConfigure.call(this, info);
        const loraName = getComboValue(this, comboName);
        if (loraName && loraName !== "None") getLoraMetadata(loraName);
        this.widgets.push(createInfoWidget(comboName));
    };

    node.widgets.push(createInfoWidget(comboName));
}
