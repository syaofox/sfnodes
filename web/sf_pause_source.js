// ==========================================================================
// sf_pause_source.js - SFPauseImage 外载图（Load / Browse / 拖放 / Ctrl+V）
// ==========================================================================
//
// 独立模块：把 image_browser / CropAPI 等重依赖隔离在此，不进入 sf_pause_kit.js
// （kit 是 image/mask/latent 共享引擎，只有 image 需要外载图）。
//
// 语义（Design D）：外载图只替换 Continue 提交的那张 temp 快照——
//   - Pause/Pass（Run）照常用接线图捕获（未接线时才回退用已物化的快照）；
//   - 加载（Load/Browse/拖放/粘贴）→ 复用 CropAPI.uploadSrc 落盘 input/
//     → 调 /api/sfnodes/pause/load 物化写入 temp 快照（按当前 flip 镜像）
//     → 预览立即显示；
//   - Continue 读 temp 快照 → 输出"显示的外载图"，接线不被断开；
//   - Clear 清除外载图标记与预览。
//
// 外载图标记 node._sfPauseSrcPath 仅内存（不持久化、不自动恢复）：temp 快照
// 本就随重启清空，避免与"最后一次 Run 结果"语义冲突（用户拍板）。
//
// kit 只提供两个接入点：cfg.extraHeight（给本行预留高度）与 els.preview
// （插入按钮行 + 挂原生拖放监听）。本模块注册第二个 extension，在 kit 的
// onNodeCreated/onConfigure 之后链式执行。
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { installPasteHandler, parseAnnotatedImageValue, buildSourceURL } from "./sf_common.js";
import { showImageBrowser } from "./image_browser.js";
import { CropAPI } from "./sf_crop_core.js";

const LOAD_PATH = "/api/sfnodes/pause/load";

function getSrc(node) {
    return (node && node._sfPauseSrcPath) || "";
}

function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = reject;
        r.readAsDataURL(file);
    });
}

function readBlobAsDataURL(blob) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = reject;
        r.readAsDataURL(blob);
    });
}

// cfg: { gate, classy, stateProp, propPrefix, cssPrefix, emptyText, logTag }
// gate = definePauseGate(...) 的返回值（state/body/flash/props）
export function attachSourceControls(cfg) {
    const { gate, classy, propPrefix, cssPrefix, emptyText, logTag } = cfg;
    const { getState } = gate.state;
    const { showFrame, renderPause } = gate.body;
    const { flash } = gate;
    const hasSnapProp = gate.props.hasSnapProp;
    const elsProp = propPrefix + "Els";
    const pasteHook = propPrefix + "SrcPaste";

    function refreshClear(node) {
        const els = node[elsProp];
        if (els && els.btnClear) els.btnClear.disabled = !getSrc(node);
    }

    // 把已落盘的外载图物化写入 temp 快照（应用当前 flip），随后显示预览。
    async function materialize(node) {
        const src = getSrc(node);
        if (!src) return false;
        const st = getState(node);
        try {
            const resp = await fetch(api.apiURL(LOAD_PATH), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ unique_id: node.id, src_path: src, flip: !!st.flip }),
            });
            if (!resp.ok) throw new Error(`load ${resp.status}`);
            const data = await resp.json().catch(() => ({}));
            const f = data.frame;
            if (!f || !f.filename) throw new Error("no frame");
            st.frame = { filename: f.filename, subfolder: f.subfolder || "", type: f.type || "temp" };
            showFrame(node, st.frame);  // onload 后置 hasSnap + renderPause
            return true;
        } catch (err) {
            console.error(`[${logTag}] materialize failed`, err);
            flash(node, "Load failed");
            return false;
        }
    }

    // 三个入口共用：dataURL → 落盘 input/ → 物化 temp 快照
    async function loadSource(node, dataURL) {
        if (!dataURL) return;
        try {
            const res = await CropAPI.uploadSrc("pause_" + Date.now(), dataURL);
            const srcPath = res?.path || "";
            if (!srcPath) { flash(node, "Upload failed"); return; }
            node._sfPauseSrcPath = srcPath;
            refreshClear(node);
            await materialize(node);
        } catch (err) {
            console.error(`[${logTag}] load source failed`, err);
            flash(node, "Load failed");
        }
    }

    function pickFile(node) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            try { await loadSource(node, await readFileAsDataURL(file)); }
            catch (err) { console.error(`[${logTag}] read file failed`, err); }
        };
        input.click();
    }

    // Browse：复用图片浏览器选择器模式，选中后经 /view 取字节 → dataURL
    function browse(node) {
        showImageBrowser(node, {
            onPick: async (annotated) => {
                const part = parseAnnotatedImageValue(annotated);
                const url = buildSourceURL(part);
                if (!url) return;
                try {
                    const resp = await fetch(url);
                    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                    await loadSource(node, await readBlobAsDataURL(await resp.blob()));
                } catch (err) {
                    console.error(`[${logTag}] browse load failed`, err);
                    flash(node, "Browse failed");
                }
            },
        });
    }

    function clearSource(node) {
        node._sfPauseSrcPath = "";
        const st = getState(node);
        st.frame = null;
        node[hasSnapProp] = false;
        const els = node[elsProp];
        if (els) {
            els.img.style.display = "none";
            els.img.removeAttribute("src");  // 不用 src=""（会触发一次空请求）
            els.empty.style.display = "flex";
            els.empty.textContent = emptyText || "Press Run to preview the image here";
            els.dims.textContent = "";
        }
        renderPause(node);
        refreshClear(node);
        flash(node, "Loaded image cleared");
    }

    function setupNode(node) {
        const els = node[elsProp];
        if (!els || !els.preview) return;
        if (node._sfPauseSrcRow) { refreshClear(node); return; }

        const row = document.createElement("div");
        row.className = `${cssPrefix}btns`;
        const btnLoad = document.createElement("button");
        btnLoad.className = `${cssPrefix}btn`;
        btnLoad.textContent = "⭳ Load";
        btnLoad.title = "从本地文件加载图片（只替换 Continue 提交的图，接线照常）";
        const btnBrowse = document.createElement("button");
        btnBrowse.className = `${cssPrefix}btn`;
        btnBrowse.textContent = "🖼 Browse";
        btnBrowse.title = "从图片浏览器选择图片";
        const btnClear = document.createElement("button");
        btnClear.className = `${cssPrefix}btn`;
        btnClear.textContent = "✕ Clear";
        btnClear.title = "清除已加载的图片";
        row.append(btnLoad, btnBrowse, btnClear);

        els.preview.before(row);
        node._sfPauseSrcRow = row;
        els.btnLoad = btnLoad;
        els.btnBrowse = btnBrowse;
        els.btnClear = btnClear;

        btnLoad.addEventListener("click", (e) => { e.stopPropagation(); pickFile(node); });
        btnBrowse.addEventListener("click", (e) => { e.stopPropagation(); browse(node); });
        btnClear.addEventListener("click", (e) => { e.stopPropagation(); clearSource(node); });

        // 拖放图片文件到预览区：与按钮共用 loadSource
        els.preview.addEventListener("dragover", (e) => {
            if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); e.stopPropagation(); }
        });
        els.preview.addEventListener("drop", async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const file = e.dataTransfer?.files?.[0];
            if (!file || !file.type?.startsWith("image/")) return;
            try { await loadSource(node, await readFileAsDataURL(file)); }
            catch (err) { console.error(`[${logTag}] drop load failed`, err); }
        });

        refreshClear(node);
    }

    app.registerExtension({
        name: `sfnodes.${classy}Source`,

        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== classy) return;

            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                origCreated?.apply(this, arguments);
                setupNode(this);
                // 粘贴只替换快照，不断开接线（Design D）
                installPasteHandler({
                    comfyClass: classy,
                    hook: pasteHook,
                    disconnectInput: false,
                    onPasteImage: (n, dataURL) => n[pasteHook](dataURL),
                });
                this[pasteHook] = (dataURL) => loadSource(this, dataURL);
            };

            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origConfigure?.apply(this, arguments);
                setupNode(this);
                refreshClear(this);
                return r;
            };

            const origMenu = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
                origMenu?.apply(this, arguments);
                if (!Array.isArray(options)) return;
                if (!getSrc(this)) return;
                options.push({ content: "清除已加载图片", callback: () => clearSource(this) });
            };
        },
    });

    return { loadSource, materialize, clearSource, setupNode, refreshClear };
}
