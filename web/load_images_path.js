// ============================================================
// SF Load Images Path — 目录切换前端（Pixaroma 风格）
// 源切换三档（input / output / images）+ 渐进式目录浏览（面包屑 +
// 当前层子目录下拉 + 左右快速步进）+ 直接输入路径模式。
// 数据通道：隐藏的 folder combo widget（值随 workflow 保存、graphToPrompt
// 自动收集；目录不存在由后端 VALIDATE_INPUTS 校验提示）。
// 目录浏览按需加载：每次进入/回退只 fetch 当前层（/subdirs?folder=）。
// ============================================================
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { applyAdaptiveCanvasOnly, sfApiUrl } from "./sf_common.js";

const SOURCES = ["input", "output", "images"];
const WIDGET_TYPE = "sf_lip_ui";
const MIN_W = 320; // 源三档按钮行 + 面包屑行容纳所需的最小节点宽度

// ── folder 值解析 ─────────────────────────────────────────────────────────
// 目录模式判定只依赖前缀（input/output/images 或 default）——不检查列表
// 包含性，否则目录列表尚未加载时（工作流恢复初期）目录值会被误判为路径
// 模式；带前缀的值在后端语义上本就是目录值。
function parseFolderValue(value) {
    const v = String(value || "").trim();
    if (!v) return { mode: "dir", source: "images", sub: "" };
    const m = v.match(/^(input|output|images)(?:\/(.*))?$/);
    if (m) return { mode: "dir", source: m[1], sub: m[2] || "" };
    if (v === "default") return { mode: "dir", source: "images", sub: "default" };
    return { mode: "path", path: v };
}

// 目录模式值：source + sub
function dirValue(source, sub) {
    return sub ? `${source}/${sub}` : source;
}

// 值 → 面包屑层级数组（["faces", "sub"]）
function pathParts(value) {
    const st = parseFolderValue(value);
    if (st.mode !== "dir" || !st.sub) return [];
    return st.sub.split("/");
}

// ── 样式 ─────────────────────────────────────────────────────────────────
function injectCSS() {
    if (document.getElementById("sf-lip-css")) return;
    const style = document.createElement("style");
    style.id = "sf-lip-css";
    style.textContent = `
    .sf-lip-root { width:100%; box-sizing:border-box; padding:8px; display:flex;
      flex-direction:column; gap:6px; background:#1e1e1e; border-radius:4px; font-size:12px;
      overflow:hidden; }
    .sf-lip-row { display:flex; gap:4px; align-items:center; }
    .sf-lip-btn { flex:1; min-width:0; padding:4px 0; border:1px solid #444; border-radius:4px;
      background:#2a2a2a; color:#aaa; font-size:11px; cursor:pointer; text-align:center;
      user-select:none; font-family:inherit; overflow:hidden; white-space:nowrap;
      text-overflow:ellipsis; }
    .sf-lip-btn:hover { color:#ddd; }
    .sf-lip-btn.on { background:${"var(--sf-acc, #f66744)"}; color:#fff; border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-lip-btn:disabled { opacity:.4; cursor:default; }
    .sf-lip-mode { flex:none; padding:3px 12px; font-size:10px; color:#888; cursor:pointer;
      border:1px solid #3a3a3a; border-radius:4px; background:#242424; }
    .sf-lip-mode.on { color:#fff; border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-lip-trigger { flex:1; min-width:0; display:flex; align-items:center; gap:4px;
      padding:5px 8px; background:#1d1d1d; color:#ccc; border:1px solid #666; border-radius:4px;
      font-size:11px; font-family:inherit; cursor:pointer; overflow:hidden; }
    .sf-lip-trigger:hover { border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-lip-trigger .name { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;
      white-space:nowrap; text-align:left; }
    .sf-lip-trigger .arrow { color:#888; font-size:9px; flex:none; }
    .sf-lip-nav { flex:none; width:26px; padding:4px 0; }
    .sf-lip-nav:active { background:#3a3a3a; }
    .sf-lip-popup { position:fixed; z-index:99999; background:#1e1e1e; border:1px solid #555;
      border-radius:6px; box-shadow:0 6px 18px rgba(0,0,0,.5); overflow:hidden; font-size:11px; }
    .sf-lip-pop-head { padding:6px 10px; background:#262626; color:#aaa; border-bottom:1px solid #3a3a3a;
      font-size:10px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .sf-lip-pop-item { padding:6px 10px; color:#ccc; cursor:pointer; display:flex; gap:6px;
      align-items:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .sf-lip-pop-item:hover { background:#333; color:#fff; }
    .sf-lip-pop-empty { padding:10px; color:#888; text-align:center; }
    .sf-lip-input { flex:1; min-width:0; background:#1d1d1d; color:#ccc; border:1px solid #666;
      border-radius:4px; outline:0; padding:5px 6px; font-size:11px; font-family:inherit; }
    .sf-lip-input:focus { border-color:${"var(--sf-acc, #f66744)"}; }
    .sf-lip-crumbs { flex:1; min-width:0; display:flex; align-items:center; gap:2px;
      overflow:hidden; white-space:nowrap; font-size:11px; color:#aaa; }
    .sf-lip-crumb { padding:2px 5px; border-radius:3px; cursor:pointer; color:#ccc; }
    .sf-lip-crumb:hover { background:#333; color:#fff; }
    .sf-lip-crumb-sep { color:#666; flex:none; }
    `;
    document.head.appendChild(style);
}

// ── 主扩展 ───────────────────────────────────────────────────────────────
app.registerExtension({
    name: "sfnodes.load_images_path",

    nodeCreated(node) {
        if (node?.comfyClass !== "SFLoadImagesPath") return;
        injectCSS();

        const folderWidget = node.widgets?.find((w) => w.name === "folder");
        if (!folderWidget) return;

        // 隐藏原生 combo：值仍是数据通道（随 workflow 保存 + 自动收集）
        folderWidget.hidden = true;
        folderWidget.computeSize = () => [0, -4];
        applyAdaptiveCanvasOnly(folderWidget);
        const hideEl = () => {
            const el = folderWidget.element || folderWidget.inputEl;
            if (el) el.style.display = "none";
        };
        hideEl();
        requestAnimationFrame(hideEl);

        // ── DOM UI ──
        const root = document.createElement("div");
        root.className = "sf-lip-root";

        // 内容真实高度（6 行 + padding + gap，~140-170px 随模式切换变化）。
        // 硬编码 138 曾让 footRow（刷新按钮行）永远落在节点边框外。
        // 首帧未布局 / 组折叠隐藏时 offsetHeight 全为 0：返回上次良好值兜底，
        // 防止高度塌缩（sf_load_image measureH 同款防塌缩）。
        let _lastGoodH = 138;
        const measureHeight = () => {
            let totalH = 0;
            let visible = 0;
            for (const child of root.children) {
                const style = window.getComputedStyle(child);
                if (style.position === "absolute" || style.position === "fixed") continue;
                if (style.display === "none") continue;
                totalH += child.offsetHeight;
                visible += 1;
            }
            const padding = 16; // root padding 8×2
            const gaps = Math.max(0, visible - 1) * 6; // flex gap 6px
            if (totalH < 20) return _lastGoodH;
            _lastGoodH = totalH + padding + gaps;
            return _lastGoodH;
        };

        let _currentSubdirs = [];   // 当前层子目录（渐进式按需加载）

        const currentValue = () => String(folderWidget.value || "");

        // 显式模式状态（随 workflow 保存的 properties，不注入 prompt）。
        // 不能用值推导：路径模式下值可能仍是目录格式（如 "input/faces"）。
        const getMode = () => (node.properties?.sfLoadImagesPathMode === "path" ? "path" : "dir");
        const setMode = (m) => {
            if (!node.properties) node.properties = {};
            node.properties.sfLoadImagesPathMode = m;
        };

        // 进入目录：面包屑前进（值 = source/parts.../sub）
        const enterSubdir = (sub) => {
            if (!sub) return;
            const st = parseFolderValue(currentValue());
            const parts = st.sub ? st.sub.split("/") : [];
            setValue(dirValue(st.source, [...parts, sub].join("/")));
        };
        // 同级切换：替换面包屑最后一段（不改变层级深度）
        const switchSibling = (sub) => {
            if (!sub) return;
            const st = parseFolderValue(currentValue());
            const parts = st.sub ? st.sub.split("/") : [];
            setValue(dirValue(st.source, [...parts.slice(0, -1), sub].join("/")));
        };
        // 回退到祖先层：parts 截断到 length 段
        const goToLevel = (length) => {
            const st = parseFolderValue(currentValue());
            const parts = pathParts(currentValue());
            setValue(dirValue(st.source, parts.slice(0, length).join("/")));
        };

        // 拉取指定目录的下一级子目录（无竞态缓存，供同级切换用）
        const fetchSubdirs = async (folderVal) => {
            try {
                const resp = await fetch(sfApiUrl(`/api/sfnodes/images_path/subdirs?folder=${encodeURIComponent(folderVal)}`));
                const data = resp.ok ? await resp.json() : null;
                return Array.isArray(data?.subdirs) ? data.subdirs : [];
            } catch {
                return [];
            }
        };

        // ── 渲染：源/模式/面包屑/值（同步部分）──
        const renderFromValue = () => {
            const raw = currentValue();
            const st = parseFolderValue(raw);
            const mode = getMode();
            const parts = pathParts(raw);

            for (const b of root.querySelectorAll("[data-role='src']")) {
                b.classList.toggle("on", mode === "dir" && b.dataset.src === st.source);
            }
            for (const b of root.querySelectorAll("[data-role='mode']")) {
                b.classList.toggle("on", b.dataset.mode === mode);
            }

            const dirRow = root.querySelector("[data-role='dir-row']");
            if (dirRow) dirRow.style.display = mode === "dir" ? "" : "none";
            const crumbRow = root.querySelector("[data-role='crumb-row']");
            if (crumbRow) crumbRow.style.display = mode === "dir" ? "" : "none";
            const pathRow = root.querySelector("[data-role='path-row']");
            if (pathRow) pathRow.style.display = mode === "path" ? "" : "none";
            const input = root.querySelector("[data-role='path-input']");
            if (input && mode === "path" && document.activeElement !== input) {
                input.value = raw;
            }

            // 面包屑：source ▸ 各级（仅目录模式渲染；路径模式 st.source 无值）
            const crumbs = root.querySelector("[data-role='crumbs']");
            if (crumbs) {
                crumbs.innerHTML = "";
                if (mode === "dir") {
                    const srcLabel = document.createElement("span");
                    srcLabel.className = "sf-lip-crumb";
                    srcLabel.textContent = st.source;
                    srcLabel.title = "回到根目录";
                    srcLabel.addEventListener("click", () => setValue(dirValue(st.source, "")));
                    crumbs.appendChild(srcLabel);
                    parts.forEach((p, i) => {
                        const sep = document.createElement("span");
                        sep.className = "sf-lip-crumb-sep";
                        sep.textContent = "▸";
                        crumbs.appendChild(sep);
                        const seg = document.createElement("span");
                        seg.className = "sf-lip-crumb";
                        seg.textContent = p;
                        if (i < parts.length - 1) {
                            seg.title = "回到这一层";
                            seg.addEventListener("click", () => goToLevel(i + 1));
                        }
                        crumbs.appendChild(seg);
                    });
                }
            }

            // 左右同级切换：根层（无父层）无同级可切 → 禁用
            const prevBtn = root.querySelector("[data-role='dir-prev']");
            const nextBtn = root.querySelector("[data-role='dir-next']");
            if (prevBtn) prevBtn.disabled = mode !== "dir" || parts.length === 0;
            if (nextBtn) nextBtn.disabled = mode !== "dir" || parts.length === 0;

            // 下拉按钮：name 显示当前目录名（末段或源根），title 完整路径
            const trigger = root.querySelector("[data-role='dir-trigger']");
            if (trigger) {
                const nameEl = trigger.querySelector(".name");
                if (nameEl) {
                    nameEl.textContent = mode === "dir"
                        ? (parts[parts.length - 1] || st.source)
                        : "—";
                }
                trigger.title = raw;
            }

            // 值变化 → 重新加载当前层子目录（loadCurrentSubdirs 内部有
            // 同值缓存，重复渲染不会重复请求）
            if (mode === "dir") loadCurrentSubdirs();
        };

        // ── 渲染：下拉按钮状态（当前目录名 + 子目录计数）──
        const renderSubdirs = () => {
            const trigger = root.querySelector("[data-role='dir-trigger']");
            if (!trigger) return;
            const counterEl = trigger.querySelector("[data-role='dir-count']");
            if (counterEl) {
                counterEl.textContent = _currentSubdirs.length ? `${_currentSubdirs.length} 目录` : "";
            }
            // popup 打开时同步其内容
            if (_openPopup) renderDirPopup(_openPopup);
        };

        // ── 目录 popup（SFLoadImageResize 下拉风格：锚点下方 fixed 列表）──
        let _openPopup = null;
        const closeDirPopup = () => {
            if (!_openPopup) return;
            const popup = _openPopup;
            _openPopup = null;
            popup.remove();
            document.removeEventListener("mousedown", onDocDown, true);
            document.removeEventListener("pointerdown", onDocDown, true);
            document.removeEventListener("wheel", onWheelClose, true);
            document.removeEventListener("keydown", onKeyClose, true);
        };
        const onDocDown = (e) => { if (!_openPopup || !_openPopup.contains(e.target)) closeDirPopup(); };
        const onWheelClose = (e) => { if (!_openPopup || !_openPopup.contains(e.target)) closeDirPopup(); };
        const onKeyClose = (e) => { if (e.key === "Escape") closeDirPopup(); };

        const renderDirPopup = (popup) => {
            const listEl = popup.querySelector("[data-role='pop-list']");
            if (!listEl) return;
            listEl.innerHTML = "";
            if (!_currentSubdirs.length) {
                const empty = document.createElement("div");
                empty.className = "sf-lip-pop-empty";
                empty.textContent = "（无子目录）";
                listEl.appendChild(empty);
                return;
            }
            for (const s of _currentSubdirs) {
                const item = document.createElement("div");
                item.className = "sf-lip-pop-item";
                item.textContent = `📁 ${s}`;
                item.addEventListener("click", () => {
                    closeDirPopup();
                    enterSubdir(s);
                });
                listEl.appendChild(item);
            }
        };

        const openDirPopup = () => {
            if (_openPopup) closeDirPopup();
            const trigger = root.querySelector("[data-role='dir-trigger']");
            if (!trigger) return;
            const popup = document.createElement("div");
            popup.className = "sf-lip-popup";
            const rect = trigger.getBoundingClientRect();
            const width = Math.max(rect.width, 240);
            Object.assign(popup.style, {
                left: `${rect.left}px`,
                top: `${rect.bottom + 2}px`,
                width: `${width}px`,
            });
            const head = document.createElement("div");
            head.className = "sf-lip-pop-head";
            head.textContent = `📁 ${currentValue() || "—"}`;
            popup.appendChild(head);
            const listEl = document.createElement("div");
            listEl.dataset.role = "pop-list";
            popup.appendChild(listEl);
            document.body.appendChild(popup);
            _openPopup = popup;
            renderDirPopup(popup);
            setTimeout(() => {
                document.addEventListener("mousedown", onDocDown, true);
                document.addEventListener("pointerdown", onDocDown, true);
                document.addEventListener("wheel", onWheelClose, true);
                document.addEventListener("keydown", onKeyClose, true);
            }, 0);
        };

        let _subdirsReq = 0;
        let _lastFetched = null;   // 同值缓存：重复渲染/恢复不重复请求
        const loadCurrentSubdirs = async (force = false) => {
            const v = currentValue();
            if (!force && v === _lastFetched) return;
            _lastFetched = v;
            const myReq = ++_subdirsReq;
            const list = await fetchSubdirs(v);
            if (myReq !== _subdirsReq) return;   // 竞态：快速切换时旧响应丢弃
            _currentSubdirs = list;
            renderSubdirs();
        };

        // ── 左右快速切换：在同级目录（父层的兄弟）间循环步进，
        // 不改变层级深度。根层（无父层）由渲染层禁用按钮。──
        const stepSubdir = async (prev) => {
            const st = parseFolderValue(currentValue());
            const parts = st.sub ? st.sub.split("/") : [];
            if (!parts.length) return;   // 根层无同级
            const parentVal = dirValue(st.source, parts.slice(0, -1).join("/"));
            const list = await fetchSubdirs(parentVal);
            if (!list.length) return;
            const cur = parts[parts.length - 1];
            let idx = list.indexOf(cur);
            if (idx < 0) idx = 0;
            idx = (idx + (prev ? -1 : 1) + list.length) % list.length;
            switchSibling(list[idx]);
        };

        const setValue = (v) => {
            folderWidget.value = v;
            renderFromValue();
            if (app.graph) app.graph.setDirtyCanvas(true, true);
        };

        // ── 源切换（三档）──
        const srcRow = document.createElement("div");
        srcRow.className = "sf-lip-row";
        for (const s of SOURCES) {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "sf-lip-btn";
            b.dataset.role = "src";
            b.dataset.src = s;
            b.textContent = s === "input" ? "IN · input" : s === "output" ? "OUT · output" : "IMAGES";
            b.addEventListener("click", () => {
                // 点源按钮 = 切到目录模式并回到该源根
                setMode("dir");
                setValue(dirValue(s, ""));
            });
            srcRow.appendChild(b);
        }
        root.appendChild(srcRow);

        // ── 模式切换：目录选择 / 直接输入路径 ──
        const modeRow = document.createElement("div");
        modeRow.className = "sf-lip-row";
        for (const m of [["dir", "Folder Mode"], ["path", "Path Mode"]]) {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "sf-lip-btn sf-lip-mode";
            b.dataset.role = "mode";
            b.dataset.mode = m[0];
            b.textContent = m[1];
            b.addEventListener("click", () => {
                if (m[0] === "dir") {
                    setMode("dir");
                    const cur = parseFolderValue(currentValue());
                    setValue(cur.mode === "dir" ? currentValue() : dirValue("images", ""));
                } else {
                    setMode("path");
                    setValue(currentValue());
                }
            });
            modeRow.appendChild(b);
        }
        root.appendChild(modeRow);

        // ── 面包屑 ──
        const crumbRow = document.createElement("div");
        crumbRow.className = "sf-lip-row";
        crumbRow.dataset.role = "crumb-row";
        const crumbs = document.createElement("div");
        crumbs.className = "sf-lip-crumbs";
        crumbs.dataset.role = "crumbs";
        crumbRow.appendChild(crumbs);
        root.appendChild(crumbRow);

        // ── 目录模式：面包屑 + 当前层子目录下拉 + 左右步进 ──
        const dirRow = document.createElement("div");
        dirRow.className = "sf-lip-row";
        dirRow.dataset.role = "dir-row";
        const navBtn = (label, title, role, prev) => {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "sf-lip-btn sf-lip-nav";
            b.dataset.role = role;
            b.textContent = label;
            b.title = title;
            b.addEventListener("click", () => stepSubdir(prev));
            return b;
        };
        const prevBtn = navBtn("◀", "上一个同级目录", "dir-prev", true);
        const nextBtn = navBtn("▶", "下一个同级目录", "dir-next", false);
        const trigger = document.createElement("button");
        trigger.type = "button";
        trigger.className = "sf-lip-trigger";
        trigger.dataset.role = "dir-trigger";
        const triggerName = document.createElement("span");
        triggerName.className = "name";
        triggerName.textContent = "—";
        const triggerCount = document.createElement("span");
        triggerCount.className = "arrow";
        triggerCount.dataset.role = "dir-count";
        triggerCount.textContent = "";
        const triggerArrow = document.createElement("span");
        triggerArrow.className = "arrow";
        triggerArrow.textContent = "▼";
        trigger.append(triggerName, triggerCount, triggerArrow);
        trigger.addEventListener("click", () => {
            loadCurrentSubdirs(true);   // 打开时刷新当前层（幂等：缓存命中直接返回）
            openDirPopup();
        });
        dirRow.append(prevBtn, trigger, nextBtn);
        root.appendChild(dirRow);

        // ── 路径模式：输入框 + 应用 ──
        const pathRow = document.createElement("div");
        pathRow.className = "sf-lip-row";
        pathRow.dataset.role = "path-row";
        const input = document.createElement("input");
        input.type = "text";
        input.className = "sf-lip-input";
        input.dataset.role = "path-input";
        input.placeholder = "绝对路径 或 input/... · output/... · images/...";
        const applyBtn = document.createElement("button");
        applyBtn.type = "button";
        applyBtn.className = "sf-lip-btn";
        applyBtn.textContent = "Apply";
        applyBtn.addEventListener("click", () => setValue(input.value.trim()));
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") setValue(input.value.trim());
        });
        pathRow.append(input, applyBtn);
        root.appendChild(pathRow);

        // ── 底部：刷新 ──
        const footRow = document.createElement("div");
        footRow.className = "sf-lip-row";
        const refreshBtn = document.createElement("button");
        refreshBtn.type = "button";
        refreshBtn.className = "sf-lip-btn";
        refreshBtn.textContent = "Refresh";
        refreshBtn.addEventListener("click", () => {
            loadCurrentSubdirs(true);   // 强制重新加载当前层
            if (app.graph) app.graph.setDirtyCanvas(true, true);
        });
        footRow.append(refreshBtn);
        root.appendChild(footRow);

        const widget = node.addDOMWidget("lip_ui", WIDGET_TYPE, root, {
            serialize: false,
            getMinHeight: measureHeight,
            getMaxHeight: measureHeight,
            margin: 4,
        });
        applyAdaptiveCanvasOnly(widget);
        // Nodes 2.0 忽略 legacy getMinHeight/getMaxHeight，改走 computeLayoutSize：
        // 同样锁住内容高度下限，并借 minWidth 兜住拖拽宽度（Vue 下 onResize 不可靠）。
        widget.computeLayoutSize = () => ({ minHeight: measureHeight(), minWidth: MIN_W });

        // 最小宽度钳制：初始只抬升过小的尺寸（已保存宽度永不变更 -> 不脏加载）；
        // legacy 拖拽路径由 onResize 兜底。
        if (!node.size || node.size[0] < MIN_W) node.size[0] = MIN_W;
        const origResize = node.onResize;
        node.onResize = function (size) {
            if (size && size[0] < MIN_W) size[0] = MIN_W;
            return origResize?.apply(this, arguments);
        };

        // 初始渲染（combo 默认值已在 INPUT_TYPES 提供）
        renderFromValue();

        // 工作流加载：nodeCreated 早于 widget 值恢复，延迟到 configure 后补同步。
        // 渲染只读（不改序列化状态），无需 isGraphLoading 门控——门控会跳过
        // 尾窗内的恢复渲染，导致 DOM 停在初始状态。
        const sync = () => renderFromValue();
        queueMicrotask(sync);
        setTimeout(sync, 250);
        const origCfg = node.onConfigure;
        node.onConfigure = function (data) {
            const r = origCfg?.apply(this, arguments);
            queueMicrotask(sync);
            setTimeout(sync, 250);
            return r;
        };
    },
});
