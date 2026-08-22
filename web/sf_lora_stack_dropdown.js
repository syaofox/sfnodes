// ==========================================================================
// SF LoRA Stack - 可搜索 LoRA 选择弹窗（点击行的名称字段打开）。输入过滤；
// 结果按子文件夹分组。外部点击 / wheel / Esc 关闭（wheel 处理器必须跳过
// 弹窗内部事件，否则滚动列表会关掉它）。
// ==========================================================================
import { BRAND } from "./sf_lora_stack_core.js";
import { listLoras } from "./sf_lora_stack_api.js";
import { injectCSSOnce, installWheelZoomPassthrough } from "./sf_common.js";

let _pop = null;
let _cleanup = null;

function injectCSS() {
    injectCSSOnce("sf-ls-dd-css", `
    .sf-ls-dd { position:fixed; z-index:10020; width:300px; max-width:92vw; background:#242424;
      border:1px solid ${"var(--sf-acc, #f66744)"}; border-radius:9px; box-shadow:0 14px 44px rgba(0,0,0,0.6);
      overflow:hidden; font:12px 'Segoe UI',system-ui,sans-serif; color:#ddd;
      display:flex; flex-direction:column; max-height:60vh; }
    .sf-ls-dd-srch { margin:8px; display:flex; align-items:center; gap:7px; background:#161616;
      border:1px solid ${"var(--sf-acc, #f66744)"}; border-radius:6px; padding:6px 9px; }
    .sf-ls-dd-srch input { flex:1; min-width:0; background:transparent; border:0; outline:none;
      color:#fff; font:12px monospace; }
    .sf-ls-dd-srch .ic { color:#888; flex:none; }
    .sf-ls-dd-crumb { display:flex; flex-wrap:wrap; align-items:center; gap:2px; padding:3px 12px 6px;
      font:10.5px 'Segoe UI',sans-serif; color:#8a8a8a; border-bottom:1px solid #1c1c1c; }
    .sf-ls-dd-crumb .c { cursor:pointer; color:#a8a8a8; }
    .sf-ls-dd-crumb .c:hover { color:${"var(--sf-acc, #f66744)"}; }
    .sf-ls-dd-crumb .c.here { color:#e0e0e0; cursor:default; }
    .sf-ls-dd-crumb .s { color:#555; }
    .sf-ls-dd-list { overflow-y:auto; overflow-x:hidden; padding:2px 0 6px; }
    .sf-ls-dd-list::-webkit-scrollbar { width:7px; }
    .sf-ls-dd-list::-webkit-scrollbar-thumb { background:#555; border-radius:3px; }
    .sf-ls-dd-grp { font-size:9.5px; text-transform:uppercase; letter-spacing:.6px; color:#7a7a7a;
      padding:6px 12px 3px; }
    .sf-ls-dd-opt { padding:6px 12px; font:11.5px monospace; color:#bbb; cursor:pointer;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .sf-ls-dd-opt:hover { background:#2f2f2f; color:#fff; }
    .sf-ls-dd-opt.cur { color:${"var(--sf-acc, #f66744)"}; }
    .sf-ls-dd-opt .sub { color:#666; }
    .sf-ls-dd-folder { display:flex; align-items:center; gap:8px; padding:7px 12px; cursor:pointer;
      font:11.5px 'Segoe UI',sans-serif; color:#d0d0d0; }
    .sf-ls-dd-folder:hover { background:#2f2f2f; color:#fff; }
    .sf-ls-dd-folder .fi { color:#e0b24a; flex:none; }
    .sf-ls-dd-folder .nm { flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .sf-ls-dd-folder .ct { color:#777; font:10px monospace; } .sf-ls-dd-folder .ch { color:#777; }
    .sf-ls-dd-back { padding:6px 12px; cursor:pointer; color:#9a9a9a; font:11px 'Segoe UI'; }
    .sf-ls-dd-back:hover { color:${"var(--sf-acc, #f66744)"}; }
    .sf-ls-dd-empty { padding:14px 12px; color:#777; text-align:center; }
  `);
}

export function closeLoraDropdown() {
    if (_cleanup) { try { _cleanup(); } catch { /* 忽略 */ } }
    _cleanup = null;
    if (_pop) { try { _pop.remove(); } catch { /* 忽略 */ } }
    _pop = null;
}

function group(name) {
    const i = name.replace(/\\/g, "/").lastIndexOf("/");
    return i < 0 ? "" : name.slice(0, i);
}
function base(name) {
    const i = name.replace(/\\/g, "/").lastIndexOf("/");
    return i < 0 ? name : name.slice(i + 1);
}

export async function openLoraDropdown(anchorEl, opts) {
    closeLoraDropdown();
    injectCSS();
    const { current = "", accent = BRAND, onPick } = opts || {};

    const pop = document.createElement("div");
    pop.className = "sf-ls-dd";
    pop.style.borderColor = accent;
    // 弹窗内部的 var(--sf-acc, …) 用节点 accent 局部覆盖（全局设置为默认）。
    pop.style.setProperty("--sf-acc", accent);

    const srch = document.createElement("div");
    srch.className = "sf-ls-dd-srch";
    srch.style.borderColor = accent;
    const ic = document.createElement("span"); ic.className = "ic"; ic.textContent = "⌕";
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Search LoRAs…";
    input.addEventListener("keydown", (e) => { if (e.ctrlKey || e.metaKey || e.altKey) return; e.stopPropagation(); }); // 不触发画布快捷键(放行修饰键组合)
    installWheelZoomPassthrough(input); // 输入框滚轮透传(缩放画布/滚动文本, 对齐原生)
    srch.append(ic, input);

    const crumb = document.createElement("div");
    crumb.className = "sf-ls-dd-crumb";
    const list = document.createElement("div");
    list.className = "sf-ls-dd-list";
    pop.append(srch, crumb, list);
    document.body.appendChild(pop);
    _pop = pop; // 先于 await：fetch 中途重开关闭的是这个弹窗，而不是无事发生

    const r = anchorEl.getBoundingClientRect();
    pop.style.width = Math.min(Math.max(r.width, 240), 360) + "px";
    pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8)) + "px";

    // 方向在打开时定一次：比较上下两侧可用空间，选大者（相等时向下——
    // 视线习惯）。展开期间永不翻转（导航/搜索改变高度只更新 top 与
    // maxHeight，弹窗不会"一会上、一会下"跳变）。
    const upSpace = r.top - 4;   // 锚点上方可用空间（含 4px 间距）
    const downSpace = window.innerHeight - 8 - (r.bottom + 4); // 下方（含底部 8px 边距）
    const goUp = upSpace > downSpace;
    function place() {
        const h = pop.offsetHeight;
        // 上限取"所选方向可用空间"而非固定 60vh：内容超高时 list 内部
        // 滚动（overflow-y:auto），弹窗实际高度 ≤ 方向空间 → 永不越界，
        // 无需再靠 top 钳制兜底。
        const maxH = Math.min(0.6 * window.innerHeight, goUp ? upSpace : downSpace);
        pop.style.maxHeight = Math.max(40, maxH) + "px";
        if (goUp) pop.style.top = Math.max(8, r.top - 4 - h) + "px"; // 底边贴锚点、顶边延伸
        else pop.style.top = (r.bottom + 4) + "px";                  // 恒定
    }
    place();

    // 每次打开都强制重取：列表很小、服务端每次请求重新校验磁盘，会话缓存
    // 正是已报告的 bug（改名、按 R、选择器仍显示旧名直到重启）。
    const all = await listLoras(true);
    if (!pop.isConnected) return; // 加载期间被关闭

    // 文件夹导航：`curPath` 是正在浏览的文件夹（"" = 根）。在搜索框打字会
    // 覆盖导航并扁平搜索所有 LoRA（仍能找到一切，无需逐层钻）。从当前
    // LoRA 所在文件夹开始。
    let curPath = current ? group(current) : "";

    function levelItems() {
        const prefix = curPath ? curPath + "/" : "";
        const folders = new Map();
        const files = [];
        for (const name of all) {
            const norm = name.replace(/\\/g, "/");
            if (curPath && !norm.startsWith(prefix)) continue;
            const rest = norm.slice(prefix.length);
            const slash = rest.indexOf("/");
            if (slash < 0) files.push(name);
            else { const f = rest.slice(0, slash); folders.set(f, (folders.get(f) || 0) + 1); }
        }
        return { folders: [...folders.entries()].sort((a, b) => a[0].localeCompare(b[0])), files };
    }

    function fileRow(name, showSub) {
        const opt = document.createElement("div");
        opt.className = "sf-ls-dd-opt" + (name === current ? " cur" : "");
        opt.title = name;
        opt.appendChild(document.createTextNode(base(name)));
        if (showSub) {
            const g = group(name);
            if (g) { const s = document.createElement("span"); s.className = "sub"; s.textContent = " · " + g; opt.appendChild(s); }
        }
        opt.addEventListener("click", () => { onPick?.(name); closeLoraDropdown(); });
        return opt;
    }

    function folderRow(f, count) {
        const row = document.createElement("div");
        row.className = "sf-ls-dd-folder";
        const fi = document.createElement("span"); fi.className = "fi"; fi.textContent = "📁";
        const nm = document.createElement("span"); nm.className = "nm"; nm.textContent = f;
        const ct = document.createElement("span"); ct.className = "ct"; ct.textContent = String(count);
        const ch = document.createElement("span"); ch.className = "ch"; ch.textContent = "›";
        row.append(fi, nm, ct, ch);
        row.addEventListener("click", () => { curPath = curPath ? curPath + "/" + f : f; input.value = ""; renderList(); });
        return row;
    }

    function crumbLink(label, path, here) {
        const c = document.createElement("span");
        c.className = "c" + (here ? " here" : "");
        c.textContent = label;
        if (!here) c.addEventListener("click", () => { curPath = path; input.value = ""; renderList(); });
        return c;
    }

    function renderCrumb() {
        crumb.innerHTML = "";
        if (input.value.trim()) { crumb.style.display = "none"; return; }
        crumb.style.display = "";
        const parts = curPath ? curPath.split("/") : [];
        crumb.appendChild(crumbLink("All", "", parts.length === 0));
        let acc = "";
        parts.forEach((p, i) => {
            const s = document.createElement("span"); s.className = "s"; s.textContent = "›"; crumb.appendChild(s);
            acc = acc ? acc + "/" + p : p;
            crumb.appendChild(crumbLink(p, acc, i === parts.length - 1));
        });
    }

    function emptyRow(text) {
        const e = document.createElement("div");
        e.className = "sf-ls-dd-empty";
        e.textContent = text;
        list.appendChild(e);
    }

    function renderList() {
        renderCrumb();
        list.innerHTML = "";
        const q = input.value.trim().toLowerCase();
        if (q) {
            const matched = all.filter((n) => n.toLowerCase().includes(q));
            if (matched.length) {
                for (const name of matched) list.appendChild(fileRow(name, true));
            } else {
                emptyRow(all.length ? "No match." : "No LoRAs in models/loras.");
            }
        } else {
            const { folders, files } = levelItems();
            if (curPath) {
                const back = document.createElement("div");
                back.className = "sf-ls-dd-back";
                back.textContent = "‹ back";
                back.addEventListener("click", () => {
                    const i = curPath.lastIndexOf("/");
                    curPath = i < 0 ? "" : curPath.slice(0, i);
                    renderList();
                });
                list.appendChild(back);
            }
            for (const [f, count] of folders) list.appendChild(folderRow(f, count));
            for (const name of files) list.appendChild(fileRow(name, false));
            if (!folders.length && !files.length) emptyRow(all.length ? "Empty folder." : "No LoRAs in models/loras.");
        }
        // 内容变化后重新定位（方向打开时已定死，这里只更新 top/maxHeight）：
        // 目录导航/搜索/回退改变弹窗高度，而 maxHeight 按所选方向空间钳制
        // （≤ 60vh），弹窗恒完整可见、方向不跳变。
        place();
    }
    renderList();
    input.addEventListener("input", renderList);
    setTimeout(() => input.focus(), 0);

    const onDown = (e) => { if (!pop.contains(e.target)) closeLoraDropdown(); };
    const onWheel = (e) => { if (!pop.contains(e.target)) closeLoraDropdown(); };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); closeLoraDropdown(); } };
    setTimeout(() => {
        if (_pop !== pop) return; // 同一 tick 内被关闭 - 不挂孤儿监听器
        document.addEventListener("pointerdown", onDown, true);
        document.addEventListener("wheel", onWheel, true);
        document.addEventListener("keydown", onKey, true);
    }, 0);
    _cleanup = () => {
        document.removeEventListener("pointerdown", onDown, true);
        document.removeEventListener("wheel", onWheel, true);
        document.removeEventListener("keydown", onKey, true);
    };
}
