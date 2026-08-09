// ==========================================================================
// SF LoRA Stack - 浮动齿轮设置面板（节点旁的主题化面板，标题可拖动，外部
// 点击或 Esc 关闭）。每节点偏好；"Set as default" 把它们存为新节点默认。
// ==========================================================================
import { app } from "/scripts/app.js";
import {
    readState, writeState, accentOf, saveDefaults, roundStrength, BRAND,
} from "./sf_lora_stack_core.js";
import { getCivitaiAccount, setCivitaiAccount } from "./sf_lora_stack_api.js";
import { repaintAll } from "./sf_lora_stack_render.js";

let _panel = null;
let _panelNode = null;
let _refresh = null;
let _followRaf = null;   // 画布跟随循环，见 startFollowing()
let _userMoved = false;  // 用户拖过面板，停止跟随

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

function injectCSS() {
    if (document.getElementById("sf-lsp-css")) return;
    const s = document.createElement("style");
    s.id = "sf-lsp-css";
    s.textContent = `
    .sf-lsp { position:fixed; z-index:10010; width:290px; max-width:94vw; background:#1a1a1a;
      border:1px solid #4a4a4a; border-radius:10px; box-shadow:0 18px 50px rgba(0,0,0,0.6);
      color:#d8d8d8; font:12px 'Segoe UI',system-ui,sans-serif; overflow:hidden; }
    .sf-lsp-t { display:flex; align-items:center; gap:8px; padding:10px 12px; background:#232323;
      border-bottom:1px solid #333; cursor:grab; user-select:none; color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-lsp-t .x { margin-left:auto; color:#8a8a8a; cursor:pointer; padding:0 4px; }
    .sf-lsp-t .x:hover { color:#fff; }
    .sf-lsp-b { padding:12px; display:flex; flex-direction:column; gap:11px; max-height:64vh; overflow-y:auto; }
    .sf-lsp-row { display:flex; align-items:center; gap:10px; }
    .sf-lsp-row .lab { flex:1; color:#c2c2c2; }
    .sf-lsp-row .hint { display:block; font-size:10px; color:#7a7a7a; margin-top:1px; }
    .sf-lsp-num { width:66px; box-sizing:border-box; background:#161616; border:1px solid #4a4a4a;
      border-radius:6px; color:#fff; text-align:center; font:12px monospace; padding:6px 4px; outline:none; }
    .sf-lsp-num:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-lsp-txt { width:70px; box-sizing:border-box; background:#161616; border:1px solid #4a4a4a;
      border-radius:6px; color:#fff; text-align:center; font:12px monospace; padding:6px 4px; outline:none; }
    .sf-lsp-txt:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-lsp-sw { flex:0 0 auto; width:34px; height:18px; border-radius:99px; background:#3a3a3a;
      position:relative; cursor:pointer; border:1px solid #000; }
    .sf-lsp-sw::after { content:""; position:absolute; top:1px; left:1px; width:14px; height:14px;
      border-radius:50%; background:#8a8a8a; transition:left .14s, background .14s; }
    .sf-lsp-sw.on { background:var(--acc, var(--sf-acc, #f66744)); } .sf-lsp-sw.on::after { left:17px; background:#fff; }
    .sf-lsp-swatch { width:30px; height:22px; border-radius:5px; border:1px solid #555; cursor:pointer; flex:0 0 auto; }
    .sf-lsp-swatch:hover { border-color:#fff; }
    .sf-lsp-seg { flex:0 0 auto; display:flex; background:rgba(0,0,0,0.25); border:1px solid #444;
      border-radius:6px; overflow:hidden; }
    .sf-lsp-segb { padding:5px 9px; font:11px 'Segoe UI',sans-serif; color:#aaa; cursor:pointer;
      user-select:none; }
    .sf-lsp-segb:hover { color:#ddd; background:rgba(255,255,255,0.08); }
    .sf-lsp-segb.on { background:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-lsp-f { display:flex; gap:8px; padding:10px 12px; border-top:1px solid #333; background:#1f1f1f; }
    .sf-lsp-btn { border:1px solid #444; background:rgba(255,255,255,0.04); color:#d8d8d8; border-radius:5px;
      padding:6px 12px; font:12px 'Segoe UI',sans-serif; cursor:pointer; }
    .sf-lsp-btn:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-lsp-push { margin-left:auto; }
    /* Civitai 块。其上方都是每节点；这些只在本机存一次，用一条规则和一个
       标题说明这一点。 */
    .sf-lsp-head { margin-top:2px; padding-top:11px; border-top:1px solid #333;
      color:var(--acc, var(--sf-acc, #f66744)); font-size:11px; letter-spacing:.04em; text-transform:uppercase; }
    .sf-lsp-head .sub { display:block; margin-top:3px; text-transform:none; letter-spacing:0;
      color:#7a7a7a; font-size:10px; line-height:1.4; }
    .sf-lsp-key { flex:1; min-width:0; box-sizing:border-box; background:#161616;
      border:1px solid #4a4a4a; border-radius:6px; color:#fff; font:12px monospace;
      padding:6px 8px; outline:none; }
    .sf-lsp-key:focus { border-color:var(--acc, var(--sf-acc, #f66744)); }
    .sf-lsp-mini { flex:0 0 auto; border:1px solid #444; background:rgba(255,255,255,0.04);
      color:#d8d8d8; border-radius:5px; padding:5px 9px; font:11px 'Segoe UI',sans-serif;
      cursor:pointer; user-select:none; }
    .sf-lsp-mini:hover { border-color:var(--acc, var(--sf-acc, #f66744)); color:#fff; }
    .sf-lsp-state { flex:1; font-size:11px; color:#7a7a7a; }
    .sf-lsp-state.set { color:#3ec371; }
    .sf-lsp-msg { font-size:10px; line-height:1.4; color:#c98a6a; }
    /* 强调色小色板：点色块直接设置，另含重置品牌橙。 */
    .sf-lsp-pal { display:flex; flex-wrap:wrap; gap:5px; }
    .sf-lsp-palb { width:20px; height:20px; border-radius:4px; border:1px solid rgba(255,255,255,0.2);
      cursor:pointer; flex:0 0 auto; }
    .sf-lsp-palb:hover { border-color:#fff; transform:scale(1.12); }
  `;
    document.head.appendChild(s);
}

// 导出供信息面板以相同几何放置（含下面 Classic 回退——
// [data-node-id] 元素只存在于 Nodes 2.0）。
export function getNodeRect(node) {
    if (node?.id != null) {
        const e = document.querySelector(`[data-node-id="${node.id}"]`);
        if (e) return e.getBoundingClientRect();
    }
    const c = app.canvas, ds = c?.ds, cv = c?.canvas;
    if (!ds || !cv || !node?.pos || !node?.size) return null;
    const cr = cv.getBoundingClientRect();
    const titleH = window.LiteGraph?.NODE_TITLE_HEIGHT || 30;
    const sc = ds.scale || 1, off = ds.offset || [0, 0];
    const left = cr.left + (node.pos[0] + off[0]) * sc;
    const top = cr.top + (node.pos[1] - titleH + off[1]) * sc;
    return { left, top, right: left + node.size[0] * sc, bottom: top + (node.size[1] + titleH) * sc };
}

function placeBeside(panel, rect) {
    const vw = window.innerWidth, vh = window.innerHeight, mw = panel.offsetWidth, mh = panel.offsetHeight;
    const gap = 12, pad = 8;
    if (!rect) { panel.style.left = Math.max(pad, (vw - mw) / 2) + "px"; panel.style.top = Math.max(pad, (vh - mh) / 2) + "px"; return; }
    let left = rect.right + gap;
    if (left + mw > vw - pad) left = rect.left - gap - mw;
    if (left < pad) left = Math.max(pad, vw - mw - pad);
    let top = Math.min(rect.top, vh - mh - pad);
    panel.style.left = left + "px";
    panel.style.top = Math.max(pad, top) + "px";
}

/**
 * 画布移动时让面板跟随其节点。
 *
 * 没有它，面板被写在固定屏幕位置一次后留在那里，缩放/平移会把它扔到
 * 不相关的地方——画布上有两个本节点时更是无从说清在编辑哪个。
 *
 * rAF 循环而非事件：LiteGraph 对变换变化不发任何事件，缩放必须平滑跟随
 * 而非事后追。每帧比三个数字即返回，空闲成本为零，只在面板打开时运行。
 *
 * 用户拖走面板即停：从那以后它就在用户放的地方，从用户手底下移走更糟。
 */
function startFollowing(panel, node) {
    let lastScale = null, lastX = null, lastY = null;
    const tick = () => {
        if (!_panel || _panel !== panel || !panel.isConnected) { _followRaf = null; return; }
        _followRaf = requestAnimationFrame(tick);
        if (_userMoved) return;
        const ds = app.canvas?.ds;
        if (!ds) return;
        const sc = ds.scale || 1;
        const ox = ds.offset?.[0] ?? 0, oy = ds.offset?.[1] ?? 0;
        // 先廉价数字比较：几乎每帧都在这返回，querySelector 只在画布真动时跑。
        if (sc === lastScale && ox === lastX && oy === lastY) return;
        lastScale = sc; lastX = ox; lastY = oy;
        placeBeside(panel, getNodeRect(node));
    };
    _followRaf = requestAnimationFrame(tick);
}

function stopFollowing() {
    if (_followRaf != null) cancelAnimationFrame(_followRaf);
    _followRaf = null;
}

function makeDraggable(panel, handle) {
    handle.addEventListener("pointerdown", (e) => {
        if (e.target.closest(".x")) return;
        e.preventDefault();
        const r = panel.getBoundingClientRect();
        const ox = e.clientX - r.left, oy = e.clientY - r.top;

        // 防拖拽粘住光标的两道防线：pointerup 真会丢（窗口外/第二显示器/
        // 被上游吞掉）。
        try { handle.setPointerCapture(e.pointerId); } catch { /* 不可捕获 */ }

        const move = (ev) => {
            if (!panel.isConnected) return up();
            // 按钮已松开：释放丢失，结束拖拽。
            if (!(ev.buttons & 1)) return up();
            // 从这里起面板在用户放的地方，停止跟随节点。
            _userMoved = true;
            panel.style.left = Math.max(0, Math.min(window.innerWidth - panel.offsetWidth, ev.clientX - ox)) + "px";
            panel.style.top = Math.max(0, Math.min(window.innerHeight - panel.offsetHeight, ev.clientY - oy)) + "px";
        };
        // 幂等：按钮守卫可调它，真实释放也可调它。
        let done = false;
        const up = () => {
            if (done) return;
            done = true;
            try { handle.releasePointerCapture(e.pointerId); } catch { /* 已离开 */ }
            handle.removeEventListener("pointermove", move, true);
            handle.removeEventListener("pointerup", up, true);
            handle.removeEventListener("pointercancel", up, true);
            handle.removeEventListener("lostpointercapture", up, true);
        };
        handle.addEventListener("pointermove", move, true);
        handle.addEventListener("pointerup", up, true);
        handle.addEventListener("pointercancel", up, true);
        handle.addEventListener("lostpointercapture", up, true);
    });
}

function outsideClose(e) {
    if (!_panel) return;
    if (_panel.contains(e.target)) return;
    closeLoraPanel();
}
function escClose(e) {
    if (e.key === "Escape" && _panel) {
        e.stopPropagation();
        closeLoraPanel();
    }
}

export function closeLoraPanel() {
    stopFollowing();
    // 在关闭而不是打开时重置：用户拖过的面板不能让下一个学着静坐。
    _userMoved = false;
    if (_panel) { try { _panel.remove(); } catch { /* 忽略 */ } }
    _panel = null; _panelNode = null; _refresh = null;
    document.removeEventListener("pointerdown", outsideClose, true);
    document.removeEventListener("keydown", escClose, true);
}
export function closeLoraPanelFor(node) { if (_panelNode === node) closeLoraPanel(); }

export function openLoraPanel(node, refresh) {
    closeLoraPanel();
    injectCSS();
    _panelNode = node;
    _refresh = refresh || null;

    const panel = el("div", "sf-lsp");
    panel.style.setProperty("--acc", accentOf(node));

    const title = el("div", "sf-lsp-t");
    title.append(el("span", null, "⚙"), el("span", null, "LoRA Stack settings"));
    const x = el("span", "x", "✕");
    x.addEventListener("click", closeLoraPanel);
    title.appendChild(x);

    const body = el("div", "sf-lsp-b");

    const fire = () => { _refresh?.(false); };
    const set = (patch) => { writeState(node, { ...readState(node), ...patch }); };

    // 开关行助手
    function toggleRow(label, hint, key, invert = false) {
        const row = el("div", "sf-lsp-row");
        const l = el("div", "lab"); l.append(el("span", null, label));
        if (hint) { const h = el("span", "hint", hint); l.appendChild(h); }
        const sw = el("div", "sf-lsp-sw");
        const cur = () => { const v = !!readState(node)[key]; return invert ? !v : v; };
        const paint = () => sw.classList.toggle("on", cur());
        paint();
        sw.addEventListener("click", () => {
            const next = !cur();
            set({ [key]: invert ? !next : next });
            paint();
            fire();
        });
        row.append(l, sw);
        return row;
    }

    // 数字行助手
    function numRow(label, key, { min = 0, round = null } = {}) {
        const row = el("div", "sf-lsp-row");
        row.appendChild(el("div", "lab", label));
        const inp = el("input", "sf-lsp-num");
        inp.type = "text";
        inp.value = String(readState(node)[key]);
        inp.addEventListener("keydown", (e) => e.stopPropagation());
        inp.addEventListener("change", () => {
            let v = parseFloat(inp.value);
            if (!Number.isFinite(v)) v = readState(node)[key];
            if (round) v = round(v);
            if (v < min) v = min;
            set({ [key]: v });
            inp.value = String(readState(node)[key]);
            fire();
        });
        row.appendChild(inp);
        return row;
    }

    body.appendChild(numRow("Default strength (new LoRAs)", "defStrength", { min: -10, round: roundStrength }));
    body.appendChild(numRow("Strength step (arrows)", "step", { min: 0.001 }));
    body.appendChild(toggleRow("Separate model / clip strength",
        "Show two strengths per row", "linkStrength", true));

    // 分隔符（文本）
    const sepRow = el("div", "sf-lsp-row");
    sepRow.appendChild(el("div", "lab", "Trigger words separator"));
    const sepIn = el("input", "sf-lsp-txt");
    sepIn.type = "text";
    sepIn.value = readState(node).sep;
    sepIn.title = "Text placed between trigger words in the output (e.g. \", \")";
    sepIn.addEventListener("keydown", (e) => e.stopPropagation());
    sepIn.addEventListener("change", () => { set({ sep: sepIn.value }); fire(); });
    sepRow.appendChild(sepIn);
    body.appendChild(sepRow);

    // 内存模式——三档分段选择
    function segRow(label, key, options) {
        const row = el("div", "sf-lsp-row");
        const l = el("div", "lab"); l.append(el("span", null, label));
        const hint = el("span", "hint", "");
        l.appendChild(hint);
        const wrap = el("div", "sf-lsp-seg");
        const paint = () => {
            const cur = readState(node)[key];
            for (const b of wrap.children) b.classList.toggle("on", b.dataset.v === cur);
            const o = options.find((x) => x.v === readState(node)[key]);
            hint.textContent = o ? o.hint : "";
        };
        for (const o of options) {
            const b = el("div", "sf-lsp-segb", o.label);
            b.dataset.v = o.v;
            b.title = o.title;
            b.addEventListener("click", () => { set({ [key]: o.v }); paint(); fire(); });
            wrap.appendChild(b);
        }
        paint();
        row.append(l, wrap);
        return row;
    }
    body.appendChild(segRow("LoRA memory use", "cacheMode", [
        { v: "last", label: "Standard", hint: "Keeps the last used LoRA in memory, like ComfyUI",
            title: "Balanced default: one LoRA stays loaded between runs" },
        { v: "all", label: "Fast", hint: "Keeps the whole stack in memory for quick re-runs",
            title: "Fastest re-runs; big stacks can hold gigabytes of RAM" },
        { v: "none", label: "Lowest", hint: "Re-reads the files on every run",
            title: "Smallest memory footprint, best for low-RAM machines" },
    ]));

    body.appendChild(toggleRow("Hide file extension",
        "Show the LoRA name without .safetensors", "hideExt"));
    body.appendChild(toggleRow("Civitai lookup button",
        "Show the optional online lookup in the info panel", "civitai"));
    body.appendChild(toggleRow("Show preview thumbnails",
        "In the info panel", "thumbs"));

    // ── Civitai 账户 ────────────────────────────────────────────────────────
    // 为什么存在：Civitai 对匿名 API 请求隐藏成人评级模型，用与"无此文件"
    // 相同的 404 应答。从节点无法区分，用户只看到查询永不工作。一个允许
    // 看到该内容的账户的 key 让同一请求返回记录。
    //
    // 与面板里其它行不同，这三行只在本机存一次，不在节点上——节点上的
    // key 会被写进工作流文件，分享给谁就带谁。标题说明了这一点，因为
    // footer 的 "Set as default" 就在几像素外。
    //
    // key 永不回发页面：服务器应答"是否已设置 + 后 4 位"，足以区分两个
    // key，截图里毫无用处。
    {
        const head = el("div", "sf-lsp-head", "Civitai account");
        head.appendChild(el("span", "sub",
            "Saved on this computer, never in your workflows. A key lets the lookup see "
            + "models that Civitai hides from anonymous requests."));
        body.appendChild(head);

        let acc = { configured: false, hint: "", host: "com", adultThumbs: false };

        const msg = el("div", "sf-lsp-msg");
        msg.style.display = "none";
        const say = (t, ok) => {
            msg.textContent = t || "";
            msg.style.display = t ? "block" : "none";
            msg.style.color = ok ? "#3ec371" : "";
        };

        // ── key 行，在显示与编辑间切换 ──
        // `editing` 是整个 key 行需要状态标记的原因：paintKeyRow 的第一件事
        // 就是清空行，任何在编辑器打开时发生的重绘都会毁掉输入框和粘贴的
        // key，连消息都没有。两件事都能触发重绘——点击 host 或 adult 行
        // （保存成功并重绘一切），和面板打开时发出的账户 GET 的应答。
        let editing = false;
        const keyRow = el("div", "sf-lsp-row");
        const paintKeyRow = () => {
            editing = false;
            keyRow.textContent = "";
            const state = el("div", "sf-lsp-state" + (acc.configured ? " set" : ""),
                acc.configured ? "✓ Key saved  " + acc.hint : "No key - anonymous lookups");
            const edit = el("div", "sf-lsp-mini", acc.configured ? "Change" : "Add key");
            edit.title = acc.configured
                ? "Replace the saved key with a different one"
                : "Paste a key from civitai.com > Account settings > API Keys";
            edit.addEventListener("click", showEditor);
            keyRow.append(state, edit);
            if (acc.configured) {
                const rm = el("div", "sf-lsp-mini", "Remove");
                rm.title = "Forget the key. Lookups go back to anonymous.";
                rm.addEventListener("click", () => save({ key: "" }, "Key removed."));
                keyRow.appendChild(rm);
            }
        };

        function showEditor() {
            editing = true;
            say("");
            keyRow.textContent = "";
            const inp = el("input", "sf-lsp-key");
            // 密码字段：窥屏/截图都读不到。
            inp.type = "password";
            inp.placeholder = "Paste your API key";
            inp.autocomplete = "off";
            inp.spellcheck = false;
            // 不拦的话，在这里打字会触发 ComfyUI 全局快捷键——Delete 会在
            // 粘贴中途删掉选中的节点。
            inp.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") { e.preventDefault(); commit(); }
            });
            const ok = el("div", "sf-lsp-mini", "Save");
            const no = el("div", "sf-lsp-mini", "Cancel");
            const commit = () => {
                const v = inp.value.trim();
                if (!v) { say("Nothing to save - paste a key first."); return; }
                // 唯一一个应把编辑器换回状态行的保存，且只在服务器确认后。
                save({ key: v }, "Key saved.", { closeEditor: true });
            };
            ok.addEventListener("click", commit);
            no.addEventListener("click", () => { say(""); paintKeyRow(); });
            keyRow.append(inp, ok, no);
            inp.focus();
        }

        // ── 先问哪个主机 ──
        const hostRow = el("div", "sf-lsp-row");
        const hostLab = el("div", "lab");
        hostLab.append(el("span", null, "Ask this site first"));
        const hostHint = el("span", "hint", "");
        hostLab.appendChild(hostHint);
        const hostSeg = el("div", "sf-lsp-seg");
        const HOSTS = [
            { v: "com", label: "Standard", hint: "civitai.com, then civitai.red as a backup",
                title: "The usual choice" },
            { v: "red", label: "Unrestricted", hint: "civitai.red first, for adult-rated models",
                title: "Civitai's unrestricted domain. Use this if your LoRAs are not found." },
        ];
        for (const o of HOSTS) {
            const b = el("div", "sf-lsp-segb", o.label);
            b.dataset.v = o.v;
            b.title = o.title;
            b.addEventListener("click", () => save({ host: o.v }, ""));
            hostSeg.appendChild(b);
        }
        hostRow.append(hostLab, hostSeg);

        // ── 成人预览图 ──
        const adultRow = el("div", "sf-lsp-row");
        const adultLab = el("div", "lab");
        adultLab.append(el("span", null, "Allow adult preview images"));
        adultLab.appendChild(el("span", "hint",
            "Off: a model whose pictures are all adult shows no thumbnail"));
        const adultSw = el("div", "sf-lsp-sw");
        adultSw.addEventListener("click", () => save({ adultThumbs: !acc.adultThumbs }, ""));
        adultRow.append(adultLab, adultSw);

        // 与 paint() 分离——见 save() 的失败分支。
        const paintRest = () => {
            for (const b of hostSeg.children) b.classList.toggle("on", b.dataset.v === acc.host);
            const h = HOSTS.find((x) => x.v === acc.host);
            hostHint.textContent = h ? h.hint : "";
            adultSw.classList.toggle("on", !!acc.adultThumbs);
        };
        // 编辑器打开时绝不重建 key 行——见 `editing`。
        const paint = () => { if (!editing) paintKeyRow(); paintRest(); };

        // 按服务器实际存储的应答重绘，绝不按我们以为它收下的。
        // `_accDirty`：用户已做过一次保存。打开面板时发出的 GET 应答可能
        // 慢于用户的第一次点击（GET 在 POST 之前就已生成），迟到落地会把
        // 面板从用户刚设的值跳回旧值——"设了 red 它显示 com"就是这么来的。
        let _accDirty = false;
        async function save(patch, okNote, opts) {
            const res = await setCivitaiAccount(patch);
            if (!res || !res.ok) {
                say((res && res.message) || "Could not save.");
                // 刻意不整面板重绘。paintKeyRow 从零重建该行，会丢掉编辑器和
                // 里面打好的 key——被拒的 key 曾清空输入框并关掉编辑器，让
                // 用户要重找重贴。错误正是文本最要紧的时刻。
                paintRest();
                return;
            }
            _accDirty = true;
            acc = res;
            if (opts?.closeEditor) editing = false;
            say(okNote, true);
            paint();
        }

        // msg 直接放在 key 行下面，不放在末尾：它带的每条消息都关于 key，
        // 放在最后会渲染在引发它的框下面两行无关内容处。
        body.append(keyRow, msg, hostRow, adultRow);
        paint();
        getCivitaiAccount().then((res) => {
            if (!res || !res.ok) { say("Could not read the Civitai settings."); return; }
            // 应答落地前面板可能已被关闭重开；写进分离的 DOM 无害但无意义。
            if (!keyRow.isConnected) return;
            // 用户已保存过：这次 GET 是保存前的旧快照，跳过（见 save 的
            // _accDirty 注释）。
            if (_accDirty) return;
            acc = res;
            paint();
        });
    }

    // 强调色：点色块直接设置，或重置品牌橙。
    const ACC_PALETTE = ["#f66744", "#4f7cff", "#3ec371", "#e9a53d", "#e2504a", "#a06ee0", "#3aa0b0", "#ffffff"];
    const accRow = el("div", "sf-lsp-row");
    accRow.appendChild(el("div", "lab", "Highlight colour"));
    const pal = el("div", "sf-lsp-pal");
    for (const c of ACC_PALETTE) {
        const b = el("div", "sf-lsp-palb");
        b.style.background = c;
        b.title = c === BRAND ? "Reset to the brand orange" : c;
        b.addEventListener("click", () => {
            const col = c === "#ffffff" ? null : c; // 白色色块 = 重置默认（品牌橙）
            set({ accent: col });
            panel.style.setProperty("--acc", col || BRAND);
            node._sfLsInner?.style.setProperty("--acc", col || BRAND);
            fire();
        });
        pal.appendChild(b);
    }
    accRow.appendChild(pal);
    body.appendChild(accRow);

    // footer
    const foot = el("div", "sf-lsp-f");
    const mkDefault = el("button", "sf-lsp-btn", "Set as default");
    mkDefault.title = "Use these settings for every new LoRA Stack node";
    mkDefault.addEventListener("click", async () => {
        const st = readState(node);
        const ok = await saveDefaults(st);
        mkDefault.textContent = ok ? "Saved as default" : "Could not save";
        setTimeout(() => { mkDefault.textContent = "Set as default"; }, 1200);
    });
    // 写入全局强调色设置（ComfyUI Settings 页的 sfnodes.Accent）：未单独设色
    // 的节点统一跟随。写当前生效色（node 级 > 节点默认 > 全局 > 品牌橙）。
    const mkAll = el("button", "sf-lsp-btn", "Every SF node");
    mkAll.title = "Use this colour for every SF node without its own colour (SF nodes setting)";
    mkAll.addEventListener("click", async () => {
        try {
            const col = accentOf(node);
            await app.ui.settings.setSettingValueAsync("sfnodes.Accent", col);
            repaintAll();
            mkAll.textContent = "Saved";
            setTimeout(() => { mkAll.textContent = "Every SF node"; }, 1200);
        } catch { /* 设置系统不可用 */ }
    });
    const done = el("button", "sf-lsp-btn sf-lsp-push", "Done");
    done.addEventListener("click", closeLoraPanel);
    foot.append(mkDefault, mkAll, done);

    panel.append(title, body, foot);
    document.body.appendChild(panel);
    placeBeside(panel, getNodeRect(node));
    makeDraggable(panel, title);

    setTimeout(() => {
        if (!_panel) return;
        document.addEventListener("pointerdown", outsideClose, true);
        document.addEventListener("keydown", escClose, true);
    }, 0);
    _panel = panel;
    // _panel 赋值之后：循环的第一件事是检查它拥有该面板。
    startFollowing(panel, node);
}
