// SF Model Info 对话框（SFPowerLoraLoader / SFLoraLoader / SFLoraLoaderModelOnly 共享）
// 的 Civitai 查询功能冒烟测试（Node 直接运行：node tests/test_lora_info_civitai_smoke.js）
// 用 mock DOM/app/fetch 真实加载 web/sf_lora_info.js，验证：
//   - 对话框构建：行、footer 按钮（↻ Civitai / Account）
//   - runCivitai：found / notfound / offline 三态状态条 + 非编辑行刷新
//   - thumb_skipped 封面替换确认流（saveCivitaiThumb）
//   - runDeleteCivitai：状态条/🗑 消失、词回文件词
//   - Account 展开区：Add key -> Save -> Remove（setCivitaiAccount）
//   - 编辑态守卫：查询不覆盖编辑中的草稿
const fs = require("fs");
const os = require("os");
const path = require("path");

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

// ── mock DOM（惰性元素 + 事件记录）──
function makeEl(tag) {
    const el = {
        tagName: (tag || "div").toUpperCase(),
        style: {}, dataset: {}, children: [],
        className: "", _textContent: "", _innerHTML: "", value: "", placeholder: "",
        type: "", title: "", rows: 1, spellcheck: false, disabled: false, checked: false,
        href: "", src: "", open: false, isConnected: true, _removed: false, _parent: null,
        classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        get firstChild() { return el.children[0] || null; },
        get innerHTML() { return el._innerHTML; },
        set innerHTML(v) { el._innerHTML = v; el.children.length = 0; }, // 真实 DOM 语义：赋值即清空子节点
        // 真实 DOM 语义：textContent 赋值同样清空子节点（否则旧子节点残留，
        // 如账户行重建后旧 Save 按钮仍被 findBtn 命中）
        get textContent() { return el._textContent || ""; },
        set textContent(v) { el._textContent = String(v == null ? "" : v); el.children.length = 0; },
        append(...kids) { for (const k of kids) el.appendChild(k); },
        appendChild(c) { c._parent = el; this.children.push(c); return c; },
        prepend(...kids) { for (const k of [...kids].reverse()) el.insertBefore(k, el.children[0]); },
        insertBefore(c, ref) {
            c._parent = el;
            const i = this.children.indexOf(ref);
            if (i === -1) this.children.unshift(c); else this.children.splice(i, 0, c);
            return c;
        },
        replaceWith() {}, replaceChildren(...kids) { this.children = kids; },
        remove() {
            this._removed = true;
            if (this._parent) {
                const i = this._parent.children.indexOf(this);
                if (i !== -1) this._parent.children.splice(i, 1);
            }
        },
        contains() { return false; }, closest() { return null; },
        querySelector() { return makeEl(); }, querySelectorAll() { return []; },
        removeAttribute() {}, setAttribute() {}, getAttribute() { return null; },
        focus() {}, blur() {}, select() {}, click() {},
        getBoundingClientRect() { return { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 }; },
        scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {}, setSelectionRange() {},
        showModal() { this.open = true; }, close() { this.open = false; },
        get value() { return this._value || ""; },
        set value(v) { this._value = v; },
    };
    el._listeners = {};
    el.addEventListener = (t, f) => { (el._listeners[t] ||= []).push(f); };
    el.removeEventListener = (t, f) => { el._listeners[t] = (el._listeners[t] || []).filter(x => x !== f); };
    el.dispatchEvent = (e) => { for (const f of el._listeners[e.type] || []) f(e); };
    el.click = () => { for (const f of el._listeners.click || []) f({ stopPropagation() {}, preventDefault() {} }); };
    return el;
}

const bodyEl = { children: [], appendChild(c) { c._parent = bodyEl; this.children.push(c); return c; } };
const docListeners = {};
globalThis.document = {
    createElement(tag) { return makeEl(tag); },
    createTextNode(t) { return { nodeType: 3, textContent: String(t), children: [] }; },
    body: bodyEl,
    head: { appendChild() {} },
    addEventListener(t, f) { (docListeners[t] ||= []).push(f); },
    removeEventListener(t, f) { docListeners[t] = (docListeners[t] || []).filter((x) => x !== f); },
    dispatchEvent(e) { for (const f of docListeners[e.type] || []) f(e); },
    getElementById() { return null; },
    activeElement: makeEl(),
};
globalThis.window = {
    addEventListener() {}, removeEventListener() {},
    innerWidth: 1280, innerHeight: 720,
    open() { return {}; },
};
globalThis.navigator = { clipboard: { writeText: async () => {} } };
let confirmCalls = [];
globalThis.confirm = (msg) => { confirmCalls.push(msg); return true; };
globalThis.CustomEvent = class { constructor(type, opts) { this.type = type; this.detail = opts?.detail; } };
globalThis.requestAnimationFrame = (f) => f();
globalThis.setTimeout = setTimeout; // 保留真实计时（流程测试用）

// ── app / fetch / api mock ──
globalThis.app = {
    graph: { setDirtyCanvas() {} },
    canvas: { ds: { scale: 1 }, editor_alpha: 1 },
    api: { fetchApi: async () => ({ ok: false }) },
};
let notesMeta = { trigger_words: "", description: "", base_model: "sd15", source_url: "", _has_custom: false };
globalThis.fetch = async (url) => {
    if (String(url).includes("lora_notes")) {
        return { ok: true, json: async () => notesMeta };
    }
    return { ok: false, status: 404 };
};

// Civitai API 层 mock（sf_lora_stack_api 的封装在此替换）
let civitaiResult = { ok: true, found: false, reason: "notfound" };
const callLog = { civitai: 0, delete: 0, saveThumb: 0, setAcc: [] };
globalThis.__apiMock = {
    loraInfo: async (name) => ({ ok: true, info: { source: "file", sidecar_triggers: [] } }),
    civitaiLookup: async (name) => { callLog.civitai++; return civitaiResult; },
    deleteCivitai: async (name) => { callLog.delete++; return { ok: true }; },
    saveCivitaiThumb: async (name) => { callLog.saveThumb++; return { ok: true, v: 99 }; },
    getCivitaiAccount: async () => ({ ok: true, configured: false, hint: "", host: "com", adultThumbs: false }),
    migrateLoraData: async () => ({ ok: true }),
    setCivitaiAccount: async (patch) => {
        callLog.setAcc.push(patch);
        return { ok: true, configured: !!(patch.key ?? null), hint: patch.key ? "1234" : "",
            host: patch.host || "com", adultThumbs: !!patch.adultThumbs };
    },
};

// ── 加载模块（改 import 为 mock）──
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "sf_li_"));
for (const n of ["sf_lora_shared_info.js"]) {
    const c = fs.readFileSync(path.join(__dirname, "..", "web", n), "utf8")
        .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
        .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
    fs.writeFileSync(path.join(tmpDir, n.replace(/\.js$/, ".mjs")), c);
}
const code = fs
    .readFileSync(path.join(__dirname, "..", "web", "sf_lora_info.js"), "utf8")
    .replaceAll('import { app } from "/scripts/app.js";', "const app = globalThis.app;")
    .replace(/from "\.\/([a-z_]+)\.js"/g, 'from "./$1.mjs"');
fs.writeFileSync(path.join(tmpDir, "sf_lora_info.mjs"), code);
fs.writeFileSync(path.join(tmpDir, "sf_markdown.mjs"),
    "export function renderMarkdown(s) { return String(s || ''); }\n");
fs.writeFileSync(path.join(tmpDir, "sf_common.mjs"),
    "export async function copyText(t){ try{ await navigator.clipboard.writeText(t); return true;}catch{return false;}}\n" +
    "export function escapeHtml(s){ return String(s); }\n" +
    "export function installWheelZoomPassthrough(){ return ()=>{}; }\n");
fs.writeFileSync(path.join(tmpDir, "sf_lora_stack_api.mjs"),
    "const m = globalThis.__apiMock;\n" +
    "export const loraInfo = m.loraInfo;\nexport const civitaiLookup = m.civitaiLookup;\n" +
    "export const deleteCivitai = m.deleteCivitai;\nexport const saveCivitaiThumb = m.saveCivitaiThumb;\n" +
    "export const getCivitaiAccount = m.getCivitaiAccount;\nexport const setCivitaiAccount = m.setCivitaiAccount;\n" +
    "export const migrateLoraData = m.migrateLoraData;\n");

function findEl(root, pred) {
    if (!root || !root.children) return null;
    if (pred(root)) return root;
    for (const c of root.children) { const r = findEl(c, pred); if (r) return r; }
    return null;
}
// mock 元素 textContent 不自动聚合子节点——显式聚合（含 text node）
function textOf(root) {
    if (!root || !root.children) return String(root.textContent || "");
    let s = root.textContent || "";
    for (const c of root.children) s += textOf(c);
    return s;
}
function findBtn(root, text) {
    return findEl(root, (e) => String(e.textContent || "").includes(text) && !!e._listeners?.click);
}
function findAll(root, pred) {
    const out = [];
    (function walk(r) {
        if (!r || !r.children) return;
        if (pred(r)) out.push(r);
        for (const c of r.children) walk(c);
    })(root);
    return out;
}
const tick = () => new Promise((r) => setTimeout(r, 20));

(async () => {
    await import(path.join(tmpDir, "sf_lora_info.mjs"));
    const { showLoraInfoDialog } = await import(path.join(tmpDir, "sf_lora_info.mjs"));

    // ── T1 对话框构建 ──
    showLoraInfoDialog(null, "test/lora_a.safetensors",
        { trigger_words: "", description: "", base_model: "sd15", source_url: "" });
    const dialog = bodyEl.children.find((c) => c.tagName === "DIALOG");
    check("对话框已挂载", !!dialog && dialog.open === true);
    check("↻ Civitai 按钮存在", !!findBtn(dialog, "↻ Civitai"));
    check("Account 按钮存在", !!findBtn(dialog, "Account"));
    check("🖍 编辑按钮存在（Trigger Words 行）", !!findEl(dialog, (e) => String(e.title||"").includes("Edit")));
    await tick(); // 预取（getCivitaiAccount/loraInfo）落地

    // ── T2 found：状态条 + 行刷新 + 🗑 + 链接 + 封面 ──
    notesMeta = { trigger_words: "cw1, cw2", description: "civ desc", base_model: "sd1.5",
        source_url: "", _has_custom: true };
    civitaiResult = { ok: true, found: true, info: { name: "Civitai Lora", model_id: 123,
        version_id: 456, triggers: ["cw1", "cw2"], description: "civ desc" } };
    findBtn(dialog, "↻ Civitai").click();
    await tick();
    const strip = findEl(dialog, (e) => String(e.className || "").includes("sf-li-civstrip"));
    check("found 状态条", !!strip && textOf(strip).includes("Found on Civitai"));
    check("View on Civitai 链接", !!strip && textOf(strip).includes("View on Civitai ↗"));
    check("Trigger Words 行已刷为 Civitai 词",
        !!findEl(dialog, (e) => e.textContent === "cw1, cw2"));
    check("🗑 删除侧车按钮出现", !!findBtn(dialog, "🗑"));
    const th = findEl(dialog, (e) => String(e.src || "").includes("lora_thumb"));
    check("封面已刷新（bust URL）", !!th && /&t=\d+/.test(th.src));
    check("无 thumb_skipped 时不弹确认", confirmCalls.length === 0);

    // ── T3 thumb_skipped + 确认替换 ──
    civitaiResult = { ok: true, found: true, info: { name: "Civitai Lora", model_id: 123,
        version_id: 456, triggers: ["cw1"], description: "civ desc" }, thumb_skipped: true };
    findBtn(dialog, "↻ Civitai").click();
    await tick();
    check("thumb_skipped 弹确认", confirmCalls.length === 1 && /preview picture/.test(confirmCalls[0]));
    check("确认后走 saveCivitaiThumb", callLog.saveThumb === 1);

    // ── T4 notfound ──
    civitaiResult = { ok: true, found: false, reason: "notfound" };
    findBtn(dialog, "↻ Civitai").click();
    await tick();
    const strip4 = findEl(dialog, (e) => String(e.className || "").includes("sf-li-civstrip"));
    check("notfound 状态条", !!strip4 && textOf(strip4).includes("Not on Civitai"));

    // ── T5 offline ──
    civitaiResult = { ok: false, reason: "offline", message: "Civitai timed out." };
    findBtn(dialog, "↻ Civitai").click();
    await tick();
    const strip5 = findEl(dialog, (e) => String(e.className || "").includes("sf-li-civstrip"));
    check("offline 状态条带原因", !!strip5 && textOf(strip5).includes("Civitai timed out"));

    // ── T6 删除侧车 ──
    notesMeta = { trigger_words: "file-word", description: "", base_model: "sd15",
        source_url: "", _has_custom: false };
    // 精确匹配 "🗑"（模糊匹配会命中 footer 的 "🗑️ Clear Notes" 按钮）
    const delBtn = findEl(dialog, (e) => String(e.textContent || "") === "🗑" && !!e._listeners?.click);
    check("🗑 按钮可点", !!delBtn);
    delBtn.click();
    await tick();
    check("删除后状态条消失", !findEl(dialog, (e) => String(e.className || "").includes("sf-li-civstrip")));
    check("删除后 🗑 消失", !findEl(dialog, (e) => String(e.textContent || "") === "🗑"));
    check("词回到文件词", !!findEl(dialog, (e) => e.textContent === "file-word"));
    check("deleteCivitai 已调用", callLog.delete === 1);

    // ── T7 账户区：Add key -> Save -> Remove ──
    findBtn(dialog, "Account").click();
    const accPanel = findEl(dialog, (e) => String(e.className || "").includes("sf-li-acc"));
    check("账户区展开", !!accPanel && accPanel.style.display === "block");
    check("未配置状态显示", !!findEl(dialog, (e) => String(e.textContent || "").includes("No key")));
    const addKey = findBtn(dialog, "Add key");
    check("Add key 按钮", !!addKey);
    addKey.click();
    const keyInput = findEl(dialog, (e) => e.tagName === "INPUT" && e.type === "password");
    check("key 输入框出现", !!keyInput);
    keyInput.value = "sec123";
    findBtn(dialog, "Save").click();
    await tick();
    check("setCivitaiAccount 收到 key", callLog.setAcc.length === 1 && callLog.setAcc[0].key === "sec123");
    check("保存后显示 Key saved", !!findEl(dialog, (e) => String(e.textContent || "").includes("Key saved")));
    const rmBtn = findBtn(dialog, "Remove");
    check("Remove 按钮出现", !!rmBtn);
    rmBtn.click();
    await tick();
    check("移除后回到 No key", !!findEl(dialog, (e) => String(e.textContent || "").includes("No key")));

    // ── T8 编辑态守卫：查询不覆盖草稿 ──
    const editBtns = findAll(dialog, (e) => String(e.title||"").includes("Edit") && !!e._listeners?.click);
    check("两个编辑按钮（Trigger Words + Description）", editBtns.length === 2);
    editBtns[1].click(); // Description 行
    const ta = findEl(dialog, (e) => e.tagName === "TEXTAREA");
    check("Description 进入编辑态", !!ta);
    ta.value = "my draft";
    notesMeta = { trigger_words: "civ-new", description: "civ-new-desc", base_model: "sd1.5",
        source_url: "", _has_custom: true };
    civitaiResult = { ok: true, found: true, info: { name: "Civitai Lora", model_id: 1,
        version_id: 2, triggers: ["civ-new"], description: "civ-new-desc" } };
    findBtn(dialog, "↻ Civitai").click();
    await tick();
    const ta2 = findEl(dialog, (e) => e.tagName === "TEXTAREA");
    check("编辑中草稿未被覆盖", !!ta2 && ta2.value === "my draft");
    check("非编辑行（Trigger Words）仍刷新", !!findEl(dialog, (e) => e.textContent === "civ-new"));

    // ── T9 保存后缓存保持（回归：saveNotes 必须先广播失效再写回自身缓存，
    // 否则事件桥同步 delete 会删掉刚写入的新值——i 图标保存后变灰，
    // 重开对话框 force 重取才恢复）──
    const { loraMetadataCache } = await import(path.join(tmpDir, "sf_lora_info.mjs"));
    loraMetadataCache.set("test/lora_a.safetensors", { _has_custom: false }); // 模拟陈旧缓存
    notesMeta = { trigger_words: "my-words", description: "my-desc", base_model: "sd15",
        source_url: "", _has_custom: true };
    const editBtns9 = findAll(dialog, (e) => String(e.title||"").includes("Edit") && !!e._listeners?.click);
    editBtns9[0].click(); // Trigger Words 行进入编辑
    const inp9 = findEl(dialog, (e) => e.tagName === "INPUT" && e.type !== "password");
    check("Trigger Words 编辑输入框", !!inp9);
    inp9.value = "my-words";
    const saveBtn9 = findBtn(dialog, "Save");
    check("Save 按钮（编辑态）", !!saveBtn9);
    saveBtn9.click();
    await tick();
    const cached9 = loraMetadataCache.get("test/lora_a.safetensors");
    check("保存后缓存保持新值（i 图标高亮不丢失）", !!cached9 && cached9._has_custom === true);

    console.log("\nFAILURES:", failures.length);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(failures.length ? 1 : 0);
})().catch((e) => {
    console.error("smoke crashed:", e);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    process.exit(1);
});
