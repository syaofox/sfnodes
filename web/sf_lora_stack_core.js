// ==========================================================================
// SF LoRA Stack - shared state + helpers（纯逻辑，无 app/DOM 依赖，可复制为
// .mjs 用 Node 直跑，见 tests/test_lora_stack_core.mjs）
//
// 状态存 node.properties.loraStackState（LiteGraph 原生序列化），由主扩展的
// graphToPrompt 钩子注入隐藏 LoraLoaderState 输入（web/sf_lora_stack.js）。
// Python 读回 loras + 分隔符，逐个应用开着的 LoRA，并把勾选的触发词连接
// 成 triggers 输出。
//
// 双端契约：parse_state（Python lora_reader.py）与这里 normalize/promptState
// 的字段名与语义 1:1 镜像，改一边必须同步另一边。
// ==========================================================================

export const BRAND = "#f66744";
export const STATE_PROP = "loraStackState";
export const HIDDEN_INPUT = "LoraLoaderState"; // 匹配 Python INPUT_TYPES 键
export const DEFAULTS_SETTING = "sfnodes.LoraStack.Defaults";

export const MAX_LORAS = 64;
export const MIN_STRENGTH = -10;
export const MAX_STRENGTH = 10;

// 新节点从保存的默认继承的每节点偏好（DEFAULTS_SETTING 里一个 JSON blob）。
// 状态里其余都是每节点数据。强调色统一走全局设置（sfnodes.Accent），
// 无节点级自定义（DEFAULT_PREFS 不再含 accent；旧 state 里的 accent 字段
// 被忽略）。
export const DEFAULT_PREFS = {
    sep: ", ",          // 触发词输出连接符
    step: 0.05,         // 强度箭头/步进增量
    defStrength: 1.0,   // 新添加 LoRA 的起始强度
    linkStrength: true, // 单一强度同时驱动 model + clip
    civitai: true,      // 信息面板允许可选 Civitai 查询按钮
    thumbs: true,       // 信息面板显示预览缩略图
    hideExt: true,      // 行上显示 "MoXin" 而非 "MoXin.safetensors"
    // Python 在 run 之间保留多少 LoRA 数据（齿轮 "LoRA memory use"）：
    // "last"（默认）= ComfyUI 对齐，只留最近使用的文件；
    // "all" = 保留整个栈（重跑最快，大栈可占 GB 级内存）；
    // "none" = 什么都不留（内存最低，每 run 重读文件）。
    cacheMode: "last",
    // LoRA 叠加方式（齿轮 "Stacking method"）：
    // "sequential"（默认）= 标准逐行相加；
    // "ortho_gs" = 输入空间 Gram-Schmidt 正交化（减少相似 LoRA 干扰；
    // 行顺序即优先级——第一个 LoRA 保持原样、后续让位、可能损失幅度；
    // 仅 UNet 层正交化，CLIP 仍顺序叠加）。改变输出，必须注入 promptState。
    mergeMethod: "sequential",
};

export const DEFAULT_STATE = {
    version: 1,
    loras: [], // { id, name, on, sm, sc, triggers:[], custom:[] }
    ...DEFAULT_PREFS,
};

let _idc = 0;
export function newId() {
    try { if (crypto?.randomUUID) return "l" + crypto.randomUUID().slice(0, 8); } catch { /* 忽略 */ }
    return "l" + Date.now().toString(36) + (_idc++).toString(36);
}

function num(v, dflt) {
    const f = parseFloat(v);
    if (!Number.isFinite(f)) return dflt;
    return f;
}

export function clampStrength(v) {
    const f = num(v, 0);
    return Math.max(MIN_STRENGTH, Math.min(MAX_STRENGTH, f));
}

// 强度四舍五入到 2 位小数用于显示/存储（杀掉浮点灰尘）。
export function roundStrength(v) {
    return Math.round(clampStrength(v) * 100) / 100;
}

function normLora(e, prefs) {
    if (!e || typeof e !== "object") return null;
    const name = typeof e.name === "string" ? e.name : "";
    const sm = roundStrength(e.sm != null ? e.sm : (e.strength != null ? e.strength : prefs.defStrength));
    const sc = roundStrength(e.sc != null ? e.sc : sm);
    return {
        id: typeof e.id === "string" && e.id ? e.id : newId(),
        name,
        on: e.on == null ? true : !!e.on,
        sm,
        sc,
        triggers: Array.isArray(e.triggers)
            ? e.triggers.map((w) => String(w)).filter((w) => w.trim()).slice(0, 64)
            : [],
        // 用户亲手为这个 LoRA 输入的词（文件没有词时也可用做 chips）。
        // UI 专用——选中的活在 `triggers`（到达输出的东西）；`custom` 在
        // promptState 里被剥掉。
        custom: Array.isArray(e.custom)
            ? e.custom.map((w) => String(w)).filter((w) => w.trim()).slice(0, 64)
            : [],
    };
}

export function normalize(raw) {
    const st = { ...DEFAULT_STATE, ...(raw && typeof raw === "object" ? raw : {}) };
    if (typeof st.sep !== "string") st.sep = DEFAULT_PREFS.sep;
    st.step = num(st.step, DEFAULT_PREFS.step);
    if (st.step <= 0) st.step = DEFAULT_PREFS.step;
    st.defStrength = roundStrength(st.defStrength);
    st.linkStrength = st.linkStrength == null ? true : !!st.linkStrength;
    st.civitai = st.civitai == null ? true : !!st.civitai;
    st.thumbs = st.thumbs == null ? true : !!st.thumbs;
    st.hideExt = st.hideExt == null ? true : !!st.hideExt;
    st.cacheMode = st.cacheMode === "all" || st.cacheMode === "none" ? st.cacheMode : "last";
    st.mergeMethod = st.mergeMethod === "ortho_gs" ? st.mergeMethod : "sequential";
    st.loras = (Array.isArray(st.loras) ? st.loras : [])
        .map((e) => normLora(e, st))
        .filter(Boolean)
        .slice(0, MAX_LORAS);
    // 联动强度不变量：单强度驱动双端时 clip 恒等于 model。在每次写/读上
    // 强制，切换 "separate" 关闭后永不留下会静默应用错误 CLIP 权重的陈旧值。
    if (st.linkStrength) st.loras.forEach((e) => { e.sc = e.sm; });
    // 去重 id：手改/复制的状态不能有按 id 不可达的行（所有按 id 操作用
    // find()，重复 id 会永远命中第一个）。
    const seenIds = new Set();
    for (const e of st.loras) {
        if (seenIds.has(e.id)) e.id = newId();
        seenIds.add(e.id);
    }
    return st;
}

export function readState(node) {
    const v = node.properties?.[STATE_PROP];
    if (typeof v === "string" && v) {
        try { return normalize(JSON.parse(v)); } catch { /* 走默认 */ }
    }
    return normalize({ ...DEFAULT_STATE, ...loadDefaults() });
}

export function writeState(node, state) {
    if (!node.properties) node.properties = {};
    const st = normalize(state);
    node.properties[STATE_PROP] = JSON.stringify(st);
    return st;
}

// ── mutations（每个返回新状态）──────────────────────────────────────────────
export function addLora(node, name) {
    const st = readState(node);
    if (st.loras.length >= MAX_LORAS) return { ok: false, reason: "max", state: st };
    st.loras.push({
        id: newId(),
        name: name || "",
        on: true,
        sm: st.defStrength,
        sc: st.defStrength,
        triggers: [],
    });
    return { ok: true, state: writeState(node, st), index: st.loras.length - 1 };
}

export function removeLora(node, id) {
    const st = readState(node);
    const i = st.loras.findIndex((e) => e.id === id);
    if (i < 0) return null;
    st.loras.splice(i, 1);
    return writeState(node, st);
}

export function duplicateLora(node, id) {
    const st = readState(node);
    const i = st.loras.findIndex((e) => e.id === id);
    if (i < 0 || st.loras.length >= MAX_LORAS) return null;
    const clone = {
        ...st.loras[i], id: newId(),
        triggers: [...st.loras[i].triggers],
        custom: [...(st.loras[i].custom || [])],
    };
    st.loras.splice(i + 1, 0, clone);
    return writeState(node, st);
}

export function moveLora(node, id, dir) {
    const st = readState(node);
    const i = st.loras.findIndex((e) => e.id === id);
    if (i < 0) return null;
    const j = i + dir;
    if (j < 0 || j >= st.loras.length) return null;
    const [m] = st.loras.splice(i, 1);
    st.loras.splice(j, 0, m);
    return writeState(node, st);
}

export function reorderLora(node, from, to) {
    const st = readState(node);
    const n = st.loras.length;
    if (from < 0 || from >= n || to < 0 || to >= n || from === to) return null;
    const [m] = st.loras.splice(from, 1);
    st.loras.splice(to, 0, m);
    return writeState(node, st);
}

export function patchLora(node, id, patch) {
    const st = readState(node);
    const e = st.loras.find((x) => x.id === id);
    if (!e) return null;
    const oldName = e.name;
    const keepId = e.id;
    Object.assign(e, patch);
    e.id = keepId; // patch 绝不改变行的身份
    if (patch.sm != null) e.sm = roundStrength(patch.sm);
    if (patch.sc != null) e.sc = roundStrength(patch.sc);
    // 换到不同 LoRA 清除选中和自定义词——它们属于旧文件。
    if (patch.name != null && patch.name !== oldName) { e.triggers = []; e.custom = []; }
    // 联动强度下，model 强度变化镜像到 clip。
    if (st.linkStrength && patch.sm != null && patch.sc == null) e.sc = e.sm;
    return writeState(node, st);
}

export function setAllOn(node, on) {
    const st = readState(node);
    st.loras.forEach((e) => { e.on = !!on; });
    return writeState(node, st);
}

export function countOn(state) {
    return state.loras.reduce((a, e) => a + (e.on ? 1 : 0), 0);
}

// ── 全局默认（一个 JSON blob，让 "Set as default" 捕获每个偏好）─────────────
export function loadDefaults() {
    try {
        const raw = globalThis.app?.ui?.settings?.getSettingValue(DEFAULTS_SETTING);
        if (raw) {
            const obj = typeof raw === "string" ? JSON.parse(raw) : raw;
            if (obj && typeof obj === "object") return obj;
        }
    } catch { /* 忽略 */ }
    return {};
}

export async function saveDefaults(prefs) {
    try {
        const keep = {};
        for (const k of Object.keys(DEFAULT_PREFS)) if (prefs[k] !== undefined) keep[k] = prefs[k];
        await globalThis.app.ui.settings.setSettingValueAsync(DEFAULTS_SETTING, JSON.stringify(keep));
        return true;
    } catch { return false; }
}

// 进入 prompt 的执行相关子集。Python 只读 loras（name/on/sm/sc/triggers）和
// 分隔符，所以外观偏好（accent/thumbs/civitai/step/defStrength/linkStrength/id）
// 剥掉——否则换个颜色或开关设置就会改节点缓存签名、白白重跑（文档化陷阱）。
export function promptState(state) {
    return {
        version: 1,
        sep: state.sep,
        // cacheMode 刻意保留（虽然它不改变输出）：Python 只看到 prompt 携带
        // 的东西，它需要内存模式决定 run 之间保留什么。代价：切换模式会多
        // 重跑一次节点——对内存行为开关可接受，与上面剥掉的外观偏好相反。
        cacheMode: state.cacheMode,
        // mergeMethod 同样保留：它改变执行语义（正交化 vs 顺序）。切换会多
        // 重跑一次节点，代价可接受。
        mergeMethod: state.mergeMethod,
        loras: state.loras.map((e) => ({
            name: e.name, on: !!e.on, sm: e.sm, sc: e.sc, triggers: e.triggers,
        })),
    };
}

// 强调色：统一遵循全局设置（sfnodes.Accent），无节点级自定义。
// 节点参数保留签名兼容（调用方传 node，返回值与 node 无关）。
export function accentOf(node) {
    try {
        const g = globalThis.app?.ui?.settings?.getSettingValue?.("sfnodes.Accent");
        if (typeof g === "string" && g.trim()) return g;
    } catch { /* 忽略 */ }
    return BRAND;
}

// ── 预设（与 SFPowerLoraLoader 共享 user/sfnodes/lora_presets.json）──────────
// 预设存 Power 的形状 {lora, on, strength, strengthTwo}（后端校验该形状），
// 两个节点互通：Stack 存的 Power 能载，Power 存的 Stack 能载。SF 行里
// 的 triggers/custom 词不入预设（词属文件级存储，不随栈走）。

// 行形状 -> 预设形状。无名行（占位/未选）跳过。
export function rowsToPreset(st) {
    return {
        loras: st.loras
            .filter((e) => e.name)
            .map((e) => ({ lora: e.name, on: e.on, strength: e.sm, strengthTwo: e.sc })),
    };
}

// 预设形状 -> 行形状。防御垃圾输入：缺字段回默认，坏行丢弃。
export function presetToRows(preset) {
    const items = Array.isArray(preset?.loras) ? preset.loras : [];
    return items
        .filter((it) => it && typeof it === "object" && typeof it.lora === "string" && it.lora.trim())
        .map((it) => {
            const smF = parseFloat(it.strength);
            const scF = parseFloat(it.strengthTwo);
            const sm = Number.isFinite(smF) ? roundStrength(smF) : 1;
            const sc = Number.isFinite(scF) ? roundStrength(scF) : sm;
            return {
                id: newId(),
                name: it.lora,
                on: it.on == null ? true : !!it.on,
                sm,
                sc,
                triggers: [],
                custom: [],
            };
        })
        .slice(0, MAX_LORAS);
}
