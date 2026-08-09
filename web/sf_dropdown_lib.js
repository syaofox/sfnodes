// ==========================================================================
// sf_dropdown_lib.js - SFValueDropdown 纯函数库（coerce + state + 游标）
// ==========================================================================
//
// 无 app/DOM 依赖，供主扩展 sf_dropdown.js、UI 模块 sf_dropdown_ui.js 与
// 设置面板 sf_dropdown_settings.js import，也供 tests/ 复制为 .mjs 直接测试。
//
// 与后端 sf_utils/dropdown.py 的镜像契约（THE PARITY RULE）：
//   `readable` 与全部强制转换规则必须与 Python 侧 1:1 一致。面板用 readable
//   给"读不成所选类型"的行打警告标记，实时预览用 coerceValue 显示该行实际会
//   发什么。两侧漂移 = 面板承诺一件事、运行做另一件事，比什么都不显示更糟。
//   规则刻意保持简单，使镜像平凡（数字语法、1e12 钳制、half-away-from-zero
//   取整）。改 Python 侧规则必须同一 commit 改这里。
//
// 状态存 node.properties.dropdownState（随工作流保存）：
//   { version:1, type:"text"|"int"|"float"|"bool", index, mode:"fixed"|"increment"|"random",
//     options:[{name,value}] }
// 运行游标（_sfDropdownPending/_sfDropdownCursor）只存节点内存，永不序列化：
//   写进工作流会把每次 Run 标成 modified（Seed 陷阱，见 AGENTS.md 经验摘要）。
//
// ==========================================================================

// ── 类型 ────────────────────────────────────────────────────────────────
export const TYPES = ["text", "int", "float", "bool"];

// 面板按钮上的长名。
export const TYPE_LABELS = {
    text: "Text",
    int: "Whole number",
    float: "Decimal",
    bool: "On / off",
};
// 输出点旁的短名——坐在节点行上不能吃宽度。
export const SOCKET_LABELS = { text: "text", int: "int", float: "float", bool: "on/off" };

// LiteGraph 输出槽类型，使画布拒绝不兼容拖拽。Python 声明 ANY；这是故事的前端
// 一半，背后没有第二次服务端类型检查。
export const SOCKET_TYPES = { text: "STRING", int: "INT", float: "FLOAT", bool: "BOOLEAN" };

export const FALLBACKS = { text: "", int: 0, float: 0.0, bool: false };

const TRUE_WORDS = new Set(["true", "yes", "on", "y", "t"]);
const FALSE_WORDS = new Set(["false", "no", "off", "n", "f"]);

// 与 Python 侧、以及 Control Panel 的 _value_of 相同的钳制。
const LIMIT = 1e12;

// THE 共享数字语法，与 sf_utils/dropdown.py 的 _NUMBER_RE 逐字符相同。
// 刻意不用 Number()：一次 228 例的 parity 运行抓到 Number("0x10") 返回 16 而
// Python float() 拒绝——于是面板说"发 16"而运行发 0。Python 同样接受 "1_0" 而
// Number() 拒绝。两边原生解析器都不是契约，这个才是。
//   accepts: 5  5.  .5  5.5  +5  -3  1e3  1E3  -1e3
//   refuses: 0x10  0b1  1_0  1,024  1024px  abc  Infinity  NaN  (and "")
const NUMBER_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

// 半数朝向远离零。Math.round 把平局朝 +Infinity 破（Math.round(-3.5) === -3），
// Python round() 是银行家舍入（round(2.5) === 2）；它们在每个精确半值上分歧。
// 远离零也是人期望的：往整数列表里敲 2.5 意味着 3，不是 2。
function roundHalfAway(value) {
    return value >= 0 ? Math.floor(value + 0.5) : -Math.floor(-value + 0.5);
}

// 数字 -> text 模式下 Python 会发出的字符串。Python 对整 float 的 str() 保留
// '.0' 而 JS 去掉；_number_to_text 那边匹配 THIS——两边都显示浏览器所显示的。
function numberToText(value) {
    return String(value);
}

export function normalizeType(kind) {
    if (typeof kind !== "string") return "text";
    const k = kind.trim().toLowerCase();
    if (TYPES.includes(k)) return k;
    if (k === "string" || k === "str") return "text";
    if (k === "integer" || k === "whole") return "int";
    if (k === "decimal" || k === "number" || k === "double") return "float";
    if (k === "boolean" || k === "toggle" || k === "onoff" || k === "on/off") return "bool";
    return "text";
}

// raw -> 有限数字，或 null。镜像 _as_number。
function asNumber(raw) {
    if (typeof raw === "boolean") return raw ? 1 : 0;
    if (typeof raw === "number") return Number.isFinite(raw) ? raw : null;
    if (typeof raw === "string") {
        const text = raw.trim();
        // 语法至少需要一个数字位，所以 "" 和 "   " 也覆盖。
        if (!NUMBER_RE.test(text)) return null;
        const value = Number(text);
        return Number.isFinite(value) ? value : null;
    }
    return null;
}

/** `raw` 能否干净地读成 `kind`？与 Python readable() 镜像，供警告标记用。 */
export function readable(raw, kind) {
    kind = normalizeType(kind);
    if (kind === "text") return true;
    if (kind === "bool") {
        if (typeof raw === "boolean") return true;
        if (typeof raw === "string") {
            const w = raw.trim().toLowerCase();
            if (TRUE_WORDS.has(w) || FALSE_WORDS.has(w)) return true;
        }
        // 钳制不会改变零/非零答案，所以开关不关心数量级。
        return asNumber(raw) !== null;
    }
    const n = asNumber(raw);
    if (n === null) return false;
    // 钳制会移动的值即使解析成功也不算可读。见 Python 侧同名注释：
    // 没有它，15 位种子得不到警告标记而运行发 1000000000000。
    return n >= -LIMIT && n <= LIMIT;
}

/** raw + 类型 -> 浏览器预览用的值。与 Python coerce_value() 镜像。 */
export function coerceValue(raw, kind) {
    kind = normalizeType(kind);

    if (kind === "text") {
        if (raw == null) return "";
        if (typeof raw === "string") return raw;
        // 与 Python 一致：发出人会打出的拼写，而非语言自己的。
        if (typeof raw === "boolean") return raw ? "true" : "false";
        if (typeof raw === "number") return numberToText(raw);
        return String(raw);
    }

    if (kind === "bool") {
        if (typeof raw === "boolean") return raw;
        if (typeof raw === "string") {
            const w = raw.trim().toLowerCase();
            if (TRUE_WORDS.has(w)) return true;
            if (FALSE_WORDS.has(w)) return false;
        }
        const n = asNumber(raw);
        if (n === null) return FALLBACKS.bool;
        return n !== 0;
    }

    let n = asNumber(raw);
    if (n === null) return FALLBACKS[kind];
    n = Math.max(-LIMIT, Math.min(LIMIT, n));
    if (kind === "int") return roundHalfAway(n);
    return n;
}

// 值的单行短渲染，给节点面和弹出列表的提示。值可能是多行（一句触发词），
// 多行原文会把单行砸烂——只取第一行。
export function previewText(raw, kind) {
    const v = coerceValue(raw, kind);
    const s = typeof v === "string" ? v : String(v);
    const firstLine = s.split("\n")[0];
    return firstLine.length === s.length ? firstLine : firstLine + "…";
}

// ── 状态（node.properties）─────────────────────────────────────────────
export const CLASS = "SFValueDropdown";

// node.properties 键（camelCase）与 Python INPUT_TYPES 键（PascalCase）。
// 大小写刻意不同，与项目其他节点一致——第二个键一旦打错，Python 永远看到默认值，
// 节点表现为无视你的一切修改，所以写两遍。
export const STATE_PROP = "dropdownState";
export const HIDDEN_INPUT = "DropdownState";   // 必须与 Python INPUT_TYPES 键一致

// 几何。Legacy 与 Nodes 2.0 都从这里推导，是调行的唯一地点。
export const ROW_H = 26;
export const MIN_W = 210;
export const DEFAULT_W = 250;
export const BODY_PAD = 7;

// 零宽空格。真值，使两个渲染器都不会回退去画原始槽名（"value"）压在我们的行上，
// 但实际什么都不画。空字符串会掉进 litegraph 的 || 链回退到 slot.name。
// 写成转义而非字面 U+200B：源码里的不可见字符不可审阅、不可 diff。
export const ZW = "\u200B";

export const OUT_NAME = "value";

// 节点每次 RUN 如何选条目。
//   fixed     - 总是你选的那个。默认，也是唯一让节点完全可预测的模式。
//   increment - 每次运行顺延到下一条，到底回绕。
//   random    - 任一条，2 条以上时永不连续相同。
export const MODES = ["fixed", "increment", "random"];
export const MODE_LETTERS = { fixed: "F", increment: "I", random: "R" };
export const MODE_LABELS = {
    fixed: "Fixed - always the entry you picked",
    increment: "In order - the next entry each run, wrapping at the end",
    random: "Random - any entry each run",
};

export function defaultState() {
    return { version: 1, type: "text", index: 0, mode: "fixed", options: [] };
}

function normalizeMode(m) {
    return MODES.includes(m) ? m : "fixed";
}

/** node -> 其状态，永远是合法对象。绝不信任找到的东西。 */
export function readState(node) {
    const raw = node?.properties?.[STATE_PROP];
    const st = defaultState();
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return st;

    st.type = normalizeType(raw.type);
    st.mode = normalizeMode(raw.mode);

    if (Array.isArray(raw.options)) {
        for (const o of raw.options) {
            // 丢掉非对象行，不让它以后炸掉列表。Control Panel 为此吃过亏：
            // 单个 null 行让画布上所有同类节点的值注入全部中止。
            if (!o || typeof o !== "object" || Array.isArray(o)) continue;
            st.options.push({
                name: typeof o.name === "string" ? o.name : "",
                value: typeof o.value === "string" ? o.value : (o.value == null ? "" : String(o.value)),
            });
        }
    }

    const n = Number(raw.index);
    st.index = Number.isFinite(n) ? Math.max(0, Math.min(st.options.length - 1, Math.trunc(n))) : 0;
    if (!st.options.length) st.index = 0;
    return st;
}

/**
 * 唯一写路径。一切改变列表的调用都经过这里，保证存储的 index 永远不会指向
 * 不存在的行。
 *
 * 刻意不做与已存对象的 diff 门控：调用方传 patch，我们总是重归一化。加载路径
 * 上安全只是因为加载路径上没人调它——见 sf_dropdown.js 中的说明。
 */
export function writeState(node, patch) {
    if (!node) return defaultState();
    if (!node.properties) node.properties = {};
    const cur = readState(node);
    const next = { ...cur, ...(patch || {}) };

    next.version = 1;
    next.type = normalizeType(next.type);
    next.mode = normalizeMode(next.mode);
    next.options = Array.isArray(next.options) ? next.options.map((o) => ({
        name: typeof o?.name === "string" ? o.name : "",
        value: typeof o?.value === "string" ? o.value : (o?.value == null ? "" : String(o.value)),
    })) : [];
    const n = Number(next.index);
    next.index = Number.isFinite(n) ? Math.max(0, Math.min(next.options.length - 1, Math.trunc(n))) : 0;
    if (!next.options.length) next.index = 0;

    node.properties[STATE_PROP] = next;
    return next;
}

/** 当前选中的条目，列表为空时为 null。 */
export function selectedOption(node) {
    const st = readState(node);
    return st.options[st.index] || null;
}

// ── 运行游标（节点内存，不序列化）─────────────────────────────────────
/**
 * 这次 BUILD 应发送的 index。
 *
 * Fixed 模式就是选中的条目。另外两种模式从运行时游标（node._sfDropdownCursor）
 * 推导，绝不碰 node.properties：把新位置写进工作流会让每次按 Run 都标记
 * "modified"——Seed 陷阱。代价是页面刷新后序列从你选中的条目重新开始，可预测、
 * 可见。
 *
 * 选出的牌先 HOLD 在 node._sfDropdownPending 直到真正被花掉：一次 queue 的
 * graphToPrompt 会跑多次（Export、保存、以及随后校验失败的 queue），因此同一
 * 条目反复给，直到 api.queuePrompt 成功才 commitPick。
 */
export function pendingIndex(node) {
    const st = readState(node);
    const n = st.options.length;
    if (!n) return 0;
    const clamp = (i) => Math.max(0, Math.min(i, n - 1));
    if (st.mode === "fixed") return clamp(st.index);

    // 已持有的牌只在仍指向真实行时有效。
    if (Number.isInteger(node._sfDropdownPending) && node._sfDropdownPending < n) return node._sfDropdownPending;

    let next;
    if (st.mode === "random") {
        // 总是随机，包括切到 R 后的第一次运行。下面的首轮分支只属于 In-order：
        // 它存在是为了让序列从用户目光所在处开始，但套到 Random 上会让第一次
        // Run 发出节点面正显示的条目，与面板承诺的"每次随机一条"矛盾。
        const avoid = Number.isInteger(node._sfDropdownCursor) ? clamp(node._sfDropdownCursor) : clamp(st.index);
        next = Math.floor(Math.random() * n);
        // 永不连续同一条——两条列表时重复读起来像模式没生效。
        if (n > 1 && next === avoid) next = (next + 1 + Math.floor(Math.random() * (n - 1))) % n;
    } else if (!Number.isInteger(node._sfDropdownCursor)) {
        // 加载后的第一次运行：发节点正在显示的条目，然后开始移动。
        next = clamp(st.index);
    } else {
        next = (clamp(node._sfDropdownCursor) + 1) % n;
    }
    node._sfDropdownPending = next;
    return next;
}

/**
 * 花掉持有的牌。只在 queue 真正被接受时调用，所以 Export 或失败的 queue
 * 不会推进 In-order 列表。
 */
export function commitPick(node) {
    if (Number.isInteger(node._sfDropdownPending)) {
        node._sfDropdownCursor = node._sfDropdownPending;
        node._sfDropdownPending = null;
    } else if (readState(node).mode === "fixed") {
        node._sfDropdownCursor = null;   // Fixed 不累积位置
    }
}

/** 节点面应显示的：已排队或上次运行的牌，而非盲目存的那个。 */
export function shownIndex(node) {
    const st = readState(node);
    const n = st.options.length;
    if (!n) return 0;
    const clamp = (i) => Math.max(0, Math.min(i, n - 1));
    if (st.mode === "fixed") return clamp(st.index);
    if (Number.isInteger(node._sfDropdownPending) && node._sfDropdownPending < n) return node._sfDropdownPending;
    if (Number.isInteger(node._sfDropdownCursor)) return clamp(node._sfDropdownCursor);
    return clamp(st.index);
}

/**
 * 浏览器发给 Python 的东西。只含影响结果的部分。
 *
 * 注入字符串成为节点 inputs 的一部分，ComfyUI 会对它哈希：任何真正只是显示用
 * 的东西（改名、重排、改你没选中的行、模式、颜色）都会在变化时重跑整个图，
 * 必须全部留在外面。Python 直接接受这个 lean 形状。
 */
export function injectedState(node) {
    const st = readState(node);
    const opt = st.options[pendingIndex(node)];
    return { version: 1, type: st.type, value: opt ? opt.value : null };
}

// ── 输出槽类型同步（前端半边）─────────────────────────────────────────
/**
 * 把所选类型放到输出槽上，让画布拒绝不兼容拖拽。Python 声明 ANY；这是前端那
 * 一半，背后没有第二次服务端检查。
 *
 * 每次写入都 diff 门控。槽位会被序列化，重写一个相同的值在某些构建上仍算一次
 * 变更，会把干净的工作流一打开就标 "modified"（Vue Compat #18）。
 */
export function syncOutput(node) {
    if (!node?.outputs?.length) return;
    const want = SOCKET_TYPES[readState(node).type] || "*";
    const out = node.outputs[0];
    if (out.name !== OUT_NAME) out.name = OUT_NAME;
    if (out.label !== ZW) out.label = ZW;
    if (out.type !== want) out.type = want;
}

// ── 槽位类型匹配（纯逻辑）──────────────────────────────────────────────
// 读 LiteGraph 槽位类型而不被 ComfyUI 的多类型输入绊倒：自 V3 schema API 起
// 槽位类型可以是逗号连接的列表（io.MultiType.Input 到达浏览器是字面的
// "FLOAT,INT,BOOLEAN"，如核心的 Math Expression）。把它读成单个名字会让
// dropIncompatibleLinks 剪掉用户刚画的线。纯模块：无 ComfyUI 导入。

/** "FLOAT,INT,BOOLEAN" -> ["FLOAT","INT","BOOLEAN"]；"FLOAT" -> ["FLOAT"]；"" -> []。 */
export function slotTypeList(type) {
    if (type == null) return [];
    return String(type)
        .split(",")
        .map((x) => x.trim().toUpperCase())
        .filter(Boolean);
}

/**
 * 通配槽与未定类型槽——都接受一切。任何 falsy 值都算，不只是 null/""：
 * LiteGraph 在若干场合用 0 作通配，把 0 折叠成 "" 被接受才对。
 */
export function isWildcardType(type) {
    if (!type) return true;
    const s = String(type).trim();
    return s === "" || s === "*";
}

/** `slotType` 的槽会接受 `ourType` 的值吗？任一侧可以是通配或逗号连接的多类型。 */
export function slotAccepts(slotType, ourType) {
    if (isWildcardType(slotType) || isWildcardType(ourType)) return true;
    const accepts = slotTypeList(slotType);
    const ours = slotTypeList(ourType);
    return ours.some((o) => accepts.includes(o));
}
