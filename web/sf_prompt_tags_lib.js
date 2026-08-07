// ==========================================================================
// sf_prompt_tags_lib.js - SFPromptTags 纯工具库
// ==========================================================================
//
// 提供 @tag 展开与标签库所需的纯函数（无 app / DOM 依赖），供主扩展
// sf_prompt_tags.js、存储层 sf_prompt_tags_store.js、游标
// sf_prompt_tags_cursors.js、全屏编辑器 sf_prompt_tags_editor.js import，
// 也供 tests/ 复制为 .mjs 直接测试。
//
// 语义与 Pixaroma Prompt 一致：
//   - @name    = 保存的标签 -> 整段文本
//   - *cat     = 该分类随机一个标签 -> 其文本（若抽中 List 标签则随机取一行）
//   - #name    = 标签文本按行拆分 -> 一行
//   未知名称一律保留字面量（token 原样输出）。
//
// 库数据模型（normalizeLibrary 的规范形，与 Pixaroma 兼容）：
//   { version, categories:[name...], listCats:[name...], tags:[{name, cat, text,
//     kind?, mode?}...], catModes:{cat: mode} }
//   - categories 是含双侧的单一有序列表；listCats 标记 List 侧分类（Text 侧为
//     其余）。每个分类属于且仅属于一侧。
//   - kind 仅在 "list" 时写入（缺省 = 文本标签）；mode 仅在非默认时写入。
//   - catModes：分类级 Picks 模式（*cat 如何选），键为规范名称。
//   - 未分类标签 cat:""，显示在 Text / List 桶（见 TEXT_BUCKET / LIST_BUCKET）。
//
// ==========================================================================

// name = 字母 / 数字 / _ / -（token 语法，与 Pixaroma 一致）
const TOKEN_RE = /([@*#])([a-zA-Z0-9_\-]+)/g;
const KIND_BY_SYM = { "@": "tag", "*": "wild", "#": "list" };

export const NAME_RE = /[^a-zA-Z0-9_\-]/g;

// 两个隐式桶：无分类标签的归属处，不是真实分类，永远不能成为分类名
export const TEXT_BUCKET = "Text";
export const LIST_BUCKET = "List";
// 2026-07-24 之前的旧桶，读取时仍识别，写入/展示不再使用
export const UNCATEGORIZED = "Uncategorized";
const RESERVED = new Set([TEXT_BUCKET, LIST_BUCKET, UNCATEGORIZED].map((s) => s.toLowerCase()));

// ── Picks 模式常量（游标系统也使用，保持一处定义）────────────────────────
export const MODES = ["shuffle", "random", "order"];
export const DEFAULT_MODE = "shuffle";
export const MODE_LABEL = { random: "Random", shuffle: "Shuffle", order: "In order" };
export function isMode(m) { return MODES.includes(m); }
export function cleanMode(m) { return isMode(m) ? m : DEFAULT_MODE; }
// 该模式是否跨 run 记住位置（Random 每次重掷，没有位置可言）
export function hasPosition(m) { return cleanMode(m) !== "random"; }

const lc = (s) => String(s == null ? "" : s).toLowerCase();

export function cleanTagName(v) {
    if (typeof v === "number" && Number.isFinite(v)) v = String(v);
    if (typeof v !== "string") return "";
    return v.trim().replace(NAME_RE, "");
}

function asName(v) {
    if (typeof v === "number" && Number.isFinite(v)) v = String(v);
    return typeof v === "string" ? v.trim() : "";
}

// ── 库规范化 ─────────────────────────────────────────────────────────────
// 任意来源（设置 / 导入文件 / 工作副本）→ 规范形。名称清洗、大小写去重、
// 分类补齐、Text/List 侧修复、catModes 规范化。旧库（无 listCats 字段）自动兼容。
export function normalizeLibrary(raw) {
    const out = { version: 1, categories: [], listCats: [], tags: [] };
    const src = raw && typeof raw === "object" ? raw : {};
    const seenCat = new Set();
    const addCat = (c) => {
        const name = asName(c);
        if (!name || RESERVED.has(lc(name)) || seenCat.has(lc(name))) return;
        seenCat.add(lc(name));
        out.categories.push(name);
    };
    for (const c of (Array.isArray(src.categories) ? src.categories : [])) addCat(c);
    // 声明为 List 侧的分类：即使 categories 数组漏了它，也是真实分类
    const listKeys = new Set();
    for (const c of (Array.isArray(src.listCats) ? src.listCats : [])) {
        const name = asName(c);
        if (!name || RESERVED.has(lc(name))) continue;
        listKeys.add(lc(name));
        addCat(name);
    }
    const seenTag = new Set();
    for (const t of (Array.isArray(src.tags) ? src.tags : [])) {
        if (!t || typeof t !== "object") continue;
        const name = cleanTagName(t.name);
        if (!name || seenTag.has(lc(name))) continue;
        seenTag.add(lc(name));
        let cat = asName(t.cat);
        let kind = t.kind === "list" ? "list" : "text";
        if (RESERVED.has(lc(cat))) {
            // 桶名不是分类："List" 还告诉我们它是列表（导入文件可能用桶名代替 kind）；
            // "Text" 不得反向强制 kind=text（会静默把显式 list 变成片段）。
            if (lc(cat) === lc(LIST_BUCKET)) kind = "list";
            cat = "";
        }
        const rec = { name, cat, text: typeof t.text === "string" ? t.text : "" };
        if (kind === "list") rec.kind = "list";
        const mode = cleanMode(t.mode);
        if (mode !== DEFAULT_MODE) rec.mode = mode;
        out.tags.push(rec);
    }
    // 标签引用了 categories 中缺失的分类 -> 补上（canonical 大小写）
    const catByKey = new Map(out.categories.map((c) => [lc(c), c]));
    for (const t of out.tags) {
        if (!t.cat) continue;
        const canon = catByKey.get(lc(t.cat));
        if (canon) t.cat = canon;
        else { out.categories.push(t.cat); catByKey.set(lc(t.cat), t.cat); seenCat.add(lc(t.cat)); }
    }
    // 侧修复：仅对 sides 出现之前的旧库运行启发式（全部是 list 标签的分类 => List 侧）
    const tagsByCat = new Map();
    for (const t of out.tags) {
        if (!t.cat) continue;
        const k = lc(t.cat);
        const arr = tagsByCat.get(k);
        if (arr) arr.push(t); else tagsByCat.set(k, [t]);
    }
    const sidesDeclared = Array.isArray(src.listCats);
    if (!sidesDeclared) {
        for (const c of out.categories) {
            if (listKeys.has(lc(c))) continue;
            const inCat = tagsByCat.get(lc(c));
            if (inCat && inCat.length && inCat.every((t) => t.kind === "list")) listKeys.add(lc(c));
        }
    }
    // 分类属于一侧，标签 kind 与分类侧冲突时标签移入自己的桶（kind 永远赢）
    for (const t of out.tags) {
        if (!t.cat) continue;
        const side = listKeys.has(lc(t.cat)) ? "list" : "text";
        if (side !== (t.kind === "list" ? "list" : "text")) t.cat = "";
    }
    out.listCats = out.categories.filter((c) => listKeys.has(lc(c)));
    // 分类 Picks 模式：键为规范名（改名/大小写变体不丢）；默认模式不写入
    const srcModes = src.catModes && typeof src.catModes === "object" ? src.catModes : {};
    const byKey = new Map(out.categories.map((c) => [lc(c), c]));
    byKey.set(lc(TEXT_BUCKET), TEXT_BUCKET);
    byKey.set(lc(LIST_BUCKET), LIST_BUCKET);
    out.catModes = {};
    for (const [k, v] of Object.entries(srcModes)) {
        const canon = byKey.get(lc(k));
        const m = cleanMode(v);
        if (canon && m !== DEFAULT_MODE) out.catModes[canon] = m;
    }
    return out;
}

// 列表标签按行拆分：每行 trim，空行丢弃
export function tagLines(text) {
    if (typeof text !== "string") return [];
    const out = [];
    for (const raw of text.split(/\r?\n/)) {
        const s = raw.trim();
        if (s) out.push(s);
    }
    return out;
}

export function isListTag(t) { return !!t && t.kind === "list"; }

// 标签归属分类：无分类时归入该侧的桶（Text / List）——桶名是隐式归属地，
// 侧栏桶行、导出桶、导入分组都依赖这个语义
export function catOf(t) { return (t && t.cat) || (isListTag(t) ? LIST_BUCKET : TEXT_BUCKET); }

// 分类属于哪一侧；两个桶各自回答自己；未知名称按 Text 侧处理
export function sideOfCat(name, data) {
    if (lc(name) === lc(LIST_BUCKET)) return "list";
    if (lc(name) === lc(TEXT_BUCKET)) return "text";
    const d = data || {};
    return (d.listCats || []).some((c) => lc(c) === lc(name)) ? "list" : "text";
}
export function isListCat(name, data) { return sideOfCat(name, data) === "list"; }

// 与 cat 同侧的分类，按显示顺序
export function catsOnSameSide(data, cat) {
    const d = data || {};
    const side = sideOfCat(cat, d);
    return (d.categories || []).filter((c) => sideOfCat(c, d) === side);
}

// 在分类数组中的位置：精确匹配优先，其次大小写不敏感
function catIndex(list, name) {
    const i = list.indexOf(name);
    if (i > -1) return i;
    const k = lc(name);
    return list.findIndex((c) => lc(c) === k);
}

// 将 cat 在其同侧块内上移（dir -1）/下移（dir +1）。返回新数组，无法移动返回 null
export function reorderCategoryStep(data, cat, dir) {
    const list = Array.isArray(data?.categories) ? data.categories : [];
    const from = catIndex(list, cat);
    if (from < 0 || (dir !== 1 && dir !== -1)) return null;
    const side = sideOfCat(list[from], data);
    let to = -1;
    for (let i = from + dir; i >= 0 && i < list.length; i += dir) {
        if (sideOfCat(list[i], data) === side) { to = i; break; }
    }
    if (to < 0) return null;
    const next = list.slice();
    const tmp = next[from]; next[from] = next[to]; next[to] = tmp;
    return next;
}

// 该步是否可行：直接由移动本身回答，展示与执行共用同一实现
export function canMoveCategory(data, cat, dir) {
    return reorderCategoryStep(data, cat, dir) !== null;
}

// 将 moved 移到 target 正上方（above true）或正下方。两者必须是同侧真实分类；
// 跨侧拒绝（分类属于一块，拖过去会清空其中所有标签的分类）。返回新数组或 null
export function reorderCategoryTo(data, moved, target, above) {
    const list = Array.isArray(data?.categories) ? data.categories : [];
    const from = catIndex(list, moved);
    const t0 = catIndex(list, target);
    if (from < 0 || t0 < 0 || from === t0) return null;
    if (sideOfCat(list[from], data) !== sideOfCat(list[t0], data)) return null;
    const side = sideOfCat(list[from], data);
    const next = list.slice();
    const [name] = next.splice(from, 1);
    const t = catIndex(next, target);
    if (t < 0) return null;
    next.splice(above ? t : t + 1, 0, name);
    // 按同侧序列比较：两侧在扁平数组中相对位置不可见，不该算一次移动
    const seq = (arr) => arr.filter((c) => sideOfCat(c, data) === side);
    const a = seq(next), b = seq(list);
    return a.every((c, i) => c === b[i]) ? null : next;
}

// 标签 / 分类的 Picks 模式
export function tagMode(t) { return cleanMode(t && t.mode); }
export function catMode(name, data) {
    const d = data || {};
    const m = d.catModes || {};
    // 只用自有属性，防 "toString"/"constructor" 读到 Object.prototype
    const own = Object.prototype.hasOwnProperty;
    return cleanMode(own.call(m, name) ? m[name] : (own.call(m, String(name)) ? m[String(name)] : undefined));
}

// 按名称查标签（大小写不敏感），无则 null
export function findTagIn(tags, name) {
    const k = String(name).toLowerCase();
    for (const t of tags) if (lc(t.name) === k) return t;
    return null;
}

// 生成不与现有名称（大小写不敏感）冲突的新标签名，追加 -2, -3, ...
export function uniqueTagName(base, tags, ignore) {
    let n = cleanTagName(base) || "tag";
    const taken = (x) => {
        const k = x.toLowerCase();
        for (const t of tags || []) {
            if (t === ignore) continue;
            if (lc(t.name) === k) return true;
        }
        return false;
    };
    if (!taken(n)) return n;
    const stem = n;
    let i = 2;
    while (taken(stem + "-" + i)) i++;
    return stem + "-" + i;
}

// 两侧是否内容一致（两侧都先规范化，键序/默认值差异不算不同）
export function isSameAsStored(data, stored) {
    try { return JSON.stringify(normalizeLibrary(data)) === JSON.stringify(normalizeLibrary(stored)); }
    catch { return false; }
}

// ── 导出 / 导入（数据变换，纯函数）──────────────────────────────────────
// 序列化库（或单个分类）为文件内容。cat 缺省/null = 全部
export function exportLibraryJSON(data, cat) {
    if (cat == null) return JSON.stringify(data, null, 2);
    const tags = (data.tags || []).filter((t) => lc(catOf(t)) === lc(cat));
    const categories = (data.categories || []).filter((c) => lc(c) === lc(cat));
    const listCats = categories.filter((c) => isListCat(c, data));
    // 桶也可导出（无分类标签），其 Picks 模式按作用域名查
    const catModes = {};
    for (const [k, v] of Object.entries(data.catModes || {})) if (lc(k) === lc(cat)) catModes[k] = v;
    return JSON.stringify({ version: 1, categories, listCats, catModes, tags }, null, 2);
}

// 导入文件包含的桶（按文件顺序 + 计数），供导入预览勾选
export function importCategories(parsed) {
    const out = [];
    const seen = new Map();
    for (const t of (parsed?.data?.tags || [])) {
        const c = catOf(t);
        const k = c.toLowerCase();
        if (!seen.has(k)) { seen.set(k, { name: c, count: 0 }); out.push(seen.get(k)); }
        seen.get(k).count += 1;
    }
    // 文件声明但无标签的分类也要提供（备份往返不能丢空分类）
    for (const c of (parsed?.data?.categories || [])) {
        const k = String(c || "").toLowerCase();
        if (!k || seen.has(k)) continue;
        seen.set(k, { name: c, count: 0 });
        out.push(seen.get(k));
    }
    return out;
}

// 把导入收窄到勾选的桶，并按当前库重算冲突。返回 { data, conflicts }
export function subsetImport(parsed, names, cur) {
    const keep = new Set((names || []).map((n) => lc(n)));
    const tags = (parsed?.data?.tags || []).filter((t) => keep.has(lc(catOf(t))));
    const categories = (parsed?.data?.categories || []).filter((c) => keep.has(lc(c)));
    const listCats = (parsed?.data?.listCats || []).filter((c) => keep.has(lc(c)));
    const catModes = {};
    for (const [k, v] of Object.entries(parsed?.data?.catModes || {})) if (keep.has(lc(k))) catModes[k] = v;
    const have = new Set((cur?.tags || []).map((t) => lc(t.name)));
    const conflicts = tags.filter((t) => have.has(lc(t.name))).map((t) => t.name);
    return { data: { version: 1, categories, listCats, catModes, tags }, conflicts };
}

// 解析导入文本为规范化库（不应用）。返回 { data, conflicts, dropped } 或 { error }
// cur 为当前库（用于计算冲突名）
export function parseImport(jsonStr, cur) {
    let raw;
    try { raw = JSON.parse(jsonStr); } catch { return { error: "That file is not valid JSON." }; }
    if (Array.isArray(raw)) raw = { tags: raw };
    else if (raw && !Array.isArray(raw.tags)) {
        raw = { categories: raw.categories, listCats: raw.listCats, catModes: raw.catModes, tags: raw.tags || raw.library || raw.snippets || raw.prompts };
    }
    // 文件内同名先去重（先按清洗后名称），否则 normalize 会静默丢弃后者
    if (raw && Array.isArray(raw.tags)) {
        const seen = new Set();
        const nextSuffix = new Map();
        raw.tags = raw.tags.map((t) => {
            if (!t || typeof t !== "object") return t;
            const base = cleanTagName(t.name);
            if (!base) return t;
            const baseKey = base.toLowerCase();
            let name = base;
            if (seen.has(baseKey)) {
                let i = nextSuffix.get(baseKey) || 2;
                while (seen.has((base + "-" + i).toLowerCase())) i++;
                name = base + "-" + i;
                nextSuffix.set(baseKey, i + 1);
            }
            seen.add(name.toLowerCase());
            return name === t.name ? t : { ...t, name };
        });
    }
    // 名称必须可输入为 @token，清不出来的（CJK/音标等）会被丢弃——先计数并报告
    let dropped = 0;
    if (raw && Array.isArray(raw.tags)) {
        for (const t of raw.tags) {
            if (!t || typeof t !== "object" || !cleanTagName(t.name)) dropped++;
        }
    }
    const data = normalizeLibrary(raw);
    if (!data.tags.length && !data.categories.length) {
        return {
            error: dropped
                ? `None of the ${dropped} tag${dropped === 1 ? "" : "s"} in that file can be used. A tag name can only contain letters a to z, numbers, - and _.`
                : "No tags found in that file.",
        };
    }
    const have = new Set((cur?.tags || []).map((t) => t.name.toLowerCase()));
    const conflicts = data.tags.filter((t) => have.has(t.name.toLowerCase())).map((t) => t.name);
    return { data, conflicts, dropped };
}

// 将导入合并进当前库（纯变换）。mode: "both"（冲突改名保留全部）/ "replace"
// （覆盖我方文本）/ "skip"（只加不冲突的）。返回 { data, added, replaced, replacedNames }
export function applyImportData(cur, parsed, mode) {
    const tags = (cur.tags || []).map((t) => ({ ...t }));
    const byKey = new Map(tags.map((t) => [t.name.toLowerCase(), t]));
    const uniqueIn = (base) => {
        let n = base;
        let i = 2;
        while (byKey.has(n.toLowerCase())) { n = base + "-" + i; i++; }
        return n;
    };
    const toAdd = [];
    const replaced = [];
    for (const inc of (parsed?.data?.tags || [])) {
        const key = inc.name.toLowerCase();
        if (!byKey.has(key)) {
            const t = { ...inc };
            toAdd.push(t);
            byKey.set(key, t);
        } else if (mode === "replace") {
            // 替换整个标签（text + kind + mode），而非只 text
            const t = byKey.get(key);
            t.text = inc.text;
            if (inc.kind === "list") t.kind = "list"; else delete t.kind;
            if (inc.mode) t.mode = inc.mode; else delete t.mode;
            replaced.push(t.name);
        } else if (mode === "both") {
            const nn = uniqueIn(inc.name);
            const t = { ...inc, name: nn };
            toAdd.push(t);
            byKey.set(nn.toLowerCase(), t);
        }
        // "skip": 什么都不做
    }
    const next = {
        version: 1,
        categories: [...(cur.categories || [])],
        listCats: [...(cur.listCats || [])],
        catModes: { ...(cur.catModes || {}) },
        tags: toAdd.concat(tags), // 新导入的在最前
    };
    const catHave = new Set(next.categories.map((c) => lc(c)));
    for (const c of (parsed?.data?.categories || [])) {
        if (c && !catHave.has(lc(c))) { catHave.add(lc(c)); next.categories.push(c); }
    }
    // 对方 List 分类只在名字空缺时补入（冲突时我的侧赢）
    const listHave = new Set(next.listCats.map((c) => lc(c)));
    for (const c of (parsed?.data?.listCats || [])) {
        if (c && !listHave.has(lc(c)) && !(cur.categories || []).some((x) => lc(x) === lc(c))) {
            listHave.add(lc(c));
            next.listCats.push(c);
        }
    }
    // Picks 模式同样：我的赢，对方只填空缺（含桶键）
    for (const [k, v] of Object.entries(parsed?.data?.catModes || {})) {
        if (!k) continue;
        const mine = (cur.categories || []).some((x) => lc(x) === lc(k)) ||
            Object.keys(cur.catModes || {}).some((x) => lc(x) === lc(k));
        if (!mine) next.catModes[k] = v;
    }
    return { data: next, added: toAdd.length, replaced: replaced.length, replacedNames: replaced };
}

// ── token 扫描 ─────────────────────────────────────────────────────────────

// `text[at-1]` 取的是 UTF-16 码元；对补充平面字符（CJK 扩展 B 等）会拆出半个代理对。
// 这里按完整码点返回前一字符，避免 "𠀀@tag" 被误判为 token。
export function prevCodePoint(text, at) {
    if (!(at > 0)) return "";
    const c = text[at - 1];
    if (at >= 2 && c >= "\uDC00" && c <= "\uDFFF") {
        const hi = text[at - 2];
        if (hi >= "\uD800" && hi <= "\uDBFF") return hi + c;
    }
    return c;
}

// 从左到右扫描 @tag / *wild / #list token。判定规则：token 需在行首、非单词字符
// 之后，或紧跟前一个同种 token（@a@b 链式有效）；邮件 user@name、算式 2*2 不误判。
// 跨种不链式（未知 *wildcard 不得把后面紧跟的 @tag 一并吞掉）。
// 返回 [{kind, sym, name, start, end, raw}]
export function scanTokens(text) {
    const out = [];
    if (typeof text !== "string" || !/[@*#]/.test(text)) return out;
    TOKEN_RE.lastIndex = 0;
    let m;
    let lastEnd = -1;
    let lastKind = null;
    while ((m = TOKEN_RE.exec(text))) {
        const at = m.index;
        const kind = KIND_BY_SYM[m[1]];
        const prev = prevCodePoint(text, at);
        const chains = at === lastEnd && kind === lastKind;
        const isTok = !prev || !/[\p{L}\p{N}\p{M}_]/u.test(prev) || chains;
        if (isTok) {
            out.push({ kind, sym: m[1], name: m[2], start: at, end: at + m[0].length, raw: m[0] });
            lastEnd = at + m[0].length;
            lastKind = kind;
        }
        // 非 token 的 @/* 不更新 lastEnd/lastKind，不能开启链式
    }
    return out;
}

export function scanTags(text) { return scanTokens(text).filter((t) => t.kind === "tag"); }
export function scanWilds(text) { return scanTokens(text).filter((t) => t.kind === "wild"); }
export function scanLists(text) { return scanTokens(text).filter((t) => t.kind === "list"); }

export function hasTags(text) {
    if (typeof text !== "string" || text.indexOf("@") === -1) return false;
    return scanTags(text).length > 0;
}
export function hasWilds(text) {
    if (typeof text !== "string" || text.indexOf("*") === -1) return false;
    return scanWilds(text).length > 0;
}
export function hasLists(text) {
    if (typeof text !== "string" || text.indexOf("#") === -1) return false;
    return scanLists(text).length > 0;
}

// ── 展开 ────────────────────────────────────────────────────────────────────

// 展开 @tags 并解析 *wildcards / #lists。resolveWild(name) / resolveList(name)
// 返回替换字符串，或 null/undefined 表示保留该 token 字面量（未知名称等）。
// 随机性由调用方控制：队列时传真随机（游标），预览时传稳定占位符。
// tags 缺省传入 []（纯函数，不触达设置）。
// 返回 { out, spans, knownTags, unknownTags, knownWilds, unknownWilds, knownLists, unknownLists }
// spans: [{start, end, kind, name, known}]，标记替换文本在 out 中的位置，供着色。
export function expandAll(text, opts = {}) {
    const { tags = [], resolveWild, resolveList } = opts;
    if (typeof text !== "string" || !/[@*#]/.test(text)) {
        return {
            out: typeof text === "string" ? text : "",
            spans: [],
            knownTags: [], unknownTags: [], knownWilds: [], unknownWilds: [], knownLists: [], unknownLists: [],
        };
    }
    const map = new Map();
    for (const t of tags) map.set(lc(t.name), t);
    const toks = scanTokens(text);
    const knownTags = [], unknownTags = [], knownWilds = [], unknownWilds = [], knownLists = [], unknownLists = [];
    const spans = [];
    let out = "";
    let i = 0;
    for (const h of toks) {
        out += text.slice(i, h.start);
        const at = out.length;
        let known = false;
        if (h.kind === "tag") {
            const t = map.get(h.name.toLowerCase());
            if (t && typeof t.text === "string") { out += t.text; knownTags.push(h.name); known = true; }
            else { out += h.raw; unknownTags.push(h.name); }
        } else if (h.kind === "wild") {
            const rep = typeof resolveWild === "function" ? resolveWild(h.name) : null;
            if (rep != null) { out += rep; knownWilds.push(h.name); known = true; }
            else { out += h.raw; unknownWilds.push(h.name); }
        } else {
            const rep = typeof resolveList === "function" ? resolveList(h.name) : null;
            if (rep != null) { out += rep; knownLists.push(h.name); known = true; }
            else { out += h.raw; unknownLists.push(h.name); }
        }
        spans.push({ start: at, end: out.length, kind: h.kind, name: h.name, known });
        i = h.end;
    }
    out += text.slice(i);
    return { out, spans, knownTags, unknownTags, knownWilds, unknownWilds, knownLists, unknownLists };
}
