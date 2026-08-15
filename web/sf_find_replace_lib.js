// ==========================================================================
// sf_find_replace_lib.js - SFTextFindReplace 纯函数库（state + 替换逻辑 + 词级 diff）
// ==========================================================================
//
// 无 app/DOM 依赖，供主扩展 sf_find_replace.js 与 UI 模块 import，也供
// tests/ 复制为 .mjs 直接测试。
//
// 状态存 node.properties.findReplaceState：
//   { version:1, caseSensitive, wholeWord, regex, tidy, rules:[{id,enabled,find,replace}] }
// LiteGraph 原生序列化 node.properties。sf_find_replace.js 的 graphToPrompt hook
// 把状态（不含预览）打包进隐藏 FindReplaceState 输入。
//
// 节点上的预览由 applyRulesJS() 驱动——nodes/text/find_replace.py::_apply_rules
// 的 1:1 镜像。Python 是权威。literal 模式（含 whole-word）对常见情况与 Python
// 的 Unicode 大小写折叠一致（构建带 /u flag 的正则，使忽略大小写对非 ASCII 的
// 折叠方式与 re.IGNORECASE 相同；个别 locale 特殊折叠如土耳其点 i 仍略异，那些
// 罕见情况预览稍有出入但运行正确）。Regex 模式对预览是尽力而为：翻译常见
// Python-only 语法（\\1/$1 反向引用、\\g<n>/\\g<0>、命名组 (?P<n>) 与反向引用
// (?P=n)），并优先 /u、回退非 /u；但仍有一些 Python 正则构造与 JS 不同、真实
// Python 运行才是权威：字符类简写 \\w \\d \\s \\b（及 \\W \\D \\S \\B）在 JS
// 预览中仅 ASCII 而 Python 中 Unicode 感知（如 \\w+ 作用于重音/希腊/CJK 文本时
// 预览比实际更窄）；替换文本以孤立反斜杠结尾（Python 报错、JS 保留）；模式开头
// 的内联 flags 如 (?s)/(?m)（请用作用域 (?s:...)）；替换文本中的 \\10 式两位
// 引用 / \\0；以及少量 Python 拒绝的非法替换模板（跳过该规则并警告）而 JS 预览
// 静默变成错误的字面文本（未知转义如 \\q、超出组数的数字反向引用、未闭合的
// \\g<name）。在这些情况下节点预览可能稍有出入；Python 运行为准。
//
// ==========================================================================

// ── state（node.properties）─────────────────────────────────────────────
export const STATE_PROP = "findReplaceState";
export const PREVIEW_PROP = "findReplacePreview";

let _idCounter = 0;
function nextId() {
    _idCounter += 1;
    // Date + 会话内计数器 + 小随机后缀，即使跨页面刷新 id 也唯一（计数器每次
    // 会话归零，而 id 是删除/排序的键）。浏览器环境 Math.random 足够。
    const rnd = Math.floor(Math.random() * 1e6).toString(36);
    return `fr${Date.now().toString(36)}_${_idCounter}_${rnd}`;
}

export function freshRule(overrides = {}) {
    return { id: nextId(), enabled: true, find: "", replace: "", ...overrides };
}

export function defaultState() {
    return {
        version: 1,
        caseSensitive: false,
        wholeWord: false,
        regex: false,
        tidy: true,
        rules: [freshRule()],
    };
}

export function readState(node) {
    const s = node.properties?.[STATE_PROP];
    if (!s || typeof s !== "object") return defaultState();
    if (!Array.isArray(s.rules) || s.rules.length === 0) return defaultState();
    if (typeof s.caseSensitive !== "boolean") s.caseSensitive = false;
    if (typeof s.wholeWord !== "boolean") s.wholeWord = false;
    if (typeof s.regex !== "boolean") s.regex = false;
    if (typeof s.tidy !== "boolean") s.tidy = true;
    // 丢弃任何非对象行（损坏/手编工作流可能带 null/string 条目）——镜像 Python
    // 的 `isinstance(rule, dict)` 防护，使畸形状态不会让 readState（它喂给每个
    // 渲染、mutator 与 graphToPrompt hook）抛错。.some() 检查意味着干净状态
    // 永不重写（无改动、加载不脏）；只有畸形状态才被重写。
    if (s.rules.some((row) => !row || typeof row !== "object")) {
        s.rules = s.rules.filter((row) => row && typeof row === "object");
    }
    if (s.rules.length === 0) return defaultState();
    for (const row of s.rules) {
        if (typeof row.id !== "string" || !row.id) row.id = nextId();
        if (typeof row.enabled !== "boolean") row.enabled = true;
        if (typeof row.find !== "string") row.find = "";
        if (typeof row.replace !== "string") row.replace = "";
    }
    return s;
}

export function writeState(node, state) {
    node.properties = node.properties || {};
    node.properties[STATE_PROP] = state;
}

export function restoreFromProperties(node) {
    writeState(node, readState(node));
}

// ── mutators ─────────────────────────────────────────────────────────────

export function addRule(node) {
    const state = readState(node);
    state.rules.push(freshRule());
    writeState(node, state);
}

export function deleteRule(node, id) {
    const state = readState(node);
    if (state.rules.length <= 1) return;
    state.rules = state.rules.filter((r) => r.id !== id);
    writeState(node, state);
}

export function toggleRuleEnabled(node, id) {
    const state = readState(node);
    const row = state.rules.find((r) => r.id === id);
    if (row) row.enabled = !row.enabled;
    writeState(node, state);
}

export function setFind(node, id, v) {
    const state = readState(node);
    const row = state.rules.find((r) => r.id === id);
    if (row) row.find = String(v || "");
    writeState(node, state);
}

export function setReplace(node, id, v) {
    const state = readState(node);
    const row = state.rules.find((r) => r.id === id);
    if (row) row.replace = String(v || "");
    writeState(node, state);
}

export function setToggle(node, key) {
    const state = readState(node);
    if (key in state) state[key] = !state[key];
    writeState(node, state);
}

export function reorderRules(node, fromIdx, toIdx) {
    const state = readState(node);
    if (fromIdx === toIdx) return;
    if (fromIdx < 0 || fromIdx >= state.rules.length) return;
    if (toIdx < 0 || toIdx >= state.rules.length) return;
    const [moved] = state.rules.splice(fromIdx, 1);
    state.rules.splice(toIdx, 0, moved);
    writeState(node, state);
}

export function resetToDefault(node) {
    writeState(node, defaultState());
}

// ── 预览持久化 ──────────────────────────────────────────────────────────
// 与规则状态分开存储，因此不会注入 prompt。

export function getPreviewInput(node) {
    const p = node.properties?.[PREVIEW_PROP];
    if (!p || typeof p !== "object" || typeof p.input !== "string") return null;
    return p;
}

const PREVIEW_CAP = 4000;
export function setPreviewInput(node, input, truncated) {
    node.properties = node.properties || {};
    // 自我保护上限（Python 已限制 4000，但永远别信任调用方）：样本会序列化进
    // 工作流 JSON，所以这里也限制，未来未封顶的调用方不能撑爆保存的文件。
    const s = String(input == null ? "" : input);
    const over = s.length > PREVIEW_CAP;
    node.properties[PREVIEW_PROP] = {
        input: over ? s.slice(0, PREVIEW_CAP) : s,
        truncated: !!truncated || over,
    };
}

// ── 替换逻辑（镜像 nodes/text/find_replace.py::_apply_rules）───────────

function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function tidy(s) {
    s = s.replace(/[ \t]+/g, " ");
    s = s.replace(/[ \t]+,/g, ",");
    s = s.replace(/,(?:[ \t]*,)+/g, ",");
    s = s.replace(/[ \t]+(\r?\n)/g, "$1");
    s = s.replace(/^[ \t]*,[ \t]*/, "");
    s = s.replace(/,[ \t]*$/, "");
    return s.trim();
}

// 把 Python re.sub() 的替换模板翻译成 JS String.replace 的替换字符串，使预览
// 与权威 Python 运行一致。处理：\\1..\\99 与 \\g<name>/\\g<number> 反向引用、
// \\n \\t \\r \\f \\v 与 \\\\ 字符转义（Python 在替换文本中处理这些，JS 不处理）、
// 以及字面 $（JS 中特殊、Python 中字面 -> 转义为 $$）。
function pyTemplateToJs(repl) {
    let out = "";
    for (let i = 0; i < repl.length; i++) {
        const ch = repl[i];
        if (ch === "$") { out += "$$"; continue; }      // 字面 $（Python 保留）
        if (ch !== "\\") { out += ch; continue; }
        const nx = repl[i + 1];
        if (nx === undefined) { out += "\\"; break; }    // 尾部反斜杠 -> 字面
        if (nx === "\\") { out += "\\"; i++; continue; } // \\ -> 一个字面反斜杠
        if (nx >= "1" && nx <= "9") {                    // \\1..\\99 组引用
            let num = nx; i++;
            if (repl[i + 1] >= "0" && repl[i + 1] <= "9") { num += repl[i + 1]; i++; }
            out += "$" + num;
            continue;
        }
        if (nx === "g") {                                // \\g<name> 或 \\g<number>
            const m = /^\\g<([^>]+)>/.exec(repl.slice(i));
            if (m) {
                const ref = m[1];
                if (/^\d+$/.test(ref)) {
                    // JS 没有 $0；整个匹配是 $&。（Python \\g<0> = 整个匹配。）
                    out += ref === "0" ? "$&" : "$" + ref;
                } else {
                    out += "$<" + ref + ">";
                }
                i += m[0].length - 1;
                continue;
            }
            out += "g"; i++; continue;                     // 畸形 \\g -> 尽力而为
        }
        const map = { n: "\n", t: "\t", r: "\r", f: "\f", v: "\v" };
        if (nx in map) { out += map[nx]; i++; continue; }
        out += nx; i++;                                   // 未知转义 -> 字面字符
    }
    return out;
}

// Python 的 \b/\w 对 str 模式是 Unicode 感知的；JS 的 \b 仅 ASCII。
// whole-word 字面匹配时构建显式 Unicode 感知边界断言，使重音/非拉丁单词与
// Python 运行匹配一致。
const _WORD = "\\p{L}\\p{N}_";
function isWordChar(c) {
    return /[\p{L}\p{N}_]/u.test(c || "");
}

// 尽力翻译 Python-only 正则 PATTERN 语法使预览可编译：命名组定义
// (?P<n>...) -> (?<n>...)，命名反向引用 (?P=n) -> \\k<n>。其他 Python-only
// 构造仍不同（头注释已记录）。
function pyPatternToJs(pat) {
    return pat
        .replace(/\(\?P</g, "(?<")
        .replace(/\(\?P=([A-Za-z_]\w*)\)/g, "\\k<$1>");
}

// 构建优先带 Unicode flag 的 RegExp，使忽略大小写匹配对非 ASCII（Kelvin 符号、
// 重音字母、土耳其点 i）的折叠方式与 Python re.IGNORECASE 一致。若用户的模式
// 不带 u 才能编译则回退（regex 模式——转义的字面量总是 /u 安全的）。返回的
// RegExp 仍可能对真正无效的模式抛错；调用方的 try/catch 把它变成 "invalid
// regex" 警告。
function makeRegexU(pattern, flags) {
    try {
        return new RegExp(pattern, flags + "u");
    } catch (_e) {
        return new RegExp(pattern, flags);
    }
}

// 位置 j 的无界量词：* 或 + 或 {n,}（开放结尾）。{n} 与 {n,m} 有界 -> 安全。
function unboundedQuantAt(src, j) {
    const c = src[j];
    if (c === "*" || c === "+") return true;
    return /^\{\d*,\}/.test(src.slice(j));
}
// 交替型指数回溯启发式（镜像 nodes/text/find_replace.py::_alternation_overlap_risk）：
// (a|aa)+ / (a|a?)+ / (a|)+ 家族——组内顶层 | 分出至少两个分支、任意两分支的
// 首字符集合重叠、且组后紧跟无界量词 → 匹配方式随输入长度指数增长。分支互斥
// （(a|b)+）不命中。与 Python 端 1:1 同步，预览与运行一致。
const _ANY_CHARS = Symbol("any");
const _EMPTY = Symbol("empty");

function splitTopLevelAlt(body) {
    const branches = [];
    let start = 0;
    let escaped = false;
    let inClass = false;
    let depth = 0;
    for (let i = 0; i < body.length; i++) {
        const c = body[i];
        if (escaped) { escaped = false; continue; }
        if (c === "\\") { escaped = true; continue; }
        if (inClass) { if (c === "]") inClass = false; continue; }
        if (c === "[") { inClass = true; continue; }
        if (c === "(") { depth++; continue; }
        if (c === ")") { depth--; continue; }
        if (c === "|" && depth === 0) { branches.push(body.slice(start, i)); start = i + 1; }
    }
    branches.push(body.slice(start));
    return branches;
}

function classFirstChars(seg) {
    let negate = false;
    const chars = new Set();
    let j = 1;
    while (j < seg.length) {
        const c = seg[j];
        if (c === "^" && j === 1) { negate = true; j++; continue; }
        if (c === "\\") {
            const e = seg[j + 1];
            if (e === undefined) break;
            if ("dDwWsS".includes(e)) return _ANY_CHARS;
            chars.add(e); j += 2; continue;
        }
        if (c === "]") break;
        if (seg[j + 1] === "-" && seg[j + 2] !== "]" && seg[j + 2] !== undefined) {
            const a = c;
            const b = seg[j + 2];
            if (b.charCodeAt(0) - a.charCodeAt(0) <= 64) {
                for (let k = a.charCodeAt(0); k <= b.charCodeAt(0); k++) chars.add(String.fromCharCode(k));
            } else { return _ANY_CHARS; }
            j += 3; continue;
        }
        chars.add(c); j++;
    }
    if (negate) return _ANY_CHARS;
    return chars;
}

function branchFirstChars(branch) {
    if (!branch) return _EMPTY;
    let j = 0;
    while (j < branch.length && (branch[j] === "^" || branch[j] === "$")) j++;
    if (j >= branch.length) return _EMPTY;
    const c = branch[j];
    if (c === "\\") {
        const e = branch[j + 1];
        if (e === undefined) return null;
        if ("dDwWsS".includes(e)) return _ANY_CHARS;
        if ("bBAZ".includes(e)) return branchFirstChars(branch.slice(j + 2));
        return new Set([e]);
    }
    if (c === "[") return classFirstChars(branch.slice(j));
    if (c === ".") return _ANY_CHARS;
    if (c === "(") return null;
    if ("*+?{".includes(c)) return null;
    return new Set([c]);
}

function alternationOverlapRisk(src) {
    const groups = [];
    const stack = [];
    let escaped = false;
    let inClass = false;
    for (let i = 0; i < src.length; i++) {
        const c = src[i];
        if (escaped) { escaped = false; continue; }
        if (c === "\\") { escaped = true; continue; }
        if (inClass) { if (c === "]") inClass = false; continue; }
        if (c === "[") { inClass = true; continue; }
        if (c === "(") { stack.push(i); continue; }
        if (c === ")") { if (stack.length) groups.push([stack.pop(), i]); continue; }
    }
    for (const [gs, ge] of groups) {
        const branches = splitTopLevelAlt(src.slice(gs + 1, ge));
        if (branches.length < 2) continue;
        const q = src[ge + 1] || "";
        let unbounded = q === "*" || q === "+";
        if (q === "{" && /^\{\d*,\}/.test(src.slice(ge + 1))) unbounded = true;
        if (!unbounded) continue;
        const firsts = branches.map(branchFirstChars);
        let skip = false;
        const seen = new Set();
        for (const f of firsts) {
            if (f === null) { skip = true; break; }
            if (f === _EMPTY || f === _ANY_CHARS) return true;
            let overlap = false;
            for (const ch of f) { if (seen.has(ch)) { overlap = true; break; } }
            if (overlap) return true;
            for (const ch of f) seen.add(ch);
        }
        if (skip) continue;
    }
    return false;
}


// 对灾难性回溯（"ReDoS"）模式的启发式防护。嵌套的无界量词——无界量词限定的组、
// 其体内还含无界量词，如 (a+)+ (a*)* (.*)* (\\w+)+——对不匹配的输入可能指数级
// 耗时。该预览在每次按键时重算（同样的模式每次 Run 也在服务端无超时运行），
// 所以此类模式会冻结标签页/卡死 worker。原生正则无法限时，因此拒绝明显的嵌套
// 量词形状并带警告跳过该规则。镜像于 nodes/text/find_replace.py::
// _is_catastrophic_regex，使预览与运行一致。启发式而非完备：捕获常见的意外
// 形状，不是每种 ReDoS。误报率低——嵌套无界量词总是冗余的（(a+)+ == a+），
// 合法模式不会使用。
export function isCatastrophicRegex(src) {
    const stack = []; // 每个打开的组一个 {inner}；inner = 组体内含无界量词
    let escaped = false;
    let inClass = false;
    for (let i = 0; i < src.length; i++) {
        const c = src[i];
        if (escaped) { escaped = false; continue; }
        if (c === "\\") { escaped = true; continue; }
        if (inClass) { if (c === "]") inClass = false; continue; }
        if (c === "[") { inClass = true; continue; }
        if (c === "(") { stack.push({ inner: false }); continue; }
        if (c === ")") {
            const grp = stack.pop() || { inner: false };
            const quant = unboundedQuantAt(src, i + 1);
            if (quant && grp.inner) return true; // 嵌套无界量词
            // 被限量的组本身对其父组是一个无界 token
            if (quant && stack.length) stack[stack.length - 1].inner = true;
            continue;
        }
        if (unboundedQuantAt(src, i)) {
            if (stack.length) stack[stack.length - 1].inner = true;
            continue;
        }
    }
    if (alternationOverlapRisk(src)) return true;
    return false;
}

// 返回 { output, warnings:[string] }。
export function applyRulesJS(text, state) {
    const rules = Array.isArray(state.rules) ? state.rules : [];
    const cs = !!state.caseSensitive;
    const ww = !!state.wholeWord;
    const rx = !!state.regex;
    const td = state.tidy !== false;
    const warnings = [];
    let out = String(text == null ? "" : text);
    const baseFlags = "g" + (cs ? "" : "i");

    rules.forEach((rule, idx) => {
        if (!rule || rule.enabled === false) return;
        // 非字符串 find/replace 强转为 ""（与 readState 及 Python 引擎一致），
        // 使畸形规则无论调用方是谁都不会在这里抛错。
        const find = typeof rule.find === "string" ? rule.find : "";
        if (!find) return;
        const repl = typeof rule.replace === "string" ? rule.replace : "";
        try {
            if (rx) {
                if (isCatastrophicRegex(find)) {
                    warnings.push(`Rule ${idx + 1}: pattern may be catastrophically slow (nested quantifier) - simplify it`);
                    return; // 跳过该规则（与 Python 相同）使预览不会冻结
                }
                const re = makeRegexU(pyPatternToJs(find), baseFlags);
                out = out.replace(re, pyTemplateToJs(repl));
            } else {
                let pat = escapeRegex(find);
                if (ww) {
                    // 镜像 Python 的 \bTERM\b：两侧断言取决于边缘字符本身是否为
                    // 词字符，因此对任意 TERM（含标点边缘）都匹配 Python，并带
                    // Unicode 词字符。
                    const lead = isWordChar(find[0]) ? `(?<![${_WORD}])` : `(?<=[${_WORD}])`;
                    const tail = isWordChar(find[find.length - 1]) ? `(?![${_WORD}])` : `(?=[${_WORD}])`;
                    pat = lead + pat + tail;
                }
                // 转义的字面量总是 /u 安全的，因此这里始终用 /u，以匹配 Python
                // 的 Unicode 大小写折叠（修复了此前忽略大小写时仅 ASCII 匹配
                // 的非 whole-word 字面情形）。
                const re = new RegExp(pat, baseFlags + "u");
                // 字面替换：原样插入 repl（JS 中只有 $ 特殊；反斜杠是字面量，
                // 匹配 Python 的反斜杠双写 safe_repl）。
                out = out.replace(re, repl.replace(/\$/g, "$$$$"));
            }
        } catch (_e) {
            warnings.push(`Rule ${idx + 1}: invalid regex`);
        }
    });

    if (td) out = tidy(out);
    return { output: out, warnings };
}

// ── 前后对比的词级 diff ─────────────────────────────────────────────────

function tokenize(s) {
    return s.match(/\s+|[^\s]+/g) || [];
}

// 基于 LCS 的 token diff。返回 [{t:'eq'|'del'|'ins', s}]。
export function diffTokens(aStr, bStr) {
    const a = tokenize(aStr);
    const b = tokenize(bStr);
    const n = a.length;
    const m = b.length;
    // 防御病态的 token 数量。预览样本上限 4000 字符，但很多短空格分隔 token 的
    // 样本仍可 tokenize 出数千个 token，而下面的 DP 在时间与内存上都是 O(n*m)，
    // 并且每次按键都会重算（按帧合并）。旧的 4M 上限下最坏 ~2000x2000 的 diff
    // 每帧分配约 16MB 并跑 4M 次迭代，在编辑长提示词的规则时短暂冻结浏览器。
    // 1M 让正常散文/标签提示词保留真实词级 diff，只把非常大的样本降级为整个
    // 字符串 diff。
    if (n * m > 1_000_000) {
        return [{ t: "del", s: aStr }, { t: "ins", s: bStr }];
    }
    const dp = [];
    for (let i = 0; i <= n; i++) dp.push(new Int32Array(m + 1));
    for (let i = n - 1; i >= 0; i--) {
        for (let j = m - 1; j >= 0; j--) {
            dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
    }
    const out = [];
    let i = 0;
    let j = 0;
    while (i < n && j < m) {
        if (a[i] === b[j]) {
            out.push({ t: "eq", s: a[i] });
            i++;
            j++;
        } else if (dp[i + 1][j] >= dp[i][j + 1]) {
            out.push({ t: "del", s: a[i] });
            i++;
        } else {
            out.push({ t: "ins", s: b[j] });
            j++;
        }
    }
    while (i < n) out.push({ t: "del", s: a[i++] });
    while (j < m) out.push({ t: "ins", s: b[j++] });
    return out;
}

export function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
