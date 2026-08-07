// ==========================================================================
// sf_prompt_tags_cursors.js - SFPromptTags Picks 游标系统
// ==========================================================================
//
// 一个 #list 或 *category 在非 Random 模式下要记住"进行到哪了"：
//   * shuffle - 一叠牌：每个选项出现一次才重复，发完重洗（新牌堆不开旧堆最后
//     那张牌）
//   * order   - 1,2,3,... 循环
// 位置存于未注册设置 "sfnodes.PromptTags.Cursors"（键 "list:<name>" /
// "cat:<category>"，小写），属于列表本身：两个节点用 #poses 走同一条序列。
// 永不写入工作流，也永不写入库 blob（导出只带标签，不带进度）。
//
// 队列语义（与 Pixaroma 一致）：
//   - app.graphToPrompt 不是队列（Export / 分享 / 保存按钮都会触发），所以掷出
//     的选择先存在 _pending 里；只有当一次 queuePrompt 真正被接受后才推进
//     （commitPicks），否则同一选择反复给下一次 run。
//   - 同一 build 内 #fruit #fruit #fruit 各拿一张新牌（按 occ 计数发牌）；
//     In order 例外：每个 run 只推进一次，所以同一 build 内全部相同。
//   - build id 挂在 prompt 对象上（WeakMap），防止窗口期内别的 graphToPrompt
//     把全局计数移走。
//
// ==========================================================================

import { app } from "/scripts/app.js";
import { MODES, DEFAULT_MODE, cleanMode } from "./sf_prompt_tags_lib.js";

const CURSOR_SETTING = "sfnodes.PromptTags.Cursors";

export function isMode(m) { return MODES.includes(m); }
export function hasPosition(m) { return cleanMode(m) !== "random"; }

export const listKey = (name) => "list:" + String(name).toLowerCase();
export const catKey = (name) => "cat:" + String(name).toLowerCase();

let _data = null;
let _loaded = false;
let _timer = null;

function settingsApi() {
    const s = app.ui?.settings;
    return s && typeof s.getSettingValue === "function" ? s : null;
}
// 游标映射，或 null（设置未就绪）。返回 null 而非 {} 有意为之：缓存空对象会让
// 真正的已存位置在设置到达后永远被遮蔽；调用方降级为纯随机（见 rollIndex）。
function all() {
    if (_loaded) return _data;
    const s = settingsApi();
    if (!s) return null;
    const raw = s.getSettingValue(CURSOR_SETTING);
    try { _data = (raw && typeof raw === "string" ? JSON.parse(raw) : raw) || {}; }
    catch { _data = {}; }
    if (!_data || typeof _data !== "object" || Array.isArray(_data)) _data = {};
    _loaded = true;
    return _data;
}
function persist() {
    const s = app.ui?.settings;
    if (!s || !_loaded || !_data) return;
    const json = JSON.stringify(_data);
    try {
        if (typeof s.setSettingValueAsync === "function") s.setSettingValueAsync(CURSOR_SETTING, json);
        else if (typeof s.setSettingValue === "function") s.setSettingValue(CURSOR_SETTING, json);
    } catch { /* 非致命：本会话内存态正确 */ }
}
// run 是成批的（一次队列十个 run 十次掷），合并写入
function touch() {
    if (_timer) clearTimeout(_timer);
    _timer = setTimeout(() => { persist(); _timer = null; }, 300);
}
export function flushCursors() {
    if (_timer) { clearTimeout(_timer); _timer = null; }
    persist();
}
// 队列后立刻关标签页，300ms 防抖内的推进会丢；尽力而为在离开时落盘
if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    window.addEventListener("pagehide", () => { try { flushCursors(); } catch { /* ignore */ } });
}

// 牌堆必须是由界内不重复下标组成；坏牌堆整副丢弃重洗（不能发 [0,0,1]）
function validBag(bag, n) {
    if (!Array.isArray(bag)) return false;
    const seen = new Set();
    for (const x of bag) {
        if (!Number.isInteger(x) || x < 0 || x >= n || seen.has(x)) return false;
        seen.add(x);
    }
    return true;
}

function shuffled(n) {
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// 已掷出但尚未被 run 消耗的选择。见文件头注释（队列语义）。
const _pending = new Map();   // key -> { i, n, mode, state, build }
let _build = 0;
const _buildOf = new WeakMap();
export function beginPickBuild(promptObj) {
    _build++;
    if (promptObj && typeof promptObj === "object") {
        try { _buildOf.set(promptObj, _build); } catch { /* not weak-mappable, fall back */ }
    }
    return _build;
}
// `queued` 是真正 POST 的 prompt 对象，消耗的恰好是产生它的那次 build 的选择
export function commitPicks(queued) {
    if (!_pending.size) return;
    let build = _build;
    if (queued && typeof queued === "object") {
        const b = _buildOf.get(queued);
        if (b != null) build = b;
    }
    const map = all();
    let wrote = false;
    for (const [key, p] of _pending) {
        if (p.build !== build) continue;           // 未入队的 build 掷的不消耗
        if (map && p.state) { map[key] = p.state; wrote = true; }
        _pending.delete(key);
    }
    // 只有确实发生了持久变化才写（全是 Random 的 run 没有可存的）
    if (wrote) flushCursors();
}

// 现在要用的下标，并推进游标。`len` 是当前池大小（编辑改长度 = 重新开始）。
// 一个 build 内同一 key 的第 occ 次使用（#fruit #fruit = 0,1,2）。
// In order 例外：每 run 只推进一次，同一 build 内全部返回同一项。
export function nextIndex(key, len, mode, occ = 0) {
    const n = Math.floor(len);
    if (!(n > 0)) return -1;
    const m = cleanMode(mode);
    const want = m === "order" ? 0 : Math.max(0, Math.floor(occ) || 0);
    const held = _pending.get(key);
    if (held && held.n === n && held.mode === m) {
        // 重贴 stamp：本 build 在用这个选择，所以它归本 build 消耗
        held.build = _build;
        while (held.picks.length <= want) {
            const more = rollIndex(key, n, m, held.state);
            if (more.i < 0) break;
            held.picks.push(more.i);
            held.state = more.state;
        }
        return held.picks[Math.min(want, held.picks.length - 1)];
    }
    const r = rollIndex(key, n, m);
    if (r.i < 0) return r.i;
    const rec = { picks: [r.i], n, mode: m, state: r.state, build: _build };
    _pending.set(key, rec);
    while (rec.picks.length <= want) {
        const more = rollIndex(key, n, m, rec.state);
        if (more.i < 0) break;
        rec.picks.push(more.i);
        rec.state = more.state;
    }
    return rec.picks[Math.min(want, rec.picks.length - 1)];
}

// 真正的抽取。返回 { i, state }：state 是该选择被 run 消耗后应存的游标。
// 不在此处写入——由 commitPicks 应用。
function rollIndex(key, n, m, from) {
    if (m === "random" || n === 1) return { i: Math.floor(Math.random() * n), state: from ?? null };
    const map = all();
    // 设置未就绪：无法记位置，降级纯随机（不假装在序列化后把结果丢掉）
    if (!map) return { i: Math.floor(Math.random() * n), state: null };
    let st = from !== undefined ? from : map[key];
    if (!st || typeof st !== "object" || st.n !== n) st = null;   // 池变了 -> 重来

    if (m === "order") {
        const i = st && Number.isInteger(st.i) ? ((st.i % n) + n) % n : 0;
        return { i, state: { n, i: (i + 1) % n, last: i } };
    }
    // shuffle：从牌堆发牌，发完重洗
    let bag = validBag(st && st.bag, n) ? st.bag.slice() : null;
    const last = st && Number.isInteger(st.last) ? st.last : -1;
    if (!bag || !bag.length) {
        bag = shuffled(n);
        // 新牌堆不开旧堆最后那张牌（这正是该模式要避免的连续重复）。
        // 随机交换而非轮转到最前：轮转会把被挡的牌堆映射到同一个允许牌堆，
        // 让那个牌堆概率翻倍；随机交换使允许牌堆均匀。
        if (n > 1 && bag[bag.length - 1] === last) {
            const j = Math.floor(Math.random() * (n - 1));
            [bag[n - 1], bag[j]] = [bag[j], bag[n - 1]];
        }
    }
    const i = bag.pop();
    return { i, state: { n, bag, last: i } };
}

// 编辑器里显示的位置文本：random 或还没跑过返回 null
export function cursorInfo(key, len, mode) {
    const n = Math.floor(len);
    const m = cleanMode(mode);
    if (m === "random" || !(n > 0)) return null;
    const map = all();
    const st = map && map[key];
    if (!st || typeof st !== "object" || st.n !== n) return m === "order" ? `next 1 of ${n}` : `${n} left`;
    if (m === "order") {
        const i = Number.isInteger(st.i) ? ((st.i % n) + n) % n : 0;
        return `next ${i + 1} of ${n}`;
    }
    const left = validBag(st.bag, n) ? st.bag.length : 0;
    return `${left || n} left in the deck`;
}

// 从头开始（牌堆重洗 / 计数回到 1）
export function resetCursor(key) {
    _pending.delete(key);          // 持有的选择属于旧序列
    const map = all();
    if (map && map[key]) { delete map[key]; touch(); }
}

// 把位置带到新名字下（改名不是内容变化，"next 4 of 12" 不应变成 "next 1 of 12"）
export function renameCursor(fromKey, toKey) {
    if (fromKey === toKey) return;
    const held = _pending.get(fromKey);
    if (held) { _pending.set(toKey, held); _pending.delete(fromKey); }
    const map = all();
    if (!map || !map[fromKey]) return;
    map[toKey] = map[fromKey];
    delete map[fromKey];
    touch();
}
