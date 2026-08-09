// ==========================================================================
// SF LoRA Stack - server routes 的薄 fetch 封装。列表会话级缓存；info 按
// LoRA 缓存，重开信息面板即时。Civitai 调用永不在此缓存（服务端以侧车文件
// 缓存，第二次调用即时且离线）。
// ==========================================================================
import { sfApiUrl } from "./sf_common.js";

let _listCache = null;
let _listPromise = null;
const _infoCache = new Map();
const _infoPromise = new Map();

export async function listLoras(force = false) {
    if (!force && _listCache) return _listCache;
    if (!force && _listPromise) return _listPromise;
    const p = (async () => {
        try {
            // no-store：路由不发缓存头，浏览器启发式缓存这份列表正是
            // "改名的文件永不出现"的 bug。
            const r = await fetch(sfApiUrl("/api/sfnodes/lora_list"), { cache: "no-store" });
            const j = await r.json();
            // j.error = 服务端目录扫描失败（不是空文件夹）。按网络失败处理：
            // 保留手头列表。把它当真 [] 会让 hasLora() 把每行标 "missing"——
            // 全网误报。
            if (!j.error && Array.isArray(j.loras)) _listCache = j.loras;
        } catch {
            // 瞬时失败：保留旧列表（旧数据胜过空——把它清 null 曾经在每次
            // 下拉打开强制重取后清掉完美的好缓存）。从未取到过的缓存保持
            // null，hasLora() 保持 "unknown"，不出现假 missing 标记。
        }
        // 只在槽位仍是我们的时清它——强制调用可能在我们飞行时替换了它。
        if (_listPromise === p) _listPromise = null;
        return _listCache || [];
    })();
    _listPromise = p;
    return p;
}

// invalidateInfo 会 bump 它。当回复在它的 LoRA 变化时仍在途中，绝不可缓存：
// 它描述变化之前的世界，而缓存比面板活得久，会整场会话持续服务旧答案。
const _infoGen = new Map();
const genOf = (name) => _infoGen.get(name) || 0;

export async function loraInfo(name, force = false) {
    if (!name) return { ok: false, message: "No LoRA selected." };
    if (!force && _infoCache.has(name)) return _infoCache.get(name);
    // 并发非强制请求去重（两个节点同一 LoRA），共享一个响应而不是竞写缓存。
    if (!force && _infoPromise.has(name)) return _infoPromise.get(name);
    const gen = genOf(name);
    // 在闭包使用它的初始化器之前声明。下面的 finally 读 `p`，若用
    // `const p = (async () => {...})()` 且 try 内同步抛错（未配对代理字符到
    // encodeURIComponent，比如），会撞上 temporal dead zone，finally 抛
    // ReferenceError 而非清槽，之后对该 LoRA 的每次调用都拿到同一 rejected
    // promise 直到永远。
    let p;
    p = (async () => {
        try {
            const r = await fetch(sfApiUrl("/api/sfnodes/lora_info?name=" + encodeURIComponent(name)),
                { cache: "no-store" });
            const j = await r.json();
            // 只缓存成功，且只在该 LoRA 未被 invalidate 时。服务端失败
            // （{ok:false}）曾像命中一样被缓存，短暂不可解析的 LoRA 的错误
            // 在原因消失后仍显示整个会话。
            if (j && j.ok && genOf(name) === gen) _infoCache.set(name, j);
            // 调用者需要知道它的答案过时了，否则会画出来。
            if (genOf(name) !== gen && j && typeof j === "object") j.stale = true;
            return j;
        } catch (e) {
            return { ok: false, message: "Could not reach the server." }; // 不缓存 -> 下次重试
        } finally {
            // 只清我们拥有的槽：强制调用落地不能丢掉并发的非强制调用的去重项。
            if (_infoPromise.get(name) === p) _infoPromise.delete(name);
        }
    })();
    if (!force) _infoPromise.set(name, p);
    return p;
}

// 丢掉缓存 info（Civitai 查询或预览变化重写了它描述的东西后）。bump 世代号
// 正是阻止在途回复把变化前的答案稍后塞回缓存的东西。
export function invalidateInfo(name) {
    _infoCache.delete(name);
    _infoGen.set(name, genOf(name) + 1);
}

// 刷新期失效器（挂 ComfyUI 的 R 键，见主扩展）。文件改名/增删的瞬间列表
// 缓存就过期；服务端永远新鲜（folder_paths 按目录 mtime 重新校验），所以
// 丢我们的副本就是全部所需。
export function invalidateList() {
    _listCache = null;
}
export function invalidateAllInfo() {
    _infoCache.clear();
    // 每个已知世代号也 bump，不只是缓存。只清缓存让 R 键可被击败：一个在
    // 途中、世代号未变的回复仍会把刷新前的答案塞回缓存——正是计数器要封的
    // 洞。每个消费者在旧答案上都会再问一次，所以这安全。
    for (const k of [..._infoGen.keys()]) _infoGen.set(k, _infoGen.get(k) + 1);
    // 从未 invalidate 过的名字没有条目，而它的在途 fetch 捕获了世代 0——
    // 给它一个，否则那条回复仍能溜进来。
    for (const k of [..._infoPromise.keys()]) if (!_infoGen.has(k)) _infoGen.set(k, 1);
}

// 名字在最近一次列表中吗？null = 列表未取（unknown）——调用方可避免首次
// 加载前的假 "missing" 标记。
export function hasLora(name) {
    return _listCache ? _listCache.includes(name) : null;
}

// 最近一次列表，未取时 null。同步读取给无法 await 的调用方；null 意为
// "unknown" 而非 "无 LoRA"；调用方可 kick listLoras() 预热（每个节点 setup
// 时都预热，实际使用中早就暖了）。
export function cachedLoras() {
    return _listCache;
}

// `bust`（时间戳或计数器）强制越过浏览器图片缓存——缩略图路由发
// max-age=3600 且 URL 永不变化，Civitai 查询替换预览后旧图会显示一小时。
export function thumbUrl(name, bust) {
    return sfApiUrl("/api/sfnodes/lora_thumb?name=" + encodeURIComponent(name) +
        (bust ? "&t=" + bust : ""));
}

// `overwrite` = 该 LoRA 已有用户自定义预览时仍用 Civitai 图片覆盖
// （前端已确认）。缺省不传 = 后端跳过保存、返回 thumb_skipped。
export async function civitaiLookup(name, overwrite) {
    try {
        const q = overwrite ? "&overwrite=1" : "";
        const r = await fetch(sfApiUrl("/api/sfnodes/lora/civitai?name=" + encodeURIComponent(name) + q));
        return await r.json();
    } catch {
        return { ok: false, reason: "offline", message: "Could not reach Civitai." };
    }
}

// ── Civitai 账户（可选 API key + 两个查询偏好）──────────────────────────────
// key 绝不发到页面。这两个调用只携带 {configured, hint, host, adultThumbs}；
// key 本体活在服务器读的文件里。刻意不缓存：设置面板打开时才读（罕见），
// 缓存一份"未配置"会让刚设置过 key 的用户看起来像保存失败。

export async function getCivitaiAccount() {
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/civitai/account"), { cache: "no-store" });
        return await r.json();
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

/** Patch {key, host, adultThumbs} 任意项。省略字段即不动；key:"" 移除 key。
 *  以服务器存储的状态应答，面板按服务器实际收下的重绘。 */
export async function setCivitaiAccount(patch) {
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/civitai/account"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch || {}),
        });
        return await r.json();
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

// 保存这个 LoRA 文件的用户自定义触发词（ComfyUI user 目录单一存储，按 LoRA
// 名键控）。它们曾只活在行上，所以行换 LoRA 再换回来就丢，别的节点也看不到。
// 发送空数组移除该 LoRA 的条目。
export async function saveCustomTriggers(name, words) {
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/lora/custom_triggers"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, words }),
        });
        const j = await r.json();
        // 面板从缓存 info 读自定义词，陈旧缓存会在下次打开时撤销保存。
        if (j?.ok) invalidateInfo(name);
        return j;
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

// 删除保存的 Civitai 侧车（<base>.civitai.info），info 回到文件自己的词。
// 调用方随后应 invalidateInfo(name)。
export async function deleteCivitai(name) {
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/lora/civitai_delete"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        return await r.json();
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

// ── 用户自己的预览图 ───────────────────────────────────────────────────────
// 存 ComfyUI user 目录，按 LoRA 名键控，胜过 .safetensors 旁的图和实时
// Civitai 缩略图。两个调用都 invalidate 缓存 info（携带 custom_preview /
// preview_v）——陈旧的那份会让面板在图片已消失时还提供删除，或从浏览器
// 一小时图片缓存里显示旧图。

/** 保存一张图作为该 LoRA 的预览。`dataUrl` 是面板降采样后的 jpeg（服务端
 *  仍查大小与 magic bytes）。 */
export async function saveLoraPreview(name, dataUrl) {
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/lora/preview"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, dataUrl }),
        });
        const j = await r.json();
        if (j?.ok) invalidateInfo(name);
        return j;
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}

/** 移除它，自动图回来。 */
export async function deleteLoraPreview(name) {
    try {
        const r = await fetch(sfApiUrl("/api/sfnodes/lora/preview_delete"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        const j = await r.json();
        if (j?.ok) invalidateInfo(name);
        return j;
    } catch {
        return { ok: false, message: "Could not reach the server." };
    }
}
