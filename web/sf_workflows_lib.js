// ==========================================================================
// sf_workflows_lib.js - SF Workflows 纯函数库（无 app/DOM 依赖）
// ==========================================================================
//
// 名称清洗、文件夹排序/层级、搜索评分——全部纯函数，供主扩展与 UI 模块
// import，也供 tests/ 复制为 .mjs 直接测试。数据与 DOM 都在别处。
//
// ==========================================================================

// ── 名称检查 ──────────────────────────────────────────────────────────────

/** 一个名字清理到可用，或 ""。 */
export const MAX_NAME = 120;

export function cleanName(raw) {
    return String(raw || "")
        .replace(/[\\/:*?"<>|]/g, "")
        // 控制字符。从终端或 PDF 粘贴会带来不可见的控制字符，\s 只覆盖
        // tab/换行/换页，其余会直接进磁盘成一个打不开的文件名。
        // 写成 \x 转义而非字面字节：字面版屏幕上看一样、diff 里不可见、
        // 审查里不可见，正则里的真实控制字节已经坑过这个项目一次。
        .replace(/[\x00-\x1F\x7F]/g, "")
        .replace(/^[.\s]+|[.\s]+$/g, "")
        .trim()
        // 远低于每个文件系统 ~255 字节上限，且给文件夹路径和 .json 留了
        // 空间。在这里截断意味着用户看到的是他们实际会得到的名字
        .slice(0, MAX_NAME)
        .trim();
}

// CON、NUL、COM1 等在任何扩展名下都是 Windows 的设备名，所以 "NUL" 和
// "NUL.json" 都会失败。服务器也会拒绝（它是权威），但这里先检查让回答
// 即时且为输入者措辞。
export const WIN_RESERVED = new Set([
    "CON", "PRN", "AUX", "NUL",
    ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
    ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
]);

/** 这个名字为什么不行（成句），或 null（没问题）。与 cleanName 分离，
 *  让每个理由都有自己的句子。 */
export function nameProblem(clean) {
    if (!clean) return "That name cannot be used.";
    if (WIN_RESERVED.has(clean.split(".")[0].trim().toUpperCase())) {
        return `"${clean}" is a name Windows keeps for itself. Pick another one.`;
    }
    return null;
}

// ── 文件夹顺序与层级 ──────────────────────────────────────────────────────

/**
 * 文件夹顺序。磁盘上没有顺序——服务器按字母列出。选定顺序以路径列表存在
 * sidecar 里，不在其中的回退到字母序，新文件夹仍出现在合理位置。
 * 按父排序、按树行走，子级永远跟随自己的父级。
 */
export function orderedFolders(folders, order) {
    const rank = new Map((order || []).map((p, i) => [p, i]));
    const kids = new Map();               // 父路径 -> 子路径
    for (const f of folders) {
        const parent = f.includes("/") ? f.slice(0, f.lastIndexOf("/")) : "";
        if (!kids.has(parent)) kids.set(parent, []);
        kids.get(parent).push(f);
    }
    const byRank = (a, b) => {
        const ra = rank.has(a) ? rank.get(a) : Infinity;
        const rb = rank.has(b) ? rank.get(b) : Infinity;
        if (ra !== rb) return ra - rb;
        return a.localeCompare(b, undefined, { sensitivity: "base" });
    };
    const out = [];
    const walk = (parent) => {
        const list = (kids.get(parent) || []).slice().sort(byRank);
        for (const f of list) { out.push(f); walk(f); }
    };
    walk("");
    // 父级缺失的文件夹永远不会被走到；追加而非从面板整个消失
    for (const f of folders) if (!out.includes(f)) out.push(f);
    return out;
}

/** 此路径之上的每个文件夹，由外到内。"a/b/c" -> ["a", "a/b"] */
export function ancestorsOf(path) {
    const parts = String(path || "").split("/");
    const out = [];
    for (let i = 1; i < parts.length; i++) out.push(parts.slice(0, i).join("/"));
    return out;
}

/** 是否有文件夹坐在这个里面？决定行是否带 twisty。 */
export function hasChildren(path, folders) {
    const prefix = path + "/";
    return (folders || []).some((f) => f.startsWith(prefix));
}

/**
 * 本次渲染哪些文件夹展开。存储列表是用户打开的——缺省即关闭，全新安装
 * 整洁而不是把每个子文件夹倒满一列。正在查看的文件夹的分支额外加入，
 * 且只对本渲染：选中子文件夹不得静默重写用户保存的选择。
 */
export function openSet(expanded, sel) {
    const open = new Set(expanded || []);
    if (sel && sel.kind === "folder" && sel.value) {
        for (const a of ancestorsOf(sel.value)) open.add(a);
    }
    return open;
}

/** 一个文件夹的同级，按显示顺序。 */
export function siblingsOf(path, folders, order) {
    const parent = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
    return orderedFolders(folders, order).filter((f) => {
        const p = f.includes("/") ? f.slice(0, f.lastIndexOf("/")) : "";
        return p === parent;
    });
}

const FOLDER_COLORS = ["#4d7ea8", "#7ea84d", "#a8794d", "#8a4da8", "#a84d4d", "#4da8a0", "#8f8f8f", "#6d78a8"];

/** 每文件夹的稳定颜色，看起来刻意、不随添加而洗牌。sidecar 可覆盖。 */
export function folderColor(path, meta) {
    const chosen = meta?.folderColors?.[path];
    if (chosen) return chosen;
    let h = 0;
    for (let i = 0; i < path.length; i++) h = (h * 31 + path.charCodeAt(i)) >>> 0;
    return FOLDER_COLORS[h % FOLDER_COLORS.length];
}

// ── 搜索评分 ──────────────────────────────────────────────────────────────

// 打分而非过滤：半记得名字的东西排前，只是用了那个模型的靠后。
// 匹配内容才是要点："哪个用了 flux krea" 和 "红大衣提示词的那个" 都是
// 文件名答不了的问题。
const FIELDS = [
    // [权重, 从条目怎么读]
    [100, (e) => e.name],
    [30, (e) => e.folder],
    [26, (e) => e._note || ""],
    [18, (e) => (e.models || []).join(" ")],
    [18, (e) => (e.loras || []).join(" ")],
    [8, (e) => (e.class_types || []).join(" ")],
    [6, (e) => e.text || ""],
];

/** 一个条目对已小写化的 terms 打分。0 表示"不匹配"。 */
function score(entry, terms) {
    let total = 0;
    for (const term of terms) {
        let best = 0;
        for (const [weight, read] of FIELDS) {
            const hay = (read(entry) || "").toLowerCase();
            if (!hay) continue;
            const at = hay.indexOf(term);
            if (at < 0) continue;
            // 精确 > 前缀 > 包含某处，输入全名不会让更长的包含名排前
            let s = weight;
            if (hay === term) s = weight * 3;
            else if (at === 0) s = weight * 2;
            else if (/\s|[-_/]/.test(hay[at - 1] || "")) s = Math.round(weight * 1.5);
            if (s > best) best = s;
        }
        // 每个 term 都要命中，"krea lora" 不能返回任何只是 lora 的东西
        if (!best) return 0;
        total += best;
    }
    return total;
}

/**
 * @param entries 索引条目，每个可选带 `_note`
 * @param query   用户输入
 * @returns 匹配条目，最优在前；query 为空时保持输入顺序
 */
export function searchEntries(entries, query) {
    const q = (query || "").trim().toLowerCase();
    if (!q) return entries;
    const terms = q.split(/\s+/).filter(Boolean);
    const hits = [];
    for (const e of entries) {
        const s = score(e, terms);
        if (s > 0) hits.push([s, e]);
    }
    // 平局按最近修改破：两个同优匹配里，上周还在做的是要找的那个
    hits.sort((a, b) => b[0] - a[0] || (b[1].modified || 0) - (a[1].modified || 0));
    return hits.map((h) => h[1]);
}
