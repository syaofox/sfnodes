// ==========================================================================
// SF LoRA 浏览器 - 纯函数（无 app/DOM 依赖，可直接拷 .mjs 用 Node 单测）
// 文件名拆分 / 搜索过滤 / 文件夹层级（立即子目录 + 当前层文件）/ 面包屑分段。
// ==========================================================================

/** "a/b/c.safetensors" -> { folder: "a/b", base: "c.safetensors" }；无子目录时 folder 为 ""。 */
export function splitName(name) {
    const norm = String(name || "").replace(/\\/g, "/");
    const i = norm.lastIndexOf("/");
    return i < 0
        ? { folder: "", base: norm }
        : { folder: norm.slice(0, i), base: norm.slice(i + 1) };
}

/** 大小写不敏感子串过滤。命中全名或文件名主体（去扩展名）任一即可。 */
export function filterLoras(list, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return (list || []).slice();
    return (list || []).filter((name) => {
        const n = String(name || "");
        if (n.toLowerCase().includes(q)) return true;
        const { base } = splitName(n);
        const stem = base.replace(/\.[^.]+$/, "").toLowerCase();
        return stem.includes(q);
    });
}

/** 按子文件夹分组（平铺式归档展示用）。根（""）在前，其余按文件夹名字典序。 */
export function groupLoras(list) {
    const out = [];
    const byFolder = new Map();
    for (const name of list || []) {
        const { folder } = splitName(name);
        let g = byFolder.get(folder);
        if (g === undefined) {
            g = { folder, items: [] };
            byFolder.set(folder, g);
            out.push(g);
        }
        g.items.push(name);
    }
    out.sort((a, b) => {
        if (!a.folder && !b.folder) return 0;
        if (!a.folder) return -1;
        if (!b.folder) return 1;
        return a.folder.localeCompare(b.folder);
    });
    return out;
}

/** 组内排序：按文件名主体（去扩展名）字典序，原地返回新数组。 */
export function sortWithinGroup(names) {
    return (names || []).slice().sort((a, b) => {
        const sa = splitName(a).base.replace(/\.[^.]+$/, "").toLowerCase();
        const sb = splitName(b).base.replace(/\.[^.]+$/, "").toLowerCase();
        return sa.localeCompare(sb);
    });
}

/** 当前文件夹层的展示内容（对齐 SF Load Image Browser 的目录下钻模型）：
 *  folders = 立即子目录名（只取第一段），files = 当前层的直接文件（全名）。
 *  folder 为 ""/undefined 表示根。尾部斜杠归一。返回值已排序：folders 字典
 *  序、files 按文件名主体字典序。 */
export function folderContents(list, folder) {
    const f = String(folder || "").replace(/\/+$/, "");
    const prefix = f ? f + "/" : "";
    const folders = new Set();
    const files = [];
    for (const item of list || []) {
        const n = String(item || "");
        if (!prefix || n.startsWith(prefix)) {
            const rest = n.slice(prefix.length);
            if (!rest) continue;
            if (rest.includes("/")) folders.add(rest.split("/")[0]);
            else if (n) files.push(n);
        }
    }
    return { folders: [...folders].sort(), files: sortWithinGroup(files) };
}

/** 面包屑分段："a/b" -> ["a","b"]；"" -> []。尾部斜杠归一。 */
export function breadcrumbParts(folder) {
    const f = String(folder || "").replace(/\/+$/, "");
    return f ? f.split("/") : [];
}
