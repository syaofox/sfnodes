// ==========================================================================
// sf_memory_menu.js — 内存清理菜单项构建器
// 供聚合菜单（web/sf_canvas_menu.js，📦 SF Menu ▶ SF Memory）提供子菜单。
// VRAM 走 ComfyUI 原生 POST /free（server.py:1192，队列 flag 有序执行卸载 +
// soft_empty_cache，与官方释放语义一致，零后端改动）；RAM 走自建路由
// POST /api/sfnodes/memory/ram（复用 nodes/utils/memory_cleanup.py 的
// RAMCleanup 逻辑——浏览器 JS 无法释放服务端进程内存，必须经后端执行）。
// 反馈经 sf_common.sfToast（禁止内联副本）。
// 本文件不再自行注册画布菜单（分散入口已收敛到聚合器），只 export 构建器。
// ==========================================================================

import { sfApiUrl, sfToast } from "./sf_common.js";

const TAG = "SF Memory";

async function freeVram() {
    try {
        const res = await fetch(sfApiUrl("/free"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ unload_models: true, free_memory: true }),
        });
        if (!res || !res.ok) throw new Error(`HTTP ${res && res.status}`);
        sfToast({ summary: TAG, detail: "VRAM 已释放（模型已卸载、缓存已清空）", severity: "success", fallbackTag: TAG });
    } catch (err) {
        sfToast({ summary: TAG, detail: `VRAM 释放失败：${(err && err.message) || err}`, severity: "error", life: 6000, fallbackTag: TAG });
    }
}

async function freeRam() {
    try {
        const res = await fetch(sfApiUrl("/api/sfnodes/memory/ram"), { method: "POST" });
        const data = await res.json();
        if (!data || !data.ok) throw new Error((data && data.message) || `HTTP ${res && res.status}`);
        sfToast({ summary: TAG, detail: `RAM 清理完成 [${data.before_usage}% → ${data.after_usage}%，释放 ${data.freed_mb}MB]`, severity: "success", fallbackTag: TAG });
    } catch (err) {
        sfToast({ summary: TAG, detail: `RAM 清理失败：${(err && err.message) || err}`, severity: "error", life: 6000, fallbackTag: TAG });
    }
}

export function buildMemoryMenuItem() {
    // sf_canvas_align 同款子菜单结构；内存清理无选中门槛，不做节点数守卫
    // （sf_lora_browser 同款无条件返回）。
    return {
        content: "SF Memory",
        has_submenu: true,
        submenu: {
            options: [
                { content: "Free VRAM", callback: freeVram },
                { content: "Free RAM", callback: freeRam },
            ],
        },
    };
}
