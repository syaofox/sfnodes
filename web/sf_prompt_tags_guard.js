// ==========================================================================
// sf_prompt_tags_guard.js - 全屏编辑器的 graph-undo 守卫
// ==========================================================================
//
// 编辑器打开期间，Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z 必须不能撤销画布上的工作流。
// ComfyUI 的 ChangeTracker 在页面启动时注册了 window+capture 的 keydown，
// 我们后注册的监听永远无法抢先（实测 preventDefault + stopImmediatePropagation
// 都拦不住）；它唯一承认的"全屏编辑器开着"信号是构造器上的静态槽
// `maskeditor_is_opended`（返回 true 则撤销链整条跳过）。填上它就是官方 API，
// 不是补丁。
//
// 要求（每一条都来自真实 bug）：
//   1. 自愈：overlay 被异常拆除（中途关工作流标签页）不能让我们永久武装
//   2. 引用计数：两个编辑器可同时打开，关一个不能扒掉另一个的保护
//   3. 单一系统：全项目共享一份实现，多份拷贝会互相打架
//   4. 只在我们仍持有槽时归还，绝不覆盖后来者
//
// 用法：
//   import { installGraphUndoGuard } from "./sf_prompt_tags_guard.js";
//   this._undoGuardOff = installGraphUndoGuard(() => !!this.overlay?.isConnected);
//   // close() 与 onRemoved 都要调用：  this._undoGuardOff?.(); this._undoGuardOff = null;
//
// ==========================================================================

const HOOK = "maskeditor_is_opended";

const _tokens = new Set(); // { isAlive: () => boolean }
let _installed = false;
let _prevHook = null; // 我们之前槽里的东西（通常为 null）
let _ourHook = null; // 我们自己的函数，用于证明槽还是我们的
let _fallbackOff = null; // 仅当槽完全不存在时使用

function _anyAlive() {
    for (const t of _tokens) {
        try {
            if (t.isAlive()) return true;
        } catch {
            /* 抛错的 isAlive 视为死亡 */
        }
    }
    return false;
}

function _release() {
    const C = window.app?.constructor;
    // 要求 4：只有槽仍装着我们自己的函数才归还
    if (C && _ourHook && C[HOOK] === _ourHook) C[HOOK] = _prevHook;
    _ourHook = null;
    _prevHook = null;
    try {
        _fallbackOff?.();
    } catch {
        /* teardown 永不抛错 */
    }
    _fallbackOff = null;
    _installed = false;
    _tokens.clear();
}

// 未来前端若移除旧槽的兜底：拦截 changeTracker 自己的 undoRedo 入口
// （返回 true = "已处理"，撤销/重做被吞掉，其余不变）
function _installFallback() {
    const ct = window.app?.extensionManager?.workflow?.activeWorkflow?.changeTracker;
    const proto = ct && Object.getPrototypeOf(ct);
    if (!proto || typeof proto.undoRedo !== "function") return null;
    const orig = proto.undoRedo;
    const patched = function (...args) {
        if (_anyAlive()) return true;
        if (proto.undoRedo === patched) proto.undoRedo = orig; // 自愈
        return orig.apply(this, args);
    };
    proto.undoRedo = patched;
    return () => {
        if (proto.undoRedo === patched) proto.undoRedo = orig;
    };
}

/**
 * 向共享守卫注册一个全屏编辑器。
 * @param {() => boolean} isAlive 编辑器 overlay 仍在 DOM 中时返回 true
 * @returns {() => void} 卸载函数——从编辑器的 close/cleanup 和 onRemoved 调用
 */
export function installGraphUndoGuard(isAlive) {
    const app = window.app;
    if (!app || !app.constructor) return () => {};

    const token = { isAlive };
    _tokens.add(token);

    if (!_installed) {
        _installed = true;
        const C = app.constructor;

        if (HOOK in C) {
            _prevHook = C[HOOK] ?? null;
            _ourHook = function () {
                if (_anyAlive()) return true;
                // 要求 1：所有 overlay 都没了且没人调卸载——立即放下武器，
                // 然后按前任所有者的方式回答
                const prev = _prevHook;
                _release();
                try {
                    return prev ? !!prev() : false;
                } catch {
                    return false;
                }
            };
            C[HOOK] = _ourHook;
        } else {
            _fallbackOff = _installFallback();
        }
    }

    // 卸载本编辑器。仅在最后一个编辑器离开时放下守卫。幂等。
    return function uninstall() {
        _tokens.delete(token);
        if (_anyAlive()) return; // 还有别的编辑器开着——保持守卫
        _release();
    };
}
