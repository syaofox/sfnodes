// ============================================================
// sf_common.js — 复刻节点共享的公共小工具
// 原 pixaroma js/shared/ 的内联副本去重（sf_crop / sf_inpaint /
// sf_load_image 曾各持一份，现已收敛到此处，避免复制后语义分叉）。
// 纯工具模块（无扩展行为），由使用者 import（同 sf_dynamic_slots.js 惯例）。
// 注意：本文件顶层自动安装 loadGraphData 守卫（幂等），任何模块 import 一次
// 即可全局生效。
// ============================================================
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// ── URL ──────────────────────────────────────────────────────────────
// 绝对安全的 URL：api.apiURL 处理托管部署基址，失败降级原样返回
export function sfApiUrl(route) {
  try {
    if (typeof api?.apiURL === "function") return api.apiURL(route);
  } catch {
    /* 降级 */
  }
  return route;
}

// ── 全局强调色（sfnodes.Accent 设置）──────────────────────────────────
// 供所有具备 accent 能力的 sf 节点统一读取（SFLoraStack 等）。优先级链在
// 各节点自行决定（如 node.accent > 节点默认 > 此处全局 > 品牌色）。
// 惰性读 ComfyUI 设置，异常/未配置返回 null（调用方回退）。空字符串视为
// 未配置（combo 的 "Default" 选项存空值）。
export function getSfAccent() {
  try {
    const v = globalThis.app?.ui?.settings?.getSettingValue?.("sfnodes.Accent");
    if (typeof v === "string" && v.trim()) return v;
  } catch {
    /* 降级 */
  }
  return null;
}

// 把全局强调色写到 <html> 的 inline CSS 变量 --sf-acc。所有 sf 节点主题色
// 统一走 var(--sf-acc, #f66744)：CSS 部分响应式自动生效，JS/canvas 部分经
// sfAccent() 读取。`value` 优先（设置 onChange 回调传入的新值——该回调在
// store 更新前触发，此时读设置会拿到旧值）；缺省回退读设置。
export function applySfAccentVar(value) {
  try {
    const v = value || getSfAccent();
    document.documentElement.style.setProperty("--sf-acc", v || "#f66744");
  } catch {
    /* 降级 */
  }
}

// 运行时读当前全局强调色（inline 变量，轻量、每帧调用无压力；未应用时
// 回退设置直读，再回退品牌橙）。canvas 绘制/动态 DOM 注入用这个。
export function sfAccent() {
  try {
    const v = document.documentElement.style.getPropertyValue("--sf-acc");
    if (v && v.trim()) return v.trim();
  } catch {
    /* 降级 */
  }
  return getSfAccent() || "#f66744";
}

// ── LoRA 名称显示（全局设置 sfnodes.Lora.DisplayName）────────────────────
// 供 SFLoraStack / SFLoraPlot 统一读取（单一真源，禁止各节点内联副本）。五档：
// full 完整相对路径（默认）/ filename 文件名含扩展名 / basename 文件名
// 去扩展名 / folder 所在文件夹名（根目录文件降级文件名）/ parent_basename
// 上级目录名 + 文件名去扩展名（根目录文件降级 basename）。
export const LORA_DISPLAY_MODES = {
  FULL: "full",
  FILENAME: "filename",
  BASENAME: "basename",
  FOLDER: "folder",
  PARENT_BASENAME: "parent_basename",
};
export const LORA_DISPLAY_SETTING = "sfnodes.Lora.DisplayName";

// 单一路径 → 显示名（纯函数，不读设置；mode 由调用方传入）
export function loraDisplayName(path, mode) {
  if (!path || path === "None") return path || "None";
  const parts = String(path).split(/[\\/]/);
  const file = parts.pop() || path;
  switch (mode) {
    case LORA_DISPLAY_MODES.FILENAME:
      return file;
    case LORA_DISPLAY_MODES.BASENAME: {
      const i = file.lastIndexOf(".");
      return i > 0 ? file.slice(0, i) : file; // 点开头（.hidden）或无扩展名原样
    }
    case LORA_DISPLAY_MODES.FOLDER:
      // 根目录文件（无文件夹）降级显示文件名
      return parts.length ? parts[parts.length - 1] : file;
    case LORA_DISPLAY_MODES.PARENT_BASENAME: {
      // 上级目录名 + 文件名去扩展名；根目录文件降级仅显示 basename
      const base = loraDisplayName(file, LORA_DISPLAY_MODES.BASENAME);
      return parts.length ? `${parts[parts.length - 1]}/${base}` : base;
    }
    default:
      return path;
  }
}

// 每渲染直读（getSettingValue 是轻量 map 查找）；未设置/异常回退 full
export function getLoraDisplayMode() {
  try {
    return (
      app.ui?.settings?.getSettingValue?.(LORA_DISPLAY_SETTING) || LORA_DISPLAY_MODES.FULL
    );
  } catch {
    return LORA_DISPLAY_MODES.FULL;
  }
}

// 只剥尾部已知模型扩展名（白名单，不是"最后一个点后的一切"），版本化名字
// 如 "MoXin_v1.0" 保留 ".0"。仅用于显示——行 title 保留真实文件名。
const LORA_EXT_RE = /\.(safetensors|safetensor|ckpt|pt|pth|bin|sft)$/i;
function loraBaseName(name) {
  if (!name) return "";
  const i = name.replace(/\\/g, "/").lastIndexOf("/");
  return i < 0 ? name : name.slice(i + 1);
}

// SFLoraStack / SFLoraPlot 行名统一入口：全局模式 ≠ full 时设置语义优先
// （basename 用 lastIndexOf 剥任意扩展名——"xyz.v1.0" → "xyz.v1"）；full
// （默认）回退每节点 hideExt 语义（白名单剥模型扩展名）——默认行为与
// 旧版逐字节一致。hideExt 仅在 full 模式参与。
export function loraRowLabel(name, hideExt) {
  const mode = getLoraDisplayMode();
  if (mode && mode !== LORA_DISPLAY_MODES.FULL) return loraDisplayName(name, mode);
  const b = loraBaseName(name);
  return hideExt ? b.replace(LORA_EXT_RE, "") : b;
}

// ── 渲染器检测（Vue 2.0 vs Classic）──────────────────────────────────
export function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}
export function applyAdaptiveCanvasOnly(widget) {
  if (!widget || !widget.options) return widget;
  try {
    Object.defineProperty(widget.options, "canvasOnly", {
      configurable: true,
      enumerable: true,
      get() {
        return !window.LiteGraph?.vueNodesMode;
      },
    });
  } catch (e) {
    widget.options.canvasOnly = !window.LiteGraph?.vueNodesMode;
  }
  return widget;
}

// ── 工作流加载守卫（全局单例，幂等）──────────────────────────────────
// 包装 app.loadGraphData 一次，加载 + 300ms 尾窗内 isGraphLoading() 为 true。
// 加载路径上的序列化状态变更必须被此守卫门控（连接恢复发生在 onConfigure
// 之后，无此守卫打开工作流会误断已保存的线）。
let _sfGraphLoading = false;
let _sfGraphLoadWrapped = false;
export function installGraphLoadingGuard() {
  if (_sfGraphLoadWrapped || !app || typeof app.loadGraphData !== "function") return;
  _sfGraphLoadWrapped = true;
  const _origLoadGraphData = app.loadGraphData.bind(app);
  app.loadGraphData = function (...args) {
    _sfGraphLoading = true;
    let r;
    try {
      r = _origLoadGraphData(...args);
    } finally {
      Promise.resolve(r).finally(() => setTimeout(() => { _sfGraphLoading = false; }, 300));
    }
    return r;
  };
}
export function isGraphLoading() {
  return _sfGraphLoading;
}
installGraphLoadingGuard();

// ── 滚轮缩放透传（仅 Classic 渲染器；增强版：滚动容器穿透检测）───────
// 根内部有可滚动区域（面板列表、编辑器等）时让容器滚动而非缩放画布；
// 无滚动容器时行为等价于直接转发（简单版），是后者的功能超集。
export function installCanvasZoomPassthrough(root) {
  if (!root || typeof root.addEventListener !== "function") return () => {};
  const onWheel = (e) => {
    if (isVueNodes()) return;
    let el = e.target;
    const vertical = Math.abs(e.deltaY) >= Math.abs(e.deltaX);
    let scrollable = false;
    while (el && el !== root.parentElement) {
      if (el.nodeType === 1) {
        const cs = getComputedStyle(el);
        if (vertical) {
          const oy = cs.overflowY;
          if ((oy === "auto" || oy === "scroll") && el.scrollHeight > el.clientHeight + 1) {
            const atTop = el.scrollTop <= 0;
            const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
            if ((e.deltaY < 0 && !atTop) || (e.deltaY > 0 && !atBottom)) { scrollable = true; break; }
          }
        } else {
          const ox = cs.overflowX;
          if ((ox === "auto" || ox === "scroll") && el.scrollWidth > el.clientWidth + 1) {
            const atLeft = el.scrollLeft <= 0;
            const atRight = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1;
            if ((e.deltaX < 0 && !atLeft) || (e.deltaX > 0 && !atRight)) { scrollable = true; break; }
          }
        }
      }
      el = el.parentElement;
    }
    if (scrollable) return;
    const canvasEl = app?.canvas?.canvas;
    if (!canvasEl) return;
    e.preventDefault();
    e.stopPropagation();
    const { clientX, clientY, deltaX, deltaY, deltaMode, ctrlKey, metaKey, shiftKey } = e;
    canvasEl.dispatchEvent(new WheelEvent("wheel", {
      clientX, clientY, deltaX, deltaY, deltaMode,
      ctrlKey, metaKey, shiftKey, bubbles: true, cancelable: true,
    }));
  };
  root.addEventListener("wheel", onWheel, { passive: false });
  return () => root.removeEventListener("wheel", onWheel);
}

// ── 输入框滚轮透传（画布缩放/滚动）───────────────────────────────────────
// sf 的 DOM widget 输入框（textarea/input）挂载在 canvas 的 DOM 层，不在
// Vue 新版 TransformPane 的 @wheel.capture 转发路径内——ComfyUI 的画布缩放
// 在编辑框上完全失效（连 Ctrl+滚轮都不缩放）。这个工具把 ComfyUI 原生输入框
// 的滚轮行为搬到 sf 输入框上（与 installCanvasZoomPassthrough 同思路，但挂在
// 输入框元素、且不因 isVueNodes() 早退）：
//   Ctrl/⌘+滚轮 → 总是转发 canvas 缩放
//   普通滚轮     → 输入框可滚动（scrollHeight>clientHeight）时滚动文本；
//                  否则转发 canvas 缩放
export function installWheelZoomPassthrough(el) {
  if (!el || typeof el.addEventListener !== "function") return () => {};
  const onWheel = (e) => {
    const canvasEl = app?.canvas?.canvas;
    if (!canvasEl) return;
    const isGesture = e.ctrlKey || e.metaKey;
    if (!isGesture) {
      // 输入框自身可滚动 → 滚动文本（与 ComfyUI 原生输入框一致）
      if (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1) return;
    }
    e.preventDefault();
    e.stopPropagation();
    canvasEl.dispatchEvent(new WheelEvent("wheel", {
      clientX: e.clientX, clientY: e.clientY,
      deltaX: e.deltaX, deltaY: e.deltaY,
      ctrlKey: e.ctrlKey, metaKey: e.metaKey, shiftKey: e.shiftKey,
      bubbles: true, cancelable: true,
    }));
  };
  el.addEventListener("wheel", onWheel, { passive: false });
  return () => el.removeEventListener("wheel", onWheel);
}

// ── 通用 DOM 工具（HTML 转义 / 下载 / 剪贴板）───────────────────────────
// 收敛自各节点内联副本（sf_crop_framework.downloadDataURL、
// sf_workflows_ui.copyText、sf_lora_stack_info.escapeHtml——复制后语义分叉
// 是 bug 温床）。注意：本文件依赖 /scripts/app.js，纯逻辑模块（*_lib.js /
// *_core.js / sf_markdown.js）不得 import 本文件（会破坏其 Node 测试拷贝
// 能力）；sf_find_replace_lib.js 与 sf_markdown.js 因纯模块独立性与测试
// 锁定保留各自的本地转义实现（转义集合更小）。
// HTML 转义（五字符全集，DOM innerHTML 注入用）。转义集合刻意大于
// find_replace/markdown 的本地版（& < >）：引号一并转义对属性上下文安全。
export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// 下载 dataURL 为文件：showSaveFilePicker 优先（AbortError 豁免），
// <a download> 回退。扩展名按 MIME 推导（jpeg→jpg，其余 png）。
export async function downloadDataURL(dataURL, suggestedName = "sf_crop.png") {
  if (!dataURL) return;
  const mimeMatch = dataURL.match(/^data:([^;]+);/);
  const mime = mimeMatch ? mimeMatch[1] : "image/png";
  const ext = mime === "image/jpeg" ? "jpg" : "png";
  const name = suggestedName.endsWith(`.${ext}`)
    ? suggestedName
    : `${suggestedName}.${ext}`;
  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: name,
        types: [{ description: "Image", accept: { [mime]: [`.${ext}`] } }],
      });
      const writable = await handle.createWritable();
      const blob = await (await fetch(dataURL)).blob();
      await writable.write(blob);
      await writable.close();
      return;
    } catch (e) { if (e?.name !== "AbortError") console.warn("[sfnodes] save picker failed:", e); }
  }
  const a = document.createElement("a");
  a.href = dataURL;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => a.remove(), 1000);
}

// 复制文本到剪贴板（返回是否成功）：navigator.clipboard 优先（LAN 明文
// http 无安全上下文时不可用），execCommand 回退。
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch { /* 无安全上下文或权限被拒 */ }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;top:-1000px;left:-1000px;";
  document.body.append(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  ta.remove();
  return ok;
}

// ── 图片值解析（LoadImage widget 值 → {filename, subfolder, type}）────
// 支持 "subfolder/name.png [output]" 注解与反斜杠路径。
export function parseAnnotatedImageValue(value) {
  let v = String(value || "");
  let type = "input";
  const m = v.match(/\s*\[(input|output|temp)\]\s*$/i);
  if (m) { type = m[1].toLowerCase(); v = v.slice(0, m.index); }
  v = v.replace(/\\/g, "/").trim();
  const i = v.lastIndexOf("/");
  return {
    filename: i >= 0 ? v.slice(i + 1) : v,
    subfolder: i >= 0 ? v.slice(0, i) : "",
    type,
  };
}

// ── /view URL 构建 ────────────────────────────────────────────────────
// Build a /view URL from a {filename, subfolder, type} record. Adds a fresh
// cache-buster timestamp at runtime; persisted records (in node.properties)
// store only the structural parts so workflow JSON stays clean.
export function buildSourceURL(part, withCacheBust) {
  if (!part || !part.filename) return null;
  // The cache-buster is part of the ROUTE handed to sfApiUrl, never appended to
  // its RESULT: a hosted ComfyUI adds its auth token to the finished url, so
  // concatenating afterwards writes our param on the far side of that token
  // (see js/shared/api_url.mjs). Locally the two produce the identical string.
  return sfApiUrl(`/view?filename=${encodeURIComponent(part.filename)}` +
              `&subfolder=${encodeURIComponent(part.subfolder || "")}` +
              `&type=${encodeURIComponent(part.type || "temp")}` +
              (withCacheBust ? `&t=${Date.now()}` : ""));
}

// ── 上游图片 URL 解析 ─────────────────────────────────────────────────
// 优先活着的接线源（刚换过的 Load Image / 实时预览），cachedUrl 是执行期
// 缓存（生成型上游上次运行保存的 temp PNG / 粘贴拖放的磁盘源）。不按这个
// 顺序，换 Load Image 文件后显示的是上次运行的旧图直到重跑。
export function getUpstreamImageURL(node, cachedUrl) {
  const input = (node.inputs || []).find((i) => i.name === "image");
  const graph = node.graph;
  if (input && input.link != null && graph) {
    // Vue Compat #3: graph.links can be a Map in newer ComfyUI versions.
    let link = graph.links?.[input.link];
    if (!link && typeof graph.links?.get === "function") link = graph.links.get(input.link);
    const src = link && graph.getNodeById(link.origin_id);
    if (src) {
      if (src.comfyClass === "LoadImage" || src.type === "LoadImage") {
        const w = (src.widgets || []).find((x) => x.name === "image");
        if (w && w.value) return buildSourceURL(parseAnnotatedImageValue(w.value), true);
      }
      if (src.imgs && src.imgs.length > 0) {
        const img = src.imgs[link.origin_slot] || src.imgs[0];
        if (typeof img === "string") return img;
        if (img && img.src) return img.src;
      }
    }
  }
  return cachedUrl || null;
}

// ── 全局粘贴 handler（剪贴板图片 → 选中的节点）────────────────────────
// Capture phase + stopImmediatePropagation 不能完全压住 ComfyUI 自己的
// paste handler（它注册更早），因此额外快照图节点 id，之后删掉 ComfyUI 从
// 同一 paste 事件自动创建的 LoadImage（"pasted/" 文件名）。
// 参数化：comfyClass 匹配活动节点；onPasteImage(node, dataURL) 执行上传；
// allowPaste(node) 返回 false 时跳过（如 inpaint 编辑器打开时）。
// 单例键是 comfyClass:hook 而非全局一次：SFImageCrop 与 SFInpaintCrop 等
// 多个类各自要一个监听器（去重为单一布尔曾让后注册的类粘贴静默失效——
// 同节点类多实例的重复安装仍幂等）。
const _pasteHandlerKeys = new Set();
export function installPasteHandler({ comfyClass, hook, onPasteImage, allowPaste }) {
  const key = `${comfyClass}:${hook || ""}`;
  if (_pasteHandlerKeys.has(key)) return;
  _pasteHandlerKeys.add(key);
  window.addEventListener("paste", async (e) => {
    // Don't steal paste from form fields (panel inputs, editor inputs, etc.)
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;

    const node = findActiveNode(comfyClass, hook);
    if (!node) return;
    if (allowPaste && !allowPaste(node)) return;

    const items = e.clipboardData?.items || [];
    const imageItem = Array.from(items).find((it) => it.type?.startsWith("image/"));
    if (!imageItem) return;

    e.preventDefault();
    e.stopImmediatePropagation();

    // If upstream wire is connected, disconnect it — pasting an image is an
    // unambiguous "use this image now" override. Without this, Python would
    // keep using the upstream tensor and the paste would have no effect on
    // workflow output.
    const imgInputIdx = (node.inputs || []).findIndex((i) => i.name === "image");
    if (imgInputIdx >= 0 && node.inputs[imgInputIdx].link != null) {
      try { node.disconnectInput(imgInputIdx); } catch {}
    }

    // Snapshot existing graph node IDs so we can remove any LoadImage that
    // ComfyUI auto-creates from this same paste event.
    const idsBefore = new Set((app.graph?._nodes || []).map((n) => n.id));

    const blob = imageItem.getAsFile();
    if (!blob) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      onPasteImage(node, ev.target.result);
    };
    reader.readAsDataURL(blob);

    // Schedule a sweep for the auto-created LoadImage node (if any).
    // 50ms is enough for ComfyUI's createNode + widget setup to settle.
    setTimeout(() => {
      const after = app.graph?._nodes || [];
      for (const n of after) {
        if (idsBefore.has(n.id)) continue;
        if (n.comfyClass !== "LoadImage" && n.type !== "LoadImage") continue;
        const w = (n.widgets || []).find((x) => x.name === "image");
        const v = w?.value;
        if (typeof v === "string" && v.startsWith("pasted/")) {
          try { app.graph.remove(n); } catch {}
        }
      }
    }, 50);
  }, true); // capture phase
}

// Find the "active" node of a class from any of the selection sources
// ComfyUI might use across versions/frontends:
//   1. app.canvas.selected_nodes  (object/array/map of selected nodes)
//   2. app.canvas.current_node    (LiteGraph's last-clicked node)
//   3. node_over                  (hovered node)
//   4. Iterate all nodes and pick one with `.is_selected` (Vue may set this)
// `hook` is the node's paste method name; matching requires it to be present
// (defensive: nodeCreated may not have finished for a just-created node).
// Returns the first matching node, or null.
function findActiveNode(comfyClass, hook) {
  const c = app.canvas;
  if (!c) return null;
  const isTarget = (n) =>
    n && n.comfyClass === comfyClass && (!hook || typeof n[hook] === "function");

  // 1. selected_nodes — try Object.values, Array, and Map .values()
  const sel = c.selected_nodes;
  if (sel) {
    let iter = null;
    if (Array.isArray(sel)) iter = sel;
    else if (typeof sel.values === "function") iter = Array.from(sel.values());
    else if (typeof sel === "object") iter = Object.values(sel);
    if (iter) {
      const hit = iter.find(isTarget);
      if (hit) return hit;
    }
  }

  // 2. current_node (LiteGraph internal)
  if (isTarget(c.current_node)) return c.current_node;

  // 3. node_over (hovered)
  if (isTarget(c.node_over)) return c.node_over;

  // 4. Fallback — scan all nodes for an is_selected flag
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    if (isTarget(n) && (n.is_selected || n.flags?.is_selected)) return n;
  }
  return null;
}
