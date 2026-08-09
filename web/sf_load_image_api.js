// SF Load Image Resize — API helpers (ported from comfyui-pixaroma js/load_image/api.mjs).

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { sfApiUrl } from "./sf_common.js";

// 通知 ComfyUI 图已变更（DOM 控制 click 提交晚于核心 mouseup 快照，若不
// 显式 capture，工作流不会被标记"已修改"，重新打开图片会回退）。
// 120ms 防抖合并连点；加载路径由调用方保证不会触发。
let _notifyTimer = null;
function notifyGraphChanged() {
  if (_notifyTimer) clearTimeout(_notifyTimer);
  _notifyTimer = setTimeout(() => {
    _notifyTimer = null;
    try {
      const ct = (
        app?.extensionManager?.workflow?.activeWorkflow?.changeTracker ||
        app?.workflowManager?.activeWorkflow?.changeTracker ||
        null
      );
      if (!ct) return;
      if (typeof ct.captureCanvasState === "function") ct.captureCanvasState();
      else if (typeof ct.checkState === "function") ct.checkState();
    } catch (e) {
      console.warn("[SFLoadImageResize] could not notify ComfyUI of a graph change", e);
    }
  }, 120);
}

// Split "Studio1/cat.png" into {subfolder:"Studio1", filename:"cat.png"}.
// ComfyUI's input/ folder can hold subfolders; the native image_upload combo
// values include the path-prefixed names (e.g. "Studio1/cat.png"). The /view
// endpoint expects subfolder + filename as SEPARATE query params - if we send
// the slash inside `filename=` and leave subfolder empty, the preview fetch
// silently 404s on some Comfy builds. Always split before building the URL.
export function splitFilenameSubfolder(path) {
  if (!path) return { subfolder: "", filename: "" };
  const norm = String(path).replace(/\\/g, "/");
  const idx = norm.lastIndexOf("/");
  if (idx < 0) return { subfolder: "", filename: norm };
  return { subfolder: norm.slice(0, idx), filename: norm.slice(idx + 1) };
}

/**
 * Split ComfyUI's type annotation off a combo value.
 *
 * A value can be `clipspace-painted-masked-123.png [input]` - MEASURED, that is
 * exactly what the widget holds after a Mask Editor save. The annotation names
 * the folder the file lives in, and it must NEVER reach the `filename` query
 * parameter: doing so produced `...png+[input]` in the URL, a guaranteed 404,
 * and because the preview then never matched the widget the reconcile refetched
 * on every setup forever. Found by masking a real image and reloading.
 */
export function splitTypeAnnotation(value) {
  const s = String(value ?? "");
  const m = s.match(/^(.*?)\s*\[(input|output|temp)\]\s*$/i);
  if (!m) return { name: s, type: "input" };
  return { name: m[1], type: m[2].toLowerCase() };
}

// Fetch the image from ComfyUI's /view route and assign it to node.imgs so
// the native bottom-of-node preview updates. ComfyUI populates node.imgs
// automatically on workflow load via the image_upload combo's setter, but
// when we set widget.value programmatically the setter does NOT fire - so
// without this helper the preview stays stuck on the previously-loaded file.
//
// Defensive race-condition fix (issue #38 family): rapid pick-A-then-B picks
// queue two concurrent fetches; img.onload fires in LOAD order, not call
// order, so a slow A landing after a fast B would clobber node.imgs back to A.
// Per-node monotonic request-id discards stale onloads.
/**
 * Is the picture currently in `node.imgs` actually the file `filename` names?
 *
 * Needed on the LOAD path. ComfyUI populates node.imgs from the image_upload
 * combo's setter, but a restored workflow does not always fire it, so a node
 * can end up holding the PREVIOUS workflow's picture while its widget, its
 * filename cache and its origName all correctly name the new one. Reported
 * 2026-08-05 after switching workflows: the picker read one file and the
 * preview showed another.
 *
 * This is not cosmetic. node.imgs feeds the INPUT size card (the node reported
 * 1024x1024 for an image that is 1376x768) and is what Mask Editor and
 * Clipspace read.
 *
 * Compares the `filename` query parameter rather than a substring of the whole
 * URL, so a name that merely appears inside a subfolder or a cache-busting
 * parameter cannot produce a false match.
 */
export function previewMatches(node, filename) {
  const img = node?.imgs?.[0];
  if (!img?.src || !filename) return false;
  let loaded = null;
  try {
    loaded = new URL(img.src, window.location.href).searchParams.get("filename");
  } catch {
    loaded = decodeURIComponent(img.src).match(/[?&]filename=([^&]*)/)?.[1] ?? null;
  }
  if (loaded == null) return false;
  // Normalise BOTH sides. Our own updateNativePreview strips the annotation
  // before building the URL, but ComfyUI's native image_upload setter does NOT:
  // on a workflow load it sets node.imgs with `filename=...png+[input]` (that
  // URL works, its /view parses the annotation server-side). MEASURED after a
  // Mask Editor save plus a reload. Stripping only the widget side meant the
  // two could never agree behind a core-populated preview, so the reconcile
  // refetched on every single setup - wasteful, and it made "match" a
  // permanently false signal.
  const bare = (v) => splitFilenameSubfolder(splitTypeAnnotation(v).name).filename;
  return bare(loaded) === bare(filename);
}

export function updateNativePreview(node, filename) {
  if (!filename) return;
  node._sfLiPreviewReqId = (node._sfLiPreviewReqId | 0) + 1;
  const myReq = node._sfLiPreviewReqId;
  // Peel the type annotation off BEFORE splitting, and use it as the /view
  // `type`. A Mask Editor save leaves the widget holding
  // "clipspace-painted-masked-123.png [input]"; without this the annotation
  // ended up inside filename= as "...png+[input]", which 404s.
  const { name: bare, type } = splitTypeAnnotation(filename);
  const { subfolder, filename: name } = splitFilenameSubfolder(bare);
  const img = new Image();
  img.onload = () => {
    if (node._sfLiPreviewReqId !== myReq) return; // stale, newer pick won
    node.imgs = [img];
    node.graph?.setDirtyCanvas?.(true, true);
    // Notify the index.js side that natural dims are now available, so
    // the input/output dims info bar can refresh. The hook is attached
    // by setupLoadImageNode and may be absent on stray calls.
    node._sfLiOnImageLoaded?.();
  };
  img.onerror = () => {
    if (node._sfLiPreviewReqId !== myReq) return;
    console.warn("[SFLoadImageResize] preview fetch failed for", filename);
  };
  img.src = sfApiUrl(`/view?filename=${encodeURIComponent(name)}&type=${encodeURIComponent(type)}&subfolder=${encodeURIComponent(subfolder)}&t=${Date.now()}`);
}

// Single source of truth for picking an image (dropdown click, arrow nav,
// upload, drag-drop, paste). Centralises:
//   - widget.value write
//   - per-node `_sfLiSelectedFilename` cache (defensive sync used by the
//     graphToPrompt hook, in case some Vue path resets widget.value back)
//   - native preview refresh (via updateNativePreview)
//   - dropdown label refresh (via the registered hook)
//   - dirty canvas
//   - telling ComfyUI's change tracker the workflow now differs from its file
// Call this instead of touching imageWidget.value directly in new code.
export function setSelectedImage(node, filename) {
  if (!filename) return;
  const w = node._sfLiImageWidget;
  if (!w) return;
  // Ensure the value exists in the combo's options - upload paths push first
  // then call this; arrow/dropdown paths already have it. Defensive only.
  if (!w.options) w.options = {};
  const values = w.options.values || (w.options.values = []);
  if (!values.includes(filename)) {
    values.push(filename);
    values.sort();
  }
  w.value = filename;
  node._sfLiSelectedFilename = filename;
  // Track the original (non-clipspace) name directly here too, not only via the
  // imageWidget.value setter — that setter is skipped when the widget's `value`
  // property is non-configurable, and every caller of this fn is a real pick
  // (dropdown / arrow / upload / paste), never a clipspace copy (issue #51).
  if (!/clipspace/i.test(filename)) node._sfLiOrigName = filename;
  updateNativePreview(node, filename);
  node._sfLiOnFilenameChanged?.(filename);
  node.graph?.setDirtyCanvas?.(true, true);
  // setDirtyCanvas is only a REDRAW flag - it tells the change tracker nothing.
  // Our pick commits on `click`, which is AFTER the `mouseup` that core
  // snapshots on, so without this the pick is never recorded: the workflow is
  // never marked modified, ComfyUI never offers to save it, and reopening
  // restores the file's original image. Safe here because every caller of this
  // function is a real user pick (dropdown / arrow / upload / paste / drop) -
  // there is no load-path caller - and the helper re-checks isGraphLoading().
  notifyGraphChanged();
}

// Upload an image File/Blob to ComfyUI's /upload/image route and update the
// node's `image` combo widget to select the new file.
//
// Returns a Promise<string> resolving to the saved filename (or rejecting on
// network/HTTP error).

export async function uploadImageToInput(node, file, filenameHint = null) {
  const form = new FormData();
  // ComfyUI's /upload/image accepts:
  //   image: the File/Blob
  //   subfolder: optional, defaults to ""
  //   overwrite: "true" / "false"
  //   type: "input" (default) or "temp"
  // When `file` is a Blob (paste path), we need to give it a name.
  if (file instanceof Blob && !(file instanceof File) && filenameHint) {
    form.append("image", file, filenameHint);
  } else if (file instanceof File && filenameHint) {
    // Rename to filenameHint
    form.append("image", new File([file], filenameHint, { type: file.type }));
  } else {
    form.append("image", file);
  }

  const resp = await fetch("/upload/image", { method: "POST", body: form });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Upload failed (${resp.status}): ${text || resp.statusText}`);
  }
  const json = await resp.json();
  const saved = json?.name;
  if (!saved) throw new Error("Upload succeeded but response had no filename");

  // Route through setSelectedImage so we hit ALL the same side effects as
  // dropdown/arrow picks (cache, preview, label refresh, dirty canvas).
  const imageWidget = node._sfLiImageWidget || (node.widgets || []).find((w) => w.name === "image");
  if (imageWidget) {
    if (!node._sfLiImageWidget) node._sfLiImageWidget = imageWidget;
    setSelectedImage(node, saved);
  }
  return saved;
}

// Opens a hidden <input type="file"> picker; on selection, uploads the file.
export function pickAndUploadFile(node) {
  return new Promise((resolve, reject) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "image/*";
    inp.style.display = "none";
    inp.addEventListener("change", async () => {
      const file = inp.files?.[0];
      if (!file) { inp.remove(); resolve(null); return; }
      try {
        const saved = await uploadImageToInput(node, file);
        resolve(saved);
      } catch (e) {
        reject(e);
      } finally {
        inp.remove();
      }
    });
    document.body.appendChild(inp);
    inp.click();
  });
}

// Reads clipboard for an image; uploads as pasted_<ts>.png.
export async function pasteFromClipboard(node) {
  if (!navigator.clipboard?.read) {
    throw new Error("Clipboard read not supported in this browser");
  }
  const items = await navigator.clipboard.read();
  for (const item of items) {
    for (const type of item.types) {
      if (type.startsWith("image/")) {
        const blob = await item.getType(type);
        const ext = type.split("/")[1] || "png";
        const name = `pasted_${Date.now()}.${ext}`;
        return uploadImageToInput(node, blob, name);
      }
    }
  }
  return null; // no image in clipboard
}
