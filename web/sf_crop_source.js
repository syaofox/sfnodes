// ==========================================================================
// sf_crop_source.js - 节点内源图加载链路（Load / Browse / 拖放 / Ctrl+V 共用）
// ==========================================================================
//
// SFImageCropExpand / SFImageBrushMask / SFImageCropExpandBrushMask 三节点的
// 源图链路单源（原两节点各持一份逐字相同的 imageDims/srcViewPart/
// loadAndStoreImage/pickFile/browseImage/restoreImage，差异仅上传前缀/文案/
// 落盘后的状态回写，收敛于此）：
//   dataURL → CropAPI.uploadSrc 落盘 input/sfnodes_crop/ → cfg.onStored 回写
//   宿主状态 → <img> 预览；工作流重载经 /view 恢复（buildSourceURL + cacheBust）。
//
// cfg: {
//   uploadPrefix: "cropexpand_" | "brushmask_" | "cebm_" | ...（upload_src 文件名前缀）,
//   logTag:       "[SF Crop Expand]" 等（console 日志标签）,
//   toastTag:     "SF Crop Expand" 等（sfToast summary 标签）,
//   imgProp:      承载 <img> 的节点属性名（如 "_sfExpandImg"）,
//   onStored({ srcPath, w, h }, node): 状态回写（宿主 setState）,
//   getState(node): 读取宿主状态（restoreSourceImage 取 src_path）,
// }
// ==========================================================================

import { app } from "/scripts/app.js";
import { CropAPI } from "./sf_crop_core.js";
import { sfToast, buildSourceURL, parseAnnotatedImageValue } from "./sf_common.js";
import { showImageBrowser } from "./image_browser.js";

export function imageDims(dataURL) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = reject;
    img.src = dataURL;
  });
}

// src_path → /view 记录（upload_src 返回 "sfnodes_crop/<file>"，前缀即子目录）
export function srcViewPart(srcPath) {
  if (!srcPath) return null;
  const norm = String(srcPath).replace(/\\/g, "/");
  const slash = norm.indexOf("/");
  return {
    filename: slash >= 0 ? norm.slice(slash + 1) : norm,
    subfolder: slash >= 0 ? norm.slice(0, slash) : "",
    type: "input",
  };
}

export function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

export function readBlobAsDataURL(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

// 三入口（Load/Browse/拖放/粘贴）共用：落盘 → 宿主回写状态 → 预览。
export async function storeSource(node, dataURL, cfg) {
  try {
    const dims = await imageDims(dataURL);
    const res = await CropAPI.uploadSrc(cfg.uploadPrefix + Date.now(), dataURL);
    const srcPath = res?.path || "";
    if (!srcPath) {
      sfToast({ summary: cfg.toastTag, detail: "源图上传失败，已取消加载", severity: "error", fallbackTag: cfg.toastTag });
      return;
    }
    cfg.onStored({ srcPath, w: dims.w, h: dims.h }, node);
    const img = new Image();
    img.onload = () => {
      node[cfg.imgProp] = img;
      if (app.graph) app.graph.setDirtyCanvas(true, true);
    };
    img.src = dataURL;
  } catch (err) {
    console.error(`${cfg.logTag} load image failed:`, err);
    sfToast({ summary: cfg.toastTag, detail: "加载图片失败", severity: "error", fallbackTag: cfg.toastTag });
  }
}

export function pickFile(node, cfg) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await storeSource(node, await readFileAsDataURL(file), cfg);
    } catch (err) {
      console.error(`${cfg.logTag} read file failed:`, err);
    }
  };
  input.click();
}

// Browse 按钮：复用 SF Load Image Browser 弹窗（选择器模式），
// 选中后经 /view 取原始字节 → dataURL → 既有落盘+状态链路
export function browseSource(node, cfg) {
  showImageBrowser(node, {
    onPick: async (annotated) => {
      const part = parseAnnotatedImageValue(annotated);
      const url = buildSourceURL(part);
      if (!url) return;
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        await storeSource(node, await readBlobAsDataURL(await resp.blob()), cfg);
      } catch (err) {
        console.error(`${cfg.logTag} browse load failed:`, err);
        sfToast({ summary: cfg.toastTag, detail: "从图片浏览器加载失败", severity: "error", fallbackTag: cfg.toastTag });
      }
    },
  });
}

// 工作流恢复：src_path → /view 预览（状态本体在 properties 里，已随 info 恢复）
export function restoreSourceImage(node, cfg) {
  const st = cfg.getState(node);
  const part = srcViewPart(st.src_path);
  if (!part) return;
  const url = buildSourceURL(part, true);
  if (!url) return;
  const img = new Image();
  img.onload = () => {
    node[cfg.imgProp] = img;
    if (app.graph) app.graph.setDirtyCanvas(true, true);
  };
  img.src = url;
}

// 拖放图片文件到节点显示区加载：cfg.getArea(node) → {x, y, w, h} 判定显示区
// （宿主按自身显示坐标系给出，拖放在区内才接管）。
export function installSourceDrop(node, cfg) {
  node.onDragOver = (e) => {
    const a = cfg.getArea(node);
    const lx = e.canvasX - node.pos[0];
    const ly = e.canvasY - node.pos[1];
    const inArea = lx >= a.x && lx <= a.x + a.w && ly >= a.y && ly <= a.y + a.h;
    if (inArea && e.dataTransfer?.types && Array.from(e.dataTransfer.types).includes("Files")) {
      e.preventDefault();
      e.stopPropagation();
      return true;
    }
    return false;
  };

  node.onDragDrop = (e) => {
    const a = cfg.getArea(node);
    const lx = e.canvasX - node.pos[0];
    const ly = e.canvasY - node.pos[1];
    if (lx < a.x || lx > a.x + a.w || ly < a.y || ly > a.y + a.h) {
      return false;
    }
    const file = e.dataTransfer?.files?.[0];
    if (!file) return false;
    if (!file.type.startsWith("image/")) {
      console.warn(`${cfg.logTag} only image files are supported`);
      return false;
    }
    const reader = new FileReader();
    reader.onload = (event) => storeSource(node, event.target.result, cfg);
    reader.onerror = (err) => console.error(`${cfg.logTag} read file failed:`, err);
    reader.readAsDataURL(file);
    e.preventDefault();
    e.stopPropagation();
    return true;
  };
}
