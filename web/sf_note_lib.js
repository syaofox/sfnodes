// ==========================================================================
// sf_note_lib.js - SF Note 纯逻辑库（复刻孤海注释文本引擎）
// ==========================================================================
//
// 无 app/DOM 依赖（纯模块边界，禁止 import sf_common.js），供主扩展
// sf_note.js 使用，也供 tests/ 复制为 .mjs 直接测试。
// 原版 GoohaiNote.js 的链接识别/字符表/CJK 换行引擎逐义移植，唯一签名
// 差异：wrapCharList 接收 measure(string)→width 函数而非 canvas ctx
// （原版三处 ctx.measureText().width，逻辑逐字一致；调用方传
// ctx.measureText.bind(ctx)，DOM 渲染与 canvas 绘制同源）。
// ==========================================================================

// 链接识别：[[...]] / http(s):// / www.（原版 LINK_REGEX 一致）
const URL_CHARS = "a-zA-Z0-9._~:/?#\\[\\]@!$&'()*+,;=%\\-";
const LINK_REGEX = new RegExp(
  "(\\[\\[(.+?)\\]\\])" +
    "|(https?:\\/\\/[" + URL_CHARS + "]+)" +
    "|(www\\.[" + URL_CHARS + "]+)",
  "g"
);

const isChinese = (ch) => /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/.test(ch);
const isBreakPoint = (ch) => /[\s\p{P}\p{S}]/u.test(ch);
const isPunct = (ch) => /\p{P}|\p{S}/.test(ch);

const CJK_PUNCT_SET = new Set(
  "，。、！？：；「」『』【】《》（）\u201C\u201D\u2018\u2019—…～．".split("")
);
const isCJKPunct = (ch) => CJK_PUNCT_SET.has(ch);
const isCJKLike = (ch) => isChinese(ch) || isCJKPunct(ch);

// 行首禁则（kinsoku）：这些标点不能出现在行首
const LINE_START_FORBIDDEN = new Set([
  ",", "，", ".", "。", ";", "；", "!", "！", "\u201D", "\u2019", "」", "』",
]);
const isLineStartForbidden = (ch) => LINE_START_FORBIDDEN.has(ch);

// 文本切分：纯文本段 / 链接段（原版 parseSegments 一致）
export function parseSegments(text) {
  LINK_REGEX.lastIndex = 0;
  const segments = [];
  let match;
  let last = 0;
  while ((match = LINK_REGEX.exec(text)) !== null) {
    if (match.index > last)
      segments.push({ type: "text", content: text.substring(last, match.index), url: null });
    if (match[1] !== undefined) {
      segments.push({ type: "link", content: match[2], url: match[2] });
    } else if (match[3] !== undefined) {
      segments.push({ type: "link", content: match[3], url: match[3] });
    } else if (match[4] !== undefined) {
      segments.push({ type: "link", content: match[4], url: match[4] });
    }
    last = match.index + match[0].length;
  }
  if (last < text.length)
    segments.push({ type: "text", content: text.substring(last), url: null });
  return segments;
}

// 段 → 逐字表（携带 type/url，原版 buildCharList 一致）
export function buildCharList(segments) {
  const list = [];
  for (const seg of segments) {
    for (const ch of seg.content) {
      list.push({ ch, type: seg.type, url: seg.url });
    }
  }
  return list;
}

// CJK 换行（原版 wrapCharList 一致；measure 替代 ctx.measureText().width）
export function wrapCharList(measure, charList, maxWidth) {
  if (!charList || charList.length === 0) return [[]];

  const lines = [];
  let line = [];
  let lineStr = "";

  const rebuild = () => {
    lineStr = line.map((c) => c.ch).join("");
  };

  for (let i = 0; i < charList.length; i++) {
    const c = charList[i];

    if (c.ch === "\n") {
      lines.push(line);
      line = [];
      lineStr = "";
      continue;
    }

    const test = lineStr + c.ch;

    if (measure(test) > maxWidth && line.length > 0) {
      if (isCJKLike(c.ch)) {
        lines.push(line);
        line = [c];
        lineStr = c.ch;
      } else {
        let lastCJK = -1;
        for (let k = line.length - 1; k >= 0; k--) {
          if (isCJKLike(line[k].ch)) {
            lastCJK = k;
            break;
          }
        }

        if (lastCJK >= 0) {
          lines.push(line.slice(0, lastCJK + 1));
          line = [...line.slice(lastCJK + 1), c];
          rebuild();
        } else {
          const limit = Math.min(20, line.length);
          let bp = -1;
          for (let j = line.length - 1; j >= line.length - limit; j--) {
            if (isBreakPoint(line[j].ch)) {
              bp = j;
              break;
            }
          }

          if (bp >= 0) {
            lines.push(line.slice(0, bp + 1));
            line = [...line.slice(bp + 1), c];
            rebuild();
            if (line.length > 0 && isPunct(line[0].ch)) {
              let pEnd = 0;
              while (pEnd < line.length && isPunct(line[pEnd].ch)) pEnd++;
              lines[lines.length - 1] = [...lines[lines.length - 1], ...line.slice(0, pEnd)];
              line = line.slice(pEnd);
              rebuild();
            }
          } else if (isPunct(c.ch)) {
            lines.push([...line, c]);
            line = [];
            lineStr = "";
          } else {
            lines.push(line);
            line = [c];
            lineStr = c.ch;
          }
        }
      }

      // 行首禁则回拉（kinsoku，原版 fix 循环一致，上限 20 次防抖）
      let fix = 0;
      while (fix++ < 20 && line.length > 0 && isLineStartForbidden(line[0].ch) && lines.length > 0) {
        const prev = lines[lines.length - 1];
        if (!prev || prev.length <= 1) break;
        const last = prev[prev.length - 1];
        if (isCJKLike(last.ch)) {
          line = [last, ...line];
          lines[lines.length - 1] = prev.slice(0, -1);
        } else if (!isBreakPoint(last.ch)) {
          let ws = prev.length - 1;
          while (ws > 0 && !isBreakPoint(prev[ws - 1].ch)) ws--;
          if (ws === 0) {
            line = [last, ...line];
            lines[lines.length - 1] = prev.slice(0, -1);
          } else {
            line = [...prev.slice(ws), ...line];
            lines[lines.length - 1] = prev.slice(0, ws);
          }
        } else break;
        rebuild();
      }
    } else {
      line.push(c);
      lineStr = test;
    }
  }

  if (line.length > 0) lines.push(line);
  return lines.length > 0 ? lines : [[]];
}

// 一站式排版：预处理（\n 转义/回车归一，原版 drawMultilineText 头部一致）→
// 分段 → 逐字 → 换行，返回行数组（行 = 字符对象数组，含 type/url）
export function layoutText(text, maxWidth, measure) {
  const processed = String(text || "")
    .replace(/\\n/g, "\n")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n");
  return wrapCharList(measure, buildCharList(parseSegments(processed)), maxWidth);
}

// ── 样式 ──
export const DEFAULT_STYLE = {
  fontSize: 24,
  fontColor: "#C8C8C8",
  backgroundColor: "#1B4669",
  backgroundAlpha: 0.25,
  borderRadius: 20,
  padding: 12,
  lineHeight: 1.4,
  textAlign: "center",
  stroke: true,
  locked: false,
};

const clampNum = (v, lo, hi, fallback) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(hi, Math.max(lo, n));
};

const isHexColor = (v) => typeof v === "string" && /^#[0-9a-fA-F]{6}$/.test(v);

// 样式归一化（非法值回默认；原版 @ 注解范围收敛于此，面板编辑与工作流恢复同源）
export function normalizeStyle(raw) {
  const r = raw && typeof raw === "object" ? raw : {};
  const d = DEFAULT_STYLE;
  const textAlign = r.textAlign === "left" || r.textAlign === "right" ? r.textAlign : "center";
  return {
    fontSize: Math.round(clampNum(r.fontSize, 8, 200, d.fontSize)),
    fontColor: isHexColor(r.fontColor) ? r.fontColor : d.fontColor,
    backgroundColor: isHexColor(r.backgroundColor) ? r.backgroundColor : d.backgroundColor,
    backgroundAlpha: clampNum(r.backgroundAlpha, 0, 1, d.backgroundAlpha),
    borderRadius: Math.round(clampNum(r.borderRadius, 0, 300, d.borderRadius)),
    padding: Math.round(clampNum(r.padding, 0, 50, d.padding)),
    lineHeight: clampNum(r.lineHeight, 0.8, 3.0, d.lineHeight),
    textAlign,
    stroke: r.stroke === undefined ? d.stroke : !!r.stroke,
    locked: !!r.locked,
  };
}

export function parseStyle(json) {
  try {
    return normalizeStyle(JSON.parse(json));
  } catch {
    return { ...DEFAULT_STYLE };
  }
}

export function hexToRgba(hex, alpha) {
  if (typeof hex !== "string" || hex.length < 7) return `rgba(120,120,120,${alpha})`;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// 描边色：背景色各通道 +80 钳制（原版 onDrawBackground 一致）
export function strokeColor(backgroundColor, alpha) {
  const r = Math.min(parseInt(backgroundColor.slice(1, 3), 16) + 80, 255);
  const g = Math.min(parseInt(backgroundColor.slice(3, 5), 16) + 80, 255);
  const b = Math.min(parseInt(backgroundColor.slice(5, 7), 16) + 80, 255);
  return `rgba(${r},${g},${b},${alpha})`;
}

// 行内同 type/url 合并成组（DOM 渲染用，原版 drawMultilineText 分组一致）
export function groupLine(lineChars) {
  const groups = [];
  let gi = 0;
  while (gi < lineChars.length) {
    const { type, url } = lineChars[gi];
    let gj = gi;
    while (gj < lineChars.length && lineChars[gj].type === type && lineChars[gj].url === url) gj++;
    groups.push({
      type,
      url,
      text: lineChars.slice(gi, gj).map((c) => c.ch).join(""),
    });
    gi = gj;
  }
  return groups;
}
