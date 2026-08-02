// ==========================================================================
// SF Markdown - minimal, dependency-free Markdown renderer
// Used by the LoRA info dialog (sf_lora_info.js) to render descriptions.
//
// Safety model: source text is HTML-escaped first, then structured into a
// fixed whitelist of elements (h1-h6, p, br, strong, em, del, code, pre,
// a, img, ul/ol/li, blockquote, hr). No raw HTML ever passes through, and
// dangerous URL schemes (javascript:/vbscript:) are neutralized.
//
// Supported syntax: headings, bold/italic/strikethrough, inline & fenced
// code, links, images, nested unordered/ordered lists, blockquotes,
// horizontal rules, line breaks, bare http(s) URLs.
// ==========================================================================

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function resolveUrl(src) {
    if (/^(?:https?:)?\/\//.test(src) || /^data:/.test(src) || src.startsWith("/")) return src;
    const base = typeof location !== "undefined" && location.origin ? location.origin + "/" : "/";
    return base + src;
}

const resolveHref = (u) => (/^(?:javascript|vbscript|data):/i.test(u) ? "#" : u);
const resolveImgSrc = (u) => {
    if (/^(?:javascript|vbscript):/i.test(u)) return "";
    const r = resolveUrl(u);
    return /^data:/.test(r) && !/^data:image\//.test(r) ? "" : r;
};

const IMG_STYLE = "max-width:100%;border-radius:6px;display:block;margin:6px 0;border:1px solid #444;";
const LINK_STYLE = "color:#7aa2ff;text-decoration:none;word-break:break-all;";
const CODE_STYLE = "background:#1a1a1e;border:1px solid #3a3a3e;border-radius:4px;padding:0 4px;font-size:0.92em;color:#ffd98a;";
const PRE_STYLE = "background:#1a1a1e;border:1px solid #3a3a3e;border-radius:6px;padding:8px 10px;overflow-x:auto;font-size:0.92em;";
const H1_STYLE = "margin:8px 0 4px;font-size:1.3em;color:#fff;";
const H2_STYLE = "margin:8px 0 4px;font-size:1.15em;color:#fff;";
const H3_STYLE = "margin:8px 0 4px;font-size:1.05em;color:#fff;";
const H4_STYLE = "margin:8px 0 4px;font-size:1em;color:#fff;";
const H5_STYLE = "margin:8px 0 4px;font-size:0.95em;color:#fff;";
const H6_STYLE = "margin:8px 0 4px;font-size:0.9em;color:#fff;";
const HEADING_STYLES = [H1_STYLE, H2_STYLE, H3_STYLE, H4_STYLE, H5_STYLE, H6_STYLE];

// Inline parsing. Input must already be HTML-escaped; the output is a
// whitelisted HTML fragment built from that escaped text.
function inline(src) {
    let s = src;

    // Code spans (content kept verbatim)
    s = s.replace(/`([^`\n]+)`/g, (m, code) => `<code style="${CODE_STYLE}">${code}</code>`);

    // Images: ![alt](url)
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)/g, (m, alt, src) => {
        const url = resolveImgSrc(src);
        if (!url) return "";
        return `<a href="${url}" target="_blank" rel="noopener noreferrer"><img src="${url}" alt="${alt}" loading="lazy" style="${IMG_STYLE}"></a>`;
    });

    // Links: [text](url)
    s = s.replace(/(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+[^)]*)?\)/g, (m, text, href) =>
        `<a href="${resolveHref(href)}" target="_blank" rel="noopener noreferrer" style="${LINK_STYLE}">${inline(text)}</a>`
    );

    // Bold / strikethrough / italic
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
    s = s.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");

    // Bare http(s) URLs
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]+)/g, (m, pre, url) => {
        const trimmed = url.replace(/[.,;:!?)\]]+$/, "");
        const tail = url.slice(trimmed.length);
        return `${pre}<a href="${trimmed}" target="_blank" rel="noopener noreferrer" style="${LINK_STYLE}">${trimmed}</a>${tail}`;
    });

    return s;
}

function renderMarkdown(src) {
    if (!src) return "";
    const lines = escapeHtml(src).split(/\r?\n/);
    const out = [];
    let i = 0;

    const openTag = (ordered) => (ordered ? "<ol>" : "<ul>");
    const closeTag = (ordered) => (ordered ? "</ol>" : "</ul>");
    const renderList = (items) => {
        let h = "";
        const stack = [];
        for (const it of items) {
            while (stack.length && stack[stack.length - 1].level > it.level) {
                h += "</li>" + closeTag(stack.pop().ordered);
            }
            if (stack.length && stack[stack.length - 1].level === it.level) {
                h += "</li>";
            } else {
                stack.push({ level: it.level, ordered: it.ordered });
                h += openTag(it.ordered);
            }
            h += `<li>${inline(it.content)}`;
        }
        while (stack.length) h += "</li>" + closeTag(stack.pop().ordered);
        return h;
    };

    while (i < lines.length) {
        const line = lines[i];

        // Fenced code blocks
        if (/^```/.test(line)) {
            const buf = [];
            i++;
            while (i < lines.length && !/^```\s*$/.test(lines[i])) {
                buf.push(lines[i]);
                i++;
            }
            i++;
            out.push(`<pre style="${PRE_STYLE}"><code>${buf.join("\n")}</code></pre>`);
            continue;
        }

        // Headings
        const h = line.match(/^(#{1,6})\s+(.+)$/);
        if (h) {
            const lvl = h[1].length;
            out.push(`<h${lvl} style="${HEADING_STYLES[lvl - 1]}">${inline(h[2])}</h${lvl}>`);
            i++;
            continue;
        }

        // Horizontal rules
        if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
            out.push('<hr style="border:none;border-top:1px solid #444;margin:8px 0;">');
            i++;
            continue;
        }

        // Blockquotes
        if (/^&gt;/.test(line)) {
            const buf = [];
            while (i < lines.length && /^&gt;/.test(lines[i])) {
                buf.push(lines[i].replace(/^&gt;\s?/, ""));
                i++;
            }
            out.push(`<blockquote style="margin:6px 0;padding:4px 12px;border-left:3px solid #6af;color:#bbb;">${inline(buf.join("<br>"))}</blockquote>`);
            continue;
        }

        // Lists
        if (/^\s*(?:[-*+]|\d+\.)\s+/.test(line)) {
            const items = [];
            while (i < lines.length && /^\s*(?:[-*+]|\d+\.)\s+/.test(lines[i])) {
                const m = lines[i].match(/^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$/);
                items.push({
                    level: Math.min(Math.floor(m[1].length / 2), 4),
                    ordered: !!m[3],
                    content: m[4],
                });
                i++;
            }
            out.push(renderList(items));
            continue;
        }

        // Blank line
        if (/^\s*$/.test(line)) {
            i++;
            continue;
        }

        // Paragraph (single line breaks preserved as <br>)
        const buf = [];
        while (i < lines.length && !/^\s*$/.test(lines[i])) {
            buf.push(lines[i]);
            i++;
        }
        out.push(`<p style="margin:4px 0;">${inline(buf.join("<br>"))}</p>`);
    }

    return out.join("");
}

export { renderMarkdown };
