"""SFPromptReader - metadata extraction helpers.

Read PNG tEXt/iTXt chunks via PIL, then walk the embedded ComfyUI workflow
JSON to trace the POSITIVE prompt text that drove the image. Falls back to
A1111 / Forge `parameters` style metadata when no ComfyUI workflow is present.

MP4 / WebM / MKV 视频同样支持：纯标准库解析（无第三方依赖）——
- MP4：moov→udta→meta 的 keys/ilst box 链（ffmpeg `-movflags use_metadata_tags`
  布局：ilst item 以 1-based index 指向 keys；iTunes 4cc 布局同样兼容），
  覆盖 VideoHelperSuite 等用 ffmpeg 写元数据的视频保存节点
- WebM / MKV：EBML 容器 Tags→Tag→SimpleTag（TagName/TagString 键值对，
  键名按 Matroska 规范大写，读取时大小写不敏感）

复刻自 comfyui-pixaroma 的 _prompt_reader_helpers.py：保留对 Pixaroma 生态
节点（Switch / PromptStack / Multi / Pack / Dropdown / FromList / Prompt /
SwitchSource / rgthree Any Switch）的兼容支持——读别人用 Pixaroma 插件生成的
图片时同样能恢复 prompt；并新增本项目（sfnodes）节点分支：SFPromptTags、
SFValueDropdown、SFTextPreset、SFAnythingIndexSwitch、SFPauseText、
SFPromptList、SFPromptPreset。

Used by both the Python node (run-time output) and the server route
(/api/sfnodes/prompt_reader/extract for the in-node live readout).
"""

import json
import os
import re
import struct
from typing import Optional

from PIL import Image

# folder_paths is a ComfyUI runtime module; not available in unit-test
# environments. The chase-through-PromptReader feature degrades silently
# when it can't resolve a path.
try:
    import folder_paths as _folder_paths
except ImportError:
    _folder_paths = None


# Extension preference when matching a bare/extension-less name back to a real
# file. PNG first because that is where ComfyUI / A1111 / Forge embed the
# prompt; videos follow so a bare video stem still resolves; the rest are
# included so a match is still found for images that simply have no prompt
# (the readout then explains that).
_MEDIA_EXT_PRIORITY = (
    ".png", ".webp", ".jpeg", ".jpg", ".bmp", ".gif", ".tiff", ".tif",
    ".mp4", ".m4v", ".mov", ".webm", ".mkv",
)

# 视频文件扩展名（read_prompt_from_image 按扩展名分流到 EBML / MP4 解析器）
_VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".webm", ".mkv")

# Upper bound on files scanned by the stem search so a pathological / huge input
# folder (or a client spamming the extract route with non-resolving names) can't
# turn a single lookup into an unbounded tree walk. Real libraries are far under
# this; on hitting it we just return the best match found so far.
_MAX_RESOLVE_SCAN = 50000


def resolve_input_image_name(name):
    """Resolve a possibly extension-less / bare image name to a real file under
    ComfyUI's input directory.

    Load Image SF's ``filename`` output is the base name WITHOUT its
    extension AND without any subfolder (built that way to double as a
    save-prefix), so a value like ``"BunnyExplorer"`` wired into Prompt Reader
    has to be matched back to the actual file (``"BunnyExplorer.png"``). PNG is
    preferred because that is where prompt metadata lives.

    Returns a name that ``folder_paths.get_annotated_filepath`` resolves to an
    existing file (which may equal the input when it already resolves), or
    ``None`` when nothing matches.
    """
    if not name or _folder_paths is None:
        return name or None
    raw = str(name).strip()
    if not raw:
        return None
    # 1) Direct hit - already a complete, valid reference (handles the "name
    #    [input]" annotation and full "sub/file.png" values untouched).
    try:
        p = _folder_paths.get_annotated_filepath(raw)
        if p and os.path.isfile(p):
            return raw
    except Exception:
        pass
    # 2) Strip any "[input]/[output]/[temp]" annotation, normalise separators,
    #    then search the input dir by file stem (extension-less base name).
    ann = re.sub(r"\s*\[(?:input|output|temp)\]\s*$", "", raw)
    ann = ann.replace("\\", "/").strip().strip("/")
    if not ann:
        return None
    sub_hint, _, base = ann.rpartition("/")
    stem = os.path.splitext(base)[0]
    if not stem:
        return None
    try:
        input_dir = _folder_paths.get_input_directory()
    except Exception:
        input_dir = None
    if not input_dir or not os.path.isdir(input_dir):
        return None
    stem_lower = stem.lower()
    best = None
    best_rank = None
    scanned = 0
    # Recursive walk: Load Image strips the subfolder from its filename output,
    # so a subfolder image arrives as a bare stem and must still be found.
    # os.walk yields the input ROOT first, so root-level files are seen before
    # any subfolder is descended into.
    for root, _dirs, fnames in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir).replace("\\", "/")
        rel_sub = "" if rel_root == "." else rel_root
        for fname in fnames:
            scanned += 1
            if scanned > _MAX_RESOLVE_SCAN:
                return best
            fstem, fext = os.path.splitext(fname)
            if fstem.lower() != stem_lower:
                continue
            ext = fext.lower()
            if ext not in _MEDIA_EXT_PRIORITY:
                continue
            rel = fname if rel_sub == "" else rel_sub + "/" + fname
            # Rank (lower is better): a subfolder-hint match wins, then PNG-first
            # extension order, then the shallowest path, then name for stability.
            sub_match = 0 if (sub_hint and rel_sub == sub_hint) else 1
            ext_rank = _MEDIA_EXT_PRIORITY.index(ext)
            depth = rel.count("/")
            rank = (sub_match, ext_rank, depth, rel.lower())
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = rel
            # Fast path for the common case (a bare name from Load Image, no
            # subfolder hint): a root-level PNG is the best rank achievable, so
            # stop instead of walking the rest of the tree. When a subfolder
            # hint IS present a deeper hint-match could still beat it, so only
            # short-circuit when there is no hint.
            if not sub_hint and rel_sub == "" and ext == ".png":
                return best
    return best


# Known text-bearing input names. Frozenset for O(1) lookup; the regex
# `_TEXT_KEY_RE` below catches the long tail of `text_X` / `string_X`
# / `prompt_X` patterns used by various concat / format / chain nodes.
_TEXT_KEYS = frozenset({
    "text", "text_g", "text_l", "string", "str", "prompt",
    "value", "wildcard_text", "input_string", "positive_prompt",
    "format", "template", "prepend", "append", "prefix", "suffix",
})
# Fallback pattern: covers rgthree-style Text Concatenate (`string_a`,
# `string_b`, ...), numbered variants (`text_1`, `text_2`), and the many
# similar concat / chain nodes in the ecosystem.
_TEXT_KEY_RE = re.compile(r"^(text|string|str|prompt)[_-][a-zA-Z0-9]+$")


def _is_text_key(name: str) -> bool:
    """Return True iff `name` looks like a text-carrying input."""
    return name in _TEXT_KEYS or bool(_TEXT_KEY_RE.match(name))


_COND_LINK_KEYS = frozenset({
    "conditioning", "conditioning_1", "conditioning_2",
    "cond", "positive", "from", "input",
})
_SAMPLER_RE = re.compile(r"sampler", re.IGNORECASE)

# ── Pixaroma 生态节点（兼容层：读他人用 Pixaroma 生成的图）──────────────────
#
# Mux / switch nodes: at workflow-run time these route ONE of several inputs
# to their output. The walker has to mirror the same selection logic, or it
# stops at the switch and the prompt comes back empty even though the wired
# upstream text node IS in the workflow JSON.
#
# Selection strategy per node class:
#   PixaromaSwitch         - inputs.SwitchState is a string "1".."32" set by
#                            the JS app.graphToPrompt hook; follow input_{N}.
#   Any Switch (rgthree)   - no widget; rgthree picks the first non-None
#                            any_NN at run-time. Mirror by scanning any_NN
#                            in numeric order and following the first one
#                            that has a wired link.
_MUX_PIX_SWITCH = "PixaromaSwitch"
_MUX_RGTHREE_ANY_SWITCH = "Any Switch (rgthree)"
_RGTHREE_ANY_KEY_RE = re.compile(r"^any_(\d+)$")

# Prompt Stack Pixaroma: ships its rows + separator as a JSON blob in the
# hidden PromptStackState STRING input. The walker rebuilds the joined output
# in pure Python (mirrors nodes/node_prompt_stack.py's build() logic).
_PROMPT_STACK_CLASS = "PixaromaPromptStack"

# Prompt Multi Pixaroma: holds a library of prompts AND can run in either
# of two modes (Queue or List). The hidden PromptMultiState STRING input is
# {"version":2, "mode":"queue"|"list", "activePrompt":"...", "rowTexts":[...]}.
_PROMPT_MULTI_CLASS = "PixaromaPromptMulti"

# Prompt Pack Pixaroma: holds prompts pasted as a text block. The hidden
# PromptPackState STRING input is {"version":1, "activePrompt":"..."}.
_PROMPT_PACK_CLASS = "PixaromaPromptPack"

# Prompt From List Pixaroma: tiny picker that grabs one row from a Prompt
# Multi's list output via an "index" widget. The walker chases its
# `prompts` input back to the upstream Multi and indexes rowTexts.
_PROMPT_FROM_LIST_CLASS = "PixaromaPromptFromList"

# Dropdown Pixaroma: a user-written list of name -> value pairs. The browser
# injects the CHOSEN value into the hidden DropdownState at submit time, so
# what produced this image is a direct read. Two shapes exist: the lean
# {"type","value"} the browser actually sends, and the full {"type","index",
# "options"} a hand-edited API file may carry.
#
# It only contributes when the node is set to TEXT. A whole number, a decimal or
# an on/off is not prompt text, and splicing "1024" into the recovered prompt
# would be worse than recovering nothing.
_DROPDOWN_CLASS = "PixaromaDropdown"

# Prompt Pixaroma: a prompt box whose typed text (with @tags ALREADY expanded)
# lives in the hidden PromptState STRING input {"text": str, "order": str,
# "sep": str}, plus an OPTIONAL wired `text_in` it JOINS with.
_PROMPT_CLASS = "PixaromaPrompt"

# Switch Source Pixaroma: N-output A/B bank switcher. The hidden
# SwitchSourceState carries active side + row count, and the walker follows
# whichever side it names.
_SWITCH_SOURCE_CLASS = "PixaromaSwitchSource"

# ── 本项目（sfnodes）节点 ────────────────────────────────────────────────────

# SFPromptTags 与 PixaromaPrompt 同构：隐藏 PromptState 携带
# {"text", "order", "sep"}，可选 text_in 接线拼接。共享 _PROMPT_CLASS 分支。
_SF_PROMPT_TAGS_CLASS = "SFPromptTags"

# SFValueDropdown 与 PixaromaDropdown 同构：隐藏 DropdownState 注入选中的值
# （lean 形状 {"type","value"}，full 形状兜底手写 API 文件）。共享
# _DROPDOWN_CLASS 分支。
_SF_VALUE_DROPDOWN_CLASS = "SFValueDropdown"

# SFTextPreset：工作流绑定的文本预设。presets_json 是 JSON 数组
# [{name, text}]，preset 是选中的条目名（widget 值，提交时进入 inputs）。
# 恢复选中条目的 text。
_SF_TEXT_PRESET_CLASS = "SFTextPreset"

# SFAnythingIndexSwitch：index widget（INT）直接选 value{index} 输入。
_SF_INDEX_SWITCH_CLASS = "SFAnythingIndexSwitch"

# SFPauseText：隐藏 PauseState {"mode", "text"}。continue 模式前端剪掉 text
# 链接，输出盒子里编辑后的文本；其他模式跟随 text 输入走通用循环。
_SF_PAUSE_TEXT_CLASS = "SFPauseText"

# SFPromptList / SFPromptPreset：文本在普通 widget（multiline_text /
# input_text）里，走特判直读（键名不在 _TEXT_KEYS / _TEXT_KEY_RE 覆盖内）。
_SF_PROMPT_LIST_CLASS = "SFPromptList"
_SF_PROMPT_PRESET_CLASS = "SFPromptPreset"

# PromptReader 自身类名集合（自追链 + "源图不在"提示）。同时认 SFPromptReader
# 与 PixaromaPromptReader，读任一生态的图都能追链。
_SELF_CLASSES = ("SFPromptReader", "PixaromaPromptReader")

_MAX_WALK_DEPTH = 24
# Chase depth caps how many PromptReader hops we follow when an image
# was generated from a workflow that itself contained a PromptReader pointing
# at yet another image. Five levels is plenty for realistic histories and
# bounds the work cleanly.
_MAX_CHASE_DEPTH = 5


def read_png_text_chunks(file_path: str) -> dict:
    """Return all tEXt/iTXt chunks from a PNG as {key: value} strings.

    Empty dict for non-PNG / unreadable files - the caller treats that as
    'no metadata found' and shows the placeholder message.
    """
    try:
        with Image.open(file_path) as img:
            info = dict(img.info or {})
    except Exception:
        return {}
    out = {}
    for k, v in info.items():
        if isinstance(v, (str, bytes)):
            out[str(k)] = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
    return out


# ── 视频元数据（纯标准库，无第三方依赖）─────────────────────────────────────

def _iter_mp4_boxes(data: bytes, start: int, end: int):
    """Yield (box_type: bytes, content_start, content_end) for every child box
    in the ISO BMFF container slice [start, end).

    Handles 64-bit sizes (size==1) and box-to-end (size==0). Malformed boxes
    (size < header) terminate iteration.
    """
    off = start
    while off + 8 <= end:
        size = int.from_bytes(data[off:off + 4], "big")
        typ = data[off + 4:off + 8]
        if size == 1:
            if off + 16 > end:
                break
            size = int.from_bytes(data[off + 8:off + 16], "big")
            hdr = 16
        elif size == 0:
            size = end - off
            hdr = 8
        elif size < 8:
            break
        else:
            hdr = 8
        yield typ, off + hdr, min(off + size, end)
        off += size


def _parse_mp4_meta(data: bytes, start: int, end: int) -> dict:
    """Parse a moov/udta/meta (or moov/meta) full box into {key: value}.

    Two ilst layouts are handled:
      - ffmpeg (`-movflags use_metadata_tags`): each ilst item's 4 bytes are a
        1-based INDEX into the keys list; the item body is a `data` box.
        This is what VideoHelperSuite etc. write.
      - iTunes-style: the item's 4 bytes are the key itself (e.g. ©nam) with a
        `data` box body.
    """
    keys: list = []
    items: dict = {}   # index(0-based) -> bytes, or 4cc str -> bytes
    for typ, b, e in _iter_mp4_boxes(data, start, end):
        if typ == b"keys":
            # full box: 4 bytes version/flags + uint32 count + entries
            if b + 8 > e:
                continue
            count = int.from_bytes(data[b + 4:b + 8], "big")
            o = b + 8
            for _ in range(count):
                if o + 8 > e:
                    break
                ksz = int.from_bytes(data[o:o + 4], "big")
                kt = data[o + 4:o + 8]
                if ksz < 8 or kt != b"mdta" or o + ksz > e:
                    break
                keys.append(data[o + 8:o + ksz].decode("utf-8", "replace"))
                o += ksz
        elif typ == b"ilst":
            for it, ib, ie in _iter_mp4_boxes(data, b, e):
                payload = None
                for styp, sb, se in _iter_mp4_boxes(data, ib, ie):
                    if styp == b"data":
                        # data box: version/flags(4) + reserved(4) + payload
                        payload = data[sb + 8:se]
                        break
                if payload is None:
                    continue
                idx = int.from_bytes(it, "big")
                if 1 <= idx <= max(1, len(keys)):
                    # ffmpeg index style (1-based)
                    items[idx - 1] = payload
                else:
                    items[it.decode("latin1", "replace")] = payload

    out = {}
    for i, k in enumerate(keys):
        if k and i in items:
            out[k] = items[i].decode("utf-8", "replace")
    for k, v in items.items():
        if isinstance(k, str) and k not in out:
            out[k] = v.decode("utf-8", "replace")
    return out


def _read_mp4_metadata(file_path: str) -> dict:
    """Stream-scan an MP4/MOV/M4V for moov→(udta→)meta metadata.

    Large boxes (mdat, etc.) are skipped via seek, only the moov box is read
    into memory, so multi-GB videos stay cheap.
    """
    try:
        f = open(file_path, "rb")
    except OSError:
        return {}
    try:
        moov = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            size, typ = struct.unpack(">I4s", hdr)
            if size == 1:
                ext = f.read(8)
                if len(ext) < 8:
                    break
                size = struct.unpack(">Q", ext)[0]
                hdr_len = 16
            elif size == 0:
                # box runs to end of file: total size = fsize - box_offset.
                # After reading the 8-byte header the file pointer sits at
                # box_offset + 8, so capture it before seeking to the end.
                cur_pos = f.tell()
                f.seek(0, 2)
                fsize = f.tell()
                size = fsize - cur_pos + 8
                hdr_len = 8
            else:
                hdr_len = 8
            if size < hdr_len:
                break
            if typ == b"moov":
                moov = f.read(size - hdr_len)
                break
            f.seek(size - hdr_len, 1)
        if not moov:
            return {}
        # meta 可能在 moov 直下（部分工具），或 moov→udta→meta（ffmpeg）
        for typ, b, e in _iter_mp4_boxes(moov, 0, len(moov)):
            if typ == b"meta":
                return _parse_mp4_meta(moov, b + 4, e)
            if typ == b"udta":
                for t2, b2, e2 in _iter_mp4_boxes(moov, b, e):
                    if t2 == b"meta":
                        return _parse_mp4_meta(moov, b2 + 4, e2)
        return {}
    finally:
        f.close()


# EBML (WebM / MKV) element ids used by the metadata walker.
_EBML_SEGMENT = 0x18538067    # top-level container
_EBML_TAGS = 0x1254C367      # Segment -> Tags
_EBML_TAG = 0x7373           # Tags -> Tag
_EBML_SIMPLE_TAG = 0x67C8    # Tag -> SimpleTag
_EBML_TAG_NAME = 0x45A3      # SimpleTag -> TagName (utf-8)
_EBML_TAG_STRING = 0x4487    # SimpleTag -> TagString (utf-8)
# Element ids whose content we descend into; everything else (Cluster,
# Tracks, ...) is skipped by seek.
_EBML_WALK_IN = (_EBML_SEGMENT, _EBML_TAGS, _EBML_TAG, _EBML_SIMPLE_TAG)


def _ebml_id_len(first: int) -> int:
    """EBML element id length from the first byte: the position of the first
    1 bit (0x80 -> 1, 0x40 -> 2, 0x20 -> 3, 0x10 -> 4). 0 = invalid."""
    if first & 0x80:
        return 1
    if first & 0x40:
        return 2
    if first & 0x20:
        return 3
    if first & 0x10:
        return 4
    return 0


def _ebml_vint(buf: bytes) -> tuple:
    """Decode an EBML variable-length integer from `buf`.

    Returns (value, length_in_bytes), or (None, 0) when the first byte marks
    an unknown size (all payload bits 1) or the buffer is too short.
    """
    if not buf:
        return None, 0
    first = buf[0]
    length = 1
    mask = 0x80
    while not (first & mask):
        mask >>= 1
        length += 1
        if length > 8 or length > len(buf):
            return None, 0
    value = first & (mask - 1)
    for i in range(1, length):
        value = (value << 8) | buf[i]
    if value == (1 << (7 * length)) - 1:
        return None, length  # unknown size
    return value, length


def _scan_ebml(f, start: int, end: int, in_simple_tag: bool = False) -> list:
    """Iterate EBML elements in [start, end), descending only into the Tags
    container chain (Tags/Tag/SimpleTag), and collect (name, value) pairs
    from SimpleTag children.

    Every other element (Cluster, Tracks, ...) is skipped by seek, so the
    scan is cheap even for large videos. Unknown-size elements extend to the
    end of the current container (standard for Segment).
    """
    out = []
    pending_name = None
    cur = start
    while cur + 2 <= end:
        f.seek(cur)
        b0 = f.read(1)
        if len(b0) < 1:
            break
        idlen = _ebml_id_len(b0[0])
        if idlen == 0 or cur + idlen > end:
            break
        f.seek(cur)
        idb = f.read(idlen)
        if len(idb) < idlen:
            break
        eid = int.from_bytes(idb, "big")
        vbuf = f.read(8)
        if not vbuf:
            break
        size, vlen = _ebml_vint(vbuf)
        cstart = cur + idlen + vlen
        if size is None:
            cend = end  # unknown size: element runs to container end
        else:
            cend = cstart + size
        if cend > end:
            break
        if in_simple_tag and eid == _EBML_TAG_NAME:
            f.seek(cstart)
            pending_name = f.read(size).decode("utf-8", "replace")
        elif in_simple_tag and eid == _EBML_TAG_STRING:
            f.seek(cstart)
            value = f.read(size).decode("utf-8", "replace")
            if pending_name is not None:
                out.append((pending_name, value))
                pending_name = None
        elif eid in _EBML_WALK_IN:
            out.extend(_scan_ebml(f, cstart, cend, eid == _EBML_SIMPLE_TAG))
        cur = cend
    return out


def _read_ebml_metadata(file_path: str) -> dict:
    """Extract {key: value} tags from a WebM / MKV (Matroska) file.

    Matroska tag names are uppercased by muxers (ffmpeg writes PROMPT /
    WORKFLOW), so keys are normalised to lowercase here; callers read
    "prompt" / "workflow" / "parameters" case-sensitively.
    """
    try:
        f = open(file_path, "rb")
    except OSError:
        return {}
    try:
        f.seek(0, 2)
        fsize = f.tell()
        pairs = _scan_ebml(f, 0, fsize, False)
    finally:
        f.close()
    out = {}
    for name, value in pairs:
        if not name:
            continue
        key = name.lower()
        if key not in out:
            out[key] = value
    return out


def read_video_text_chunks(file_path: str) -> dict:
    """Return metadata from a video file as {key: value} strings (keys
    lowercased), or {} when unreadable / no metadata.

    MP4 / MOV / M4V go through the ISO BMFF parser, WebM / MKV through the
    EBML parser. The extraction is pure stdlib - no ffmpeg / PyAV needed.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".mp4", ".m4v", ".mov"):
        return _read_mp4_metadata(file_path)
    if ext in (".webm", ".mkv"):
        return _read_ebml_metadata(file_path)
    return {}


def _chase_prompt_reader(node: dict, chase_depth: int) -> Optional[str]:
    """When the walker hits a PromptReader node, the embedded workflow
    only records `inputs.image = "<filename>"` - the actual prompt text was a
    runtime output, never saved into the prompt JSON. To recover it, resolve
    the image filename and recursively read THAT file's metadata.

    Returns None when the source file is missing (e.g. the user deleted it),
    when folder_paths isn't available, or when the chase cap is reached.
    """
    if chase_depth >= _MAX_CHASE_DEPTH or _folder_paths is None:
        return None
    inputs = node.get("inputs") or {}
    image_name = inputs.get("image")
    if not isinstance(image_name, str) or not image_name:
        return None
    try:
        image_path = _folder_paths.get_annotated_filepath(image_name)
    except Exception:
        return None
    if not image_path or not os.path.isfile(image_path):
        return None
    chunks = read_png_text_chunks(image_path)
    if "prompt" in chunks:
        positive = extract_positive_from_comfy_prompt(
            chunks["prompt"], _chase_depth=chase_depth + 1,
        )
        if positive:
            return positive
    if "parameters" in chunks:
        positive = extract_positive_from_a1111(chunks["parameters"])
        if positive:
            return positive
    return None


def _pix_switch_active_link(inputs: dict):
    """Return the active-row link tuple [upstream_id, upstream_output_slot] of
    a PixaromaSwitch.

    SwitchState is injected by the JS app.graphToPrompt hook as a string
    "1".."32". A wired input is stored as [origin_id, origin_slot].
    Returns the full tuple (so the caller can pass origin_slot when recursing
    in case the upstream is itself a Switch Source). Returns None when
    nothing is connected on the active row.
    """
    state = inputs.get("SwitchState")
    try:
        idx = int(str(state)) if state is not None else 0
    except (TypeError, ValueError):
        idx = 0
    if idx < 1:
        return None
    wire = inputs.get(f"input_{idx}")
    if isinstance(wire, list) and len(wire) >= 1:
        return wire
    return None


def _sf_index_switch_active_link(inputs: dict):
    """Return the selected input link tuple [upstream_id, upstream_output_slot]
    of a SFAnythingIndexSwitch.

    The index is a plain INT widget (already in inputs at submit time); the
    selected lane is value{index}. Mirrors the node's run-time selection.
    """
    idx = inputs.get("index", 0)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = 0
    if idx < 0:
        return None
    wire = inputs.get(f"value{idx}")
    if isinstance(wire, list) and len(wire) >= 1:
        return wire
    return None


def _pix_prompt_stack_extract(inputs: dict) -> Optional[str]:
    """Rebuild the joined text from a PixaromaPromptStack's saved state.

    The hidden PromptStackState input is a JSON string of shape:
        { "version": 1, "rows": [{"enabled": bool, "label": str, "text": str}, ...],
          "separator": str }

    Returns the joined text (mirrors node_prompt_stack.py build()), or None
    when nothing is enabled / all rows empty / state malformed.
    """
    raw = inputs.get("PromptStackState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    rows = state.get("rows")
    if not isinstance(rows, list):
        return None
    parts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("enabled"):
            continue
        txt = row.get("text", "") or ""
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if txt.endswith(","):
            txt = txt[:-1].rstrip()
        if not txt:
            continue
        parts.append(txt)
    if not parts:
        return None
    sep = state.get("separator", ", ")
    if not isinstance(sep, str):
        sep = ", "
    return sep.join(parts)


def _pix_prompt_multi_row_at(inputs: dict, index_1based: int) -> Optional[str]:
    """Return the prompt text at the given 1-based index from a Multi's
    enabled-rows list, or None.

    Used by the From List walker to resolve which library row a downstream
    Prompt From List node was pointing at when this image was generated.
    The rowTexts field already contains ONLY enabled non-empty rows in
    display order, so a From List index of 1 maps to rowTexts[0].
    """
    raw = inputs.get("PromptMultiState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    rows = state.get("rowTexts")
    if not isinstance(rows, list):
        return None
    idx0 = int(index_1based) - 1
    if idx0 < 0 or idx0 >= len(rows):
        return None
    item = rows[idx0]
    if not isinstance(item, str):
        return None
    item = item.strip()
    return item or None


def _pix_prompt_from_list_resolve(node: dict, nodes: dict) -> Optional[str]:
    """Resolve a PixaromaPromptFromList node to its picked text.

    Reads the node's `index` widget value, follows the `prompts` input back
    to an upstream PixaromaPromptMulti (in List mode), and returns
    rowTexts[index-1]. Returns None when the upstream isn't a Multi, the
    index is missing / out of range, or the resolved row is empty.

    ComfyUI prompt JSON shape: widget values live in inputs (since they
    were promoted to the inputs dict at submit time); link values are
    [upstream_node_id, output_slot_idx] tuples.
    """
    inputs = node.get("inputs") or {}
    if not isinstance(inputs, dict):
        return None
    idx = inputs.get("index", 1)
    if not isinstance(idx, (int, float)):
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
    idx = int(idx)
    link = inputs.get("prompts")
    if not (isinstance(link, list) and len(link) >= 1):
        return None
    upstream_id = link[0]
    upstream = nodes.get(str(upstream_id))
    if not isinstance(upstream, dict):
        return None
    if upstream.get("class_type") != _PROMPT_MULTI_CLASS:
        # Some other node is feeding the list - we can't resolve it here.
        return None
    return _pix_prompt_multi_row_at(upstream.get("inputs") or {}, idx)


def _pix_prompt_multi_extract(inputs: dict) -> Optional[str]:
    """Read the active prompt from a PixaromaPromptMulti's saved state.

    The hidden PromptMultiState input is a JSON string of shape:
        { "version": 2, "mode": "queue"|"list",
          "activePrompt": str, "rowTexts": [str, ...] }
    (v1 schema {version:1, activePrompt} also handled - same field name.)

    Used when the walker reaches a Multi node via its `text` output (queue
    mode). Each queue iteration bakes that row's text into activePrompt at
    submit time, so the PNG embedded workflow captures exactly the prompt
    that produced that image. Recovery is a direct read.

    For Multi nodes reached via a Prompt From List node (list mode), use
    _pix_prompt_from_list_resolve instead - it indexes rowTexts properly.

    Returns the active prompt, or None when missing / malformed / empty.
    """
    raw = inputs.get("PromptMultiState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    txt = state.get("activePrompt", "")
    if not isinstance(txt, str):
        return None
    txt = txt.strip()
    return txt or None


def _pix_dropdown_extract(inputs: dict) -> Optional[str]:
    """Read the chosen value from a Dropdown (Pixaroma or SFValueDropdown)
    node's saved state.

    The hidden DropdownState is normally the LEAN shape the browser injects:
        { "version": 1, "type": "text"|"int"|"float"|"bool", "value": <chosen> }
    A hand-written API file may instead carry the FULL shape:
        { "type": ..., "index": N, "options": [{"name","value"}, ...] }

    Returns the value only when the node is set to TEXT, else None: a number or
    a true/false is not prompt text, and splicing it into the recovered prompt
    would corrupt the reading rather than improve it.
    """
    raw = inputs.get("DropdownState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(state, dict):
        return None

    kind = state.get("type")
    # Anything that is not explicitly text is refused, INCLUDING a missing type.
    # An unknown/absent type most likely means a newer schema, and guessing
    # "probably text" is how a stray number ends up inside someone's prompt.
    if not isinstance(kind, str) or kind.strip().lower() not in ("text", "string", "str"):
        return None

    if "value" in state:                       # lean shape
        value = state.get("value")
    else:                                      # full shape
        options = state.get("options")
        if not isinstance(options, list):
            return None
        idx = state.get("index")
        if isinstance(idx, bool) or not isinstance(idx, (int, float)):
            idx = 0
        try:
            # int() RAISES on NaN (ValueError) and on +/-Infinity (OverflowError),
            # and json.loads accepts both of those as bare literals. Uncaught,
            # that propagated out of _walk_for_text and took down the reading of
            # the WHOLE image, not just this node's contribution - so a single
            # hand-written state in a shared PNG could stop Prompt Reader
            # recovering anything at all. Nothing on this path may raise.
            idx = int(idx)
        except (ValueError, OverflowError):
            return None
        if idx < 0 or idx >= len(options):
            return None
        entry = options[idx]
        if not isinstance(entry, dict):
            return None
        value = entry.get("value")

    if not isinstance(value, str):
        return None
    return value or None


def _pix_prompt_pack_extract(inputs: dict) -> Optional[str]:
    """Read the active prompt from a PixaromaPromptPack's saved state.

    The hidden PromptPackState input is a JSON string of shape:
        { "version": 1, "activePrompt": str }

    Each queue iteration bakes that prompt's text into activePrompt at
    submit time, so the PNG embedded workflow captures exactly the prompt
    that produced that image. Recovery is a direct read.

    Returns the active prompt, or None when missing / malformed / empty.
    """
    raw = inputs.get("PromptPackState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    txt = state.get("activePrompt", "")
    if not isinstance(txt, str):
        return None
    txt = txt.strip()
    return txt or None


def _pix_prompt_parse_state(inputs: dict):
    """Parse a Prompt node's hidden PromptState into (mine, order, sep).

    Mirrors nodes/node_prompt.py _parse_state: `mine` is the typed prompt with
    @tags expanded and every *category / #list random slot rolled (baked at
    submit time, so this is exactly the text that made the image), `order` is
    "mine"|"wired", `sep` the join separator. Defaults on any malformed /
    missing state. Same schema as SFPromptTags (shared branch).
    """
    raw = inputs.get("PromptState")
    if not isinstance(raw, str) or not raw:
        return "", "mine", ", "
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return "", "mine", ", "
    if not isinstance(state, dict):
        return "", "mine", ", "
    mine = state.get("text", "")
    if not isinstance(mine, str):
        mine = ""
    order = state.get("order")
    order = order if order in ("mine", "wired") else "mine"
    sep = state.get("sep")
    if not isinstance(sep, str) or len(sep) > 16:
        sep = ", "
    return mine, order, sep


def _pix_prompt_join(mine, other, order: str, sep: str) -> Optional[str]:
    """Combine a Prompt node's typed text with its wired text_in, exactly
    like the run() implementation: nothing wired -> just mine; empty mine ->
    just other; else join in the chosen order with the chosen separator.
    Returns the stripped result, or None when both are empty.
    """
    mine = mine if isinstance(mine, str) else ""
    other = other if isinstance(other, str) else ""
    if not other.strip():
        return mine.strip() or None
    if not mine.strip():
        return other.strip() or None
    combined = (other + sep + mine) if order == "wired" else (mine + sep + other)
    return combined.strip() or None


def _sf_text_preset_extract(inputs: dict) -> Optional[str]:
    """Read the selected preset's text from a SFTextPreset node.

    presets_json is a JSON array of {name, text} (the workflow-bound preset
    store); preset is the selected entry's name (widget value, promoted to
    inputs at submit time). Returns the selected entry's text, or None when
    nothing is selected / the store is malformed.
    """
    name = inputs.get("preset")
    if not isinstance(name, str) or not name:
        return None
    raw = inputs.get("presets_json")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and str(item.get("name", "")) == name:
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return None


def _sf_pause_text_extract(inputs: dict) -> Optional[str]:
    """Read a SFPauseText node's editor text in continue mode.

    PauseState is {"mode": "pause"|"continue"|"pass", "text": <box content>}.
    In continue mode the frontend already cut the `text` input link, so the
    box text is the only copy of what flowed downstream - read it directly.

    Returns None for any other mode: the caller falls through to the generic
    input loop, which follows the wired `text` input (a _TEXT_KEYS member).
    """
    raw = inputs.get("PauseState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    if state.get("mode") != "continue":
        return None
    txt = state.get("text", "")
    if not isinstance(txt, str):
        return None
    txt = txt.strip()
    return txt or None


def _rgthree_any_switch_active_link(inputs: dict):
    """Return the active-input link tuple [upstream_id, upstream_output_slot]
    of rgthree's Any Switch.

    rgthree's Any Switch has no widget: at run-time it picks the first non-
    None any_NN value. The walker mirrors that by scanning any_NN keys in
    numeric order and returning the first one that has a wired link.
    Returns the full tuple (for origin_slot threading when the upstream is a
    Switch Source). Returns None when nothing is connected.
    """
    candidates = []
    for key, v in inputs.items():
        m = _RGTHREE_ANY_KEY_RE.match(key)
        if not m:
            continue
        if isinstance(v, list) and len(v) >= 1:
            candidates.append((int(m.group(1)), v))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def _pix_switch_source_active_link(inputs: dict, row: int):
    """Return the active-side link tuple [upstream_id, upstream_output_slot]
    for the given row of a PixaromaSwitchSource, or None.

    SwitchSourceState is injected by the JS graphToPrompt hook as
    {"version":1, "active":"A"|"B", "rows":N, ...}. We resolve the side
    from that state, so this holds whether the inactive side is present or not
    (a browser-submitted prompt is pruned to the active side at submit time; an
    API-exported one keeps both banks).

    Returns None when state is unparseable, row out of range, or the active
    side has no link on that row (the walker bails cleanly in that case).
    """
    raw = inputs.get("SwitchSourceState")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        state = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(state, dict):
        return None
    active = state.get("active", "A")
    if active not in ("A", "B"):
        return None
    key = f"a_{row}" if active == "A" else f"b_{row}"
    wire = inputs.get(key)
    if isinstance(wire, list) and len(wire) >= 1:
        return wire
    return None


def _walk_for_text(
    node_id: str,
    nodes: dict,
    captured: list,
    visited: set,
    depth: int = 0,
    chase_depth: int = 0,
    origin_slot: Optional[int] = None,
) -> None:
    """DFS from `node_id` collecting string text-widget values.

    Follows known conditioning / text link inputs backwards through the graph
    so chains like KSampler -> ConditioningCombine -> CLIPTextEncode resolve
    to the underlying text. Visited-set + depth cap guard against cycles.
    PromptReader nodes are special-cased: their text output is a runtime value
    (not stored in the prompt JSON), so we chase the source image file and
    recursively extract its prompt.

    `origin_slot` is the upstream output slot index of the link that led
    here - needed for multi-output nodes like Switch Source where row R is
    fed by output_R. For other nodes it's harmless extra info.
    """
    if depth > _MAX_WALK_DEPTH:
        return
    sid = str(node_id)
    if sid in visited:
        return
    visited.add(sid)

    node = nodes.get(sid)
    if not isinstance(node, dict):
        return

    # Special-case Prompt Reader (sf / Pixaroma): chase the source file.
    if node.get("class_type") in _SELF_CLASSES:
        chased = _chase_prompt_reader(node, chase_depth)
        if chased:
            captured.append(chased)
        return

    inputs = node.get("inputs") or {}
    if not isinstance(inputs, dict):
        return

    # Mux / switch nodes: pick the active input and recurse through it. If we
    # fall through to the per-input loop instead, the switch's own input names
    # (input_1, any_01, value0, ...) don't match the text/cond heuristics and
    # the walker stops cold at the switch.
    cls = node.get("class_type")
    if cls == _MUX_PIX_SWITCH:
        link = _pix_switch_active_link(inputs)
        if link is not None:
            _walk_for_text(
                link[0], nodes, captured, visited, depth + 1, chase_depth,
                origin_slot=link[1] if len(link) >= 2 else None,
            )
        return
    if cls == _MUX_RGTHREE_ANY_SWITCH:
        link = _rgthree_any_switch_active_link(inputs)
        if link is not None:
            _walk_for_text(
                link[0], nodes, captured, visited, depth + 1, chase_depth,
                origin_slot=link[1] if len(link) >= 2 else None,
            )
        return
    if cls == _SF_INDEX_SWITCH_CLASS:
        link = _sf_index_switch_active_link(inputs)
        if link is not None:
            _walk_for_text(
                link[0], nodes, captured, visited, depth + 1, chase_depth,
                origin_slot=link[1] if len(link) >= 2 else None,
            )
        return

    # Switch Source Pixaroma: 16 outputs, each fed by a_r or b_r depending on
    # active side. We need origin_slot to know which row (R = origin_slot + 1)
    # this output corresponds to. The submit-time hook already pruned the
    # inactive side, so only the active a_r OR b_r carries a link.
    if cls == _SWITCH_SOURCE_CLASS:
        if origin_slot is not None:
            link = _pix_switch_source_active_link(inputs, origin_slot + 1)
            if link is not None:
                _walk_for_text(
                    link[0], nodes, captured, visited, depth + 1, chase_depth,
                    origin_slot=link[1] if len(link) >= 2 else None,
                )
        return

    # Prompt Stack Pixaroma: text is NOT a wired input - all rows live as a
    # JSON blob inside the hidden PromptStackState string. Rebuild the joined
    # output the same way the Python node does at run-time.
    if cls == _PROMPT_STACK_CLASS:
        joined = _pix_prompt_stack_extract(inputs)
        if joined:
            captured.append(joined)
        return

    # Prompt Multi Pixaroma: each generated image carries only the prompt
    # that produced THIS image (the active row at queue time), baked into the
    # hidden PromptMultiState as {"activePrompt": "..."}. Read it directly.
    if cls == _PROMPT_MULTI_CLASS:
        text = _pix_prompt_multi_extract(inputs)
        if text:
            captured.append(text)
        return

    # Prompt Pack Pixaroma: same shape as Prompt Multi - each generated
    # image carries only the prompt that produced THIS image, baked into
    # the hidden PromptPackState. Read it directly.
    if cls == _PROMPT_PACK_CLASS:
        text = _pix_prompt_pack_extract(inputs)
        if text:
            captured.append(text)
        return

    # Dropdown (Pixaroma or SFValueDropdown): the chosen value is baked into
    # the hidden DropdownState at submit time, so the image carries exactly
    # the entry that produced it. Contributes only when the node is set to
    # text - see _pix_dropdown_extract.
    if cls in (_DROPDOWN_CLASS, _SF_VALUE_DROPDOWN_CLASS):
        text = _pix_dropdown_extract(inputs)
        if text:
            captured.append(text)
        return

    # Prompt From List Pixaroma: a tiny picker that grabs one row from a
    # Prompt Multi's `list` output. Read its index widget, walk back to the
    # upstream Multi, and resolve rowTexts[index-1].
    if cls == _PROMPT_FROM_LIST_CLASS:
        text = _pix_prompt_from_list_resolve(node, nodes)
        if text:
            captured.append(text)
        return

    # Prompt (PixaromaPrompt or SFPromptTags): the typed prompt (with @tags
    # already expanded) lives in the hidden PromptState; an OPTIONAL wired
    # text_in is JOINED with it. Read PromptState, resolve text_in via a
    # sub-walk, and combine exactly like the Python node does at run-time.
    if cls in (_PROMPT_CLASS, _SF_PROMPT_TAGS_CLASS):
        mine, order, sep = _pix_prompt_parse_state(inputs)
        other = ""
        ti = inputs.get("text_in")
        if isinstance(ti, list) and len(ti) >= 1:
            sub: list = []
            _walk_for_text(
                ti[0], nodes, sub, visited, depth + 1, chase_depth,
                origin_slot=ti[1] if len(ti) >= 2 else None,
            )
            other = sep.join(s for s in sub if s)
        combined = _pix_prompt_join(mine, other, order, sep)
        if combined:
            captured.append(combined)
        return

    # SFTextPreset: the selected preset's text lives in the workflow-bound
    # presets_json store; preset holds the selected entry name.
    if cls == _SF_TEXT_PRESET_CLASS:
        text = _sf_text_preset_extract(inputs)
        if text:
            captured.append(text)
        return

    # SFPauseText: in continue mode the text link was pruned and the editor
    # text is the only copy; other modes fall through to the generic loop
    # which follows the wired `text` input.
    if cls == _SF_PAUSE_TEXT_CLASS:
        text = _sf_pause_text_extract(inputs)
        if text:
            captured.append(text)
            return
        # fall through (pause/pass/keep with a wired input)

    # SFPromptList: text lives in plain widgets (multiline_text with
    # prepend/append). Recover each resulting row.
    if cls == _SF_PROMPT_LIST_CLASS:
        body = inputs.get("multiline_text")
        prepend = inputs.get("prepend_text") or ""
        append = inputs.get("append_text") or ""
        if isinstance(body, str) and body.strip():
            for line in body.split("\n"):
                s = (prepend + line + append).strip()
                if s:
                    captured.append(s)
        return

    # SFPromptPreset: the typed base prompt is a plain widget; the preset
    # categories are resolved at run-time from a local JSON store (not in the
    # workflow), so only the base text is recoverable. Best effort.
    if cls == _SF_PROMPT_PRESET_CLASS:
        txt = inputs.get("input_text")
        if isinstance(txt, str) and txt.strip():
            captured.append(txt.strip())
        return

    # Single pass over inputs. For each one, classify as text-carrying
    # (capture string OR recurse into linked node), conditioning-link
    # (recurse only), or ignore. Pass v[1] as origin_slot so a downstream
    # Switch Source knows which row this output came from.
    for key, v in inputs.items():
        if _is_text_key(key):
            if isinstance(v, str):
                s = v.strip()
                if s:
                    captured.append(s)
            elif isinstance(v, list) and len(v) >= 1:
                _walk_for_text(
                    v[0], nodes, captured, visited, depth + 1, chase_depth,
                    origin_slot=v[1] if len(v) >= 2 else None,
                )
        elif key in _COND_LINK_KEYS:
            if isinstance(v, list) and len(v) >= 1:
                _walk_for_text(
                    v[0], nodes, captured, visited, depth + 1, chase_depth,
                    origin_slot=v[1] if len(v) >= 2 else None,
                )


def extract_positive_from_comfy_prompt(
    prompt_json: str, _chase_depth: int = 0,
) -> Optional[str]:
    """Parse the ComfyUI 'prompt' PNG chunk and return the positive prompt.

    Strategy: find every sampler-like node (class_type matches /sampler/i),
    follow its 'positive' input backwards through the graph, collect every
    string text-widget value reached. De-duplicate while preserving order
    and join with paragraph separators when multiple distinct texts are
    found (e.g. SDXL CLIPTextEncodeSDXL with text_g + text_l).

    `_chase_depth` is internal - used when the walker follows a PromptReader
    node into its source image's metadata.

    Returns None when no sampler exists OR no text is reached - the caller
    then tries the A1111 fallback.
    """
    try:
        nodes = json.loads(prompt_json)
    except Exception:
        return None
    if not isinstance(nodes, dict):
        return None

    samplers = []
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type") or ""
        if isinstance(cls, str) and _SAMPLER_RE.search(cls):
            samplers.append(nid)

    if not samplers:
        return None

    captured: list = []
    visited: set = set()
    for sid in samplers:
        node = nodes.get(sid)
        if not isinstance(node, dict):
            continue
        pos = (node.get("inputs") or {}).get("positive")
        if isinstance(pos, list) and len(pos) >= 1:
            _walk_for_text(
                pos[0], nodes, captured, visited, 0, _chase_depth,
                origin_slot=pos[1] if len(pos) >= 2 else None,
            )
        elif isinstance(pos, str) and pos.strip():
            captured.append(pos.strip())

    if not captured:
        return None

    seen = set()
    unique = []
    for s in captured:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return "\n\n".join(unique)


_A1111_PARAM_LINE_RE = re.compile(
    r"^(Steps|Sampler|Schedule type|CFG scale|Seed|Size|Model hash|"
    r"Model|VAE|Denoising strength|Clip skip|ENSD|Eta|Hires upscale|"
    r"Hires steps|Hires upscaler|Version):",
    re.MULTILINE,
)


def extract_positive_from_a1111(parameters: str) -> Optional[str]:
    """Pull the positive portion out of an A1111 / Forge 'parameters' string.

    A1111 stores all three sections in one PNG tEXt chunk keyed 'parameters':
        masterpiece, cat
        Negative prompt: ugly, blurry
        Steps: 20, Sampler: Euler, ...

    The positive is everything before either the 'Negative prompt:' marker or
    the first known param line.
    """
    if not isinstance(parameters, str) or not parameters.strip():
        return None
    text = parameters

    neg_idx = text.find("\nNegative prompt:")
    if neg_idx > 0:
        positive = text[:neg_idx]
    else:
        m = _A1111_PARAM_LINE_RE.search(text)
        positive = text[: m.start()] if m else text

    positive = positive.strip()
    return positive or None


def read_prompt_from_image(file_path: str) -> dict:
    """Orchestrator. Returns one of:

      { "found": True,  "text": "<prompt>", "source": "comfyui" | "a1111" }
      { "found": False, "message": "..." }

    PNG goes through the tEXt/iTXt chunk reader; MP4 / WebM / MKV through the
    video metadata parser (pure stdlib).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _VIDEO_EXTENSIONS:
        chunks = read_video_text_chunks(file_path)
    else:
        chunks = read_png_text_chunks(file_path)
    if not chunks:
        return {
            "found": False,
            "message": "No prompt metadata found in this file.",
        }

    if "prompt" in chunks:
        positive = extract_positive_from_comfy_prompt(chunks["prompt"])
        if positive:
            return {"found": True, "text": positive, "source": "comfyui"}

    if "parameters" in chunks:
        positive = extract_positive_from_a1111(chunks["parameters"])
        if positive:
            return {"found": True, "text": positive, "source": "a1111"}

    # If the workflow contains a Prompt Reader and we still got
    # nothing, the failure mode is almost always "the original source image
    # is no longer in input/" (the chase couldn't resolve it). Surface a
    # specific message rather than the generic one so the user knows what
    # to do.
    if "prompt" in chunks:
        try:
            nodes = json.loads(chunks["prompt"])
            if isinstance(nodes, dict) and any(
                isinstance(n, dict) and n.get("class_type") in _SELF_CLASSES
                for n in nodes.values()
            ):
                return {
                    "found": False,
                    "message": (
                        "The prompt came from a Prompt Reader node, but its "
                        "source image is no longer in the input folder so the "
                        "prompt couldn't be traced."
                    ),
                }
        except Exception:
            pass

    return {
        "found": False,
        "message": "Image has metadata but no positive prompt was found.",
    }
