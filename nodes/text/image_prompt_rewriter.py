"""SFImagePromptRewriter - 把文字 brief（可选 0-10 张有序参考图）改写为图像生成提示词。

自动判定：连接了参考图 = 图像编辑（按 Image 1..N 顺序），无图 = 文生图。
协议与契约校验纯逻辑在 sf_utils/prompt_rewrite.py；LLM 配置/请求/LRU 缓存复用
sf_utils/llm_client.py（API key/模型在 设置 → SF LLM，节点内不接触密钥）。

输出 rewritten_prompt（英文观察者式画面描述）、wh_ratio（独立比例元数据，不写入
正文）与 report_json（结构化状态/纠正次数/超长/错误，不含密钥）。契约不满足或超长
时最多一次有界纠正（温度 0.1）；仍失败保留非空原文 + 空 wh_ratio 并在 report 记错；
空响应抛错（工作流红）。
"""

import json

try:
    from ...sf_utils.common import frame_to_pil, ordered_slot_items
    from ...sf_utils.llm_client import (
        chat_completion_sync,
        get_llm_config,
        image_content_parts,
        image_to_data_url,
    )
    from ...sf_utils.prompt_rewrite import (
        MAX_IMAGES,
        MAX_PROMPT_CHARS,
        RATIO_OPTIONS,
        PromptRewriteError,
        build_messages,
        correction_messages,
        extract_json,
        validate_response,
    )
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.common import frame_to_pil, ordered_slot_items  # type: ignore
    from sf_utils.llm_client import (  # type: ignore
        chat_completion_sync,
        get_llm_config,
        image_content_parts,
        image_to_data_url,
    )
    from sf_utils.prompt_rewrite import (  # type: ignore
        MAX_IMAGES,
        MAX_PROMPT_CHARS,
        RATIO_OPTIONS,
        PromptRewriteError,
        build_messages,
        correction_messages,
        extract_json,
        validate_response,
    )

_CATEGORY = "sfnodes/text"

_CORRECTION_TEMPERATURE = 0.1


def _chat(config, messages, *, temperature, max_tokens, seed, send_seed):
    """统一调用入口（测试可打桩本模块的 chat_completion_sync）。

    send_seed 关闭时不下发 seed，但 seed 恒进缓存键——随机化 seed 取新结果、
    固定 seed 命中缓存，与 SFImageInterrogatorAPI 语义一致。
    """
    return chat_completion_sync(
        config,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=(seed if send_seed else None),
        cache_key_extra=(seed,),
    )


class SFImagePromptRewriter:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE",)
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "图像 brief（中文/英文均可）。固定事实——要显示的文字、指名物体、数量、"
                               "颜色、位置——会逐字保留；未提到的内容由模型补全",
                }),
                "wh_ratio": (RATIO_OPTIONS, {
                    "default": "auto",
                    "tooltip": "画幅比例。auto = 由模型按主体自选（Qwen 官方 7 档之一）；固定值会作为"
                               "独立输出 wh_ratio，且不允许写进提示词正文",
                }),
                "max_output_chars": ("INT", {
                    "default": 0, "min": 0, "max": MAX_PROMPT_CHARS, "step": 100,
                    "tooltip": "改写字数上限（字符）。0 = 交给模型决定完整长度；非零超长时执行一次"
                               "有界纠正，仍超长保留完整结果并在 report 标 over_limit",
                }),
                "transparent_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "透明通道：要求正文明确 RGBA、alpha channel 与 transparent background "
                               "三语义（否则触发一次纠正）",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "采样温度。越低越稳定（建议 0.1~0.4）；纠正调用固定 0.1",
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 8192,
                    "tooltip": "单次返回最大 token 数（DeepSeek thinking 已关闭，不占推理预算）。"
                               "截断时先调大此值",
                }),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "参考图发送前的缩放上限（百万像素，按面积等比，只缩小不放大）。控制请求体积/成本",
                }),
                "detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "视觉细节级别（OpenAI 兼容字段）：low 更省，high 保留原分辨率细节",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "随机种子（best-effort 复现）。是否真正发给 API 取决于 send_seed；"
                               "即使不发送也会区分 LRU 缓存——随机化取新结果，固定值命中缓存",
                }),
                "send_seed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否把 seed 发给 API。官方 DeepSeek Chat Completions 未文档化该字段，"
                               "默认关闭；第三方 OpenAI 兼容端点可开启",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "wh_ratio", "report_json")
    OUTPUT_TOOLTIPS = (
        "改写后的英文画面描述（观察者式，不含比例/分辨率）。契约失败时保留模型原文",
        "比例元数据（auto 时为模型所选），如 3:2；未结构化成功时为空串",
        "脱敏报告 JSON：structured_response / correction_calls / over_limit / prompt_chars /"
        "image_count / wh_ratio / provider / model / error",
    )
    FUNCTION = "rewrite"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "图像提示词改写（LLM API）：把文字 brief 改写为英文观察者式画面描述（协议吸收 Qwen Image 2.1 "
        "提示词实践、文本自写），可选连接 1–10 张有序参考图（有图 = 图像编辑，逐帧计数按序标注 "
        "Image 1..N）。输出 rewritten_prompt、独立 wh_ratio（auto 或 Qwen 官方 7 档比例）与脱敏 "
        "report_json。契约不满足/超长时最多一次有界纠正，失败保留原文不静默截断；API key/模型在 "
        "设置 → SF LLM 配置；相同输入命中内存 LRU 缓存不再出网（设置 → SF LLM 可关闭）。"
    )

    def rewrite(self, prompt, wh_ratio, max_output_chars, transparent_alpha, temperature,
                max_tokens, vision_megapixels, detail, seed, send_seed, **kwargs):
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("SF Image Prompt Rewriter: prompt 不能为空，请输入图像描述")
        ratio = str(wh_ratio or "auto")
        if ratio not in RATIO_OPTIONS:
            raise ValueError(f"SF Image Prompt Rewriter: 不支持的 wh_ratio：{ratio}")
        max_chars = int(max_output_chars or 0)
        transparent = bool(transparent_alpha)

        frames = []
        for _name, tensor in ordered_slot_items(kwargs, "image_"):
            for index in range(int(tensor.shape[0])):
                frames.append((tensor, index))
        if len(frames) > MAX_IMAGES:
            raise ValueError(
                f"SF Image Prompt Rewriter: 参考图最多 {MAX_IMAGES} 张（含批次逐帧计数），当前 {len(frames)} 张"
            )
        data_urls = [
            image_to_data_url(frame_to_pil(tensor, index), max_megapixels=vision_megapixels)
            for tensor, index in frames
        ]
        image_count = len(data_urls)
        messages = build_messages(
            text, image_count, ratio, max_chars, transparent, image_content_parts(data_urls, detail),
        )
        config = get_llm_config()
        call = dict(max_tokens=int(max_tokens), seed=int(seed), send_seed=bool(send_seed))

        corrections = 0
        structured = False
        report_error = ""
        rewritten = ""
        chosen_ratio = ""
        meta = {}
        raw = _chat(config, messages, temperature=float(temperature), **call)
        try:
            payload = extract_json(raw)
            if payload is None:
                raise PromptRewriteError("模型返回不是合法 JSON")
            rewritten, chosen_ratio, meta = validate_response(
                payload, requested_ratio=ratio, transparent=transparent, max_chars=max_chars,
            )
            if meta["over_limit"]:
                corrections += 1
                try:
                    corrected = _chat(
                        config,
                        correction_messages(
                            messages, raw,
                            f"Keep rewritten_prompt at or below {max_chars} characters without cutting it mid-sentence.",
                        ),
                        temperature=_CORRECTION_TEMPERATURE, **call,
                    )
                    corrected_payload = extract_json(corrected)
                    if corrected_payload is None:
                        raise PromptRewriteError("纠正返回不是合法 JSON")
                    rewritten, chosen_ratio, meta = validate_response(
                        corrected_payload, requested_ratio=ratio, transparent=transparent, max_chars=max_chars,
                    )
                    raw = corrected
                except Exception as exc:
                    # 首次结果已合法，纠正失败（网络/解析/校验）不丢稿
                    report_error = f"长度纠正未生效，已保留原结果：{exc}"
            structured = True
        except PromptRewriteError as first_error:
            corrections += 1
            try:
                corrected = _chat(
                    config,
                    correction_messages(messages, raw, str(first_error)),
                    temperature=_CORRECTION_TEMPERATURE, **call,
                )
                corrected_payload = extract_json(corrected)
                if corrected_payload is None:
                    raise first_error
                rewritten, chosen_ratio, meta = validate_response(
                    corrected_payload, requested_ratio=ratio, transparent=transparent, max_chars=max_chars,
                )
                raw = corrected
                structured = True
            except Exception as exc:
                report_error = str(exc)
                rewritten = str(raw or "").strip()
                chosen_ratio = ""
                if not rewritten:
                    raise ValueError(f"SF Image Prompt Rewriter: {report_error}") from exc
        report = json.dumps({
            "structured_response": structured,
            "correction_calls": corrections,
            "over_limit": bool(meta.get("over_limit")),
            "prompt_chars": len(rewritten),
            "image_count": image_count,
            "requested_ratio": ratio,
            "wh_ratio": chosen_ratio,
            "provider": str(config.get("provider") or ""),
            "model": str(config.get("model") or ""),
            "error": report_error,
        }, ensure_ascii=False)
        return (rewritten, chosen_ratio, report)
