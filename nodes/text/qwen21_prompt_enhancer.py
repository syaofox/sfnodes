"""SFQwenImage21PromptEnhancer - Qwen Image 2.1 提示词增强（复刻 TE_MAN TE_Qwen_Image_2_1_Prompt_Enhancer 能力）。

三种增强方式（干净室复刻：功能对齐，协议文本不抄 TE_MAN）：
- 本地官方PE：clip 输入接 CLIPLoader 加载的官方 PE 模型（Qwen3.5-VL 9B 微调，
  ComfyUI 自动识别为 qwen35_9b），用官方系统提示词原文 + JSON 契约 + thinking 预填。
- 本地LLM：clip 输入接任意指令 VLM（如 qwen3vl_8b safetensors / ComfyUI-GGUF 的
  GGUF+mmproj），用 sf_utils/qwen21_enhance.py 的自写通用协议。
- 本地LLaMA：llama_model 输入接 ComfyUI-llama-cpp_vlm 的 Llama-cpp Model Loader
  （GGUF + mmproj，自带视觉），复用该插件已加载的 Llama 实例与 chat_handler（软依赖）。
- API：OpenAI 兼容 LLM，复用 sfnodes.LLM.* 设置与 sf_utils/llm_client.py（节点不接触密钥）。

任务模式显式选择：官方 PE 按任务分权重（PE-T2I / PE-I2I 不可混用），图生图必须连接
参考图、文生图不接收参考图（错配直接报错）。采样参数用官方生产 profile（presence_penalty
文生图 1.5 / 图生图 0；max token 16256 / 24000），节点暴露 temperature/top_k/top_p/
min_p/repetition_penalty/seed（API 模式仅 temperature/seed 可透传，见 tooltip）。

本地生成复用 SFImageInterrogator 的 clip.tokenize/generate/decode 范式；答案解析
（严格 JSON → 字段级恢复 → 原文兜底）纯逻辑在 sf_utils/qwen21_enhance.py。
"""

import json
import math
import sys

try:
    from ...sf_utils.common import flatten_to_rgb, frame_to_pil, ordered_slot_items
    from ...sf_utils.llm_client import (
        build_image_content,
        chat_completion_sync,
        get_llm_config,
        image_to_data_url,
    )
    from ...sf_utils.qwen21_enhance import (
        LANGUAGE_EN,
        LANGUAGE_ZH,
        MAX_IMAGES,
        TASK_EDIT,
        TASK_T2I,
        build_local_chat_text,
        generic_system_prompt,
        get_profile,
        official_system_prompt,
        parse_answer,
        split_thinking,
    )
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.common import flatten_to_rgb, frame_to_pil, ordered_slot_items  # type: ignore
    from sf_utils.llm_client import (  # type: ignore
        build_image_content,
        chat_completion_sync,
        get_llm_config,
        image_to_data_url,
    )
    from sf_utils.qwen21_enhance import (  # type: ignore
        LANGUAGE_EN,
        LANGUAGE_ZH,
        MAX_IMAGES,
        TASK_EDIT,
        TASK_T2I,
        build_local_chat_text,
        generic_system_prompt,
        get_profile,
        official_system_prompt,
        parse_answer,
        split_thinking,
    )

_CATEGORY = "sfnodes/text"

MODE_OFFICIAL_PE = "本地官方PE"
MODE_LOCAL_LLM = "本地LLM"
MODE_LLAMA = "本地LLaMA"
MODE_API = "API"
MODE_OPTIONS = [MODE_OFFICIAL_PE, MODE_LOCAL_LLM, MODE_LLAMA, MODE_API]

TASK_LABELS = {"文生图": TASK_T2I, "图生图": TASK_EDIT}
LANGUAGE_LABELS = {"英文": LANGUAGE_EN, "中文": LANGUAGE_ZH}


def _find_llama_plugin():
    """查找已加载的 ComfyUI-llama-cpp_vlm 插件模块（软依赖；未安装返回 None）。

    插件目录名含连字符、由 ComfyUI 以路径式模块名加载，不能按包名 import；按属性
    指纹在 sys.modules 里找已加载实例，避免二次实例化导致 LLAMA_CPP_STORAGE 分裂
    （用户 Loader 加载的模型必须与本节点的调用是同一份）。

    指纹必须验到 storage 具备 load_model/clean（仅查名字会撞上插件注册的
    `torch.ops.LLAMA_CPP_STORAGE` 命名空间——实测踩坑）。
    """
    for module in list(sys.modules.values()):
        try:
            storage = getattr(module, "LLAMA_CPP_STORAGE", None)
            if storage is None or not hasattr(storage, "load_model") or not hasattr(storage, "clean"):
                continue
            if not hasattr(module, "llama_cpp_instruct_adv"):
                continue
            return module
        except Exception:  # 惰性加载模块的 __getattr__ 可能抛非 AttributeError
            continue
    return None


class SFQwenImage21PromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"clip": ("CLIP", {
            "tooltip": "本地模式必接：CLIPLoader 加载官方 PE 模型（text_encoders 下的 "
                       "qwen3.5_9b_qwen_image_2.1_pe_t2i/i2i，自动识别为 qwen35_9b，"
                       "类型任选）或任意指令 VLM（如 qwen3vl_8b；GGUF+mmproj 走 ComfyUI-GGUF）。"
                       "API 模式忽略此输入",
        })}
        for i in range(1, MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": "图生图参考图（最多 8 张，批次逐帧计数；按连接槽序即 <image1>..<imageN>）。"
                           "文生图模式不接；官方 PE-I2I 训练时每图约 1MP，见 vision_megapixels",
            })
        optional["system_prompt"] = ("STRING", {
            "forceInput": True,
            "tooltip": "可选系统指令，覆盖内置（官方 PE / 通用自写协议 + 输出语言指令）。"
                       "适合接入自定义协议或调试；不连接则按增强方式自动选择",
        })
        optional["llama_model"] = ("LLAMACPPMODEL", {
            "tooltip": "本地LLaMA 模式必接：ComfyUI-llama-cpp_vlm 的 Llama-cpp Model Loader 输出"
                       "（GGUF 语言模型 + mmproj 视觉投影；图生图必须有 mmproj）。"
                       "该模式下 clip 输入被忽略；其他模式忽略此输入",
        })
        return {
            "required": {
                "mode": (MODE_OPTIONS, {
                    "default": MODE_OFFICIAL_PE,
                    "tooltip": "增强方式。本地官方PE = 官方 PE 微调模型 + 官方系统提示词（JSON 契约，接 clip）；"
                               "本地LLM = 任意本地指令 VLM + 通用自写协议（接 clip）；"
                               "本地LLaMA = ComfyUI-llama-cpp_vlm 的 GGUF+mmproj 模型（接 llama_model，自带视觉）；"
                               "API = sfnodes.LLM.* 设置里的 OpenAI 兼容端点",
                }),
                "task": (list(TASK_LABELS), {
                    "default": "文生图",
                    "tooltip": "任务模式。官方 PE 按任务分权重：文生图用 PE-T2I 模型、图生图用 PE-I2I 模型，"
                               "选错模型效果会明显下降；图生图必须连接参考图，文生图不接收参考图",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "输入要增强的文生图需求或图生图编辑指令（中文/英文均可）。"
                               "用户明确要求的画面文字、指名物体、数量、颜色、位置会逐字保留",
                }),
                "output_language": (list(LANGUAGE_LABELS), {
                    "default": "英文",
                    "tooltip": "控制增强提示词的描述语言（官方 PE 默认英文，中文时追加覆盖指令）。"
                               "用户明确要求的画面文字保持原文与语言，不翻译",
                }),
                "max_tokens": ("INT", {
                    "default": 8192, "min": 0, "max": 32768, "step": 256,
                    "tooltip": "本地模式生成上限（token，含思考链；模型生成到 EOS 自动停止，此值只是上限）。"
                               "0 = 官方 profile 上限（文生图 16256 / 图生图 24000）；本地LLaMA 模式 0 = 不限制"
                               "（用模型/插件默认）。低显存机器调小可压缩最坏耗时（进度条按此值估时），"
                               "撞上限被截断会报错；API 模式不发送此参数",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "采样温度（官方生产值 1.0；0 = 贪心解码）。本地/API 模式均生效",
                }),
                "top_k": ("INT", {
                    "default": 20, "min": 0, "max": 1000,
                    "tooltip": "top-k 采样（官方生产值 20，0 = 禁用）。仅本地模式生效（API 请求体不支持）",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "核采样（官方生产值 0.95）。仅本地模式生效（API 请求体不支持）",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "最小概率阈值（官方生产值 0.0 = 禁用）。仅本地模式生效",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "重复惩罚（官方生产值 1.0 = 无惩罚）。仅本地模式生效",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "随机种子。本地模式直接传给采样；API 模式始终进 LRU 缓存键"
                               "（随机化取新结果、固定值命中缓存）并 best-effort 下发",
                }),
                "thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "思考模式（仅本地官方PE / 本地LLM 的 CLIP 模式）。开启时预填 `<think>` 让模型先推理"
                               "再作答（官方 PE 默认开启），思考链自动剥离；关闭时预填空 think 块抑制推理。"
                               "本地LLaMA 模式由插件的 chat_handler 决定（选 Qwen3-VL-Thinking 即思考）；"
                               "API 模式由 llm_client 处理（DeepSeek 恒关）",
                }),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "参考图缩放上限（百万像素，按面积等比，只缩小不放大）。"
                               "官方 PE 训练时每图约 1MP；调大可保留细节但更慢更占显存",
                }),
                "detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "视觉细节级别（仅 API 模式，OpenAI 兼容字段）：low 更省，high 保留原分辨率细节",
                }),
                "unload_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "生成完成后卸载模型释放显存：CLIP 模式走 ComfyUI（会连带卸载其他节点已加载的模型，"
                               "按需重载）；本地LLaMA 模式走 llama-cpp 插件 storage.clean()（关闭 GGUF 模型，"
                               "其他 Instruct 节点下次执行会自动重载）",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "wh_ratio", "ratio_follow", "report_json")
    OUTPUT_TOOLTIPS = (
        "增强后的提示词（官方 PE 为英文长段落；JSON 解析失败时保留模型原文）",
        "推荐画幅比例（Qwen Image 2.1 官方 7 档之一；ratio_follow 有值时为空）",
        "图生图跟随的画布图标签（如 <image1>；文生图恒为空）",
        "脱敏报告 JSON：mode/task/output_language/thinking/image_count/generated_tokens/"
        "parse_ok/recovered/prompt_chars/wh_ratio/ratio_follow/model/warning/error",
    )
    FUNCTION = "enhance"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Qwen Image 2.1 提示词增强（复刻 TE_MAN TE_Qwen_Image_2_1_Prompt_Enhancer 能力）：把简短"
        "需求/编辑指令改写为 Qwen Image 2.1 偏好的提示词，输出 enhanced_prompt + wh_ratio + "
        "ratio_follow + report_json。三种增强方式——本地官方PE（CLIPLoader 加载官方 PE 模型，"
        "qwen35_9b 自动识别，官方系统提示词 + JSON 契约）、本地LLM（任意指令 VLM + 自写协议）、"
        "本地LLaMA（ComfyUI-llama-cpp_vlm 的 GGUF+mmproj，自带视觉）、"
        "API（sfnodes.LLM.* 设置，节点不接触密钥）。任务模式显式选择（图生图需接图，官方 PE "
        "按任务分权重），最多 8 张参考图；采样用官方生产 profile（presence_penalty 1.5/0），"
        "max_tokens 默认 8192（0 = 官方上限 16256/24000，模型到 EOS 自动停止，此值只是上限），"
        "本地生成后可选卸载模型释放显存。"
    )

    @staticmethod
    def _scale_frame(image, index, megapixels):
        """单帧 -> [1,H,W,C] 张量：alpha 黑底预乘 + 面积上限等比缩小（只缩小不放大）。"""
        import comfy.utils

        frame = flatten_to_rgb(image[index:index + 1])
        samples = frame.movedim(-1, 1)
        total = int(float(megapixels) * 1024 * 1024)
        scale_by = min(1.0, math.sqrt(total / float(samples.shape[3] * samples.shape[2])))
        width = max(1, round(samples.shape[3] * scale_by))
        height = max(1, round(samples.shape[2] * scale_by))
        samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        return [samples.movedim(1, -1)]

    @staticmethod
    def _unload_models():
        """生成后释放显存（对齐 SF VRAMCleanup：卸载缓存模型 + GC + 清缓存）。"""
        import comfy.model_management

        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models_gc()
        comfy.model_management.soft_empty_cache()

    def _request_local(self, clip, text, frames, system, thinking, temperature, top_k,
                       top_p, min_p, repetition_penalty, seed, vision_megapixels, max_tokens, profile):
        """本地 clip 生成：官方聊天原文 + 视觉占位 + 官方 profile 采样。返回 (raw, 生成 token 数, 模型族名)。"""
        images = []
        for tensor, index in frames:
            images.extend(self._scale_frame(tensor, index, vision_megapixels))
        chat = build_local_chat_text(system, text, image_count=len(images), thinking=bool(thinking))
        tokens = clip.tokenize(chat, images=images, thinking=bool(thinking))
        max_length = int(max_tokens) if int(max_tokens) > 0 else int(profile.max_new_tokens)
        generated = clip.generate(
            tokens,
            do_sample=float(temperature) > 0.0,
            max_length=max_length,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            min_p=float(min_p),
            repetition_penalty=float(repetition_penalty),
            seed=int(seed),
            presence_penalty=float(profile.presence_penalty),
        )
        raw = clip.decode(generated)
        tokenizer = getattr(clip, "tokenizer", None)
        return raw, len(generated), str(getattr(tokenizer, "clip_name", "") or "")

    def _request_llama(self, llama_model, text, frames, system, temperature, top_k, top_p,
                       min_p, repetition_penalty, seed, max_tokens, vision_megapixels):
        """本地LLaMA 生成：走 ComfyUI-llama-cpp_vlm 的 storage（GGUF + mmproj，软依赖）。

        返回 (raw, 生成 token 数, 模型名)。与插件 Instruct 节点同源：复用其已加载的
        Llama 实例与 chat_handler（图像走 OpenAI 风格 image_url 部分，由 handler 消费）。
        """
        plugin = _find_llama_plugin()
        if plugin is None:
            raise ValueError(
                "SF Qwen Image 2.1 Prompt Enhancer: 未检测到 ComfyUI-llama-cpp_vlm 插件；"
                "本地LLaMA 模式需要它提供 Llama-cpp Model Loader（GGUF + mmproj）"
            )
        storage = plugin.LLAMA_CPP_STORAGE
        if storage.llm is None or storage.current_config != llama_model:
            storage.load_model(llama_model)
        if frames and getattr(storage.chat_handler, "clip_model_path", None) is None:
            raise ValueError(
                "SF Qwen Image 2.1 Prompt Enhancer: 本地LLaMA 图生图需要 mmproj 视觉投影；"
                "请在 Llama-cpp Model Loader 的 mmproj 下拉选择与模型匹配的文件"
            )
        data_urls = [
            image_to_data_url(frame_to_pil(tensor, index), max_megapixels=vision_megapixels)
            for tensor, index in frames
        ]
        if data_urls:
            user_content = [{"type": "text", "text": text}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": url}} for url in data_urls
            )
        else:
            user_content = text
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        params = {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repeat_penalty": float(repetition_penalty),
            "seed": int(seed),
        }
        if int(max_tokens) > 0:
            params["max_tokens"] = int(max_tokens)
        output = storage.llm.create_chat_completion(messages=messages, **params)
        choices = output.get("choices") or [{}]
        raw = str((choices[0].get("message") or {}).get("content") or "").removeprefix(": ").lstrip()
        generated_tokens = (output.get("usage") or {}).get("completion_tokens")
        return raw, generated_tokens, str((llama_model or {}).get("model") or "")

    def _request_api(self, text, frames, system, temperature, seed, vision_megapixels, detail):
        """API 生成：复用 llm_client（设置读取/请求/LRU）。返回 (raw, 模型名)。"""
        data_urls = [
            image_to_data_url(frame_to_pil(tensor, index), max_megapixels=vision_megapixels)
            for tensor, index in frames
        ]
        user_content = build_image_content(text, data_urls, detail) if data_urls else text
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        config = get_llm_config()
        raw = chat_completion_sync(
            config,
            messages,
            temperature=float(temperature),
            seed=int(seed),
            cache_key_extra=(seed,),
        )
        return raw, str(config.get("model") or "")

    def enhance(self, mode, task, prompt, output_language, max_tokens, temperature, top_k, top_p, min_p,
                repetition_penalty, seed, thinking, vision_megapixels, detail, unload_after,
                clip=None, system_prompt=None, llama_model=None, **kwargs):
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("SF Qwen Image 2.1 Prompt Enhancer: prompt 不能为空，请输入需求或编辑指令")
        task_key = TASK_LABELS.get(task)
        if task_key is None:
            raise ValueError(f"SF Qwen Image 2.1 Prompt Enhancer: 未知任务模式：{task!r}")
        language = LANGUAGE_LABELS.get(output_language, LANGUAGE_EN)
        profile = get_profile(task_key)

        frames = []
        for _name, tensor in ordered_slot_items(kwargs, "image_"):
            for index in range(int(tensor.shape[0])):
                frames.append((tensor, index))
        image_count = len(frames)
        if image_count > MAX_IMAGES:
            raise ValueError(
                f"SF Qwen Image 2.1 Prompt Enhancer: 参考图最多 {MAX_IMAGES} 张"
                f"（含批次逐帧计数），当前 {image_count} 张"
            )
        if profile.requires_images and image_count == 0:
            raise ValueError(
                "SF Qwen Image 2.1 Prompt Enhancer: 图生图模式需要连接至少一张参考图（image_1..8）"
            )
        if not profile.requires_images and image_count:
            raise ValueError(
                "SF Qwen Image 2.1 Prompt Enhancer: 文生图模式不接收参考图，"
                "请断开 image_* 连线或改选图生图"
            )

        system = (system_prompt or "").strip()
        model_warning = ""
        generated_tokens = None
        if mode == MODE_API:
            system = system or generic_system_prompt(task_key, language)
            raw, model_label = self._request_api(
                text, frames, system, temperature, seed, vision_megapixels, detail,
            )
        elif mode == MODE_LLAMA:
            if llama_model is None:
                raise ValueError(
                    "SF Qwen Image 2.1 Prompt Enhancer: 本地LLaMA 模式需要连接 llama_model 输入"
                    "（ComfyUI-llama-cpp_vlm 的 Llama-cpp Model Loader 输出）"
                )
            system = system or generic_system_prompt(task_key, language)
            raw, generated_tokens, model_label = self._request_llama(
                llama_model, text, frames, system, temperature, top_k, top_p, min_p,
                repetition_penalty, seed, max_tokens, vision_megapixels,
            )
        else:
            if clip is None:
                raise ValueError(
                    f"SF Qwen Image 2.1 Prompt Enhancer: {mode} 模式需要连接 clip 输入"
                    "（CLIPLoader 加载官方 PE 模型或指令 VLM）"
                )
            if not system:
                system = (
                    official_system_prompt(task_key, language)
                    if mode == MODE_OFFICIAL_PE
                    else generic_system_prompt(task_key, language)
                )
            raw, generated_tokens, model_label = self._request_local(
                clip, text, frames, system, thinking, temperature, top_k, top_p, min_p,
                repetition_penalty, seed, vision_megapixels, max_tokens, profile,
            )
            if mode == MODE_OFFICIAL_PE and model_label and not model_label.startswith("qwen35"):
                model_warning = (
                    f"当前 clip 模型族为 {model_label}，不是官方 PE 的 qwen35_9b；"
                    "官方提示词与 JSON 契约可能不被完整遵循"
                )

        thinking_text, answer = split_thinking(raw)
        if not answer:
            if thinking_text:
                raise ValueError(
                    "SF Qwen Image 2.1 Prompt Enhancer: 模型只生成了思考内容但没有最终提示词"
                    "（可能被输出上限截断）；请确认模型与任务匹配后重试"
                )
            raise ValueError("SF Qwen Image 2.1 Prompt Enhancer: 模型返回为空")

        parsed = parse_answer(answer, task_key, image_count)
        if (not parsed["parse_ok"] and not parsed["recovered"]
                and mode == MODE_OFFICIAL_PE and bool(thinking) and "</think>" not in raw):
            raise ValueError(
                "SF Qwen Image 2.1 Prompt Enhancer: 输出既无 </think> 也不是合法 JSON，"
                "疑似思考链被截断；请确认 clip 为对应任务的官方 PE 模型"
            )
        error = ""
        if not parsed["parse_ok"]:
            error = (
                "返回不是合法 JSON，已从原文恢复提示词字段"
                if parsed["recovered"] else "返回不是合法 JSON，已保留模型原文"
            )

        report = json.dumps({
            "mode": mode,
            "task": task_key,
            "output_language": language,
            "thinking": bool(thinking) if mode in (MODE_OFFICIAL_PE, MODE_LOCAL_LLM) else None,
            "image_count": image_count,
            "generated_tokens": generated_tokens,
            "parse_ok": parsed["parse_ok"],
            "recovered": parsed["recovered"],
            "prompt_chars": len(parsed["rewritten_prompt"]),
            "wh_ratio": parsed["wh_ratio"],
            "ratio_follow": parsed["ratio_follow"],
            "model": model_label,
            "warning": "；".join(x for x in (parsed.get("warning", ""), model_warning) if x),
            "error": error,
        }, ensure_ascii=False)

        if unload_after:
            try:
                if mode == MODE_LLAMA:
                    plugin = _find_llama_plugin()
                    if plugin is not None:
                        plugin.LLAMA_CPP_STORAGE.clean()
                else:
                    self._unload_models()
            except Exception as exc:  # 卸载失败不影响结果输出
                print(f"[SFQwenImage21PromptEnhancer] 模型卸载失败：{exc}")

        return (parsed["rewritten_prompt"], parsed["wh_ratio"], parsed["ratio_follow"], report)
