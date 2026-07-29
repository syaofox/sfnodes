import random
from datetime import datetime

from ...sf_utils.logger import get_logger

logger = get_logger("SFSeed")

_CATEGORY = "sfnodes/utils"

initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
_seed_random_state = random.getstate()
random.setstate(initial_random_state)


def _new_random_seed():
    global _seed_random_state
    prev = random.getstate()
    random.setstate(_seed_random_state)
    seed = random.randint(1, 1125899906842624)
    _seed_random_state = random.getstate()
    random.setstate(prev)
    return seed


class SFSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": -1125899906842624,
                    "max": 1125899906842624,
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "输出种子值，支持 -1/-2/-3 自动生成随机种子"

    @classmethod
    def IS_CHANGED(cls, seed, prompt=None, extra_pnginfo=None, unique_id=None):
        if seed in (-1, -2, -3):
            return _new_random_seed()
        return seed

    def execute(self, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):
        if seed in (-1, -2, -3):
            logger.warning(f'Got "{seed}" as passed seed.')
            original_seed = seed
            seed = _new_random_seed()
            logger.info(f"Server-generated random seed {seed}.")
            if unique_id is not None and prompt is not None:
                prompt_node = prompt.get(str(unique_id))
                if prompt_node is not None and "inputs" in prompt_node and "seed" in prompt_node["inputs"]:
                    prompt_node["inputs"]["seed"] = seed
            if unique_id is not None and extra_pnginfo is not None:
                workflow = extra_pnginfo.get("workflow", extra_pnginfo)
                if isinstance(workflow, dict):
                    nodes = workflow.get("nodes", [])
                    for node in nodes:
                        if str(node.get("id")) == str(unique_id):
                            widgets_values = node.get("widgets_values", [])
                            for i, v in enumerate(widgets_values):
                                if v == original_seed:
                                    widgets_values[i] = seed
        return {"ui": {"SEED": (seed,)}, "result": (seed,)}
