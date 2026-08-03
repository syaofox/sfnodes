import random

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"

HAIRSTYLES = [
    "long straight hair",
    "short bob haircut",
    "wavy shoulder-length hair",
    "curly hair",
    "twin braids",
    "high ponytail",
    "low ponytail",
    "side-swept bangs",
    "pixie cut",
    "low bun",
    "two low buns",
    "mid-length layered hair",
    "french braid",
    "half-up half-down hair",
    "long afro curly hair",
    "slicked-back hair",
]

HAIR_COLORS = [
    "black hair",
    "brown hair",
    "blonde hair",
    "silver-white hair",
    "pink hair",
    "blue hair",
    "purple hair",
    "red hair",
    "dark blue hair",
    "gradient two-tone hair (pink and blue)",
    "green hair",
    "copper hair",
    "platinum blonde hair",
]

CLOTHINGS = [
    "a white T-shirt and blue jeans",
    "a black cocktail dress",
    "a business suit",
    "a red hoodie and sweatpants",
    "a plaid skirt and knit sweater",
    "a long trench coat",
    "a white blouse and pencil skirt",
    "a leather jacket and skinny jeans",
    "a sportswear outfit",
    "a fitted crop top and high-waisted jeans",
    "a bodycon dress",
    "a low-cut evening gown",
    "a tank top and yoga leggings",
    "a form-fitting knit sweater and mini skirt",
    "a corset top and wide-leg pants",
    "a slim-fit dress shirt and tight trousers",
    "a sports bra and leggings",
    "a halter top and denim shorts",
    "a tight mini dress",
    "a sheer blouse and skinny jeans",
]

POSES = [
    "standing upright facing the camera",
    "sitting on a chair",
    "leaning against a wall",
    "walking",
    "looking over the shoulder",
    "sitting cross-legged on the floor",
    "lying on a bed",
    "hugging the knees",
    "kneeling",
    "holding the hands behind the back",
    "standing with the hands on the hips",
    "crossing the arms in front of the chest",
    "resting the chin on the hand",
    "raising one hand to wave",
    "sitting on a window sill",
    "leaning over a table with both hands",
    "crouching down",
    "stretching the arms overhead",
    "dancing",
    "running",
    "sitting sideways on a stool with crossed legs",
    "lying on the side propped on one elbow",
    "standing on tiptoes reaching upward",
    "sitting with the legs stretched out straight",
    "doing a yoga pose",
    "standing with a seductive S-curve body arch",
    "sitting on the edge of the bed with crossed legs",
    "lying on the side with the legs slightly bent",
    "kneeling with the back arched and looking back",
    "posing with one hand caressing the hair",
    "lying on the back on a bed, supporting the head with one hand",
    "leaning forward with the back arched",
    "standing in side profile with the chest out and hips back",
    "sitting with the legs crossed and leaning back",
]

CAMERA_DIRECTIONS = [
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
]

CAMERA_ANGLES = [
    "low-angle shot",
    "eye-level shot",
    "elevated shot",
    "high-angle shot",
]

CAMERA_DISTANCES = [
    "wide shot",
    "medium shot",
    "close-up",
]

_RANDOM = "随机"
_UNCHANGED = "不修改"

IDENTITY_STATEMENT = (
    "Keep the character's identity, gender, facial features and facial structure unchanged."
)


def _resolve_value(selection, rng, options):
    if selection == _RANDOM:
        return rng.choice(options)
    if selection == _UNCHANGED:
        return None
    return selection


def _build_prompt(hairstyle, hair_color, clothing, pose, keep_identity, white_background):
    changes = []
    if hairstyle:
        changes.append(f"hairstyle to {hairstyle}")
    if hair_color:
        changes.append(f"hair color to {hair_color}")
    if clothing:
        changes.append(f"clothing to {clothing}")
    if pose:
        changes.append(f"pose to {pose}")

    if changes:
        if len(changes) == 1:
            sentence = "Change the character's " + changes[0]
        else:
            sentence = "Change the character's " + ", ".join(changes[:-1]) + ", and " + changes[-1]
    else:
        sentence = ""

    if white_background:
        sentence = (sentence + " " if sentence else "") + "Set the background to pure white."

    if keep_identity:
        sentence += " " + IDENTITY_STATEMENT
    return sentence


def _build_camera_prompt(direction, angle, distance):
    parts = [p for p in (direction, angle, distance) if p]
    if not parts:
        return "", ""
    joined = " ".join(parts)
    return f"<sks> {joined}", joined


class SFRandomEditPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1125899906842624,
                    "tooltip": "-1: 每次运行随机生成；>=0: 固定种子可复现",
                }),
                "hairstyle": ([_RANDOM, _UNCHANGED] + HAIRSTYLES, {
                    "default": _RANDOM,
                    "tooltip": "发型：随机 / 不修改 / 指定发型",
                }),
                "hair_color": ([_RANDOM, _UNCHANGED] + HAIR_COLORS, {
                    "default": _RANDOM,
                    "tooltip": "发色：随机 / 不修改 / 指定发色",
                }),
                "clothing": ([_RANDOM, _UNCHANGED] + CLOTHINGS, {
                    "default": _RANDOM,
                    "tooltip": "服装：随机 / 不修改 / 指定服装",
                }),
                "pose": ([_RANDOM, _UNCHANGED] + POSES, {
                    "default": _RANDOM,
                    "tooltip": "姿势：随机 / 不修改 / 指定姿势",
                }),
                "keep_identity": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否在提示词中附加保持身份（性别、五官）不变的语句",
                }),
                "white_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否将背景改为纯白",
                }),
                "camera_direction": ([_RANDOM, _UNCHANGED] + CAMERA_DIRECTIONS, {
                    "default": _UNCHANGED,
                    "tooltip": "镜头水平方向：随机 / 不修改 / 指定方向",
                }),
                "camera_angle": ([_RANDOM, _UNCHANGED] + CAMERA_ANGLES, {
                    "default": _UNCHANGED,
                    "tooltip": "镜头垂直角度：随机 / 不修改 / 指定角度",
                }),
                "camera_distance": ([_RANDOM, _UNCHANGED] + CAMERA_DISTANCES, {
                    "default": _UNCHANGED,
                    "tooltip": "镜头景别：随机 / 不修改 / 指定景别",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING, IO.STRING, IO.STRING, IO.STRING, IO.STRING)
    RETURN_NAMES = ("prompt", "hairstyle", "hair_color", "clothing", "pose", "camera")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "随机生成图片编辑提示词：保持人物身份（性别、五官）不变，随机改变发型、发色、服装、姿势、镜头"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        if seed == -1:
            return random.random()
        return seed

    def execute(self, seed, hairstyle, hair_color, clothing, pose, keep_identity,
                white_background, camera_direction, camera_angle, camera_distance):
        rng = random.Random(None if seed == -1 else seed)
        hairstyle = _resolve_value(hairstyle, rng, HAIRSTYLES)
        hair_color = _resolve_value(hair_color, rng, HAIR_COLORS)
        clothing = _resolve_value(clothing, rng, CLOTHINGS)
        pose = _resolve_value(pose, rng, POSES)
        direction = _resolve_value(camera_direction, rng, CAMERA_DIRECTIONS)
        angle = _resolve_value(camera_angle, rng, CAMERA_ANGLES)
        distance = _resolve_value(camera_distance, rng, CAMERA_DISTANCES)

        sentence = _build_prompt(hairstyle, hair_color, clothing, pose, keep_identity, white_background)
        camera_prompt, camera_value = _build_camera_prompt(direction, angle, distance)

        if camera_prompt:
            prompt = camera_prompt + (", " + sentence if sentence else "")
        else:
            prompt = sentence
        return (prompt, hairstyle or "", hair_color or "", clothing or "", pose or "", camera_value)
