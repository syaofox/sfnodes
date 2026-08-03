import random

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"

HAIRSTYLES = {
    "长直发": "long straight hair",
    "波波头短发": "short bob haircut",
    "波浪及肩发": "wavy shoulder-length hair",
    "卷发": "curly hair",
    "双麻花辫": "twin braids",
    "高马尾": "high ponytail",
    "低马尾": "low ponytail",
    "侧分刘海": "side-swept bangs",
    "精灵短发": "pixie cut",
    "低发髻": "low bun",
    "双低丸子头": "two low buns",
    "中长发": "mid-length layered hair",
    "法式辫": "french braid",
    "半扎发": "half-up half-down hair",
    "长爆炸卷发": "long afro curly hair",
    "背头发型": "slicked-back hair",
}

HAIR_COLORS = {
    "黑发": "black hair",
    "棕发": "brown hair",
    "金发": "blonde hair",
    "银白发": "silver-white hair",
    "粉发": "pink hair",
    "蓝发": "blue hair",
    "紫发": "purple hair",
    "红发": "red hair",
    "深蓝发": "dark blue hair",
    "粉蓝渐变双色发": "gradient two-tone hair (pink and blue)",
    "绿发": "green hair",
    "铜色发": "copper hair",
    "白金发": "platinum blonde hair",
}

CLOTHINGS = {
    "白T恤配牛仔裤": "a white T-shirt and blue jeans",
    "黑色小礼服裙": "a black cocktail dress",
    "商务西装": "a business suit",
    "红色连帽衫配运动裤": "a red hoodie and sweatpants",
    "格纹裙配针织衫": "a plaid skirt and knit sweater",
    "长风衣": "a long trench coat",
    "白衬衫配铅笔裙": "a white blouse and pencil skirt",
    "皮夹克配紧身牛仔裤": "a leather jacket and skinny jeans",
    "运动套装": "a sportswear outfit",
    "紧身短上衣配高腰牛仔裤": "a fitted crop top and high-waisted jeans",
    "紧身连衣裙": "a bodycon dress",
    "低胸晚礼服": "a low-cut evening gown",
    "背心配瑜伽裤": "a tank top and yoga leggings",
    "修身针织衫配迷你裙": "a form-fitting knit sweater and mini skirt",
    "束腰上衣配阔腿裤": "a corset top and wide-leg pants",
    "修身衬衫配紧身裤": "a slim-fit dress shirt and tight trousers",
    "运动内衣配紧身裤": "a sports bra and leggings",
    "吊带背心配牛仔短裤": "a halter top and denim shorts",
    "紧身迷你裙": "a tight mini dress",
    "透视衬衫配紧身牛仔裤": "a sheer blouse and skinny jeans",
}

POSES = {
    "正立面对镜头": "standing upright facing the camera",
    "坐椅": "sitting on a chair",
    "靠墙": "leaning against a wall",
    "行走": "walking",
    "回眸": "looking over the shoulder",
    "盘腿坐地": "sitting cross-legged on the floor",
    "躺床": "lying on a bed",
    "抱膝": "hugging the knees",
    "跪姿": "kneeling",
    "背手站立": "holding the hands behind the back",
    "叉腰站立": "standing with the hands on the hips",
    "抱臂站立": "crossing the arms in front of the chest",
    "手托下巴": "resting the chin on the hand",
    "举手挥舞": "raising one hand to wave",
    "坐窗台": "sitting on a window sill",
    "撑桌前倾": "leaning over a table with both hands",
    "下蹲": "crouching down",
    "伸臂过顶": "stretching the arms overhead",
    "跳舞": "dancing",
    "奔跑": "running",
    "侧坐凳子翘腿": "sitting sideways on a stool with crossed legs",
    "侧卧撑肘": "lying on the side propped on one elbow",
    "踮脚伸手": "standing on tiptoes reaching upward",
    "伸腿坐": "sitting with the legs stretched out straight",
    "瑜伽姿势": "doing a yoga pose",
    "S曲线站姿": "standing with a seductive S-curve body arch",
    "床沿交叉腿坐": "sitting on the edge of the bed with crossed legs",
    "侧卧微屈腿": "lying on the side with the legs slightly bent",
    "跪姿弓背回眸": "kneeling with the back arched and looking back",
    "手抚发丝": "posing with one hand caressing the hair",
    "仰卧单手撑头": "lying on the back on a bed, supporting the head with one hand",
    "前倾弓背": "leaning forward with the back arched",
    "侧身挺胸翘臀": "standing in side profile with the chest out and hips back",
    "翘腿后仰坐": "sitting with the legs crossed and leaning back",
}

CAMERA_DIRECTIONS = {
    "正面": "front view",
    "右前四分之三": "front-right quarter view",
    "右侧面": "right side view",
    "右后四分之三": "back-right quarter view",
    "背面": "back view",
    "左后四分之三": "back-left quarter view",
    "左侧面": "left side view",
    "左前四分之三": "front-left quarter view",
}

CAMERA_ANGLES = {
    "仰角": "low-angle shot",
    "平视": "eye-level shot",
    "轻俯视": "elevated shot",
    "高俯视": "high-angle shot",
}

CAMERA_DISTANCES = {
    "全景": "wide shot",
    "中景": "medium shot",
    "特写": "close-up",
}

_RANDOM = "随机"
_UNCHANGED = "不修改"

IDENTITY_STATEMENT = (
    "Keep the character's identity, gender, facial features and facial structure unchanged."
)


def _resolve_value(selection, rng, options):
    if selection == _RANDOM:
        return rng.choice(list(options.values()))
    if selection == _UNCHANGED:
        return None
    return options[selection]


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
                "hairstyle": ([_RANDOM, _UNCHANGED] + list(HAIRSTYLES), {
                    "default": _RANDOM,
                    "tooltip": "发型：随机 / 不修改 / 指定发型",
                }),
                "hair_color": ([_RANDOM, _UNCHANGED] + list(HAIR_COLORS), {
                    "default": _RANDOM,
                    "tooltip": "发色：随机 / 不修改 / 指定发色",
                }),
                "clothing": ([_RANDOM, _UNCHANGED] + list(CLOTHINGS), {
                    "default": _RANDOM,
                    "tooltip": "服装：随机 / 不修改 / 指定服装",
                }),
                "pose": ([_RANDOM, _UNCHANGED] + list(POSES), {
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
                "camera_direction": ([_RANDOM, _UNCHANGED] + list(CAMERA_DIRECTIONS), {
                    "default": _UNCHANGED,
                    "tooltip": "镜头水平方向：随机 / 不修改 / 指定方向",
                }),
                "camera_angle": ([_RANDOM, _UNCHANGED] + list(CAMERA_ANGLES), {
                    "default": _UNCHANGED,
                    "tooltip": "镜头垂直角度：随机 / 不修改 / 指定角度",
                }),
                "camera_distance": ([_RANDOM, _UNCHANGED] + list(CAMERA_DISTANCES), {
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
