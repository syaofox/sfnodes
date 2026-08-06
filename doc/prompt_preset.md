# SFPromptPreset 节点使用指南

`SF Prompt Preset`：按 10 个分类（963 个预设 / 60 个分组）组合提示词的预设选择器。下拉选项为中文（欧美名人显示英文名），输出保持英文提示词。针对 Krea2 Turbo 优化（自然语言、无 SD 质量标签、分类正交），同时兼容 SD/Flux。

## 功能一览

- **组合拼接**：`input_text → 名人 → 表情 → 服装 → 单人动作 → 双人动作 → 环境 → 灯光 → 风格 → 镜头角度 → 镜头距离 → 镜头`
- **随机机制**：全分类随机（`随机`）/ 组内随机（`随机·组名`，如 `随机·NSFW`）/ `[选项A, 选项B]` 括号随机；同 seed 结果可复现（`IS_CHANGED` 随 seed 重跑）
- **pose/couple 互斥**：单人/双人动作同时启用时保留 pose（前端联动 + 后端兜底）
- **分组选择器弹窗**（节点上"☰ 预设"按钮）：
  - 10 个分类 tab（记忆上次打开的 tab 与组筛选）
  - group 筛选 chips（`🎲 全随机` / `全部` / 各分组）
  - 组标题行 `🎲 随机`（组内随机）/ 选项 hover 显示 description 预览 / 搜索过滤
- **动态说明**：combo 悬停显示当前选中预设的英文 description（写入 `widget.tooltip`）
- **热加载**：编辑 `data/prompt_presets.json` 后无需重启容器，自动重载

## 输入 / 输出

| 输入 | 说明 |
|---|---|
| `input_text` | 基础提示词（建议提供主体，如 "a beautiful woman"） |
| `seed` | 随机种子（0 ~ 2^64-1） |
| 10 个 combo | 各分类选择：`禁用` / `随机` / `随机·组名` / 具体预设 |

输出 2 个：

| 输出 | 类型 | 说明 |
|---|---|---|
| `combined_prompt` | STRING | 全部分类拼接（预设片段首字母小写化，与 Krea2 官方短语流一致） |
| `prompt_pack` | SF_PROMPT_PACK | 11 个分类文本打包（dict：Celebrity/Expression/Outfit/Pose/Couple Pose/Environment/Lighting/Style/Camera Angle/Camera Distance/Camera Lens），配合 **SFUnpackPromptPreset** 解包 |

> **破坏性变更提示**：旧版本输出 12 条 STRING，现改为 2 条（combined + pack），需在 `SFPromptPreset` 后接 `SFUnpackPromptPreset` 节点还原 11 条分类文本。打包值为运行时对象，**不可**接入 Primitive/保存类节点。

`SFUnpackPromptPreset` 输出 11 个 STRING（celebrity_text/expression_text/outfit_text/pose_text/couple_text/environment_text/lighting_text/style_text/camera_angle_text/camera_distance_text/camera_lens_text），顺序与旧版一致。

随机 seed 偏移：名人 `seed+1`、表情 `+2`、服装 `+3`、单人 `+4`、双人 `+5`、环境 `+6`、灯光 `+7`、风格 `+8`、角度 `+9`、距离 `+10`、镜头 `+11`。

## 分组体系（53 组）

| 分类 | 分组 |
|---|---|
| 名人（429） | 歌手/女演员/男演员/说唱歌手/喜剧演员/摔角手/模特/运动员/亚洲名人/其他 |
| 表情（38） | 开心微笑/诱惑挑逗/色气娇媚/情绪波动/高冷淡漠/害羞脸红 |
| 服装（71） | SFW / NSFW |
| 单人动作（76） | SFW / NSFW |
| 双人动作（47） | SFW / NSFW |
| 环境（111） | 自然风光/城市街景/室内空间/历史复古/科幻未来/恐怖暗黑/日系生活/私密场所 |
| 灯光（62） | 自然日光/人工光源/光效氛围/柔光漫射/人像布光/夜晚星光/黄昏日落 |
| 风格（49） | 写实 / 非写实 |
| 镜头角度（26） | 机位高度/俯仰角度/水平朝向/视角叙事/创意特殊 |
| 镜头距离（11） | 特写景别/近景景别/全景景别/远景景别 |
| 镜头（43） | 广角/标准/人像/长焦/微距/特殊/变焦/电影定焦/变形宽银幕/电影变焦/复古 |

## 数据文件格式（`data/prompt_presets.json`）

```json
{
  "Outfit": {
    "Black Bikini": {
      "name_zh": "黑色比基尼",                    // 下拉显示名（combo 持久值）
      "prompt": "Wearing a black bikini, ...",    // 英文输出（自然语言，无 SD 标签）
      "description": "Solid black bikini swimwear", // 悬浮说明（英文）
      "tags": ["bikini", "black", "beach"],        // 分类元数据
      "weight": 1.0,                              // 加权随机权重
      "group": "SFW"                              // 分组（弹窗筛选维度）
    }
  }
}
```

**添加自定义预设**：按上述结构在对应分类追加条目即可（热加载生效）。建议：
- `name_zh` 用中文（或欧美名人用英文名），同分类内唯一
- `prompt` 保持 Krea2 自然语言风格，**不要**内嵌他类职责（见"正交原则"）
- `group` 任意字符串（弹窗自动按出现顺序展示）

## 正交原则（分类职责单一）

各分类 prompt 不得内嵌其他分类的职责，否则组合会互相污染：

| 禁止 | 示例（错误） |
|---|---|
| 动作内嵌场景 | 姿势里写 "on a bed"（床由环境分类提供） |
| 动作内嵌灯光 | 姿势里写 "soft natural light"（光由灯光分类提供） |
| 服装内嵌灯光 | 服装里写 "soft warm lighting"（光由灯光分类提供） |
| 风格内嵌镜头参数 | 风格里写 "85mm lens, f/1.8"（镜头由镜头分类提供） |
| 名人内嵌风格 | 名人描述写 "photorealistic portrait"（风格由风格分类提供） |
| 动作内嵌裸体 | 姿势里写 "fully nude"（裸体由服装分类"全裸"预设控制） |

允许的例外：动作要素（靠墙必须有墙、坐床边必须有床、撑伞必须有伞）、场景固有照明（停车场荧光灯）、风格光效特征（巴洛克戏剧光）、服装场合属性（通勤装/沙滩装）。

> **NSFW 动作正交**：Pose/Couple Pose 的 NSFW 组只描述动作（姿态/情绪），不再内嵌 "fully nude" 等裸体词。需要裸体时在服装分类选"全裸"（或 NSFW 服装），配合任意动作使用。曾有的"全裸站立"预设因去裸体词后与"站立肖像"重复已删除，旧工作流中该值自动降级为空串（`VALIDATE_INPUTS` 接管校验）。

## 伦理提示

- 数据文件包含 **NSFW 成人内容预设**（NSFW 组），仓库分发注意许可与政策
- **名人 + NSFW 组合**会生成真实人物成人内容，存在肖像权与伦理风险，由使用者自行判断
- 亚洲名人（22 个）无社区实测依据，可自行生成测试验证（模型不认识的可在 JSON 增删，热加载生效）

## 部署

- 后端改动（`nodes/text/prompt_preset.py`、`nodes/model/krea2.py`）：同步 docker 目录并**重启容器**
- 前端改动（`web/prompt_preset.js`）：同步 docker 目录并**硬刷新**（Ctrl+Shift+R）
- 数据改动（`data/prompt_presets.json`）：同步 docker 目录，热加载生效

## 测试

```bash
python3 tests/test_prompt_preset.py   # 后端 214 项断言（mock 环境，无需 ComfyUI）
node tests/test_prompt_preset_js.js   # 前端 42 项断言（Node 直接运行）
```
