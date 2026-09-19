# AGENTS.md - sfnodes (ComfyUI Custom Node Pack)

## Project Overview

sfnodes 是一个 ComfyUI 自定义节点包，提供图像处理、人脸操作、遮罩编辑、文本处理、模型管理等增强功能。

ComfyUI 源码根目录即 `../..`（`custom_nodes/` 的父目录，含 `comfy/`、`nodes.py` 等，**仅为源码副本**，实际运行实例为 docker 部署——以实际挂载路径为准），可用于查阅 API 和参考实现。**不要尝试在本机启动 ComfyUI 或安装运行时依赖。**

**ComfyUI 前端（`comfyui-frontend-package` / `/scripts/app.js` 等）不在宿主机上**——宿主机没有 `web/` 前端源码与 `node_modules`。需查前端实现时在容器内查看：容器名 `comfyui-docker`，容器内代码位置 `/mnt/github/comfyui-docker`（如 `docker exec comfyui-docker cat /mnt/github/comfyui-docker/...`）。

## Architecture

```
sfnodes/
├── __init__.py      # 注册入口：NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS + WEB_DIRECTORY="web"
├── requirements.txt # Python 依赖（仅声明，不在本机安装）
├── nodes/           # 节点实现：face/ image/ mask/ model/ text/ utils/ inpaint/ latent/ video/ 子目录 + logic.py（循环/Any 打包）、workflow_routes.py
├── tools/           # 一次性脚本（extract_lora_diff.py 模型差异提 LoRA，自带 README，不进 requirements.txt）
├── sf_utils/        # 共享工具库（无状态纯函数为主）：image/mask 转换、lora_* 系列、resize_engine / dropdown / regional_engine / krea2_presets / disk_state / prompt_reader 等纯逻辑模块
├── web/             # 前端 JS Widget：sf_common.js（公共小工具/微工具 injectCSSOnce·sfToast·el·hideJsonWidget/强调色/LoRA 行名）+ sf_popup.js（弹层三件套）+ 各节点模块（单文件或 *_lib/*_core/_ui 多模块系列）
├── data/            # 静态数据（prompt_presets.json、styles/ 内置风格库+samples、anime_char/characters/face_distance 子数据，含 CSV/字体）
├── tests/           # 前端/后端模拟测试（Node/Python 直接运行，无测试框架）
├── user/            # 用户数据目录回退占位（真源 `<ComfyUI user dir>/sfnodes`，见 user/sfnodes/README.md；仅 README 入库）
└── doc/             # 文档：architecture.md 逐文件细目 / experience/ 经验归档（README 索引 + 七主题文件）
```

**逐文件职责与机制说明见 `doc/architecture.md`**——新增/删除文件必须同步其条目。

## Node Registration & Class Convention

根 `__init__.py` 两字典同步注册：

- `NODE_CLASS_MAPPINGS`: 键 `"SF<ClassName>"`，值为类本身（现 198 键全部带 SF 前缀；新增一律带前缀）
- `NODE_DISPLAY_NAME_MAPPINGS`: 键同上，显示名 `"SF <Display Name>"`

```python
class SFMyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}

    RETURN_TYPES = ("TYPE",)
    RETURN_NAMES = ("name",)
    FUNCTION = "execute"          # 执行方法名
    CATEGORY = "sfnodes/<group>"  # 统一 sfnodes/<功能组>：face/image/mask/model/text/video/utils/logic/inpaint/latent
    DESCRIPTION = "..."           # 必填

    def execute(self, ...):
        return (result,)
```

## Dependencies & ComfyUI APIs

- 运行时第三方依赖见 `requirements.txt`（带用途注释的完整清单在 `doc/architecture.md`）；`torch/torchvision` 由 ComfyUI 运行时提供，不入 requirements。**新增依赖必须同步 requirements.txt。**
- ComfyUI 运行时提供的常用 import 清单见 `doc/architecture.md`；易错点：
  - `nodes.NODE_CLASS_MAPPINGS` **运行时才包含全部自定义节点**——函数内 import 最安全
  - `ExecutionBlocker` 官方位置 `comfy_execution.graph_utils`（graph.py 只是 re-export）；DYNPROMPT 隐藏输入对象是 `comfy_execution.graph.DynamicPrompt`

## Code Style

- Python 3.10+，无类型注解强制；用 `_CATEGORY` 模块级常量定义分类前缀；工具函数放 `sf_utils/`（无状态纯函数），节点放 `nodes/<组>/`
- JS Widget 放 `web/`；**动手前先查公共模块复用**：前端 `sf_common.js`（小工具/微工具 injectCSSOnce·sfToast·el·hideJsonWidget/强调色/LoRA 行名）、`sf_dynamic_slots.js`（动态槽位）、`sf_popup.js`（新弹层优先，见 experience/patterns.md §26）、`sf_crop_framework.js`（编辑器框架）；后端 `disk_state.py` 与 `sf_utils/` 各纯逻辑模块。有公共实现必须复用，**禁止内联副本**
- **纯模块边界**：`*_lib.js` / `*_core.js` / `sf_markdown.js` 纯逻辑模块（无 app 依赖、可拷 .mjs 单测）**不得 import sf_common.js**（它依赖 /scripts/app.js）；通用纯函数共享放无依赖模块
- **注册规范**（由 `tests/check_web_imports.py` 固化校验）：registerExtension 文件必须直接 `import { app } from "/scripts/app.js"`（禁相对路径）、扩展注册名 `sfnodes.*` 前缀、相对导入目标必须存在；新增 web 模块需加入该脚本 MODS 列表
- 子目录 `__init__.py` 为空（`nodes/utils/` 无此文件走 namespace package），仅根 `__init__.py` 负责注册

## Development Rules

1. **不要启动 ComfyUI 或运行 pip install**——本机仅代码编辑环境；一次性生成工具装 `/tmp` 用，产物内联进 web/ 模块，不得进 requirements.txt
2. 可阅读 `../..` 源码理解 API 与参考实现；**前端代码不在宿主机**，查前端实现须在容器 `comfyui-docker` 的 `/mnt/github/comfyui-docker` 内查看（见 Project Overview）
3. 新增节点同步更新根 `__init__.py` 两个注册字典
4. 新增依赖同步 `requirements.txt`
5. 实现类 PascalCase，注册键 `"SF"` 前缀
6. 图像张量 `[B,H,W,C]`；遮罩张量 `[B,H,W]`
7. JS Widget 用 `app.registerExtension` 注册；纯工具模块仅 export 函数由使用者 import
8. 动态槽位类 JS 复用 `web/sf_dynamic_slots.js` 公共库，勿重复实现
9. 部署为 docker：宿主机工作副本与运行实例是**独立副本**，改动须同步到实际挂载目录（当前 `/mnt/github/comfyui-docker/custom_nodes/sfnodes`，以实际挂载为准）；后端改动需重启容器——**重启会打断用户任务，必须先征得用户同意**；`web/` JS 改动同步后需浏览器硬刷新（Ctrl+Shift+R）才生效
10. **实机调试**：禁止浏览器/浏览器自动化访问用户 ComfyUI 页面（干扰用户 tab 与工作流）。① **后端/API 层可自行调试**：`docker exec comfyui-docker curl -s http://127.0.0.1:8188/...` 或宿主 `curl http://localhost:8188/...`——`GET /object_info/{节点}` 验注册与输入/输出槽、`/api/sfnodes/...` 路由自测、`POST /prompt` 跑轻量测试工作流（不加载大模型）、`docker logs` 查错、`docker exec ... python3` 验容器运行时行为；测试数据勿污染用户数据/队列 → experience/platform.md §106；② **前端/UI 层仍须用户配合**：分段 console 诊断脚本（版本检查→节点状态→事件日志→数据层→UI 层）交用户执行反馈，节点请用户 UI 添加（新版前端无 graph.createNode）→ experience/platform.md §2.9
11. **新增节点/功能前先查复用**（见 Code Style），禁止内联副本——语义分叉是 bug 温床。去重/重构注意：① 独立语句的包装块不在函数体内按名删除会漏；② 文件已有某模块 import 时脚本补 import 可能跳过致缺符号（被 try/catch 吞掉极难排查）；③ ESM 结构错误用 `node --input-type=module --check < file` 验证
12. 新增/删除 py/js 文件同步 `doc/architecture.md` 条目；沉淀新经验按主题写入 `doc/experience/` 对应主题文件（下一个全局 §N）并同步 README.md 索引表；**确属新类别且现有主题均不适配时可新建主题文件**（英文短名对齐节点族，标题注明所含章节）

## Workflow（任务流程）

1. **沟通语言**：中文
2. **前置评估**（先回答再提方案）：开工前 `git status` 确认基线，有未提交改动先说明；说明影响范围/副作用/性能风险、最优解与替代方案、复用检索结论（复用了哪个公共模块、哪些为新增及不可复用理由）；bug 修复/文案/注释类小改动可简化为一句检索结论
3. **确认拦截**：方案需详细解释并经用户确认后方可编码
4. **最小改动**：只改任务范围内代码，不顺手重构无关部分；发现的无关问题先报告，过时/错误注释可顺手修正
5. **后置复查**：按「Testing」跑三条回归；核对两注册字典键一致；`git diff` 对照开工基线逐文件复查，确保无错漏与无关改动；未经明确要求不执行 git commit/push
6. **硬约束查询**：涉及 ComfyUI API/节点注册/工具函数/JS Widget 的改动，先读本文件对应章节与 `doc/architecture.md` 逐文件细目，遵循既有约定
7. **不确定时**：务必追问，禁止猜测

## Testing

本项目无自动化测试框架。验证方式：

- 静态检查：两注册字典键一致、所有节点类正确导入、requirements.txt 含全部第三方依赖
- 后端模拟测试：mock torch/comfy.utils 加载节点模块，FakeDynPrompt 断言图结构与返回值（循环节点有先例）
- 前端模拟测试：无 DOM 依赖的公共库复制 `.mjs` 用 Node 直跑（FakeNode + 事件序列，tests/ 有先例）
- 快速回归命令（文件自含断言，任一失败非零退出即停）：
  - 后端：`for f in tests/test_*.py; do python3 "$f" || break; done`
  - 前端：`for f in tests/test_*.js tests/test_*.mjs; do node "$f" || break; done`
  - 静态一致性：`python3 tests/check_web_imports.py`

## 经验归档（按主题，细则见 `doc/experience/`）

> 本包所有具体机制、节点族约定与踩坑经验统一归档在 `doc/experience/`（七主题文件：platform / patterns / nodes-text(简写 text) / nodes-image(image) / nodes-lora(lora) / nodes-video(video) / apps）。全局章节号 §N 与文件的映射见 `doc/experience/README.md`；**动手改动某功能前，先按 §N 到对应主题文件查阅**，本文档不重复收录细节。
>
> 收录规则：节点专属机制与横切踩坑一律写进对应主题文件（下一个全局 §N，编号只增不复用、不重排；允许最小事实订正与旧节「已被 §N 取代」标注，细则见 `doc/experience/README.md`「维护规则」），并同步 `doc/experience/README.md` 索引表；本文档只保留通用规则（「Node Registration & Class Convention」「Dependencies & ComfyUI APIs」「Code Style」「Development Rules」「Workflow」「Testing」）与本节引用。
