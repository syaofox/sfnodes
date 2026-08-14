## 任务目标
[需求]

## 项目背景
- 语言/框架：Python 3.10+, ComfyUI 自定义节点包
- 依赖管理：`requirements.txt`（仅声明，不在本机安装）
- 运行时环境：ComfyUI（源码根目录在 `../..`，可查阅 API，**不要启动或安装依赖**）
- 张量格式：图像 `[B, H, W, C]`，遮罩 `[B, H, W]`（ComfyUI 标准）
- 前端：JS Widget 使用 `app.registerExtension`（LiteGraph API），放在 `web/` 目录；根 `__init__.py` 需声明 `WEB_DIRECTORY = "web"`

## 执行约束
0. **优先使用 codebase-memory 图搜索**（`search_graph`、`trace_path`、`get_code_snippet`）代替 grep/glob 查找函数、类、节点定义和调用关系。找不到时再回退 grep/glob。
1. **沟通语言**：中文。
2. **前置评估**（回答以下几个问题后再提交方案）：
   - 修改影响范围？有无副作用？会不会影响性能或导致相关功能受损？有没有过度想象？
   - 最优解？替代方案利弊？
3. **确认拦截**：方案需详细解释并经我确认，方可实施编码。
4. **代码质量**：
   - 遵循 `AGENTS.md` 中的节点注册规范（`__init__.py` 两个字典同步添加键）。
   - 遵循 `AGENTS.md` 中的 CATEGORY 命名规范（`sfnodes/<功能组>`）。
   - 新增/修改函数保持项目代码风格（无类型注解强制、使用 `_CATEGORY` 常量、纯函数工具等）。
   - 新增节点必须添加 `DESCRIPTION` 类属性。
   - 新增依赖必须同步更新 `requirements.txt`。
   - **新增/修改节点或功能后同步文档**：根 `__init__.py` 两字典、`AGENTS.md` 架构树与经验摘要（或归档至 `doc/experience.md` 并同步其目录锚点）、相关专项文档（如 `doc/prompt_preset.md` 在预设数据改动时）。
   - 发现过时或错误的注释应一并修正。
   - JS Widget 遵循 `app.registerExtension` 注册方式。
   - 子目录 `__init__.py` 为空，仅根 `__init__.py` 负责注册。
5. **后置处理**（手动检查，无自动化测试框架）：
   - 检查 `NODE_CLASS_MAPPINGS` 与 `NODE_DISPLAY_NAME_MAPPINGS` 键是否一致。
   - 确认所有节点类在根 `__init__.py` 中正确导入。
   - 若新增了 `doc/experience.md` 章节，确认「目录」锚点已同步。
   - 复查所有修改，确保无错漏。
6. **硬约束查询**：涉及 ComfyUI API、节点注册、工具函数、JS Widget 等改动时，务必先阅读 `AGENTS.md` 中的对应章节，遵循既有约定。
7. **不确定时**：务必追问，禁止猜测。
8. **实际环境验证**：需要真实环境验证前端行为时，编写分段 console 诊断脚本（版本检查 → 节点状态 → 事件日志包装 → 数据层 → UI 层）交用户执行并反馈输出，流程与模板见 `doc/experience.md` 前端机制 §9 与 `AGENTS.md` 经验摘要。**禁止自行浏览器访问 ComfyUI**（会 404 且可能干扰用户运行中的工作流），也不要程序化创建节点（`graph.createNode` 在新版前端不可用，节点请用户用 UI 添加）。
