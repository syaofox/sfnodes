## 任务目标
[需求] <!-- 示例：新增 SFImageFoo 节点，输入 IMAGE + 阈值，实现二值化输出 MASK；含前端预览与隐藏状态持久化 -->
<!-- 模板说明：此文件为任务发起模板，发起新任务时将 [需求] 替换为具体描述；保留“项目背景/执行约束”结构供复用，示例行可删 -->

## 项目背景
- 语言/框架：Python 3.10+, ComfyUI 自定义节点包
- 依赖管理：`requirements.txt`（仅声明，不在本机安装）
- 运行时环境：ComfyUI（源码根目录在 `../..`，可查阅 API，**不要启动或安装依赖**）
- 张量格式：图像 `[B, H, W, C]`，遮罩 `[B, H, W]`（ComfyUI 标准）
- 前端：JS Widget 使用 `app.registerExtension`（LiteGraph API），放在 `web/` 目录；根 `__init__.py` 需声明 `WEB_DIRECTORY = "web"`

## 执行约束
0. **优先使用 codebase-memory 图搜索**（`search_graph`、`trace_path`、`get_code_snippet`）代替 grep/glob 查找函数、类、节点定义和调用关系。找不到时再回退 grep/glob。
1. **沟通语言**：中文。
2. **前置评估**（回答以下几个问题后再提出方案）：
   - 开工前运行 `git status` 确认工作区基线；若已有未提交改动，先向我说明。
   - 修改影响范围？有无副作用？会不会影响性能或导致相关功能受损？有没有过度想象？
   - 最优解？替代方案利弊？
   - 有无可直接复用的公共实现？
3. **复用优先，禁止重复造轮子**：提出方案前必须先检索项目既有实现——后端 `sf_utils/` 纯逻辑模块与 `disk_state.py`，前端 `sf_common.js`、`sf_dynamic_slots.js`、`sf_popup.js`、`sf_crop_framework.js` 等公共模块与功能相近的既有节点（**完整清单以 `AGENTS.md`「Code Style」为准**，此处仅为常用示例）。有公共实现必须 import 复用，禁止内联副本。方案中需逐项说明：哪些部分复用了哪个现有模块、哪些为新增及不可复用的理由；bug 修复/文案/注释类小改动可简化为一句检索结论。
4. **确认拦截**：方案需详细解释并经我确认，方可实施编码。
5. **代码质量**：
   - 遵循 `AGENTS.md` 中的节点注册规范（`__init__.py` 两个字典同步添加键）。
   - 遵循 `AGENTS.md` 中的 CATEGORY 命名规范（`sfnodes/<功能组>`）。
   - 新增/修改函数保持项目代码风格（无类型注解强制、使用 `_CATEGORY` 常量、纯函数工具等）。
   - 新增节点必须添加 `DESCRIPTION` 类属性。
   - 新增依赖必须同步更新 `requirements.txt`。
   - **能测则测**：新增纯函数/mock 可测逻辑时，按 `AGENTS.md`「Testing」先例补 `tests/test_*` 模拟测试并纳入快速回归循环。
   - **最小改动原则**：只改任务范围内代码，不顺手重构无关部分；发现的无关问题先报告，经我确认再处理（过时/错误注释的顺手修正不受此限）。
   - **新增/修改节点或功能后同步文档**：根 `__init__.py` 两字典；新增/删除 py/js 文件同步 `doc/architecture.md` 对应条目；新经验按主题写入 `doc/experience/` 对应主题文件（platform/patterns/nodes-text/nodes-image/nodes-lora/apps，下一个全局 §N）并同步该目录 README.md 索引表，再在 `AGENTS.md` 经验摘要增补/更新对应的索引行。**仅当确属新类别、现有主题均不适配时，才新建主题文件**（英文短名对齐节点族/领域，标题注明所含 § 章节，README 索引表加行）——勿把无关章节塞进既有文件。
   - 发现过时或错误的注释应一并修正。
   - JS Widget 遵循 `app.registerExtension` 注册方式。
   - 子目录 `__init__.py` 为空，仅根 `__init__.py` 负责注册。
6. **后置处理**：
   - 运行 `AGENTS.md`「Testing」的三条快速回归命令（静态一致性 / 后端 / 前端），任一失败先修复再汇报结果。
   - 检查 `NODE_CLASS_MAPPINGS` 与 `NODE_DISPLAY_NAME_MAPPINGS` 键是否一致。
   - 确认所有节点类在根 `__init__.py` 中正确导入。
   - 若新增/删除了 py/js 文件，确认 `doc/architecture.md` 条目已同步；若新增了 `doc/experience/` 章节或新建了主题文件，确认 README.md 索引表已加行且 `AGENTS.md` 经验摘要索引行已增补。
   - 复查无与公共模块重复的内联实现。
   - 通过 `git diff` 对照开工基线逐文件复查所有修改，确保无错漏且不含无关改动。
   - 未经我在任务目标中明确要求，不执行 git commit/push。
7. **硬约束查询**：涉及 ComfyUI API、节点注册、工具函数、JS Widget 等改动时，务必先阅读 `AGENTS.md` 中的对应章节与 `doc/architecture.md` 逐文件细目，遵循既有约定。
8. **不确定时**：务必追问，禁止猜测。
9. **实际环境验证**：需要真实环境验证前端行为时，编写分段 console 诊断脚本（版本检查 → 节点状态 → 事件日志包装 → 数据层 → UI 层）交用户执行并反馈输出，流程与模板见 `doc/experience/platform.md` §2.9 与 `AGENTS.md` 经验摘要。**禁止自行浏览器访问 ComfyUI**（会 404 且可能干扰用户运行中的工作流），也不要程序化创建节点（`graph.createNode` 在新版前端不可用，节点请用户用 UI 添加）。
