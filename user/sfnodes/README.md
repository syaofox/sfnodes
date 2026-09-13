# user/sfnodes 数据目录说明

本目录是 sfnodes 的用户数据统一目录（`sf_utils/disk_state.py::sf_user_dir()` 单源，
即 `<ComfyUI user dir>/sfnodes`；本包内 `user/` 仅为拿不到 `folder_paths` 时的回退占位）。
docker 部署下 bind mount 存活，容器重建不丢失。目录与文件组织约定如下：

## 根 JSON（机器级，全局共享，Lock + 原子写 + mtime 缓存）

| 文件 | 归属节点/模块 |
|---|---|
| `lora_triggers.json` | SFLoraStack / LoRA 加载器系（`sf_utils/lora_notes.py` 统一网关） |
| `lora_presets.json` | SFLoraStack 复合预设（`sf_utils/lora_presets.py`） |
| `dmodels.json` | SF Load Diffusion Model（diffusion 域，`sf_utils/lora_routes.py` 数据域分派；**懒创建：只看信息面板不生成，首次保存描述/预览才落盘**） |
| `civitai.json` | Civitai 账户（key/host/adult_thumbs，key 永不离开服务器） |
| `krea2_presets.json` / `interrogator_presets.json` | Krea2 / 反推预设（`sf_utils/krea2_presets.py` register 范式） |
| `text_presets.json` | SFTextPreset（`sf_utils/text_presets.py`） |
| `crop_expand_presets.json` | SFImageCropExpand 自定义比例（`sf_utils/crop_expand_presets.py`） |

## 子目录

| 目录 | 归属 | 命名规范 |
|---|---|---|
| `styles/*.json` + `samples/` | SFStylesSelector（内置 `data/styles/` 同名覆盖） | 风格库；证件照模板用 `id_` 前缀 JSON + 独立 `samples_id_clothing/`，禁止与 `samples/` 混放 |
| `characters/character_*.json` + `samples_id_chara/<库>/` | SFCharacterSelect | 库文件必须 `character_` 前缀；角色图放对应该库的子目录 |
| `lora_previews/` | LoRA 域预览图 | 按 LoRA 键 sha1 命名，勿手改 |
| `previews_model/` | diffusion 域预览图（与 `lora_previews/` 物理隔离防撞槽） | 同上；**查看面板即创建目录**，属正常 |
| `lut/*.cube` | SFLoadLUT / SFExtractLUT | LUT 文件；`filename` 自由输入经 `sanitize_filename` 净化 + 强制 `.cube` |
| `prompt/<子目录>/*.txt` | 历史遗留（2026-09 起 SF Prompt Batcher 改读 `output/prompt/`；本目录保留但节点不再读取） | 新提示词建到 `output/prompt/<子目录>/`；空子目录请删除 |
| `images/<子目录>/` | 历史遗留（2026-09 起 SFLoadImagesPath 不再读取；批量入口改用 `input/` 子目录或 Path Mode 绝对路径） | 保留在磁盘，仅作归档；批量加载大目录仍建议按主题拆分（整目录加载模式，单目录过大拖慢执行） |

## 不变式

- 新增用户数据一律经 `sf_user_dir()` 读写，禁止各写一份取径（反例已收敛：`nodes/image/lut.py`）。
- 个人训练集放 `input/` 子目录（SFLoadImagesPath 两源之一）或用 Path Mode 绝对路径；废弃目录直接删除，不留孤儿（2026-09 已清理无代码引用的 `face_pieces/`，并移除 `images` 数据源）。
