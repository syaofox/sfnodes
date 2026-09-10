# 角色库数据格式（SFCharacterSelect）

每角色一条 JSON（数组），字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 角色名（选择键，全库唯一） |
| `prompt` | 否 | 角色提示词（选中后输出，可前端手改覆盖） |
| `face` | 否 | 脸部特写图相对路径 |
| `half` | 否 | 半身像相对路径 |
| `full` | 否 | 全身像相对路径 |

图片路径相对角色库目录解析（`samples_id_chara/<库名>/...` 独立目录，
勿与风格库 `samples/` 混放）；缺图的分镜输出 1×1 黑占位，不影响其他分镜。

内置库放本目录 `character_*.json`；用户自定义放
`<ComfyUI user>/sfnodes/characters/*.json`，同名覆盖内置。
`example.json` 为格式示例（无配图，各分镜占位）。
