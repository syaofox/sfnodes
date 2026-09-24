// 原生 LoadImage filename 输出开关（后端 sf_utils/native_load_image_filename.py
// 的注册表补丁开关）：设置项与 BrowseButton 同段（sfnodes.LoadImage.*），默认开。
// 后端在启动时读取该值决定是否给原生 LoadImage 追加 filename(STRING) 输出，
// 改动需重启 ComfyUI 生效——切换时弹提示，避免"设置变了但节点没变"的困惑。
import { app } from "/scripts/app.js";
import { sfToast } from "./sf_common.js";

export const LOAD_IMAGE_FILENAME_SETTING = "sfnodes.LoadImage.FilenameOutput.Enabled";

const TOAST_TAG = "SF LoadImage filename";

let _registered = false;

export function registerLoadImageFilenameSetting() {
    if (_registered) return;
    _registered = true;
    try {
        app.ui.settings.addSetting({
            id: LOAD_IMAGE_FILENAME_SETTING,
            name: "SF: add filename output to native LoadImage (restart required)",
            type: "boolean",
            defaultValue: true,
            onChange: () => {
                sfToast({
                    summary: TOAST_TAG,
                    detail: "设置已保存，重启 ComfyUI 后生效",
                    severity: "info",
                    life: 5000,
                    fallbackTag: TOAST_TAG,
                });
            },
        });
    } catch { /* 设置系统不可用则退化为默认值（开） */ }
}

app.registerExtension({
    name: "sfnodes.native_load_image_filename",
    init() {
        registerLoadImageFilenameSetting();
    },
});
