// SF Load Images Path 前端扩展：刷新 user/sfnodes/images/ 子目录列表

import { app } from "/scripts/app.js";

async function fetchFolders() {
    try {
        const resp = await fetch("/api/sfnodes/images_path/folders");
        if (resp.ok) {
            const data = await resp.json();
            return Array.isArray(data?.folders) ? data.folders : [];
        }
    } catch (e) {
        console.error("[SFLoadImagesPath] Failed to load folders:", e);
    }
    return [];
}

function refreshFolderWidget(node) {
    const folderWidget = node.widgets?.find((w) => w.name === "folder");
    if (!folderWidget) return;
    fetchFolders().then((folders) => {
        const values = folders.length > 0 ? folders : ["default"];
        const current = folderWidget.value;
        folderWidget.options.values = values;
        if (!values.includes(current)) {
            folderWidget.value = values[0];
        }
        node.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "sfnodes.load_images_path",

    nodeCreated(node) {
        if (node?.comfyClass !== "SFLoadImagesPath") return;

        node.addWidget("button", "刷新目录列表", null, () => {
            refreshFolderWidget(node);
        });
    },
});
