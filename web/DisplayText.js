import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const _DISPLAY_TEXT_NODES = new Set([
    "SFDisplayAny",
    "SFImageScalerForSDModels",
    "SFImageScalerByPixels",
    "SFImageScaleBySpecifiedSide",
    "SFComputeImageScaleRatio",
    "SFImageRotate",
    "SFTrimImageBorders",
    "SFAddImageBorder",
    "SFGetImageSize",
]);

app.registerExtension({
    name: "sfnodes.SFDisplayText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!_DISPLAY_TEXT_NODES.has(nodeData?.name)) {
            return;
        }

        if (nodeData.name === "SFDisplayAny") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                updateWidget(this, "displaytext", message["text"].join(""));
            };
        }
        
        switch (nodeData.name) {  
            case "SFImageScalerForSDModels":
            case "SFImageScalerByPixels":
            case "SFImageScaleBySpecifiedSide":
            case "SFComputeImageScaleRatio":
            case "SFImageRotate":
            case "SFTrimImageBorders":
            case "SFAddImageBorder":
            case "SFGetImageSize":
                const onExecutedImage = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecutedImage?.apply(this, arguments);
                    let value = message["width"].join("") + "x" + message["height"].join("");
                    if (nodeData.name === "SFGetImageSize") {
                        value += "_" + message["count"].join("");
                        value += "_" + message["min_dimension"].join("");
                        value += "_" + message["max_dimension"].join("");
                    }
                    
                    updateWidget(this, "return_text", value);
                };
                break;
        }

        // 辅助函数用于更新或创建widget
        function updateWidget(node, widgetName, value) {
            let textWidget = node.widgets && node.widgets.find(w => w.name === widgetName);
            if (!textWidget) {
                textWidget = ComfyWidgets["STRING"](node, widgetName, ["STRING", { multiline: true }], app).widget;
                textWidget.inputEl.readOnly = true;
                textWidget.inputEl.style.border = "none";
                textWidget.inputEl.style.backgroundColor = "transparent";
            }
            textWidget.value = value;
        }
                
    },
});
