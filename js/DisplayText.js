import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "sfnodes.DisplayText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("fnodes")) {
            return;
        }
        
        switch (nodeData.name) {  
            case "ImageScalerForSDModels":
            case "ImageScalerByPixels":
            case "ImageScaleBySpecifiedSide":
            case "ComputeImageScaleRatio":
            case "ImageRotate":
            case "TrimImageBorders":
            case "AddImageBorder":
            case "GetImageSize":           
                const onExecutedImage = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecutedImage?.apply(this, arguments);
                    let value = message["width"].join("") + "x" + message["height"].join("");
                    if (nodeData.name === "GetImageSize") {
                        value += "x" + message["count"].join("");
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
