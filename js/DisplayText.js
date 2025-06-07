import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "sfnodes.SFDisplayText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("sfnodes")) {
            return;
        }

        if (nodeData.name === "SFDisplayAny") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
					for (let i = 1; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = 1;
				}

                // Check if the "text" widget already exists.
                let textWidget = this.widgets && this.widgets.find(w => w.name === "displaytext");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }
                textWidget.value = message["text"].join("");
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
