import { app } from "/scripts/app.js";

// Some fragments of this code are from https://github.com/LucianoCirino/efficiency-nodes-comfyui

// 节点类型与对应widget配置
const NODE_WIDGETS_CONFIG = {
    "SFFaceMorph": {
        simpleWidgets: ["landmark_type", "align_type", "onnx_device"]
    },
    "SFGeneratePreciseFaceMask": {
        simpleWidgets: ["grow", "grow_percent", "grow_tapered"]
    },
    "SFGenerateRegionFaceMask": {
        simpleWidgets: ["grow", "grow_percent", "grow_tapered"]
    }
};

function inpaintCropAndStitchHandler(node) {
    const nodeConfig = NODE_WIDGETS_CONFIG[node.comfyClass];
    if (!nodeConfig) return;

    // 处理简单的widget（默认隐藏）
    if (nodeConfig.simpleWidgets) {
        nodeConfig.simpleWidgets.forEach(widgetName => {
            const widget = findWidgetByName(node, widgetName);
            if (widget) {
                toggleWidget(node, widget);
            }
        });
    }

    // 处理条件widget
    if (nodeConfig.conditionalWidgets) {
        nodeConfig.conditionalWidgets.forEach(condConfig => {
            // 检查条件是否满足
            if (condConfig.condition(node)) {
                // 处理所有满足条件的widget
                condConfig.widgets.forEach(widgetConfig => {
                    const widget = findWidgetByName(node, widgetConfig.name);
                    if (widget) {
                        // 处理显示条件 - 可以是布尔值或函数
                        const showWidget = typeof widgetConfig.show === 'function' 
                            ? widgetConfig.show(node) 
                            : widgetConfig.show;
                        toggleWidget(node, widget, showWidget);
                    }
                });
            }
        });
    }

    return;
}

let origProps = {};

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const doesInputWithNameExist = (node, name) => {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
};

const HIDDEN_TAG = "tschide";
// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
            
    // Store the original properties of the widget if not already stored
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }       
        
    const origSize = node.size;

    // Set the widget type and computeSize based on the show flag
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    
    // Recursively handle linked widgets if they exist
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show, ":" + widget.name));
        
    // Calculate the new height for the node based on its computeSize method
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}   

app.registerExtension({
    name: "sfnodes.showcontrol",
    nodeCreated(node) {
        if (!node.comfyClass.startsWith("SFInpaint") && !NODE_WIDGETS_CONFIG[node.comfyClass]) {
            return;
        }

        inpaintCropAndStitchHandler(node);
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists 
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value') || 
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(w), 'value');

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else { 
                        widgetValue = newVal;
                    }

                    inpaintCropAndStitchHandler(node);
                }
            });
        }
    }
});
