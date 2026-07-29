import { app } from "/scripts/app.js";

function getWorkflowName() {
    const title = document.title;
    if (title && title.endsWith(" - ComfyUI")) {
        return title.slice(0, -" - ComfyUI".length);
    }
    if (app.ui?.titleWidget?.value) {
        return app.ui.titleWidget.value;
    }
    const tabEl = document.querySelector(".workflow-tabs-container span");
    if (tabEl?.textContent) {
        return tabEl.textContent.trim();
    }
    return "Untitled";
}

const workflowNameNodes = new Set();
let workflownameHooked = false;

app.registerExtension({
    name: "sfnodes.WorkflowName",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SFWorkflowName") return;

        if (!workflownameHooked) {
            workflownameHooked = true;
            const orig = app.queuePrompt;
            app.queuePrompt = function (...args) {
                for (const node of workflowNameNodes) {
                    if (node.mode === LiteGraph.NEVER) continue;
                    const widget = node.widgets?.find(w => w.name === "workflow_name");
                    if (widget) {
                        widget.value = getWorkflowName();
                    }
                }
                return orig?.apply(app, args);
            };
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            workflowNameNodes.add(this);
            const widget = this.widgets?.find(w => w.name === "workflow_name");
            if (widget) {
                widget.value = getWorkflowName();
                if (widget.inputEl) {
                    widget.inputEl.readOnly = true;
                    widget.inputEl.style.border = "none";
                    widget.inputEl.style.backgroundColor = "transparent";
                    widget.inputEl.style.cursor = "default";
                    widget.inputEl.style.color = "var(--input-text, #ccc)";
                }
            }
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            workflowNameNodes.delete(this);
            onRemoved?.apply(this, arguments);
        };
    }
});
