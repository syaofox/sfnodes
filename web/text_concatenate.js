import { app } from "/scripts/app.js";

const MAX_INPUTS = 16;
const INITIAL_INPUTS = 1;

app.registerExtension({
    name: "sfnodes.TextConcatenate",

    nodeCreated(node) {
        if (node.comfyClass !== "SFTextConcatenate") return;

        const originalOnConnectionsChange = node.onConnectionsChange;

        const getDynamicInputs = () =>
            node.inputs.filter((inp) => inp.name.startsWith("text_"));

        const resizeNode = () => {
            const sz = node.computeSize();
            if (sz) node.setSize([node.size[0] || sz[0], sz[1]]);
        };

        const trimInputs = () => {
            const dynamic = getDynamicInputs();
            while (dynamic.length > INITIAL_INPUTS) {
                const idx = node.inputs.indexOf(dynamic[dynamic.length - 1]);
                node.removeInput(idx);
                dynamic.pop();
            }
        };

        trimInputs();
        resizeNode();

        node.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
            if (type !== 1) {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                return;
            }

            const dynamicInputs = getDynamicInputs();

            if (connected) {
                const allConnected = dynamicInputs.every(
                    (inp) => inp.link !== null && inp.link !== undefined,
                );
                if (allConnected) {
                    if (dynamicInputs.length >= MAX_INPUTS) return;
                    const newName = "text_" + (dynamicInputs.length + 1);
                    this.addInput(newName, "STRING");
                }
            } else {
                const reversed = [...node.inputs].reverse();
                for (const inp of reversed) {
                    if (!inp.name.startsWith("text_")) break;
                    if (
                        (inp.link === null || inp.link === undefined) &&
                        getDynamicInputs().length > INITIAL_INPUTS
                    ) {
                        const idx = node.inputs.indexOf(inp);
                        node.removeInput(idx);
                    } else {
                        break;
                    }
                }
                resizeNode();
            }

            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
        };
    },
});
