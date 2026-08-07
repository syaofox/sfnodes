// ==========================================================================
// sf_pause_text_lib.js - SFPauseText 纯函数库（prune + state）
// ==========================================================================
//
// 无 app/DOM 依赖，供主扩展 sf_pause_text.js 与 UI 模块 import，也供
// tests/ 复制为 .mjs 直接测试。
//
// 两部分：
//   1. state：节点状态存 node.properties.pauseTextState（gate/text/original），
//      随工作流保存（保留你的编辑是设计目标）。
//   2. prune：对 ComfyUI API prompt 对象（app.graphToPrompt().output 的
//      id -> {class_type, inputs}）按闸门生效模式修剪：
//      - pause    删除闸门下游（闸门是 OUTPUT_NODE，成为该分支终点）
//      - continue 跳过整个上游模型链，只从编辑文本跑其余；只删除会拉活被跳过
//                  上游的输出节点，无关分支照跑
//      - pass     不剪，整条工作流照跑
//
// ==========================================================================

// ── state（node.properties）─────────────────────────────────────────────
export const STATE_PROP = "pauseTextState";

// 持久化形状：
//   gate:     "pause"（默认）| "pass" | "keep"
//   text:     当前盒子内容（Continue 时输出；未接线/continue 时作为 PauseState
//             的盒子文本注入）
//   original: 模型最近一次文本（Revert 与 "edited" 指示用）
// text/original 有意序列化：重开工作流保留你的编辑。只在真实用户动作（打字、
// Revert）或一次 RUN（executed 事件）时变化——纯加载/恢复路径永不改动，
// 所以打开已保存工作流不会误标 "modified"。
export function getState(node) {
    node.properties = node.properties || {};
    let s = node.properties[STATE_PROP];
    if (!s || typeof s !== "object") {
        s = { gate: "pause", text: "", original: "" };
        node.properties[STATE_PROP] = s;
    }
    if (s.gate !== "pause" && s.gate !== "pass" && s.gate !== "keep") s.gate = "pause";
    if (typeof s.text !== "string") s.text = "";
    if (typeof s.original !== "string") s.original = "";
    return s;
}

export function setGate(node, gate) {
    const s = getState(node);
    s.gate = (gate === "pass" || gate === "keep") ? gate : "pause";
}

// 更新当前盒子文本（一次按键）。标记工作流已修改是正确行为——用户确实改了东西。
export function setText(node, text) {
    const s = getState(node);
    s.text = typeof text === "string" ? text : "";
}

// 一次 run 的新鲜模型捕获：替换盒子与 revert 基线。
export function setModelText(node, text) {
    const s = getState(node);
    const t = typeof text === "string" ? text : "";
    s.text = t;
    s.original = t;
}

// 把模型的原始文本放回去（Revert）。
export function revertText(node) {
    const s = getState(node);
    s.text = s.original;
}

// 盒子与模型最近文本是否不同。
export function isEdited(node) {
    const s = getState(node);
    return s.text !== s.original;
}

// ── prune（对 API prompt 对象操作，纯函数）─────────────────────────────
// ComfyUI prompt 输入链接的形式是 EXACTLY [originNodeId, originSlot]
export function isLink(v) {
    return Array.isArray(v) && v.length === 2
        && (typeof v[0] === "string" || typeof v[0] === "number")
        && typeof v[1] === "number";
}

// 从 prompt 构建 origin -> Set(consumerIds)
export function buildConsumers(output) {
    const consumers = new Map();
    for (const id in output) {
        const inputs = output[id]?.inputs;
        if (!inputs) continue;
        for (const k in inputs) {
            if (!isLink(inputs[k])) continue;
            const origin = String(inputs[k][0]);
            if (!consumers.has(origin)) consumers.set(origin, new Set());
            consumers.get(origin).add(String(id));
        }
    }
    return consumers;
}

// 从 startId 前向 BFS：所有下游可达节点集合（不含 start）
export function collectDownstream(consumers, startId) {
    const seen = new Set();
    const stack = [String(startId)];
    while (stack.length) {
        const cur = stack.pop();
        const next = consumers.get(cur);
        if (!next) continue;
        for (const c of next) {
            if (!seen.has(c)) { seen.add(c); stack.push(c); }
        }
    }
    return seen;
}

// 把 keep 中每个节点的全部祖先并入 keep
export function addAncestors(output, keep) {
    const stack = [...keep];
    while (stack.length) {
        const cur = stack.pop();
        const inputs = output[cur]?.inputs;
        if (!inputs) continue;
        for (const k in inputs) {
            if (!isLink(inputs[k])) continue;
            const origin = String(inputs[k][0]);
            if (output[origin] && !keep.has(origin)) {
                keep.add(origin);
                stack.push(origin);
            }
        }
    }
}

// 应用一个闸门的生效模式到 prompt `out`。
//   opts = { inputKey = "text", editedText = "" }
// `isOutput(classType)` 返回 true 表示 class_type 是 OUTPUT_NODE。Continue 只
// 删除其他输出节点；非输出节点留作无害孤儿（永不校验/运行），下游 Save 节点
// 因此保留完整生成元数据——与 Pause Image 同理。
export function applyGateMode(out, id, entry, mode, isOutput, HIDDEN_INPUT = "PauseState", opts = {}) {
    const inputKey = opts.inputKey || "text";
    const editedText = typeof opts.editedText === "string" ? opts.editedText : "";
    entry.inputs = entry.inputs || {};

    if (mode === "pause") {
        // 让 run 停在闸门：删除其下游全部，使闸门（OUTPUT_NODE）成为本分支终点。
        // 并行分支不受影响。
        const consumers = buildConsumers(out);
        const downstream = collectDownstream(consumers, id);
        for (const d of downstream) delete out[d];
        // 模式旁带上盒子文本：未接线的 pause 保留盒子内容。
        entry.inputs[HIDDEN_INPUT] = JSON.stringify({ mode: "pause", text: editedText });
    } else if (mode === "continue") {
        // 完全跳过上游，只从编辑文本跑其余。
        const gateSrc = isLink(entry.inputs[inputKey])
            ? [String(entry.inputs[inputKey][0]), Number(entry.inputs[inputKey][1])]
            : null;

        delete entry.inputs[inputKey];
        entry.inputs[HIDDEN_INPUT] = JSON.stringify({ mode: "continue", text: editedText });

        const consumers = buildConsumers(out);
        const downstream = collectDownstream(consumers, id);

        // 菱形重路由：闸门之后还有节点直接读闸门原文本源（gateSrc）的话，
        // 会把整个上游拉活。闸门现在发出的是同一份（编辑后的）文本，把这些
        // 精确匹配的链接改指向闸门自己的输出（slot 0），闸门之后的东西就不再
        // 触及闸门之前。
        if (gateSrc) {
            for (const dId of downstream) {
                const dInputs = out[dId]?.inputs;
                if (!dInputs) continue;
                for (const k in dInputs) {
                    const v = dInputs[k];
                    if (isLink(v) && String(v[0]) === gateSrc[0] && Number(v[1]) === gateSrc[1]) {
                        dInputs[k] = [String(id), 0];
                    }
                }
            }
        }

        const keep = new Set(downstream);
        keep.add(String(id));
        addAncestors(out, keep);

        // 删除哪些输出节点：只删会重新拉活被跳过上游（喂给闸门的模型链）的那些。
        // 无关输出分支（自有来源、不经过闸门上游）必须继续跑——Continue/Keep
        // 跳的是模型，不是图的其余部分。
        // upstream = gateSrc 的节点 + 其全部祖先（被跳过的链）。
        const upstream = new Set();
        if (gateSrc) { upstream.add(gateSrc[0]); addAncestors(out, upstream); }
        // pullsUpstream = 从 upstream 前向可达的一切（其消费者）；执行其中任何
        // 一个都会跑被跳过的模型。菱形重路由后再重建 consumers，被重路由的
        // 下游节点不再算作消费者。
        const postConsumers = buildConsumers(out);
        const pullsUpstream = new Set();
        const stack = [...upstream];
        while (stack.length) {
            const next = postConsumers.get(String(stack.pop()));
            if (!next) continue;
            for (const c of next) if (!pullsUpstream.has(c)) { pullsUpstream.add(c); stack.push(c); }
        }
        const canDetect = typeof isOutput === "function";
        for (const nid of Object.keys(out)) {
            const s = String(nid);
            if (keep.has(s)) continue;
            if (!pullsUpstream.has(s)) continue;   // 与闸门上游无关 -> 保留照跑
            if (!canDetect || isOutput(out[nid] && out[nid].class_type)) delete out[nid];
        }
    } else {
        // Pass：不剪，整条工作流照跑。未接线的情况携带盒子文本。
        entry.inputs[HIDDEN_INPUT] = JSON.stringify({ mode: "pass", text: editedText });
    }
}
