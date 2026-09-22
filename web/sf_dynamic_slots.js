// ==========================================================================
// sf_dynamic_slots.js - 动态槽位公共库
// ==========================================================================
//
// 提供配置化的动态输入/输出槽位管理，供需要"连线自动增删槽位"的节点复用：
//
//   import { installDynamicSlots } from "./sf_dynamic_slots.js";
//   installDynamicSlots(node, {
//       inputPrefix: "text_",   // 动态输入前缀（省略则无动态输入）
//       inputStart: 1,          // 编号起点
//       inputCount: 16,         // 槽位上限（数量）
//       inputType: "STRING",    // 槽位类型
//       initialInputs: 1,       // 初始显示数量
//       inputMatch: (name) => /^[a-z]$/.test(name), // 可选：自定义匹配（优先级高于前缀）
//       outputPrefix: "value",  // 动态输出前缀（省略则无动态输出）
//       outputStart: 0,
//       outputCount: 20,
//       outputType: "*",
//       initialOutputs: 1,
//       nameFor: (cfg, count) => "abcdefghijklmnopqrstuvwxyz"[count], // 可选自定义命名
//   });
//
// 行为：
// - 创建时裁剪到初始数量并重置尺寸
// - 全部动态槽已连接 → 追加下一个槽位（上限内）
// - 断开尾部空槽 → 回收，直到只剩初始数量
// - 固定槽位（前缀不匹配）不受影响
//
// ==========================================================================

export const isSlotConnected = (slot) => {
    if (!slot) return false;
    if (slot.link !== null && slot.link !== undefined && slot.link !== -1) return true;
    return Array.isArray(slot.links) && slot.links.length > 0;
};

// 具名输入槽是否已接线（link 断开为 null，旧版可能 -1）。
export function isWired(node, name) {
    const inp = node?.inputs?.find((i) => i.name === name);
    return !!(inp && inp.link != null && inp.link !== -1);
}

// 具名输入槽的上游接线：{node, link, slot}（未接线/链接表缺项/上游缺失 → null）。
// slot = link.origin_slot，调用方可按上游输出槽名/序号判断语义。
export function linkedInput(node, name) {
    const inp = node?.inputs?.find((i) => i.name === name);
    if (!inp || inp.link == null || inp.link === -1) return null;
    let l = node?.graph?.links?.[inp.link];
    if (!l && typeof node?.graph?.links?.get === "function") l = node.graph.links.get(inp.link);
    if (!l) return null;
    const up = node.graph.getNodeById(l.origin_id);
    if (!up) return null;
    return { node: up, link: l, slot: l.origin_slot };
}

// 尽力读取已接线 INT 输入在编辑时的值（SFImageResize/SFImageCropExpandBrushMask
// 共用）：仅信任"恰好一个数值 widget"的上游（无歧义）；多数值 widget
// （seed/steps/cfg…）与 combo/字符串来源返回 null——调用方回退为"由接线输入
// 决定"而不是显示错误数字。数值按 int 截断（2.7 -> 2，镜像后端 int()；
// Math.round 会把 2.7 报成 3，预览对输出说谎）。
export function readWiredInt(node, name) {
    const li = linkedInput(node, name);
    if (!li) return null;
    const nums = (li.node.widgets || []).filter((x) => typeof x.value === "number");
    return nums.length === 1 && Number.isFinite(nums[0].value) ? Math.trunc(nums[0].value) : null;
}

// 返回与槽位数组其他元素不重名的名字（重名会破坏 prompt 序列化的输入键）。
export const uniqueName = (slots, selfIndex, base) => {
    if (!Array.isArray(slots) || slots.length === 0) return base;
    let name = base;
    let i = 2;
    while (slots.some((s, idx) => idx !== selfIndex && s && s.name === name)) {
        name = base + "_" + i++;
    }
    return name;
};

function installSide(node, side, cfg, nameFor) {
    const slotsProp = side === "input" ? "inputs" : "outputs";
    const isInput = side === "input";
    const match = cfg.match || ((name) => name.startsWith(cfg.prefix));

    const addSlot = (name) => {
        if (isInput) {
            node.addInput(name, cfg.type);
        } else {
            node.addOutput(name, cfg.type);
        }
    };

    const removeSlot = (slot) => {
        const slots = node[slotsProp];
        const idx = slots.indexOf(slot);
        if (idx >= 0) {
            if (isInput) {
                node.removeInput(idx);
            } else {
                node.removeOutput(idx);
            }
        }
    };

    const getDynamic = () =>
        (node[slotsProp] ?? []).filter((s) => s && match(s.name));

    const trimToInitial = () => {
        const dyn = getDynamic();
        while (dyn.length > cfg.initial) {
            removeSlot(dyn[dyn.length - 1]);
            dyn.pop();
        }
    };

    const resizeNode = () => {
        const sz = node.computeSize();
        if (sz) node.setSize([node.size[0] || sz[0], sz[1]]);
    };

    trimToInitial();
    resizeNode();

    return {
        handleChange(slotType, connected) {
            if (slotType !== (isInput ? 1 : 2)) return;
            const dyn = getDynamic();

            if (connected) {
                const allConnected = dyn.length > 0 && dyn.every(isSlotConnected);
                if (allConnected && dyn.length < cfg.count) {
                    addSlot(nameFor(cfg, dyn.length));
                }
            } else {
                const reversed = [...node[slotsProp]].reverse();
                for (const slot of reversed) {
                    if (!match(slot.name)) break;
                    if (!isSlotConnected(slot) && getDynamic().length > cfg.initial) {
                        removeSlot(slot);
                    } else {
                        break;
                    }
                }
                resizeNode();
            }
        },
    };
}

// 加载/粘贴恢复：configure 直赋 links 不触发 onConnectionsChange，按实际链接数
// 补齐到 linked+1（上限内），回收多余尾部空槽。包装 node.onAfterGraphConfigured。
// 单侧（input/output）恢复：按实际已连接数补齐到 linked+1（上限内），回收多余尾部空槽。
function recoverDynamicSide(node, side, cfg) {
    const slotsProp = side === "input" ? "inputs" : "outputs";
    const isInput = side === "input";
    const slots = node[slotsProp];
    if (!Array.isArray(slots)) return;
    const dynamic = () => (node[slotsProp] ?? []).filter((s) => s && s.name && cfg.match(s.name));
    let linked = 0;
    for (const slot of slots) {
        if (slot && slot.name && cfg.match(slot.name) && isSlotConnected(slot)) {
            linked += 1;
        }
    }
    const want = Math.min(Math.max(linked + 1, cfg.initial), cfg.max);
    // 补齐
    while (dynamic().length < want) {
        const name = cfg.prefix + (cfg.start + dynamic().length);
        if (isInput) node.addInput(name, cfg.type);
        else node.addOutput(name, cfg.type);
    }
    // 回收尾部空槽
    const reversed = [...node[slotsProp]].reverse();
    for (const slot of reversed) {
        if (!slot || !slot.name || !cfg.match(slot.name)) break;
        if (!isSlotConnected(slot) && dynamic().length > want) {
            if (isInput) node.removeInput(node[slotsProp].indexOf(slot));
            else node.removeOutput(node[slotsProp].indexOf(slot));
        } else {
            break;
        }
    }
}

// 与 installDynamicSlots 配套使用（同一 prefix/type/初始值与上限）。// config 同时给 inputPrefix 与 outputPrefix 时两侧都恢复（如循环节点 value 槽）。
export function installConfiguredSlotRecovery(node, config) {
    const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
        if (originalOnAfterGraphConfigured) {
            originalOnAfterGraphConfigured.apply(this, arguments);
        }
        if (!Array.isArray(this.inputs)) return;
        if (config.inputPrefix) {
            // 可选自定义匹配（优先级高于前缀）：与 installDynamicSlots 的 inputMatch 同款，
            // 用于前缀会误伤固定槽的场景（如 track_data 与 track_N 共用 track_ 前缀）。
            const inputMatch = config.inputMatch
                || ((name) => typeof name === "string" && name.startsWith(config.inputPrefix));
            recoverDynamicSide(this, "input", {
                prefix: config.inputPrefix,
                type: config.inputType || "*",
                initial: config.initialInputs ?? 1,
                max: config.inputCount ?? 20,
                start: config.inputStart ?? 0,
                match: inputMatch,
            });
        }
        if (config.outputPrefix) {
            const outputMatch = config.outputMatch
                || ((name) => typeof name === "string" && name.startsWith(config.outputPrefix));
            recoverDynamicSide(this, "output", {
                prefix: config.outputPrefix,
                type: config.outputType || "*",
                initial: config.initialOutputs ?? 1,
                max: config.outputCount ?? 20,
                start: config.outputStart ?? 0,
                match: outputMatch,
            });
        }
        this.setSize(this.computeSize());
    };
}

export function installDynamicSlots(node, config) {
    const originalOnConnectionsChange = node.onConnectionsChange;
    const nameFor = config.nameFor || ((cfg, count) => cfg.prefix + (cfg.start + count));

    const input = config.inputPrefix || config.inputMatch
        ? installSide(
              node,
              "input",
              {
                  prefix: config.inputPrefix ?? "",
                  start: config.inputStart ?? 0,
                  count: config.inputCount,
                  type: config.inputType || "*",
                  initial: config.initialInputs ?? 1,
                  match: config.inputMatch,
              },
              nameFor
          )
        : null;

    const output = config.outputPrefix || config.outputMatch
        ? installSide(
              node,
              "output",
              {
                  prefix: config.outputPrefix ?? "",
                  start: config.outputStart ?? 0,
                  count: config.outputCount,
                  type: config.outputType || "*",
                  initial: config.initialOutputs ?? 1,
                  match: config.outputMatch,
              },
              nameFor
          )
        : null;

    node.onConnectionsChange = function (type, index, connected, link_info, slot_info) {
        if (input) input.handleChange(type, connected);
        if (output) output.handleChange(type, connected);
        if (originalOnConnectionsChange) {
            originalOnConnectionsChange.apply(this, arguments);
        }
    };
}

// ── 固定输入槽的按需增删（模式/条件驱动的源输入显隐，SFQwenImage21PromptEnhancer 用）──
// 移除指定输入槽：优先原生 removeInput（会一并断开该槽连线），旧版回退断开 + splice。
export function removeInputAt(node, index) {
    if (typeof node?.removeInput === "function") {
        node.removeInput(index);
        return;
    }
    if (node?.inputs?.[index]?.link != null) {
        node.disconnectInput?.(index);
    }
    node?.inputs?.splice(index, 1);
}

// 增删中间输入槽后修正其余已连线输入的 link.target_slot（索引位移，否则连线错位）。
// graph 缺省取 node.graph；纯函数式传参便于测试。
export function syncInputLinkTargets(node, graph) {
    const g = graph || node?.graph;
    if (!g || !Array.isArray(node?.inputs)) return;
    node.inputs.forEach((input, index) => {
        if (input?.link == null) return;
        const link = g.links?.[input.link];
        if (link) link.target_slot = index;
    });
}
