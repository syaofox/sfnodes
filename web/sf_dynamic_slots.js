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
// 与 installDynamicSlots 配套使用（同一 prefix/type/初始值与上限）。
export function installConfiguredSlotRecovery(node, config) {
    const prefix = config.inputPrefix;
    const type = config.inputType || "*";
    const initial = config.initialInputs ?? 1;
    const max = config.inputCount ?? 20;
    const start = config.inputStart ?? 0;
    // 可选自定义匹配（优先级高于前缀）：与 installDynamicSlots 的 inputMatch 同款，
    // 用于前缀会误伤固定槽的场景（如 track_data 与 track_N 共用 track_ 前缀）。
    const match = config.inputMatch || ((name) => typeof name === "string" && name.startsWith(prefix));

    const originalOnAfterGraphConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
        if (originalOnAfterGraphConfigured) {
            originalOnAfterGraphConfigured.apply(this, arguments);
        }
        if (!this.inputs) return;
        let linked = 0;
        for (const slot of this.inputs) {
            if (slot && slot.name && match(slot.name) && isSlotConnected(slot)) {
                linked += 1;
            }
        }
        const want = Math.min(Math.max(linked + 1, initial), max);
        // 补齐
        let dynamic = this.inputs.filter((s) => s && s.name && match(s.name));
        while (dynamic.length < want) {
            this.addInput(prefix + (start + dynamic.length), type);
            dynamic = this.inputs.filter((s) => s && s.name && match(s.name));
        }
        // 回收尾部空槽
        const reversed = [...this.inputs].reverse();
        for (const slot of reversed) {
            if (!slot || !slot.name || !match(slot.name)) break;
            const current = this.inputs.filter((s) => s && s.name && match(s.name));
            if (!isSlotConnected(slot) && current.length > want) {
                this.removeInput(this.inputs.indexOf(slot));
            } else {
                break;
            }
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
