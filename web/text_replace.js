import { app } from "/scripts/app.js";

const MAX_SLOTS = 20;

const MARKER_MENU_ITEMS = [
    { token: "{date}", label: "插入 {date}（日期）" },
    { token: "{time}", label: "插入 {time}（时间）" },
    { token: "{datetime}", label: "插入 {datetime}（日期时间）" },
    { token: "{timestamp}", label: "插入 {timestamp}（Unix 时间戳）" },
    null,
    { token: "{random}", label: "插入 {random}（随机数）" },
    { token: "{random:100}", label: "插入 {random:100}（0-99）" },
    { token: "{uuid}", label: "插入 {uuid}（短 UUID）" },
];

function caretOffsetAt(el, x, y) {
    const candidates = [];
    if (document.caretRangeFromPoint) candidates.push(document.caretRangeFromPoint(x, y));
    if (document.caretPositionFromPoint) candidates.push(document.caretPositionFromPoint(x, y));
    for (const pos of candidates) {
        if (!pos) continue;
        const node = pos.offsetNode !== undefined ? pos.offsetNode : pos.startContainer;
        const offset = pos.offset !== undefined ? pos.offset : pos.startOffset;
        if (node === el || (node && node.nodeType === Node.TEXT_NODE && node.parentNode === el)) {
            return offset;
        }
    }
    return null;
}

function showMarkerMenu(x, y, onSelect) {
    const menuEl = document.createElement("div");
    menuEl.style.cssText = [
        "position:fixed",
        "z-index:99999",
        "min-width:230px",
        "background:#2c2c2c",
        "border:1px solid #555",
        "border-radius:4px",
        "box-shadow:0 4px 12px rgba(0,0,0,0.5)",
        "padding:4px 0",
        "font-family:sans-serif",
        "font-size:13px",
        "color:#ddd",
        "user-select:none",
    ].join(";");

    const cleanup = () => {
        document.removeEventListener("mousedown", onDocMouseDown, true);
        document.removeEventListener("keydown", onKeyDown, true);
        document.removeEventListener("wheel", onDocMouseDown, true);
        menuEl.remove();
    };

    const onDocMouseDown = (e) => {
        if (!menuEl.contains(e.target)) cleanup();
    };

    const onKeyDown = (e) => {
        if (e.key === "Escape") cleanup();
    };

    for (const item of MARKER_MENU_ITEMS) {
        if (item === null) {
            const sep = document.createElement("div");
            sep.style.cssText = "height:1px;background:#555;margin:4px 8px;";
            menuEl.appendChild(sep);
            continue;
        }
        const row = document.createElement("div");
        row.textContent = item.label;
        row.style.cssText = "padding:5px 12px;cursor:pointer;white-space:nowrap;";
        row.addEventListener("mouseenter", () => {
            row.style.background = "#4a4a4a";
        });
        row.addEventListener("mouseleave", () => {
            row.style.background = "transparent";
        });
        row.addEventListener("click", () => {
            cleanup();
            onSelect(item.token);
        });
        menuEl.appendChild(row);
    }

    document.addEventListener("mousedown", onDocMouseDown, true);
    document.addEventListener("keydown", onKeyDown, true);
    document.addEventListener("wheel", onDocMouseDown, true);

    document.body.appendChild(menuEl);

    const rect = menuEl.getBoundingClientRect();
    const px = Math.min(x, window.innerWidth - rect.width - 8);
    const py = Math.min(y, window.innerHeight - rect.height - 8);
    menuEl.style.left = `${Math.max(8, px)}px`;
    menuEl.style.top = `${Math.max(8, py)}px`;
}

app.registerExtension({
    name: "sfnodes.TextReplace",

    nodeCreated(node) {
        if (node.comfyClass !== "SFTextReplace") return;

        node.visibleSlotCount = node.visibleSlotCount ?? 3;

        const getSlotWidget = (slotIndex) =>
            node.widgets.find((w) => w.name === `replace_${slotIndex}`);

        const setSlotVisibility = (slotIndex, visible) => {
            const widget = getSlotWidget(slotIndex);
            if (widget) {
                widget.hidden = !visible;
                if (!visible) widget.value = "";
            }
        };

        const updateNodeSize = () => {
            const baseHeight = 160;
            const slotHeight = 44;
            const calculatedHeight = baseHeight + node.visibleSlotCount * slotHeight;
            const currentWidth = node.size[0] || 420;
            node.setSize([currentWidth, calculatedHeight]);
        };

        const initializeSlots = () => {
            for (let i = 1; i <= MAX_SLOTS; i++) {
                setSlotVisibility(i, i <= node.visibleSlotCount);
            }
            updateNodeSize();
        };

        const addSlot = () => {
            if (node.visibleSlotCount < MAX_SLOTS) {
                node.visibleSlotCount++;
                setSlotVisibility(node.visibleSlotCount, true);
                updateNodeSize();
                updateButtonStates();
                node.setDirtyCanvas(true, true);
            }
        };

        const removeSlot = () => {
            if (node.visibleSlotCount > 1) {
                setSlotVisibility(node.visibleSlotCount, false);
                node.visibleSlotCount--;
                updateNodeSize();
                updateButtonStates();
                node.setDirtyCanvas(true, true);
            }
        };

        const updateButtonStates = () => {
            if (node.removeSlotButton) {
                node.removeSlotButton.disabled = node.visibleSlotCount <= 1;
            }
        };

        const templateWidget = node.widgets.find((w) => w.name === "template");
        if (templateWidget) {
            let pendingInsertPos = null;

            const insertToken = (token) => {
                const el = templateWidget.inputEl;
                let value = templateWidget.value || "";
                let start = null;
                let end = null;
                if (pendingInsertPos !== null) {
                    start = end = pendingInsertPos;
                    pendingInsertPos = null;
                } else if (el && el === document.activeElement && typeof el.selectionStart === "number") {
                    start = el.selectionStart;
                    end = el.selectionEnd;
                }
                if (start !== null) {
                    value = value.slice(0, start) + token + value.slice(end ?? start);
                    templateWidget.value = value;
                    if (el) {
                        el.value = value;
                        el.selectionStart = el.selectionEnd = start + token.length;
                    }
                } else {
                    templateWidget.value = value + token;
                }
                node.setDirtyCanvas(true, true);
            };

            templateWidget.options = templateWidget.options || {};
            templateWidget.options.contextMenu = MARKER_MENU_ITEMS.map((item) =>
                item === null ? null : { content: item.label, callback: () => insertToken(item.token) }
            );

            if (templateWidget.inputEl) {
                templateWidget.inputEl.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const pos = caretOffsetAt(templateWidget.inputEl, e.clientX, e.clientY);
                    if (pos !== null) {
                        pendingInsertPos = Math.max(0, Math.min(pos, (templateWidget.value || "").length));
                    }
                    showMarkerMenu(e.clientX, e.clientY, insertToken);
                });
            }
        }

        initializeSlots();

        node.addWidget("button", "➕ 添加替换项", null, addSlot);
        node.removeSlotButton = node.addWidget("button", "➖ 删除替换项", null, removeSlot);
        updateButtonStates();

        const originalSerialize = node.serialize;
        node.serialize = function () {
            const data = originalSerialize.call(this);
            data.visibleSlotCount = this.visibleSlotCount;
            return data;
        };

        const originalConfigure = node.configure;
        node.configure = function (data) {
            originalConfigure.apply(this, arguments);
            if (data.visibleSlotCount !== undefined) {
                this.visibleSlotCount = data.visibleSlotCount;
                initializeSlots();
                updateButtonStates();
            }
        };
    },
});
