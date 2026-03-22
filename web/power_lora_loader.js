// ==========================================================================
// SF Power Lora Loader - Custom Node
// ==========================================================================
import { app } from "/scripts/app.js";

// ---------------------------------------------------------------------------
// Canvas Drawing Utilities
// ---------------------------------------------------------------------------

function isLowQuality() {
    return (app.canvas.ds?.scale || 1) <= 0.5;
}

function fitString(ctx, str, maxWidth) {
    let width = ctx.measureText(str).width;
    const ellipsis = "...";
    const ellipsisWidth = ctx.measureText(ellipsis).width;
    if (width <= maxWidth || width <= ellipsisWidth) return str;
    let min = 0, max = str.length;
    while (min <= max) {
        let guess = Math.floor((min + max) / 2);
        const val = ctx.measureText(str.substring(0, guess)).width;
        if (val < maxWidth - ellipsisWidth) min = guess + 1;
        else max = guess - 1;
    }
    return str.substring(0, max) + ellipsis;
}

function drawRoundedRect(ctx, pos, size, radius, fillColor, strokeColor) {
    ctx.save();
    if (strokeColor) {
        ctx.strokeStyle = strokeColor;
    }
    ctx.fillStyle = fillColor || LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(pos[0], pos[1], size[0], size[1], radius);
    ctx.fill();
    if (strokeColor && strokeColor !== "transparent") ctx.stroke();
    ctx.restore();
}

function drawToggle(ctx, posX, posY, height, value) {
    const lq = isLowQuality();
    const toggleRadius = height * 0.36;
    const bgWidth = height * 1.5;
    ctx.save();
    if (!lq) {
        ctx.beginPath();
        ctx.roundRect(posX + 4, posY + 4, bgWidth - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = app.canvas.editor_alpha;
    }
    ctx.fillStyle = value === true ? "#89B" : "#888";
    const tx = lq || value === false ? posX + height * 0.5 : value === true ? posX + height : posX + height * 0.75;
    ctx.beginPath();
    ctx.arc(tx, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return [posX, bgWidth];
}

function drawNumberWidget(ctx, posX, posY, height, value, direction) {
    const AW = 9, AH = 10, IM = 3, NW = 32;
    ctx.save();
    if (direction === -1) posX = posX - AW - IM - NW - IM - AW;
    const midY = posY + height / 2;
    // Left arrow
    ctx.fill(new Path2D(`M ${posX} ${midY} l ${AW} ${AH/2} l 0 -${AH} L ${posX} ${midY} z`));
    const leftBounds = [posX, AW];
    posX += AW + IM;
    // Number
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, value.toFixed(2), NW), posX + NW / 2, midY);
    const numBounds = [posX, NW];
    posX += NW + IM;
    // Right arrow
    ctx.fill(new Path2D(`M ${posX} ${midY - AH/2} l ${AW} ${AH/2} l -${AW} ${AH/2} v -${AH} z`));
    const rightBounds = [posX, AW];
    ctx.restore();
    return [leftBounds, numBounds, rightBounds];
}
drawNumberWidget.WIDTH = 9 + 3 + 32 + 3 + 9;

// ---------------------------------------------------------------------------
// Array Utils
// ---------------------------------------------------------------------------
function moveArrayItem(arr, item, to) {
    const from = arr.indexOf(item);
    if (from === -1) return;
    arr.splice(to, 0, arr.splice(from, 1)[0]);
}
function removeArrayItem(arr, item) {
    const idx = arr.indexOf(item);
    if (idx > -1) arr.splice(idx, 1);
}

// ---------------------------------------------------------------------------
// Lora Chooser
// ---------------------------------------------------------------------------
function showLoraChooser(event, callback) {
    const loras = ["None", ...(app.loras || [])];
    new LiteGraph.ContextMenu(loras, { event, title: "Choose a lora", scale: Math.max(1, app.canvas.ds?.scale ?? 1), className: "dark", callback });
}

function fetchLoraList() {
    return fetch("/models/loras")
        .then(r => r.json())
        .then(list => { app.loras = list; })
        .catch(() => { app.loras = app.loras || []; });
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PROP_KEY = "SF_ShowStrengths";
const OPT_SINGLE = "Single Strength";
const OPT_SEPARATE = "Separate Model & Clip";
const NODE_TYPE = "SFPowerLoraLoader";

// ---------------------------------------------------------------------------
// Setup node instance
// ---------------------------------------------------------------------------
function setupNode(node) {
    node.loraWidgetsCounter = 0;
    node.widgetButtonSpacer = null;
    node.serialize_widgets = true;

    node.properties = node.properties || {};
    node.properties[PROP_KEY] = node.properties[PROP_KEY] || OPT_SINGLE;

    if (!app.loras) fetchLoraList();

    // ---- helper: is this widget a lora row? ----
    function isLoraWidget(w) { return w?.name?.startsWith?.("lora_"); }

    // ---- add a new lora row widget ----
    node.addNewLoraWidget = function (loraName) {
        this.loraWidgetsCounter++;
        const widget = createLoraWidget("lora_" + this.loraWidgetsCounter, node);
        if (loraName) widget._value.lora = loraName;
        // Insert before the spacer
        if (this.widgetButtonSpacer) {
            const idx = this.widgets.indexOf(this.widgetButtonSpacer);
            if (idx !== -1) this.widgets.splice(idx, 0, widget);
            else this.widgets.push(widget);
        } else {
            this.widgets.push(widget);
        }
        return widget;
    };

    // ---- add header / spacer / button ----
    node.addNonLoraWidgets = function () {
        const header = createHeaderWidget(node);
        const spacer = createSpacerWidget();
        const btnWidget = node.addWidget("button", "\u2795 Add Lora", null, () => {
            showLoraChooser(window._sfLastEvent, (value) => {
                if (typeof value === "string" && value !== "None") {
                    node.addNewLoraWidget(value);
                    const s = node.computeSize();
                    node.setSize([node.size[0], Math.max(node.size[1], s[1])]);
                    node.setDirtyCanvas(true, true);
                }
            });
        });

        // Insert header and spacer at the beginning
        this.widgets.splice(0, 0, header, spacer);
        this.widgetButtonSpacer = spacer;
    };

    // ---- queries ----
    node.hasLoraWidgets = function () {
        return this.widgets.some(w => w.name?.startsWith("lora_"));
    };
    node.allLorasState = function () {
        let allOn = true, allOff = true;
        for (const w of this.widgets) {
            if (!w.name?.startsWith("lora_")) continue;
            const on = w._value?.on;
            allOn = allOn && on === true;
            allOff = allOff && on === false;
            if (!allOn && !allOff) return null;
        }
        return allOn && this.widgets.length ? true : false;
    };
    node.toggleAllLoras = function () {
        const allOn = this.allLorasState();
        const to = !allOn;
        for (const w of this.widgets) {
            if (w.name?.startsWith("lora_") && w._value) w._value.on = to;
        }
        node.setDirtyCanvas(true, true);
    };

    // ---- override configure ----
    const _origConfigure = node.configure;
    node.configure = function (info) {
        // Remove all existing widgets
        while (this.widgets?.length) this.removeWidget(0);
        this.widgetButtonSpacer = null;
        this.loraWidgetsCounter = 0;
        // Call original configure (restores properties etc.)
        if (_origConfigure) _origConfigure.call(this, info);
        // Re-add lora widgets from saved data
        for (const v of info.widgets_values || []) {
            if (v && typeof v === "object" && v.lora !== undefined) {
                const w = this.addNewLoraWidget();
                w._value = { ...v };
            }
        }
        this.addNonLoraWidgets();
        const s = this.computeSize();
        this.setSize([this.size[0], Math.max(this.size[1], s[1])]);
    };

    // ---- override onNodeCreated ----
    const _origCreated = node.onNodeCreated;
    node.onNodeCreated = function () {
        if (_origCreated) _origCreated.call(this);
        this.addNonLoraWidgets();
        const s = this.computeSize();
        this.setSize([Math.max(this.size[0], s[0]), Math.max(this.size[1], s[1])]);
        this.setDirtyCanvas(true, true);
    };

    // ---- context menu ----
    const _origGetSlotMenu = node.getSlotMenuOptions;
    node.getSlotMenuOptions = function (slot) {
        if (slot?.widget?.name?.startsWith("lora_")) {
            const widget = slot.widget;
            const idx = this.widgets.indexOf(widget);
            const canUp = idx > 0 && this.widgets[idx - 1]?.name?.startsWith("lora_");
            const canDown = idx < this.widgets.length - 1 && this.widgets[idx + 1]?.name?.startsWith("lora_");
            new LiteGraph.ContextMenu([
                { content: `${widget._value?.on ? "Disable" : "Enable"} Toggle`, callback: () => { widget._value.on = !widget._value.on; node.setDirtyCanvas(true, true); } },
                null,
                { content: "Move Up", disabled: !canUp, callback: () => { moveArrayItem(this.widgets, widget, idx - 1); node.setDirtyCanvas(true, true); } },
                { content: "Move Down", disabled: !canDown, callback: () => { moveArrayItem(this.widgets, widget, idx + 1); node.setDirtyCanvas(true, true); } },
                { content: "Remove", callback: () => { removeArrayItem(this.widgets, widget); node.setDirtyCanvas(true, true); } },
            ], { title: "LORA WIDGET", event: window._sfLastEvent });
            return undefined;
        }
        if (_origGetSlotMenu) return _origGetSlotMenu.call(this, slot);
        return [];
    };

    // ---- refresh ----
    node.refreshComboInNode = function () { fetchLoraList(); };

    // Capture last mouse event for context menu
    const origProcessMouse = node.processMouseDown;
    if (!window._sfMouseHooked) {
        window._sfMouseHooked = true;
        document.addEventListener("pointerdown", (e) => { window._sfLastEvent = e; });
    }
}

// ---------------------------------------------------------------------------
// Create custom widget objects
// ---------------------------------------------------------------------------

function createSpacerWidget() {
    return {
        name: "_spacer",
        type: "custom",
        options: { serialize: false },
        value: {},
        draw() { /* invisible spacer */ },
        computeSize(width) { return [width, 8]; },
    };
}

function createHeaderWidget(node) {
    const w = {
        name: "_header",
        type: "custom",
        options: { serialize: false },
        value: { type: "header" },
        _hitToggle: [0, 0],
        draw(ctx, n, width, posY, height) {
            if (!n.hasLoraWidgets()) return;
            const showSep = n.properties[PROP_KEY] === OPT_SEPARATE;
            const margin = 10, im = margin * 0.33;
            const lq = isLowQuality();
            posY += 2;
            const midY = posY + height * 0.5;
            let posX = 10;
            w._hitToggle = drawToggle(ctx, posX, posY, height, n.allLorasState());
            if (!lq) {
                posX += w._hitToggle[1] + im;
                ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText("Toggle All", posX, midY);
                let rposX = width - margin - im - im;
                ctx.textAlign = "center";
                ctx.fillText(showSep ? "Clip" : "Strength", rposX - drawNumberWidget.WIDTH / 2, midY);
                if (showSep) {
                    rposX = rposX - drawNumberWidget.WIDTH - im * 2;
                    ctx.fillText("Model", rposX - drawNumberWidget.WIDTH / 2, midY);
                }
            }
        },
        mouse(event, pos, n) {
            if (event.type === "pointerdown") {
                const b = w._hitToggle;
                if (pos[0] >= b[0] && pos[0] <= b[0] + b[1]) {
                    n.toggleAllLoras();
                    return true;
                }
            }
            return false;
        },
    };
    return w;
}

function createLoraWidget(name, node) {
    const w = {
        name,
        type: "custom",
        value: null, // will be set by getter
        _value: { on: true, lora: null, strength: 1, strengthTwo: null },
        _showSep: null,
        _haveMouseMoved: false,
        _hit: {},
        get value() { return this._value; },
        set value(v) {
            if (typeof v === "object" && v) this._value = v;
            else this._value = { on: true, lora: null, strength: 1, strengthTwo: null };
        },
        serializeValue() {
            const v = { ...this._value };
            if (!this._showSep) delete v.strengthTwo;
            else v.strengthTwo = v.strengthTwo ?? 1;
            return v;
        },
        draw(ctx, n, width, posY, height) {
            const showSep = n.properties[PROP_KEY] === OPT_SEPARATE;
            // Handle mode switch
            if (this._showSep !== showSep) {
                const old = this._showSep;
                this._showSep = showSep;
                if (showSep) {
                    if (old != null) this._value.strengthTwo = this._value.strength ?? 1;
                } else {
                    this._value.strengthTwo = null;
                    this._hit = {};
                }
            }
            const margin = 10, im = margin * 0.33;
            const lq = isLowQuality();
            const midY = posY + height * 0.5;
            let posX = margin;
            // Background
            drawRoundedRect(ctx, [posX, posY], [width - margin * 2, height], height * 0.5, LiteGraph.WIDGET_BGCOLOR, LiteGraph.WIDGET_OUTLINE_COLOR);
            // Toggle
            this._hit.toggle = drawToggle(ctx, posX, posY, height, this._value.on);
            posX += this._hit.toggle[1] + im;
            if (lq) return;
            if (!this._value.on) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            let rposX = width - margin - im - im;
            // Strength (rightmost / clip)
            const sVal = showSep ? (this._value.strengthTwo ?? 1) : (this._value.strength ?? 1);
            const [la, ta, ra] = drawNumberWidget(ctx, rposX, posY, height, sVal, -1);
            this._hit.strengthTwoDec = la;
            this._hit.strengthTwoVal = ta;
            this._hit.strengthTwoInc = ra;
            this._hit.strengthTwoAny = [la[0], ra[0] + ra[1] - la[0]];
            rposX = la[0] - im;
            // Model strength (second from right)
            if (showSep) {
                rposX -= im;
                const [la2, ta2, ra2] = drawNumberWidget(ctx, rposX, posY, height, this._value.strength ?? 1, -1);
                this._hit.strengthDec = la2;
                this._hit.strengthVal = ta2;
                this._hit.strengthInc = ra2;
                this._hit.strengthAny = [la2[0], ra2[0] + ra2[1] - la2[0]];
                rposX = la2[0] - im;
            }
            // Lora name
            const loraWidth = rposX - posX;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(fitString(ctx, String(this._value?.lora || "None"), loraWidth), posX, midY);
            this._hit.lora = [posX, loraWidth];
            ctx.globalAlpha = app.canvas.editor_alpha;
        },
        mouse(event, pos, n) {
            if (event.type === "pointerdown") {
                w._mouseDown = [...pos];
                w._haveMouseMoved = false;
                // Check toggle
                if (hitTest(pos, w._hit.toggle)) {
                    w._value.on = !w._value.on;
                    n.setDirtyCanvas(true, true);
                    return true;
                }
                // Check lora name click
                if (hitTest(pos, w._hit.lora)) {
                    showLoraChooser(event, (val) => {
                        if (typeof val === "string" && val !== "None") {
                            w._value.lora = val;
                            n.setDirtyCanvas(true, true);
                        }
                    });
                    return true;
                }
                // Check strengthTwo arrows
                if (hitTest(pos, w._hit.strengthTwoDec)) { stepStrength(w, -1, true); n.setDirtyCanvas(true, true); return true; }
                if (hitTest(pos, w._hit.strengthTwoInc)) { stepStrength(w, 1, true); n.setDirtyCanvas(true, true); return true; }
                // Check model strength arrows
                if (hitTest(pos, w._hit.strengthDec)) { stepStrength(w, -1, false); n.setDirtyCanvas(true, true); return true; }
                if (hitTest(pos, w._hit.strengthInc)) { stepStrength(w, 1, false); n.setDirtyCanvas(true, true); return true; }
                // Check number value click (prompt)
                if (hitTest(pos, w._hit.strengthTwoVal)) {
                    app.canvas.prompt("Value", w._value.strengthTwo ?? 1, (v) => { w._value.strengthTwo = Number(v); n.setDirtyCanvas(true, true); }, event);
                    return true;
                }
                if (hitTest(pos, w._hit.strengthVal)) {
                    app.canvas.prompt("Value", w._value.strength ?? 1, (v) => { w._value.strength = Number(v); n.setDirtyCanvas(true, true); }, event);
                    return true;
                }
                // Check strength area drag
                if (hitTest(pos, w._hit.strengthTwoAny)) {
                    w._dragTarget = "strengthTwo";
                    return true;
                }
                if (hitTest(pos, w._hit.strengthAny)) {
                    w._dragTarget = "strength";
                    return true;
                }
            }
            if (event.type === "pointermove" && w._mouseDown) {
                if (w._dragTarget && event.deltaX) {
                    w._haveMouseMoved = true;
                    w._value[w._dragTarget] = (w._value[w._dragTarget] ?? 1) + event.deltaX * 0.05;
                    n.setDirtyCanvas(true, true);
                }
            }
            if (event.type === "pointerup") {
                w._mouseDown = null;
                w._dragTarget = null;
            }
            return false;
        },
    };

    function hitTest(pos, bounds) {
        if (!bounds || bounds.length < 2) return false;
        return pos[0] >= bounds[0] && pos[0] <= bounds[0] + bounds[1];
    }
    function stepStrength(widget, dir, isTwo) {
        const prop = isTwo ? "strengthTwo" : "strength";
        const val = (widget._value[prop] ?? 1) + dir * 0.05;
        widget._value[prop] = Math.round(val * 100) / 100;
    }

    return w;
}

// ---------------------------------------------------------------------------
// Register Extension
// ---------------------------------------------------------------------------
app.registerExtension({
    name: "sfnodes.SFPowerLoraLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        nodeType["@SF_ShowStrengths"] = { type: "combo", values: [OPT_SINGLE, OPT_SEPARATE] };
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (orig) orig.call(this);
            setupNode(this);
        };
    },
});
