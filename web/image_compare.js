// ==========================================================================
// SF Image Compare
// ==========================================================================

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const imageCache = new Map();

const restoreImagesFromCache = (node) => {
    const cached = imageCache.get(node.id);
    if (!cached || (!cached.imageAData && !cached.imageBData)) return;

    node.imageAData = cached.imageAData;
    node.imageBData = cached.imageBData;

    let loadedCount = 0;
    const assetsToLoad = (cached.imageAData ? 1 : 0) + (cached.imageBData ? 1 : 0);
    const onLoaded = () => {
        loadedCount++;
        if (loadedCount === assetsToLoad) {
            if (node.imageA && typeof node.autosize === 'function') {
                node.autosize(node.imageA);
            }
            node.setDirtyCanvas(true, true);
        }
    };

    node.imageA = cached.imageAData
        ? Object.assign(new Image(), { src: `data:image/png;base64,${cached.imageAData}`, onload: onLoaded })
        : null;
    node.imageB = cached.imageBData
        ? Object.assign(new Image(), { src: `data:image/png;base64,${cached.imageBData}`, onload: onLoaded })
        : null;
};

app.registerExtension({
    name: "sfnodes.SFImageCompare",

    nodeCreated(node) {
        if (node.comfyClass === "SFImageCompare") {
            const PADDING = 10;
            const HEADER_HEIGHT = 100;
            const MIN_HEIGHT = 300;
            const NEUTRALPOS = 0.5;

            node.imageA = null;
            node.imageB = null;
            node.isHovering = false;
            node.isManuallyResized = false;
            node.slider_pos = NEUTRALPOS;
            node.showA = true;
            node.setSize([320, 440]);

            node.addWidget("combo", "mode", "Slide", () => {
                node.setDirtyCanvas(true, true);
            }, {
                values: ["Slide", "Click", "diff", "lighten", "darken", "screen", "multiply"]
            });

            node.getMode = function () {
                const w = this.widgets.find(w => w.name === "mode");
                return w ? w.value : "Slide";
            };

            node.addWidget("button", "Reset Node Size", null, () => {
                node.isManuallyResized = false;
                node.slider_pos = NEUTRALPOS;
                node.showA = true;
                if (node.imageA) {
                    autosize(node.imageA);
                }
                node.setDirtyCanvas(true, true);
            });

            const autosize = (img) => {
                if (!node.isManuallyResized && img) {
                    const aspectRatio = img.naturalWidth / img.naturalHeight;
                    const baseWidth = 300;
                    node.size[0] = baseWidth;
                    const drawAreaHeight = (baseWidth - PADDING * 2) / aspectRatio;
                    let newHeight = drawAreaHeight + HEADER_HEIGHT + PADDING;
                    if (newHeight < MIN_HEIGHT) {
                        newHeight = MIN_HEIGHT;
                    }
                    node.size[1] = newHeight;
                    node.setDirtyCanvas(true, true);
                }
            };

            node.autosize = autosize;

            const originalConfigure = node.configure;
            node.configure = function (data) {
                originalConfigure.apply(this, arguments);
                if (data.isManuallyResized) this.isManuallyResized = data.isManuallyResized;
                if (data.slider_pos !== undefined) this.slider_pos = data.slider_pos;
                if (data.showA !== undefined) this.showA = data.showA;
                restoreImagesFromCache(this);
            };

            const originalSerialize = node.serialize;
            node.serialize = function () {
                const data = originalSerialize.call(this);
                data.isManuallyResized = this.isManuallyResized;
                data.slider_pos = this.slider_pos;
                data.showA = this.showA;
                return data;
            };

            node.onResize = function () {
                this.isManuallyResized = true;
                if (this.size[1] < MIN_HEIGHT) {
                    this.size[1] = MIN_HEIGHT;
                }
            };

            const drawLabelWithBackground = (ctx, text, x, y, textAlign) => {
                const textMetrics = ctx.measureText(text);
                const boxPadding = 2;
                const fontSize = 8;
                const boxHeight = fontSize + (boxPadding * 2);
                const boxWidth = textMetrics.width + (boxPadding * 2);
                const boxRadius = 1.5;
                let boxX;
                if (textAlign === "left") {
                    boxX = x - boxPadding;
                } else {
                    boxX = x - textMetrics.width - boxPadding;
                }
                const boxY = y - (fontSize / 2) - boxPadding - 0.3;
                ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
                ctx.beginPath();
                ctx.moveTo(boxX + boxRadius, boxY);
                ctx.arcTo(boxX + boxWidth, boxY, boxX + boxWidth, boxY + boxHeight, boxRadius);
                ctx.arcTo(boxX + boxWidth, boxY + boxHeight, boxX, boxY + boxHeight, boxRadius);
                ctx.arcTo(boxX, boxY + boxHeight, boxX, boxY, boxRadius);
                ctx.arcTo(boxX, boxY, boxX + boxWidth, boxY, boxRadius);
                ctx.closePath();
                ctx.fill();
                ctx.fillStyle = "white";
                ctx.textAlign = textAlign;
                ctx.textBaseline = "middle";
                ctx.fillText(text, x, y);
            };

            Object.assign(node, {
                getContainerArea() {
                    const area = {
                        x: PADDING,
                        y: HEADER_HEIGHT,
                        width: this.size[0] - PADDING * 2,
                        height: this.size[1] - HEADER_HEIGHT - PADDING
                    };
                    if (area.height < 0) area.height = 0;
                    return (area.width < 1 || area.height < 1) ? null : area;
                },

                getImageRenderData(img, container) {
                    const imgRatio = img.naturalWidth / img.naturalHeight;
                    const containerRatio = container.width / container.height;
                    let renderWidth, renderHeight, renderX, renderY;
                    if (imgRatio > containerRatio) {
                        renderWidth = container.width;
                        renderHeight = container.width / imgRatio;
                    } else {
                        renderHeight = container.height;
                        renderWidth = container.height * imgRatio;
                    }
                    renderX = container.x + (container.width - renderWidth) / 2;
                    renderY = container.y + (container.height - renderHeight) / 2;
                    return { x: renderX, y: renderY, width: renderWidth, height: renderHeight };
                },

                drawSliderView(ctx, renderData) {
                    const sliderValue = this.slider_pos;
                    const sliderPx = renderData.x + sliderValue * renderData.width;

                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(sliderPx, renderData.y, renderData.x + renderData.width - sliderPx, renderData.height);
                    ctx.clip();
                    ctx.drawImage(this.imageA, renderData.x, renderData.y, renderData.width, renderData.height);
                    ctx.restore();

                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(renderData.x, renderData.y, sliderPx - renderData.x, renderData.height);
                    ctx.clip();
                    ctx.drawImage(this.imageB, renderData.x, renderData.y, renderData.width, renderData.height);
                    ctx.restore();

                    if (this.isHovering && sliderValue > 0 && sliderValue < 1) {
                        ctx.font = "100 8px Arial";
                        ctx.textBaseline = "top";
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(sliderPx, renderData.y, renderData.x + renderData.width - sliderPx, renderData.height);
                        ctx.clip();
                        drawLabelWithBackground(ctx, "A", renderData.x + renderData.width - 5, renderData.y + 9, "right");
                        ctx.restore();
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(renderData.x, renderData.y, sliderPx - renderData.x, renderData.height);
                        ctx.clip();
                        drawLabelWithBackground(ctx, "B", renderData.x + 5, renderData.y + 9, "left");
                        ctx.restore();
                    }

                    const smallValue = 0.001;

                    if (this.slider_pos > smallValue && this.slider_pos < 1.0 - smallValue) {
                        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
                        ctx.lineWidth = 0.5;
                        ctx.beginPath();
                        ctx.moveTo(sliderPx, renderData.y);
                        ctx.lineTo(sliderPx, renderData.y + renderData.height);
                        ctx.stroke();

                        ctx.fillStyle = "rgba(255, 255, 255, 1.0)";
                        const handleY = renderData.y + renderData.height / 2;
                        const triangleSize = 3.5;
                        const triangleGap = 2.5;

                        ctx.beginPath();
                        ctx.moveTo(sliderPx - triangleGap, handleY - triangleSize);
                        ctx.lineTo(sliderPx - triangleGap, handleY + triangleSize);
                        ctx.lineTo(sliderPx - triangleGap - triangleSize, handleY);
                        ctx.closePath();
                        ctx.fill();

                        ctx.beginPath();
                        ctx.moveTo(sliderPx + triangleGap, handleY - triangleSize);
                        ctx.lineTo(sliderPx + triangleGap, handleY + triangleSize);
                        ctx.lineTo(sliderPx + triangleGap + triangleSize, handleY);
                        ctx.closePath();
                        ctx.fill();
                    }
                },

                drawBlendView(ctx, renderData) {
                    ctx.drawImage(this.imageB, renderData.x, renderData.y, renderData.width, renderData.height);
                    const mode = this.getMode();
                    ctx.globalCompositeOperation = mode;
                    ctx.drawImage(this.imageA, renderData.x, renderData.y, renderData.width, renderData.height);
                    ctx.globalCompositeOperation = "source-over";
                },

                onDrawForeground(ctx) {
                    if (this.flags.collapsed) return;

                    ctx.save();
                    const containerArea = this.getContainerArea();
                    if (!containerArea) {
                        ctx.restore();
                        return;
                    }

                    if (this.imageA) {
                        const renderData = this.getImageRenderData(this.imageA, containerArea);

                        if (!this.imageB) {
                            ctx.drawImage(this.imageA, renderData.x, renderData.y, renderData.width, renderData.height);
                            ctx.restore();
                            return;
                        }

                        const mode = this.getMode();

                        if (mode === "Slide") {
                            this.drawSliderView(ctx, renderData);
                        } else if (mode === "Click") {
                            if (this.showA) {
                                ctx.drawImage(this.imageA, renderData.x, renderData.y, renderData.width, renderData.height);
                            } else {
                                ctx.drawImage(this.imageB, renderData.x, renderData.y, renderData.width, renderData.height);
                            }
                        } else {
                            this.drawBlendView(ctx, renderData);
                        }
                    } else {
                        ctx.font = "11px Arial";
                        ctx.fillStyle = "#CCCCCC";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("Connect Images and run workflow", containerArea.x + containerArea.width / 2, containerArea.y + containerArea.height / 2);
                    }

                    ctx.restore();
                },

                updateSliderFromEvent(event) {
                    if (!this.imageA) return;
                    const renderData = this.getImageRenderData(this.imageA, this.getContainerArea());
                    const localPos = app.canvas.convertEventToCanvasOffset(event);
                    const mouseX = localPos[0] - this.pos[0];
                    let newSliderValue = (mouseX - renderData.x) / renderData.width;
                    this.slider_pos = Math.max(0.0, Math.min(1.0, newSliderValue));
                    this.setDirtyCanvas(true, true);
                },

                onMouseDown(event) {
                    if (event.button !== 0 || !this.imageA || !this.imageB) return false;

                    const mode = this.getMode();

                    if (mode === "Click") {
                        this.showA = !this.showA;
                        this.setDirtyCanvas(true, true);
                        return true;
                    }

                    if (mode === "Slide") {
                        const renderData = this.getImageRenderData(this.imageA, this.getContainerArea());
                        const localPos = app.canvas.convertEventToCanvasOffset(event);
                        const mouseX = localPos[0] - this.pos[0];
                        const mouseY = localPos[1] - this.pos[1];
                        if (mouseX >= renderData.x && mouseX <= renderData.x + renderData.width &&
                            mouseY >= renderData.y && mouseY <= renderData.y + renderData.height) {
                            this.isDragging = true;
                            this.updateSliderFromEvent(event);
                            return true;
                        }
                    }

                    return false;
                },

                onMouseEnter(event) {
                    if (!this.imageA) return;
                    this.isHovering = true;
                    if (this.imageB) {
                        this.setDirtyCanvas(true, true);
                    }
                },

                onMouseLeave(event) {
                    if (!this.imageA) return;
                    this.isHovering = false;
                    if (this.getMode() === "Slide") {
                        this.slider_pos = 0;
                    }
                    document.body.style.cursor = 'default';
                    this.setDirtyCanvas(true, true);
                },

                onMouseMove(event, pos, canvas) {
                    if (!this.imageA) return;

                    const mode = this.getMode();
                    if (mode !== "Slide") {
                        document.body.style.cursor = 'default';
                        return;
                    }

                    const renderData = this.getImageRenderData(this.imageA, this.getContainerArea());
                    const isOverImage = pos[0] >= renderData.x && pos[0] <= renderData.x + renderData.width &&
                        pos[1] >= renderData.y && pos[1] <= renderData.y + renderData.height;

                    if (isOverImage) {
                        document.body.style.cursor = 'ew-resize';
                        if (this.imageB) {
                            let newSliderValue = (pos[0] - renderData.x) / renderData.width;
                            this.slider_pos = Math.max(0.0, Math.min(1.0, newSliderValue));
                            this.setDirtyCanvas(true, true);
                        }
                    } else {
                        document.body.style.cursor = 'default';
                    }
                },
            });
        }
    },
});

api.addEventListener("sfnodes.image_compare_preview", ({ detail }) => {
    const node = app.graph.getNodeById(detail.node_id);
    if (!node) return;

    imageCache.set(detail.node_id, {
        imageAData: detail.image_a_data,
        imageBData: detail.image_b_data,
    });

    let assetsToLoad = (detail.image_a_data ? 1 : 0) + (detail.image_b_data ? 1 : 0);
    if (assetsToLoad === 0) {
        node.imageA = null;
        node.imageB = null;
        node.setDirtyCanvas(true, true);
        return;
    }

    let loadedCount = 0;
    const onAssetLoaded = () => {
        loadedCount++;
        if (loadedCount === assetsToLoad) {
            if (node.imageA && typeof node.autosize === 'function') {
                node.autosize(node.imageA);
            }
            node.setDirtyCanvas(true, true);
        }
    };

    node.imageA = detail.image_a_data ? Object.assign(new Image(), { src: `data:image/png;base64,${detail.image_a_data}`, onload: onAssetLoaded }) : null;
    node.imageB = detail.image_b_data ? Object.assign(new Image(), { src: `data:image/png;base64,${detail.image_b_data}`, onload: onAssetLoaded }) : null;
});
