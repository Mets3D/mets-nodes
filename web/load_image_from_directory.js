import { app } from "../../scripts/app.js";

const _ID = "LoadImageFromDirectory";
const _PAD = 10;
const _INFO_H = 22;

app.registerExtension({
    name: "Mets.BatchVideoLoadImage",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== _ID) return;

        const origDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            origDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed || !this._showPreview || !this._previewImg?.complete) return;
            const img = this._previewImg;
            const [w, h] = this.size;
            const availW = w - _PAD * 2;
            const imgH = availW * img.naturalHeight / img.naturalWidth;
            ctx.drawImage(img, _PAD, h - imgH - _INFO_H, availW, imgH);
            if (this._previewInfo) {
                ctx.save();
                ctx.font = "11px sans-serif";
                ctx.fillStyle = "rgba(255,255,255,0.6)";
                ctx.textAlign = "center";
                ctx.fillText(this._previewInfo, w / 2, h - 6);
                ctx.restore();
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            const dirWidget = node.widgets?.find(w => w.name === "directory");
            const idxWidget = node.widgets?.find(w => w.name === "index");
            if (!dirWidget || !idxWidget) return result;

            node.addWidget("button", "Browse...", null, async () => {
                try {
                    const resp = await fetch("/mets/browse_directory");
                    const { path } = await resp.json();
                    if (path) {
                        dirWidget.value = path;
                        scheduleUpdate();
                        app.graph.setDirtyCanvas(true);
                    }
                } catch (e) {
                    console.error("[Mets] browse_directory failed:", e);
                }
            });
            node.widgets.unshift(node.widgets.pop());

            node._previewImg = null;
            node._showPreview = true;
            let _debounce = null;

            const togglePreviewWidget = node.addWidget("button", "▼ Hide Preview", null, () => {
                node._showPreview = !node._showPreview;
                togglePreviewWidget.name = node._showPreview ? "▼ Hide Preview" : "► Show Preview";
                fitToPreview(node);
                app.graph?.setDirtyCanvas(true);
            });

            const scheduleUpdate = () => {
                clearTimeout(_debounce);
                _debounce = setTimeout(async () => {
                    const dir = dirWidget.value?.trim();
                    const idx = idxWidget.value ?? 0;
                    if (!dir) {
                        node._previewImg = null;
                        app.graph?.setDirtyCanvas(true);
                        return;
                    }
                    const url = `/mets/preview_image?directory=${encodeURIComponent(dir)}&index=${encodeURIComponent(idx)}&t=${Date.now()}`;
                    try {
                        const resp = await fetch(url);
                        if (!resp.ok) { node._previewImg = null; app.graph?.setDirtyCanvas(true); return; }
                        const filename = resp.headers.get("X-Filename") ?? "";
                        const total = parseInt(resp.headers.get("X-Total") ?? "0");
                        const objectUrl = URL.createObjectURL(await resp.blob());
                        const img = new Image();
                        img.onload = () => {
                            URL.revokeObjectURL(objectUrl);
                            node._previewImg = img;
                            node._previewInfo = `${filename}  •  ${(idx % total) + 1} / ${total}`;
                            fitToPreview(node);
                            app.graph?.setDirtyCanvas(true);
                        };
                        img.onerror = () => {
                            URL.revokeObjectURL(objectUrl);
                            node._previewImg = null;
                            node._previewInfo = null;
                            app.graph?.setDirtyCanvas(true);
                        };
                        img.src = objectUrl;
                    } catch (e) {
                        console.error("[Mets] preview fetch failed:", e);
                    }
                }, 300);
            };

            const wrapCallback = (widget) => {
                const orig = widget.callback;
                widget.callback = function (...args) { orig?.apply(this, args); scheduleUpdate(); };
            };
            wrapCallback(dirWidget);
            wrapCallback(idxWidget);

            node.onSerialize = (data) => { data.showPreview = node._showPreview; };

            node.onConfigure = (data) => {
                if (typeof data.showPreview === "boolean") {
                    node._showPreview = data.showPreview;
                    togglePreviewWidget.name = node._showPreview ? "▼ Hide Preview" : "► Show Preview";
                }
                scheduleUpdate();
            };

            node.onResize = () => {
                if (node._fittingToPreview) return;
                node._fittingToPreview = true;
                fitToPreview(node);
                app.graph?.setDirtyCanvas(true);
                node._fittingToPreview = false;
            };

            return result;
        };
    },
});

function fitToPreview(node) {
    const img = node._previewImg;
    if (!img || !node._showPreview) {
        node.setSize([node.size[0], node.computeSize()[1]]);
        return;
    }
    const baseH = node.computeSize()[1];
    const availW = node.size[0] - _PAD * 2;
    const imgH = availW * img.naturalHeight / img.naturalWidth;
    node.setSize([node.size[0], baseH + imgH + _INFO_H]);
}
