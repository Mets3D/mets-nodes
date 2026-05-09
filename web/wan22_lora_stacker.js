import { app } from "../../scripts/app.js";

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

const _ID = "Wan22LoRAStacker";
const _PREFIX = "dual_lora";
const _TYPE = "WAN_DUAL_LORA";

app.registerExtension({
    name: "Mets.Wan22LoRAStacker",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== _ID) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            this.addInput(_PREFIX, _TYPE);
            const slot = this.inputs[this.inputs.length - 1];
            if (slot) {
                slot.color_off = "#666";
            }
            return me;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, _link_info, _node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);

            if (slotType === TypeSlot.Input) {
                if (event === TypeSlotEvent.Disconnect) {
                    this.removeInput(slot_idx);
                }

                // Renumber all connected slots sequentially
                let count = 0;
                for (const slot of this.inputs) {
                    if (slot.link !== null) {
                        count += 1;
                        slot.name = `${_PREFIX}_${count}`;
                    }
                }

                // Always keep one empty slot at the end
                const last = this.inputs[this.inputs.length - 1];
                if (last === undefined || last.link !== null) {
                    this.addInput(_PREFIX, _TYPE);
                    const newLast = this.inputs[this.inputs.length - 1];
                    if (newLast) {
                        newLast.color_off = "#666";
                    }
                }

                this?.graph?.setDirtyCanvas(true);
                return me;
            }
        };
    },
});
