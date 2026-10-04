import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Reports this browser tab's open workflows to the server (open_workflows.py), so an
// external agent can see what is open, which tab is active and what is unsaved.
// Only a small summary is pushed, and only when it changes (plus a heartbeat so the
// server can tell a live tab from a closed one). Full graphs are sent on request,
// and edits requested by the server are applied to the live canvas.
const POLL_MS = 2000;
const HEARTBEAT_MS = 20000;

function workflowStore() {
    return app.extensionManager?.workflow;
}

function summarize() {
    const store = workflowStore();
    if (!store?.openWorkflows) return null;
    const active = store.activeWorkflow;
    return store.openWorkflows.map((wf) => ({
        path: wf.path,
        filename: wf.filename,
        active: wf === active,
        modified: !!wf.isModified,
        persisted: !!wf.isPersisted,
        temporary: !!wf.isTemporary,
    }));
}

function report(body) {
    return api.fetchApi("/mets/open_workflows/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: api.clientId, ...body }),
    });
}

function respond(request_id, result) {
    return api.fetchApi("/mets/open_workflows/response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id, client_id: api.clientId, ...result }),
    });
}

async function serializeWorkflow(wf) {
    const store = workflowStore();
    if (wf === store.activeWorkflow) {
        // The live canvas, including edits not yet captured by the change tracker.
        return { graph: (app.rootGraph ?? app.graph).serialize(), source: "canvas" };
    }
    const state = wf.changeTracker?.activeState ?? wf.activeState;
    if (state) return { graph: state, source: "tab_state" };
    // Tabs not visited since page load have no in-memory state yet. Do NOT call
    // wf.load() here: it applies any auto-saved draft from browser storage and
    // flags the tab as modified, just like clicking the tab would. The server
    // falls back to reading the saved file instead.
    return { graph: null, source: "not_loaded" };
}

// Read (never modify) the auto-saved draft the frontend keeps in localStorage for a
// workflow path. Mirrors the frontend's draft store v2 layout:
//   "Comfy.Workflow.DraftIndex.v2:<workspace>" -> {entries: {<hash>: {path, ...}}}
//   "Comfy.Workflow.Draft.v2:<workspace>:<hash>" -> {data: "<graph json>", updatedAt}
// Useful because loading a tab deletes its draft if the file on disk is newer.
function readDraft(path) {
    const indexPrefix = "Comfy.Workflow.DraftIndex.v2:";
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (!key?.startsWith(indexPrefix)) continue;
        const workspace = key.slice(indexPrefix.length);
        try {
            const index = JSON.parse(localStorage.getItem(key));
            for (const [hash, entry] of Object.entries(index?.entries ?? {})) {
                if (entry?.path !== path) continue;
                const payload = JSON.parse(localStorage.getItem(`Comfy.Workflow.Draft.v2:${workspace}:${hash}`));
                if (payload?.data) return { workspace, updatedAt: payload.updatedAt, graph: JSON.parse(payload.data) };
            }
        } catch (e) {
            console.warn("[Mets] open_workflows: could not read draft index", key, e);
        }
    }
    return null;
}

async function answerGraphRequest({ request_id, path, client_id, include_draft }) {
    if (client_id && client_id !== api.clientId) return;
    const store = workflowStore();
    let result;
    const wf = path ? store?.openWorkflows?.find((w) => w.path === path) : store?.activeWorkflow;
    if (!wf) {
        result = { error: `workflow not open in this tab: ${path ?? "(active)"}` };
    } else {
        result = {
            path: wf.path,
            filename: wf.filename,
            active: wf === store.activeWorkflow,
            modified: !!wf.isModified,
            loaded: !!wf.isLoaded,
            ...(await serializeWorkflow(wf)),
        };
        if (include_draft) {
            result.file_last_modified = wf.lastModified ?? null;
            result.draft = readDraft(wf.path);
        }
    }
    await respond(request_id, result);
}

// LiteGraph node modes, as shown in the node's right-click "Mode" menu.
const MODES = { always: 0, mute: 2, bypass: 4 };

function resolveOp(graph, op) {
    const node = graph.getNodeById(op.node);
    if (!node) throw new Error(`no node with id ${JSON.stringify(op.node)}`);
    switch (op.op) {
        case "set_widget": {
            const widget = node.widgets?.find((w) => w.name === op.widget);
            if (!widget) {
                const names = (node.widgets ?? []).map((w) => w.name);
                throw new Error(`node ${op.node} has no widget "${op.widget}" (has: ${names.join(", ")})`);
            }
            return { node, widget, apply: () => {
                const old = widget.value;
                widget.value = op.value;
                widget.callback?.(widget.value, app.canvas, node);
                return { old, new: widget.value };
            } };
        }
        case "set_title":
            return { node, apply: () => {
                const old = node.title;
                node.title = op.title;
                return { old, new: node.title };
            } };
        case "set_mode": {
            const mode = typeof op.mode === "number" ? op.mode : MODES[op.mode];
            if (mode === undefined) throw new Error(`unknown mode ${JSON.stringify(op.mode)} (use ${Object.keys(MODES).join("/")})`);
            return { node, apply: () => {
                const old = node.mode;
                node.mode = mode;
                return { old, new: node.mode };
            } };
        }
        case "reconfigure":
            // Re-apply the node's own current state via configure(). Nodes with a custom UI
            // that keeps its own copy of widget data (e.g. MiniMaxH3Extender's clip cards)
            // re-read their widgets in onConfigure; without this they keep showing, and
            // later write back, the values from before a set_widget.
            return { node, apply: () => {
                node.configure(node.serialize());
                return {};
            } };
        default:
            throw new Error(`unknown op ${JSON.stringify(op.op)}`);
    }
}

// Switch the browser to an already-open tab, the same way clicking the tab does
// (mirrors the frontend's internal workflowService.openWorkflow). A tab not visited
// since page load gets load()ed first, which applies any auto-saved draft - exactly
// what a user click would do.
async function activateWorkflow(path) {
    const store = workflowStore();
    const wf = store?.openWorkflows?.find((w) => w.path === path);
    if (!wf) {
        const open = (store?.openWorkflows ?? []).map((w) => w.path);
        throw new Error(`workflow not open in this tab: ${path} (open: ${open.join(", ")})`);
    }
    if (wf === store.activeWorkflow) return { activated: path, was_active: true };
    const previous = store.activeWorkflow?.path ?? null;
    if (!wf.isLoaded) await wf.load();
    await app.loadGraphData(wf.activeState, true, true, wf, { checkForRerouteMigration: false, deferWarnings: true });
    if (store.activeWorkflow !== wf) throw new Error(`switching to ${path} did not take effect`);
    return { activated: path, previous };
}

async function answerEditRequest({ request_id, ops, path, client_id }) {
    if (client_id && client_id !== api.clientId) return;
    let result;
    try {
        // {"op": "activate", "path": "workflows/x.json"} switches tabs first, so the
        // remaining ops (and "queue") apply to that tab. It is applied before the other
        // ops are resolved: if a later op is invalid, the tab stays switched but nothing
        // is edited.
        const activateOp = (ops ?? []).find((op) => op.op === "activate");
        const activation = activateOp ? await activateWorkflow(activateOp.path) : null;
        const wf = workflowStore()?.activeWorkflow;
        if (!wf) throw new Error("no active workflow");
        if (path && path !== wf.path) throw new Error(`${path} is not the active tab (active: ${wf.path})`);
        const graph = app.rootGraph ?? app.graph;
        // Resolve every op before applying any, so a bad op leaves the canvas untouched.
        // {"op": "queue", "batch_count": n} is not a node edit: it presses Run after the edits.
        // {"op": "refresh_combos"} reloads the dropdown lists (files in input/, models, ...)
        // before the edits, so a file staged since page load can be selected.
        const nonNodeOps = ["queue", "activate", "refresh_combos"];
        const editOps = (ops ?? []).filter((op) => !nonNodeOps.includes(op.op));
        const queueOp = (ops ?? []).find((op) => op.op === "queue");
        if ((ops ?? []).some((op) => op.op === "refresh_combos")) await app.refreshComboInNodes();
        const resolved = editOps.map((op) => resolveOp(graph, op));
        const changes = resolved.map((r, i) => ({ ...editOps[i], ...r.apply() }));
        if (changes.length) {
            graph.setDirtyCanvas(true, true);
            // Record the change like a user edit: marks the tab modified and makes it undoable.
            const tracker = wf.changeTracker;
            if (tracker?.captureCanvasState) tracker.captureCanvasState();
            else tracker?.checkState?.();
        }
        result = { path: wf.path, changes };
        if (activation) result.activation = activation;
        if (queueOp) {
            // Same call as the Run button, so the full workflow is embedded in the outputs.
            await app.queuePrompt(0, queueOp.batch_count ?? 1);
            result.queued = queueOp.batch_count ?? 1;
        }
    } catch (e) {
        result = { error: String(e.message ?? e) };
    }
    await respond(request_id, result);
}

app.registerExtension({
    name: "Mets.OpenWorkflows",
    async setup() {
        let lastSent = null;
        let lastSentAt = 0;

        setInterval(() => {
            const workflows = summarize();
            if (!workflows || !api.clientId) return;
            const serialized = JSON.stringify(workflows);
            if (serialized === lastSent && Date.now() - lastSentAt < HEARTBEAT_MS) return;
            report({ workflows })
                .then(() => { lastSent = serialized; lastSentAt = Date.now(); })
                .catch(() => {});
        }, POLL_MS);

        // app.loadGraphData is the same public method ComfyUI's own File > Open
        // uses, so a workflow file edited on disk shows up without a page reload.
        api.addEventListener("mets-open-workflows-load-file", (event) => {
            const { graph, workflow } = event.detail || {};
            if (!graph) return;
            app.loadGraphData(graph, true, true, workflow ?? null);
        });

        api.addEventListener("mets-open-workflows-request-graph", (event) => {
            answerGraphRequest(event.detail || {}).catch((e) => console.error("[Mets] open_workflows:", e));
        });

        api.addEventListener("mets-open-workflows-edit", (event) => {
            answerEditRequest(event.detail || {}).catch((e) => console.error("[Mets] open_workflows:", e));
        });

        window.addEventListener("pagehide", () => {
            navigator.sendBeacon(
                api.apiURL("/mets/open_workflows/report"),
                new Blob([JSON.stringify({ client_id: api.clientId, closing: true })], { type: "application/json" }),
            );
        });
    },
});
