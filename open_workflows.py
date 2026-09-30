import asyncio
import json
import time
import uuid
from pathlib import Path


# Bridge between external agents (plain HTTP) and the workflows open in the browser.
# Browser tabs push a summary of their open workflows here (web/open_workflows.js),
# keyed by the frontend's clientId, so an agent can ask "what is open?" with a plain
# GET. The full graph of a workflow is not pushed routinely - it is requested on
# demand over the websocket and awaited (see /mets/open_workflows/graph).
# Going the other way, /mets/open_workflows/edit changes widgets/titles/modes on the
# live canvas, and /mets/open_workflows/load_file replaces a tab with a file on disk.
_STALE_AFTER_SECONDS = 45
_BROWSER_TIMEOUT_SECONDS = 5

_clients: dict[str, dict] = {}
_pending: dict[str, asyncio.Future] = {}


async def _ask_browser(event: str, payload: dict) -> dict:
    """Send a websocket event to the browser tabs and await the first reply to /mets/open_workflows/response."""
    from server import PromptServer

    request_id = uuid.uuid4().hex
    future = asyncio.get_running_loop().create_future()
    _pending[request_id] = future
    try:
        PromptServer.instance.send_sync(event, {"request_id": request_id, **payload})
        return await asyncio.wait_for(future, timeout=_BROWSER_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return {"error": "no browser tab answered", "timeout": True}
    finally:
        _pending.pop(request_id, None)


def _read_user_file(path: str) -> dict:
    """Read a workflow by its frontend path (e.g. 'workflows/foo.json'), relative to the default user dir."""
    import folder_paths

    root = Path(folder_paths.get_user_directory(), "default").resolve()
    full = (root / path).resolve()
    if not full.is_relative_to(root):
        raise ValueError(f"path escapes user directory: {path}")
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        routes = PromptServer.instance.routes

        @routes.post("/mets/open_workflows/load_file")
        async def load_file(request: web.Request) -> web.Response:
            # Push a workflow file from disk into the browser, replacing the canvas of
            # its open tab (or opening it). Replaces the whole graph: any unsaved edits
            # in that tab are lost, so check /mets/open_workflows for modified first.
            data = await request.json()
            path = data.get("path", "")
            try:
                with open(path, encoding="utf-8") as f:
                    graph = json.load(f)
            except OSError as e:
                return web.json_response({"error": str(e)}, status=400)
            # The frontend tracks open workflows by a name relative to its own
            # "workflows/" root (ComfyWorkflow.basePath + appendJsonExt(value)),
            # not by filesystem path - passing the absolute path here made it
            # fail to match the already-open tab and open a duplicate instead.
            workflow_name = Path(path).stem
            PromptServer.instance.send_sync("mets-open-workflows-load-file", {"graph": graph, "workflow": workflow_name})
            return web.json_response({"ok": True, "path": path, "workflow": workflow_name})

        @routes.post("/mets/open_workflows/report")
        async def report(request: web.Request) -> web.Response:
            data = await request.json()
            client_id = data.get("client_id")
            if not client_id:
                return web.json_response({"error": "client_id required"}, status=400)
            if data.get("closing"):
                _clients.pop(client_id, None)
            else:
                _clients[client_id] = {
                    "client_id": client_id,
                    "workflows": data.get("workflows", []),
                    "updated_at": time.time(),
                }
            return web.json_response({"ok": True})

        @routes.get("/mets/open_workflows")
        async def open_workflows(request: web.Request) -> web.Response:
            now = time.time()
            clients = []
            for client in _clients.values():
                age = now - client["updated_at"]
                clients.append({**client, "age_seconds": round(age, 1), "stale": age > _STALE_AFTER_SECONDS})
            clients.sort(key=lambda c: c["age_seconds"])
            return web.json_response({"clients": clients})

        @routes.get("/mets/open_workflows/graph")
        async def get_graph(request: web.Request) -> web.Response:
            # ?path=<workflow path as listed by /mets/open_workflows>; omit for the active tab.
            # ?client_id=... picks one browser window; omit to take the first that answers.
            result = await _ask_browser("mets-open-workflows-request-graph", {
                "path": request.query.get("path"), "client_id": request.query.get("client_id"),
            })
            if result.get("source") == "not_loaded" and result.get("path"):
                # Tab not visited since page load - what it would show is the saved file.
                try:
                    result["graph"] = _read_user_file(result["path"])
                    result["source"] = "saved_file"
                except (OSError, ValueError) as e:
                    result["error"] = f"could not read saved file: {e}"
            status = 504 if result.get("timeout") else 404 if result.get("error") else 200
            return web.json_response(result, status=status)

        @routes.post("/mets/open_workflows/edit")
        async def edit(request: web.Request) -> web.Response:
            # Body: {"ops": [{"op": "set_widget", "node": 12, "widget": "text", "value": "..."},
            #                {"op": "set_title", "node": 12, "title": "..."},
            #                {"op": "set_mode", "node": 12, "mode": "always" | "mute" | "bypass"}],
            #        "path": optional - must be the active tab, "client_id": optional}
            # Applied to the active tab's live canvas as one undoable change, all-or-nothing.
            data = await request.json()
            result = await _ask_browser("mets-open-workflows-edit", {
                "ops": data.get("ops", []), "path": data.get("path"), "client_id": data.get("client_id"),
            })
            status = 504 if result.get("timeout") else 400 if result.get("error") else 200
            return web.json_response(result, status=status)

        # Browser replies to _ask_browser(). Deliberately not sharing a path with any GET:
        # LG_HotReload (before the upstream fix) swapped route handlers by path only.
        @routes.post("/mets/open_workflows/response")
        async def response(request: web.Request) -> web.Response:
            data = await request.json()
            future = _pending.get(data.get("request_id"))
            if future and not future.done():
                future.set_result(data)
            return web.json_response({"ok": True})

    except Exception:
        pass


_register_routes()
