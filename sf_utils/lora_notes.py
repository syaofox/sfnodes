import json
import os
import folder_paths

from aiohttp import web
from .logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _get_notes_path() -> str:
    p = os.path.join(folder_paths.get_user_directory(), "sfnodes", "lora_notes.json")
    logger.debug(f"Notes path: {p}")
    return p


# ---------------------------------------------------------------------------
# Notes CRUD
# ---------------------------------------------------------------------------

def load_all_notes() -> dict:
    path = _get_notes_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_all_notes(notes: dict) -> None:
    path = _get_notes_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(notes)} note(s) to {path}")


def get_custom_notes(lora_name: str) -> dict | None:
    notes = load_all_notes()
    entry = notes.get(lora_name)
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        return {"description": entry}
    return None


def set_custom_notes(lora_name: str, data: dict) -> dict:
    notes = load_all_notes()
    if data and any(v for v in data.values()):
        notes[lora_name] = {k: v for k, v in data.items() if v}
    else:
        notes.pop(lora_name, None)
    save_all_notes(notes)
    return get_custom_notes(lora_name) or {}


# ---------------------------------------------------------------------------
# Embedded metadata reader (lightweight, no tensor loading)
# ---------------------------------------------------------------------------

def read_embedded_metadata(lora_name: str) -> dict | None:
    if not lora_name or not lora_name.endswith(".safetensors"):
        return None
    try:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.isfile(lora_path):
            return None
        from comfy.utils import safetensors_header
        header_bytes = safetensors_header(lora_path, max_size=1024 * 1024)
        if header_bytes is None:
            return None
        header = json.loads(header_bytes)
        return header.get("__metadata__", {}) or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trigger word extraction (mirrors frontend logic for consistency)
# ---------------------------------------------------------------------------

def _extract_trigger_words(meta: dict) -> str:
    if not meta:
        return ""

    tag_freq = meta.get("ss_tag_frequency")
    if isinstance(tag_freq, str):
        try:
            data = json.loads(tag_freq)
            words = set()
            for tags in data.values():
                if isinstance(tags, dict):
                    words.update(tags.keys())
            if words:
                return ", ".join(sorted(words)[:50])
        except Exception:
            pass

    tw = meta.get("trained_words")
    if isinstance(tw, str):
        if tw.strip().startswith("["):
            try:
                parsed = json.loads(tw)
                if isinstance(parsed, list):
                    return ", ".join(str(x) for x in parsed)
            except Exception:
                pass
        parts = [w.strip() for w in tw.split(",") if w.strip()]
        if parts:
            return ", ".join(parts)
    if isinstance(tw, list):
        return ", ".join(str(x) for x in tw)

    return ""


# ---------------------------------------------------------------------------
# Merge logic: custom notes > embedded metadata (field-level merge)
# ---------------------------------------------------------------------------

def get_merged_metadata(lora_name: str) -> dict:
    custom = get_custom_notes(lora_name)
    embedded = read_embedded_metadata(lora_name)

    not_found = not custom and not embedded
    if not_found:
        return {"_not_found": True}

    result = {}

    if embedded:
        result["trigger_words"] = _extract_trigger_words(embedded) or ""
        result["description"] = (
            embedded.get("modelspec.description")
            or embedded.get("modelspec.title")
            or embedded.get("modelspec_description")
            or embedded.get("modelspec_title")
            or ""
        )
        result["base_model"] = (
            embedded.get("ss_base_model_version")
            or embedded.get("modelspec.base_model")
            or embedded.get("modelspec_base_model")
            or embedded.get("base_model")
            or ""
        )
        result["source_url"] = embedded.get("source_url", "")
        result["_has_embedded"] = True

    if custom:
        if custom.get("trigger_words"):
            result["trigger_words"] = custom["trigger_words"]
        if custom.get("description"):
            result["description"] = custom["description"]
        if custom.get("base_model"):
            result["base_model"] = custom["base_model"]
        if custom.get("source_url"):
            result["source_url"] = custom["source_url"]
        result["_has_custom"] = True

    return result


# ---------------------------------------------------------------------------
# HTTP route registration
# ---------------------------------------------------------------------------

def _register_routes():
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            logger.warning("PromptServer instance not available, routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/lora_notes")
        async def _get_notes(request: web.Request) -> web.Response:
            try:
                filename = request.query.get("filename", "")
                if not filename:
                    return web.json_response({"error": "filename required"}, status=400)
                data = get_merged_metadata(filename)
                return web.json_response(data)
            except Exception as e:
                logger.error(f"GET /api/sfnodes/lora_notes failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.post("/api/sfnodes/lora_notes")
        async def _save_notes(request: web.Request) -> web.Response:
            try:
                filename = request.query.get("filename", "")
                if not filename:
                    return web.json_response({"error": "filename required"}, status=400)
                body = await request.json()
                if not isinstance(body, dict):
                    return web.json_response({"error": "json object required"}, status=400)
                set_custom_notes(filename, body)
                data = get_merged_metadata(filename)
                logger.info(f"Saved notes for {filename}")
                return web.json_response(data)
            except Exception as e:
                logger.error(f"POST /api/sfnodes/lora_notes failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        @routes.delete("/api/sfnodes/lora_notes")
        async def _delete_notes(request: web.Request) -> web.Response:
            try:
                filename = request.query.get("filename", "")
                if not filename:
                    return web.json_response({"error": "filename required"}, status=400)
                notes = load_all_notes()
                notes.pop(filename, None)
                save_all_notes(notes)
                data = get_merged_metadata(filename)
                return web.json_response(data)
            except Exception as e:
                logger.error(f"DELETE /api/sfnodes/lora_notes failed: {e}")
                return web.json_response({"error": "internal error"}, status=500)

        logger.info("LoRA notes API routes registered")

    except Exception as e:
        logger.error(f"Failed to register LoRA notes routes: {e}")


_register_routes()
