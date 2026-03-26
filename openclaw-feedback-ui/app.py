from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATE_DIR = Path(os.environ.get("OPENCLAW_RL_UI_STATE_DIR", APP_DIR / "state")).expanduser()
PROFILES_DIR = STATE_DIR / "profiles"
LEGACY_GUIDANCE_FILE = STATE_DIR / "steering_notes.txt"
LEGACY_SYSTEM_PROMPT_FILE = STATE_DIR / "system_prompt.txt"
ACTIVE_PROFILE_FILE = STATE_DIR / "active_profile.txt"
LEGACY_THINKING_FILE = STATE_DIR / "thinking_enabled.txt"
FEEDBACK_LOG_FILE_VALUE = os.environ.get("OPENCLAW_RL_UI_FEEDBACK_LOG_FILE", "").strip()
FEEDBACK_LOG_FILE = Path(FEEDBACK_LOG_FILE_VALUE).expanduser() if FEEDBACK_LOG_FILE_VALUE else None

COOKIE_NAME = "openclaw_rl_ui_session"
PROXY_BASE_URL = os.environ.get("OPENCLAW_RL_PROXY_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
BACKEND_MODE = os.environ.get("OPENCLAW_RL_UI_BACKEND_MODE", "rl_proxy").strip().lower()
SPLIT_BACKEND_MODES = {"split_trainer_api", "split_backends", "sglang_trainer"}
SPLIT_BACKENDS = BACKEND_MODE in SPLIT_BACKEND_MODES
CHAT_BACKEND_MODE = os.environ.get(
    "OPENCLAW_RL_CHAT_BACKEND_MODE",
    "openai_chat" if SPLIT_BACKENDS else BACKEND_MODE,
).strip().lower()
TRAINING_BACKEND_MODE = os.environ.get(
    "OPENCLAW_RL_TRAINING_BACKEND_MODE",
    "trainer_api" if SPLIT_BACKENDS else BACKEND_MODE,
).strip().lower()
CHAT_PROXY_BASE_URL = os.environ.get("OPENCLAW_RL_CHAT_PROXY_BASE_URL", PROXY_BASE_URL).rstrip("/")
TRAINING_PROXY_BASE_URL = os.environ.get("OPENCLAW_RL_TRAINING_PROXY_BASE_URL", PROXY_BASE_URL).rstrip("/")
PROFILE_PROXY_BASE_URL = os.environ.get("OPENCLAW_RL_PROFILE_PROXY_BASE_URL", TRAINING_PROXY_BASE_URL).rstrip("/")
CHAT_PATH = os.environ.get("OPENCLAW_RL_CHAT_PATH", "/v1/chat/completions")
FEEDBACK_PATH = os.environ.get("OPENCLAW_RL_FEEDBACK_PATH", "/v1/feedback")
DEFAULT_CHAT_HEALTH_PATH = "/healthz" if CHAT_BACKEND_MODE == "rl_proxy" else "/health"
DEFAULT_TRAINING_HEALTH_PATH = "/healthz" if TRAINING_BACKEND_MODE == "rl_proxy" else "/health"
CHAT_HEALTH_PATH = os.environ.get("OPENCLAW_RL_CHAT_HEALTH_PATH", DEFAULT_CHAT_HEALTH_PATH)
TRAINING_HEALTH_PATH = os.environ.get("OPENCLAW_RL_TRAINING_HEALTH_PATH", DEFAULT_TRAINING_HEALTH_PATH)
CHAT_LOAD_LORA_PATH = os.environ.get("OPENCLAW_RL_CHAT_LOAD_LORA_PATH", "/load_lora_adapter")
CHAT_URL = f"{CHAT_PROXY_BASE_URL}{CHAT_PATH}"
CHAT_HEALTH_URL = f"{CHAT_PROXY_BASE_URL}{CHAT_HEALTH_PATH}"
CHAT_LOAD_LORA_URL = f"{CHAT_PROXY_BASE_URL}{CHAT_LOAD_LORA_PATH}"
TRAINING_HEALTH_URL = f"{TRAINING_PROXY_BASE_URL}{TRAINING_HEALTH_PATH}"
FEEDBACK_URL = f"{TRAINING_PROXY_BASE_URL}{FEEDBACK_PATH}"
API_KEY = os.environ.get("OPENCLAW_RL_API_KEY", "openclaw-local")
CHAT_API_KEY = os.environ.get("OPENCLAW_RL_CHAT_API_KEY", API_KEY)
TRAINING_API_KEY = os.environ.get("OPENCLAW_RL_TRAINING_API_KEY", API_KEY)
MODEL_NAME = os.environ.get("OPENCLAW_RL_MODEL", "qwen3-0.6b-local")
DEFAULT_PROFILE_ID = os.environ.get("OPENCLAW_RL_DEFAULT_PROFILE_ID", "default").strip() or "default"
REQUEST_TIMEOUT = float(os.environ.get("OPENCLAW_RL_UI_TIMEOUT_SECONDS", "600"))
MAX_HISTORY_TURNS = int(os.environ.get("OPENCLAW_RL_UI_MAX_HISTORY_TURNS", "16"))
UI_MAX_TOKENS = int(os.environ.get("OPENCLAW_RL_UI_MAX_TOKENS", "384"))
UI_THINKING_MAX_TOKENS = int(
    os.environ.get("OPENCLAW_RL_UI_THINKING_MAX_TOKENS", str(max(UI_MAX_TOKENS * 2, UI_MAX_TOKENS + 192)))
)
UI_HOST = os.environ.get("OPENCLAW_RL_UI_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("OPENCLAW_RL_UI_PORT", "30001"))
RL_SERVICE_NAME = os.environ.get("OPENCLAW_RL_SERVICE_NAME", "openclaw-rl.service")
CHAT_SERVICE_NAME = os.environ.get(
    "OPENCLAW_RL_CHAT_SERVICE_NAME",
    RL_SERVICE_NAME if CHAT_BACKEND_MODE == BACKEND_MODE else "",
).strip()
TRAINING_SERVICE_NAME = os.environ.get(
    "OPENCLAW_RL_TRAINING_SERVICE_NAME",
    RL_SERVICE_NAME if TRAINING_BACKEND_MODE == BACKEND_MODE else "",
).strip()
FORCE_NO_THINK = os.environ.get("OPENCLAW_RL_FORCE_NO_THINK", "1").strip().lower() in {"1", "true", "yes", "on"}
USE_NATIVE_CHAT_THINKING = os.environ.get("OPENCLAW_RL_USE_NATIVE_CHAT_THINKING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROFILES_SUPPORTED = TRAINING_BACKEND_MODE == "trainer_api"
PROFILES_URL = f"{PROFILE_PROXY_BASE_URL}/v1/profiles"
PROFILE_CREATE_URL = f"{PROFILE_PROXY_BASE_URL}/v1/profiles"
PROFILE_SELECT_URL = f"{PROFILE_PROXY_BASE_URL}/v1/profiles/select"
DEFAULT_SESSION_TITLE = "New chat"
SESSION_TITLE_WORDS = 7
SESSION_TITLE_MAX_CHARS = 64
RAW_EXTRA_CHAT_BODY = os.environ.get("OPENCLAW_RL_UI_EXTRA_CHAT_BODY_JSON", "").strip()
try:
    EXTRA_CHAT_BODY = json.loads(RAW_EXTRA_CHAT_BODY) if RAW_EXTRA_CHAT_BODY else {}
except json.JSONDecodeError as exc:
    raise RuntimeError(f"Invalid OPENCLAW_RL_UI_EXTRA_CHAT_BODY_JSON: {exc}") from exc
if not isinstance(EXTRA_CHAT_BODY, dict):
    raise RuntimeError("OPENCLAW_RL_UI_EXTRA_CHAT_BODY_JSON must decode to a JSON object")


app = FastAPI(title="OpenClaw-RL Feedback UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_state_lock = RLock()
_browser_sessions: dict[str, dict[str, Any]] = {}
_last_service_restart_at: dict[str, float] = {}
_last_loaded_chat_overlay_path = ""


class ChatRequest(BaseModel):
    prompt: str
    thinking_enabled: bool | None = None


class FeedbackRequest(BaseModel):
    assistant_turn_id: str = ""
    score: int | None = None
    rating: str | None = None
    note: str = ""


class GuidanceRequest(BaseModel):
    text: str = ""


class SystemPromptRequest(BaseModel):
    text: str = ""


class ProfileSettingsRequest(BaseModel):
    guidance_text: str = ""
    system_prompt_text: str = ""


class ProfileCreateRequest(BaseModel):
    name: str = ""
    select_after_create: bool = True


class ProfileSelectRequest(BaseModel):
    profile_id: str = ""


class SessionSelectRequest(BaseModel):
    session_id: str = ""


class ThinkingRequest(BaseModel):
    enabled: bool = False


def _profile_dir(profile_id: str) -> Path:
    return PROFILES_DIR / profile_id


def _profile_guidance_file(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "steering_notes.txt"


def _profile_system_prompt_file(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "system_prompt.txt"


def _profile_thinking_file(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "thinking_enabled.txt"


def _profile_sessions_dir(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "sessions"


def _profile_active_session_file(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "active_session.txt"


def _session_file(profile_id: str, session_id: str) -> Path:
    return _profile_sessions_dir(profile_id) / f"{session_id}.json"


def _ensure_profile_storage() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    default_guidance_path = _profile_guidance_file(DEFAULT_PROFILE_ID)
    if LEGACY_GUIDANCE_FILE.exists() and not default_guidance_path.exists():
        default_guidance_path.parent.mkdir(parents=True, exist_ok=True)
        default_guidance_path.write_text(LEGACY_GUIDANCE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    default_system_prompt_path = _profile_system_prompt_file(DEFAULT_PROFILE_ID)
    if LEGACY_SYSTEM_PROMPT_FILE.exists() and not default_system_prompt_path.exists():
        default_system_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        default_system_prompt_path.write_text(LEGACY_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    default_thinking_path = _profile_thinking_file(DEFAULT_PROFILE_ID)
    if LEGACY_THINKING_FILE.exists() and not default_thinking_path.exists():
        default_thinking_path.parent.mkdir(parents=True, exist_ok=True)
        default_thinking_path.write_text(LEGACY_THINKING_FILE.read_text(encoding="utf-8"), encoding="utf-8")


def _load_active_profile_id() -> str:
    _ensure_profile_storage()
    try:
        profile_id = ACTIVE_PROFILE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return DEFAULT_PROFILE_ID
    return profile_id or DEFAULT_PROFILE_ID


def _save_active_profile_id(profile_id: str) -> str:
    _ensure_profile_storage()
    ACTIVE_PROFILE_FILE.write_text(f"{profile_id}\n", encoding="utf-8")
    return profile_id


def _load_guidance_text(profile_id: str | None = None) -> str:
    resolved_profile_id = profile_id or _load_active_profile_id()
    guidance_file = _profile_guidance_file(resolved_profile_id)
    try:
        return guidance_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _save_guidance_text(profile_id: str, text: str) -> str:
    clean_text = text.strip()
    guidance_file = _profile_guidance_file(profile_id)
    guidance_file.parent.mkdir(parents=True, exist_ok=True)
    guidance_file.write_text(f"{clean_text}\n" if clean_text else "", encoding="utf-8")
    return clean_text


def _load_system_prompt_text(profile_id: str | None = None) -> str:
    resolved_profile_id = profile_id or _load_active_profile_id()
    system_prompt_file = _profile_system_prompt_file(resolved_profile_id)
    try:
        return system_prompt_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _save_system_prompt_text(profile_id: str, text: str) -> str:
    clean_text = text.strip()
    system_prompt_file = _profile_system_prompt_file(profile_id)
    system_prompt_file.parent.mkdir(parents=True, exist_ok=True)
    system_prompt_file.write_text(f"{clean_text}\n" if clean_text else "", encoding="utf-8")
    return clean_text


def _load_thinking_enabled(profile_id: str | None = None) -> bool:
    resolved_profile_id = profile_id or _load_active_profile_id()
    try:
        raw = _profile_thinking_file(resolved_profile_id).read_text(encoding="utf-8").strip().lower()
    except FileNotFoundError:
        return False
    return raw in {"1", "true", "yes", "on"}


def _save_thinking_enabled(profile_id: str, enabled: bool) -> bool:
    thinking_file = _profile_thinking_file(profile_id)
    thinking_file.parent.mkdir(parents=True, exist_ok=True)
    thinking_file.write_text("1\n" if enabled else "0\n", encoding="utf-8")
    return enabled


def _session_title_from_prompt(prompt: str) -> str:
    clean = re.sub(r"\s+", " ", prompt).strip()
    if not clean:
        return DEFAULT_SESSION_TITLE
    words = clean.split(" ")
    title = " ".join(words[:SESSION_TITLE_WORDS]).strip()
    if len(words) > SESSION_TITLE_WORDS or len(clean) > len(title):
        title = f"{title}..."
    title = title[:SESSION_TITLE_MAX_CHARS].rstrip()
    return title or DEFAULT_SESSION_TITLE


def _default_session_record(
    *,
    session_id: str | None = None,
    title: str = DEFAULT_SESSION_TITLE,
    created_at: float | None = None,
) -> dict[str, Any]:
    timestamp = float(created_at or time.time())
    return {
        "id": session_id or uuid.uuid4().hex,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "transcript": [],
        "context_messages": [],
        "feedback_candidates": {},
    }


def _normalize_session_record(payload: dict[str, Any] | None, *, session_id: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _default_session_record(session_id=session_id)

    created_at = float(payload.get("created_at") or time.time())
    updated_at = float(payload.get("updated_at") or created_at)
    transcript = payload.get("transcript")
    context_messages = payload.get("context_messages")
    feedback_candidates = payload.get("feedback_candidates")
    session = {
        "id": str(payload.get("id") or session_id or uuid.uuid4().hex),
        "title": str(payload.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
        "created_at": created_at,
        "updated_at": updated_at,
        "transcript": transcript if isinstance(transcript, list) else [],
        "context_messages": context_messages if isinstance(context_messages, list) else [],
        "feedback_candidates": feedback_candidates if isinstance(feedback_candidates, dict) else {},
    }
    return session


def _load_session_record(profile_id: str, session_id: str) -> dict[str, Any]:
    path = _session_file(profile_id, session_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        session = _default_session_record(session_id=session_id)
        _save_session_record(profile_id, session)
        return session
    except json.JSONDecodeError:
        session = _default_session_record(session_id=session_id)
        _save_session_record(profile_id, session)
        return session
    return _normalize_session_record(payload, session_id=session_id)


def _save_session_record(profile_id: str, session: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_session_record(session, session_id=str(session.get("id") or uuid.uuid4().hex))
    normalized["updated_at"] = float(normalized.get("updated_at") or time.time())
    sessions_dir = _profile_sessions_dir(profile_id)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    _session_file(profile_id, normalized["id"]).write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized


def _session_summary(session: dict[str, Any]) -> dict[str, Any]:
    transcript = session.get("transcript") or []
    preview = ""
    for item in transcript:
        if item.get("role") == "assistant" and str(item.get("content") or "").strip():
            preview = str(item.get("content") or "").strip()
            break
    if not preview:
        for item in transcript:
            if str(item.get("content") or "").strip():
                preview = str(item.get("content") or "").strip()
                break
    preview = re.sub(r"\s+", " ", preview).strip()
    return {
        "id": str(session.get("id") or ""),
        "title": str(session.get("title") or DEFAULT_SESSION_TITLE).strip() or DEFAULT_SESSION_TITLE,
        "created_at": float(session.get("created_at") or time.time()),
        "updated_at": float(session.get("updated_at") or time.time()),
        "message_count": len(transcript),
        "preview": preview[:120],
    }


def _upsert_session_summary(
    sessions: list[dict[str, Any]] | None,
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = _session_summary(session)
    next_sessions = [
        item
        for item in (sessions or [])
        if str(item.get("id") or "").strip() != summary["id"]
    ]
    next_sessions.append(summary)
    next_sessions.sort(
        key=lambda item: (float(item.get("updated_at") or 0), float(item.get("created_at") or 0)),
        reverse=True,
    )
    return next_sessions


def _list_profile_sessions(profile_id: str) -> list[dict[str, Any]]:
    sessions_dir = _profile_sessions_dir(profile_id)
    if not sessions_dir.exists():
        return []

    sessions: list[dict[str, Any]] = []
    for path in sessions_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        sessions.append(_session_summary(_normalize_session_record(payload)))

    sessions.sort(key=lambda item: (float(item.get("updated_at") or 0), float(item.get("created_at") or 0)), reverse=True)
    return sessions


def _load_profile_active_session_id(profile_id: str) -> str | None:
    active_file = _profile_active_session_file(profile_id)
    try:
        session_id = active_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return session_id or None


def _save_profile_active_session_id(profile_id: str, session_id: str) -> str:
    active_file = _profile_active_session_file(profile_id)
    active_file.parent.mkdir(parents=True, exist_ok=True)
    active_file.write_text(f"{session_id}\n", encoding="utf-8")
    return session_id


def _create_profile_session(profile_id: str, *, title: str = DEFAULT_SESSION_TITLE) -> dict[str, Any]:
    session = _default_session_record(title=title)
    saved = _save_session_record(profile_id, session)
    _save_profile_active_session_id(profile_id, saved["id"])
    return saved


def _ensure_profile_active_session(profile_id: str) -> dict[str, Any]:
    _ensure_profile_storage()
    _profile_dir(profile_id).mkdir(parents=True, exist_ok=True)
    _profile_sessions_dir(profile_id).mkdir(parents=True, exist_ok=True)

    active_session_id = _load_profile_active_session_id(profile_id)
    if active_session_id:
        session_path = _session_file(profile_id, active_session_id)
        if session_path.exists():
            return _load_session_record(profile_id, active_session_id)

    sessions = _list_profile_sessions(profile_id)
    if sessions:
        first_session_id = str(sessions[0]["id"])
        _save_profile_active_session_id(profile_id, first_session_id)
        return _load_session_record(profile_id, first_session_id)

    return _create_profile_session(profile_id)


def _load_state_from_session(state: dict[str, Any], profile_id: str, session_id: str | None = None) -> dict[str, Any]:
    _ensure_profile_storage()
    profile_id = profile_id or DEFAULT_PROFILE_ID

    if session_id:
        session = _load_session_record(profile_id, session_id)
        _save_profile_active_session_id(profile_id, session["id"])
    else:
        session = _ensure_profile_active_session(profile_id)

    state["active_profile_id"] = profile_id
    state["guidance_text"] = _load_guidance_text(profile_id)
    state["system_prompt_text"] = _load_system_prompt_text(profile_id)
    state["thinking_enabled"] = _load_thinking_enabled(profile_id)
    state["active_session_id"] = session["id"]
    state["sessions"] = _list_profile_sessions(profile_id)
    state["transcript"] = list(session.get("transcript") or [])
    state["context_messages"] = list(session.get("context_messages") or [])
    state["feedback_candidates"] = dict(session.get("feedback_candidates") or {})
    state["busy"] = False
    _touch_state(state)
    return session


def _new_browser_state(profile_id: str | None = None) -> dict[str, Any]:
    state = {
        "transcript": [],
        "context_messages": [],
        "feedback_candidates": {},
        "sessions": [],
        "active_session_id": "",
        "busy": False,
        "guidance_text": "",
        "system_prompt_text": "",
        "thinking_enabled": False,
        "active_profile_id": profile_id or _load_active_profile_id(),
        "updated_at": time.time(),
    }
    _load_state_from_session(state, state["active_profile_id"])
    return state


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    awaiting_feedback = any(
        item.get("role") == "assistant" and item.get("feedback_pending")
        for item in state["transcript"]
    )
    return {
        "transcript": state["transcript"],
        "awaiting_feedback": awaiting_feedback,
        "busy": state["busy"],
        "model": MODEL_NAME,
        "backend_mode": BACKEND_MODE,
        "chat_backend_mode": CHAT_BACKEND_MODE,
        "training_backend_mode": TRAINING_BACKEND_MODE,
        "proxy_base_url": CHAT_PROXY_BASE_URL,
        "chat_proxy_base_url": CHAT_PROXY_BASE_URL,
        "training_proxy_base_url": TRAINING_PROXY_BASE_URL,
        "guidance_text": state.get("guidance_text", ""),
        "system_prompt_text": state.get("system_prompt_text", ""),
        "thinking_enabled": bool(state.get("thinking_enabled")),
        "active_profile_id": state.get("active_profile_id", DEFAULT_PROFILE_ID),
        "profiles_supported": PROFILES_SUPPORTED,
        "sessions": state.get("sessions", []),
        "active_session_id": state.get("active_session_id", ""),
    }


def _is_split_backend() -> bool:
    return SPLIT_BACKENDS or CHAT_PROXY_BASE_URL != TRAINING_PROXY_BASE_URL or CHAT_BACKEND_MODE != TRAINING_BACKEND_MODE


def _touch_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()


def _persist_active_session_state(state: dict[str, Any], *, title_from_prompt: str | None = None) -> None:
    profile_id = str(state.get("active_profile_id") or DEFAULT_PROFILE_ID)
    session_id = str(state.get("active_session_id") or "").strip()
    if not session_id:
        session = _ensure_profile_active_session(profile_id)
        session_id = session["id"]
        state["active_session_id"] = session_id

    session = _load_session_record(profile_id, session_id)
    had_user_messages = any(item.get("role") == "user" for item in session.get("transcript") or [])
    session["transcript"] = list(state.get("transcript") or [])
    session["context_messages"] = list(state.get("context_messages") or [])
    session["feedback_candidates"] = dict(state.get("feedback_candidates") or {})
    session["updated_at"] = time.time()
    if title_from_prompt and not had_user_messages:
        session["title"] = _session_title_from_prompt(title_from_prompt)
    saved_session = _save_session_record(profile_id, session)
    _save_profile_active_session_id(profile_id, session_id)
    state["sessions"] = _upsert_session_summary(state.get("sessions"), saved_session)
    _touch_state(state)


def _create_and_activate_session(state: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(state.get("active_profile_id") or DEFAULT_PROFILE_ID)
    session = _create_profile_session(profile_id)
    _load_state_from_session(state, profile_id, session["id"])
    return session


def _set_state_profile(state: dict[str, Any], profile_id: str) -> None:
    _save_active_profile_id(profile_id)
    _load_state_from_session(state, profile_id)


def _select_state_session(state: dict[str, Any], session_id: str) -> None:
    profile_id = str(state.get("active_profile_id") or DEFAULT_PROFILE_ID)
    _load_state_from_session(state, profile_id, session_id)


def _ensure_browser_session(request: Request) -> tuple[str, dict[str, Any], bool]:
    browser_id = request.cookies.get(COOKIE_NAME)
    created = False
    with _state_lock:
        if not browser_id or browser_id not in _browser_sessions:
            browser_id = uuid.uuid4().hex
            _browser_sessions[browser_id] = _new_browser_state()
            created = True
        state = _browser_sessions[browser_id]
        _touch_state(state)
        snapshot = state
    return browser_id, snapshot, created


def _set_cookie_on_response(response: Response, browser_id: str, created: bool) -> Response:
    if created:
        response.set_cookie(
            key=COOKIE_NAME,
            value=browser_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 14,
        )
    return response


def _response_with_cookie(payload: Any, browser_id: str, created: bool) -> JSONResponse:
    response = JSONResponse(payload)
    _set_cookie_on_response(response, browser_id, created)
    return response


def _trim_context(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) <= MAX_HISTORY_TURNS * 2:
        return messages
    return messages[-MAX_HISTORY_TURNS * 2 :]


def _normalize_openai_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_chunks: list[str] = []
    normalized: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role == "system":
            content = str(message.get("content") or "").strip()
            if content:
                system_chunks.append(content)
            continue
        normalized.append(message)

    if not system_chunks:
        return normalized

    return [
        {
            "role": "system",
            "content": "\n\n".join(system_chunks),
        },
        *normalized,
    ]


def _feedback_text(score: int, note: str) -> str:
    prefix = (
        f"[OpenClaw feedback] score={score}/10. "
        "This is explicit user feedback from the local rating UI. "
        "Interpret 1 as very bad, 5 as neutral, and 10 as excellent."
    )
    clean_note = note.strip()
    if not clean_note:
        return prefix
    return f"{prefix} Comment: {clean_note}"


def _guidance_message(guidance_text: str) -> dict[str, str] | None:
    clean_text = guidance_text.strip()
    if not clean_text:
        return None
    return {
        "role": "system",
        "content": (
            "Standing user guidance for this local training session. Treat this as durable "
            "steering context, including preferred approaches, pitfalls to avoid, and "
            "fictional example scenarios to generalize from.\n\n"
            f"{clean_text}"
        ),
    }


def _system_prompt_message(system_prompt_text: str) -> dict[str, str] | None:
    clean_text = system_prompt_text.strip()
    if not clean_text:
        return None
    return {
        "role": "system",
        "content": clean_text,
    }


def _chat_control_message(thinking_enabled: bool) -> dict[str, str] | None:
    if thinking_enabled and not FORCE_NO_THINK:
        if not USE_NATIVE_CHAT_THINKING:
            return {
                "role": "system",
                "content": (
                    "Thinking mode is on. Answer using exactly this structure:\n"
                    "Thinking:\n"
                    "- one to three short bullets\n"
                    "Final Answer:\n"
                    "<the user-facing answer>\n"
                    "Keep the Thinking section brief and always include Final Answer."
                ),
            }
        return None
    return {
        "role": "system",
        "content": (
            "For this local feedback UI, answer in non-thinking mode and return only the final answer. "
            "Do not expose chain-of-thought or a <think> block. /no_think"
        ),
    }


_FINAL_ANSWER_PATTERNS = (
    re.compile(r"(?:^|\n)\s*(?:Final Answer|Final Response|Answer|Response|Construct Response|Output)\s*:\s*(.+)", re.I),
    re.compile(r"(?:^|\n)\s*\*+\s*(?:Final Answer|Final Response|Answer|Response|Construct Response|Output)\s*:\s*(.+)", re.I),
)


def _extract_final_answer(reasoning_text: str) -> str:
    clean_text = reasoning_text.strip()
    if not clean_text:
        return ""

    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = [match.group(1).strip() for match in pattern.finditer(clean_text) if match.group(1).strip()]
        if matches:
            return matches[-1].strip("\"' ")
    return ""


def _looks_like_reasoning_leak(content: str, reasoning_text: str) -> bool:
    clean_content = content.strip()
    clean_reasoning = reasoning_text.strip()
    if not clean_content:
        return False
    if clean_reasoning and clean_content == clean_reasoning:
        return True
    if clean_content.lower().startswith("thinking process:"):
        return True
    return bool(clean_reasoning) and bool(
        re.search(r"(analyze the request|final decision|construct response|instruction priority)", clean_content, re.I)
    )


def _split_visible_thinking_response(text: str, *, final: bool) -> tuple[str, str]:
    clean_text = text.strip()
    if not clean_text:
        return "", ""

    matches = list(re.finditer(r"(?:^|\n)\s*Final Answer\s*:\s*", clean_text, flags=re.I))
    if matches:
        match = matches[-1]
        reasoning = clean_text[: match.start()].strip()
        content = clean_text[match.end() :].strip()
        reasoning = re.sub(r"^\s*Thinking\s*:\s*", "", reasoning, flags=re.I).strip()
        return reasoning, content

    partial_final_match = re.search(r"(?:^|\n)\s*Final(?:\s+Answer)?\s*:?\s*$", clean_text, flags=re.I)
    if partial_final_match and not final:
        reasoning = clean_text[: partial_final_match.start()].strip()
        reasoning = re.sub(r"^\s*Thinking\s*:?\s*", "", reasoning, flags=re.I).strip()
        return reasoning, ""

    thinking_match = re.match(r"^\s*Thinking\s*:\s*(.*)$", clean_text, flags=re.I | re.S)
    if thinking_match:
        reasoning = thinking_match.group(1).strip()
        if not final:
            partial_final_match = re.search(r"(?:^|\n)\s*Final(?:\s+Answer)?\s*:?\s*$", reasoning, flags=re.I)
            if partial_final_match:
                reasoning = reasoning[: partial_final_match.start()].strip()
        return reasoning, ""

    normalized_text = clean_text.lower()
    if not final:
        compact_text = re.sub(r"\s+", " ", normalized_text).strip()
        if compact_text and "thinking:".startswith(compact_text):
            return "", ""

    if final:
        return "", clean_text
    return "", clean_text


def _resolve_assistant_text(content: str, reasoning_text: str) -> str:
    clean_content = content.strip()
    clean_reasoning = reasoning_text.strip()
    if clean_content and not _looks_like_reasoning_leak(clean_content, clean_reasoning):
        return clean_content

    extracted_answer = _extract_final_answer(clean_reasoning)
    if extracted_answer:
        return extracted_answer
    return ""


_TOKEN_ESTIMATE_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _estimate_token_count(text: str) -> int:
    clean_text = str(text or "").strip()
    if not clean_text:
        return 0
    return len(_TOKEN_ESTIMATE_PATTERN.findall(clean_text))


def _stream_metrics_fallback(
    metrics: dict[str, Any] | None,
    *,
    visible_text: str,
    reasoning_text: str,
    started_at: float,
    first_token_at: float | None,
) -> dict[str, Any]:
    merged = dict(metrics) if isinstance(metrics, dict) else {}

    visible_tokens = merged.get("visible_tokens")
    if not isinstance(visible_tokens, (int, float)) or visible_tokens < 0:
        visible_tokens = _estimate_token_count(visible_text)
    visible_tokens = max(int(round(float(visible_tokens))), 0)

    generated_tokens = merged.get("generated_tokens")
    if not isinstance(generated_tokens, (int, float)) or generated_tokens < visible_tokens:
        combined_text = "\n".join(part for part in (reasoning_text.strip(), visible_text.strip()) if part)
        generated_tokens = _estimate_token_count(combined_text) if combined_text else visible_tokens
    generated_tokens = max(int(round(float(generated_tokens))), visible_tokens)

    elapsed_seconds = merged.get("elapsed_seconds")
    elapsed_started_at = first_token_at if first_token_at is not None else started_at
    if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds <= 0:
        elapsed_seconds = max(time.perf_counter() - elapsed_started_at, 1e-3)
    elapsed_seconds = round(float(elapsed_seconds), 3)

    tokens_per_second = merged.get("tokens_per_second")
    if not isinstance(tokens_per_second, (int, float)) or tokens_per_second <= 0:
        tokens_per_second = round(generated_tokens / max(elapsed_seconds, 1e-3), 2) if generated_tokens > 0 else 0.0
    else:
        tokens_per_second = round(float(tokens_per_second), 2)

    merged.update(
        {
            "visible_tokens": visible_tokens,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": tokens_per_second,
        }
    )
    return merged


def _chat_backend_label() -> str:
    return "Chat backend" if _is_split_backend() else "Training backend"


def _training_backend_label() -> str:
    return "Training backend" if _is_split_backend() else "Backend"


def _get_loaded_chat_overlay_path() -> str:
    with _state_lock:
        return _last_loaded_chat_overlay_path


def _set_loaded_chat_overlay_path(path: str) -> None:
    global _last_loaded_chat_overlay_path
    with _state_lock:
        _last_loaded_chat_overlay_path = path


def _clear_loaded_chat_overlay_path() -> None:
    _set_loaded_chat_overlay_path("")


def _should_resync_chat_overlay(detail: str) -> bool:
    lowered = detail.lower()
    markers = (
        "never been loaded",
        "not loaded",
        "requested lora adapters are not loaded",
    )
    return any(marker in lowered for marker in markers)


def _build_chat_request_body(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    thinking_enabled: bool,
    max_tokens: int | None = None,
    stream: bool,
) -> dict[str, Any]:
    request_messages = list(messages)
    if CHAT_BACKEND_MODE == "openai_chat":
        request_messages = _normalize_openai_chat_messages(request_messages)

    native_thinking = USE_NATIVE_CHAT_THINKING and CHAT_BACKEND_MODE == "openai_chat"
    effective_thinking = bool(thinking_enabled) and not FORCE_NO_THINK and native_thinking
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": request_messages,
        "stream": stream,
        "enable_thinking": effective_thinking,
    }
    if CHAT_BACKEND_MODE == "rl_proxy":
        body.update(
            {
                "session_id": session_id,
                "turn_type": turn_type,
                "session_done": session_done,
            }
        )
    elif CHAT_BACKEND_MODE == "openai_chat":
        chat_template_kwargs = {"enable_thinking": effective_thinking}
        extra_chat_template_kwargs = EXTRA_CHAT_BODY.get("chat_template_kwargs")
        if isinstance(extra_chat_template_kwargs, dict):
            chat_template_kwargs.update(extra_chat_template_kwargs)
        body["chat_template_kwargs"] = chat_template_kwargs
        if native_thinking:
            body.update(
                {
                    "separate_reasoning": True,
                    "stream_reasoning": True,
                }
            )
        for key, value in EXTRA_CHAT_BODY.items():
            if key == "chat_template_kwargs":
                continue
            body[key] = value
    elif CHAT_BACKEND_MODE == "trainer_api":
        body.update(EXTRA_CHAT_BODY)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported chat backend mode: {CHAT_BACKEND_MODE}")
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


async def _resolve_split_chat_overlay(*, force_sync: bool = False) -> dict[str, Any]:
    if not _is_split_backend() or CHAT_BACKEND_MODE != "openai_chat":
        return {}

    training_health = await _fetch_health(TRAINING_HEALTH_URL)
    if not bool(training_health.get("ok")):
        detail = str(training_health.get("detail") or training_health.get("status") or "unavailable").strip()
        raise HTTPException(status_code=503, detail=f"{_training_backend_label()} unavailable: {detail}")

    serving_adapter_path = str(training_health.get("serving_adapter_path") or "").strip()
    if not serving_adapter_path:
        _clear_loaded_chat_overlay_path()
        return {}

    if not force_sync and _get_loaded_chat_overlay_path() == serving_adapter_path:
        return {"lora_path": serving_adapter_path}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                CHAT_LOAD_LORA_URL,
                json={
                    "lora_name": serving_adapter_path,
                    "lora_path": serving_adapter_path,
                    "pinned": False,
                },
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"{_chat_backend_label()} LoRA sync failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text[:1200]
        if "already loaded" not in detail.lower():
            raise HTTPException(status_code=502, detail=f"{_chat_backend_label()} LoRA sync failed: {detail}")

    _set_loaded_chat_overlay_path(serving_adapter_path)
    return {"lora_path": serving_adapter_path}


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _aiter_sse_payloads(response: httpx.Response):
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                yield payload
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def _append_feedback_log(entry: dict[str, Any]) -> None:
    if FEEDBACK_LOG_FILE is None:
        return
    FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _commit_chat_turn(
    state: dict[str, Any],
    *,
    user_turn: dict[str, Any],
    assistant_turn: dict[str, Any],
    prompt: str,
    assistant_text: str,
    training_prompt_messages: list[dict[str, Any]],
    training_session_id: str,
    guidance_text: str,
) -> None:
    state["transcript"].extend([user_turn, assistant_turn])
    state["context_messages"] = _trim_context(
        state["context_messages"]
        + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    state["feedback_candidates"][assistant_turn["id"]] = {
        "training_session_id": training_session_id,
        "prompt_messages": training_prompt_messages,
        "user_prompt": prompt,
        "guidance_text": guidance_text,
        "assistant_message": {"role": "assistant", "content": assistant_text},
    }
    _persist_active_session_state(state, title_from_prompt=prompt)


async def _proxy_chat(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    thinking_enabled: bool,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    backend_label = _chat_backend_label()
    headers = {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for attempt in range(2):
            body = _build_chat_request_body(
                messages=messages,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                thinking_enabled=thinking_enabled,
                max_tokens=max_tokens,
                stream=False,
            )
            body.update(await _resolve_split_chat_overlay(force_sync=attempt > 0))
            try:
                response = await client.post(CHAT_URL, json=body, headers=headers)
            except httpx.HTTPError as exc:
                detail = str(exc)
                if _should_restart_backend(detail):
                    _clear_loaded_chat_overlay_path()
                    await _restart_service_if_needed(CHAT_SERVICE_NAME)
                    raise HTTPException(
                        status_code=503,
                        detail=f"The {backend_label.lower()} disconnected and is being restarted. Wait about 30 seconds, then try again.",
                    ) from exc
                raise HTTPException(status_code=502, detail=f"{backend_label} connection error: {detail}") from exc
            if response.status_code == 200:
                break
            detail = response.text[:1200]
            if response.status_code == 400 and attempt == 0 and _should_resync_chat_overlay(detail):
                _clear_loaded_chat_overlay_path()
                continue
            if response.status_code >= 500 and _should_restart_backend(detail):
                _clear_loaded_chat_overlay_path()
                await _restart_service_if_needed(CHAT_SERVICE_NAME)
                raise HTTPException(
                    status_code=503,
                    detail=f"The {backend_label.lower()} worker crashed and is being restarted. Wait about 30 seconds, then try again.",
                )
            raise HTTPException(status_code=502, detail=f"{backend_label} error {response.status_code}: {detail}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Training backend returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Training backend returned an unexpected payload shape")
    if isinstance(payload.get("response"), dict):
        return payload["response"]
    if isinstance(payload.get("choices"), list):
        return payload
    raise HTTPException(status_code=502, detail="Training backend returned an unexpected payload shape")


async def _proxy_chat_stream(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    thinking_enabled: bool,
    max_tokens: int | None = None,
):
    backend_label = _chat_backend_label()
    headers = {
        "Authorization": f"Bearer {CHAT_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for attempt in range(2):
            body = _build_chat_request_body(
                messages=messages,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                thinking_enabled=thinking_enabled,
                max_tokens=max_tokens,
                stream=True,
            )
            body.update(await _resolve_split_chat_overlay(force_sync=attempt > 0))
            try:
                async with client.stream("POST", CHAT_URL, json=body, headers=headers) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", errors="replace")[:1200]
                        if response.status_code == 400 and attempt == 0 and _should_resync_chat_overlay(detail):
                            _clear_loaded_chat_overlay_path()
                            continue
                        if response.status_code >= 500 and _should_restart_backend(detail):
                            _clear_loaded_chat_overlay_path()
                            await _restart_service_if_needed(CHAT_SERVICE_NAME)
                            raise HTTPException(
                                status_code=503,
                                detail=f"The {backend_label.lower()} crashed and is being restarted. Wait about 30 seconds, then try again.",
                            )
                        raise HTTPException(status_code=502, detail=f"{backend_label} error {response.status_code}: {detail}")

                    async for raw_payload in _aiter_sse_payloads(response):
                        if raw_payload == "[DONE]":
                            break
                        try:
                            payload = json.loads(raw_payload)
                        except json.JSONDecodeError as exc:
                            raise HTTPException(status_code=502, detail="Streaming backend returned invalid JSON") from exc
                        if not isinstance(payload, dict):
                            continue
                        choices = payload.get("choices") or []
                        if not choices or not isinstance(choices[0], dict):
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        reasoning_content = delta.get("reasoning_content") or ""
                        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
                        if content or reasoning_content or metrics:
                            yield {
                                "type": "delta",
                                "content": content,
                                "reasoning_content": reasoning_content,
                                "metrics": metrics,
                                "payload": payload,
                            }
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            yield {
                                "type": "finish",
                                "finish_reason": finish_reason,
                                "metrics": metrics,
                                "payload": payload,
                            }
                    return
            except httpx.HTTPError as exc:
                detail = str(exc)
                if _should_restart_backend(detail):
                    _clear_loaded_chat_overlay_path()
                    await _restart_service_if_needed(CHAT_SERVICE_NAME)
                    raise HTTPException(
                        status_code=503,
                        detail=f"The {backend_label.lower()} disconnected and is being restarted. Wait about 30 seconds, then try again.",
                    ) from exc
                raise HTTPException(status_code=502, detail=f"{backend_label} connection error: {detail}") from exc


async def _proxy_feedback(
    *,
    messages: list[dict[str, Any]],
    assistant_response: str,
    score: int,
    note: str,
    session_id: str,
) -> dict[str, Any]:
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "assistant_response": assistant_response,
        "score": score,
        "note": note,
        "session_id": session_id,
    }
    backend_label = _training_backend_label()
    headers = {
        "Authorization": f"Bearer {TRAINING_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(FEEDBACK_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            detail = str(exc)
            if _should_restart_backend(detail):
                await _restart_service_if_needed(TRAINING_SERVICE_NAME)
                raise HTTPException(
                    status_code=503,
                    detail=f"The {backend_label.lower()} disconnected and is being restarted. Wait about 30 seconds, then try again.",
                ) from exc
            raise HTTPException(status_code=502, detail=f"{backend_label} connection error: {detail}") from exc
        if response.status_code != 200:
            detail = response.text[:1200]
            if response.status_code >= 500 and _should_restart_backend(detail):
                await _restart_service_if_needed(TRAINING_SERVICE_NAME)
                raise HTTPException(
                    status_code=503,
                    detail=f"The {backend_label.lower()} crashed and is being restarted. Wait about 30 seconds, then try again.",
                )
            raise HTTPException(status_code=502, detail=f"{backend_label} error {response.status_code}: {detail}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Training backend returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Training backend returned an unexpected payload shape")
    return payload


def _guidance_size_bytes(profile_id: str) -> int:
    guidance_file = _profile_guidance_file(profile_id)
    try:
        return guidance_file.stat().st_size
    except FileNotFoundError:
        return 0


def _system_prompt_size_bytes(profile_id: str) -> int:
    system_prompt_file = _profile_system_prompt_file(profile_id)
    try:
        return system_prompt_file.stat().st_size
    except FileNotFoundError:
        return 0


def _augment_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    profiles = []
    for raw_profile in payload.get("profiles") or []:
        if not isinstance(raw_profile, dict):
            continue
        profile_id = str(raw_profile.get("id") or "").strip()
        if not profile_id:
            continue
        guidance_bytes = _guidance_size_bytes(profile_id)
        guidance_text = _load_guidance_text(profile_id)
        system_prompt_bytes = _system_prompt_size_bytes(profile_id)
        system_prompt_text = _load_system_prompt_text(profile_id)
        profile = dict(raw_profile)
        profile["guidance_size_bytes"] = guidance_bytes
        profile["guidance_text"] = guidance_text
        profile["system_prompt_size_bytes"] = system_prompt_bytes
        profile["system_prompt_text"] = system_prompt_text
        profile["total_size_bytes"] = int(profile.get("size_bytes") or 0) + guidance_bytes + system_prompt_bytes
        profiles.append(profile)
    augmented = {
        "ok": bool(payload.get("ok", True)),
        "active_profile_id": str(payload.get("active_profile_id") or _load_active_profile_id()),
        "profiles": profiles,
    }
    if "created_profile_id" in payload:
        augmented["created_profile_id"] = payload.get("created_profile_id")
    if "profile" in payload and isinstance(payload.get("profile"), dict):
        augmented["profile"] = payload.get("profile")
    return augmented


def _fallback_profile_payload() -> dict[str, Any]:
    _ensure_profile_storage()
    profile_ids = []
    if PROFILES_DIR.exists():
        for child in sorted(PROFILES_DIR.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                profile_ids.append(child.name)
    if not profile_ids:
        profile_ids = [DEFAULT_PROFILE_ID]
    return _augment_profile_payload(
        {
            "ok": False,
            "active_profile_id": _load_active_profile_id(),
            "profiles": [
                {
                    "id": profile_id,
                    "name": profile_id,
                    "size_bytes": 0,
                }
                for profile_id in profile_ids
            ],
        }
    )


def _save_profile_text_settings(
    profile_id: str,
    *,
    guidance_text: str | None = None,
    system_prompt_text: str | None = None,
) -> tuple[str, str]:
    saved_guidance_text = _load_guidance_text(profile_id) if guidance_text is None else _save_guidance_text(profile_id, guidance_text)
    saved_system_prompt_text = (
        _load_system_prompt_text(profile_id)
        if system_prompt_text is None
        else _save_system_prompt_text(profile_id, system_prompt_text)
    )
    return saved_guidance_text, saved_system_prompt_text


async def _proxy_profiles() -> dict[str, Any]:
    if not PROFILES_SUPPORTED:
        return {
            "ok": False,
            "active_profile_id": _load_active_profile_id(),
            "profiles": [],
        }
    headers = {
        "Authorization": f"Bearer {TRAINING_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(PROFILES_URL, headers=headers)
        except httpx.HTTPError as exc:
            return _fallback_profile_payload()
    if response.status_code != 200:
        if response.status_code >= 500:
            return _fallback_profile_payload()
        raise HTTPException(status_code=502, detail=f"Profile backend error {response.status_code}: {response.text[:1200]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Profile backend returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Profile backend returned an unexpected payload shape")
    return _augment_profile_payload(payload)


async def _proxy_create_profile(name: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {TRAINING_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(PROFILE_CREATE_URL, json={"name": name}, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Profile backend connection error: {exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Profile backend error {response.status_code}: {response.text[:1200]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Profile backend returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Profile backend returned an unexpected payload shape")
    return _augment_profile_payload(payload)


async def _proxy_select_profile(profile_id: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {TRAINING_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(PROFILE_SELECT_URL, json={"profile_id": profile_id}, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Profile backend connection error: {exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Profile backend error {response.status_code}: {response.text[:1200]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Profile backend returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Profile backend returned an unexpected payload shape")
    return _augment_profile_payload(payload)


def _should_restart_backend(detail: str) -> bool:
    lowered = detail.lower()
    markers = (
        "call_upstream_request_error",
        "error sending request for url",
        "connection refused",
        "internal server error",
    )
    return any(marker in lowered for marker in markers)


def _restart_service(service_name: str) -> None:
    if not service_name:
        return
    hard_restart = service_name == CHAT_SERVICE_NAME and CHAT_BACKEND_MODE == "openai_chat"
    if hard_restart:
        subprocess.run(
            ["systemctl", "--user", "kill", "-s", "SIGKILL", service_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        time.sleep(1)
        subprocess.run(
            ["systemctl", "--user", "reset-failed", service_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        subprocess.run(
            ["systemctl", "--user", "start", service_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return

    subprocess.run(
        ["systemctl", "--user", "restart", service_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )


async def _restart_service_if_needed(service_name: str) -> None:
    if not service_name:
        return
    now = time.time()
    last_restart_at = _last_service_restart_at.get(service_name, 0.0)
    if now - last_restart_at < 30:
        return
    _last_service_restart_at[service_name] = now
    await asyncio.to_thread(_restart_service, service_name)


async def _fetch_health(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(url)
        except Exception as exc:  # pragma: no cover - surfaced in UI instead
            return {"ok": False, "detail": str(exc), "url": url}

    try:
        payload = response.json()
    except ValueError:
        payload = {
            "ok": response.status_code == 200,
            "detail": {"status_code": response.status_code, "body": response.text[:300]},
            "url": url,
        }

    if isinstance(payload, dict):
        payload.setdefault("ok", response.status_code == 200)
        payload.setdefault("url", url)
        return payload

    return {"ok": response.status_code == 200, "detail": {"status_code": response.status_code}, "url": url}


async def _proxy_health() -> dict[str, Any]:
    chat_health = await _fetch_health(CHAT_HEALTH_URL)
    if not _is_split_backend():
        return chat_health

    training_health = await _fetch_health(TRAINING_HEALTH_URL)
    return {
        "ok": bool(chat_health.get("ok")) and bool(training_health.get("ok")),
        "chat": chat_health,
        "training": training_health,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def api_state(request: Request) -> JSONResponse:
    browser_id, state, created = _ensure_browser_session(request)
    payload = {"ok": True, "state": _serialize_state(state)}
    return _response_with_cookie(payload, browser_id, created)


@app.get("/api/status")
async def api_status(request: Request) -> JSONResponse:
    browser_id, state, created = _ensure_browser_session(request)
    payload = {
        "ok": True,
        "state": {
            **_serialize_state(state),
            "proxy": await _proxy_health(),
        },
    }
    return _response_with_cookie(payload, browser_id, created)


@app.get("/api/profiles")
async def api_profiles(request: Request) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    profile_payload = await _proxy_profiles()
    active_profile_id = str(profile_payload.get("active_profile_id") or _load_active_profile_id())
    _save_active_profile_id(active_profile_id)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if not state["busy"]:
            _set_state_profile(state, active_profile_id)
        payload = {
            "ok": True,
            "profiles": profile_payload.get("profiles") or [],
            "active_profile_id": active_profile_id,
            "state": _serialize_state(state),
        }
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/profiles")
async def api_create_profile(request: Request, body: ProfileCreateRequest) -> JSONResponse:
    if not PROFILES_SUPPORTED:
        raise HTTPException(status_code=400, detail="Training profiles are not supported by the current backend")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Profile name cannot be empty")

    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")

    profile_payload = await _proxy_create_profile(name)
    created_profile_id = str(profile_payload.get("created_profile_id") or profile_payload.get("active_profile_id") or _load_active_profile_id())
    if body.select_after_create:
        profile_payload = await _proxy_select_profile(created_profile_id)

    active_profile_id = str(profile_payload.get("active_profile_id") or created_profile_id)
    _save_active_profile_id(active_profile_id)

    with _state_lock:
        state = _browser_sessions[browser_id]
        _set_state_profile(state, active_profile_id)
        payload = {
            "ok": True,
            "profiles": profile_payload.get("profiles") or [],
            "active_profile_id": active_profile_id,
            "state": _serialize_state(state),
        }
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/profiles/select")
async def api_select_profile(request: Request, body: ProfileSelectRequest) -> JSONResponse:
    if not PROFILES_SUPPORTED:
        raise HTTPException(status_code=400, detail="Training profiles are not supported by the current backend")
    profile_id = body.profile_id.strip()
    if not profile_id:
        raise HTTPException(status_code=400, detail="profile_id cannot be empty")

    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")

    profile_payload = await _proxy_select_profile(profile_id)
    active_profile_id = str(profile_payload.get("active_profile_id") or profile_id)
    _save_active_profile_id(active_profile_id)

    with _state_lock:
        state = _browser_sessions[browser_id]
        _set_state_profile(state, active_profile_id)
        payload = {
            "ok": True,
            "profiles": profile_payload.get("profiles") or [],
            "active_profile_id": active_profile_id,
            "state": _serialize_state(state),
        }
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/sessions")
async def api_create_session(request: Request) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        _create_and_activate_session(state)
        payload = {"ok": True, "state": _serialize_state(state)}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/sessions/select")
async def api_select_session(request: Request, body: SessionSelectRequest) -> JSONResponse:
    session_id = body.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        profile_id = str(state.get("active_profile_id") or DEFAULT_PROFILE_ID)
        if not _session_file(profile_id, session_id).exists():
            raise HTTPException(status_code=404, detail="Session not found for the active profile")
        _select_state_session(state, session_id)
        payload = {"ok": True, "state": _serialize_state(state)}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/chat/stream")
async def api_chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        state["busy"] = True
        context_messages = list(state["context_messages"])
        guidance_text = state.get("guidance_text", "")
        system_prompt_text = state.get("system_prompt_text", "")
        thinking_enabled = bool(body.thinking_enabled) if body.thinking_enabled is not None else bool(state.get("thinking_enabled"))
        if FORCE_NO_THINK:
            thinking_enabled = False
        _touch_state(state)

    training_session_id = uuid.uuid4().hex
    user_turn = {
        "id": uuid.uuid4().hex,
        "role": "user",
        "content": prompt,
        "created_at": time.time(),
    }
    assistant_turn = {
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": "",
        "created_at": time.time(),
        "feedback": None,
        "feedback_pending": False,
        "reasoning": "",
        "thinking_enabled": thinking_enabled,
        "streaming": True,
        "metrics": None,
    }

    prompt_messages: list[dict[str, Any]] = []
    training_prompt_messages: list[dict[str, Any]] = []
    chat_control_message = _chat_control_message(thinking_enabled)
    if chat_control_message is not None:
        prompt_messages.append(chat_control_message)
        training_prompt_messages.append(chat_control_message)
    system_prompt_message = _system_prompt_message(system_prompt_text)
    if system_prompt_message is not None:
        prompt_messages.append(system_prompt_message)
    guidance_message = _guidance_message(guidance_text)
    if guidance_message is not None:
        prompt_messages.append(guidance_message)
        training_prompt_messages.append(guidance_message)
    prompt_messages.extend(context_messages)
    training_prompt_messages.extend(context_messages)
    user_message = {"role": "user", "content": prompt}
    prompt_messages.append(user_message)
    training_prompt_messages.append(user_message)

    async def event_stream():
        committed = False
        final_text = ""
        final_reasoning = ""
        last_metrics: dict[str, Any] | None = None
        finish_reason = "stop"
        streamed_raw_text = ""
        stream_started_at = time.perf_counter()
        first_token_at: float | None = None
        structured_visible_thinking = thinking_enabled and not USE_NATIVE_CHAT_THINKING and not FORCE_NO_THINK
        requested_max_tokens = UI_THINKING_MAX_TOKENS if thinking_enabled else UI_MAX_TOKENS

        try:
            yield _sse_event(
                "start",
                {
                    "user_turn": user_turn,
                    "assistant_turn": assistant_turn,
                },
            )

            async for event in _proxy_chat_stream(
                messages=prompt_messages,
                session_id=training_session_id,
                turn_type="main",
                session_done=False,
                thinking_enabled=thinking_enabled,
                max_tokens=requested_max_tokens,
            ):
                if await request.is_disconnected():
                    raise asyncio.CancelledError()

                if event["type"] == "delta":
                    raw_metrics = event.get("metrics") or {}
                    delta = str(event.get("content") or "")
                    reasoning_delta = str(event.get("reasoning_content") or "")
                    if not delta and not reasoning_delta and not raw_metrics:
                        continue
                    if first_token_at is None and (delta or reasoning_delta or raw_metrics):
                        first_token_at = time.perf_counter()
                    if structured_visible_thinking:
                        if delta:
                            streamed_raw_text += delta
                        next_reasoning, next_visible = _split_visible_thinking_response(streamed_raw_text, final=False)
                        reasoning_delta = (
                            next_reasoning[len(final_reasoning) :] if next_reasoning.startswith(final_reasoning) else next_reasoning
                        )
                        delta = next_visible[len(final_text) :] if next_visible.startswith(final_text) else next_visible
                        final_reasoning = next_reasoning
                        final_text = next_visible
                    else:
                        if delta:
                            final_text += delta
                        if reasoning_delta:
                            final_reasoning += reasoning_delta
                    metrics = _stream_metrics_fallback(
                        raw_metrics,
                        visible_text=final_text,
                        reasoning_text=final_reasoning,
                        started_at=stream_started_at,
                        first_token_at=first_token_at,
                    )
                    last_metrics = metrics
                    yield _sse_event(
                        "delta",
                        {
                            "assistant_turn_id": assistant_turn["id"],
                            "delta": delta,
                            "reasoning_delta": reasoning_delta,
                            "metrics": metrics,
                        },
                    )
                    continue

                if event["type"] == "finish":
                    finish_reason = str(event.get("finish_reason") or "stop")
                    last_metrics = _stream_metrics_fallback(
                        event.get("metrics") or last_metrics,
                        visible_text=final_text,
                        reasoning_text=final_reasoning,
                        started_at=stream_started_at,
                        first_token_at=first_token_at,
                    )

            if structured_visible_thinking:
                assistant_reasoning, assistant_text = _split_visible_thinking_response(streamed_raw_text, final=True)
                assistant_reasoning = assistant_reasoning.strip()
                assistant_text = assistant_text.strip()
            else:
                assistant_reasoning = final_reasoning.strip()
                assistant_text = _resolve_assistant_text(final_text, assistant_reasoning)
            needs_reasoning_backfill = structured_visible_thinking and bool(assistant_text) and not assistant_reasoning
            if needs_reasoning_backfill or (not assistant_text and not assistant_reasoning):
                fallback_response = await _proxy_chat(
                    messages=prompt_messages,
                    session_id=training_session_id,
                    turn_type="main",
                    session_done=False,
                    thinking_enabled=thinking_enabled,
                    max_tokens=requested_max_tokens,
                )
                choice = (fallback_response.get("choices") or [{}])[0]
                assistant_message = choice.get("message") or {}
                raw_assistant_content = str(assistant_message.get("content") or "")
                if structured_visible_thinking:
                    assistant_reasoning, assistant_text = _split_visible_thinking_response(raw_assistant_content, final=True)
                    assistant_reasoning = assistant_reasoning.strip()
                    assistant_text = assistant_text.strip()
                else:
                    assistant_reasoning = str(assistant_message.get("reasoning_content") or "").strip()
                    assistant_text = _resolve_assistant_text(raw_assistant_content, assistant_reasoning)

                delta = assistant_text[len(final_text) :] if assistant_text.startswith(final_text) else assistant_text
                reasoning_delta = (
                    assistant_reasoning[len(final_reasoning) :]
                    if assistant_reasoning.startswith(final_reasoning)
                    else assistant_reasoning
                )
                final_text = assistant_text
                final_reasoning = assistant_reasoning

                if delta or reasoning_delta:
                    last_metrics = _stream_metrics_fallback(
                        last_metrics,
                        visible_text=assistant_text,
                        reasoning_text=assistant_reasoning,
                        started_at=stream_started_at,
                        first_token_at=first_token_at,
                    )
                    yield _sse_event(
                        "delta",
                        {
                            "assistant_turn_id": assistant_turn["id"],
                            "delta": delta,
                            "reasoning_delta": reasoning_delta,
                            "metrics": last_metrics or {},
                        },
                    )

            if not assistant_text and not assistant_reasoning:
                raise HTTPException(status_code=502, detail="Model returned an empty response")

            last_metrics = _stream_metrics_fallback(
                last_metrics,
                visible_text=assistant_text,
                reasoning_text=assistant_reasoning,
                started_at=stream_started_at,
                first_token_at=first_token_at,
            )
            yield _sse_event(
                "final",
                {
                    "assistant_turn_id": assistant_turn["id"],
                    "finish_reason": finish_reason,
                    "metrics": last_metrics or {},
                },
            )

            completed_assistant_turn = {
                **assistant_turn,
                "content": assistant_text,
                "reasoning": assistant_reasoning,
                "feedback_pending": True,
                "streaming": False,
                "metrics": last_metrics,
            }

            with _state_lock:
                state = _browser_sessions[browser_id]
                _commit_chat_turn(
                    state,
                    user_turn=user_turn,
                    assistant_turn=completed_assistant_turn,
                    prompt=prompt,
                    assistant_text=assistant_text,
                    training_prompt_messages=training_prompt_messages,
                    training_session_id=training_session_id,
                    guidance_text=guidance_text,
                )
                state["busy"] = False
                _touch_state(state)
                serialized_state = _serialize_state(state)

            committed = True
            yield _sse_event("state", {"state": serialized_state})
        except asyncio.CancelledError:
            with _state_lock:
                state = _browser_sessions[browser_id]
                state["busy"] = False
                _touch_state(state)
            raise
        except Exception as exc:
            with _state_lock:
                state = _browser_sessions[browser_id]
                state["busy"] = False
                _touch_state(state)
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            yield _sse_event("error", {"detail": detail or "Streaming failed"})
        finally:
            if not committed:
                with _state_lock:
                    state = _browser_sessions[browser_id]
                    state["busy"] = False
                    _touch_state(state)

    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    _set_cookie_on_response(response, browser_id, created)
    return response


@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest) -> JSONResponse:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        state["busy"] = True
        context_messages = list(state["context_messages"])
        guidance_text = state.get("guidance_text", "")
        system_prompt_text = state.get("system_prompt_text", "")
        thinking_enabled = bool(body.thinking_enabled) if body.thinking_enabled is not None else bool(state.get("thinking_enabled"))
        if FORCE_NO_THINK:
            thinking_enabled = False

    training_session_id = uuid.uuid4().hex
    prompt_messages: list[dict[str, Any]] = []
    training_prompt_messages: list[dict[str, Any]] = []
    chat_control_message = _chat_control_message(thinking_enabled)
    if chat_control_message is not None:
        prompt_messages.append(chat_control_message)
        training_prompt_messages.append(chat_control_message)
    system_prompt_message = _system_prompt_message(system_prompt_text)
    if system_prompt_message is not None:
        prompt_messages.append(system_prompt_message)
    guidance_message = _guidance_message(guidance_text)
    if guidance_message is not None:
        prompt_messages.append(guidance_message)
        training_prompt_messages.append(guidance_message)
    prompt_messages.extend(context_messages)
    training_prompt_messages.extend(context_messages)
    user_message = {"role": "user", "content": prompt}
    prompt_messages.append(user_message)
    training_prompt_messages.append(user_message)

    try:
        proxy_response = await _proxy_chat(
            messages=prompt_messages,
            session_id=training_session_id,
            turn_type="main",
            session_done=False,
            thinking_enabled=thinking_enabled,
            max_tokens=UI_MAX_TOKENS,
        )
        choice = (proxy_response.get("choices") or [{}])[0]
        assistant_message = choice.get("message") or {}
        raw_assistant_content = str(assistant_message.get("content") or "")
        if thinking_enabled and not USE_NATIVE_CHAT_THINKING and not FORCE_NO_THINK:
            assistant_reasoning, assistant_text = _split_visible_thinking_response(raw_assistant_content, final=True)
            assistant_reasoning = assistant_reasoning.strip()
            assistant_text = assistant_text.strip()
        else:
            assistant_reasoning = (assistant_message.get("reasoning_content") or "").strip()
            assistant_text = _resolve_assistant_text(raw_assistant_content, assistant_reasoning)
        if not assistant_text and not assistant_reasoning:
            raise HTTPException(status_code=502, detail="Model returned an empty response")

        user_turn = {
            "id": uuid.uuid4().hex,
            "role": "user",
            "content": prompt,
            "created_at": time.time(),
        }
        assistant_turn = {
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "content": assistant_text,
            "created_at": time.time(),
            "feedback": None,
            "feedback_pending": True,
            "reasoning": assistant_reasoning,
            "thinking_enabled": thinking_enabled,
            "metrics": None,
        }

        with _state_lock:
            state = _browser_sessions[browser_id]
            _commit_chat_turn(
                state,
                user_turn=user_turn,
                assistant_turn=assistant_turn,
                prompt=prompt,
                assistant_text=assistant_text,
                training_prompt_messages=training_prompt_messages,
                training_session_id=training_session_id,
                guidance_text=guidance_text,
            )
            state["busy"] = False
            _touch_state(state)
            payload = {"ok": True, "state": _serialize_state(state)}
        return _response_with_cookie(payload, browser_id, created)
    except Exception:
        with _state_lock:
            state = _browser_sessions[browser_id]
            state["busy"] = False
            _touch_state(state)
        raise


@app.post("/api/feedback")
async def api_feedback(request: Request, body: FeedbackRequest) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    score = body.score
    if score is None:
        if body.rating == "good":
            score = 8
        elif body.rating == "bad":
            score = 2
    if score is None or score < 1 or score > 10:
        raise HTTPException(status_code=400, detail="Feedback score must be between 1 and 10")

    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        candidate_id = body.assistant_turn_id.strip()
        feedback_candidates = state.get("feedback_candidates") or {}
        if candidate_id:
            pending_episode = feedback_candidates.get(candidate_id)
        else:
            pending_episode = None
            candidate_id = ""
            for item in reversed(state["transcript"]):
                if item.get("role") == "assistant" and item.get("feedback_pending"):
                    candidate_id = str(item.get("id") or "")
                    pending_episode = feedback_candidates.get(candidate_id)
                    if pending_episode is not None:
                        break
        if pending_episode is None or not candidate_id:
            raise HTTPException(status_code=409, detail="There is no assistant response available for feedback")
        state["busy"] = True

    feedback_message = _feedback_text(score, body.note)
    clean_note = body.note.strip()
    if score >= 7:
        sentiment = "positive"
    elif score <= 4:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    try:
        if TRAINING_BACKEND_MODE == "rl_proxy":
            messages = list(pending_episode["prompt_messages"]) + [
                pending_episode["assistant_message"],
                {"role": "user", "content": feedback_message},
            ]
            await _proxy_chat(
                messages=messages,
                session_id=pending_episode["training_session_id"],
                turn_type="main",
                session_done=True,
                thinking_enabled=False,
                max_tokens=16,
            )
        elif TRAINING_BACKEND_MODE == "trainer_api":
            await _proxy_feedback(
                messages=list(pending_episode["prompt_messages"]),
                assistant_response=pending_episode["assistant_message"]["content"],
                score=score,
                note=clean_note,
                session_id=pending_episode["training_session_id"],
            )
        else:
            _append_feedback_log(
                {
                    "created_at": time.time(),
                    "model": MODEL_NAME,
                    "score": score,
                    "sentiment": sentiment,
                    "note": clean_note,
                    "feedback_text": feedback_message,
                    "user_prompt": pending_episode.get("user_prompt", ""),
                    "assistant_response": pending_episode["assistant_message"]["content"],
                    "guidance_text": pending_episode.get("guidance_text", ""),
                    "messages": list(pending_episode["prompt_messages"]) + [pending_episode["assistant_message"]],
                }
            )

        with _state_lock:
            state = _browser_sessions[browser_id]
            for item in reversed(state["transcript"]):
                if item.get("id") == candidate_id:
                    item["feedback"] = {
                        "score": score,
                        "sentiment": sentiment,
                        "note": clean_note,
                        "sent_text": feedback_message,
                        "created_at": time.time(),
                    }
                    item["feedback_pending"] = False
                    break
            state.get("feedback_candidates", {}).pop(candidate_id, None)
            state["busy"] = False
            _persist_active_session_state(state)
            payload = {"ok": True, "state": _serialize_state(state)}
        return _response_with_cookie(payload, browser_id, created)
    except Exception:
        with _state_lock:
            state = _browser_sessions[browser_id]
            state["busy"] = False
            _touch_state(state)
        raise


@app.post("/api/guidance")
async def api_guidance(request: Request, body: GuidanceRequest) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        current_profile_id = _browser_sessions[browser_id].get("active_profile_id", _load_active_profile_id())
    clean_text, system_prompt_text = _save_profile_text_settings(current_profile_id, guidance_text=body.text)

    with _state_lock:
        for state in _browser_sessions.values():
            if state.get("active_profile_id", DEFAULT_PROFILE_ID) == current_profile_id:
                state["guidance_text"] = clean_text
                state["system_prompt_text"] = system_prompt_text
                _touch_state(state)
        payload = {"ok": True, "state": _serialize_state(_browser_sessions[browser_id])}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/system-prompt")
async def api_system_prompt(request: Request, body: SystemPromptRequest) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        current_profile_id = _browser_sessions[browser_id].get("active_profile_id", _load_active_profile_id())
    guidance_text, system_prompt_text = _save_profile_text_settings(current_profile_id, system_prompt_text=body.text)

    with _state_lock:
        for state in _browser_sessions.values():
            if state.get("active_profile_id", DEFAULT_PROFILE_ID) == current_profile_id:
                state["guidance_text"] = guidance_text
                state["system_prompt_text"] = system_prompt_text
                _touch_state(state)
        payload = {"ok": True, "state": _serialize_state(_browser_sessions[browser_id])}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/profile-settings")
async def api_profile_settings(request: Request, body: ProfileSettingsRequest) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        current_profile_id = _browser_sessions[browser_id].get("active_profile_id", _load_active_profile_id())
    guidance_text, system_prompt_text = _save_profile_text_settings(
        current_profile_id,
        guidance_text=body.guidance_text,
        system_prompt_text=body.system_prompt_text,
    )

    with _state_lock:
        for state in _browser_sessions.values():
            if state.get("active_profile_id", DEFAULT_PROFILE_ID) == current_profile_id:
                state["guidance_text"] = guidance_text
                state["system_prompt_text"] = system_prompt_text
                _touch_state(state)
        payload = {"ok": True, "state": _serialize_state(_browser_sessions[browser_id])}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/thinking")
async def api_thinking(request: Request, body: ThinkingRequest) -> Response:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Cannot change thinking mode while a response is streaming")
        profile_id = str(state.get("active_profile_id") or DEFAULT_PROFILE_ID)
        enabled = False if FORCE_NO_THINK else bool(body.enabled)
        _save_thinking_enabled(profile_id, enabled)
        state["thinking_enabled"] = enabled
        _touch_state(state)
        payload = {"ok": True, "state": _serialize_state(state)}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/reset")
async def api_reset(request: Request) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        _create_and_activate_session(state)
        payload = {"ok": True, "state": _serialize_state(state)}
    return _response_with_cookie(payload, browser_id, created)


if __name__ == "__main__":
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="info")
