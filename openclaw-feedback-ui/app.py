from __future__ import annotations

import asyncio
import json
import os
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
ACTIVE_PROFILE_FILE = STATE_DIR / "active_profile.txt"
FEEDBACK_LOG_FILE_VALUE = os.environ.get("OPENCLAW_RL_UI_FEEDBACK_LOG_FILE", "").strip()
FEEDBACK_LOG_FILE = Path(FEEDBACK_LOG_FILE_VALUE).expanduser() if FEEDBACK_LOG_FILE_VALUE else None

COOKIE_NAME = "openclaw_rl_ui_session"
PROXY_BASE_URL = os.environ.get("OPENCLAW_RL_PROXY_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
BACKEND_MODE = os.environ.get("OPENCLAW_RL_UI_BACKEND_MODE", "rl_proxy").strip().lower()
CHAT_PATH = os.environ.get("OPENCLAW_RL_CHAT_PATH", "/v1/chat/completions")
FEEDBACK_PATH = os.environ.get("OPENCLAW_RL_FEEDBACK_PATH", "/v1/feedback")
DEFAULT_HEALTH_PATH = "/healthz" if BACKEND_MODE == "rl_proxy" else "/health"
HEALTH_PATH = os.environ.get("OPENCLAW_RL_HEALTH_PATH", DEFAULT_HEALTH_PATH)
CHAT_URL = f"{PROXY_BASE_URL}{CHAT_PATH}"
HEALTH_URL = f"{PROXY_BASE_URL}{HEALTH_PATH}"
FEEDBACK_URL = f"{PROXY_BASE_URL}{FEEDBACK_PATH}"
API_KEY = os.environ.get("OPENCLAW_RL_API_KEY", "openclaw-local")
MODEL_NAME = os.environ.get("OPENCLAW_RL_MODEL", "qwen3-0.6b-local")
DEFAULT_PROFILE_ID = os.environ.get("OPENCLAW_RL_DEFAULT_PROFILE_ID", "default").strip() or "default"
REQUEST_TIMEOUT = float(os.environ.get("OPENCLAW_RL_UI_TIMEOUT_SECONDS", "600"))
MAX_HISTORY_TURNS = int(os.environ.get("OPENCLAW_RL_UI_MAX_HISTORY_TURNS", "16"))
UI_MAX_TOKENS = int(os.environ.get("OPENCLAW_RL_UI_MAX_TOKENS", "384"))
UI_HOST = os.environ.get("OPENCLAW_RL_UI_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("OPENCLAW_RL_UI_PORT", "30001"))
RL_SERVICE_NAME = os.environ.get("OPENCLAW_RL_SERVICE_NAME", "openclaw-rl.service")
FORCE_NO_THINK = os.environ.get("OPENCLAW_RL_FORCE_NO_THINK", "1").strip().lower() in {"1", "true", "yes", "on"}
PROFILES_SUPPORTED = BACKEND_MODE == "trainer_api"
PROFILES_URL = f"{PROXY_BASE_URL}/v1/profiles"
PROFILE_CREATE_URL = f"{PROXY_BASE_URL}/v1/profiles"
PROFILE_SELECT_URL = f"{PROXY_BASE_URL}/v1/profiles/select"
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
_last_backend_restart_at = 0.0


class ChatRequest(BaseModel):
    prompt: str


class FeedbackRequest(BaseModel):
    assistant_turn_id: str = ""
    score: int | None = None
    rating: str | None = None
    note: str = ""


class GuidanceRequest(BaseModel):
    text: str = ""


class ProfileCreateRequest(BaseModel):
    name: str = ""
    select_after_create: bool = True


class ProfileSelectRequest(BaseModel):
    profile_id: str = ""


def _profile_dir(profile_id: str) -> Path:
    return PROFILES_DIR / profile_id


def _profile_guidance_file(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "steering_notes.txt"


def _ensure_profile_storage() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    default_guidance_path = _profile_guidance_file(DEFAULT_PROFILE_ID)
    if LEGACY_GUIDANCE_FILE.exists() and not default_guidance_path.exists():
        default_guidance_path.parent.mkdir(parents=True, exist_ok=True)
        default_guidance_path.write_text(LEGACY_GUIDANCE_FILE.read_text(encoding="utf-8"), encoding="utf-8")


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


def _new_browser_state(profile_id: str | None = None) -> dict[str, Any]:
    active_profile_id = profile_id or _load_active_profile_id()
    return {
        "transcript": [],
        "context_messages": [],
        "feedback_candidates": {},
        "busy": False,
        "guidance_text": _load_guidance_text(active_profile_id),
        "active_profile_id": active_profile_id,
        "updated_at": time.time(),
    }


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
        "proxy_base_url": PROXY_BASE_URL,
        "guidance_text": state.get("guidance_text", ""),
        "active_profile_id": state.get("active_profile_id", DEFAULT_PROFILE_ID),
        "profiles_supported": PROFILES_SUPPORTED,
    }


def _touch_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()


def _set_state_profile(state: dict[str, Any], profile_id: str, *, reset_session: bool) -> None:
    state["active_profile_id"] = profile_id
    state["guidance_text"] = _load_guidance_text(profile_id)
    if reset_session:
        state["transcript"] = []
        state["context_messages"] = []
        state["feedback_candidates"] = {}
        state["busy"] = False
    _touch_state(state)


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


def _chat_control_message() -> dict[str, str] | None:
    if not FORCE_NO_THINK:
        return None
    return {
        "role": "system",
        "content": (
            "For this local feedback UI, answer in non-thinking mode and return only the final answer. "
            "Do not expose chain-of-thought or a <think> block. /no_think"
        ),
    }


def _assistant_text(message: dict[str, Any]) -> str:
    content = (message.get("content") or "").strip()
    if content:
        return content
    return (message.get("reasoning_content") or "").strip()


def _build_chat_request_body(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    max_tokens: int | None = None,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": stream,
    }
    if BACKEND_MODE == "rl_proxy":
        body.update(
            {
                "session_id": session_id,
                "turn_type": turn_type,
                "session_done": session_done,
            }
        )
    elif BACKEND_MODE in {"openai_chat", "trainer_api"}:
        body.update(EXTRA_CHAT_BODY)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported backend mode: {BACKEND_MODE}")
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


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
    prompt_messages: list[dict[str, Any]],
    training_session_id: str,
    guidance_text: str,
) -> None:
    state["transcript"].extend([user_turn, assistant_turn])
    state["context_messages"] = _trim_context(
        state["context_messages"] + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    state["feedback_candidates"][assistant_turn["id"]] = {
        "training_session_id": training_session_id,
        "prompt_messages": prompt_messages,
        "user_prompt": prompt,
        "guidance_text": guidance_text,
        "assistant_message": {"role": "assistant", "content": assistant_text},
    }


async def _proxy_chat(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    body = _build_chat_request_body(
        messages=messages,
        session_id=session_id,
        turn_type=turn_type,
        session_done=session_done,
        max_tokens=max_tokens,
        stream=False,
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(CHAT_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            detail = str(exc)
            if _should_restart_backend(detail):
                await _restart_backend_if_needed()
                raise HTTPException(
                    status_code=503,
                    detail="The RL backend disconnected and is being restarted. Wait about 30 seconds, then try again.",
                ) from exc
            raise HTTPException(status_code=502, detail=f"RL proxy connection error: {detail}") from exc
        if response.status_code != 200:
            detail = response.text[:1200]
            if response.status_code >= 500 and _should_restart_backend(detail):
                await _restart_backend_if_needed()
                raise HTTPException(
                    status_code=503,
                    detail="The RL backend worker crashed and is being restarted. Wait about 30 seconds, then try again.",
                )
            raise HTTPException(status_code=502, detail=f"RL proxy error {response.status_code}: {detail}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="RL proxy returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="RL proxy returned an unexpected payload shape")
    if isinstance(payload.get("response"), dict):
        return payload["response"]
    if isinstance(payload.get("choices"), list):
        return payload
    raise HTTPException(status_code=502, detail="RL proxy returned an unexpected payload shape")


async def _proxy_chat_stream(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    max_tokens: int | None = None,
):
    body = _build_chat_request_body(
        messages=messages,
        session_id=session_id,
        turn_type=turn_type,
        session_done=session_done,
        max_tokens=max_tokens,
        stream=True,
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            async with client.stream("POST", CHAT_URL, json=body, headers=headers) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:1200]
                    if response.status_code >= 500 and _should_restart_backend(detail):
                        await _restart_backend_if_needed()
                        raise HTTPException(
                            status_code=503,
                            detail="The training backend crashed and is being restarted. Wait about 30 seconds, then try again.",
                        )
                    raise HTTPException(status_code=502, detail=f"Streaming backend error {response.status_code}: {detail}")

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
                    if content:
                        yield {"type": "delta", "content": content, "payload": payload}
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        yield {"type": "finish", "finish_reason": finish_reason, "payload": payload}
        except httpx.HTTPError as exc:
            detail = str(exc)
            if _should_restart_backend(detail):
                await _restart_backend_if_needed()
                raise HTTPException(
                    status_code=503,
                    detail="The training backend disconnected and is being restarted. Wait about 30 seconds, then try again.",
                ) from exc
            raise HTTPException(status_code=502, detail=f"Streaming backend connection error: {detail}") from exc


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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(FEEDBACK_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            detail = str(exc)
            if _should_restart_backend(detail):
                await _restart_backend_if_needed()
                raise HTTPException(
                    status_code=503,
                    detail="The training backend disconnected and is being restarted. Wait about 30 seconds, then try again.",
                ) from exc
            raise HTTPException(status_code=502, detail=f"Training backend connection error: {detail}") from exc
        if response.status_code != 200:
            detail = response.text[:1200]
            if response.status_code >= 500 and _should_restart_backend(detail):
                await _restart_backend_if_needed()
                raise HTTPException(
                    status_code=503,
                    detail="The training backend crashed and is being restarted. Wait about 30 seconds, then try again.",
                )
            raise HTTPException(status_code=502, detail=f"Training backend error {response.status_code}: {detail}")
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
        profile = dict(raw_profile)
        profile["guidance_size_bytes"] = guidance_bytes
        profile["guidance_text"] = guidance_text
        profile["total_size_bytes"] = int(profile.get("size_bytes") or 0) + guidance_bytes
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


async def _proxy_profiles() -> dict[str, Any]:
    if not PROFILES_SUPPORTED:
        return {
            "ok": False,
            "active_profile_id": _load_active_profile_id(),
            "profiles": [],
        }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(PROFILES_URL, headers=headers)
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


async def _proxy_create_profile(name: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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
        "Authorization": f"Bearer {API_KEY}",
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


def _restart_backend() -> None:
    subprocess.run(
        ["systemctl", "--user", "restart", RL_SERVICE_NAME],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )


async def _restart_backend_if_needed() -> None:
    global _last_backend_restart_at
    now = time.time()
    if now - _last_backend_restart_at < 30:
        return
    _last_backend_restart_at = now
    await asyncio.to_thread(_restart_backend)


async def _proxy_health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(HEALTH_URL)
        except Exception as exc:  # pragma: no cover - surfaced in UI instead
            return {"ok": False, "detail": str(exc)}

    try:
        payload = response.json()
    except ValueError:
        payload = {
            "ok": response.status_code == 200,
            "detail": {"status_code": response.status_code, "body": response.text[:300]},
        }

    if isinstance(payload, dict):
        payload.setdefault("ok", response.status_code == 200)
        return payload

    return {"ok": response.status_code == 200, "detail": {"status_code": response.status_code}}


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
        current_profile_id = state.get("active_profile_id")
        _set_state_profile(state, active_profile_id, reset_session=current_profile_id != active_profile_id and not state["busy"])
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
    created_profile_id = str(profile_payload.get("active_profile_id") or _load_active_profile_id())
    if body.select_after_create:
        created_profile_id = str(profile_payload.get("created_profile_id") or created_profile_id)
        profile_payload = await _proxy_select_profile(created_profile_id)

    active_profile_id = str(profile_payload.get("active_profile_id") or _load_active_profile_id())
    _save_active_profile_id(active_profile_id)

    with _state_lock:
        state = _browser_sessions[browser_id]
        _set_state_profile(state, active_profile_id, reset_session=True)
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
        _set_state_profile(state, active_profile_id, reset_session=True)
        payload = {
            "ok": True,
            "profiles": profile_payload.get("profiles") or [],
            "active_profile_id": active_profile_id,
            "state": _serialize_state(state),
        }
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
        "streaming": True,
    }

    prompt_messages: list[dict[str, Any]] = []
    chat_control_message = _chat_control_message()
    if chat_control_message is not None:
        prompt_messages.append(chat_control_message)
    guidance_message = _guidance_message(guidance_text)
    if guidance_message is not None:
        prompt_messages.append(guidance_message)
    prompt_messages.extend(context_messages)
    prompt_messages.append({"role": "user", "content": prompt})

    async def event_stream():
        committed = False
        final_text = ""

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
                max_tokens=UI_MAX_TOKENS,
            ):
                if await request.is_disconnected():
                    raise asyncio.CancelledError()

                if event["type"] == "delta":
                    delta = str(event.get("content") or "")
                    if not delta:
                        continue
                    final_text += delta
                    yield _sse_event(
                        "delta",
                        {
                            "assistant_turn_id": assistant_turn["id"],
                            "delta": delta,
                        },
                    )
                    continue

                if event["type"] == "finish":
                    yield _sse_event(
                        "final",
                        {
                            "assistant_turn_id": assistant_turn["id"],
                            "finish_reason": event.get("finish_reason", "stop"),
                        },
                    )

            assistant_text = final_text.strip()
            if not assistant_text:
                raise HTTPException(status_code=502, detail="Model returned an empty response")

            completed_assistant_turn = {
                **assistant_turn,
                "content": assistant_text,
                "feedback_pending": True,
                "streaming": False,
            }

            with _state_lock:
                state = _browser_sessions[browser_id]
                _commit_chat_turn(
                    state,
                    user_turn=user_turn,
                    assistant_turn=completed_assistant_turn,
                    prompt=prompt,
                    assistant_text=assistant_text,
                    prompt_messages=prompt_messages,
                    training_session_id=training_session_id,
                    guidance_text=guidance_text,
                )
                state["busy"] = False
                _touch_state(state)
                serialized_state = _serialize_state(state)

            committed = True
            yield _sse_event(
                "state",
                {
                    "state": serialized_state,
                },
            )
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

    training_session_id = uuid.uuid4().hex
    prompt_messages = []
    chat_control_message = _chat_control_message()
    if chat_control_message is not None:
        prompt_messages.append(chat_control_message)
    guidance_message = _guidance_message(guidance_text)
    if guidance_message is not None:
        prompt_messages.append(guidance_message)
    prompt_messages.extend(context_messages)
    prompt_messages.append({"role": "user", "content": prompt})

    try:
        proxy_response = await _proxy_chat(
            messages=prompt_messages,
            session_id=training_session_id,
            turn_type="main",
            session_done=False,
            max_tokens=UI_MAX_TOKENS,
        )
        choice = (proxy_response.get("choices") or [{}])[0]
        assistant_message = choice.get("message") or {}
        assistant_text = _assistant_text(assistant_message)
        if not assistant_text:
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
            "reasoning": (assistant_message.get("reasoning_content") or "").strip(),
        }

        with _state_lock:
            state = _browser_sessions[browser_id]
            _commit_chat_turn(
                state,
                user_turn=user_turn,
                assistant_turn=assistant_turn,
                prompt=prompt,
                assistant_text=assistant_text,
                prompt_messages=prompt_messages,
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
    rating_label = body.rating
    if rating_label not in {"good", "bad"}:
        if score >= 7:
            rating_label = "good"
        elif score <= 4:
            rating_label = "bad"
        else:
            rating_label = "neutral"

    try:
        if BACKEND_MODE == "rl_proxy":
            messages = (
                list(pending_episode["prompt_messages"])
                + [pending_episode["assistant_message"], {"role": "user", "content": feedback_message}]
            )
            await _proxy_chat(
                messages=messages,
                session_id=pending_episode["training_session_id"],
                turn_type="main",
                session_done=True,
                max_tokens=16,
            )
        elif BACKEND_MODE == "trainer_api":
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
                    "rating": rating_label,
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
                if item["id"] == candidate_id:
                    item["feedback"] = {
                        "rating": rating_label,
                        "score": score,
                        "note": clean_note,
                        "sent_text": feedback_message,
                        "created_at": time.time(),
                    }
                    item["feedback_pending"] = False
                    break
            state.get("feedback_candidates", {}).pop(candidate_id, None)
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


@app.post("/api/guidance")
async def api_guidance(request: Request, body: GuidanceRequest) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)
    with _state_lock:
        current_profile_id = _browser_sessions[browser_id].get("active_profile_id", _load_active_profile_id())
    clean_text = _save_guidance_text(current_profile_id, body.text)

    with _state_lock:
        for state in _browser_sessions.values():
            if state.get("active_profile_id", DEFAULT_PROFILE_ID) == current_profile_id:
                state["guidance_text"] = clean_text
                _touch_state(state)
        payload = {"ok": True, "state": _serialize_state(_browser_sessions[browser_id])}
    return _response_with_cookie(payload, browser_id, created)


@app.post("/api/reset")
async def api_reset(request: Request) -> JSONResponse:
    browser_id, _, created = _ensure_browser_session(request)

    with _state_lock:
        state = _browser_sessions[browser_id]
        if state["busy"]:
            raise HTTPException(status_code=409, detail="Another request is already running")
        feedback_candidates = list((state.get("feedback_candidates") or {}).values())
        state["busy"] = True

    try:
        if BACKEND_MODE == "rl_proxy":
            for candidate in feedback_candidates:
                await _proxy_chat(
                    messages=[{"role": "user", "content": "Reset conversation."}],
                    session_id=candidate["training_session_id"],
                    turn_type="side",
                    session_done=True,
                    max_tokens=1,
                )

        with _state_lock:
            _browser_sessions[browser_id] = _new_browser_state(_load_active_profile_id())
            _touch_state(_browser_sessions[browser_id])
            payload = {"ok": True, "state": _serialize_state(_browser_sessions[browser_id])}
        return _response_with_cookie(payload, browser_id, created)
    except Exception:
        with _state_lock:
            state = _browser_sessions[browser_id]
            state["busy"] = False
            _touch_state(state)
        raise


if __name__ == "__main__":
    uvicorn.run(app, host=UI_HOST, port=UI_PORT, log_level="info")
