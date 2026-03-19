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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATE_DIR = Path(os.environ.get("OPENCLAW_RL_UI_STATE_DIR", APP_DIR / "state")).expanduser()
GUIDANCE_FILE = STATE_DIR / "steering_notes.txt"
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
REQUEST_TIMEOUT = float(os.environ.get("OPENCLAW_RL_UI_TIMEOUT_SECONDS", "600"))
MAX_HISTORY_TURNS = int(os.environ.get("OPENCLAW_RL_UI_MAX_HISTORY_TURNS", "16"))
UI_MAX_TOKENS = int(os.environ.get("OPENCLAW_RL_UI_MAX_TOKENS", "384"))
UI_HOST = os.environ.get("OPENCLAW_RL_UI_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("OPENCLAW_RL_UI_PORT", "30001"))
RL_SERVICE_NAME = os.environ.get("OPENCLAW_RL_SERVICE_NAME", "openclaw-rl.service")
FORCE_NO_THINK = os.environ.get("OPENCLAW_RL_FORCE_NO_THINK", "1").strip().lower() in {"1", "true", "yes", "on"}
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
    score: int | None = None
    rating: str | None = None
    note: str = ""


class GuidanceRequest(BaseModel):
    text: str = ""


def _load_guidance_text() -> str:
    try:
        return GUIDANCE_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _save_guidance_text(text: str) -> str:
    clean_text = text.strip()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    GUIDANCE_FILE.write_text(f"{clean_text}\n" if clean_text else "", encoding="utf-8")
    return clean_text


def _new_browser_state() -> dict[str, Any]:
    return {
        "transcript": [],
        "context_messages": [],
        "pending_episode": None,
        "busy": False,
        "guidance_text": _load_guidance_text(),
        "updated_at": time.time(),
    }


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "transcript": state["transcript"],
        "awaiting_feedback": state["pending_episode"] is not None,
        "busy": state["busy"],
        "model": MODEL_NAME,
        "backend_mode": BACKEND_MODE,
        "proxy_base_url": PROXY_BASE_URL,
        "guidance_text": state.get("guidance_text", ""),
    }


def _touch_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()


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


def _response_with_cookie(payload: Any, browser_id: str, created: bool) -> JSONResponse:
    response = JSONResponse(payload)
    if created:
        response.set_cookie(
            key=COOKIE_NAME,
            value=browser_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 14,
        )
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


def _append_feedback_log(entry: dict[str, Any]) -> None:
    if FEEDBACK_LOG_FILE is None:
        return
    FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def _proxy_chat(
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    turn_type: str,
    session_done: bool,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
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
        if state["pending_episode"] is not None:
            raise HTTPException(status_code=409, detail="Label the last assistant response before sending another prompt")
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
            "reasoning": (assistant_message.get("reasoning_content") or "").strip(),
        }

        with _state_lock:
            state = _browser_sessions[browser_id]
            state["transcript"].extend([user_turn, assistant_turn])
            state["context_messages"] = _trim_context(
                state["context_messages"] + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            state["pending_episode"] = {
                "training_session_id": training_session_id,
                "prompt_messages": prompt_messages,
                "user_prompt": prompt,
                "guidance_text": guidance_text,
                "assistant_message": {"role": "assistant", "content": assistant_text},
                "assistant_turn_id": assistant_turn["id"],
            }
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
        pending_episode = state["pending_episode"]
        if pending_episode is None:
            raise HTTPException(status_code=409, detail="There is no assistant response waiting for feedback")
        state["busy"] = True

    feedback_message = _feedback_text(score, body.note)

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
                note=body.note.strip(),
                session_id=pending_episode["training_session_id"],
            )
        else:
            _append_feedback_log(
                {
                    "created_at": time.time(),
                    "model": MODEL_NAME,
                    "score": score,
                    "note": body.note.strip(),
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
                if item["id"] == pending_episode["assistant_turn_id"]:
                    item["feedback"] = {
                        "score": score,
                        "note": body.note.strip(),
                        "sent_text": feedback_message,
                        "created_at": time.time(),
                    }
                    break
            state["pending_episode"] = None
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
    clean_text = _save_guidance_text(body.text)

    with _state_lock:
        for state in _browser_sessions.values():
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
        pending_episode = state["pending_episode"]
        state["busy"] = True

    try:
        if BACKEND_MODE == "rl_proxy" and pending_episode is not None:
            await _proxy_chat(
                messages=[{"role": "user", "content": "Reset conversation."}],
                session_id=pending_episode["training_session_id"],
                turn_type="side",
                session_done=True,
                max_tokens=1,
            )

        with _state_lock:
            _browser_sessions[browser_id] = _new_browser_state()
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
