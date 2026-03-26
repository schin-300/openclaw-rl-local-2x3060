from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
import uuid
import gc
from pathlib import Path
from typing import Any

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from pydantic import BaseModel
from transformers import (
    AsyncTextIteratorStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - optional runtime optimization
    bnb = None


MODEL_ID = os.environ.get("QWEN35_MODEL_ID", "Qwen/Qwen3.5-4B")
SERVED_MODEL_NAME = os.environ.get("QWEN35_SERVED_MODEL_NAME", "qwen3.5-4b-local")
HOST = os.environ.get("QWEN35_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN35_PORT", "30100"))
MAX_INPUT_TOKENS = int(os.environ.get("QWEN35_MAX_INPUT_TOKENS", "2048"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QWEN35_DEFAULT_MAX_NEW_TOKENS", "192"))
DEFAULT_TEMPERATURE = float(os.environ.get("QWEN35_DEFAULT_TEMPERATURE", "0.6"))
DEFAULT_TOP_P = float(os.environ.get("QWEN35_DEFAULT_TOP_P", "0.9"))
STREAM_TIMEOUT = float(os.environ.get("QWEN35_STREAM_TIMEOUT_SECONDS", "600"))
ATTN_IMPLEMENTATION = os.environ.get("QWEN35_ATTN_IMPLEMENTATION", "sdpa")
TRUST_REMOTE_CODE = os.environ.get("QWEN35_TRUST_REMOTE_CODE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOAD_IN_4BIT = os.environ.get("QWEN35_LOAD_IN_4BIT", "1").strip().lower() in {"1", "true", "yes", "on"}
COMPUTE_DTYPE_NAME = os.environ.get("QWEN35_COMPUTE_DTYPE", "float16").strip().lower()
STATE_DIR = Path(os.environ.get("QWEN35_STATE_DIR", Path(__file__).resolve().parent / "state")).expanduser()
PROFILES_DIR = STATE_DIR / "profiles"
ACTIVE_PROFILE_PATH = STATE_DIR / "active_profile.txt"
DEFAULT_PROFILE_ID = os.environ.get("QWEN35_DEFAULT_PROFILE_ID", "default").strip() or "default"
DEFAULT_PROFILE_NAME = os.environ.get("QWEN35_DEFAULT_PROFILE_NAME", "Default").strip() or "Default"
LORA_R = int(os.environ.get("QWEN35_LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("QWEN35_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("QWEN35_LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = tuple(
    part.strip()
    for part in os.environ.get(
        "QWEN35_LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ).split(",")
    if part.strip()
)
TRAIN_LR = float(os.environ.get("QWEN35_TRAIN_LR", "8e-5"))
TRAIN_WEIGHT_DECAY = float(os.environ.get("QWEN35_TRAIN_WEIGHT_DECAY", "0.0"))
TRAIN_CLIP_NORM = float(os.environ.get("QWEN35_TRAIN_CLIP_NORM", "1.0"))
TRAIN_MAX_RESPONSE_TOKENS = int(os.environ.get("QWEN35_TRAIN_MAX_RESPONSE_TOKENS", "224"))
TRAIN_GRADIENT_CHECKPOINTING = os.environ.get("QWEN35_TRAIN_GRADIENT_CHECKPOINTING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRAIN_SAVE_EVERY_STEPS = int(os.environ.get("QWEN35_TRAIN_SAVE_EVERY_STEPS", "1"))
SGLANG_ENABLED = os.environ.get("QWEN35_USE_SGLANG", "1").strip().lower() in {"1", "true", "yes", "on"}
SGLANG_BASE_URL = os.environ.get("QWEN35_SGLANG_BASE_URL", "http://127.0.0.1:30101").rstrip("/")
SGLANG_CHAT_URL = f"{SGLANG_BASE_URL}/v1/chat/completions"
SGLANG_HEALTH_URL = f"{SGLANG_BASE_URL}/health"
SGLANG_REQUEST_TIMEOUT = float(os.environ.get("QWEN35_SGLANG_TIMEOUT_SECONDS", str(max(STREAM_TIMEOUT, 600.0))))
SGLANG_SERVED_MODEL_NAME = os.environ.get("QWEN35_SGLANG_SERVED_MODEL_NAME", SERVED_MODEL_NAME).strip() or SERVED_MODEL_NAME
SERVING_ALIASES_DIRNAME = os.environ.get("QWEN35_SERVING_ALIASES_DIRNAME", "serving-adapters").strip() or "serving-adapters"
SERVING_ALIASES_KEEP = max(1, int(os.environ.get("QWEN35_SERVING_ALIASES_KEEP", "3")))

_EXPLICIT_FEEDBACK_REWARD_BY_SCORE = {
    1: -1.00,
    2: -0.92,
    3: -0.68,
    4: -0.30,
    5: 0.00,
    6: 0.18,
    7: 0.45,
    8: 0.78,
    9: 0.94,
    10: 1.00,
}
_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


app = FastAPI(title="Qwen3.5 Experimental Server")

_model_lock = threading.RLock()
_tokenizer = None
_model = None
_optimizer = None
_marker_ids = None
_loaded_at = None
_train_state: dict[str, Any] | None = None
_trainable_param_count = 0
_active_profile_id: str | None = None


class _StopOnEventCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self._stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._stop_event.is_set()


class ProfileCreateRequest(BaseModel):
    name: str


class ProfileSelectRequest(BaseModel):
    profile_id: str


def _slugify_profile_id(name: str) -> str:
    lowered = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or DEFAULT_PROFILE_ID


def _profile_dir(profile_id: str) -> Path:
    return PROFILES_DIR / profile_id


def _profile_meta_path(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "meta.json"


def _profile_adapter_dir(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "adapter"


def _profile_optimizer_path(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "optimizer.pt"


def _profile_train_state_path(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "train_state.json"


def _profile_feedback_log_path(profile_id: str) -> Path:
    return _profile_dir(profile_id) / "feedback.jsonl"


def _profile_serving_aliases_dir(profile_id: str) -> Path:
    return _profile_dir(profile_id) / SERVING_ALIASES_DIRNAME


def _default_profile_meta(profile_id: str, name: str, *, created_at: float | None = None) -> dict[str, Any]:
    timestamp = created_at or time.time()
    return {
        "id": profile_id,
        "name": name.strip() or profile_id,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _write_profile_meta(meta: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(meta["id"])
    profile_dir = _profile_dir(profile_id)
    profile_dir.mkdir(parents=True, exist_ok=True)
    meta["updated_at"] = float(meta.get("updated_at") or time.time())
    _profile_meta_path(profile_id).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def _load_profile_meta(profile_id: str) -> dict[str, Any]:
    meta_path = _profile_meta_path(profile_id)
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = _default_profile_meta(profile_id, profile_id)
    except json.JSONDecodeError:
        payload = _default_profile_meta(profile_id, profile_id)
    if not isinstance(payload, dict):
        payload = _default_profile_meta(profile_id, profile_id)
    payload.setdefault("id", profile_id)
    payload.setdefault("name", profile_id)
    payload.setdefault("created_at", time.time())
    payload.setdefault("updated_at", payload["created_at"])
    return payload


def _iter_profile_ids() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    profile_ids: list[str] = []
    for child in sorted(PROFILES_DIR.iterdir(), key=lambda item: item.name):
        if child.is_dir():
            profile_ids.append(child.name)
    return profile_ids


def _profile_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _read_active_profile_id() -> str:
    try:
        profile_id = ACTIVE_PROFILE_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return DEFAULT_PROFILE_ID
    return profile_id or DEFAULT_PROFILE_ID


def _write_active_profile_id(profile_id: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROFILE_PATH.write_text(f"{profile_id}\n", encoding="utf-8")


def _ensure_profile_storage_initialized() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    profile_ids = [child.name for child in PROFILES_DIR.iterdir() if child.is_dir()]
    legacy_paths = [
        STATE_DIR / "adapter",
        STATE_DIR / "optimizer.pt",
        STATE_DIR / "train_state.json",
        STATE_DIR / "feedback.jsonl",
    ]
    has_legacy_root_state = any(path.exists() for path in legacy_paths)
    default_dir = _profile_dir(DEFAULT_PROFILE_ID)

    if not profile_ids:
        default_meta = _default_profile_meta(DEFAULT_PROFILE_ID, DEFAULT_PROFILE_NAME)
        default_dir.mkdir(parents=True, exist_ok=True)
        _write_profile_meta(default_meta)

        if has_legacy_root_state:
            move_targets = {
                STATE_DIR / "adapter": _profile_adapter_dir(DEFAULT_PROFILE_ID),
                STATE_DIR / "optimizer.pt": _profile_optimizer_path(DEFAULT_PROFILE_ID),
                STATE_DIR / "train_state.json": _profile_train_state_path(DEFAULT_PROFILE_ID),
                STATE_DIR / "feedback.jsonl": _profile_feedback_log_path(DEFAULT_PROFILE_ID),
            }
            for source, destination in move_targets.items():
                if source.exists() and not destination.exists():
                    source.replace(destination)

    if not _profile_meta_path(DEFAULT_PROFILE_ID).exists():
        _write_profile_meta(_default_profile_meta(DEFAULT_PROFILE_ID, DEFAULT_PROFILE_NAME))

    active_profile_id = _read_active_profile_id()
    if active_profile_id not in _iter_profile_ids():
        active_profile_id = DEFAULT_PROFILE_ID
        _write_active_profile_id(active_profile_id)


def _get_active_profile_id() -> str:
    global _active_profile_id
    _ensure_profile_storage_initialized()
    if _active_profile_id is None:
        _active_profile_id = _read_active_profile_id()
    if _active_profile_id not in _iter_profile_ids():
        _active_profile_id = DEFAULT_PROFILE_ID
        _write_active_profile_id(_active_profile_id)
    return _active_profile_id


def _active_profile_dir() -> Path:
    return _profile_dir(_get_active_profile_id())


def _active_adapter_dir() -> Path:
    return _profile_adapter_dir(_get_active_profile_id())


def _active_optimizer_path() -> Path:
    return _profile_optimizer_path(_get_active_profile_id())


def _active_train_state_path() -> Path:
    return _profile_train_state_path(_get_active_profile_id())


def _active_feedback_log_path() -> Path:
    return _profile_feedback_log_path(_get_active_profile_id())


def _profile_summary(profile_id: str) -> dict[str, Any]:
    meta = _load_profile_meta(profile_id)
    train_state = _load_train_state(_profile_train_state_path(profile_id))
    profile_dir = _profile_dir(profile_id)
    size_bytes = _profile_dir_size(profile_dir)
    return {
        "id": profile_id,
        "name": str(meta.get("name") or profile_id),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "size_bytes": size_bytes,
        "adapter_ready": (_profile_adapter_dir(profile_id) / "adapter_config.json").exists(),
        "training": train_state,
    }


def _reset_loaded_runtime() -> None:
    global _model, _optimizer, _train_state, _loaded_at, _trainable_param_count
    _model = None
    _optimizer = None
    _train_state = None
    _loaded_at = None
    _trainable_param_count = 0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _activate_profile(profile_id: str, *, load_model: bool = True) -> dict[str, Any]:
    global _active_profile_id
    _ensure_profile_storage_initialized()
    if profile_id not in _iter_profile_ids():
        raise HTTPException(status_code=404, detail=f"Unknown profile: {profile_id}")
    _active_profile_id = profile_id
    _write_active_profile_id(profile_id)
    _reset_loaded_runtime()
    if load_model:
        _load_model()
        _ensure_active_serving_adapter_path(force_refresh=False)
    return _profile_summary(profile_id)


def _compute_dtype() -> torch.dtype:
    return _DTYPE_BY_NAME.get(COMPUTE_DTYPE_NAME, torch.float16 if torch.cuda.is_available() else torch.float32)


def _default_train_state() -> dict[str, Any]:
    return {
        "train_steps": 0,
        "total_feedback": 0,
        "total_updates": 0,
        "last_feedback_at": None,
        "last_feedback_score": None,
        "last_reward": None,
        "last_ce_loss": None,
        "last_objective": None,
        "last_grad_norm": None,
        "last_note": "",
        "last_session_id": "",
        "serving_adapter_path": "",
        "last_serving_synced_at": None,
    }


def _load_train_state(path: Path | None = None) -> dict[str, Any]:
    train_state_path = path or _active_train_state_path()
    try:
        raw = json.loads(train_state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _default_train_state()
    except json.JSONDecodeError:
        return _default_train_state()
    if not isinstance(raw, dict):
        return _default_train_state()
    state = _default_train_state()
    state.update(raw)
    return state


def _save_train_state() -> None:
    if _train_state is None:
        return
    active_dir = _active_profile_dir()
    active_dir.mkdir(parents=True, exist_ok=True)
    _active_train_state_path().write_text(
        json.dumps(_train_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except FileNotFoundError:
        return


def _refresh_serving_adapter_alias(profile_id: str) -> str | None:
    adapter_dir = _profile_adapter_dir(profile_id)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    aliases_dir = _profile_serving_aliases_dir(profile_id)
    aliases_dir.mkdir(parents=True, exist_ok=True)

    train_steps = 0
    if _train_state is not None:
        train_steps = int(_train_state.get("train_steps", 0) or 0)
    alias_path = aliases_dir / f"adapter-step-{train_steps:06d}-{int(time.time())}"
    if alias_path.exists() or alias_path.is_symlink():
        _remove_path(alias_path)
    alias_path.symlink_to(adapter_dir, target_is_directory=True)

    alias_entries = sorted(
        aliases_dir.iterdir(),
        key=lambda item: item.lstat().st_mtime if item.exists() or item.is_symlink() else 0.0,
        reverse=True,
    )
    for stale in alias_entries[SERVING_ALIASES_KEEP:]:
        _remove_path(stale)

    return str(alias_path)


def _ensure_active_serving_adapter_path(*, force_refresh: bool = False) -> str | None:
    global _train_state

    if _train_state is None:
        _train_state = _load_train_state()

    adapter_ready = (_active_adapter_dir() / "adapter_config.json").exists()
    if not adapter_ready:
        _train_state["serving_adapter_path"] = ""
        _train_state["last_serving_synced_at"] = None
        _save_train_state()
        return None

    existing_value = str(_train_state.get("serving_adapter_path") or "").strip()
    existing_path = Path(existing_value).expanduser() if existing_value else None
    if not force_refresh and existing_path is not None and existing_path.exists():
        return str(existing_path)

    refreshed = _refresh_serving_adapter_alias(_get_active_profile_id())
    _train_state["serving_adapter_path"] = refreshed or ""
    _train_state["last_serving_synced_at"] = time.time() if refreshed else None
    _save_train_state()
    return refreshed


def _active_serving_adapter_path() -> str | None:
    return _ensure_active_serving_adapter_path(force_refresh=False)


def _append_feedback_log(entry: dict[str, Any]) -> None:
    active_dir = _active_profile_dir()
    active_dir.mkdir(parents=True, exist_ok=True)
    with _active_feedback_log_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        role = str(message.get("role") or "").strip()
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"unsupported role: {role or '<empty>'}")
        content = _normalize_content(message.get("content"))
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise HTTPException(status_code=400, detail="messages contained no usable text content")
    return normalized


def _prepare_qwen_template_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            content = message["content"].strip()
            if content:
                system_parts.append(content)
            continue
        conversation.append(message)

    if not system_parts:
        return list(messages)

    merged_system = "\n\n".join(system_parts).strip()
    if not merged_system:
        return conversation
    return [{"role": "system", "content": merged_system}, *conversation]


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


def _first_model_device() -> torch.device:
    if _model is None:
        raise RuntimeError("Model has not been loaded")
    return next(_model.parameters()).device


def _bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _chat_template_kwargs(*, enable_thinking: bool) -> dict[str, Any]:
    return {
        "tokenize": True,
        "enable_thinking": bool(enable_thinking),
    }


def _tokenize_messages(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> torch.Tensor:
    template_messages = _prepare_qwen_template_messages(messages)
    ids = _tokenizer.apply_chat_template(
        template_messages,
        add_generation_prompt=add_generation_prompt,
        **_chat_template_kwargs(enable_thinking=enable_thinking),
    )
    if not isinstance(ids, str) and hasattr(ids, "get"):
        extracted_ids = ids.get("input_ids")
        if extracted_ids is not None:
            ids = extracted_ids
    if isinstance(ids, torch.Tensor):
        return ids.to(dtype=torch.long).view(-1)
    if isinstance(ids, str):
        ids = _tokenizer(ids, add_special_tokens=False)["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return torch.tensor(ids, dtype=torch.long)


def _truncate_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    global _marker_ids

    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    if input_ids.shape[0] <= MAX_INPUT_TOKENS:
        return batch

    if _marker_ids is None:
        marker_text = "\n[OMITTED MIDDLE CONTEXT]\n"
        marker_ids = _tokenizer.encode(marker_text, add_special_tokens=False)
        _marker_ids = torch.tensor(marker_ids, dtype=input_ids.dtype)

    marker_ids = _marker_ids
    keep_head = max(128, int(MAX_INPUT_TOKENS * 0.2))
    tail = MAX_INPUT_TOKENS - keep_head - int(marker_ids.shape[0])
    if tail <= 0:
        tail = max(1, MAX_INPUT_TOKENS - keep_head)
        marker_ids = marker_ids[:0]

    truncated_ids = torch.cat([input_ids[:keep_head], marker_ids, input_ids[-tail:]])
    truncated_mask = torch.ones_like(truncated_ids, dtype=attention_mask.dtype)
    batch["input_ids"] = truncated_ids.unsqueeze(0)
    batch["attention_mask"] = truncated_mask.unsqueeze(0)
    return batch


def _strip_thinking(text: str) -> str:
    cleaned = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    cleaned = re.sub(r"^\s*<think>\s*</think>\s*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _strip_thinking_partial(text: str, *, final: bool) -> str:
    cleaned = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

    open_idx = cleaned.rfind("<think>")
    close_idx = cleaned.rfind("</think>")
    if open_idx != -1 and open_idx > close_idx:
        cleaned = cleaned[:open_idx]

    if not final:
        tail_lt = cleaned.rfind("<")
        tail_gt = cleaned.rfind(">")
        if tail_lt > tail_gt and len(cleaned) - tail_lt <= 32:
            cleaned = cleaned[:tail_lt]
        return cleaned.lstrip()

    cleaned = re.sub(r"^\s*<think>\s*</think>\s*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _trim_partial_markup(text: str, *, final: bool) -> str:
    if final:
        return text.strip()
    tail_lt = text.rfind("<")
    tail_gt = text.rfind(">")
    if tail_lt > tail_gt and len(text) - tail_lt <= 32:
        text = text[:tail_lt]
    return text.lstrip()


def _split_labeled_final_answer(text: str, *, final: bool) -> tuple[str, str] | None:
    matches = list(re.finditer(r"(?:^|\n)\s*(?:Final Answer|Final Response|Answer|Response)\s*:\s*", text, flags=re.I))
    if not matches:
        return None
    match = matches[-1]
    reasoning = _trim_partial_markup(text[: match.start()], final=final)
    content = _trim_partial_markup(text[match.end() :], final=final)
    return reasoning, content


def _looks_like_reasoning_transcript(text: str) -> bool:
    stripped = text.lstrip()
    return bool(re.match(r"(?is)(thinking process:|1\.\s+\*\*analyze the request|\d+\.\s+\*\*)", stripped))


def _split_thinking_text(text: str, *, final: bool) -> tuple[str, str]:
    cleaned = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    labeled_split = _split_labeled_final_answer(cleaned, final=final)
    if labeled_split is not None:
        return labeled_split
    open_idx = cleaned.find("<think>")
    if open_idx == -1:
        close_idx = cleaned.find("</think>")
        if close_idx != -1:
            reasoning = _trim_partial_markup(cleaned[:close_idx], final=final)
            content = _trim_partial_markup(cleaned[close_idx + len("</think>") :], final=final)
            return reasoning, content
        if not final:
            reasoning = _trim_partial_markup(cleaned, final=False)
            return reasoning, ""
        if _looks_like_reasoning_transcript(cleaned):
            reasoning = _trim_partial_markup(cleaned, final=final)
            return reasoning, ""
        content = _trim_partial_markup(cleaned, final=final)
        return "", content

    before = cleaned[:open_idx]
    after_open = cleaned[open_idx + len("<think>") :]
    close_idx = after_open.find("</think>")
    if close_idx == -1:
        reasoning = after_open
        content = before
    else:
        reasoning = after_open[:close_idx]
        content = before + after_open[close_idx + len("</think>") :]

    reasoning = _trim_partial_markup(reasoning, final=final)
    content = _trim_partial_markup(content, final=final)
    return reasoning, content


def _generation_kwargs(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    streamer: AsyncTextIteratorStreamer | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    do_sample = temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": _tokenizer.pad_token_id or _tokenizer.eos_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
    if stop_event is not None:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnEventCriteria(stop_event)])
    return generation_kwargs


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_metrics(*, visible_text: str, raw_text: str, started_at: float) -> dict[str, Any]:
    visible_tokens = 0
    if visible_text.strip():
        visible_tokens = len(_tokenizer.encode(visible_text, add_special_tokens=False))
    generated_tokens = 0
    cleaned_raw = raw_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    if cleaned_raw.strip():
        generated_tokens = len(_tokenizer.encode(cleaned_raw, add_special_tokens=False))
    elapsed = max(time.perf_counter() - started_at, 1e-3)
    return {
        "visible_tokens": visible_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / elapsed, 2),
        "elapsed_seconds": round(elapsed, 3),
    }


def _score_to_reward(score: int) -> float:
    return _EXPLICIT_FEEDBACK_REWARD_BY_SCORE.get(score, 0.0)


def _build_sglang_request_body(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    stream: bool,
    lora_path: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": SGLANG_SERVED_MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "separate_reasoning": True,
        "stream_reasoning": True,
        "chat_template_kwargs": {
            "enable_thinking": bool(enable_thinking),
        },
    }
    if lora_path:
        body["lora_path"] = lora_path
    return body


def _sglang_health_snapshot() -> dict[str, Any]:
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(SGLANG_HEALTH_URL)
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}

    try:
        payload = response.json()
    except ValueError:
        payload = {"status_code": response.status_code, "body": response.text[:300]}

    if isinstance(payload, dict):
        payload.setdefault("ok", response.status_code == 200)
        return payload
    return {"ok": response.status_code == 200, "detail": payload}


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )


def _build_optimizer(model) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if bnb is not None and torch.cuda.is_available():
        return bnb.optim.PagedAdamW8bit(params, lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    return torch.optim.AdamW(params, lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)


def _move_optimizer_state_to_device(device: torch.device) -> None:
    if _optimizer is None:
        return
    for state in _optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _save_adapter_checkpoint() -> None:
    if _model is None:
        return
    active_dir = _active_profile_dir()
    active_dir.mkdir(parents=True, exist_ok=True)
    _model.save_pretrained(_active_adapter_dir())
    if _optimizer is not None:
        torch.save(_optimizer.state_dict(), _active_optimizer_path())
    _ensure_active_serving_adapter_path(force_refresh=True)
    _save_train_state()


def _load_model() -> None:
    global _loaded_at, _model, _optimizer, _tokenizer, _train_state, _trainable_param_count

    if _model is not None and _tokenizer is not None and _optimizer is not None and _train_state is not None:
        return

    _ensure_profile_storage_initialized()
    active_profile_id = _get_active_profile_id()
    active_dir = _profile_dir(active_profile_id)
    active_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    dtype = _compute_dtype()
    quantization_config = None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "torch_dtype": dtype,
    }
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["quantization_config"] = quantization_config

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    if LOAD_IN_4BIT:
        base_model = prepare_model_for_kbit_training(base_model)
    if TRAIN_GRADIENT_CHECKPOINTING and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    adapter_dir = _profile_adapter_dir(active_profile_id)
    optimizer_path = _profile_optimizer_path(active_profile_id)
    train_state_path = _profile_train_state_path(active_profile_id)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        _model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    else:
        _model = get_peft_model(base_model, _lora_config())

    _optimizer = _build_optimizer(_model)
    _train_state = _load_train_state(train_state_path)
    _trainable_param_count = sum(param.numel() for param in _model.parameters() if param.requires_grad)

    if optimizer_path.exists():
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        _optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(_first_model_device())

    _model.eval()
    _loaded_at = time.time()


def _build_inputs(messages: list[dict[str, str]], *, enable_thinking: bool = False) -> dict[str, torch.Tensor]:
    template_messages = _prepare_qwen_template_messages(messages)
    batch = _tokenizer.apply_chat_template(
        template_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=enable_thinking,
    )
    batch = _truncate_inputs(batch)
    device = _first_model_device()
    return {name: value.to(device) for name, value in batch.items()}


def _build_training_tensors(
    prompt_messages: list[dict[str, str]],
    assistant_text: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prompt_ids = _tokenize_messages(prompt_messages, add_generation_prompt=True)
    full_ids = _tokenize_messages(
        prompt_messages + [{"role": "assistant", "content": assistant_text}],
        add_generation_prompt=False,
    )
    if full_ids.shape[0] <= prompt_ids.shape[0]:
        raise HTTPException(status_code=400, detail="Assistant response did not produce any trainable tokens")

    response_ids = full_ids[prompt_ids.shape[0] :]
    if response_ids.shape[0] > TRAIN_MAX_RESPONSE_TOKENS:
        response_ids = response_ids[:TRAIN_MAX_RESPONSE_TOKENS]

    train_ids = torch.cat([prompt_ids, response_ids], dim=0)
    assistant_start = int(prompt_ids.shape[0])
    if train_ids.shape[0] > MAX_INPUT_TOKENS:
        overflow = int(train_ids.shape[0] - MAX_INPUT_TOKENS)
        train_ids = train_ids[overflow:]
        assistant_start = max(0, assistant_start - overflow)
    if assistant_start >= train_ids.shape[0]:
        raise HTTPException(status_code=400, detail="Prompt consumed the full training window")

    attention_mask = torch.ones_like(train_ids)
    labels = train_ids.clone()
    labels[:assistant_start] = -100
    device = _first_model_device()
    return (
        train_ids.unsqueeze(0).to(device),
        attention_mask.unsqueeze(0).to(device),
        labels.unsqueeze(0).to(device),
        int((labels != -100).sum().item()),
    )


def _finalize_generated_text(raw_text: str, *, enable_thinking: bool) -> tuple[str, str]:
    if enable_thinking:
        reasoning_text, visible_text = _split_thinking_text(raw_text, final=True)
    else:
        reasoning_text = ""
        visible_text = _strip_thinking(raw_text)
    reasoning_text = reasoning_text.strip()
    visible_text = visible_text.strip()
    return reasoning_text, visible_text


def _chat_completion_payload(
    *,
    response_id: str,
    created_at: int,
    visible_text: str,
    reasoning_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created_at,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": visible_text,
                    "reasoning_content": reasoning_text,
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "metrics": metrics,
    }


def _generate_local_chat_completion(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> dict[str, Any]:
    with _model_lock:
        _load_model()
        model_inputs = _build_inputs(messages, enable_thinking=enable_thinking)
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        generation_kwargs = _generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids = _model.generate(**model_inputs, **generation_kwargs)

    generated_ids = output_ids[0][prompt_tokens:]
    completion_tokens = int(generated_ids.shape[0])
    raw_text = _tokenizer.decode(generated_ids, skip_special_tokens=False)
    reasoning_text, visible_text = _finalize_generated_text(raw_text, enable_thinking=enable_thinking)
    if not visible_text and not reasoning_text:
        raise HTTPException(status_code=502, detail="Local model returned an empty response")

    metrics = _stream_metrics(
        visible_text=visible_text or reasoning_text,
        raw_text=raw_text,
        started_at=started_at,
    )
    return _chat_completion_payload(
        response_id=f"chatcmpl-{uuid.uuid4().hex}",
        created_at=int(time.time()),
        visible_text=visible_text,
        reasoning_text=reasoning_text if enable_thinking else "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        metrics=metrics,
    )


async def _stream_local_chat_completion(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
):
    with _model_lock:
        _load_model()
        model_inputs = _build_inputs(messages, enable_thinking=enable_thinking)
        prompt_tokens = int(model_inputs["input_ids"].shape[1])

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_at = int(time.time())
    started_at: float | None = None
    raw_text = ""
    visible_text = ""
    reasoning_text = ""
    sent_role = False
    stop_event = threading.Event()
    error_holder: dict[str, Exception] = {}
    streamer = AsyncTextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        timeout=STREAM_TIMEOUT,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    generation_kwargs = _generation_kwargs(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        streamer=streamer,
        stop_event=stop_event,
    )

    def _run_generation() -> None:
        try:
            with _model_lock:
                with torch.inference_mode():
                    _model.generate(**model_inputs, **generation_kwargs)
        except Exception as exc:  # pragma: no cover - background generation path
            error_holder["error"] = exc
            streamer.on_finalized_text("", stream_end=True)

    thread = threading.Thread(target=_run_generation, daemon=True)
    thread.start()

    try:
        async for chunk in streamer:
            if started_at is None:
                started_at = time.perf_counter()

            raw_text += chunk
            if enable_thinking:
                next_reasoning_text, next_visible_text = _split_thinking_text(raw_text, final=False)
            else:
                next_reasoning_text = ""
                next_visible_text = _strip_thinking_partial(raw_text, final=False)

            reasoning_delta = next_reasoning_text[len(reasoning_text) :] if next_reasoning_text.startswith(reasoning_text) else next_reasoning_text
            content_delta = next_visible_text[len(visible_text) :] if next_visible_text.startswith(visible_text) else next_visible_text
            reasoning_text = next_reasoning_text
            visible_text = next_visible_text

            metrics = (
                _stream_metrics(visible_text=visible_text, raw_text=raw_text, started_at=started_at)
                if started_at is not None
                else {}
            )

            if not content_delta and not reasoning_delta:
                continue

            delta_payload: dict[str, Any] = {}
            if content_delta:
                delta_payload["content"] = content_delta
            if reasoning_delta:
                delta_payload["reasoning_content"] = reasoning_delta
            if not sent_role:
                delta_payload["role"] = "assistant"
                sent_role = True

            yield _sse_data(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": SERVED_MODEL_NAME,
                    "metrics": metrics,
                    "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                }
            )

        if "error" in error_holder:
            raise HTTPException(status_code=502, detail=f"Local generation failed: {error_holder['error']}")

        final_reasoning_text, final_visible_text = _finalize_generated_text(raw_text, enable_thinking=enable_thinking)
        if not final_visible_text and not final_reasoning_text:
            raise HTTPException(status_code=502, detail="Local model returned an empty response")

        final_metrics = _stream_metrics(
            visible_text=final_visible_text or final_reasoning_text,
            raw_text=raw_text,
            started_at=started_at or time.perf_counter(),
        )
        yield _sse_data(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": SERVED_MODEL_NAME,
                "metrics": final_metrics,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(_tokenizer.encode(raw_text, add_special_tokens=False)),
                    "total_tokens": prompt_tokens + len(_tokenizer.encode(raw_text, add_special_tokens=False)),
                },
            }
        )
        yield "data: [DONE]\n\n"
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


def _generate_chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    messages = _normalize_messages(body.get("messages"))
    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
    max_tokens = max(1, min(max_tokens, 1024))
    temperature = float(body.get("temperature") if body.get("temperature") is not None else DEFAULT_TEMPERATURE)
    top_p = float(body.get("top_p") if body.get("top_p") is not None else DEFAULT_TOP_P)
    enable_thinking = _bool_flag(body.get("enable_thinking") or body.get("thinking"))
    if not SGLANG_ENABLED:
        return _generate_local_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
    with _model_lock:
        _load_model()
        lora_path = _active_serving_adapter_path() if SGLANG_ENABLED else None

    request_body = _build_sglang_request_body(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        stream=False,
        lora_path=lora_path,
    )

    try:
        with httpx.Client(timeout=SGLANG_REQUEST_TIMEOUT) as client:
            response = client.post(SGLANG_CHAT_URL, json=request_body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"SGLang request failed: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"SGLang error {response.status_code}: {response.text[:1200]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="SGLang returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="SGLang returned an unexpected payload shape")

    payload["model"] = SERVED_MODEL_NAME
    if isinstance(payload.get("choices"), list):
        for choice in payload["choices"]:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            if not enable_thinking:
                message["reasoning_content"] = ""
            message.setdefault("role", "assistant")
    return payload


async def _stream_chat_completion(body: dict[str, Any]):
    messages = _normalize_messages(body.get("messages"))
    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
    max_tokens = max(1, min(max_tokens, 1024))
    temperature = float(body.get("temperature") if body.get("temperature") is not None else DEFAULT_TEMPERATURE)
    top_p = float(body.get("top_p") if body.get("top_p") is not None else DEFAULT_TOP_P)
    enable_thinking = _bool_flag(body.get("enable_thinking") or body.get("thinking"))
    if not SGLANG_ENABLED:
        async for payload in _stream_local_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        ):
            yield payload
        return
    with _model_lock:
        _load_model()
        lora_path = _active_serving_adapter_path() if SGLANG_ENABLED else None

    request_body = _build_sglang_request_body(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        stream=True,
        lora_path=lora_path,
    )

    started_at: float | None = None
    visible_text = ""
    reasoning_text = ""
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_at = int(time.time())
    sent_role = False

    timeout = httpx.Timeout(SGLANG_REQUEST_TIMEOUT, read=SGLANG_REQUEST_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", SGLANG_CHAT_URL, json=request_body) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:1200]
                    raise HTTPException(status_code=502, detail=f"SGLang error {response.status_code}: {detail}")

                async for raw_payload in _aiter_sse_payloads(response):
                    if raw_payload == "[DONE]":
                        break

                    try:
                        payload = json.loads(raw_payload)
                    except json.JSONDecodeError as exc:
                        raise HTTPException(status_code=502, detail="SGLang returned invalid streaming JSON") from exc

                    if not isinstance(payload, dict):
                        continue

                    if started_at is None:
                        started_at = time.perf_counter()
                    response_id = str(payload.get("id") or response_id)
                    created_at = int(payload.get("created") or created_at)

                    choices = payload.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")

                    content_delta = str(delta.get("content") or "")
                    reasoning_delta = str(delta.get("reasoning_content") or "") if enable_thinking else ""
                    if content_delta:
                        visible_text += content_delta
                    if reasoning_delta:
                        reasoning_text += reasoning_delta

                    raw_text = f"{reasoning_text}{visible_text}"
                    metrics = (
                        _stream_metrics(visible_text=visible_text, raw_text=raw_text, started_at=started_at)
                        if started_at is not None
                        else {}
                    )

                    if content_delta or reasoning_delta or finish_reason is not None:
                        emitted_delta: dict[str, Any] = {}
                        if content_delta:
                            emitted_delta["content"] = content_delta
                        if reasoning_delta:
                            emitted_delta["reasoning_content"] = reasoning_delta
                        if not sent_role:
                            emitted_delta["role"] = "assistant"
                            sent_role = True
                        elif "role" in delta:
                            emitted_delta["role"] = delta["role"]
                        yield _sse_data(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created_at,
                                "model": SERVED_MODEL_NAME,
                                "metrics": metrics,
                                "choices": [{"index": 0, "delta": emitted_delta, "finish_reason": finish_reason}],
                            }
                        )

                if not visible_text:
                    raise HTTPException(status_code=502, detail="SGLang returned an empty response")

                final_metrics = (
                    _stream_metrics(
                        visible_text=visible_text,
                        raw_text=f"{reasoning_text}{visible_text}",
                        started_at=started_at or time.perf_counter(),
                    )
                )
                yield _sse_data(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": SERVED_MODEL_NAME,
                        "metrics": final_metrics,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
                yield "data: [DONE]\n\n"
    except asyncio.CancelledError:  # pragma: no cover - client disconnect path
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"SGLang streaming request failed: {exc}") from exc


def _train_on_feedback(body: dict[str, Any]) -> dict[str, Any]:
    score = int(body.get("score") or 0)
    if score < 1 or score > 10:
        raise HTTPException(status_code=400, detail="score must be between 1 and 10")

    prompt_messages = _normalize_messages(body.get("messages"))
    assistant_text = _normalize_content(body.get("assistant_response"))
    if not assistant_text:
        raise HTTPException(status_code=400, detail="assistant_response must contain text")

    note = str(body.get("note") or "").strip()
    session_id = str(body.get("session_id") or uuid.uuid4().hex)
    reward = _score_to_reward(score)

    with _model_lock:
        _load_model()
        input_ids, attention_mask, labels, response_token_count = _build_training_tensors(prompt_messages, assistant_text)
        ce_loss_value = None
        objective_value = None
        grad_norm_value = None
        updated = False

        if abs(reward) > 1e-9:
            _model.train()
            _optimizer.zero_grad(set_to_none=True)
            outputs = _model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            ce_loss = outputs.loss
            objective = ce_loss * reward
            objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in _model.parameters() if param.requires_grad],
                TRAIN_CLIP_NORM,
            )
            _optimizer.step()
            _optimizer.zero_grad(set_to_none=True)
            _model.eval()

            ce_loss_value = float(ce_loss.detach().cpu().item())
            objective_value = float(objective.detach().cpu().item())
            grad_norm_value = float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            updated = True
        else:
            _model.eval()

        if _train_state is None:
            raise RuntimeError("Training state failed to initialize")
        _train_state["total_feedback"] = int(_train_state.get("total_feedback", 0)) + 1
        _train_state["last_feedback_at"] = time.time()
        _train_state["last_feedback_score"] = score
        _train_state["last_reward"] = reward
        _train_state["last_ce_loss"] = ce_loss_value
        _train_state["last_objective"] = objective_value
        _train_state["last_grad_norm"] = grad_norm_value
        _train_state["last_note"] = note
        _train_state["last_session_id"] = session_id
        if updated:
            _train_state["train_steps"] = int(_train_state.get("train_steps", 0)) + 1
            _train_state["total_updates"] = int(_train_state.get("total_updates", 0)) + 1
            if TRAIN_SAVE_EVERY_STEPS <= 1 or (_train_state["train_steps"] % TRAIN_SAVE_EVERY_STEPS) == 0:
                _save_adapter_checkpoint()
            else:
                _save_train_state()
        else:
            _save_train_state()

        _append_feedback_log(
            {
                "created_at": time.time(),
                "profile_id": _get_active_profile_id(),
                "session_id": session_id,
                "model": SERVED_MODEL_NAME,
                "model_id": MODEL_ID,
                "score": score,
                "reward": reward,
                "updated": updated,
                "note": note,
                "assistant_response": assistant_text,
                "messages": prompt_messages,
                "response_tokens": response_token_count,
                "ce_loss": ce_loss_value,
                "objective": objective_value,
                "grad_norm": grad_norm_value,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "ok": True,
        "model": SERVED_MODEL_NAME,
        "score": score,
        "reward": reward,
        "updated": updated,
        "session_id": session_id,
        "response_tokens": response_token_count,
        "train_steps": _train_state["train_steps"],
        "total_feedback": _train_state["total_feedback"],
        "total_updates": _train_state["total_updates"],
        "ce_loss": ce_loss_value,
        "objective": objective_value,
        "grad_norm": grad_norm_value,
    }


@app.on_event("startup")
def startup_event() -> None:
    _load_model()
    _ensure_active_serving_adapter_path(force_refresh=False)


@app.get("/health")
def health() -> JSONResponse:
    active_profile_id = _get_active_profile_id()
    loaded = _model is not None and _tokenizer is not None and _optimizer is not None and _train_state is not None
    serving_adapter_path = _active_serving_adapter_path() if loaded else None
    detail = {
        "status": "ok" if loaded else "loading",
        "ok": loaded,
        "model": SERVED_MODEL_NAME,
        "model_id": MODEL_ID,
        "load_in_4bit": LOAD_IN_4BIT,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "train_max_response_tokens": TRAIN_MAX_RESPONSE_TOKENS,
        "loaded_at": _loaded_at,
        "active_profile_id": active_profile_id,
        "active_profile": _profile_summary(active_profile_id),
        "compute_dtype": str(_compute_dtype()).replace("torch.", ""),
        "adapter_dir": str(_active_adapter_dir()),
        "adapter_ready": (_active_adapter_dir() / "adapter_config.json").exists(),
        "serving_adapter_path": serving_adapter_path,
        "serving_via_sglang": SGLANG_ENABLED,
        "sglang_base_url": SGLANG_BASE_URL,
        "sglang": _sglang_health_snapshot() if SGLANG_ENABLED else {"ok": False, "detail": "disabled"},
        "trainable_params": _trainable_param_count,
        "training": dict(_train_state or _default_train_state()),
    }
    return JSONResponse(detail)


@app.get("/v1/profiles")
def list_profiles() -> JSONResponse:
    active_profile_id = _get_active_profile_id()
    profiles = [_profile_summary(profile_id) for profile_id in _iter_profile_ids()]
    return JSONResponse(
        {
            "ok": True,
            "active_profile_id": active_profile_id,
            "profiles": profiles,
        }
    )


@app.post("/v1/profiles")
def create_profile(body: ProfileCreateRequest) -> JSONResponse:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Profile name cannot be empty")

    with _model_lock:
        _ensure_profile_storage_initialized()
        base_slug = _slugify_profile_id(name)
        profile_id = base_slug
        suffix = 2
        while profile_id in _iter_profile_ids():
            profile_id = f"{base_slug}-{suffix}"
            suffix += 1
        _write_profile_meta(_default_profile_meta(profile_id, name))
        profile = _profile_summary(profile_id)
        profiles = [_profile_summary(existing_id) for existing_id in _iter_profile_ids()]

    return JSONResponse(
        {
            "ok": True,
            "created_profile_id": profile_id,
            "profile": profile,
            "active_profile_id": _get_active_profile_id(),
            "profiles": profiles,
        }
    )


@app.post("/v1/profiles/select")
def select_profile(body: ProfileSelectRequest) -> JSONResponse:
    profile_id = body.profile_id.strip()
    if not profile_id:
        raise HTTPException(status_code=400, detail="profile_id cannot be empty")

    with _model_lock:
        profile = _activate_profile(profile_id, load_model=True)
        profiles = [_profile_summary(existing_id) for existing_id in _iter_profile_ids()]

    return JSONResponse(
        {
            "ok": True,
            "active_profile_id": profile_id,
            "profile": profile,
            "profiles": profiles,
        }
    )


@app.get("/v1/models")
def models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": SERVED_MODEL_NAME,
                    "object": "model",
                    "owned_by": "local",
                    "created": int(_loaded_at or time.time()),
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]):
    if bool(body.get("stream", False)):
        return StreamingResponse(
            _stream_chat_completion(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return JSONResponse(_generate_chat_completion(body))


@app.post("/v1/feedback")
def feedback(body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(_train_on_feedback(body))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
