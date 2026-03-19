from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
ADAPTER_DIR = STATE_DIR / "adapter"
OPTIMIZER_PATH = STATE_DIR / "optimizer.pt"
TRAIN_STATE_PATH = STATE_DIR / "train_state.json"
FEEDBACK_LOG_PATH = STATE_DIR / "feedback.jsonl"
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
    }


def _load_train_state() -> dict[str, Any]:
    try:
        raw = json.loads(TRAIN_STATE_PATH.read_text(encoding="utf-8"))
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
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_STATE_PATH.write_text(json.dumps(_train_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_feedback_log(entry: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
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


def _first_model_device() -> torch.device:
    if _model is None:
        raise RuntimeError("Model has not been loaded")
    return next(_model.parameters()).device


def _chat_template_kwargs() -> dict[str, Any]:
    return {
        "tokenize": True,
        "enable_thinking": False,
    }


def _tokenize_messages(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> torch.Tensor:
    ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        **_chat_template_kwargs(),
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


def _score_to_reward(score: int) -> float:
    return _EXPLICIT_FEEDBACK_REWARD_BY_SCORE.get(score, 0.0)


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
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _model.save_pretrained(ADAPTER_DIR)
    if _optimizer is not None:
        torch.save(_optimizer.state_dict(), OPTIMIZER_PATH)
    _save_train_state()


def _load_model() -> None:
    global _loaded_at, _model, _optimizer, _tokenizer, _train_state, _trainable_param_count

    if _model is not None and _tokenizer is not None and _optimizer is not None and _train_state is not None:
        return

    STATE_DIR.mkdir(parents=True, exist_ok=True)

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

    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if adapter_config_path.exists():
        _model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=True)
    else:
        _model = get_peft_model(base_model, _lora_config())

    _optimizer = _build_optimizer(_model)
    _train_state = _load_train_state()
    _trainable_param_count = sum(param.numel() for param in _model.parameters() if param.requires_grad)

    if OPTIMIZER_PATH.exists():
        optimizer_state = torch.load(OPTIMIZER_PATH, map_location="cpu")
        _optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(_first_model_device())

    _model.eval()
    _loaded_at = time.time()


def _build_inputs(messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    batch = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
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


def _generate_chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    stream = bool(body.get("stream", False))
    if stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this experimental server")

    messages = _normalize_messages(body.get("messages"))
    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
    max_tokens = max(1, min(max_tokens, 1024))
    temperature = float(body.get("temperature") if body.get("temperature") is not None else DEFAULT_TEMPERATURE)
    top_p = float(body.get("top_p") if body.get("top_p") is not None else DEFAULT_TOP_P)
    do_sample = temperature > 0.0

    with _model_lock:
        _load_model()
        _model.eval()
        inputs = _build_inputs(messages)
        prompt_len = int(inputs["input_ids"].shape[-1])
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": _tokenizer.pad_token_id or _tokenizer.eos_token_id,
            "eos_token_id": _tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.inference_mode():
            generated = _model.generate(**inputs, **generation_kwargs)

        new_tokens = generated[0][prompt_len:]
        raw_text = _tokenizer.decode(new_tokens, skip_special_tokens=False)
        text = _strip_thinking(raw_text)
        if not text:
            text = raw_text.strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completion_tokens = int(new_tokens.shape[0])
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        },
    }


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


@app.get("/health")
def health() -> JSONResponse:
    loaded = _model is not None and _tokenizer is not None and _optimizer is not None and _train_state is not None
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
        "compute_dtype": str(_compute_dtype()).replace("torch.", ""),
        "adapter_dir": str(ADAPTER_DIR),
        "adapter_ready": (ADAPTER_DIR / "adapter_config.json").exists(),
        "trainable_params": _trainable_param_count,
        "training": dict(_train_state or _default_train_state()),
    }
    return JSONResponse(detail)


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
def chat_completions(body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(_generate_chat_completion(body))


@app.post("/v1/feedback")
def feedback(body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(_train_on_feedback(body))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
