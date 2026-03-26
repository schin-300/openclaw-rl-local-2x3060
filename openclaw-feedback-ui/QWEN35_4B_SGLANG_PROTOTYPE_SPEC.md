# Qwen3.5-4B + SGLang Prototype Spec

## Goal

Ship the first real prototype of the live RL chat app with:

- `Qwen3.5-4B`
- `SGLang`
- ChatGPT-style UI
- real streaming
- thinking toggle
- seamless notes/settings
- persistent chats and profiles
- live feedback training flow

## Product Shape

One main chat product, not a fake desktop.

- left rail for chat history
- central transcript
- compact composer
- compact model/status/thinking controls
- notes and settings integrated into the same app surface

## Hard Requirements

1. The app must not silently run `qwen3-0.6b`.
2. The UI must truthfully show the connected model and backend.
3. Streaming must be real token streaming, not delayed full-response dumps.
4. Thinking toggle must actually change request behavior/state.
5. Notes and settings must be compact and not feel like separate bulky windows.
6. Browser and CLI must hit the same backend path.

## Status Update

As of `2026-03-22`, the requested target is live and verified locally:

1. `Qwen/Qwen3.5-4B` is the active model.
2. Chat traffic is served through `SGLang`.
3. The training/control plane still runs through `qwen35-experimental/server.py`
   and syncs the serving LoRA adapter into the SGLang worker.
4. The ChatGPT-style UI on `30001` talks to the same local app contract for
   chat, streaming, health, profiles, and feedback.

## Implementation Notes

1. The local SGLang install needed a `qwen3_5` LoRA hidden-dimension fix in the
   model implementation so the Qwen3.5-4B adapter could load without crashing.
   `main` now carries that patched file in `qwen35-experimental/sglang_patches/`
   and the SGLang launcher syncs it into the active venv on startup.
2. The UI now avoids requesting separated reasoning from SGLang unless native
   SGLang thinking is intentionally enabled.
3. The stream adapter has a non-stream rescue path for rare cases where the
   stream completes without a usable visible answer.
4. Thinking-mode streaming has a reasoning backfill path so the final turn can
   still populate the Thinking panel when the streamed output omits the expected
   wrapper.
5. The Notes and Profiles controls stay as compact integrated panels inside the
   same chat surface.

## Remaining Operational Note

The SGLang worker still needs a brief warm-up window after restart. Browser
evaluation should only begin once `/api/status` reports the chat proxy healthy.

## UI Work That Still Applies Regardless

Once the backend gate is resolved:

1. remove the fake multi-window desktop feel
2. keep chat as the dominant surface
3. fold notes/settings into compact integrated panels
4. keep streaming metrics and feedback inline
5. keep the thinking toggle near the composer/header

## Eval Checklist

1. `/api/status` reports the correct model and backend.
2. Browser chat streams a real reply.
3. Thinking toggle changes the request/state correctly.
4. Feedback submission succeeds and updates the training state.
5. Notes/settings are compact and usable without leaving the main chat flow.
6. CLI and browser both hit the same live backend.
7. After service restart, the app returns in the same correct state.

## Decision Rule

Do not silently compromise the requested target.

If the live worker falls out of health, restore it and re-evaluate against the
healthy `Qwen3.5-4B + SGLang` path instead of masking the failure with another
runtime.
