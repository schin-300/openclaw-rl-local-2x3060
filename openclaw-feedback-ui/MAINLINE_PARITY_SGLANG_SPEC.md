# Live RL Prototype Spec

## Product Goal

Ship one clean first prototype of the local live RL app:

- ChatGPT-style chat UI
- `Qwen3.5-4B`
- streaming replies
- thinking toggle
- compact notes and profile controls
- browser UI and CLI talking to the same backend

## Runtime Goal

The target runtime is `Qwen3.5-4B` served through `SGLang`.

Hard rule:

- Never claim the app is using `SGLang` unless the live backend health check
  confirms it.

## UI Requirements

- The app is one focused chat surface, not a fake desktop.
- Chats stay in the left rail, transcript in the center, composer at the
  bottom.
- Notes and profile controls open as compact integrated panels instead of
  separate fake apps or bulky tabs.
- The header must show the real connected model and backend health.
- Thinking output must stay in the thinking area instead of polluting the main
  assistant answer.

## Runtime Requirements

- Active model must be `Qwen/Qwen3.5-4B`.
- Streaming must work end to end in the browser.
- Thinking toggle must persist and affect requests.
- Browser and CLI must hit the same local API path.
- Session state, notes, and profile state must persist across restart.

## Eval Checklist

The prototype only passes when all of these are true:

1. `/api/status` reports `qwen3.5-4b-local` and a healthy backend.
2. The UI streams a real reply in the browser.
3. Thinking toggle changes the request path and renders reasoning cleanly.
4. Notes and profiles are available without fake desktop chrome or wasted space.
5. The model label and backend status text are truthful.
6. The CLI can send a real prompt through the same app API.
7. Restarting the services preserves the working state.

## Current Status

As of `2026-03-22`, the live prototype is running on:

- `Qwen/Qwen3.5-4B`
- `SGLang` on `http://127.0.0.1:30101`
- trainer/control plane on `http://127.0.0.1:30100`
- ChatGPT-style UI on `http://127.0.0.1:30001`

Implementation notes:

- SGLang needed a local `qwen3_5.py` LoRA shape fix so the live adapter could
  load correctly for `Qwen3.5-4B`.
- `main` now vendors that working `qwen3_5.py` patch and the SGLang launcher
  syncs it into the active venv before startup, so the live path no longer
  depends on an undocumented manual edit.
- The UI now only requests separated reasoning from SGLang when native chat
  thinking is actually enabled.
- The streaming path has an empty-stream rescue that falls back to the
  non-stream call when SGLang finishes a turn without usable visible tokens.
- Thinking-mode streaming has a reasoning backfill pass so the Thinking panel
  still gets content when the streamed answer lands without the expected
  structured wrapper.
- The transcript no longer flashes the misleading “Final answer missing” copy
  while a thinking-mode turn is still streaming.

## Residual Rule

If the `SGLang` worker is still warming or restarting, do not count that as a
passing eval run. Wait for `/api/status` to show the chat proxy healthy before
judging browser behavior.
