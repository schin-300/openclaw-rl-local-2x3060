# Mainline Parity + SGLang Spec

## Goal

Bring the ChatGPT-parity UI experiment onto the stable `main` branch while keeping
`main` on the existing OpenClaw-RL runtime path that already uses SGLang for
serving.

## Why

- The parity branch contains the stronger chat-first layout and parity docs.
- The stable `main` branch is the better landing zone for daily use.
- The stable OpenClaw-RL path already speaks to SGLang-backed services.
- Replacing the experimental Qwen3.5 trainer server with SGLang in the same
  change would add unnecessary risk and mix two different migrations.

## In Scope

- Merge the ChatGPT-parity UI assets and docs into `main`.
- Preserve the stable `main` behavior around sessions, thinking toggle, and
  inline feedback.
- Keep the default `main` runtime model on the existing `rl_proxy` /
  SGLang-backed flow.
- Keep the Qwen3.5 experimental server improvements that are safe to merge as
  part of the branch history.

## Out of Scope

- Rewriting the Qwen3.5 experimental trainer server to use SGLang internally.
- Switching the stable stack to vLLM or llama.cpp.
- Changing the live service topology beyond what `main` already uses.

## Acceptance Criteria

- `main` contains the parity branch UI layout and supporting docs.
- The stable `main` UI still works against the SGLang-backed OpenClaw-RL stack.
- No existing profile/session/thinking-toggle behavior on `main` regresses.
- Qwen3.5 experimental files still load cleanly after the merge.

## Runtime Decision

Use SGLang as the mainline serving path because OpenClaw-RL already integrates
with it directly. Keep vLLM and llama.cpp as sidecar runtime options outside the
mainline UI merge.
