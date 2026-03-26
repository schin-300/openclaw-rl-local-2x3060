# OpenClaw OS Shell Spec

## Goal

Evolve the current website from a single-purpose chat app into a lightweight
`OpenClaw OS`:

- one persistent web shell
- multiple apps living inside that shell
- `OpenClaw Chat` as the flagship app
- the ability to leave chat, open another app, and come back without losing the
  session context

The first proof that the shell is real is simple:

1. close or switch away from the current chat view
2. open a different app such as `Calendar`
3. return to chat and find the same conversation state still intact

## Product Thesis

This should feel like an intentional agent workspace, not a fake desktop
novelty.

Preferred references:

- ChatGPT projects/workspace simplicity
- Arc / Linear style app-shell clarity
- calm productivity tool, not skeuomorphic Windows cosplay

Avoid:

- draggable fake windows everywhere
- cluttered desktop metaphors
- lots of chrome that steals focus from the active app

## Core User Story

Schin opens the site and lands inside `OpenClaw OS`.

- The shell remembers the last active app.
- `Chat` is one app, not the whole product.
- Schin can switch to `Calendar`, `Notes`, `Tasks`, or other apps from the same
  shell.
- Switching apps should be fast and preserve in-app state wherever reasonable.
- The RL chat remains special, but no longer traps the whole UI inside a single
  surface.

## Hard Requirements

1. The current `Qwen3.5-4B + SGLang` chat flow must remain functional inside the
   new shell.
2. `OpenClaw Chat` must become an app module, not a separate website.
3. The shell must support opening another app such as `Calendar` without
   destroying current chat state.
4. The shell must support returning to `Chat` with the same active session and
   transcript state intact.
5. The shell must feel cohesive and minimal, not like tabs bolted onto the old
   chat page.
6. Notes/settings/profile flows must make sense in the OS model:
   either as compact utility surfaces inside the shell or as their own apps, but
   not as oversized awkward leftovers.
7. New apps must be registered through a single app registry instead of ad-hoc
   buttons and conditional DOM.
8. The architecture must allow adding future apps without rewriting the shell.

## First Prototype Scope

The first OS-shell prototype does not need a full ecosystem.

It should include:

- `OpenClaw Chat`
- one additional non-chat app: `Calendar`
- a shell-level launcher/navigation model
- persisted active-app state

It does not need, yet:

- drag-and-drop windows
- multi-window tiling
- desktop icons
- file manager complexity
- notifications system

## Information Architecture

### Shell

The shell owns:

- top bar or left rail app launcher
- active app container
- shell-level profile identity/status
- remembered active app
- shared layout and transitions

### Apps

Each app is a module mounted inside the shell.

Minimum app contract:

- `id`
- `name`
- `icon`
- `route` or shell key
- `component`
- `default_open`
- `persisted_state_key`

### Suggested Initial App Registry

- `chat`
- `calendar`
- `notes`
- `profiles`
- `tasks`
- `heretic-chat` (planned next phase)
- `model-control` (planned next phase)

Only `chat` and `calendar` must exist in the first OS-shell prototype. The rest
can remain planned entries.

## UX Direction

### Shell Layout

Preferred structure:

- compact top bar or left launcher rail
- strong active-app frame
- minimal global status row

The shell should answer:

- where am I
- which app is active
- how do I switch apps

without burying the main content area.

### Chat App

The current chat app should remain recognizably ChatGPT-like:

- session rail
- transcript
- composer
- streaming metrics
- thinking toggle

But it should now live inside the OS shell and coexist with other apps.

### Calendar App

The first non-chat app exists to prove the shell is real.

For v1 it can be simple:

- month or agenda view
- local-only mock or persisted events
- lightweight create/edit interaction

It does not need deep integrations at first.

## State Model

There are now two state layers.

### Shell State

- active app id
- launcher open/closed state if needed
- shell layout preferences
- last opened apps if multitasking is later added

### App State

Each app owns its own state namespace.

Examples:

- `chat`: active session, transcript, notes/profile utility state
- `calendar`: current month, selected day, local events

Shell state must never wipe app state unnecessarily when switching apps.

## Architecture Notes

1. The OS shell should be introduced as a frontend composition layer first.
2. The current backend contract for chat can remain mostly unchanged.
3. The shell should avoid coupling app-switching logic to RL/trainer concerns.
4. The app registry should become the single source of truth for available apps.
5. Utility panels that still conceptually belong to chat may remain chat-scoped
   during the transition, but the shell should not hardcode them forever.

## Implementation Plan

### Phase 1: Shell Foundation

1. Create a shell layout component around the current app body.
2. Add an app registry.
3. Move current chat UI into a `chat` app module.
4. Persist shell-level active app selection.

### Phase 2: First Additional App

1. Add a minimal `calendar` app.
2. Make app switching instant and stable.
3. Ensure chat state survives switching away and back.

### Phase 3: Cleanup

1. Rework notes/settings/profiles to fit the shell model.
2. Remove leftover single-app assumptions from the old layout.
3. Smooth transitions and status presentation.

### Phase 4: Heretic Sidecar Apps

After the OS shell is working, add two focused apps:

1. `Heretic Chat`
   A simple chat app dedicated to the specific
   `llmfan46/Qwen3.5-27B-heretic-v3-Q4_K_M.gguf` model path.
2. `Model Control`
   A minimal control app for switching the `2x3060` GPUs between:
   - the current `OpenClaw RL` runtime
   - the `Heretic Chat` runtime

These should be treated as shell-level product extensions, not ad-hoc side
scripts bolted onto the repo.

## Post-Shell Requirements

When Phase 4 starts:

1. `Heretic Chat` must be clearly separated from the trainable RL chat path.
2. `Model Control` must present a single obvious mode switch for GPU ownership.
3. The shell must make it clear which model lane is currently active.
4. GPU switching should prioritize clarity and safety over cleverness.

## Eval Checklist

1. Site loads into an OS shell instead of a standalone chat page.
2. `Chat` opens and works on the existing live backend.
3. `Calendar` opens from the shell.
4. Switching from `Chat` to `Calendar` does not lose current chat state.
5. Switching back to `Chat` restores the same active session and transcript.
6. Shell navigation is obvious and compact.
7. The interface feels like one coherent product, not stitched-together pages.
8. Future apps can be added by extending the app registry rather than rewriting
   shell logic.

## Non-Goals For The First OS Pass

- fake desktop wallpaper experience
- arbitrary resizable windows
- full OS parody
- replacing the chat backend
- deep calendar integrations

## Decision Rule

If a design choice makes the product feel more like a gimmick desktop than a
clean agent workspace, reject it.

The shell must increase capability without weakening the identity of the live
RL chat app.
