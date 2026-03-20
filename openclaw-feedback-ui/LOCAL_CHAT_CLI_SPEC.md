# Local Chat CLI Spec

## Goal

Provide a terminal-first CLI for the SGLang-backed local feedback stack so the same chat, session, feedback, and guidance workflow available in the ChatGPT-style UI can also be used from a shell.

## Constraints

- The CLI must use the existing `openclaw-feedback-ui` HTTP API.
- The CLI must not bypass the UI layer and talk directly to SGLang or the RL proxy.
- The CLI must preserve the same server-side browser session semantics by persisting the session cookie locally.
- The CLI must work against the stable local baseline at `http://127.0.0.1:30001`.
- The CLI must stay lightweight and dependency-compatible with the existing Python environment.

## In Scope

- `status`: show connected model, backend mode, proxy health, active profile, and active session
- `chat`: send one prompt and stream the assistant reply
- `shell`: interactive REPL on top of the same session cookie
- `reset`: create and switch to a new chat session
- `session list|select|new`: inspect saved chats and move between them
- `feedback`: score the most recent pending assistant reply or a specific assistant turn id
- `guidance show|set|clear`: inspect or update saved steering notes
- `thinking on|off`: toggle thinking mode through the UI API

## Out of Scope

- Starting or stopping systemd services
- Direct SGLang, vLLM, or `llama.cpp` transport logic
- Full TUI work
- Multi-user auth beyond the existing local cookie behavior

## UX Rules

- Default output should be plain text and shell-friendly.
- `chat` should stream assistant text to stdout as tokens arrive.
- Errors should print a single useful message to stderr and exit non-zero.
- `shell` should support slash commands for the same in-scope actions without leaving the REPL.

## Persistence

- Store the CLI cookie jar in `openclaw-feedback-ui/state/cli/cookies.txt`.
- If the server restarts and the cookie is no longer valid, the CLI should recover automatically on the next request.

## Acceptance Checks

1. `python openclaw-feedback-ui/cli.py status` reports the live `qwen3-0.6b-local` stack on `rl_proxy`.
2. `python openclaw-feedback-ui/cli.py chat "Reply with exactly: cli working."` prints `cli working.`.
3. `python openclaw-feedback-ui/cli.py reset` creates a new active session.
4. `python openclaw-feedback-ui/cli.py guidance set "Be concise."` persists guidance, and `guidance show` returns it.
5. `python openclaw-feedback-ui/cli.py thinking off` succeeds even when forced non-thinking mode is active.
