#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from http.cookiejar import LWPCookieJar
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


APP_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = os.environ.get("OPENCLAW_RL_UI_URL", "http://127.0.0.1:30001").rstrip("/")
DEFAULT_COOKIE_FILE = Path(
    os.environ.get("OPENCLAW_RL_UI_CLI_COOKIE_FILE", APP_DIR / "state" / "cli" / "cookies.txt")
).expanduser()


class CliError(RuntimeError):
    pass


class LocalChatClient:
    def __init__(self, *, base_url: str, cookie_file: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookie_file = cookie_file
        self.cookie_jar = LWPCookieJar(str(self.cookie_file))
        if self.cookie_file.exists():
            try:
                self.cookie_jar.load(ignore_discard=True, ignore_expires=True)
            except Exception:
                pass
        self.opener = urllib_request.build_opener(urllib_request.HTTPCookieProcessor(self.cookie_jar))

    def close(self) -> None:
        self._save_cookies()

    def _save_cookies(self) -> None:
        self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
        self.cookie_jar.save(ignore_discard=True, ignore_expires=True)

    def _build_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> urllib_request.Request:
        headers = {
            "Accept": accept,
            "User-Agent": "openclaw-local-cli/1.0",
        }
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib_request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

    def _open(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = 60.0,
        accept: str = "application/json",
    ):
        request = self._build_request(method, path, json_body=json_body, accept=accept)
        try:
            response = self.opener.open(request, timeout=timeout)
        except urllib_error.HTTPError as exc:
            self._save_cookies()
            raise CliError(self._error_detail(exc)) from exc
        except urllib_error.URLError as exc:
            reason = exc.reason if getattr(exc, "reason", None) else exc
            raise CliError(str(reason)) from exc
        self._save_cookies()
        return response

    @staticmethod
    def _error_detail(response) -> str:
        try:
            body = response.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, str) and detail.strip():
                    return detail.strip()
            clean_body = body.strip()
            if clean_body:
                return clean_body
        code = getattr(response, "code", None) or getattr(response, "status", None)
        return f"Request failed with {code or 'unknown status'}"

    @staticmethod
    def _read_json_response(response) -> dict[str, Any]:
        with response:
            raw = response.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CliError(f"Response did not contain valid JSON: {exc}") from exc

    def get_status(self) -> dict[str, Any]:
        return self._read_json_response(self._open("GET", "/api/status"))

    def get_state(self) -> dict[str, Any]:
        return self._read_json_response(self._open("GET", "/api/state"))

    def reset(self) -> dict[str, Any]:
        return self._read_json_response(self._open("POST", "/api/reset"))

    def create_session(self) -> dict[str, Any]:
        return self._read_json_response(self._open("POST", "/api/sessions"))

    def select_session(self, session_id: str) -> dict[str, Any]:
        return self._read_json_response(
            self._open("POST", "/api/sessions/select", json_body={"session_id": session_id})
        )

    def set_guidance(self, text: str) -> dict[str, Any]:
        return self._read_json_response(self._open("POST", "/api/guidance", json_body={"text": text}))

    def set_system_prompt(self, text: str) -> dict[str, Any]:
        return self._read_json_response(self._open("POST", "/api/system-prompt", json_body={"text": text}))

    def set_thinking(self, enabled: bool) -> dict[str, Any]:
        return self._read_json_response(self._open("POST", "/api/thinking", json_body={"enabled": enabled}))

    def send_feedback(
        self,
        *,
        score: int,
        note: str = "",
        assistant_turn_id: str = "",
    ) -> dict[str, Any]:
        return self._read_json_response(
            self._open(
                "POST",
                "/api/feedback",
                json_body={
                    "score": score,
                    "note": note,
                    "assistant_turn_id": assistant_turn_id,
                },
            )
        )

    def stream_chat(self, prompt: str, *, thinking_enabled: bool | None = None) -> dict[str, Any] | None:
        if not prompt.strip():
            raise CliError("Prompt cannot be empty")

        payload: dict[str, Any] = {"prompt": prompt}
        if thinking_enabled is not None:
            payload["thinking_enabled"] = bool(thinking_enabled)

        response = self._open(
            "POST",
            "/api/chat/stream",
            json_body=payload,
            timeout=None,
            accept="text/event-stream",
        )
        try:
            collected_text = ""
            collected_reasoning = ""
            printed_text = False
            final_state: dict[str, Any] | None = None
            event_name = "message"
            data_lines: list[str] = []

            def flush_event() -> None:
                nonlocal event_name, data_lines, collected_text, collected_reasoning, printed_text, final_state
                if not data_lines:
                    event_name = "message"
                    return
                raw_payload = "\n".join(data_lines)
                data_lines = []
                current_event = event_name
                event_name = "message"
                if raw_payload == "[DONE]":
                    return
                try:
                    parsed = json.loads(raw_payload)
                except json.JSONDecodeError as exc:
                    raise CliError(f"Streaming payload contained invalid JSON: {exc}") from exc

                if current_event == "delta":
                    delta = str(parsed.get("delta") or "")
                    reasoning_delta = str(parsed.get("reasoning_delta") or "")
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        printed_text = True
                        collected_text += delta
                    if reasoning_delta:
                        collected_reasoning += reasoning_delta
                    return

                if current_event == "state":
                    candidate = parsed.get("state")
                    final_state = candidate if isinstance(candidate, dict) else None
                    return

                if current_event == "error":
                    detail = parsed.get("detail") if isinstance(parsed, dict) else None
                    raise CliError(str(detail or "Streaming failed"))

            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line == "":
                        flush_event()
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip() or "message"
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                flush_event()

            if not printed_text and collected_reasoning:
                sys.stdout.write(collected_reasoning)
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._save_cookies()
            return final_state
        finally:
            self._save_cookies()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for the OpenClaw local feedback UI")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"UI base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument(
        "--cookie-file",
        default=str(DEFAULT_COOKIE_FILE),
        help=f"Cookie file path (default: {DEFAULT_COOKIE_FILE})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show local stack status")
    status_parser.add_argument("--json", action="store_true", help="Print raw JSON")

    chat_parser = subparsers.add_parser("chat", help="Send one prompt and stream the reply")
    chat_parser.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin when omitted.")
    thinking_group = chat_parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--thinking-on", action="store_true", help="Request thinking mode for this turn")
    thinking_group.add_argument("--thinking-off", action="store_true", help="Force non-thinking mode for this turn")

    subparsers.add_parser("shell", help="Run an interactive chat shell")
    subparsers.add_parser("reset", help="Create and switch to a new chat session")

    session_parser = subparsers.add_parser("session", help="Inspect or switch saved chat sessions")
    session_subparsers = session_parser.add_subparsers(dest="session_command", required=True)
    session_subparsers.add_parser("list", help="List saved sessions for the active profile")
    session_new_parser = session_subparsers.add_parser("new", help="Create and switch to a new saved session")
    session_new_parser.set_defaults(session_new=True)
    session_select_parser = session_subparsers.add_parser("select", help="Switch to an existing session")
    session_select_parser.add_argument("session_id", help="Session id to activate")

    feedback_parser = subparsers.add_parser("feedback", help="Score the latest pending assistant reply")
    feedback_parser.add_argument("score", type=int, help="Feedback score from 1 to 10")
    feedback_parser.add_argument("note", nargs="*", help="Optional note to attach")
    feedback_parser.add_argument("--assistant-turn-id", default="", help="Specific assistant turn id")

    guidance_parser = subparsers.add_parser("guidance", help="Show or update saved guidance")
    guidance_subparsers = guidance_parser.add_subparsers(dest="guidance_command", required=True)
    guidance_subparsers.add_parser("show", help="Show saved guidance")
    guidance_set_parser = guidance_subparsers.add_parser("set", help="Replace saved guidance")
    guidance_set_parser.add_argument("text", nargs="+", help="Guidance text")
    guidance_subparsers.add_parser("clear", help="Clear saved guidance")

    system_prompt_parser = subparsers.add_parser("system-prompt", help="Show or update the profile system prompt")
    system_prompt_subparsers = system_prompt_parser.add_subparsers(dest="system_prompt_command", required=True)
    system_prompt_subparsers.add_parser("show", help="Show the saved system prompt")
    system_prompt_set_parser = system_prompt_subparsers.add_parser("set", help="Replace the saved system prompt")
    system_prompt_set_parser.add_argument("text", nargs="+", help="System prompt text")
    system_prompt_subparsers.add_parser("clear", help="Clear the saved system prompt")

    thinking_parser = subparsers.add_parser("thinking", help="Toggle stored thinking mode")
    thinking_parser.add_argument("mode", choices=("on", "off"), help="Stored thinking mode")

    return parser


def resolve_prompt(prompt_parts: list[str]) -> str:
    if prompt_parts:
        return " ".join(prompt_parts).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise CliError("Provide a prompt argument or pipe prompt text on stdin")


def active_session_label(state: dict[str, Any]) -> str:
    active_session_id = str(state.get("active_session_id") or "")
    for session in state.get("sessions") or []:
        if str(session.get("id") or "") == active_session_id:
            title = str(session.get("title") or "New chat").strip() or "New chat"
            return f"{title} ({active_session_id})"
    return active_session_id or "(none)"


def format_status(payload: dict[str, Any]) -> str:
    state = payload.get("state") or {}
    proxy = state.get("proxy") or {}
    router = proxy.get("router") or {}
    healthy = router.get("healthy_count", 0)
    workers = router.get("worker_count", 0)
    reason = proxy.get("reason") or ("ready" if proxy.get("ok") else "unknown")
    has_system_prompt = bool(str(state.get("system_prompt_text") or "").strip())
    lines = [
        f"Model: {state.get('model') or '(unknown)'}",
        f"Backend: {state.get('backend_mode') or '(unknown)'}",
        f"Profile: {state.get('active_profile_id') or '(none)'}",
        f"Session: {active_session_label(state)}",
        f"System prompt: {'set' if has_system_prompt else 'empty'}",
        f"Thinking: {'on' if state.get('thinking_enabled') else 'off'}",
        f"Proxy: {'ready' if proxy.get('ok') else 'not ready'} ({reason})",
        f"Workers: {healthy}/{workers} healthy",
    ]
    return "\n".join(lines)


def guidance_text_from_state(payload: dict[str, Any]) -> str:
    state = payload.get("state") or {}
    return str(state.get("guidance_text") or "").strip()


def system_prompt_text_from_state(payload: dict[str, Any]) -> str:
    state = payload.get("state") or {}
    return str(state.get("system_prompt_text") or "").strip()


def run_status(client: LocalChatClient, args: argparse.Namespace) -> int:
    payload = client.get_status()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(format_status(payload))
    return 0


def run_chat(client: LocalChatClient, args: argparse.Namespace) -> int:
    prompt = resolve_prompt(args.prompt)
    thinking_enabled = None
    if args.thinking_on:
        thinking_enabled = True
    if args.thinking_off:
        thinking_enabled = False
    client.stream_chat(prompt, thinking_enabled=thinking_enabled)
    return 0


def run_reset(client: LocalChatClient) -> int:
    payload = client.reset()
    state = payload.get("state") or {}
    print(f"New session: {active_session_label(state)}")
    return 0


def run_feedback(client: LocalChatClient, args: argparse.Namespace) -> int:
    note = " ".join(args.note).strip()
    payload = client.send_feedback(
        score=args.score,
        note=note,
        assistant_turn_id=args.assistant_turn_id.strip(),
    )
    state = payload.get("state") or {}
    print(f"Saved feedback {args.score}/10 for session {active_session_label(state)}")
    return 0


def run_guidance(client: LocalChatClient, args: argparse.Namespace) -> int:
    if args.guidance_command == "show":
        payload = client.get_state()
        text = guidance_text_from_state(payload)
        print(text if text else "(empty)")
        return 0
    if args.guidance_command == "set":
        text = " ".join(args.text).strip()
        client.set_guidance(text)
        print("Saved guidance.")
        return 0
    if args.guidance_command == "clear":
        client.set_guidance("")
        print("Cleared guidance.")
        return 0
    raise CliError(f"Unsupported guidance command: {args.guidance_command}")


def run_system_prompt(client: LocalChatClient, args: argparse.Namespace) -> int:
    if args.system_prompt_command == "show":
        payload = client.get_state()
        text = system_prompt_text_from_state(payload)
        print(text if text else "(empty)")
        return 0
    if args.system_prompt_command == "set":
        text = " ".join(args.text).strip()
        client.set_system_prompt(text)
        print("Saved system prompt.")
        return 0
    if args.system_prompt_command == "clear":
        client.set_system_prompt("")
        print("Cleared system prompt.")
        return 0
    raise CliError(f"Unsupported system prompt command: {args.system_prompt_command}")


def run_thinking(client: LocalChatClient, args: argparse.Namespace) -> int:
    enabled = args.mode == "on"
    payload = client.set_thinking(enabled)
    state = payload.get("state") or {}
    print(f"Thinking is {'on' if state.get('thinking_enabled') else 'off'}.")
    return 0


def run_session(client: LocalChatClient, args: argparse.Namespace) -> int:
    if args.session_command == "list":
        payload = client.get_state()
        state = payload.get("state") or {}
        active_session_id = str(state.get("active_session_id") or "")
        sessions = state.get("sessions") or []
        if not sessions:
            print("(no saved sessions)")
            return 0
        for session in sessions:
            session_id = str(session.get("id") or "")
            marker = "*" if session_id == active_session_id else " "
            title = str(session.get("title") or "New chat").strip() or "New chat"
            preview = str(session.get("preview") or "").strip()
            print(f"{marker} {session_id}  {title}")
            if preview:
                print(f"  {preview}")
        return 0
    if args.session_command == "new":
        payload = client.create_session()
        state = payload.get("state") or {}
        print(f"New session: {active_session_label(state)}")
        return 0
    if args.session_command == "select":
        payload = client.select_session(args.session_id)
        state = payload.get("state") or {}
        print(f"Active session: {active_session_label(state)}")
        return 0
    raise CliError(f"Unsupported session command: {args.session_command}")


def print_shell_help() -> None:
    print(
        "Slash commands: /help, /status, /reset, /sessions, /use <session-id>, /guidance show, "
        "/guidance set <text>, /guidance clear, /system show, /system set <text>, /system clear, "
        "/thinking on|off, /feedback <1-10> [note], /exit"
    )


def run_shell(client: LocalChatClient) -> int:
    print("OpenClaw local shell. Type a prompt to chat, or /help for commands.")
    while True:
        try:
            line = input("openclaw> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not line:
            continue
        if not line.startswith("/"):
            try:
                client.stream_chat(line)
            except CliError as exc:
                print(f"Error: {exc}", file=sys.stderr)
            continue

        try:
            parts = shlex.split(line[1:])
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        if not parts:
            continue

        command = parts[0]
        try:
            if command in {"exit", "quit"}:
                return 0
            if command == "help":
                print_shell_help()
                continue
            if command == "status":
                print(format_status(client.get_status()))
                continue
            if command in {"new", "reset"}:
                run_reset(client)
                continue
            if command == "sessions":
                payload = client.get_state()
                state = payload.get("state") or {}
                active_session_id = str(state.get("active_session_id") or "")
                sessions = state.get("sessions") or []
                if not sessions:
                    print("(no saved sessions)")
                    continue
                for session in sessions:
                    session_id = str(session.get("id") or "")
                    marker = "*" if session_id == active_session_id else " "
                    title = str(session.get("title") or "New chat").strip() or "New chat"
                    print(f"{marker} {session_id}  {title}")
                continue
            if command == "use":
                if len(parts) < 2:
                    raise CliError("Use /use <session-id>")
                payload = client.select_session(parts[1])
                state = payload.get("state") or {}
                print(f"Active session: {active_session_label(state)}")
                continue
            if command == "guidance":
                if len(parts) == 1 or parts[1] == "show":
                    payload = client.get_state()
                    text = guidance_text_from_state(payload)
                    print(text if text else "(empty)")
                    continue
                if parts[1] == "clear":
                    client.set_guidance("")
                    print("Cleared guidance.")
                    continue
                if parts[1] == "set":
                    text = " ".join(parts[2:]).strip()
                    if not text:
                        raise CliError("Provide guidance text after /guidance set")
                    client.set_guidance(text)
                    print("Saved guidance.")
                    continue
                raise CliError("Use /guidance show, /guidance set <text>, or /guidance clear")
            if command in {"system", "system-prompt"}:
                if len(parts) == 1 or parts[1] == "show":
                    payload = client.get_state()
                    text = system_prompt_text_from_state(payload)
                    print(text if text else "(empty)")
                    continue
                if parts[1] == "clear":
                    client.set_system_prompt("")
                    print("Cleared system prompt.")
                    continue
                if parts[1] == "set":
                    text = " ".join(parts[2:]).strip()
                    if not text:
                        raise CliError("Provide system prompt text after /system set")
                    client.set_system_prompt(text)
                    print("Saved system prompt.")
                    continue
                raise CliError("Use /system show, /system set <text>, or /system clear")
            if command == "thinking":
                if len(parts) < 2 or parts[1] not in {"on", "off"}:
                    raise CliError("Use /thinking on or /thinking off")
                payload = client.set_thinking(parts[1] == "on")
                state = payload.get("state") or {}
                print(f"Thinking is {'on' if state.get('thinking_enabled') else 'off'}.")
                continue
            if command == "feedback":
                if len(parts) < 2:
                    raise CliError("Use /feedback <1-10> [note]")
                score = int(parts[1])
                note = " ".join(parts[2:]).strip()
                client.send_feedback(score=score, note=note)
                print(f"Saved feedback {score}/10.")
                continue
            raise CliError("Unknown slash command. Use /help for the list.")
        except (CliError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    client = LocalChatClient(base_url=args.base_url, cookie_file=Path(args.cookie_file).expanduser())
    try:
        if args.command == "status":
            return run_status(client, args)
        if args.command == "chat":
            return run_chat(client, args)
        if args.command == "shell":
            return run_shell(client)
        if args.command == "reset":
            return run_reset(client)
        if args.command == "feedback":
            return run_feedback(client, args)
        if args.command == "guidance":
            return run_guidance(client, args)
        if args.command == "system-prompt":
            return run_system_prompt(client, args)
        if args.command == "thinking":
            return run_thinking(client, args)
        if args.command == "session":
            return run_session(client, args)
        raise CliError(f"Unsupported command: {args.command}")
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
