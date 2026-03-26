#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:30100"
DEFAULT_MODEL = "qwen3.5-4b-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small local CLI for the Qwen3.5 SGLang-backed stack.")
    parser.add_argument("prompt", nargs="?", help="Optional one-shot prompt.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the local Qwen3.5 trainer/control API.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to send in the OpenAI-compatible request.")
    parser.add_argument("--thinking", action="store_true", help="Enable visible reasoning output.")
    parser.add_argument("--system", default="", help="Optional persistent system prompt.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds.")
    return parser.parse_args()


def iter_sse_payloads(response: httpx.Response):
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def build_messages(history: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    clean_system = system_prompt.strip()
    if clean_system:
        messages.append({"role": "system", "content": clean_system})
    messages.extend(history)
    return messages


def stream_chat(
    *,
    client: httpx.Client,
    base_url: str,
    model: str,
    history: list[dict[str, str]],
    system_prompt: str,
    prompt: str,
    thinking_enabled: bool,
) -> tuple[str, str, dict[str, Any]]:
    messages = build_messages(history + [{"role": "user", "content": prompt}], system_prompt)
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "enable_thinking": thinking_enabled,
    }

    response_text = ""
    reasoning_text = ""
    latest_metrics: dict[str, Any] = {}
    reasoning_started = False
    answer_started = False

    with client.stream("POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for raw_payload in iter_sse_payloads(response):
            if raw_payload == "[DONE]":
                break
            payload = json.loads(raw_payload)
            if not isinstance(payload, dict):
                continue
            latest_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else latest_metrics
            choices = payload.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta") or {}
            reasoning_delta = str(delta.get("reasoning_content") or "")
            content_delta = str(delta.get("content") or "")

            if reasoning_delta:
                if not reasoning_started:
                    print("\n[thinking]")
                    reasoning_started = True
                print(reasoning_delta, end="", flush=True)
                reasoning_text += reasoning_delta

            if content_delta:
                if reasoning_started and not answer_started:
                    print("\n\n[answer]")
                answer_started = True
                print(content_delta, end="", flush=True)
                response_text += content_delta

    print()
    return response_text.strip(), reasoning_text.strip(), latest_metrics


def interactive_loop(args: argparse.Namespace) -> int:
    history: list[dict[str, str]] = []
    system_prompt = args.system
    thinking_enabled = bool(args.thinking)

    timeout = httpx.Timeout(args.timeout, read=args.timeout)
    with httpx.Client(timeout=timeout) as client:
        print("Qwen3.5 local CLI")
        print("Commands: /new, /thinking on|off, /system <text>, /quit")
        while True:
            try:
                prompt = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not prompt:
                continue
            if prompt in {"/quit", "/exit"}:
                return 0
            if prompt == "/new":
                history = []
                print("Started a new chat.")
                continue
            if prompt.startswith("/thinking "):
                thinking_enabled = prompt.split(" ", 1)[1].strip().lower() in {"1", "true", "yes", "on"}
                print(f"Thinking {'on' if thinking_enabled else 'off'}.")
                continue
            if prompt.startswith("/system"):
                system_prompt = prompt[len("/system") :].strip()
                print("System prompt updated." if system_prompt else "System prompt cleared.")
                continue

            try:
                answer, _reasoning, metrics = stream_chat(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    history=history,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    thinking_enabled=thinking_enabled,
                )
            except httpx.HTTPError as exc:
                print(f"\n[error] {exc}")
                continue

            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": answer})
            if metrics:
                tok_s = metrics.get("tokens_per_second")
                generated = metrics.get("generated_tokens")
                if tok_s is not None:
                    print(f"[metrics] {tok_s} tok/s, generated={generated}")


def main() -> int:
    args = parse_args()
    timeout = httpx.Timeout(args.timeout, read=args.timeout)
    if args.prompt:
        with httpx.Client(timeout=timeout) as client:
            try:
                _answer, _reasoning, metrics = stream_chat(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    history=[],
                    system_prompt=args.system,
                    prompt=args.prompt,
                    thinking_enabled=bool(args.thinking),
                )
            except httpx.HTTPError as exc:
                print(f"[error] {exc}", file=sys.stderr)
                return 1
            if metrics:
                tok_s = metrics.get("tokens_per_second")
                generated = metrics.get("generated_tokens")
                if tok_s is not None:
                    print(f"[metrics] {tok_s} tok/s, generated={generated}", file=sys.stderr)
        return 0

    return interactive_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
