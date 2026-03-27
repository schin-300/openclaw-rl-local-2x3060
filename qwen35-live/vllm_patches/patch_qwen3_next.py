from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
)


def main() -> None:
    lines = TARGET.read_text().splitlines()
    start = None
    end = None

    for index, line in enumerate(lines):
        if line.startswith("    def _warmup_prefill_kernels("):
            start = index
            continue
        if start is not None and index > start and line.startswith("    def _forward_core("):
            end = index
            break

    if start is None or end is None:
        raise SystemExit("failed to patch qwen3_next.py warmup function")

    replacement = [
        "    def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:",
        '        if hasattr(self, "_prefill_kernels_warmed_up"):',
        "            return",
        "        self._prefill_kernels_warmed_up = True",
        "        logger.warning_once(",
        '            "OpenClaw patch: skipping Qwen3.5 GDN prefill warmup during startup; "',
        '            "the first prompt may compile kernels."',
        "        )",
        "        return",
        "",
    ]

    patched = lines[:start] + replacement + lines[end:]
    TARGET.write_text("\n".join(patched) + "\n")
    print("patched", TARGET)


if __name__ == "__main__":
    main()
