## 27B Chat + Model Control Spec

Date: 2026-03-26
Status: Working prototype on the live site. The functional sidecar lane now runs through
`vLLM` with `groxaxo/Qwen3.5-27B-heretic-v3-autoround-w4a16` at `32768` context. The
critical no-think fix was aligning the mounted chat template with the model's official
Qwen3.5 template so disabled-thinking requests emit normal assistant `content` instead of
reasoning-only output. The sidecar app and model-control app are working, but the
`>20 tok/s` acceptance target is still open.

### Goal

Extend `OpenClaw OS` with two new apps:

1. `27B Chat`
   A simple sidecar chat app that always shows the actual active 27B model.
Preferred target:
`llmfan46/Qwen3.5-27B-heretic-v3-GGUF`
file:
`Qwen3.5-27B-heretic-v3-Q4_K_M.gguf`

2. `Model Control`
   A minimal control app that assigns the machine's `2x3060` GPUs to one of
   two modes:
   - `OpenClaw RL`
   - `27B Chat`

### Exact Runtime Targets

- `OpenClaw RL` mode
  - `qwen35-4b.service` active
  - `qwen35-sglang.service` active
  - `heretic-sglang.service` inactive

- `27B Chat` mode
  - `qwen35-4b.service` inactive
  - `qwen35-sglang.service` inactive
  - `heretic-sglang.service` active

### 27B Runtime

- Final runtime must be whichever path satisfies both:
  - truthful model/backend labeling in the UI
  - `>20 tok/s` on this machine before the feature is considered fully complete
- Exact preferred target:
  - model file: `Qwen3.5-27B-heretic-v3-Q4_K_M.gguf`
  - model repo: `llmfan46/Qwen3.5-27B-heretic-v3-GGUF`
- Current best working runtime on this box:
  - backend: `vLLM`
  - model repo: `groxaxo/Qwen3.5-27B-heretic-v3-autoround-w4a16`
  - quantization: `auto-round` / `W4A16`
  - context: `32768`
  - production topology: tensor parallel `2`, pipeline parallel `1`
  - no-think requests require the model's official empty `<think></think>` assistant prefill
  - steady-state browser-visible speed currently lands around `8` to `12 tok/s`
- If the current `vLLM` runtime fails the quality, stability, or startup bar,
  fallback runtimes may be used:
  - `llama.cpp`
  - `vLLM` with a different Heretic-family quant
- The control surface must always show the real active model identifier and
  backend instead of pretending the sidecar lane is something else.
- Sidecar lane target config:
  - GPUs: `0,1`
  - GPU memory utilization target: `0.9` to `0.92`
  - context target: `32768`
  - production UI reply cap: `128`
  - no-think template override: required
  - prompt cache: disabled via `--cache-ram 0`

### UX Rules

1. `27B Chat` must be clearly separate from the RL training chat.
2. `27B Chat` must not show RL feedback controls.
3. `Model Control` must expose only two obvious actions:
   `Use OpenClaw RL` and `Use 27B Chat`.
4. The active GPU mode must be visible in the shell header.
5. If a user tries to use the inactive lane, the UI must fail clearly:
   - OpenClaw Chat should say the machine is in `27B Chat` mode.
   - `27B Chat` should say the machine is in `OpenClaw RL` mode.
6. Mode switching should be sequential and truthful:
   stop the old lane, then start the new one, then report health.

### Backend Plan

Add UI backend support for:

- model control status
- model control switching
- sidecar backend health
- sidecar streaming chat
- clear inactive-lane errors

### Frontend Plan

Add two new apps to the shell registry:

- `heretic`
- `model-control`

`27B Chat` includes:
- one transcript
- one composer
- one model label
- one clear backend/mode status
- stop-generation support

`Model Control` includes:
- current mode summary
- health cards for both lanes
- one button to activate `OpenClaw RL`
- one button to activate `27B Chat`

### Persistence Rules

- `OpenClaw Chat` keeps existing persisted profile/session behavior.
- `27B Chat` may use a lightweight local/browser session model for v1.
- Selected GPU mode must survive reload by being derived from live service
  state, not guessed client-side.

### Acceptance

1. `OpenClaw OS` shows `Chat`, `27B Chat`, `Model Control`, `Calendar`,
   `Notes`, and `Profiles`.
2. `Model Control` truthfully reports which lane owns the GPUs.
3. Switching to `27B Chat` stops the OpenClaw RL services and starts the
   sidecar 27B service.
4. Switching back to `OpenClaw RL` stops the sidecar 27B service and restores the
   RL chat + trainer services.
5. `27B Chat` can send and receive a real streamed response from the active
   27B target, and the UI truthfully names that target.
6. `OpenClaw Chat` still works after switching back.
7. The 27B lane reaches `>20 tok/s` on the chosen production runtime.
8. The 27B lane exposes `32768` context on the chosen production runtime.
9. Playwright interactive passes for:
   - launcher navigation
   - model control mode switching
   - 27B Chat request
   - OpenClaw Chat request after switching back
   - truthful shell header mode display

### Measured Runtime Findings

Tested on this machine with `32768` context:

- `llmfan46/Qwen3.5-27B-heretic-v3-GGUF:Q4_K_M`
  - `llama.cpp`
  - about `13.3` to `13.5 tok/s`
- `llmfan46/Qwen3.5-27B-heretic-v3-GGUF:Q3_K_S`
  - `llama.cpp`
  - about `11.0 tok/s`
- `mradermacher/Qwen3.5-27B-heretic-v3-i1-GGUF:IQ1_M`
  - `llama.cpp`
  - about `13.5 tok/s`
- `mradermacher/Qwen3.5-27B-heretic-v3-i1-GGUF:IQ2_XS`
  - `llama.cpp`
  - about `14.1` to `15.8 tok/s` on direct API checks
  - about `18.7 tok/s` on the current live browser `Reply with exactly: hello.` smoke check
- `cyankiwi/Qwen3.5-27B-AWQ-4bit`
  - `vLLM`
  - failed to complete `32768` startup on this box
- `groxaxo/Qwen3.5-27B-heretic-v3-autoround-w4a16`
  - `vLLM`
  - direct API with mismatched custom template produced reasoning-only output and severe lag
  - after switching the mounted template to the model's official empty-think prefill,
    disabled-thinking requests emitted normal assistant `content`
  - cold first request after restart paid a one-time compile tax of about `78.6s` to first
    token because startup warmup is intentionally skipped by the local patch
  - immediate steady-state retest dropped to about `0.258s` first token and about `9.739s`
    total for a four-sentence moon prompt
- `groxaxo/Qwen3.5-27B-heretic-v3-autoround-w4a16`
  - `vLLM` with `pipeline_parallel_size=2`
  - failed startup on this box at `32768` context with a compile-time CUDA OOM on GPU 1

Current conclusion:

- `SGLang` does not currently support the exact Heretic GGUF on this machine.
- `vLLM` did not boot the tested `27B AWQ` fallback at `32768`.
- `vLLM` is now the current working path for the live 27B lane.
- The essential parser fix was template correctness, not extra UI post-processing:
  when thinking is disabled, Qwen3.5 expects an empty `<think></think>` assistant prefill.
- `pipeline_parallel_size=2` is not currently viable on this `2x3060` box at `32768`
  because it OOMs during startup compilation, so the stable live config remains tensor
  parallel `2`.
- The current stable production setting is:
  - model: `groxaxo/Qwen3.5-27B-heretic-v3-autoround-w4a16`
  - backend: `vLLM`
  - context: `32768`
  - topology: tensor parallel `2`
  - reasoning parser: `qwen3`
  - mounted template aligned to the model's official no-think assistant prefill
  - UI sampling follows the model's generation defaults: `temperature 0.6`, `top_p 0.95`,
    `top_k 20`, `presence_penalty 0.0`
  - cold first request after restart is still slow because the local patch skips the heavy
    startup warmup and pushes that compile cost onto the first real prompt
- The sidecar lane is functionally complete for v1 app behavior, but it is still not signed
  off against the `>20 tok/s` performance target.

### Decision Rule

If a choice would blur the boundary between the trainable RL lane and the
27B sidecar lane, reject it. Clarity matters more than clever reuse.
