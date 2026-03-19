# Frutiger Aero UI Spec

## Project

Redesign the local feedback UI into a **Frutiger Aero desktop-chat experience** for the Qwen local trainer stack.

The target is not a cosmetic recolor. It should feel like:

- a late-2000s glossy desktop environment
- a real local chat application running inside that environment
- a clearer, friendlier version of the current training workflow

## Product Goal

Make the local trainer feel inviting and legible enough to use for long feedback sessions, while preserving the current mechanics:

- prompt the local model
- read the answer
- optionally mark any assistant reply with `like` or `dislike`
- optionally leave a short note with that feedback
- save persistent steering notes and scenario guidance
- monitor whether the backend is connected and actively learning

The new UI should read as **"ChatGPT, but as a Frutiger Aero local app."**

## Critical Corrections

The following requirements were added after live browser review and should be treated as binding:

- The desktop shell must resize cleanly to the browser window without awkward extra page scroll or oversized blank space.
- The main desktop should stay visually simple: wallpaper, icons, app windows, and a taskbar.
- Do **not** use passive “information boxes” that look like clickable controls.
- If the interface needs explanatory or reference text, put it inside a dedicated app such as `notes.txt`, not as repeated floating cards.
- Keep Playwright interactive working for iterative UI QA.
- The chat app should be easier to read than the current compact layout; it can take over most of the desktop when opened.
- Prioritize a stable `1920x1080` experience over ambitious resize logic.
- Do not spend complexity on freeform dragging or resizing until the fixed `1920x1080` app is fully solid.
- The fake OS should keep the desktop home screen and double-click shortcuts, but once an app is opened the experience should behave like a single fixed tab surface, not overlapping draggable windows.
- The chat app should fill the usable desktop above the taskbar with a slim top bar and a ChatGPT-like main layout.
- The feedback flow must be inline under assistant messages. No dedicated feedback sidebar or dashboard card.
- Sending a new message must never be blocked by unrated older replies.
- If a reply is never rated, it should not create training feedback.

## Persistent Feature Requirements

These are important enough to preserve across context resets:

- Add a real **`notes.txt` app** for saved steering notes, scenarios, and other durable guidance.
- Add a real **settings app** for **multiple training profiles**.
- A profile must be creatable from the UI and saved persistently.
- The user must be able to switch between saved profiles later.
- Each profile should keep its own steering notes and training state.
- The UI should show the size of each profile so growth from training / notes is visible.
- The architecture should remain easy to extend with more desktop apps later.

## Approved Chat UX Revision

These requirements were explicitly confirmed after the first desktop-shell pass and should override older chat-layout decisions where they conflict.

### Core direction

- Keep the Frutiger Aero desktop home screen and double-click app launch.
- Keep the Vista-style taskbar at the bottom.
- Do **not** keep fake overlapping-window complexity inside the chat app.
- `local-chat.exe` should behave like a real full app that fills the usable desktop above the taskbar.
- The app may keep a slim Windows-app style title bar, but the rest of the screen should read primarily as chat.

### Chat layout

- The chat app should feel much closer to ChatGPT than to a dashboard.
- The chat app should use a **two-column layout**:
  - a left sidebar for conversations
  - a main conversation surface for the transcript and composer
- The left sidebar should contain:
  - a `Chats` heading
  - a `New chat` action at the top right of the sidebar header
  - a scrollable list of saved sessions
- Remove `notes.txt` and `settings.cpl` from the chat sidebar.
- Remove the active-profile section from the chat sidebar.
- `notes.txt` and `settings.cpl` should remain available as separate desktop/taskbar apps.

### Sessions

- Sessions must persist across refreshes and service restarts.
- Sessions must be linked to the currently selected training profile.
- Each profile should maintain its own independent chat/session list.
- Session titles should be generated from the first user prompt, using the first few words only.
- Do not add LLM-based conversation title generation.
- The session list must be scrollable.
- Selecting a different session should restore its transcript and pending feedback state.

### Composer

- The composer should be visually compact by default, closer to ChatGPT or iMessage than the current large panel.
- The composer should take up roughly `10%` of screen height or less in the normal idle state.
- The input should start as a small single-line control.
- The input should auto-grow vertically as the user adds more lines.
- Pressing `Enter` should send the message.
- Pressing `Shift+Enter` should create a new line.
- Remove the helper text that says `Ctrl/Cmd + Enter` to send.
- Replace the current large CTA with a compact real send button.

### Feedback / training controls

- Feedback should stay inline under assistant messages.
- Do not require feedback before the next prompt can be sent.
- Unrated replies should not create training data.
- Use a **single 1-10 rating system** instead of separate thumbs-up / thumbs-down controls.
- Each assistant message can include:
  - a compact 1-10 rating control
  - a small optional note field
  - a compact send/apply feedback action
- The feedback UI should stay lighter and less boxy than the current dashboard-like controls.

### Streaming telemetry

- Show `tok/s` for the currently streaming assistant response.
- Place the stream speed on or directly under the active assistant bubble.
- If practical, update the `tok/s` display live during generation instead of only after completion.

### Verification target

- Prioritize correct behavior at `1920x1080`.
- A fixed, polished `1920x1080` build is more important than generalized resize behavior for now.

## Future Experimental Branch

After the approved mainline plan above is complete and stable, a separate experimental branch should be created for a much stronger ChatGPT-parity pass.

### Branching rule

- Do not overwrite the known-good mainline build when starting the parity experiment.
- Create a dedicated experimental git branch from the verified mainline state first.
- Keep the verified mainline state easily reversible in git before the parity pass begins.

### Goal

- Redesign the UI to look **much closer to ChatGPT itself**, not just “ChatGPT-like inside Frutiger Aero”.
- Use the real ChatGPT web UI as the comparison target.
- Validate similarity using Playwright interactive, not by intuition alone.

### Required process

- Define explicit, reusable similarity metrics before judging the result.
- Use those same metrics as the north star during the entire experimental pass.
- Compare the live experimental app side-by-side with ChatGPT.
- Assign a numeric similarity score out of `10`.
- The target similarity score is at least `9/10`.

### Example metric categories

- overall page structure
- sidebar width and hierarchy
- top bar proportions
- transcript column width and alignment
- message spacing and grouping
- composer size, placement, and controls
- typography scale and weight
- surface contrast and background behavior
- icon/button placement and density
- scroll behavior and empty-state composition

## Streaming Design

This section captures the planned approach for adding **real streaming** to the current Qwen3.5 local stack before implementation work starts.

### Current state

- The current Qwen3.5 chat path is still request/response.
- The UI server currently sends chat requests with `stream: false`.
- The Qwen3.5 backend currently rejects `stream=true`.
- The browser currently waits for the full JSON response, then renders the assistant message all at once.

### Goal

Add **real token-by-token streaming** for the Qwen3.5 chat app while preserving:

- the existing profile system
- the existing inline per-message feedback flow
- the current browser-session transcript state on the UI server
- the rule that only rated replies create training feedback

### Chosen transport

Use **server-sent events over a POST request** to the UI server.

Reasoning:

- The browser needs to send a prompt body, so plain `EventSource` is not enough on its own.
- `fetch()` with a streamed response body is a good fit for the existing app.
- SSE keeps the payload simple and easy to debug in FastAPI and Playwright.
- WebSockets are not necessary for this feature.

### Chosen architecture

#### 1. Qwen3.5 backend: real incremental generation

Add real streaming to the Qwen3.5 backend in `qwen35-experimental/server.py`.

Planned approach:

- keep using the existing Hugging Face generation path
- switch the chat endpoint to support `stream=true`
- use `TextIteratorStreamer` or `AsyncTextIteratorStreamer`
- run `model.generate(...)` in a background thread while the streamer yields decoded text pieces
- return a `StreamingResponse` with `text/event-stream`

Expected behavior:

- emit incremental text deltas as the model generates
- emit a final completion event when generation finishes
- preserve the current non-streaming JSON response path for callers that do not request streaming

#### 2. UI server: streaming proxy plus transcript commit

Add a new streaming chat route in `openclaw-feedback-ui/app.py`.

Planned route:

- `POST /api/chat/stream`

Responsibilities:

- build the same prompt/context/guidance payload as the current `/api/chat`
- call the Qwen3.5 backend with `stream=true`
- read upstream SSE incrementally with `httpx.AsyncClient.stream(...)`
- accumulate the assistant text while forwarding deltas to the browser
- only commit the final assistant turn to the browser-session transcript after the stream finishes successfully

Why the UI server remains in the middle:

- it owns browser-session transcript state
- it owns guidance injection and context trimming
- it creates the feedback candidate used later by inline training feedback
- it keeps the browser API stable even if backend details change

#### 3. Browser client: incremental assistant rendering

Update `openclaw-feedback-ui/static/app.js` to stream into the visible transcript.

Planned browser behavior:

- send prompts with `fetch("/api/chat/stream", { method: "POST", body: ... })`
- read the response body as a stream
- parse SSE lines in JavaScript
- create a temporary assistant message immediately when streaming starts
- append incoming text deltas into that message as they arrive
- on final event, replace the temporary message with the committed server state

The UI should feel close to ChatGPT:

- user message appears immediately
- assistant bubble grows in place while text streams in
- inline feedback controls appear only after the assistant reply is fully finished

### Stream event contract

The UI server should emit normalized SSE events to the browser instead of passing raw upstream chunks through unchanged.

Preferred event types:

- `start`
  - contains request id and temporary assistant turn id
- `delta`
  - contains the next text fragment
- `final`
  - contains the final assistant text and any lightweight metadata
- `state`
  - contains the serialized committed state after transcript save
- `error`
  - contains a user-safe error message

This keeps the browser code simple and decouples it from backend chunk quirks.

### Thinking / hidden reasoning rule

Streaming must preserve the current **non-thinking** behavior.

Requirements:

- do not expose `<think>` blocks or hidden reasoning text in the live stream
- keep `enable_thinking=False`
- if needed, filter or buffer streamed chunks so internal thinking markers never appear in the UI

Because the current backend already strips thinking at the end, streaming needs an incremental-safe version of that same rule instead of relying only on final cleanup.

### Locking and concurrency

The current backend uses one model lock for generation and training.

Streaming should preserve that safety model:

- hold model generation ownership for the full life of the stream
- block profile switches and training updates while a reply is actively streaming
- allow new feedback only after the stream is fully finished and the final assistant turn is committed

This is acceptable for the current single-user local setup.

### Failure and cancellation behavior

Requirements:

- if generation errors mid-stream, the browser should receive an `error` event and the temporary assistant bubble should be marked as failed instead of committed
- if the browser disconnects, the UI server should stop reading upstream and clean up
- incomplete assistant output must **not** become a feedback candidate
- reset/profile-switch should continue to work after a failed or cancelled stream

### Feedback interaction rules

Streaming must not change the feedback semantics:

- only completed assistant replies get inline feedback controls
- partially streamed or failed replies are not trainable
- unrated completed replies remain allowed
- only rated replies are submitted to the trainer

### Recommended implementation order

1. Add streaming support to `qwen35-experimental/server.py` while preserving the old JSON path.
2. Add `/api/chat/stream` to `openclaw-feedback-ui/app.py` and commit transcript state only after stream completion.
3. Add streamed rendering to `openclaw-feedback-ui/static/app.js`.
4. Keep the existing `/api/chat` as a fallback until streaming is stable.
5. Verify in Playwright at `1920x1080`:
   - streamed text visibly grows
   - completed replies still get inline feedback controls
   - another prompt can be sent without rating older replies
   - profile switching still resets chat cleanly

### Non-goals for the first streaming pass

- no WebSocket migration
- no multi-user concurrency redesign
- no speculative decoding or advanced token batching
- no streaming of training updates themselves
- no streaming in the older RL proxy path unless handled as a separate follow-up

## Reference Findings

### Source references

- `https://frutiger-aero.neocities.org/`
- `https://frutiger-aero.org/`
- `https://frutigeraeroarchive.org/`
- `https://tsotchke.net/`
- User-provided screenshots from those references

### Common visual traits across references

- **Glass chrome:** translucent title bars, glossy pills, bright edge highlights
- **Nature-first backdrops:** grass, sky, water, aurora, dew, soft focus photography
- **Airy composition:** large breathing room, fewer dense clusters, generous panel padding
- **Rounded geometry:** pill buttons, soft cards, capsule nav, rounded window corners
- **Blue-green emphasis:** cyan, aqua, teal, leaf green, sky white, with occasional lime or sun-yellow accents
- **Dark glass accents:** black or charcoal glossy rails framing brighter content
- **Desktop metaphor:** windows, launchers, start bars, icons, “sites as software”
- **Friendly futurism:** polished and optimistic, not cyberpunk, not brutalist, not minimal-flat

### Traits to avoid

- matte dark SaaS panels
- flat monochrome iconography
- purple gradients
- cramped density
- stark terminal aesthetics
- generic “AI dashboard” styling

## Experience Direction

### Chosen concept

Build the interface as a **fake local OS desktop** with one primary executable window:

- wallpaper backdrop
- translucent Vista-style system/task bar
- desktop icons on the home screen
- a large central chat window that behaves like the real app
- separate utility apps instead of stuffing passive information into the chat window

This keeps the app whimsical without sacrificing usability.

### OS behavior requirement

- The **home screen is the first view**.
- The main chat app is **not open by default** on desktop.
- The user must **double-click** the chat app icon to open it.
- When opened, the chat app should feel like the primary application and may launch **maximized by default**.
- The app window can later be reopened from desktop or taskbar affordances.
- The structure should make it straightforward to add more launchable apps later.

### Why this concept

- It matches the strongest signals in the references.
- It gives the user the “local app” feeling they explicitly asked for.
- It still supports the ChatGPT-style mental model:
  - history rail
  - main conversation surface
  - bottom composer
  - side utility panel

## Information Architecture

### Global structure

1. **Desktop Shell**
   - wallpaper
   - soft light bloom overlays
   - bottom taskbar
   - desktop icons and home-screen guidance

2. **Primary App Window**
   - main chat experience
   - hidden until launched from the desktop icon

3. **Utility Surface**
   - trainer status
   - rating controls
   - steering notes / scenarios

### Desktop home-screen layout

1. **Wallpaper**
   - use the bright green grass wallpaper from `https://frutiger-aero.neocities.org/img/BG.png`
   - keep it as the main desktop backdrop

2. **Desktop icons**
   - one real launchable chat app icon
   - a real `notes.txt` app
   - a real settings/profile app
   - optional placeholder icons for future apps
   - icon layout should look native to the desktop, not like card buttons

3. **Taskbar**
   - glossy Vista-like start orb/button
   - task buttons for open apps
   - clock/status cluster

### Primary app window layout

1. **Slim glass title bar**
   - app name
   - local model label
   - connection state pill
   - active profile
   - close control
   - keep it sleek and small, like a browser or Windows app top bar

2. **Side rail**
   - keep only actionable items
   - “New chat”
   - launcher-style links for `notes.txt` and `settings.cpl`
   - should resemble a glossy sidebar or program launcher, not a modern flat nav

3. **Main transcript area**
   - large readable conversation stream
   - should dominate the app visually, closer to ChatGPT than a tiny widget
   - strong empty state
   - assistant messages in luminous glass cards
   - user messages in brighter aqua/sky-tinted cards

4. **Composer dock**
   - fixed at bottom of chat window
   - large text box
   - glossy send button
   - helper copy for `Ctrl/Cmd + Enter`

5. **Inline feedback controls**
   - appear under assistant messages only
   - `like` / `dislike` buttons first
   - small optional note field beside or below them
   - a send action that applies training for that specific assistant reply only
   - once feedback is sent, show a compact saved summary instead of the controls

### Utility apps

1. **`notes.txt`**
   - editable, saved, persistent
   - stores steering notes and made-up scenarios
   - tied to the currently active training profile

2. **Settings / Profiles**
   - list saved training profiles
   - create a new profile
   - switch active profile
   - show profile size and lightweight training stats

## Visual Spec

### Color system

Core colors should come from:

- sky blue
- aqua cyan
- teal
- fresh green
- frosted white
- charcoal glass

Accent colors:

- lime glow for healthy status
- warm sun-yellow for active call-to-action
- soft coral only for failure/unavailable states

### Materials

- frosted glass with subtle blur where supported
- glossy gradients with bright top edges
- inner highlights and soft outer shadows
- translucent dark underlays for contrast
- occasional reflective streaks or light sweeps

### Background treatment

Use a layered background:

- wallpaper image or wallpaper-like gradient field
- aurora light arcs
- bokeh or dew-like glow particles
- gentle vignette for readability

It should feel alive, but not noisy enough to fight the text.

### Typography

Use a humanist or neo-grotesque stack that feels closer to Vista-era UI than developer tooling.

Preferred direction:

- `"Segoe UI", "Trebuchet MS", "Tahoma", sans-serif`

Display copy can be slightly more dramatic, but body copy should stay highly legible.

### Iconography

Use simple glossy or soft system-like markers:

- orb
- signal/status dot
- folder/program chips
- stars/bubbles/light flares sparingly

If no icon set is added, build the look with shape, chrome, and text before inventing custom art.

## Interaction Spec

### Chat flow

- Double-clicking the chat desktop icon launches the window.
- The open/closed state of the chat app should be separate from the conversation state.
- The chat app should default to a large maximized presentation for readability.
- Restored windows should be draggable and, if practical, resizable inside the desktop.
- Sending a prompt should feel like launching a local request.
- Busy state should visually lock the composer and relevant controls.
- Awaiting-feedback state should clearly explain why the next prompt is blocked.
- Transcript should remain the visual center of gravity.

### Feedback flow

- The rating module should feel like a **tuning console** rather than a plain form.
- The score display should be large and prominent.
- Positive, neutral, and negative zones should be color-coded.
- Optional note stays secondary but easy to access.

### Steering notes

- Steering notes should feel like a persistent local notes document.
- They should be edited inside `notes.txt`, not a passive info card.
- Saved/unsaved states must remain obvious.

### Training profiles

- Profiles must be real persisted entities, not temporary chat presets.
- Switching profiles should switch the active saved training state for the local backend.
- New profiles should be creatable without leaving the UI.
- Each profile should expose a readable size indicator.
- Profile switching should reset or clearly separate chat session context so turns do not bleed across identities.

### Reset

- Reset should feel like starting a fresh session, not deleting data dangerously.
- Keep confirmation.

## Motion Spec

- subtle intro fade/slide on load
- soft reflective hover lift on buttons
- smooth glass highlight transitions
- minimal panel shimmer or parallax if it stays cheap and readable

Do not add noisy looping animations.

## Content Direction

Use local-app language where helpful:

- “Local trainer”
- “Session”
- “Guidance profile”
- “Feedback console”
- “Connection online”

Avoid roleplay language that hides what the app actually does.

## Responsiveness

### Desktop

Desktop is the hero layout.

- The fake desktop should remain visible.
- The main chat window should feel centered and intentional.
- Utility widgets can float or dock beside the chat window.

### Mobile

On small screens, reduce the desktop metaphor and prioritize:

- one stacked main window
- readable transcript
- reachable composer
- collapsible utility sections

The design still needs to look themed, not like a broken desktop screenshot.

## Functional Requirements To Preserve

- desktop launches chat app with double-click
- `/api/chat` flow unchanged
- `/api/feedback` flow unchanged
- `/api/guidance` flow unchanged
- reset button still works
- transcript rendering still supports reasoning details
- score slider remains `1-10`
- note field remains optional
- guidance remains persistent
- backend mode and connection status remain visible
- architecture should allow more launchable apps to be added later with the same desktop shell

## QA Inventory

### Claims to verify

- The app now looks like a Frutiger Aero local desktop app, not a generic dark dashboard.
- The main chat flow is still easy to use.
- Rating and notes are still obvious and usable.
- Steering notes are still editable and savable.
- Connection state remains easy to read.
- Desktop and mobile layouts both work.

### Functional checks

1. Send prompt successfully.
2. Receive assistant response successfully.
3. Submit feedback successfully.
4. Save steering notes successfully.
5. Reset chat successfully.
6. Busy and awaiting-feedback states disable the correct controls.
7. Desktop icon double-click opens the chat app window.

### Visual checks

1. Desktop layout at large viewport.
2. Mobile layout around `390x844`.
3. Empty state.
4. Awaiting-feedback state.
5. Connected status state.
6. Unavailable status state if easy to simulate.

### Exploratory checks

1. Very long assistant response should still read cleanly inside the glass transcript layout.
2. Long steering note text should not break the utility panel.

## Implementation Notes

- Keep the existing HTML/JS architecture unless a small structural expansion meaningfully improves the design.
- Prefer semantic sections over div soup.
- Use CSS custom properties for theme tokens.
- A wallpaper image is acceptable if licensing/source is safe, but a strong CSS-only atmospheric background is also acceptable.
- Prioritize coherent chrome and layout before decorative extras.
