# ChatGPT Parity Analysis

## Goal

Redesign the experimental branch UI so the primary chat surface resembles the modern ChatGPT web app as closely as possible while still supporting:

- persistent local sessions
- profile-linked training state
- inline numeric training feedback
- notes and settings access

## Reference Basis

- User-provided ChatGPT screenshot from the active desktop session
- Direct visual familiarity with the modern ChatGPT layout
- Existing live app behavior and constraints in this repo

## Main Sections

1. Left sidebar
   - dark charcoal surface
   - compact top controls
   - stacked conversation list
   - subtle session rows with light hover

2. Top bar
   - minimal, low-height chrome
   - model label on the left
   - compact utility controls on the right

3. Main transcript area
   - dark neutral background
   - centered readable content column
   - assistant replies mostly unframed / low-chrome
   - user replies as compact rounded bubbles aligned right

4. Composer
   - compact rounded dark panel
   - single-line by default
   - auto-grows vertically
   - send button as a compact circular action

5. Secondary controls
   - training-only features should be quieter than the core chat structure
   - notes and settings access should not dominate the surface

## Extracted Design Tokens

### Colors

- Page background: near-black charcoal
- Sidebar: slightly darker than main panel
- Primary text: warm off-white
- Secondary text: muted gray
- User bubble: muted blue
- Borders: soft translucent gray
- Composer: dark raised rounded rectangle

### Typography

- Clean sans serif
- Tight heading usage
- Body text around normal app/chat reading size
- Sidebar labels smaller and softer than message text

### Spacing

- Generous horizontal breathing room in transcript
- Compact but not cramped sidebar list density
- Slim top bar
- Small composer footprint relative to total height

## Proposed Experimental File Focus

- `openclaw-feedback-ui/static/index.html`
  - simplify chrome to ChatGPT-like structure
- `openclaw-feedback-ui/static/styles.css`
  - replace Frutiger-heavy glass styling with dark ChatGPT-like parity styling
- `openclaw-feedback-ui/static/app.js`
  - keep current session/profile/training behavior while adapting interaction details

## Special Constraints

- Similarity should be judged primarily on the active chat surface, not legacy desktop-shell affordances.
- Training controls may remain product-specific, but they should visually defer to the core chat layout.
- The result should still function with the current API/session model.
