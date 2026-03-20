# ChatGPT Parity Verification

Branch: `experimental/chatgpt-parity-ui`

Viewport used for verification: `1920x1080`

## Measured UI Values

- Sidebar width: `260px`
- Top bar height: `52px`
- Transcript width: `860px`
- Composer width: `720px`
- Composer height at rest: `64px`
- Page background: `rgb(33, 33, 33)`
- Desktop/taskbar chrome hidden in parity mode: `yes`

## Functional Checks

- Chat view opens immediately on load
- New chat creates a fresh persistent session
- Session titles come from the first prompt
- Streaming text grows in place
- Live `tok/s` appears under the active assistant response
- Inline `1-10` feedback still works
- Notes and settings still open and close back into chat
- Sessions persist after reload
- Profile switching still changes the session set

## Scorecard

1. Sidebar structure and density: `14 / 15`
2. Top bar proportion and placement: `9 / 10`
3. Main canvas color and surface treatment: `10 / 10`
4. Transcript column geometry: `14 / 15`
5. Message styling parity: `9 / 10`
6. Composer parity: `14 / 15`
7. Typography parity: `9 / 10`
8. Interaction parity: `9 / 10`
9. Product-specific control restraint: `4 / 5`

Total: `92 / 100`

10-point similarity score: `9.2 / 10`
