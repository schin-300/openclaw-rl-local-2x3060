const state = {
  transcript: [],
  awaitingFeedback: false,
  busy: false,
  model: "",
  proxyBaseUrl: "",
  proxyOk: null,
  guidanceText: "",
};

const els = {
  transcript: document.getElementById("transcript"),
  emptyState: document.getElementById("empty-state"),
  sendButton: document.getElementById("send-button"),
  promptInput: document.getElementById("prompt-input"),
  feedbackCard: document.getElementById("feedback-card"),
  feedbackNote: document.getElementById("feedback-note"),
  feedbackScore: document.getElementById("feedback-score"),
  feedbackScoreValue: document.getElementById("feedback-score-value"),
  feedbackSubmit: document.getElementById("feedback-submit"),
  guidanceInput: document.getElementById("guidance-input"),
  guidanceSaveButton: document.getElementById("guidance-save-button"),
  guidanceStatus: document.getElementById("guidance-status"),
  proxyStatus: document.getElementById("proxy-status"),
  modelLabel: document.getElementById("model-label"),
  resetButton: document.getElementById("reset-button"),
  messageTemplate: document.getElementById("message-template"),
};

function setBusy(isBusy) {
  state.busy = isBusy;
  renderControls();
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdownLite(text) {
  const escaped = escapeHtml(text);
  const withCode = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
  const blocks = withCode.split(/\n{2,}/).map((block) => {
    const lines = block.split("\n").map((line) => `${line}<br />`).join("");
    return `<p>${lines}</p>`;
  });
  return blocks.join("");
}

function renderTranscript() {
  els.transcript.innerHTML = "";
  els.emptyState.style.display = state.transcript.length ? "none" : "grid";

  for (const item of state.transcript) {
    const fragment = els.messageTemplate.content.cloneNode(true);
    const message = fragment.querySelector(".message");
    const roleLabel = fragment.querySelector(".role-label");
    const body = fragment.querySelector(".message-body");
    const feedbackPill = fragment.querySelector(".feedback-pill");
    const feedbackNote = fragment.querySelector(".feedback-note");
    const reasoningBlock = fragment.querySelector(".reasoning-block");
    const reasoningBody = fragment.querySelector(".reasoning-body");

    message.classList.add(item.role === "assistant" ? "assistant-message" : "user-message");
    roleLabel.textContent = item.role === "assistant" ? "Assistant" : "You";
    body.innerHTML = renderMarkdownLite(item.content);

    if (item.reasoning) {
      reasoningBlock.classList.remove("hidden");
      reasoningBody.textContent = item.reasoning;
    }

    if (item.feedback) {
      const score = Number(item.feedback.score ?? NaN);
      feedbackPill.classList.remove("hidden");
      if (Number.isFinite(score)) {
        feedbackPill.textContent = `${score}/10`;
        feedbackPill.classList.add(score >= 7 ? "good-pill" : score <= 4 ? "bad-pill" : "neutral-pill");
      } else {
        const legacyGood = item.feedback.rating === "good";
        feedbackPill.textContent = legacyGood ? "Good" : "Bad";
        feedbackPill.classList.add(legacyGood ? "good-pill" : "bad-pill");
      }
      if (item.feedback.note) {
        feedbackNote.classList.remove("hidden");
        feedbackNote.textContent = `Note: ${item.feedback.note}`;
      }
    }

    els.transcript.appendChild(fragment);
  }

  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function updateFeedbackScoreLabel() {
  els.feedbackScoreValue.textContent = els.feedbackScore.value;
}

function setGuidanceStatus(text) {
  els.guidanceStatus.textContent = text;
}

function renderControls() {
  const disabled = state.busy;
  els.sendButton.disabled = disabled || state.awaitingFeedback || !els.promptInput.value.trim();
  els.promptInput.disabled = disabled || state.awaitingFeedback;
  els.feedbackSubmit.disabled = disabled || !state.awaitingFeedback;
  els.feedbackNote.disabled = disabled || !state.awaitingFeedback;
  els.feedbackScore.disabled = disabled || !state.awaitingFeedback;
  els.guidanceInput.disabled = disabled;
  els.guidanceSaveButton.disabled = disabled;
  els.feedbackCard.classList.toggle("disabled-card", !state.awaitingFeedback);
  els.resetButton.disabled = disabled;
  els.modelLabel.textContent = state.model || "Model";
  updateFeedbackScoreLabel();
}

function renderStatus() {
  els.proxyStatus.className = "status-chip";
  if (state.proxyOk === null) {
    els.proxyStatus.textContent = "Checking proxy...";
    return;
  }
  if (state.proxyOk) {
    els.proxyStatus.textContent = "Proxy connected";
    els.proxyStatus.classList.add("status-good");
    return;
  }
  els.proxyStatus.textContent = "Proxy unavailable";
  els.proxyStatus.classList.add("status-bad");
}

function applyServerState(serverState, proxyState = null) {
  state.transcript = serverState.transcript || [];
  state.awaitingFeedback = Boolean(serverState.awaiting_feedback);
  state.busy = Boolean(serverState.busy);
  state.model = serverState.model || "";
  state.proxyBaseUrl = serverState.proxy_base_url || "";
  state.guidanceText = serverState.guidance_text || "";
  if (els.guidanceInput.value !== state.guidanceText) {
    els.guidanceInput.value = state.guidanceText;
  }
  setGuidanceStatus(
    state.guidanceText
      ? "Saved steering notes are active for future prompts."
      : "No saved steering notes yet."
  );
  if (proxyState) {
    state.proxyOk = Boolean(proxyState.ok);
  }
  renderTranscript();
  renderControls();
  renderStatus();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || `Request failed with ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

async function refreshState() {
  const [statePayload, statusPayload] = await Promise.all([
    fetchJson("/api/state"),
    fetchJson("/api/status"),
  ]);
  applyServerState(statePayload.state, statusPayload.state.proxy);
}

async function sendPrompt() {
  const prompt = els.promptInput.value.trim();
  if (!prompt || state.awaitingFeedback) {
    return;
  }
  setBusy(true);
  try {
    const payload = await fetchJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    });
    els.promptInput.value = "";
    applyServerState(payload.state, { ok: true });
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function sendFeedback() {
  if (!state.awaitingFeedback) {
    return;
  }
  setBusy(true);
  try {
    const payload = await fetchJson("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        score: Number(els.feedbackScore.value),
        note: els.feedbackNote.value.trim(),
      }),
    });
    els.feedbackNote.value = "";
    applyServerState(payload.state, { ok: true });
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function saveGuidance() {
  setBusy(true);
  try {
    const payload = await fetchJson("/api/guidance", {
      method: "POST",
      body: JSON.stringify({
        text: els.guidanceInput.value,
      }),
    });
    applyServerState(payload.state, { ok: state.proxyOk !== false });
    setGuidanceStatus(
      payload.state.guidance_text
        ? "Steering notes saved and active on future prompts."
        : "Saved notes cleared."
    );
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function resetChat() {
  const shouldReset = window.confirm("Reset the visible transcript and start a fresh chat?");
  if (!shouldReset) {
    return;
  }
  setBusy(true);
  try {
    const payload = await fetchJson("/api/reset", { method: "POST", body: "{}" });
    els.feedbackNote.value = "";
    els.promptInput.value = "";
    applyServerState(payload.state, { ok: true });
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

els.sendButton.addEventListener("click", sendPrompt);
els.feedbackSubmit.addEventListener("click", sendFeedback);
els.guidanceSaveButton.addEventListener("click", saveGuidance);
els.resetButton.addEventListener("click", resetChat);
els.promptInput.addEventListener("input", renderControls);
els.feedbackScore.addEventListener("input", updateFeedbackScoreLabel);
els.guidanceInput.addEventListener("input", () => {
  setGuidanceStatus("Unsaved steering changes.");
});
els.promptInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendPrompt();
  }
});

updateFeedbackScoreLabel();
refreshState().catch((error) => {
  state.proxyOk = false;
  renderStatus();
  window.alert(error.message);
});
