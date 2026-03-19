const state = {
  transcript: [],
  hasPendingFeedback: false,
  busy: false,
  model: "",
  backendMode: "rl_proxy",
  proxyBaseUrl: "",
  proxyOk: null,
  guidanceText: "",
  activeProfileId: "",
  profiles: [],
  openApps: {
    chat: false,
    notes: false,
    settings: false,
  },
  activeApp: "",
};

const feedbackDrafts = new Map();

const els = {
  desktopShell: document.querySelector(".desktop-shell"),
  transcript: document.getElementById("transcript"),
  emptyState: document.getElementById("empty-state"),
  sendButton: document.getElementById("send-button"),
  promptInput: document.getElementById("prompt-input"),
  guidanceInput: document.getElementById("guidance-input"),
  guidanceSaveButton: document.getElementById("guidance-save-button"),
  guidanceStatus: document.getElementById("guidance-status"),
  proxyStatus: document.getElementById("proxy-status"),
  modelLabel: document.getElementById("model-label"),
  resetButton: document.getElementById("reset-button"),
  desktopClock: document.getElementById("desktop-clock"),
  taskbarNetwork: document.getElementById("taskbar-network"),
  notesProfileLabel: document.getElementById("notes-profile-label"),
  settingsActiveProfile: document.getElementById("settings-active-profile"),
  chatActiveProfile: document.getElementById("chat-active-profile"),
  chatSidebarProfile: document.getElementById("chat-sidebar-profile"),
  profileNameInput: document.getElementById("profile-name-input"),
  profileCreateButton: document.getElementById("profile-create-button"),
  profileCreateStatus: document.getElementById("profile-create-status"),
  profilesList: document.getElementById("profiles-list"),
  profilesEmpty: document.getElementById("profiles-empty"),
  appLaunchers: Array.from(document.querySelectorAll("[data-app-launch]")),
  appWindows: Array.from(document.querySelectorAll("[data-window]")),
  taskbarApps: Array.from(document.querySelectorAll("[data-task-app]")),
  windowActions: Array.from(document.querySelectorAll("[data-window-action]")),
  openAppButtons: Array.from(document.querySelectorAll("[data-open-app]")),
  messageTemplate: document.getElementById("message-template"),
};

const APP_DEFINITIONS = {
  chat: { hasWindow: true, label: "local-chat.exe" },
  notes: { hasWindow: true, label: "notes.txt" },
  settings: { hasWindow: true, label: "settings.cpl" },
};

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdownLite(text) {
  const escaped = escapeHtml(text);
  const withCode = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
  const blocks = withCode.split(/\n{2,}/).map((block) => {
    const lines = block
      .split("\n")
      .map((line) => `${line}<br />`)
      .join("");
    return `<p>${lines}</p>`;
  });
  return blocks.join("");
}

function setBusy(isBusy) {
  state.busy = isBusy;
  renderControls();
  renderTranscript();
}

function upsertStreamingTurn(turn) {
  const existingIndex = state.transcript.findIndex((item) => item.id === turn.id);
  if (existingIndex === -1) {
    state.transcript = [...state.transcript, turn];
    return;
  }
  const nextTranscript = [...state.transcript];
  nextTranscript[existingIndex] = { ...nextTranscript[existingIndex], ...turn };
  state.transcript = nextTranscript;
}

function appendStreamingDelta(turnId, delta) {
  if (!delta) {
    return;
  }
  const existingIndex = state.transcript.findIndex((item) => item.id === turnId);
  if (existingIndex === -1) {
    return;
  }
  const nextTranscript = [...state.transcript];
  const current = nextTranscript[existingIndex];
  nextTranscript[existingIndex] = {
    ...current,
    content: `${current.content || ""}${delta}`,
    streaming: true,
  };
  state.transcript = nextTranscript;
}

function clearStreamingTurns() {
  state.transcript = state.transcript.filter((item) => !item.streaming);
}

function getActiveProfile() {
  return state.profiles.find((profile) => profile.id === state.activeProfileId) || null;
}

function getFeedbackDraft(turnId) {
  if (!feedbackDrafts.has(turnId)) {
    feedbackDrafts.set(turnId, { rating: "", note: "" });
  }
  return feedbackDrafts.get(turnId);
}

function getFeedbackSummary(feedback) {
  if (!feedback) {
    return null;
  }

  const explicitRating = feedback.rating;
  const score = Number(feedback.score ?? NaN);
  let label = "Rated";
  if (explicitRating === "good" || (!explicitRating && Number.isFinite(score) && score >= 7)) {
    label = "Liked";
  } else if (explicitRating === "bad" || (!explicitRating && Number.isFinite(score) && score <= 4)) {
    label = "Disliked";
  } else if (Number.isFinite(score)) {
    label = `${score}/10`;
  }

  return {
    label,
    score,
  };
}

function setGuidanceStatus(text) {
  if (els.guidanceStatus) {
    els.guidanceStatus.textContent = text;
  }
}

function setProfileCreateStatus(text) {
  if (els.profileCreateStatus) {
    els.profileCreateStatus.textContent = text;
  }
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  const scaled = value / 1024 ** exponent;
  const digits = scaled >= 100 || exponent === 0 ? 0 : scaled >= 10 ? 1 : 2;
  return `${scaled.toFixed(digits)} ${units[exponent]}`;
}

function updateProfileLabels() {
  const activeProfile = getActiveProfile();
  const profileLabel = activeProfile ? activeProfile.name : state.activeProfileId || "Default";
  if (els.chatActiveProfile) {
    els.chatActiveProfile.textContent = profileLabel;
  }
  if (els.chatSidebarProfile) {
    els.chatSidebarProfile.textContent = profileLabel;
  }
  if (els.notesProfileLabel) {
    els.notesProfileLabel.textContent = activeProfile ? `Profile: ${profileLabel}` : "Profile";
  }
  if (els.settingsActiveProfile) {
    els.settingsActiveProfile.textContent = activeProfile ? `Active: ${profileLabel}` : "Profile";
  }
}

function renderProfiles() {
  if (!els.profilesList || !els.profilesEmpty) {
    return;
  }

  els.profilesList.innerHTML = "";
  const profiles = Array.isArray(state.profiles) ? state.profiles : [];
  els.profilesEmpty.classList.toggle("hidden", profiles.length > 0);

  for (const profile of profiles) {
    const card = document.createElement("article");
    card.className = "profile-card";
    if (profile.id === state.activeProfileId) {
      card.classList.add("active-profile");
    }

    const head = document.createElement("div");
    head.className = "profile-head";

    const titleWrap = document.createElement("div");
    const title = document.createElement("h3");
    title.className = "profile-name";
    title.textContent = profile.name || profile.id;
    const subtitle = document.createElement("p");
    subtitle.className = "profile-id";
    subtitle.textContent = profile.id;
    titleWrap.append(title, subtitle);
    head.appendChild(titleWrap);

    if (profile.id === state.activeProfileId) {
      const badge = document.createElement("span");
      badge.className = "profile-badge";
      badge.textContent = "Active";
      head.appendChild(badge);
    }

    const stats = document.createElement("div");
    stats.className = "profile-stats";
    const training = profile.training || {};
    const statValues = [
      `Total: ${formatBytes(profile.total_size_bytes ?? profile.size_bytes ?? 0)}`,
      `Notes: ${formatBytes(profile.guidance_size_bytes ?? 0)}`,
      `Updates: ${Number(training.total_updates || 0)}`,
      `Feedback: ${Number(training.total_feedback || 0)}`,
    ];
    for (const text of statValues) {
      const chip = document.createElement("span");
      chip.className = "profile-stat";
      chip.textContent = text;
      stats.appendChild(chip);
    }

    const footer = document.createElement("div");
    footer.className = "profile-footer";
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent =
      profile.id === state.activeProfileId
        ? "Current notes and future live updates stay on this profile."
        : "Switching loads this profile's saved notes and training state.";

    const button = document.createElement("button");
    button.className = "profile-activate-button";
    button.type = "button";
    button.textContent = profile.id === state.activeProfileId ? "Current profile" : "Switch to profile";
    button.disabled = profile.id === state.activeProfileId || state.busy;
    button.addEventListener("click", () => {
      selectProfile(profile.id);
    });
    footer.append(hint, button);

    card.append(head, stats, footer);
    els.profilesList.append(card);
  }

  updateProfileLabels();
}

function updateDesktopClock() {
  if (!els.desktopClock) {
    return;
  }
  const now = new Date();
  els.desktopClock.textContent = now.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function getWindowByApp(appName) {
  return els.appWindows.find((item) => item.dataset.window === appName) || null;
}

function setDesktopSelection(appName) {
  for (const launcher of els.appLaunchers) {
    launcher.classList.toggle("active", launcher.dataset.appLaunch === appName);
  }
}

function syncAppChrome() {
  const anyOpen = Object.values(state.openApps).some(Boolean);
  if (els.desktopShell) {
    els.desktopShell.classList.toggle("app-open", anyOpen);
  }

  for (const appName of Object.keys(APP_DEFINITIONS)) {
    const appWindow = getWindowByApp(appName);
    if (appWindow) {
      appWindow.classList.toggle("window-hidden", !state.openApps[appName]);
    }
  }

  for (const button of els.taskbarApps) {
    const appName = button.dataset.taskApp;
    const isActive = state.activeApp === appName;
    const isOpen = Boolean(state.openApps[appName]);
    button.classList.toggle("active", isActive);
    button.classList.toggle("open-pill", isOpen && !isActive);
    button.classList.toggle("inactive-pill", !isOpen);
  }
}

function openApp(appName) {
  const definition = APP_DEFINITIONS[appName];
  if (!definition) {
    return;
  }
  setDesktopSelection(appName);
  if (!definition.hasWindow) {
    window.alert(`${definition.label} is not installed yet. The desktop shell is ready for more apps later.`);
    return;
  }
  for (const otherName of Object.keys(APP_DEFINITIONS)) {
    state.openApps[otherName] = false;
  }
  state.openApps[appName] = true;
  state.activeApp = appName;
  syncAppChrome();
}

function closeApp(appName) {
  if (!APP_DEFINITIONS[appName]) {
    return;
  }
  state.openApps[appName] = false;
  state.activeApp = "";
  setDesktopSelection("");
  syncAppChrome();
}

function renderStatus() {
  if (!els.proxyStatus) {
    return;
  }
  els.proxyStatus.className = "status-chip";
  const backendLabel =
    state.backendMode === "rl_proxy"
      ? "Proxy"
      : state.backendMode === "trainer_api"
        ? "Trainer"
        : "Backend";

  if (state.proxyOk === null) {
    els.proxyStatus.textContent = `Checking ${backendLabel.toLowerCase()}...`;
    if (els.taskbarNetwork) {
      els.taskbarNetwork.textContent = "Link check";
    }
    return;
  }

  if (state.proxyOk) {
    els.proxyStatus.textContent = `${backendLabel} connected`;
    els.proxyStatus.classList.add("status-good");
    if (els.taskbarNetwork) {
      els.taskbarNetwork.textContent = "Local online";
    }
    return;
  }

  els.proxyStatus.textContent = `${backendLabel} unavailable`;
  els.proxyStatus.classList.add("status-bad");
  if (els.taskbarNetwork) {
    els.taskbarNetwork.textContent = "Link offline";
  }
}

function renderControls() {
  const disabled = state.busy;
  els.sendButton.disabled = disabled || !els.promptInput.value.trim();
  els.promptInput.disabled = disabled;
  els.guidanceInput.disabled = disabled;
  els.guidanceSaveButton.disabled = disabled;
  els.resetButton.disabled = disabled;
  els.modelLabel.textContent = state.model || "Model";

  if (els.profileNameInput) {
    els.profileNameInput.disabled = disabled;
  }
  if (els.profileCreateButton) {
    els.profileCreateButton.disabled = disabled || !els.profileNameInput.value.trim();
  }

  updateProfileLabels();
}

function renderTranscript() {
  els.transcript.innerHTML = "";
  const hasMessages = state.transcript.length > 0;
  els.emptyState.style.display = hasMessages ? "none" : "grid";

  for (const item of state.transcript) {
    const fragment = els.messageTemplate.content.cloneNode(true);
    const message = fragment.querySelector(".message");
    const roleLabel = fragment.querySelector(".role-label");
    const body = fragment.querySelector(".message-body");
    const feedbackPill = fragment.querySelector(".feedback-pill");
    const reasoningBlock = fragment.querySelector(".reasoning-block");
    const reasoningBody = fragment.querySelector(".reasoning-body");
    const feedbackControls = fragment.querySelector(".message-feedback-controls");
    const feedbackGoodButton = fragment.querySelector('[data-feedback-choice="good"]');
    const feedbackBadButton = fragment.querySelector('[data-feedback-choice="bad"]');
    const feedbackInput = fragment.querySelector(".message-feedback-input");
    const feedbackSendButton = fragment.querySelector(".message-feedback-send");
    const feedbackSummary = fragment.querySelector(".message-feedback-summary");

    message.classList.add(item.role === "assistant" ? "assistant-message" : "user-message");
    if (item.role === "assistant") {
      message.dataset.turnId = item.id || "";
    }
    if (item.streaming) {
      message.classList.add("streaming-message");
    }
    roleLabel.textContent = item.role === "assistant" ? "Assistant" : "You";
    body.innerHTML = renderMarkdownLite(item.content);

    if (item.reasoning) {
      reasoningBlock.classList.remove("hidden");
      reasoningBody.textContent = item.reasoning;
    }

    if (item.role !== "assistant") {
      feedbackControls.classList.add("hidden");
    } else if (item.streaming) {
      feedbackControls.classList.add("hidden");
      feedbackPill.classList.remove("hidden");
      feedbackPill.textContent = "Streaming";
      feedbackPill.classList.add("neutral-pill");
    } else if (item.feedback) {
      const summary = getFeedbackSummary(item.feedback);
      feedbackControls.classList.add("hidden");
      feedbackPill.classList.remove("hidden");
      feedbackPill.textContent = summary?.label || "Rated";
      feedbackPill.classList.add(
        summary?.label === "Liked" ? "good-pill" : summary?.label === "Disliked" ? "bad-pill" : "neutral-pill"
      );

      feedbackSummary.classList.remove("hidden");
      feedbackSummary.textContent = item.feedback.note
        ? `${summary?.label || "Rated"} for training. Note: ${item.feedback.note}`
        : `${summary?.label || "Rated"} for training.`;
    } else if (item.feedback_pending !== false) {
      const draft = getFeedbackDraft(item.id);
      feedbackControls.classList.remove("hidden");
      feedbackGoodButton.classList.toggle("selected", draft.rating === "good");
      feedbackBadButton.classList.toggle("selected", draft.rating === "bad");
      feedbackInput.value = draft.note;
      feedbackInput.disabled = state.busy;
      feedbackSendButton.disabled = state.busy || !draft.rating;

      feedbackGoodButton.addEventListener("click", () => {
        const nextDraft = getFeedbackDraft(item.id);
        nextDraft.rating = nextDraft.rating === "good" ? "" : "good";
        renderTranscript();
      });

      feedbackBadButton.addEventListener("click", () => {
        const nextDraft = getFeedbackDraft(item.id);
        nextDraft.rating = nextDraft.rating === "bad" ? "" : "bad";
        renderTranscript();
      });

      feedbackInput.addEventListener("input", (event) => {
        getFeedbackDraft(item.id).note = event.target.value;
      });

      feedbackSendButton.addEventListener("click", () => {
        sendFeedback(item.id);
      });
    } else {
      feedbackControls.classList.add("hidden");
    }

    els.transcript.appendChild(fragment);
  }

  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function applyProfilesState(profilePayload) {
  if (!profilePayload) {
    return;
  }
  state.profiles = Array.isArray(profilePayload.profiles) ? profilePayload.profiles : [];
  if (profilePayload.active_profile_id) {
    state.activeProfileId = profilePayload.active_profile_id;
  }
  renderProfiles();
  renderControls();
}

function applyServerState(serverState, proxyState = null, profilePayload = null) {
  state.transcript = serverState.transcript || [];
  state.hasPendingFeedback = Boolean(serverState.awaiting_feedback);
  state.busy = Boolean(serverState.busy);
  state.model = serverState.model || "";
  state.backendMode = serverState.backend_mode || "rl_proxy";
  state.proxyBaseUrl = serverState.proxy_base_url || "";
  state.activeProfileId = serverState.active_profile_id || state.activeProfileId || "";
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
  if (profilePayload) {
    applyProfilesState(profilePayload);
  } else {
    updateProfileLabels();
  }
  syncAppChrome();
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

function processSseBlock(rawBlock, onEvent) {
  const lines = rawBlock.split("\n");
  let eventName = "message";
  const dataLines = [];

  for (const line of lines) {
    if (!line) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!dataLines.length) {
    return;
  }

  const payloadText = dataLines.join("\n");
  if (payloadText === "[DONE]") {
    return;
  }

  let payload = {};
  try {
    payload = JSON.parse(payloadText);
  } catch (error) {
    throw new Error("Streaming payload contained invalid JSON.");
  }
  onEvent(eventName, payload);
}

async function readSseStream(response, onEvent) {
  if (!response.body) {
    throw new Error("Streaming response body was unavailable.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r/g, "");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (block.trim()) {
        processSseBlock(block, onEvent);
      }
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    processSseBlock(buffer, onEvent);
  }
}

async function refreshState() {
  const [statePayload, statusPayload, profilesPayload] = await Promise.all([
    fetchJson("/api/state"),
    fetchJson("/api/status"),
    fetchJson("/api/profiles"),
  ]);
  applyServerState(statePayload.state, statusPayload.state.proxy, profilesPayload);
}

async function refreshProfiles() {
  const payload = await fetchJson("/api/profiles");
  applyServerState(payload.state, { ok: state.proxyOk !== false }, payload);
}

async function sendPrompt() {
  const prompt = els.promptInput.value.trim();
  if (!prompt || state.busy) {
    return;
  }

  setBusy(true);
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = payload.detail || `Request failed with ${response.status}`;
      throw new Error(detail);
    }

    els.promptInput.value = "";
    renderControls();

    await readSseStream(response, (eventName, payload) => {
      if (eventName === "start") {
        if (payload.user_turn) {
          upsertStreamingTurn(payload.user_turn);
        }
        if (payload.assistant_turn) {
          upsertStreamingTurn(payload.assistant_turn);
        }
        renderTranscript();
        return;
      }

      if (eventName === "delta") {
        appendStreamingDelta(payload.assistant_turn_id, payload.delta || "");
        renderTranscript();
        return;
      }

      if (eventName === "state") {
        applyServerState(payload.state, { ok: true });
        return;
      }

      if (eventName === "error") {
        clearStreamingTurns();
        renderTranscript();
        throw new Error(payload.detail || "Streaming failed.");
      }
    });

    if (state.busy) {
      await refreshState();
    }
  } catch (error) {
    try {
      await refreshState();
    } catch (refreshError) {
      // Keep the original error if refresh also fails.
    }
    window.alert(error.message);
    setBusy(false);
  }
}

async function sendFeedback(assistantTurnId) {
  if (state.busy) {
    return;
  }

  const draft = getFeedbackDraft(assistantTurnId);
  if (!draft.rating) {
    return;
  }

  setBusy(true);
  try {
    const payload = await fetchJson("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        assistant_turn_id: assistantTurnId,
        rating: draft.rating,
        score: draft.rating === "good" ? 9 : 2,
        note: draft.note.trim(),
      }),
    });
    feedbackDrafts.delete(assistantTurnId);
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
      body: JSON.stringify({ text: els.guidanceInput.value }),
    });
    applyServerState(payload.state, { ok: state.proxyOk !== false });
    setGuidanceStatus(
      payload.state.guidance_text
        ? "Steering notes saved and active on future prompts."
        : "Saved notes cleared."
    );
    await refreshProfiles();
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function selectProfile(profileId) {
  if (!profileId || profileId === state.activeProfileId || state.busy) {
    return;
  }

  if (state.transcript.length > 0) {
    const shouldSwitch = window.confirm("Switch profiles and start a fresh local chat session?");
    if (!shouldSwitch) {
      return;
    }
  }

  setBusy(true);
  setProfileCreateStatus("Switching profile...");
  try {
    const payload = await fetchJson("/api/profiles/select", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    els.promptInput.value = "";
    feedbackDrafts.clear();
    applyServerState(payload.state, { ok: state.proxyOk !== false }, payload);
    setGuidanceStatus(
      payload.state.guidance_text
        ? "Saved notes loaded for this profile."
        : "No saved notes yet for this profile."
    );
    setProfileCreateStatus("Profile switched.");
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
    setProfileCreateStatus("Profile switch failed.");
  }
}

async function createProfile() {
  if (state.busy) {
    return;
  }

  const name = els.profileNameInput.value.trim();
  if (!name) {
    return;
  }

  setBusy(true);
  setProfileCreateStatus("Creating profile...");
  try {
    const payload = await fetchJson("/api/profiles", {
      method: "POST",
      body: JSON.stringify({ name, select_after_create: true }),
    });
    els.profileNameInput.value = "";
    els.promptInput.value = "";
    feedbackDrafts.clear();
    applyServerState(payload.state, { ok: state.proxyOk !== false }, payload);
    setGuidanceStatus(
      payload.state.guidance_text
        ? "Saved notes loaded for the new profile."
        : "New profile created. Add notes in notes.txt when you want."
    );
    setProfileCreateStatus("Profile created and activated.");
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
    setProfileCreateStatus("Profile creation failed.");
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
    els.promptInput.value = "";
    feedbackDrafts.clear();
    applyServerState(payload.state, { ok: true });
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

els.sendButton.addEventListener("click", sendPrompt);
els.guidanceSaveButton.addEventListener("click", saveGuidance);
els.resetButton.addEventListener("click", resetChat);

if (els.profileCreateButton) {
  els.profileCreateButton.addEventListener("click", createProfile);
}

els.promptInput.addEventListener("input", renderControls);
els.promptInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendPrompt();
  }
});

els.guidanceInput.addEventListener("input", () => {
  setGuidanceStatus("Unsaved steering changes.");
});

if (els.profileNameInput) {
  els.profileNameInput.addEventListener("input", () => {
    setProfileCreateStatus("Create a new saved training profile.");
    renderControls();
  });
  els.profileNameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      createProfile();
    }
  });
}

for (const launcher of els.appLaunchers) {
  launcher.addEventListener("click", () => {
    setDesktopSelection(launcher.dataset.appLaunch);
  });
  launcher.addEventListener("dblclick", () => {
    openApp(launcher.dataset.appLaunch);
  });
}

for (const taskbarButton of els.taskbarApps) {
  taskbarButton.addEventListener("click", () => {
    openApp(taskbarButton.dataset.taskApp);
  });
}

for (const appButton of els.openAppButtons) {
  appButton.addEventListener("click", () => {
    openApp(appButton.dataset.openApp);
  });
}

for (const actionButton of els.windowActions) {
  actionButton.addEventListener("click", () => {
    const targetWindow = actionButton.dataset.targetWindow;
    const action = actionButton.dataset.windowAction;
    if (action === "close" || action === "minimize") {
      closeApp(targetWindow);
      return;
    }
    openApp(targetWindow);
  });
}

updateDesktopClock();
window.setInterval(updateDesktopClock, 30000);
syncAppChrome();
setProfileCreateStatus("Create a new saved training profile.");

refreshState().catch((error) => {
  state.proxyOk = false;
  renderStatus();
  window.alert(error.message);
});
