const state = {
  transcript: [],
  hereticTranscript: [],
  sessions: [],
  activeSessionId: "",
  busy: false,
  hereticBusy: false,
  stopRequested: false,
  hereticStopRequested: false,
  model: "",
  hereticModel: "",
  backendMode: "rl_proxy",
  chatBackendMode: "rl_proxy",
  trainingBackendMode: "rl_proxy",
  proxyBaseUrl: "",
  trainingProxyBaseUrl: "",
  proxyOk: null,
  proxyHealth: null,
  guidanceText: "",
  systemPromptText: "",
  thinkingEnabled: false,
  activeProfileId: "",
  profiles: [],
  activeApp: "chat",
  modelControl: null,
  gpuMode: "",
  gpuModeLabel: "",
  calendarMonthAnchor: "",
  calendarSelectedDateKey: "",
  calendarEvents: {},
};

const feedbackDrafts = new Map();
let activeStreamController = null;
const ACTIVE_APP_STORAGE_KEY = "openclaw_os_active_app_v1";
const CALENDAR_STORAGE_KEY = "openclaw_os_calendar_v1";
const APP_REGISTRY = {
  chat: {
    id: "chat",
    title: "OpenClaw Chat",
    eyebrow: "LIVE RL CHAT",
    summary: "Stay in the feedback loop, stream replies live, and keep your training flow intact.",
  },
  heretic: {
    id: "heretic",
    title: "27B Chat",
    eyebrow: "SIDECAR CHAT",
    summary: "A separate simple 27B chat lane with no RL controls mixed in.",
  },
  "model-control": {
    id: "model-control",
    title: "Model Control",
    eyebrow: "GPU OWNERSHIP",
    summary: "Assign the machine's two GPUs to either the live OpenClaw RL lane or the separate 27B chat lane.",
  },
  calendar: {
    id: "calendar",
    title: "Calendar",
    eyebrow: "OPENCLAW OS APP",
    summary: "A simple local planning surface that lives beside the chat instead of kicking you out of it.",
  },
  notes: {
    id: "notes",
    title: "Notes",
    eyebrow: "PROFILE MEMORY",
    summary: "Shape the active profile with a lightweight system prompt and steering notes.",
  },
  profiles: {
    id: "profiles",
    title: "Profiles",
    eyebrow: "TRAINING TRACKS",
    summary: "Switch between saved chats, notes, system prompts, and learned state.",
  },
};

const els = {
  appShell: document.getElementById("app-shell"),
  appViews: Array.from(document.querySelectorAll("[data-app-view]")),
  appLauncherButtons: Array.from(document.querySelectorAll("[data-open-app]")),
  transcript: document.getElementById("transcript"),
  emptyState: document.getElementById("empty-state"),
  sendButton: document.getElementById("send-button"),
  sendButtonIcon: document.querySelector("#send-button .send-button-icon"),
  promptInput: document.getElementById("prompt-input"),
  hereticTranscript: document.getElementById("heretic-transcript"),
  hereticEmptyState: document.getElementById("heretic-empty-state"),
  hereticPromptInput: document.getElementById("heretic-prompt-input"),
  hereticSendButton: document.getElementById("heretic-send-button"),
  hereticSendButtonIcon: document.getElementById("heretic-send-button-icon"),
  hereticResetButton: document.getElementById("heretic-reset-button"),
  hereticStatusChip: document.getElementById("heretic-status-chip"),
  systemPromptInput: document.getElementById("system-prompt-input"),
  guidanceInput: document.getElementById("guidance-input"),
  guidanceSaveButton: document.getElementById("guidance-save-button"),
  guidanceStatus: document.getElementById("guidance-status"),
  proxyStatus: document.getElementById("proxy-status"),
  gpuModePill: document.getElementById("gpu-mode-pill"),
  modelLabel: document.getElementById("model-label"),
  windowModelTitle: document.getElementById("window-model-title"),
  windowModelEyebrow: document.getElementById("window-model-eyebrow"),
  windowAppSummary: document.getElementById("window-app-summary"),
  thinkingToggle: document.getElementById("thinking-toggle"),
  newSessionButton: document.getElementById("new-session-button"),
  sessionList: document.getElementById("session-list"),
  sessionsEmpty: document.getElementById("sessions-empty"),
  notesProfileLabel: document.getElementById("notes-profile-label"),
  settingsActiveProfile: document.getElementById("settings-active-profile"),
  chatActiveProfile: document.getElementById("chat-active-profile"),
  profileNameInput: document.getElementById("profile-name-input"),
  profileCreateButton: document.getElementById("profile-create-button"),
  profileCreateStatus: document.getElementById("profile-create-status"),
  profilesList: document.getElementById("profiles-list"),
  profilesEmpty: document.getElementById("profiles-empty"),
  calendarGrid: document.getElementById("calendar-grid"),
  calendarMonthLabel: document.getElementById("calendar-month-label"),
  calendarSelectedLabel: document.getElementById("calendar-selected-label"),
  calendarSelectedMeta: document.getElementById("calendar-selected-meta"),
  calendarDayEvents: document.getElementById("calendar-day-events"),
  calendarEventTitle: document.getElementById("calendar-event-title"),
  calendarEventNote: document.getElementById("calendar-event-note"),
  calendarSaveButton: document.getElementById("calendar-save-button"),
  calendarClearButton: document.getElementById("calendar-clear-button"),
  calendarPrevMonth: document.getElementById("calendar-prev-month"),
  calendarNextMonth: document.getElementById("calendar-next-month"),
  modelControlModePill: document.getElementById("model-control-mode-pill"),
  openclawModeCard: document.getElementById("openclaw-mode-card"),
  hereticModeCard: document.getElementById("heretic-mode-card"),
  openclawModeStatus: document.getElementById("openclaw-mode-status"),
  hereticModeStatus: document.getElementById("heretic-mode-status"),
  openclawModeModel: document.getElementById("openclaw-mode-model"),
  hereticModeModel: document.getElementById("heretic-mode-model"),
  openclawModeHealth: document.getElementById("openclaw-mode-health"),
  hereticModeHealth: document.getElementById("heretic-mode-health"),
  openclawModeButton: document.getElementById("openclaw-mode-button"),
  hereticModeButton: document.getElementById("heretic-mode-button"),
  messageTemplate: document.getElementById("message-template"),
};

function todayDateKey() {
  return formatDateKey(new Date());
}

function formatDateKey(value) {
  const date = value instanceof Date ? value : new Date(value);
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseDateKey(dateKey) {
  const [year, month, day] = String(dateKey || "")
    .split("-")
    .map((value) => Number(value));
  if (!year || !month || !day) {
    return new Date();
  }
  return new Date(year, month - 1, day);
}

function startOfMonth(date) {
  return new Date(date.getFullYear(), date.getMonth(), 1);
}

function loadActiveApp() {
  const saved = window.localStorage.getItem(ACTIVE_APP_STORAGE_KEY);
  return APP_REGISTRY[saved] ? saved : "chat";
}

function saveActiveApp() {
  window.localStorage.setItem(ACTIVE_APP_STORAGE_KEY, state.activeApp);
}

function loadCalendarState() {
  const today = new Date();
  const fallback = {
    calendarMonthAnchor: formatDateKey(startOfMonth(today)),
    calendarSelectedDateKey: todayDateKey(),
    calendarEvents: {},
  };

  try {
    const raw = window.localStorage.getItem(CALENDAR_STORAGE_KEY);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw);
    return {
      calendarMonthAnchor:
        typeof parsed.calendarMonthAnchor === "string" && parsed.calendarMonthAnchor
          ? parsed.calendarMonthAnchor
          : fallback.calendarMonthAnchor,
      calendarSelectedDateKey:
        typeof parsed.calendarSelectedDateKey === "string" && parsed.calendarSelectedDateKey
          ? parsed.calendarSelectedDateKey
          : fallback.calendarSelectedDateKey,
      calendarEvents:
        parsed.calendarEvents && typeof parsed.calendarEvents === "object" && !Array.isArray(parsed.calendarEvents)
          ? parsed.calendarEvents
          : fallback.calendarEvents,
    };
  } catch (_error) {
    return fallback;
  }
}

function saveCalendarState() {
  window.localStorage.setItem(
    CALENDAR_STORAGE_KEY,
    JSON.stringify({
      calendarMonthAnchor: state.calendarMonthAnchor,
      calendarSelectedDateKey: state.calendarSelectedDateKey,
      calendarEvents: state.calendarEvents,
    })
  );
}

state.activeApp = loadActiveApp();
Object.assign(state, loadCalendarState());

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

function extractFinalAnswer(text) {
  const cleanText = String(text || "").trim();
  if (!cleanText) {
    return "";
  }

  const patterns = [
    /(?:^|\n)\s*(?:Final Answer|Final Response|Answer|Response|Construct Response|Output)\s*:\s*(.+)/gi,
    /(?:^|\n)\s*\*+\s*(?:Final Answer|Final Response|Answer|Response|Construct Response|Output)\s*:\s*(.+)/gi,
  ];

  let candidate = "";
  for (const pattern of patterns) {
    let match = pattern.exec(cleanText);
    while (match) {
      candidate = String(match[1] || "").trim();
      match = pattern.exec(cleanText);
    }
    if (candidate) {
      break;
    }
  }

  return candidate.replace(/^["']+|["']+$/g, "").trim();
}

function looksLikeReasoningLeak(content, reasoning = "") {
  const cleanContent = String(content || "").trim();
  const cleanReasoning = String(reasoning || "").trim();
  if (!cleanContent) {
    return false;
  }
  if (cleanReasoning && cleanContent === cleanReasoning) {
    return true;
  }
  if (/^thinking process:/i.test(cleanContent)) {
    return true;
  }
  return Boolean(cleanReasoning) && /(analyze the request|final decision|construct response|instruction priority)/i.test(cleanContent);
}

function getDisplayMessageContent(item) {
  const content = String(item.content || "").trim();
  const reasoning = String(item.reasoning || "").trim();
  if (item.role !== "assistant") {
    return content;
  }
  if (!reasoning) {
    return content;
  }
  if (!content || looksLikeReasoningLeak(content, reasoning)) {
    return extractFinalAnswer(reasoning);
  }
  return content;
}

function getActiveProfile() {
  return state.profiles.find((profile) => profile.id === state.activeProfileId) || null;
}

function getFeedbackDraft(turnId) {
  if (!feedbackDrafts.has(turnId)) {
    feedbackDrafts.set(turnId, { score: null, note: "" });
  }
  return feedbackDrafts.get(turnId);
}

function getFeedbackSummary(feedback) {
  if (!feedback) {
    return null;
  }
  const score = Number(feedback.score ?? NaN);
  if (Number.isFinite(score)) {
    return `${score}/10`;
  }
  return "Rated";
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

function formatSessionTime(timestamp) {
  const value = Number(timestamp || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "";
  }
  const date = new Date(value * 1000);
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  if (sameDay) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
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

function autoResizeComposer() {
  if (!els.promptInput) {
    return;
  }
  els.promptInput.style.height = "auto";
  const nextHeight = Math.min(Math.max(els.promptInput.scrollHeight, 24), 160);
  els.promptInput.style.height = `${nextHeight}px`;
}

function autoResizeHereticComposer() {
  if (!els.hereticPromptInput) {
    return;
  }
  els.hereticPromptInput.style.height = "auto";
  const nextHeight = Math.min(Math.max(els.hereticPromptInput.scrollHeight, 24), 160);
  els.hereticPromptInput.style.height = `${nextHeight}px`;
}

function isContainerNearBottom(container) {
  if (!container) {
    return true;
  }
  const threshold = 96;
  const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
  return distanceFromBottom <= threshold;
}

function isTranscriptNearBottom() {
  return isContainerNearBottom(els.transcript);
}

function isHereticTranscriptNearBottom() {
  return isContainerNearBottom(els.hereticTranscript);
}

function scrollContainerToBottom(container) {
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

function scrollTranscriptToBottom() {
  scrollContainerToBottom(els.transcript);
}

function scrollHereticTranscriptToBottom() {
  scrollContainerToBottom(els.hereticTranscript);
}

function renderMessageList(container, emptyState, items, { shouldStick = false, messageOptions = {} } = {}) {
  if (!container || !emptyState) {
    return;
  }
  container.innerHTML = "";
  const hasMessages = items.length > 0;
  emptyState.classList.toggle("hidden", hasMessages);
  for (const item of items) {
    container.appendChild(buildTranscriptMessage(item, messageOptions));
  }
  if (shouldStick) {
    scrollContainerToBottom(container);
  }
}

function syncTranscriptBusyState() {
  if (!els.transcript) {
    return;
  }
  for (const input of els.transcript.querySelectorAll(".message-feedback-input")) {
    input.disabled = state.busy;
  }
  for (const button of els.transcript.querySelectorAll(".message-feedback-send, .score-pill")) {
    button.disabled = state.busy;
  }
}

function setBusy(isBusy) {
  state.busy = isBusy;
  if (!isBusy) {
    state.stopRequested = false;
  }
  document.body.classList.toggle("busy", isBusy || state.hereticBusy);
  renderControls();
  syncTranscriptBusyState();
  renderSessions();
}

function setHereticBusy(isBusy) {
  state.hereticBusy = isBusy;
  if (!isBusy) {
    state.hereticStopRequested = false;
  }
  document.body.classList.toggle("busy", isBusy || state.busy);
  renderControls();
}

function upsertStreamingTurn(turn) {
  const existingIndex = state.transcript.findIndex((item) => item.id === turn.id);
  if (existingIndex === -1) {
    state.transcript.push(turn);
    return;
  }
  Object.assign(state.transcript[existingIndex], turn);
}

function hasStreamMetrics(metrics) {
  return Boolean(metrics && typeof metrics === "object" && Object.keys(metrics).length > 0);
}

function appendStreamingDelta(turnId, delta, reasoningDelta = "", metrics = null) {
  const existingIndex = state.transcript.findIndex((item) => item.id === turnId);
  if (existingIndex === -1) {
    return;
  }
  const current = state.transcript[existingIndex];
  const nextMetrics = hasStreamMetrics(metrics)
    ? { ...(current.metrics && typeof current.metrics === "object" ? current.metrics : {}), ...metrics }
    : current.metrics || null;
  state.transcript[existingIndex] = {
    ...current,
    content: `${current.content || ""}${delta || ""}`,
    reasoning: `${current.reasoning || ""}${reasoningDelta || ""}`,
    metrics: nextMetrics,
    streaming: true,
  };
}

function setStreamingMetrics(turnId, metrics = null) {
  if (!hasStreamMetrics(metrics)) {
    return;
  }
  const existingIndex = state.transcript.findIndex((item) => item.id === turnId);
  if (existingIndex === -1) {
    return;
  }
  state.transcript[existingIndex] = {
    ...state.transcript[existingIndex],
    metrics: {
      ...(state.transcript[existingIndex].metrics && typeof state.transcript[existingIndex].metrics === "object"
        ? state.transcript[existingIndex].metrics
        : {}),
      ...metrics,
    },
  };
}

function clearStreamingTurns() {
  state.transcript = state.transcript.filter((item) => !item.ephemeral);
}

function finalizeStoppedStream() {
  const persistedTurns = state.transcript.filter((item) => !item.ephemeral);
  const userTurn = state.transcript.find((item) => item.ephemeral && item.role === "user");
  const assistantTurn = state.transcript.find((item) => item.ephemeral && item.role === "assistant");
  const hasAssistantContent = Boolean(
    assistantTurn && (String(assistantTurn.content || "").trim() || String(assistantTurn.reasoning || "").trim())
  );

  if (userTurn) {
    persistedTurns.push({ ...userTurn, ephemeral: false });
  }
  if (assistantTurn && hasAssistantContent) {
    persistedTurns.push({
      ...assistantTurn,
      ephemeral: false,
      streaming: false,
      feedback_pending: false,
      stopped: true,
    });
  }

  state.transcript = persistedTurns;
  setBusy(false);
  renderTranscript();
}

function finalizeStoppedHereticStream() {
  const persistedTurns = state.hereticTranscript.filter((item) => !item.ephemeral);
  const userTurn = state.hereticTranscript.find((item) => item.ephemeral && item.role === "user");
  const assistantTurn = state.hereticTranscript.find((item) => item.ephemeral && item.role === "assistant");
  const hasAssistantContent = Boolean(
    assistantTurn && (String(assistantTurn.content || "").trim() || String(assistantTurn.reasoning || "").trim())
  );

  if (userTurn) {
    persistedTurns.push({ ...userTurn, ephemeral: false });
  }
  if (assistantTurn && hasAssistantContent) {
    persistedTurns.push({
      ...assistantTurn,
      ephemeral: false,
      streaming: false,
      feedback_pending: false,
      stopped: true,
    });
  }

  state.hereticTranscript = persistedTurns;
  setHereticBusy(false);
  renderHereticTranscript();
}

function updateProfileLabels() {
  const activeProfile = getActiveProfile();
  const profileLabel = activeProfile ? activeProfile.name : state.activeProfileId || "Default";
  if (els.chatActiveProfile) {
    els.chatActiveProfile.textContent = profileLabel;
  }
  if (els.notesProfileLabel) {
    els.notesProfileLabel.textContent = activeProfile ? `Profile: ${profileLabel}` : "Profile";
  }
  if (els.settingsActiveProfile) {
    els.settingsActiveProfile.textContent = activeProfile ? `Active: ${profileLabel}` : "Profile";
  }
}

function getAppMeta(appId = state.activeApp) {
  return APP_REGISTRY[appId] || APP_REGISTRY.chat;
}

function getActiveModeModelLabel() {
  if (state.gpuMode === "heretic_chat") {
    return state.hereticModel || "27B sidecar model";
  }
  return state.model || "Model";
}

function renderShell() {
  const appMeta = getAppMeta();
  const chatActive = state.activeApp === "chat";
  const showProfilePill = ["chat", "notes", "profiles"].includes(state.activeApp);
  const keyboardScrollApps = new Set(["calendar", "notes", "profiles", "model-control"]);

  if (els.windowModelEyebrow) {
    els.windowModelEyebrow.textContent = appMeta.eyebrow;
  }
  if (els.windowModelTitle) {
    els.windowModelTitle.textContent = appMeta.title;
  }
  if (els.windowAppSummary) {
    els.windowAppSummary.textContent = appMeta.summary;
  }
  if (els.thinkingToggle) {
    els.thinkingToggle.classList.toggle("hidden", !chatActive);
  }
  if (els.chatActiveProfile) {
    els.chatActiveProfile.classList.toggle("hidden", !showProfilePill);
  }

  for (const button of els.appLauncherButtons) {
    const isActive = button.dataset.openApp === state.activeApp;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  }

  for (const view of els.appViews) {
    const isActive = view.dataset.appView === state.activeApp;
    view.classList.toggle("hidden", !isActive);
    view.setAttribute("aria-hidden", isActive ? "false" : "true");
    view.tabIndex = isActive && keyboardScrollApps.has(state.activeApp) ? 0 : -1;
  }
}

function focusActiveAppView() {
  if (["chat", "heretic"].includes(state.activeApp)) {
    return;
  }
  const activeView = els.appViews.find((view) => view.dataset.appView === state.activeApp);
  if (!activeView) {
    return;
  }
  window.requestAnimationFrame(() => {
    activeView.focus({ preventScroll: true });
  });
}

function activateApp(appId) {
  if (!APP_REGISTRY[appId]) {
    return;
  }
  state.activeApp = appId;
  saveActiveApp();
  renderShell();
  renderControls();
  renderStatus();
  if (appId === "calendar") {
    renderCalendar();
  }
  if (appId === "heretic") {
    renderHereticTranscript();
  }
  if (appId === "model-control") {
    renderModelControl();
  }
  focusActiveAppView();
}

function getCalendarEventsForDay(dateKey) {
  const events = state.calendarEvents[dateKey];
  return Array.isArray(events) ? events : [];
}

function renderCalendarSelectedDay() {
  if (!els.calendarSelectedLabel || !els.calendarSelectedMeta || !els.calendarDayEvents) {
    return;
  }

  const selectedDate = parseDateKey(state.calendarSelectedDateKey);
  const dateKey = state.calendarSelectedDateKey;
  const events = getCalendarEventsForDay(dateKey);

  els.calendarSelectedLabel.textContent = selectedDate.toLocaleDateString([], {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
  els.calendarSelectedMeta.textContent = `${events.length} event${events.length === 1 ? "" : "s"}`;
  els.calendarDayEvents.innerHTML = "";

  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "calendar-empty";
    const copy = document.createElement("p");
    copy.textContent = "No events saved for this day yet. Add one below to prove the shell is a real workspace.";
    empty.appendChild(copy);
    els.calendarDayEvents.appendChild(empty);
    return;
  }

  for (const event of events) {
    const card = document.createElement("article");
    card.className = "calendar-event-card";
    const title = document.createElement("h4");
    title.textContent = event.title || "Untitled event";
    const note = document.createElement("p");
    note.textContent = event.note || "No extra notes.";
    card.append(title, note);
    els.calendarDayEvents.appendChild(card);
  }
}

function renderCalendar() {
  if (!els.calendarGrid || !els.calendarMonthLabel) {
    return;
  }

  const anchor = startOfMonth(parseDateKey(state.calendarMonthAnchor));
  const monthStart = startOfMonth(anchor);
  const firstVisibleDate = new Date(monthStart);
  firstVisibleDate.setDate(monthStart.getDate() - monthStart.getDay());

  els.calendarMonthLabel.textContent = monthStart.toLocaleDateString([], {
    month: "long",
    year: "numeric",
  });
  els.calendarGrid.innerHTML = "";

  const todayKey = todayDateKey();
  for (let offset = 0; offset < 42; offset += 1) {
    const date = new Date(firstVisibleDate);
    date.setDate(firstVisibleDate.getDate() + offset);
    const dateKey = formatDateKey(date);
    const events = getCalendarEventsForDay(dateKey);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "calendar-day-button";
    button.classList.toggle("outside-month", date.getMonth() !== monthStart.getMonth());
    button.classList.toggle("selected", dateKey === state.calendarSelectedDateKey);
    button.classList.toggle("today", dateKey === todayKey);
    button.addEventListener("click", () => {
      state.calendarSelectedDateKey = dateKey;
      saveCalendarState();
      renderCalendar();
    });

    const number = document.createElement("span");
    number.className = "calendar-day-number";
    number.textContent = String(date.getDate());
    const count = document.createElement("span");
    count.className = "calendar-day-note-count";
    count.textContent = events.length ? `${events.length} saved` : "Open day";
    button.append(number, count);
    els.calendarGrid.appendChild(button);
  }

  renderCalendarSelectedDay();
}

function shiftCalendarMonth(offset) {
  const currentAnchor = startOfMonth(parseDateKey(state.calendarMonthAnchor));
  const nextAnchor = new Date(currentAnchor.getFullYear(), currentAnchor.getMonth() + offset, 1);
  state.calendarMonthAnchor = formatDateKey(nextAnchor);
  saveCalendarState();
  renderCalendar();
}

function saveCalendarEvent() {
  const title = String(els.calendarEventTitle?.value || "").trim();
  const note = String(els.calendarEventNote?.value || "").trim();
  if (!title && !note) {
    return;
  }

  const nextEvents = [...getCalendarEventsForDay(state.calendarSelectedDateKey)];
  nextEvents.push({
    id: `${Date.now()}`,
    title: title || "Untitled event",
    note,
  });
  state.calendarEvents[state.calendarSelectedDateKey] = nextEvents;
  saveCalendarState();
  if (els.calendarEventTitle) {
    els.calendarEventTitle.value = "";
  }
  if (els.calendarEventNote) {
    els.calendarEventNote.value = "";
  }
  renderCalendar();
}

function clearCalendarDay() {
  delete state.calendarEvents[state.calendarSelectedDateKey];
  saveCalendarState();
  if (els.calendarEventTitle) {
    els.calendarEventTitle.value = "";
  }
  if (els.calendarEventNote) {
    els.calendarEventNote.value = "";
  }
  renderCalendar();
}

function renderProfiles() {
  if (!els.profilesList || !els.profilesEmpty) {
    return;
  }

  const openclawLaneActive = !state.gpuMode || state.gpuMode === "openclaw_rl";
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
      `Prompt: ${formatBytes(profile.system_prompt_size_bytes ?? 0)}`,
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
    hint.className = "profile-hint";
    hint.textContent =
      profile.id === state.activeProfileId
        ? "Current profile for chats, system prompt, notes, and live updates."
        : "Switch to this profile's chats, system prompt, notes, and learned state.";

    const button = document.createElement("button");
    button.className = "profile-activate-button";
    button.type = "button";
    button.textContent = profile.id === state.activeProfileId ? "Current profile" : "Switch to profile";
    button.disabled = profile.id === state.activeProfileId || state.busy || !openclawLaneActive;
    button.addEventListener("click", () => {
      selectProfile(profile.id);
    });

    footer.append(hint, button);
    card.append(head, stats, footer);
    els.profilesList.appendChild(card);
  }

  updateProfileLabels();
}

function renderSessions() {
  if (!els.sessionList || !els.sessionsEmpty) {
    return;
  }

  const openclawLaneActive = !state.gpuMode || state.gpuMode === "openclaw_rl";
  els.sessionList.innerHTML = "";
  const sessions = Array.isArray(state.sessions) ? state.sessions : [];
  els.sessionsEmpty.classList.toggle("hidden", sessions.length > 0);

  for (const session of sessions) {
    const button = document.createElement("button");
    button.className = "session-item";
    button.type = "button";
    if (session.id === state.activeSessionId) {
      button.classList.add("active-session");
    }
    button.disabled = !openclawLaneActive || (state.busy && session.id !== state.activeSessionId);
    button.addEventListener("click", () => {
      selectSession(session.id);
    });

    const titleRow = document.createElement("div");
    titleRow.className = "session-item-head";

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = session.title || "New chat";

    const meta = document.createElement("span");
    meta.className = "session-meta";
    meta.textContent = formatSessionTime(session.updated_at);

    titleRow.append(title, meta);

    const preview = document.createElement("p");
    preview.className = "session-preview";
    const previewText = String(session.preview || "").trim();
    preview.textContent = previewText
      ? looksLikeReasoningLeak(previewText) ? extractFinalAnswer(previewText) || "Thinking trace only." : previewText
      : "No messages yet.";

    button.append(titleRow, preview);
    els.sessionList.appendChild(button);
  }
}

function renderStatus() {
  if (!els.proxyStatus) {
    return;
  }

  if (state.activeApp === "chat" && state.gpuMode && state.gpuMode !== "openclaw_rl") {
    els.proxyStatus.className = "status-chip status-bad";
    els.proxyStatus.textContent = "OpenClaw lane inactive";
    return;
  }

  if (state.activeApp === "heretic") {
    els.proxyStatus.className = "status-chip";
    if (!state.modelControl) {
      els.proxyStatus.textContent = "Checking 27B Chat...";
      return;
    }
    if (state.gpuMode !== "heretic_chat") {
      els.proxyStatus.textContent = "27B lane inactive";
      els.proxyStatus.classList.add("status-bad");
      return;
    }
    const hereticReady = Boolean(state.modelControl?.heretic?.ready);
    els.proxyStatus.textContent = hereticReady ? "27B chat connected" : "27B chat unavailable";
    els.proxyStatus.classList.add(hereticReady ? "status-good" : "status-bad");
    return;
  }

  if (state.activeApp === "model-control") {
    els.proxyStatus.className = "status-chip";
    if (!state.modelControl) {
      els.proxyStatus.textContent = "Checking GPU mode...";
      return;
    }
    els.proxyStatus.textContent = `GPU mode: ${state.gpuModeLabel || "Unknown"}`;
    els.proxyStatus.classList.add(
      state.gpuMode === "openclaw_rl" || state.gpuMode === "heretic_chat" ? "status-good" : "status-bad"
    );
    return;
  }

  if (state.activeApp !== "chat" && state.activeApp !== "heretic") {
    els.proxyStatus.className = "status-chip";
    if (!state.modelControl) {
      els.proxyStatus.textContent = "Checking GPU mode...";
      return;
    }
    if (state.gpuMode === "heretic_chat") {
      const hereticReady = Boolean(state.modelControl?.heretic?.ready);
      els.proxyStatus.textContent = hereticReady ? "27B chat connected" : "27B chat unavailable";
      els.proxyStatus.classList.add(hereticReady ? "status-good" : "status-bad");
      return;
    }
    if (state.gpuMode && state.gpuMode !== "openclaw_rl") {
      els.proxyStatus.textContent = `GPU mode: ${state.gpuModeLabel || "Unknown"}`;
      els.proxyStatus.classList.add(
        state.gpuMode === "openclaw_rl" || state.gpuMode === "heretic_chat" ? "status-good" : "status-bad"
      );
      return;
    }
  }

  const splitMode =
    state.backendMode === "split_trainer_api" ||
    state.backendMode === "split_backends" ||
    state.backendMode === "sglang_trainer" ||
    state.chatBackendMode !== state.trainingBackendMode ||
    state.trainingProxyBaseUrl !== state.proxyBaseUrl;
  const backendLabel = splitMode
    ? "SGLang chat + trainer"
    : state.backendMode === "rl_proxy"
      ? "Proxy"
      : state.backendMode === "trainer_api"
        ? "Trainer"
        : "Backend";

  els.proxyStatus.className = "status-chip";

  if (state.proxyOk === null) {
    els.proxyStatus.textContent = `Checking ${backendLabel.toLowerCase()}...`;
    return;
  }

  if (state.proxyOk) {
    els.proxyStatus.textContent = `${backendLabel} connected`;
    els.proxyStatus.classList.add("status-good");
    return;
  }

  if (splitMode && state.proxyHealth) {
    const chatOk = Boolean(state.proxyHealth.chat?.ok);
    const trainingOk = Boolean(state.proxyHealth.training?.ok);
    if (!chatOk && trainingOk) {
      els.proxyStatus.textContent = "SGLang unavailable";
    } else if (chatOk && !trainingOk) {
      els.proxyStatus.textContent = "Trainer unavailable";
    } else {
      els.proxyStatus.textContent = `${backendLabel} unavailable`;
    }
  } else {
    els.proxyStatus.textContent = `${backendLabel} unavailable`;
  }
  els.proxyStatus.classList.add("status-bad");
}

function renderControls() {
  const disabled = state.busy;
  const hereticDisabled = state.hereticBusy;
  const openclawLaneActive = !state.gpuMode || state.gpuMode === "openclaw_rl";
  const hereticLaneActive = !state.gpuMode || state.gpuMode === "heretic_chat";
  if (state.busy) {
    els.sendButton.disabled = state.stopRequested;
    els.sendButton.setAttribute("aria-label", state.stopRequested ? "Stopping response" : "Stop generating");
    els.sendButton.classList.add("stop-mode");
    els.sendButton.classList.toggle("stop-pending", state.stopRequested);
    if (els.sendButtonIcon) {
      els.sendButtonIcon.textContent = state.stopRequested ? "…" : "■";
    }
  } else {
    els.sendButton.disabled = !openclawLaneActive || !els.promptInput.value.trim();
    els.sendButton.setAttribute("aria-label", "Send message");
    els.sendButton.classList.remove("stop-mode", "stop-pending");
    if (els.sendButtonIcon) {
      els.sendButtonIcon.textContent = "➤";
    }
  }
  els.promptInput.disabled = disabled || !openclawLaneActive;
  if (els.hereticSendButton) {
    if (state.hereticBusy) {
      els.hereticSendButton.disabled = state.hereticStopRequested;
      els.hereticSendButton.setAttribute(
        "aria-label",
        state.hereticStopRequested ? "Stopping 27B response" : "Stop generating"
      );
      els.hereticSendButton.classList.add("stop-mode");
      els.hereticSendButton.classList.toggle("stop-pending", state.hereticStopRequested);
      if (els.hereticSendButtonIcon) {
        els.hereticSendButtonIcon.textContent = state.hereticStopRequested ? "…" : "■";
      }
    } else {
      els.hereticSendButton.disabled = !hereticLaneActive || !els.hereticPromptInput.value.trim();
      els.hereticSendButton.setAttribute("aria-label", "Send 27B chat message");
      els.hereticSendButton.classList.remove("stop-mode", "stop-pending");
      if (els.hereticSendButtonIcon) {
        els.hereticSendButtonIcon.textContent = "➤";
      }
    }
  }
  if (els.hereticPromptInput) {
    els.hereticPromptInput.disabled = hereticDisabled || !hereticLaneActive;
  }
  if (els.hereticResetButton) {
    els.hereticResetButton.disabled = hereticDisabled || disabled;
  }
  els.systemPromptInput.disabled = disabled || !openclawLaneActive;
  els.guidanceInput.disabled = disabled || !openclawLaneActive;
  els.guidanceSaveButton.disabled = disabled || !openclawLaneActive;
  els.newSessionButton.disabled = disabled || !openclawLaneActive;
  const connectedModel =
    state.activeApp === "chat"
      ? state.model || "Model"
      : state.activeApp === "heretic"
        ? state.hereticModel || "27B sidecar model"
        : getActiveModeModelLabel();
  els.modelLabel.textContent = connectedModel;
  if (els.gpuModePill) {
    els.gpuModePill.textContent = state.gpuModeLabel ? `GPU mode: ${state.gpuModeLabel}` : "GPU mode";
  }
  if (els.thinkingToggle) {
    els.thinkingToggle.disabled = disabled || !openclawLaneActive;
    els.thinkingToggle.classList.toggle("active", state.thinkingEnabled);
    els.thinkingToggle.setAttribute("aria-pressed", state.thinkingEnabled ? "true" : "false");
    els.thinkingToggle.textContent = state.thinkingEnabled ? "Thinking on" : "Thinking off";
  }

  if (els.profileNameInput) {
    els.profileNameInput.disabled = disabled || !openclawLaneActive;
  }
  if (els.profileCreateButton) {
    els.profileCreateButton.disabled = disabled || !openclawLaneActive || !els.profileNameInput.value.trim();
  }

  updateProfileLabels();
  autoResizeComposer();
  autoResizeHereticComposer();
}

function renderMessageMetrics(message, item) {
  const metricsWrap = message.querySelector(".message-stream-stats");
  const tokenCount = message.querySelector(".stream-token-count");
  const speed = message.querySelector(".stream-speed");
  const metrics = item.metrics && typeof item.metrics === "object" ? item.metrics : null;
  const visibleTokens = Number(metrics?.visible_tokens ?? metrics?.generated_tokens ?? NaN);
  const tokensPerSecond = Number(metrics?.tokens_per_second ?? NaN);
  const shouldShow = item.streaming || (Number.isFinite(visibleTokens) && visibleTokens > 0);

  metricsWrap.classList.toggle("hidden", !shouldShow);
  if (!shouldShow) {
    return;
  }

  tokenCount.textContent = Number.isFinite(visibleTokens) && visibleTokens > 0 ? `${visibleTokens} tok` : "";
  speed.textContent = Number.isFinite(tokensPerSecond) && tokensPerSecond > 0 ? `${tokensPerSecond.toFixed(1)} tok/s` : "";
}

function buildTranscriptMessage(item, { allowFeedback = true } = {}) {
  const fragment = els.messageTemplate.content.cloneNode(true);
  const message = fragment.querySelector(".message");
  const roleLabel = message.querySelector(".role-label");
  const body = message.querySelector(".message-body");
  const feedbackPill = message.querySelector(".feedback-pill");
  const reasoningBlock = message.querySelector(".reasoning-block");
  const reasoningBody = message.querySelector(".reasoning-body");
  const feedbackControls = message.querySelector(".message-feedback-controls");
  const scorePills = message.querySelector(".message-score-pills");
  const scoreSummary = message.querySelector(".message-rating-summary");
  const feedbackInput = message.querySelector(".message-feedback-input");
  const feedbackSendButton = message.querySelector(".message-feedback-send");
  const feedbackSummary = message.querySelector(".message-feedback-summary");

  message.classList.add(item.role === "assistant" ? "assistant-message" : "user-message");
  if (item.id) {
    message.dataset.messageId = item.id;
  }
  if (item.role === "assistant") {
    message.dataset.turnId = item.id || "";
  }
  if (item.streaming) {
    message.classList.add("streaming-message");
  }

  roleLabel.textContent = item.role === "assistant" ? "Assistant" : "You";
  const displayContent = getDisplayMessageContent(item);
  if (item.role === "assistant" && !displayContent && item.reasoning) {
    if (item.streaming) {
      body.innerHTML = "";
    } else if (item.stopped) {
      body.innerHTML =
        '<p class="assistant-placeholder-copy">Generation stopped before a final answer was produced. Open Thinking to inspect the partial trace.</p>';
    } else {
      body.innerHTML = '<p class="assistant-placeholder-copy">Final answer missing. Open Thinking to inspect the reasoning trace.</p>';
    }
  } else if (item.role === "assistant" && !displayContent && item.stopped) {
    body.innerHTML =
      '<p class="assistant-placeholder-copy">Generation stopped before any visible answer was produced.</p>';
  } else {
    body.innerHTML = renderMarkdownLite(displayContent);
  }

  if (item.reasoning) {
    reasoningBlock.classList.remove("hidden");
    reasoningBlock.open = Boolean(item.streaming);
    reasoningBody.textContent = item.reasoning;
  }

  renderMessageMetrics(message, item);

  if (!allowFeedback) {
    feedbackControls.remove();
    feedbackPill.remove();
  } else if (item.role !== "assistant") {
    feedbackControls.classList.add("hidden");
  } else if (item.streaming) {
    feedbackControls.classList.add("hidden");
    feedbackPill.classList.remove("hidden");
    feedbackPill.classList.add("neutral-pill");
    feedbackPill.textContent = "Streaming";
  } else if (item.feedback) {
    const summary = getFeedbackSummary(item.feedback);
    feedbackPill.classList.remove("hidden");
    feedbackPill.classList.add("good-pill");
    feedbackPill.textContent = summary;
    feedbackSummary.classList.remove("hidden");
    feedbackSummary.textContent = item.feedback.note
      ? `Rated ${summary} for training. Note: ${item.feedback.note}`
      : `Rated ${summary} for training.`;
  } else if (item.stopped) {
    feedbackControls.classList.add("hidden");
    feedbackPill.classList.remove("hidden");
    feedbackPill.classList.add("neutral-pill");
    feedbackPill.textContent = "Stopped";
  } else if (item.feedback_pending !== false) {
    feedbackControls.classList.remove("hidden");
    const draft = getFeedbackDraft(item.id);
    scoreSummary.textContent = draft.score ? `Current score: ${draft.score}/10` : "Choose a score from 1 to 10";
    feedbackInput.value = draft.note;
    feedbackInput.disabled = state.busy;
    feedbackSendButton.disabled = state.busy || !draft.score;

    for (let score = 1; score <= 10; score += 1) {
      const button = document.createElement("button");
      button.className = "score-pill";
      button.type = "button";
      button.textContent = String(score);
      button.classList.toggle("selected", draft.score === score);
      button.disabled = state.busy;
      button.addEventListener("click", () => {
        const nextDraft = getFeedbackDraft(item.id);
        nextDraft.score = nextDraft.score === score ? null : score;
        patchTranscriptTurn(item.id);
      });
      scorePills.appendChild(button);
    }

    feedbackInput.addEventListener("input", (event) => {
      getFeedbackDraft(item.id).note = event.target.value;
    });

    feedbackSendButton.addEventListener("click", () => {
      sendFeedback(item.id);
    });
  } else {
    feedbackControls.classList.add("hidden");
  }

  return message;
}

function patchTranscriptTurn(turnId) {
  if (!turnId || !els.transcript) {
    renderTranscript();
    return;
  }
  const item = state.transcript.find((entry) => entry.id === turnId);
  if (!item) {
    renderTranscript();
    return;
  }
  const selector = `.message[data-message-id="${window.CSS && typeof window.CSS.escape === "function" ? window.CSS.escape(turnId) : turnId}"]`;
  const existing = els.transcript.querySelector(selector);
  if (!existing) {
    renderTranscript();
    return;
  }
  const shouldStick = isTranscriptNearBottom();
  existing.replaceWith(buildTranscriptMessage(item));
  if (shouldStick) {
    scrollTranscriptToBottom();
  }
}

function renderTranscript() {
  const shouldStick = !els.transcript.childElementCount || isTranscriptNearBottom();
  renderMessageList(els.transcript, els.emptyState, state.transcript, { shouldStick });
}

function patchHereticTranscriptTurn(turnId) {
  if (!turnId || !els.hereticTranscript) {
    renderHereticTranscript();
    return;
  }
  const item = state.hereticTranscript.find((entry) => entry.id === turnId);
  if (!item) {
    renderHereticTranscript();
    return;
  }
  const selector = `.message[data-message-id="${window.CSS && typeof window.CSS.escape === "function" ? window.CSS.escape(turnId) : turnId}"]`;
  const existing = els.hereticTranscript.querySelector(selector);
  if (!existing) {
    renderHereticTranscript();
    return;
  }
  const shouldStick = isHereticTranscriptNearBottom();
  existing.replaceWith(buildTranscriptMessage(item, { allowFeedback: false }));
  if (shouldStick) {
    scrollHereticTranscriptToBottom();
  }
}

function renderHereticTranscript() {
  if (!els.hereticTranscript || !els.hereticEmptyState) {
    return;
  }
  const shouldStick = !els.hereticTranscript.childElementCount || isHereticTranscriptNearBottom();
  renderMessageList(els.hereticTranscript, els.hereticEmptyState, state.hereticTranscript, {
    shouldStick,
    messageOptions: { allowFeedback: false },
  });
}

function applyModelControlState(modelControlPayload) {
  if (!modelControlPayload || typeof modelControlPayload !== "object") {
    return;
  }
  state.modelControl = modelControlPayload;
  state.gpuMode = modelControlPayload.active_mode || "";
  state.gpuModeLabel = modelControlPayload.mode_label || "";
  state.hereticModel = modelControlPayload.heretic?.model || state.hereticModel || "";
  renderModelControl();
  renderControls();
  renderStatus();
}

function renderModelControl() {
  if (!els.modelControlModePill) {
    return;
  }

  const modelControl = state.modelControl;
  const openclawLane = modelControl?.openclaw || null;
  const hereticLane = modelControl?.heretic || null;
  const openclawActive = state.gpuMode === "openclaw_rl";
  const hereticActive = state.gpuMode === "heretic_chat";

  els.modelControlModePill.textContent = state.gpuModeLabel ? `Active: ${state.gpuModeLabel}` : "Checking mode...";

  if (els.openclawModeCard) {
    els.openclawModeCard.classList.toggle("active-mode-card", openclawActive);
  }
  if (els.hereticModeCard) {
    els.hereticModeCard.classList.toggle("active-mode-card", hereticActive);
  }

  if (els.openclawModeStatus) {
    els.openclawModeStatus.textContent = openclawActive ? "Active" : "Inactive";
  }
  if (els.hereticModeStatus) {
    els.hereticModeStatus.textContent = hereticActive ? "Active" : "Inactive";
  }
  if (els.openclawModeModel) {
    els.openclawModeModel.textContent = openclawLane?.model ? `Model: ${openclawLane.model}` : "Model: unavailable";
  }
  if (els.hereticModeModel) {
    els.hereticModeModel.textContent = hereticLane?.model ? `Model: ${hereticLane.model}` : "Model: unavailable";
  }
  if (els.openclawModeHealth) {
    els.openclawModeHealth.textContent = `Health: ${openclawLane?.ready ? "ready" : "unavailable"}`;
  }
  if (els.hereticModeHealth) {
    els.hereticModeHealth.textContent = `Health: ${hereticLane?.ready ? "ready" : "unavailable"}`;
  }
  if (els.openclawModeButton) {
    els.openclawModeButton.disabled = state.busy || state.hereticBusy || openclawActive;
    els.openclawModeButton.textContent = openclawActive ? "OpenClaw RL active" : "Use OpenClaw RL";
  }
  if (els.hereticModeButton) {
    els.hereticModeButton.disabled = state.busy || state.hereticBusy || hereticActive;
    els.hereticModeButton.textContent = hereticActive ? "27B Chat active" : "Use 27B Chat";
  }
  if (els.hereticStatusChip) {
    if (!modelControl) {
      els.hereticStatusChip.textContent = "Checking 27B Chat...";
    } else if (state.gpuMode !== "heretic_chat") {
      els.hereticStatusChip.textContent = "Mode set to OpenClaw RL";
    } else {
      els.hereticStatusChip.textContent = hereticLane?.ready ? "27B chat connected" : "27B chat unavailable";
    }
  }
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

function applyServerState(serverState, profilePayload = null) {
  state.transcript = serverState.transcript || [];
  state.hereticTranscript = serverState.heretic_transcript || [];
  state.busy = Boolean(serverState.busy);
  state.hereticBusy = Boolean(serverState.heretic_busy);
  state.stopRequested = Boolean(serverState.stop_requested) && state.busy;
  state.hereticStopRequested = Boolean(serverState.heretic_stop_requested) && state.hereticBusy;
  document.body.classList.toggle("busy", state.busy || state.hereticBusy);
  state.model = serverState.model || "";
  state.hereticModel = serverState.heretic_model || state.hereticModel || "";
  state.backendMode = serverState.backend_mode || "rl_proxy";
  state.chatBackendMode = serverState.chat_backend_mode || state.backendMode;
  state.trainingBackendMode = serverState.training_backend_mode || state.backendMode;
  state.proxyBaseUrl = serverState.proxy_base_url || "";
  state.trainingProxyBaseUrl = serverState.training_proxy_base_url || state.proxyBaseUrl || "";
  state.activeProfileId = serverState.active_profile_id || state.activeProfileId || "";
  state.activeSessionId = serverState.active_session_id || "";
  state.sessions = Array.isArray(serverState.sessions) ? serverState.sessions : [];
  state.guidanceText = serverState.guidance_text || "";
  state.systemPromptText = serverState.system_prompt_text || "";
  state.thinkingEnabled = Boolean(serverState.thinking_enabled);

  if (els.systemPromptInput.value !== state.systemPromptText) {
    els.systemPromptInput.value = state.systemPromptText;
  }
  if (els.guidanceInput.value !== state.guidanceText) {
    els.guidanceInput.value = state.guidanceText;
  }

  setGuidanceStatus(
    state.systemPromptText || state.guidanceText
      ? "Saved system prompt and notes are active for this profile."
      : "No saved system prompt or notes yet."
  );

  if (serverState.proxy) {
    state.proxyHealth = serverState.proxy;
    state.proxyOk = Boolean(serverState.proxy.ok);
  }

  if (serverState.model_control) {
    applyModelControlState(serverState.model_control);
  }

  renderTranscript();
  renderHereticTranscript();
  renderSessions();
  renderControls();
  renderStatus();
  if (profilePayload) {
    applyProfilesState(profilePayload);
  } else {
    updateProfileLabels();
  }
  renderShell();
  renderCalendar();
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
  const [statusPayload, profilesPayload] = await Promise.all([
    fetchJson("/api/status"),
    fetchJson("/api/profiles"),
  ]);
  applyServerState(statusPayload.state, profilesPayload);
}

async function refreshProfiles() {
  const payload = await fetchJson("/api/profiles");
  applyServerState(payload.state, payload);
}

async function refreshModelControl() {
  const payload = await fetchJson("/api/model-control");
  applyModelControlState(payload.model_control);
  if (payload.state) {
    applyServerState(payload.state);
  }
}

async function sendPrompt() {
  const prompt = els.promptInput.value.trim();
  if (!prompt || state.busy) {
    return;
  }

  state.stopRequested = false;
  setBusy(true);
  const controller = new AbortController();
  activeStreamController = controller;
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ prompt, thinking_enabled: state.thinkingEnabled }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = payload.detail || `Request failed with ${response.status}`;
      throw new Error(detail);
    }

    els.promptInput.value = "";
    autoResizeComposer();
    renderControls();

    await readSseStream(response, (eventName, payload) => {
      if (eventName === "start") {
        if (payload.user_turn) {
          upsertStreamingTurn({ ...payload.user_turn, ephemeral: true });
        }
        if (payload.assistant_turn) {
          upsertStreamingTurn({ ...payload.assistant_turn, ephemeral: true });
        }
        renderTranscript();
        return;
      }

      if (eventName === "delta") {
        appendStreamingDelta(
          payload.assistant_turn_id,
          payload.delta || "",
          payload.reasoning_delta || "",
          payload.metrics || null
        );
        patchTranscriptTurn(payload.assistant_turn_id);
        return;
      }

      if (eventName === "final") {
        setStreamingMetrics(payload.assistant_turn_id, payload.metrics || null);
        patchTranscriptTurn(payload.assistant_turn_id);
        return;
      }

      if (eventName === "state") {
        applyServerState(payload.state);
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
    if (error.name === "AbortError" && state.stopRequested) {
      finalizeStoppedStream();
      return;
    }
    try {
      await refreshState();
    } catch (_refreshError) {
      // Keep original error.
    }
    window.alert(error.message);
    setBusy(false);
  } finally {
    if (activeStreamController === controller) {
      activeStreamController = null;
    }
  }
}

async function stopPrompt() {
  if (!state.busy || state.stopRequested) {
    return;
  }

  state.stopRequested = true;
  renderControls();
  try {
    fetch("/api/chat/stop", { method: "POST", credentials: "same-origin", keepalive: true }).catch(() => {});
    if (activeStreamController) {
      activeStreamController.abort();
    }
  } catch (error) {
    state.stopRequested = false;
    renderControls();
    window.alert(error.message);
  }
}

function upsertHereticStreamingTurn(turn) {
  const existingIndex = state.hereticTranscript.findIndex((item) => item.id === turn.id);
  if (existingIndex === -1) {
    state.hereticTranscript.push(turn);
    return;
  }
  Object.assign(state.hereticTranscript[existingIndex], turn);
}

function appendHereticStreamingDelta(turnId, delta, reasoningDelta = "", metrics = null) {
  const existingIndex = state.hereticTranscript.findIndex((item) => item.id === turnId);
  if (existingIndex === -1) {
    return;
  }
  const current = state.hereticTranscript[existingIndex];
  const nextMetrics = hasStreamMetrics(metrics)
    ? { ...(current.metrics && typeof current.metrics === "object" ? current.metrics : {}), ...metrics }
    : current.metrics || null;
  state.hereticTranscript[existingIndex] = {
    ...current,
    content: `${current.content || ""}${delta || ""}`,
    reasoning: `${current.reasoning || ""}${reasoningDelta || ""}`,
    metrics: nextMetrics,
    streaming: true,
  };
}

function setHereticStreamingMetrics(turnId, metrics = null) {
  if (!hasStreamMetrics(metrics)) {
    return;
  }
  const existingIndex = state.hereticTranscript.findIndex((item) => item.id === turnId);
  if (existingIndex === -1) {
    return;
  }
  state.hereticTranscript[existingIndex] = {
    ...state.hereticTranscript[existingIndex],
    metrics: {
      ...(state.hereticTranscript[existingIndex].metrics && typeof state.hereticTranscript[existingIndex].metrics === "object"
        ? state.hereticTranscript[existingIndex].metrics
        : {}),
      ...metrics,
    },
  };
}

function clearHereticStreamingTurns() {
  state.hereticTranscript = state.hereticTranscript.filter((item) => !item.ephemeral);
}

async function sendHereticPrompt() {
  const prompt = els.hereticPromptInput.value.trim();
  if (!prompt || state.hereticBusy) {
    return;
  }

  state.hereticStopRequested = false;
  setHereticBusy(true);
  const controller = new AbortController();
  try {
    const response = await fetch("/api/heretic/chat/stream", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = payload.detail || `Request failed with ${response.status}`;
      throw new Error(detail);
    }

    els.hereticPromptInput.value = "";
    autoResizeHereticComposer();
    renderControls();

    await readSseStream(response, (eventName, payload) => {
      if (eventName === "start") {
        if (payload.user_turn) {
          upsertHereticStreamingTurn({ ...payload.user_turn, ephemeral: true, feedback_pending: false });
        }
        if (payload.assistant_turn) {
          upsertHereticStreamingTurn({ ...payload.assistant_turn, ephemeral: true, feedback_pending: false });
        }
        renderHereticTranscript();
        return;
      }

      if (eventName === "delta") {
        appendHereticStreamingDelta(
          payload.assistant_turn_id,
          payload.delta || "",
          payload.reasoning_delta || "",
          payload.metrics || null
        );
        patchHereticTranscriptTurn(payload.assistant_turn_id);
        return;
      }

      if (eventName === "final") {
        setHereticStreamingMetrics(payload.assistant_turn_id, payload.metrics || null);
        patchHereticTranscriptTurn(payload.assistant_turn_id);
        return;
      }

      if (eventName === "state") {
        applyServerState(payload.state);
        return;
      }

      if (eventName === "error") {
        clearHereticStreamingTurns();
        renderHereticTranscript();
        throw new Error(payload.detail || "Streaming failed.");
      }
    });

    if (state.hereticBusy) {
      await refreshState();
    }
  } catch (error) {
    if (error.name === "AbortError" && state.hereticStopRequested) {
      finalizeStoppedHereticStream();
      return;
    }
    try {
      await refreshState();
    } catch (_refreshError) {
      // Keep original error.
    }
    window.alert(error.message);
    setHereticBusy(false);
  }
}

async function stopHereticPrompt() {
  if (!state.hereticBusy || state.hereticStopRequested) {
    return;
  }

  state.hereticStopRequested = true;
  renderControls();
  try {
    fetch("/api/heretic/chat/stop", { method: "POST", credentials: "same-origin", keepalive: true }).catch(() => {});
  } catch (error) {
    state.hereticStopRequested = false;
    renderControls();
    window.alert(error.message);
  }
}

async function resetHereticChat() {
  if (state.busy || state.hereticBusy) {
    return;
  }
  setHereticBusy(true);
  try {
    const payload = await fetchJson("/api/heretic/reset", {
      method: "POST",
      body: "{}",
    });
    applyServerState(payload.state);
  } catch (error) {
    window.alert(error.message);
    setHereticBusy(false);
  }
}

async function switchGpuMode(mode) {
  if (!mode || state.busy || state.hereticBusy) {
    return;
  }

  document.body.classList.add("busy");
  try {
    const payload = await fetchJson("/api/model-control/switch", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    applyModelControlState(payload.model_control);
    applyServerState(payload.state);
    await refreshState();
  } catch (error) {
    window.alert(error.message);
    document.body.classList.toggle("busy", state.busy || state.hereticBusy);
  }
}

async function sendFeedback(assistantTurnId) {
  if (state.busy) {
    return;
  }

  const draft = getFeedbackDraft(assistantTurnId);
  if (!draft.score) {
    return;
  }

  setBusy(true);
  try {
    const payload = await fetchJson("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        assistant_turn_id: assistantTurnId,
        score: draft.score,
        note: draft.note.trim(),
      }),
    });
    feedbackDrafts.delete(assistantTurnId);
    applyServerState(payload.state);
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function saveGuidance() {
  setBusy(true);
  try {
    const payload = await fetchJson("/api/profile-settings", {
      method: "POST",
      body: JSON.stringify({
        guidance_text: els.guidanceInput.value,
        system_prompt_text: els.systemPromptInput.value,
      }),
    });
    applyServerState(payload.state);
    setGuidanceStatus(
      payload.state.system_prompt_text || payload.state.guidance_text
        ? "System prompt and notes saved for this profile."
        : "Saved system prompt and notes cleared."
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

  setBusy(true);
  setProfileCreateStatus("Switching profile...");
  try {
    const payload = await fetchJson("/api/profiles/select", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    els.promptInput.value = "";
    autoResizeComposer();
    feedbackDrafts.clear();
    applyServerState(payload.state, payload);
    setGuidanceStatus(
      payload.state.system_prompt_text || payload.state.guidance_text
        ? "Loaded saved system prompt and notes for this profile."
        : "No saved system prompt or notes yet for this profile."
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
    autoResizeComposer();
    feedbackDrafts.clear();
    applyServerState(payload.state, payload);
    setGuidanceStatus(
      payload.state.system_prompt_text || payload.state.guidance_text
        ? "Loaded saved system prompt and notes for the new profile."
        : "New profile created. Add a system prompt or notes whenever you want."
    );
    setProfileCreateStatus("Profile created and activated.");
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
    setProfileCreateStatus("Profile creation failed.");
  }
}

async function createSession() {
  if (state.busy) {
    return;
  }
  setBusy(true);
  try {
    const payload = await fetchJson("/api/sessions", {
      method: "POST",
      body: "{}",
    });
    els.promptInput.value = "";
    autoResizeComposer();
    feedbackDrafts.clear();
    applyServerState(payload.state);
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function toggleThinking() {
  if (state.busy || !els.thinkingToggle) {
    return;
  }

  setBusy(true);
  try {
    const payload = await fetchJson("/api/thinking", {
      method: "POST",
      body: JSON.stringify({ enabled: !state.thinkingEnabled }),
    });
    applyServerState(payload.state);
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

async function selectSession(sessionId) {
  if (!sessionId || sessionId === state.activeSessionId || state.busy) {
    return;
  }

  setBusy(true);
  try {
    const payload = await fetchJson("/api/sessions/select", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
    els.promptInput.value = "";
    autoResizeComposer();
    feedbackDrafts.clear();
    applyServerState(payload.state);
  } catch (error) {
    window.alert(error.message);
    setBusy(false);
  }
}

els.sendButton.addEventListener("click", () => {
  if (state.busy) {
    stopPrompt();
    return;
  }
  sendPrompt();
});
if (els.hereticSendButton) {
  els.hereticSendButton.addEventListener("click", () => {
    if (state.hereticBusy) {
      stopHereticPrompt();
      return;
    }
    sendHereticPrompt();
  });
}
els.guidanceSaveButton.addEventListener("click", saveGuidance);
els.newSessionButton.addEventListener("click", createSession);
if (els.thinkingToggle) {
  els.thinkingToggle.addEventListener("click", toggleThinking);
}

if (els.profileCreateButton) {
  els.profileCreateButton.addEventListener("click", createProfile);
}

els.promptInput.addEventListener("input", () => {
  renderControls();
});
if (els.hereticPromptInput) {
  els.hereticPromptInput.addEventListener("input", () => {
    renderControls();
  });
}

els.promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendPrompt();
  }
});
if (els.hereticPromptInput) {
  els.hereticPromptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendHereticPrompt();
    }
  });
}

if (els.systemPromptInput) {
  els.systemPromptInput.addEventListener("input", () => {
    setGuidanceStatus("Unsaved system prompt or notes changes.");
  });
}

els.guidanceInput.addEventListener("input", () => {
  setGuidanceStatus("Unsaved system prompt or notes changes.");
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

for (const button of els.appLauncherButtons) {
  button.addEventListener("click", () => {
    activateApp(button.dataset.openApp);
  });
}

if (els.calendarPrevMonth) {
  els.calendarPrevMonth.addEventListener("click", () => {
    shiftCalendarMonth(-1);
  });
}

if (els.calendarNextMonth) {
  els.calendarNextMonth.addEventListener("click", () => {
    shiftCalendarMonth(1);
  });
}

if (els.calendarSaveButton) {
  els.calendarSaveButton.addEventListener("click", saveCalendarEvent);
}

if (els.calendarClearButton) {
  els.calendarClearButton.addEventListener("click", clearCalendarDay);
}
if (els.hereticResetButton) {
  els.hereticResetButton.addEventListener("click", resetHereticChat);
}
if (els.openclawModeButton) {
  els.openclawModeButton.addEventListener("click", () => {
    switchGpuMode("openclaw_rl");
  });
}
if (els.hereticModeButton) {
  els.hereticModeButton.addEventListener("click", () => {
    switchGpuMode("heretic_chat");
  });
}

setProfileCreateStatus("Create a new saved training profile.");
autoResizeComposer();
autoResizeHereticComposer();
renderShell();
focusActiveAppView();
renderCalendar();
renderHereticTranscript();
renderModelControl();

refreshState().catch((error) => {
  state.proxyOk = false;
  renderStatus();
  window.alert(error.message);
});
