const storedSessions = JSON.parse(localStorage.getItem("nanovllm-sessions") || "[]");
const activeSession = storedSessions.find((item) => item.active);

const state = {
  model: "nano-vllm",
  backend: "",
  messages: activeSession?.messages?.map((item) => ({ ...item })) || [],
  sessions: storedSessions,
  priority: 0,
  controller: null,
  running: false,
};

const $ = (id) => document.getElementById(id);
const messageList = $("messageList");
const promptInput = $("promptInput");

function initializeIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value ?? "0";
}

function formatMs(value) {
  return Number.isFinite(value) ? Math.round(value).toString() : "--";
}

function updateRequestMetrics(metrics = {}, usage = {}) {
  setText("promptTokensValue", usage.prompt_tokens ?? metrics.prompt_tokens ?? 0);
  setText("completionTokensValue", usage.completion_tokens ?? metrics.completion_tokens ?? 0);
  if (metrics.ttft_ms != null) setText("ttftValue", formatMs(metrics.ttft_ms));
  if (metrics.e2e_ms != null) setText("e2eValue", formatMs(metrics.e2e_ms));
}

function updateSchedulerMetrics(metrics = {}) {
  setText("runningValue", metrics.serving_active_requests ?? metrics.running_requests ?? 0);
  setText("waitingValue", metrics.waiting_requests ?? metrics.serving_queued_requests ?? 0);
  setText("cacheHitsValue", metrics.prefix_cache_hit_blocks ?? 0);
  setText("cacheBackendValue", metrics.prefix_cache_backend ?? "--");
  setText("restoreCountValue", metrics.remote_restore_completed ?? 0);
  setText("restoreTokensValue", metrics.kv_restored_tokens ?? 0);
  setText("cancelledValue", metrics.cancelled_requests ?? 0);
  setText("reclaimPolicyValue", metrics.kv_reclaim_policy ?? "--");
  setText("reclaimFreedValue", metrics.kv_reclaim_freed_blocks ?? 0);
  setText("reclaimRetainedValue", metrics.kv_reclaim_retained_blocks ?? 0);
  setText("recomputedTokensValue", metrics.kv_recomputed_tokens ?? 0);
  setText("reclaimFallbackValue", metrics.kv_reclaim_forced_fallbacks ?? 0);
  setText("compressionPolicyValue", metrics.kv_compression_policy ?? "--");
  setText("compressionEventsValue", metrics.kv_compression_events ?? 0);
  setText("compressionFreedValue", metrics.kv_compression_freed_blocks ?? 0);
  setText("compressionTokensValue", metrics.kv_compression_dropped_tokens ?? 0);
  const rate = Math.max(0, Math.min(1, Number(metrics.prefix_cache_block_hit_rate || 0)));
  setText("cacheRateValue", `${Math.round(rate * 100)}%`);
  $("cacheRateBar").style.width = `${rate * 100}%`;
  if (metrics.policy) setText("policyBadge", String(metrics.policy).toUpperCase());
  if (metrics.max_model_len) {
    $("maxTokens").max = String(metrics.max_model_len);
  }
}

function scrollToBottom() {
  messageList.scrollTop = messageList.scrollHeight;
}

function createMessage(role, content = "", generating = false) {
  $("emptyState")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}${generating ? " generating" : ""}`;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "assistant" ? "nV" : "YOU";
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "assistant" ? state.model : "You";
  const text = document.createElement("div");
  text.className = "message-content";
  text.textContent = content;
  body.append(label, text);
  article.append(avatar, body);
  messageList.append(article);
  scrollToBottom();
  return { article, text };
}

function renderMessages() {
  messageList.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.id = "emptyState";
    empty.className = "empty-state";
    empty.innerHTML = `<div class="empty-logo">nV</div><h1>开始本地对话</h1><p id="emptyModel">${state.model}</p>`;
    messageList.append(empty);
    return;
  }
  for (const message of state.messages) createMessage(message.role, message.content);
}

function saveSession() {
  if (!state.messages.length) return;
  const title = state.messages.find((item) => item.role === "user")?.content.slice(0, 24) || "新对话";
  const existing = state.sessions.findIndex((item) => item.active);
  state.sessions.forEach((item) => { item.active = false; });
  const snapshot = { title, messages: state.messages, active: true, updated: Date.now() };
  if (existing >= 0) state.sessions[existing] = snapshot;
  else state.sessions.unshift(snapshot);
  state.sessions = state.sessions.slice(0, 12);
  localStorage.setItem("nanovllm-sessions", JSON.stringify(state.sessions));
  renderSessions();
}

function renderSessions() {
  const list = $("sessionList");
  list.replaceChildren();
  for (const [index, session] of state.sessions.entries()) {
    const row = document.createElement("div");
    row.className = `session-row${session.active ? " active" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.textContent = session.title;
    button.title = session.title;
    button.addEventListener("click", () => {
      if (state.running) return;
      state.sessions.forEach((item) => { item.active = false; });
      session.active = true;
      state.messages = session.messages.map((item) => ({ ...item }));
      localStorage.setItem("nanovllm-sessions", JSON.stringify(state.sessions));
      renderSessions();
      renderMessages();
    });
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete";
    deleteButton.title = `删除对话：${session.title}`;
    deleteButton.setAttribute("aria-label", `删除对话：${session.title}`);
    deleteButton.innerHTML = '<i data-lucide="trash-2">删除</i>';
    deleteButton.addEventListener("click", () => {
      if (state.running) return;
      if (!window.confirm(`确定删除“${session.title}”吗？此操作无法撤销。`)) return;
      const deletingActiveSession = session.active;
      state.sessions.splice(index, 1);
      if (deletingActiveSession) state.messages = [];
      localStorage.setItem("nanovllm-sessions", JSON.stringify(state.sessions));
      renderSessions();
      if (deletingActiveSession) renderMessages();
      promptInput.focus();
    });
    row.append(button, deleteButton);
    list.append(row);
  }
  initializeIcons();
}

function newConversation() {
  if (state.running) state.controller?.abort();
  state.sessions.forEach((item) => { item.active = false; });
  localStorage.setItem("nanovllm-sessions", JSON.stringify(state.sessions));
  state.messages = [];
  renderMessages();
  renderSessions();
  promptInput.focus();
}

function setRunning(running) {
  state.running = running;
  $("sendButton").hidden = running;
  $("stopButton").hidden = !running;
  $("requestState").textContent = running ? "STREAMING" : "IDLE";
  $("requestState").classList.toggle("running", running);
}

function parseSSEBuffer(buffer, onEvent) {
  const frames = buffer.split("\n\n");
  const remainder = frames.pop();
  for (const frame of frames) {
    const data = frame.split("\n").filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()).join("\n");
    if (data) onEvent(data);
  }
  return remainder;
}

async function streamCompletion(messages, assistantView) {
  const controller = new AbortController();
  state.controller = controller;
  const started = performance.now();
  let firstTokenAt = null;
  let serverMetrics = {};
  let usage = {};
  const apiKey = $("apiKey").value.trim();
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
    },
    signal: controller.signal,
    body: JSON.stringify({
      model: state.model,
      messages,
      stream: true,
      stream_options: { include_usage: true },
      temperature: Number($("temperature").value),
      max_tokens: Number($("maxTokens").value),
      priority: state.priority,
      request_class: state.priority >= 10 ? "urgent" : "interactive",
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error?.message || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done = false;
  const consume = (raw) => {
    if (raw === "[DONE]") { done = true; return; }
    const chunk = JSON.parse(raw);
    if (chunk.error) throw new Error(chunk.error.message);
    const delta = chunk.choices?.[0]?.delta?.content;
    if (delta) {
      if (firstTokenAt == null) {
        firstTokenAt = performance.now();
        setText("ttftValue", formatMs(firstTokenAt - started));
      }
      assistantView.text.textContent += delta;
      scrollToBottom();
    }
    if (chunk.x_nanovllm_metrics) serverMetrics = chunk.x_nanovllm_metrics;
    if (chunk.usage) usage = chunk.usage;
  };
  while (!done) {
    const result = await reader.read();
    buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
    buffer = parseSSEBuffer(buffer, consume);
    if (result.done) break;
  }
  const e2e = performance.now() - started;
  setText("e2eValue", formatMs(e2e));
  updateRequestMetrics({ ...serverMetrics, e2e_ms: serverMetrics.e2e_ms ?? e2e }, usage);
  return assistantView.text.textContent;
}

async function submitPrompt(event) {
  event.preventDefault();
  const content = promptInput.value.trim();
  if (!content || state.running) return;
  if (!$("maxTokens").reportValidity()) return;
  $("settingsPanel").hidden = true;
  const userMessage = { role: "user", content };
  state.messages.push(userMessage);
  createMessage("user", content);
  promptInput.value = "";
  resizeComposer();
  setText("tokenEstimate", "0 字符");
  const assistant = createMessage("assistant", "", true);
  setRunning(true);
  try {
    const text = await streamCompletion(state.messages, assistant);
    assistant.article.classList.remove("generating");
    state.messages.push({ role: "assistant", content: text });
    saveSession();
  } catch (error) {
    assistant.article.classList.remove("generating");
    if (error.name === "AbortError") {
      if (!assistant.text.textContent) assistant.text.textContent = "已停止生成";
      state.messages.push({ role: "assistant", content: assistant.text.textContent });
      saveSession();
    } else {
      assistant.text.textContent = error.message;
      assistant.text.classList.add("message-error");
    }
  } finally {
    setRunning(false);
    state.controller = null;
    promptInput.focus();
  }
}

function resizeComposer() {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 150)}px`;
  setText("tokenEstimate", `${promptInput.value.length} 字符`);
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    state.model = health.model;
    state.backend = health.backend;
    setText("modelLabel", health.model);
    setText("backendLabel", health.backend);
    setText("emptyModel", health.model);
    document.querySelectorAll(".message.assistant .message-role")
      .forEach((element) => { element.textContent = health.model; });
    const supportsQos = !health.backend.includes("Transformers");
    $("priorityLabel").hidden = !supportsQos;
    $("priorityControl").hidden = !supportsQos;
    $("statusDot").classList.toggle("online", health.status === "ok");
  } catch {
    setText("backendLabel", "Disconnected");
    $("statusDot").classList.remove("online");
  }
}

async function refreshMetrics() {
  try {
    const apiKey = $("apiKey").value.trim();
    const response = await fetch("/v1/metrics", {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    });
    if (response.ok) updateSchedulerMetrics(await response.json());
  } catch { /* health polling handles connection state */ }
}

$("composer").addEventListener("submit", submitPrompt);
$("stopButton").addEventListener("click", () => state.controller?.abort());
$("newChat").addEventListener("click", newConversation);
$("newChatTop").addEventListener("click", newConversation);
$("settingsToggle").addEventListener("click", () => {
  $("settingsPanel").hidden = !$("settingsPanel").hidden;
});
$("metricsToggle").addEventListener("click", () => $("metricsPanel").classList.add("open"));
$("metricsClose").addEventListener("click", () => $("metricsPanel").classList.remove("open"));
$("temperature").addEventListener("input", (event) => setText("temperatureValue", event.target.value));
$("apiKey").value = localStorage.getItem("nanovllm-api-key") || "";
$("apiKey").addEventListener("change", (event) => localStorage.setItem("nanovllm-api-key", event.target.value));
promptInput.addEventListener("input", resizeComposer);
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("priorityControl").addEventListener("click", (event) => {
  const button = event.target.closest("[data-priority]");
  if (!button) return;
  state.priority = Number(button.dataset.priority);
  document.querySelectorAll(".segment").forEach((item) => item.classList.toggle("active", item === button));
});

renderSessions();
renderMessages();
refreshHealth();
refreshMetrics();
setInterval(refreshMetrics, 1500);
setInterval(refreshHealth, 10000);
