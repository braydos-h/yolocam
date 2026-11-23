(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    cameraName: $("camera-name"),
    statusPill: $("status-pill"),
    statusText: $("status-text"),
    btnStart: $("btn-start"),
    btnStop: $("btn-stop"),
    btnReconnect: $("btn-reconnect"),
    aiBanner: $("ai-banner"),
    aiBannerText: $("ai-banner-text"),
    liveLabel: $("live-label"),
    liveDot: $("live-dot"),
    liveFps: $("live-fps"),
    liveInference: $("live-inference"),
    stage: $("video-stage"),
    video: $("live-video"),
    placeholder: $("video-placeholder"),
    placeholderTitle: $("placeholder-title"),
    placeholderCopy: $("placeholder-copy"),
    boxLayer: $("box-layer"),
    stageStatus: $("stage-status"),
    stageAnalysis: $("stage-analysis"),
    analysisCaption: $("analysis-caption"),
    currentPeople: $("current-people"),
    currentCars: $("current-cars"),
    totalPeople: $("total-people"),
    totalCars: $("total-cars"),
    cameraFps: $("camera-fps"),
    inferenceTime: $("inference-time"),
    pipelineTime: $("pipeline-time"),
    lastDetection: $("last-detection"),
    lastAnalysis: $("last-analysis"),
    detectionInterval: $("detection-interval"),
    btnFullscreen: $("btn-fullscreen"),
    historyFilters: $("history-filters"),
    filterClass: $("filter-class"),
    filterFrom: $("filter-from"),
    filterTo: $("filter-to"),
    btnResetFilters: $("btn-reset-filters"),
    historyBody: $("history-body"),
    historyCount: $("history-count"),
    settingsForm: $("settings-form"),
    settingsEnvNote: $("settings-env-note"),
    settingsStatus: $("settings-status"),
    credentialsHint: $("credentials-hint"),
    serverNote: $("server-note"),
    deviceHint: $("device-hint"),
    btnSaveSettings: $("btn-save-settings"),
    cameraNameInput: $("camera-name-input"),
    rtspUrl: $("rtsp-url"),
    rtspUsername: $("rtsp-username"),
    rtspPassword: $("rtsp-password"),
    clearCredentials: $("clear-credentials"),
    reconnectDelay: $("reconnect-delay"),
    ffmpegPath: $("ffmpeg-path"),
    modelPath: $("model-path"),
    confidence: $("confidence"),
    intervalMs: $("interval-ms"),
    detectPerson: $("detect-person"),
    detectCar: $("detect-car"),
    device: $("device"),
    onnxruntimePath: $("onnxruntime-path"),
    saveScreenshots: $("save-screenshots"),
    screenshotDir: $("screenshot-dir"),
    retentionDays: $("retention-days"),
    toast: $("toast"),
  };

  const state = {
    snapshot: null,
    settings: null,
    ws: null,
    wsRetryMs: 1000,
    wsRetryTimer: null,
    streamActive: false,
    streamRetryTimer: null,
    status: "stopped",
    historyBusy: false,
    toastTimer: null,
    actionBusy: false,
  };

  const ACTIVE_STATES = new Set(["streaming", "connecting", "reconnecting"]);

  init();

  function init() {
    els.btnStart.addEventListener("click", () => cameraAction("start"));
    els.btnStop.addEventListener("click", () => cameraAction("stop"));
    els.btnReconnect.addEventListener("click", () => cameraAction("reconnect"));
    els.btnFullscreen.addEventListener("click", toggleFullscreen);
    els.settingsForm.addEventListener("submit", saveSettings);
    els.historyFilters.addEventListener("submit", (event) => {
      event.preventDefault();
      loadHistory(false);
    });
    els.btnResetFilters.addEventListener("click", () => {
      els.filterClass.value = "";
      els.filterFrom.value = "";
      els.filterTo.value = "";
      loadHistory(false);
    });

    els.video.addEventListener("load", () => {
      els.stage.classList.add("has-frame");
      alignOverlay();
    });
    els.video.addEventListener("error", handleStreamError);

    window.addEventListener("resize", debounce(alignOverlay, 120));
    document.addEventListener("fullscreenchange", () => {
      window.setTimeout(alignOverlay, 60);
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && (!state.ws || state.ws.readyState !== WebSocket.OPEN)) {
        connectWebSocket();
      }
    });

    loadSettings().catch((error) => showToast(error.message, "error"));
    loadHistory(true);
    refreshSnapshotFallback();
    connectWebSocket();

    window.setInterval(() => {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        refreshSnapshotFallback();
      }
    }, 3000);

    window.setInterval(() => {
      if (!document.hidden) {
        loadHistory(true);
      }
    }, 5000);
  }

  async function api(path, options = {}) {
    const request = { cache: "no-store", ...options };
    if (request.body && typeof request.body !== "string") {
      request.headers = { ...(request.headers || {}), "Content-Type": "application/json" };
      request.body = JSON.stringify(request.body);
    }
    const response = await fetch(path, request);
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const data = await response.json();
        if (data && data.error) {
          message = data.error;
        }
      } catch {
        // Keep the status-based message.
      }
      throw new Error(message);
    }
    if (response.status === 204) {
      return null;
    }
    return response.json();
  }

  function connectWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    window.clearTimeout(state.wsRetryTimer);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
    state.ws = socket;

    socket.addEventListener("open", () => {
      state.wsRetryMs = 1000;
      refreshSnapshotFallback();
    });

    socket.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "snapshot" && message.data) {
          applySnapshot(message.data);
        }
      } catch (error) {
        console.warn("Ignoring malformed WebSocket message", error);
      }
    });

    socket.addEventListener("close", () => {
      state.ws = null;
      scheduleWsReconnect();
    });

    socket.addEventListener("error", () => {
      socket.close();
    });
  }

  function scheduleWsReconnect() {
    window.clearTimeout(state.wsRetryTimer);
    const delay = state.wsRetryMs;
    state.wsRetryMs = Math.min(Math.round(state.wsRetryMs * 1.6), 12000);
    state.wsRetryTimer = window.setTimeout(connectWebSocket, delay);
  }

  async function refreshSnapshotFallback() {
    try {
      const snapshot = await api("/api/stats");
      applySnapshot(snapshot);
    } catch (error) {
      console.warn("Stats refresh failed:", error.message);
    }
  }

  function applySnapshot(snapshot) {
    if (!snapshot || !snapshot.camera) {
      return;
    }
    state.snapshot = snapshot;
    const status = snapshot.camera.status || { state: "stopped", message: null };
    state.status = status.state || "stopped";

    els.cameraName.textContent = snapshot.camera.name || "Camera";
    els.statusPill.className = `status-pill ${state.status}`;
    els.statusText.textContent = statusLabel(status);
    els.statusPill.title = status.message || "";

    const busy = ACTIVE_STATES.has(state.status);
    els.btnStart.disabled = busy || state.actionBusy;
    els.btnStop.disabled = state.status === "stopped" || state.actionBusy;
    els.btnReconnect.disabled = state.actionBusy;

    const fps = Number(snapshot.camera.fps || 0);
    const analysis = snapshot.analysis;
    const people = analysis ? Number(analysis.people || 0) : 0;
    const cars = analysis ? Number(analysis.cars || 0) : 0;
    const totals = snapshot.totals || {};

    els.currentPeople.textContent = String(people);
    els.currentCars.textContent = String(cars);
    els.totalPeople.textContent = formatCount(totals.people_events);
    els.totalCars.textContent = formatCount(totals.car_events);
    els.cameraFps.textContent = fps.toFixed(1);
    els.liveFps.textContent = fps.toFixed(1);
    els.liveInference.textContent = analysis ? formatMs(analysis.inference_ms) : "—";
    els.inferenceTime.textContent = analysis ? formatMs(analysis.inference_ms) : "—";
    els.pipelineTime.textContent = analysis ? formatMs(analysis.pipeline_ms) : "—";
    els.lastDetection.textContent = snapshot.last_detection_at
      ? formatTime(snapshot.last_detection_at)
      : "—";
    els.lastAnalysis.textContent = analysis ? formatTime(analysis.analyzed_at) : "—";
    els.detectionInterval.textContent = `${Number(snapshot.detection_interval_ms || 0)} ms`;

    const liveLabel = state.status === "streaming"
      ? "LIVE"
      : state.status === "connecting"
        ? "CONNECTING"
        : state.status === "reconnecting"
          ? "RECONNECTING"
          : "OFFLINE";
    els.liveLabel.textContent = liveLabel;
    els.liveDot.className = `dot live-dot ${state.status}`;
    els.stageStatus.textContent = liveLabel;

    if (analysis) {
      els.stageAnalysis.textContent = `People ${people} · Cars ${cars}`;
      els.analysisCaption.textContent =
        `${analysis.source_width}×${analysis.source_height} · analyzed ${formatTime(analysis.analyzed_at)} · AI ${formatMs(analysis.inference_ms)}`;
    } else {
      els.stageAnalysis.textContent = "No analysis yet";
      els.analysisCaption.textContent =
        "Per-frame counts reset each analysis; totals are cumulative detection events and do not represent unique objects.";
    }

    renderDetections(analysis);
    updateStreamState(status);
    updatePlaceholder(status);
    updateAiBanner(snapshot);
    alignOverlay();
  }

  function statusLabel(status) {
    switch (status.state) {
      case "streaming":
        return "Live";
      case "connecting":
        return "Connecting";
      case "reconnecting":
        return status.attempt > 0 ? `Reconnecting · ${status.attempt}` : "Reconnecting";
      case "error":
        return "Error";
      default:
        return "Stopped";
    }
  }

  function updateStreamState(status) {
    if (status.state === "stopped" || status.state === "error") {
      stopStream();
      return;
    }
    if (ACTIVE_STATES.has(status.state)) {
      ensureStream();
    }
  }

  function ensureStream() {
    if (state.streamActive) {
      return;
    }
    state.streamActive = true;
    window.clearTimeout(state.streamRetryTimer);
    els.video.src = `/api/stream?ts=${Date.now()}`;
  }

  function stopStream() {
    state.streamActive = false;
    window.clearTimeout(state.streamRetryTimer);
    els.video.removeAttribute("src");
    els.stage.classList.remove("has-frame");
    els.boxLayer.replaceChildren();
  }

  function handleStreamError() {
    if (!state.streamActive) {
      return;
    }
    state.streamActive = false;
    els.stage.classList.remove("has-frame");
    window.clearTimeout(state.streamRetryTimer);
    if (ACTIVE_STATES.has(state.status)) {
      els.placeholderTitle.textContent = "Stream interrupted";
      els.placeholderCopy.textContent = "Waiting for the server to resume the MJPEG stream…";
      state.streamRetryTimer = window.setTimeout(() => {
        if (ACTIVE_STATES.has(state.status)) {
          ensureStream();
        }
      }, 2500);
    }
  }

  function updatePlaceholder(status) {
    if (els.stage.classList.contains("has-frame")) {
      return;
    }
    const message = status.message ? ` ${status.message}` : "";
    if (status.state === "connecting") {
      els.placeholderTitle.textContent = "Connecting to camera";
      els.placeholderCopy.textContent = "Opening the RTSP stream through FFmpeg…";
    } else if (status.state === "reconnecting") {
      els.placeholderTitle.textContent = "Camera disconnected";
      const retry = status.retry_in_seconds ? ` Retrying in ${status.retry_in_seconds}s.` : "";
      els.placeholderCopy.textContent = `${message || "The stream stopped."}${retry}`.trim();
    } else if (status.state === "error") {
      els.placeholderTitle.textContent = "Camera error";
      els.placeholderCopy.textContent = status.message || "Check the RTSP URL and credentials.";
    } else if (status.state === "streaming") {
      els.placeholderTitle.textContent = "Waiting for video";
      els.placeholderCopy.textContent = "The camera is connected; waiting for the first decoded frame.";
    } else {
      els.placeholderTitle.textContent = "Camera stopped";
      els.placeholderCopy.textContent = "Press Start to connect to the RTSP stream.";
    }
  }

  function updateAiBanner(snapshot) {
    const ai = snapshot.ai || {};
    if (ai.state === "error") {
      showBanner("error", ai.message || "YOLO inference is unavailable.");
      return;
    }
    if (snapshot.camera.status && snapshot.camera.status.state === "error") {
      showBanner("error", snapshot.camera.status.message || "Camera error.");
      return;
    }
    if (ai.state === "loading") {
      showBanner("info", ai.message || "Loading YOLO model…");
      return;
    }
    if (ai.message) {
      showBanner("warning", ai.message);
      return;
    }
    hideBanner();
  }

  function showBanner(kind, text) {
    els.aiBanner.className = `banner ${kind}`;
    els.aiBannerText.textContent = text;
  }

  function hideBanner() {
    els.aiBanner.className = "banner hidden";
    els.aiBannerText.textContent = "";
  }

  function renderDetections(analysis) {
    els.boxLayer.replaceChildren();
    if (!analysis || !Array.isArray(analysis.detections) || analysis.detections.length === 0) {
      return;
    }
    if (!ACTIVE_STATES.has(state.status)) {
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const detection of analysis.detections) {
      const bbox = detection.bbox;
      if (!bbox || !Number.isFinite(bbox.x) || !Number.isFinite(bbox.y)) {
        continue;
      }
      const box = document.createElement("div");
      box.className = `detection-box ${detection.class === "car" ? "car" : "person"}`;
      box.style.left = `${clampPercent(bbox.x) * 100}%`;
      box.style.top = `${clampPercent(bbox.y) * 100}%`;
      box.style.width = `${clampPercent(bbox.w) * 100}%`;
      box.style.height = `${clampPercent(bbox.h) * 100}%`;
      const label = document.createElement("span");
      label.className = "detection-label";
      if (Number(bbox.x) + Number(bbox.w) > 0.78) {
        label.classList.add("label-right");
      }
      label.textContent = `${detection.class === "car" ? "Car" : "Person"} ${Math.round(Number(detection.confidence) * 100)}%`;
      box.appendChild(label);
      fragment.appendChild(box);
    }
    els.boxLayer.appendChild(fragment);
  }

  function alignOverlay() {
    const stageRect = els.stage.getBoundingClientRect();
    if (!stageRect.width || !stageRect.height) {
      return;
    }
    const snapshot = state.snapshot;
    const analysis = snapshot && snapshot.analysis;
    const naturalWidth = els.video.naturalWidth || (analysis && analysis.source_width) || 0;
    const naturalHeight = els.video.naturalHeight || (analysis && analysis.source_height) || 0;
    const hasFrame = els.stage.classList.contains("has-frame");

    let left = 0;
    let top = 0;
    let width = stageRect.width;
    let height = stageRect.height;

    if (hasFrame && naturalWidth > 0 && naturalHeight > 0) {
      const imageRect = els.video.getBoundingClientRect();
      const scale = Math.min(imageRect.width / naturalWidth, imageRect.height / naturalHeight);
      width = naturalWidth * scale;
      height = naturalHeight * scale;
      left = imageRect.left - stageRect.left + (imageRect.width - width) / 2;
      top = imageRect.top - stageRect.top + (imageRect.height - height) / 2;
    }

    els.boxLayer.style.left = `${Math.max(0, left)}px`;
    els.boxLayer.style.top = `${Math.max(0, top)}px`;
    els.boxLayer.style.width = `${Math.max(0, width)}px`;
    els.boxLayer.style.height = `${Math.max(0, height)}px`;
  }

  async function cameraAction(action) {
    if (state.actionBusy) {
      return;
    }
    state.actionBusy = true;
    setActionButtonsBusy(true);
    try {
      const result = await api(`/api/camera/${action}`, { method: "POST" });
      if (result && result.snapshot) {
        applySnapshot(result.snapshot);
      }
      if (action === "start") {
        showToast(result && result.started ? "Camera start requested." : "Camera is already running.", "success");
      } else if (action === "stop") {
        showToast("Camera stopped.", "success");
      } else {
        showToast("Reconnecting camera…", "success");
      }
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      state.actionBusy = false;
      setActionButtonsBusy(false);
      if (state.snapshot) {
        applySnapshot(state.snapshot);
      }
    }
  }

  function setActionButtonsBusy(busy) {
    if (busy) {
      els.btnStart.disabled = true;
      els.btnStop.disabled = true;
      els.btnReconnect.disabled = true;
    }
  }

  async function loadHistory(quiet) {
    if (state.historyBusy || document.hidden) {
      return;
    }
    state.historyBusy = true;
    try {
      const params = new URLSearchParams();
      const classValue = els.filterClass.value;
      if (classValue) {
        params.set("class", classValue);
      }
      const from = localDateTimeToIso(els.filterFrom.value);
      const to = localDateTimeToIso(els.filterTo.value);
      if (els.filterFrom.value && !from) {
        throw new Error("Invalid From timestamp.");
      }
      if (els.filterTo.value && !to) {
        throw new Error("Invalid To timestamp.");
      }
      if (from && to && new Date(from) > new Date(to)) {
        throw new Error("From must be before To.");
      }
      if (from) {
        params.set("from", from);
      }
      if (to) {
        params.set("to", to);
      }
      params.set("limit", "100");
      const data = await api(`/api/history?${params.toString()}`);
      renderHistory(data.items || []);
      els.historyCount.textContent = `${(data.items || []).length} shown`;
    } catch (error) {
      if (!quiet) {
        showToast(error.message, "error");
      }
    } finally {
      state.historyBusy = false;
    }
  }

  function renderHistory(items) {
    if (!items.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.className = "empty-cell";
      cell.colSpan = 5;
      cell.textContent = "No detection events match the current filters.";
      row.appendChild(cell);
      els.historyBody.replaceChildren(row);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      const row = document.createElement("tr");

      const timeCell = document.createElement("td");
      timeCell.textContent = formatTime(item.timestamp);
      row.appendChild(timeCell);

      const classCell = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `class-badge ${item.class === "car" ? "car" : "person"}`;
      badge.textContent = item.class === "car" ? "Car" : "Person";
      classCell.appendChild(badge);
      row.appendChild(classCell);

      const confidenceCell = document.createElement("td");
      confidenceCell.className = "confidence-cell";
      confidenceCell.textContent = `${Math.round(Number(item.confidence) * 100)}%`;
      row.appendChild(confidenceCell);

      const cameraCell = document.createElement("td");
      cameraCell.textContent = item.camera_name || "—";
      row.appendChild(cameraCell);

      const screenshotCell = document.createElement("td");
      if (item.screenshot) {
        const link = document.createElement("a");
        link.className = "screenshot-link";
        link.href = `/api/screenshots/${encodeURIComponent(item.screenshot)}`;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "View JPEG";
        link.title = item.screenshot;
        screenshotCell.appendChild(link);
      } else {
        screenshotCell.textContent = "—";
      }
      row.appendChild(screenshotCell);
      fragment.appendChild(row);
    }
    els.historyBody.replaceChildren(fragment);
  }

  async function loadSettings() {
    const settings = await api("/api/settings");
    state.settings = settings;
    populateSettings(settings);
  }

  function populateSettings(settings) {
    const camera = settings.camera || {};
    const yolo = settings.yolo || {};
    const storage = settings.storage || {};

    els.cameraNameInput.value = camera.name || "";
    els.rtspUrl.value = camera.rtsp_url || "";
    els.rtspUsername.value = "";
    els.rtspPassword.value = "";
    els.clearCredentials.checked = false;
    els.reconnectDelay.value = camera.reconnect_delay_seconds ?? 5;
    els.ffmpegPath.value = camera.ffmpeg_path || "ffmpeg";
    els.rtspUsername.placeholder = camera.username_set ? "Leave blank to keep" : "Not set";
    els.rtspPassword.placeholder = camera.password_set ? "Leave blank to keep" : "Not set";

    const envLocked = Boolean(camera.env_locked);
    els.rtspUrl.disabled = envLocked;
    els.rtspUsername.disabled = envLocked;
    els.rtspPassword.disabled = envLocked;
    els.clearCredentials.disabled = envLocked;
    els.settingsEnvNote.classList.toggle("hidden", !envLocked);
    els.credentialsHint.textContent = envLocked
      ? "RTSP URL and credentials are managed by environment variables. Only non-secret camera options can be changed here."
      : "Credentials are never returned to the browser. Leave the fields blank to keep existing values.";

    els.modelPath.value = yolo.model_path || "";
    els.confidence.value = yolo.confidence_threshold ?? 0.5;
    els.intervalMs.value = yolo.interval_ms ?? 1000;
    els.detectPerson.checked = Boolean(yolo.detect_person);
    els.detectCar.checked = Boolean(yolo.detect_car);

    const capabilities = settings.capabilities || {};
    for (const option of els.device.options) {
      if (option.value === "cuda") {
        option.disabled = !capabilities.cuda;
      }
      if (option.value === "directml") {
        option.disabled = !capabilities.directml;
      }
    }
    els.device.value = yolo.device || "cpu";
    els.deviceHint.textContent = capabilities.cuda || capabilities.directml
      ? "Execution providers compiled into this build are available below."
      : "CPU inference requires no GPU. CUDA/DirectML builds are optional.";
    els.onnxruntimePath.value = yolo.onnxruntime_path || "";

    els.saveScreenshots.checked = Boolean(storage.save_screenshots);
    els.screenshotDir.value = storage.screenshot_directory || "./detections";
    els.retentionDays.value = storage.retention_days ?? 7;

    const server = settings.server || {};
    els.serverNote.textContent = `Dashboard server: ${server.host || "127.0.0.1"}:${server.port || 8080}. Change server settings in config.toml and restart.`;
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (!state.settings) {
      showToast("Settings are not loaded yet.", "error");
      return;
    }
    const envLocked = Boolean(state.settings.camera && state.settings.camera.env_locked);
    const camera = {
      name: els.cameraNameInput.value.trim(),
      reconnect_delay_seconds: Number(els.reconnectDelay.value),
      ffmpeg_path: els.ffmpegPath.value.trim(),
    };
    if (!envLocked) {
      camera.rtsp_url = els.rtspUrl.value.trim();
      camera.username = els.rtspUsername.value.trim() || null;
      camera.password = els.rtspPassword.value || null;
      camera.clear_credentials = els.clearCredentials.checked;
    }

    const yolo = {
      model_path: els.modelPath.value.trim(),
      confidence_threshold: Number(els.confidence.value),
      interval_ms: Number(els.intervalMs.value),
      detect_person: els.detectPerson.checked,
      detect_car: els.detectCar.checked,
      device: els.device.value,
      onnxruntime_path: els.onnxruntimePath.value.trim(),
    };
    const storage = {
      save_screenshots: els.saveScreenshots.checked,
      screenshot_directory: els.screenshotDir.value.trim(),
      retention_days: Number(els.retentionDays.value),
    };

    if (!camera.name) {
      showSettingsStatus("Camera name is required.", "error");
      return;
    }
    if (!(yolo.confidence_threshold >= 0 && yolo.confidence_threshold <= 1)) {
      showSettingsStatus("Confidence must be between 0 and 1.", "error");
      return;
    }
    if (!(yolo.interval_ms >= 100 && yolo.interval_ms <= 60000)) {
      showSettingsStatus("Detection interval must be between 100 and 60000 ms.", "error");
      return;
    }
    if (!yolo.detect_person && !yolo.detect_car) {
      showSettingsStatus("Enable at least one of person or car detection.", "error");
      return;
    }
    if (!(storage.retention_days >= 1 && storage.retention_days <= 3650)) {
      showSettingsStatus("Retention must be between 1 and 3650 days.", "error");
      return;
    }

    els.btnSaveSettings.disabled = true;
    showSettingsStatus("Saving…");
    try {
      const result = await api("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { camera, yolo, storage },
      });
      state.settings = result.settings;
      populateSettings(result.settings);
      const restart = Array.isArray(result.restart_required) ? result.restart_required : [];
      const restartText = restart.length
        ? ` Restart the app to apply: ${restart.map(restartLabel).join(", ")}.`
        : "";
      showSettingsStatus(`Saved.${restartText}`, restart.length ? "success" : "success");
      showToast(`Settings saved.${restartText}`, "success");
      els.rtspUsername.value = "";
      els.rtspPassword.value = "";
      els.clearCredentials.checked = false;
      if (result.camera_reconnecting) {
        showToast("Camera settings changed; reconnecting.", "success");
      }
    } catch (error) {
      showSettingsStatus(error.message, "error");
      showToast(error.message, "error");
    } finally {
      els.btnSaveSettings.disabled = false;
    }
  }

  function restartLabel(field) {
    switch (field) {
      case "model_path":
        return "model path";
      case "device":
        return "execution provider";
      case "onnxruntime_path":
        return "ONNX Runtime path";
      default:
        return field;
    }
  }

  function showSettingsStatus(message, kind) {
    els.settingsStatus.textContent = message;
    els.settingsStatus.className = `form-status${kind ? ` ${kind}` : ""}`;
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if (els.stage.requestFullscreen) {
        await els.stage.requestFullscreen();
      }
      window.setTimeout(alignOverlay, 80);
    } catch (error) {
      showToast(`Fullscreen failed: ${error.message}`, "error");
    }
  }

  function showToast(message, kind = "") {
    window.clearTimeout(state.toastTimer);
    els.toast.textContent = message;
    els.toast.className = `toast${kind ? ` ${kind}` : ""}`;
    state.toastTimer = window.setTimeout(() => {
      els.toast.className = "toast hidden";
    }, 4500);
  }

  function formatTime(value) {
    if (!value) {
      return "—";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "—";
    }
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function formatMs(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    if (number >= 1000) {
      return `${(number / 1000).toFixed(2)} s`;
    }
    return `${Math.round(number)} ms`;
  }

  function formatCount(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString() : "0";
  }

  function localDateTimeToIso(value) {
    if (!value) {
      return "";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }
    return date.toISOString();
  }

  function clampPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return 0;
    }
    return Math.min(1, Math.max(0, number));
  }

  function debounce(fn, delay) {
    let timer = null;
    return () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(fn, delay);
    };
  }

  window.addEventListener("beforeunload", () => {
    if (state.ws) {
      state.ws.close();
    }
  });
})();
