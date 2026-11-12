const video = document.getElementById("video");
const canvas = document.getElementById("captureCanvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const calibrateBtn = document.getElementById("calibrateBtn");
const cameraMessage = document.getElementById("cameraMessage");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const faceStatus = document.getElementById("faceStatus");
const sessionState = document.getElementById("sessionState");

let stream = null;
let running = false;
let sending = false;
let sessionTimer = null;

async function postJSON(url) {
  const response = await fetch(url, { method: "POST" });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || "Request failed");
  return data;
}

function setActive(active) {
  running = active;
  startBtn.disabled = active;
  stopBtn.disabled = !active;
  calibrateBtn.disabled = !active;
  sessionState.classList.toggle("active", active);
  sessionState.innerHTML = `<i></i> ${active ? "Monitoring" : "Not started"}`;
}

async function startMonitoring() {
  try {
    cameraMessage.textContent = "Requesting camera permission…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    video.style.display = "block";
    cameraPlaceholder.style.display = "none";
    await postJSON("/api/start");
    setActive(true);
    cameraMessage.textContent = "Monitoring started. Sit approximately 50 cm away, then calibrate.";
    sessionTimer = setInterval(captureFrame, 100);
  } catch (error) {
    cameraMessage.textContent = error.message || "Unable to start camera. Check browser permissions.";
    stopCamera();
  }
}

async function captureFrame() {
  if (!running || sending || video.readyState < 2) return;
  sending = true;
  try {
    canvas.width = 480;
    canvas.height = Math.round(480 * video.videoHeight / video.videoWidth);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.72));
    if (!blob) return;
    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    const response = await fetch("/api/frame", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Frame processing failed");
    updateDashboard(data);
  } catch (error) {
    cameraMessage.textContent = error.message;
  } finally {
    sending = false;
  }
}

function updateDashboard(data) {
  document.getElementById("esiValue").textContent = data.esi.toFixed(1);
  document.getElementById("esiLevel").textContent = data.level;
  document.getElementById("esiBar").style.width = `${data.esi}%`;
  document.getElementById("esiBar").style.background =
    data.esi >= 60 ? "#c34d4d" : data.esi >= 30 ? "#c49a4a" : "#4d9b83";
  // Update initial observation status
  const observationStatus =
    document.getElementById("observationStatus");  
  const observationTimer =
    document.getElementById("observationTimer");
  const observationBar =
    document.getElementById("observationBar");
  const observationMessage =
    document.getElementById("observationMessage");
  const observationBox =
    document.getElementById("observationBox");
  if (data.observation_complete) {

  observationStatus.textContent = "ESI calculation active";
  observationTimer.textContent = "Complete";
  observationMessage.textContent =
    "Initial observation completed. Blink factor is now included in ESI.";

  observationBar.style.width = "100%";
  observationBox.classList.add("observation-complete");

} else {

  const remaining = Math.ceil(data.observation_remaining);
  const minutes = Math.floor(remaining / 60);
  const seconds = remaining % 60;

  observationStatus.textContent = "Initial observation";
  observationTimer.textContent =
    `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

  observationMessage.textContent =
    "Collecting blink measurements. Blink factor will be included after 60 seconds.";

  observationBar.style.width =
    `${data.observation_progress}%`;

  observationBox.classList.remove("observation-complete");
}
  document.getElementById("recommendation").textContent = data.recommendation;
  document.getElementById("blinkCount").textContent = data.blink_count;
  document.getElementById("blinkRate").textContent = data.blink_rate.toFixed(1);
  document.getElementById("distance").textContent =
    data.distance_cm === null ? (data.calibrated ? "—" : "Calibrate") : data.distance_cm.toFixed(1);
  document.getElementById("screenTime").textContent = data.elapsed_display;
  document.getElementById("ear").textContent = data.ear === null ? "—" : data.ear.toFixed(3);
  document.getElementById("unsafeTime").textContent = `${Math.round(data.unsafe_distance_seconds)}s`;
  faceStatus.textContent = data.face_detected ? "Face detected" : "No face detected";
  cameraMessage.textContent = data.calibrated
    ? "Calibration complete. Keep your face visible for continuous measurements."
    : "For distance measurements, sit 50 cm from the screen and select Calibrate.";
  const factors = data.factors;
  [["blink", factors.blink], ["closure", factors.closure], ["distance", factors.distance], ["time", factors.screen_time]]
    .forEach(([key, value]) => {
      document.getElementById(`factor${key[0].toUpperCase()+key.slice(1)}`).style.width = `${value}%`;
      document.getElementById(`factor${key[0].toUpperCase()+key.slice(1)}Value`).textContent = `${value.toFixed(0)}%`;
    });
}

async function calibrate() {
  try {
    const result = await postJSON("/api/calibrate");
    cameraMessage.textContent = result.message;
  } catch (error) {
    cameraMessage.textContent = error.message;
  }
}

function stopCamera() {
  if (sessionTimer) clearInterval(sessionTimer);
  sessionTimer = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  video.style.display = "none";
  cameraPlaceholder.style.display = "flex";
  faceStatus.textContent = "Camera stopped";
}

async function stopMonitoring() {
  try {
    const result = await postJSON("/api/stop");
    cameraMessage.textContent = result.summary
      ? `Session saved. Final ESI: ${result.summary.esi}/100.`
      : "Session ended.";
    setActive(false);
    stopCamera();
    loadHistory();
  } catch (error) {
    cameraMessage.textContent = error.message;
  }
}

async function loadHistory() {
  const body = document.getElementById("historyBody");
  try {
    const response = await fetch("/api/history");
    const rows = await response.json();
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty">No saved sessions yet.</td></tr>';
      return;
    }
    body.innerHTML = rows.map(row => {
      const duration = `${Math.floor(row.duration_seconds / 60)}m ${Math.floor(row.duration_seconds % 60)}s`;
      const date = new Date(row.started_at).toLocaleString();
      return `<tr><td>${date}</td><td>${duration}</td><td>${row.blink_count}</td><td>${Number(row.average_blink_rate).toFixed(1)}</td><td>${Number(row.final_esi).toFixed(1)}</td></tr>`;
    }).join("");
  } catch {
    body.innerHTML = '<tr><td colspan="5" class="empty">Unable to load session history.</td></tr>';
  }
}

startBtn.addEventListener("click", startMonitoring);
stopBtn.addEventListener("click", stopMonitoring);
calibrateBtn.addEventListener("click", calibrate);
document.getElementById("refreshHistory").addEventListener("click", loadHistory);
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
});
loadHistory();
