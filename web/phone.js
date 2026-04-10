/**
 * Phone Camera — Capture + WebSocket Stream
 *
 * Captures frames from phone camera via getUserMedia,
 * sends JPEG frames to server via WebSocket.
 */

// ── State ──────────────────────────────────────────────────
let stream = null;
let ws = null;
let sending = false;
let sendInterval = null;
let targetFps = 5;
let frameCount = 0;
let facingMode = 'environment'; // 'user' = front, 'environment' = back
let fpsTracker = { count: 0, lastTime: Date.now() };

const video = document.getElementById('camera-video');
const canvas = document.getElementById('capture-canvas');
const ctx = canvas.getContext('2d');

// ── Camera ─────────────────────────────────────────────────
async function startCamera() {
    try {
        // Stop previous stream
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
        }

        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        await video.play();

        // Set canvas size to match video
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }, { once: true });

        updateStatus('camera-ready', 'Camera ready');
        return true;
    } catch (err) {
        console.error('Camera error:', err);
        updateStatus('error', 'Cannot access camera: ' + err.message);
        return false;
    }
}

async function switchCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    const btn = document.getElementById('btn-switch');
    btn.querySelector('span').textContent = facingMode === 'environment' ? 'Back camera' : 'Front camera';
    await startCamera();
}

// ── WebSocket ──────────────────────────────────────────────
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/phone-camera`;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateStatus('connected', 'Connected to server');
        document.getElementById('conn-status').textContent = 'Connected';
        document.getElementById('conn-status').style.color = 'var(--green)';
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        updateStatus('disconnected', 'Disconnected');
        document.getElementById('conn-status').textContent = 'Disconnected';
        document.getElementById('conn-status').style.color = 'var(--red)';

        // Auto-reconnect if still sending
        if (sending) {
            setTimeout(() => {
                if (sending) connectWebSocket();
            }, 2000);
        }
    };

    ws.onerror = (e) => {
        console.error('WebSocket error:', e);
    };
}

// ── Frame Capture & Send ───────────────────────────────────
function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth) return;

    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Compress to JPEG and send
    canvas.toBlob((blob) => {
        if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then(buf => {
                ws.send(buf);
                frameCount++;
                document.getElementById('frame-count').textContent = frameCount;

                // FPS tracking
                fpsTracker.count++;
                const now = Date.now();
                if (now - fpsTracker.lastTime >= 1000) {
                    document.getElementById('fps-display').textContent =
                        fpsTracker.count + ' FPS';
                    fpsTracker.count = 0;
                    fpsTracker.lastTime = now;
                }
            });
        }
    }, 'image/jpeg', 0.7);
}

// ── Controls ───────────────────────────────────────────────
async function toggleConnection() {
    if (sending) {
        // Stop
        sending = false;
        if (sendInterval) clearInterval(sendInterval);
        if (ws) ws.close();
        ws = null;

        document.getElementById('btn-connect-text').textContent = 'Start';
        document.getElementById('btn-connect').classList.remove('ctrl-btn-danger');
        document.getElementById('btn-connect').classList.add('ctrl-btn-primary');
        updateStatus('stopped', 'Stopped');
        document.getElementById('conn-status').textContent = 'Stopped';
        document.getElementById('conn-status').style.color = '';
    } else {
        // Start
        const cameraOk = await startCamera();
        if (!cameraOk) return;

        connectWebSocket();
        sending = true;
        sendInterval = setInterval(captureAndSend, 1000 / targetFps);

        document.getElementById('btn-connect-text').textContent = 'Stop';
        document.getElementById('btn-connect').classList.remove('ctrl-btn-primary');
        document.getElementById('btn-connect').classList.add('ctrl-btn-danger');
        document.getElementById('instructions').style.display = 'none';
    }
}

function setFps(val) {
    targetFps = parseInt(val) || 5;
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = setInterval(captureAndSend, 1000 / targetFps);
    }
}

// ── Status Updates ─────────────────────────────────────────
function updateStatus(type, text) {
    const badge = document.getElementById('status-badge');
    const overlay = document.getElementById('status-overlay');
    const statusText = document.getElementById('status-text');

    badge.className = 'badge';
    switch (type) {
        case 'connected':
            badge.classList.add('badge-ok');
            badge.textContent = 'Connected';
            overlay.style.display = 'none';
            break;
        case 'disconnected':
            badge.classList.add('badge-error');
            badge.textContent = 'Disconnected';
            overlay.style.display = 'flex';
            statusText.textContent = 'Reconnecting...';
            break;
        case 'camera-ready':
            badge.classList.add('badge-warn');
            badge.textContent = 'Camera OK';
            overlay.style.display = 'none';
            break;
        case 'error':
            badge.classList.add('badge-error');
            badge.textContent = 'Error';
            overlay.style.display = 'flex';
            statusText.textContent = text;
            break;
        case 'stopped':
            badge.textContent = 'Stopped';
            overlay.style.display = 'none';
            break;
        default:
            badge.textContent = text;
    }
}

// ── Init ───────────────────────────────────────────────────
startCamera();
