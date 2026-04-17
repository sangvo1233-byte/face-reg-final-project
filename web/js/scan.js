import { api } from './api.js';
import { state } from './state.js';
import { hideElement, setTone, showElement, showToast } from './ui.js';

const SCAN_MODES = ['auto', 'local_direct', 'browser_ws'];
const SUCCESS_OVERLAY_DISPLAY_MS = 3400;
const SUCCESS_OVERLAY_EXIT_MS = 220;
const SUCCESS_COUNTDOWN_START = 3;
const FAILURE_CARD_THROTTLE_MS = 1800;
const LOCAL_WS_RECONNECT_MS = 1500;
const BROWSER_WS_RECONNECT_MS = 1500;

const browserCanvas = document.createElement('canvas');
const browserCtx = browserCanvas.getContext('2d');
const successOverlayQueue = [];

let localSocket = null;
let localReconnectTimer = null;
let browserSocket = null;
let browserReconnectTimer = null;
let browserSendTimer = null;
let browserSending = false;
let browserFrameWidth = 960;
let browserJpegQuality = 0.82;
let browserTargetFps = 10;
let latestBrowserFrameBlob = null;
let currentTransportMode = null;
let successOverlayTimer = null;
let successCountdownTimer = null;
let activeSuccessResult = null;
let isShowingSuccessOverlay = false;
let isClosingSuccessOverlay = false;
let lastFailureCardAt = 0;

function fallbackCapabilities() {
    return {
        scan_mode_default: 'auto',
        scan_modes: ['auto', 'local_direct', 'browser_ws'],
        local_direct_available: true,
        browser_stream_available: true
    };
}

export async function initializeScanMode() {
    if (!SCAN_MODES.includes(state.scanModePreference)) {
        state.scanModePreference = 'auto';
    }
    await loadScanCapabilities(true);
    await syncScanModeUI();
}

async function loadScanCapabilities(force = false) {
    if (!force && state.scanCapabilities) return state.scanCapabilities;
    try {
        state.scanCapabilities = { ...fallbackCapabilities(), ...(await api('/api/system/capabilities')) };
    } catch (error) {
        console.warn('Capability load failed:', error);
        state.scanCapabilities = fallbackCapabilities();
    }
    return state.scanCapabilities;
}

async function getLocalStatus() {
    try {
        return await api('/api/scan/v3/local/status');
    } catch (error) {
        console.warn('Local scan status failed:', error);
        return null;
    }
}

function isLocalOrLanHost(hostname) {
    if (['localhost', '127.0.0.1', '::1'].includes(hostname)) return true;
    if (/^10\.\d+\.\d+\.\d+$/.test(hostname)) return true;
    if (/^192\.168\.\d+\.\d+$/.test(hostname)) return true;
    const match = hostname.match(/^172\.(\d+)\.\d+\.\d+$/);
    return !!match && Number(match[1]) >= 16 && Number(match[1]) <= 31;
}

async function resolveScanMode() {
    const capabilities = await loadScanCapabilities();
    let mode = state.scanModePreference || 'auto';
    if (mode === 'auto') {
        const localCandidate = capabilities.local_direct_available && isLocalOrLanHost(window.location.hostname);
        if (localCandidate) {
            const status = await getLocalStatus();
            const camera = status?.camera_status || {};
            mode = camera.running || camera.opened || camera.has_frame ? 'local_direct' : 'browser_ws';
        } else {
            mode = capabilities.browser_stream_available ? 'browser_ws' : 'local_direct';
        }
    }
    if (!['local_direct', 'browser_ws'].includes(mode)) mode = 'browser_ws';
    state.effectiveScanMode = mode;
    return mode;
}

async function syncScanModeUI() {
    const mode = await resolveScanMode();
    const preference = state.scanModePreference || 'auto';
    const localMode = mode === 'local_direct';
    const sourceText = localMode
        ? 'Server Webcam - Local Direct Scan'
        : 'Browser Camera - Browser Stream Scan';
    const note = preference === 'auto'
        ? localMode
            ? 'Auto selected Local Direct because this page is running on localhost/LAN and the server camera is available.'
            : 'Auto selected Browser Stream because this page is remote or the server camera is unavailable.'
        : localMode
            ? 'Local Direct uses the webcam attached to the server machine.'
            : 'Browser Stream uses this browser camera and streams frames to the backend runtime.';

    document.querySelectorAll('[data-scan-mode]').forEach(button => {
        button.classList.toggle('active', button.dataset.scanMode === preference);
    });
    const sourceInfo = document.getElementById('source-info-text');
    if (sourceInfo) sourceInfo.textContent = sourceText;
    const cameraHint = document.getElementById('camera-hint');
    if (cameraHint) cameraHint.textContent = sourceText;
    const noteEl = document.getElementById('scan-mode-note');
    if (noteEl) noteEl.textContent = note;
}

export async function setScanMode(mode) {
    if (!SCAN_MODES.includes(mode)) return;
    state.scanModePreference = mode;
    localStorage.setItem('scanMode', mode);
    await syncScanModeUI();
    if (state.sessionActive) await checkAutoScanStatus();
}

export async function checkAutoScanStatus() {
    if (!state.sessionActive) return;
    try {
        const mode = await resolveScanMode();
        await syncScanModeUI();
        if (mode === 'local_direct') await activateLocalDirect();
        else await activateBrowserStream();
    } catch (error) {
        console.error('Scan start failed:', error);
        document.getElementById('autoscan-status').textContent = error.message || 'Failed to start scan';
        showToast(error.message || 'Failed to start scan', 'error');
        state.isAutoScanning = false;
        updateAutoScanUI();
    }
}

async function activateLocalDirect() {
    if (currentTransportMode === 'browser_ws') stopBrowserTransport();
    applyPreviewMode('local_direct');
    state.isAutoScanning = true;
    updateAutoScanUI();
    const status = await getLocalStatus();
    if (!['running', 'starting', 'challenge', 'cooldown'].includes(status?.runner_state)) {
        const result = await api('/api/scan/v3/local/start', { method: 'POST' });
        if (!result.success) throw new Error(result.message || 'Failed to start local direct scan');
    }
    openLocalSocket();
    currentTransportMode = 'local_direct';
    document.getElementById('autoscan-status').textContent = 'Local Direct: starting...';
}

async function activateBrowserStream() {
    if (currentTransportMode === 'local_direct') await stopLocalTransport(true);
    applyPreviewMode('browser_ws');
    await startBrowserCamera();
    openBrowserSocket();
    state.isAutoScanning = true;
    updateAutoScanUI();
    currentTransportMode = 'browser_ws';
    document.getElementById('autoscan-status').textContent = 'Browser Stream: starting...';
}

function applyPreviewMode(mode) {
    const feed = document.getElementById('camera-feed');
    const browserCam = document.getElementById('browser-cam');
    const demo = document.getElementById('demo-video');
    if (demo) hideElement(demo);
    state.demoPlaying = false;

    if (mode === 'local_direct') {
        if (!feed.getAttribute('src')) feed.setAttribute('src', feed.dataset.src || '/api/live/stream');
        showElement(feed, 'block');
        hideElement(browserCam);
        return;
    }

    hideElement(feed);
    showElement(browserCam, 'block');
}

export function startAutoScan() {
    void checkAutoScanStatus();
}

export function stopAutoScan() {
    closeLocalSocket();
    stopBrowserTransport();
    state.isAutoScanning = false;
    currentTransportMode = null;
    updateAutoScanUI();
}

export async function startLocalRunner() {
    await activateLocalDirect();
}

export async function stopLocalRunner() {
    await stopLocalTransport(true);
    state.isAutoScanning = false;
    updateAutoScanUI();
}

async function stopLocalTransport(stopRunner = false) {
    closeLocalSocket();
    if (stopRunner) {
        try {
            await api('/api/scan/v3/local/stop', { method: 'POST' });
        } catch (error) {
            console.warn('Local runner stop failed:', error);
        }
    }
    if (currentTransportMode === 'local_direct') currentTransportMode = null;
}

function updateAutoScanUI() {
    const bar = document.getElementById('autoscan-bar');
    if (state.isAutoScanning) showElement(bar, 'flex');
    else hideElement(bar);
}

function openLocalSocket() {
    if (localSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(localSocket.readyState)) return;
    closeLocalSocket();
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    localSocket = new WebSocket(`${protocol}//${window.location.host}/ws/scan-v3/local`);
    localSocket.onopen = () => document.getElementById('autoscan-status').textContent = 'Local Direct: connected';
    localSocket.onmessage = event => handleSocketMessage(event, 'local_direct');
    localSocket.onerror = () => document.getElementById('autoscan-status').textContent = 'Local Direct: connection error';
    localSocket.onclose = () => {
        localSocket = null;
        if (state.sessionActive && state.isAutoScanning && currentTransportMode === 'local_direct') {
            localReconnectTimer = setTimeout(openLocalSocket, LOCAL_WS_RECONNECT_MS);
        }
    };
}

function closeLocalSocket() {
    clearTimeout(localReconnectTimer);
    localReconnectTimer = null;
    if (!localSocket) return;
    const socket = localSocket;
    localSocket = null;
    socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
    if ([WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) socket.close();
}

async function startBrowserCamera() {
    if (state.localStream) return;
    const browserCam = document.getElementById('browser-cam');
    state.localStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
    });
    browserCam.srcObject = state.localStream;
    await browserCam.play().catch(() => {});
}

function stopBrowserCamera() {
    if (state.localStream) {
        state.localStream.getTracks().forEach(track => track.stop());
        state.localStream = null;
    }
    const browserCam = document.getElementById('browser-cam');
    if (browserCam) {
        browserCam.pause();
        browserCam.srcObject = null;
    }
}

function openBrowserSocket() {
    if (browserSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(browserSocket.readyState)) return;
    closeBrowserSocket();
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    browserSocket = new WebSocket(`${protocol}//${window.location.host}/ws/scan-v3`);
    browserSocket.binaryType = 'arraybuffer';
    browserSocket.onopen = () => document.getElementById('autoscan-status').textContent = 'Browser Stream: connected';
    browserSocket.onmessage = event => handleSocketMessage(event, 'browser_ws');
    browserSocket.onerror = () => document.getElementById('autoscan-status').textContent = 'Browser Stream: connection error';
    browserSocket.onclose = () => {
        browserSocket = null;
        stopBrowserFrameLoop();
        if (state.sessionActive && state.isAutoScanning && currentTransportMode === 'browser_ws') {
            browserReconnectTimer = setTimeout(openBrowserSocket, BROWSER_WS_RECONNECT_MS);
        }
    };
}

function closeBrowserSocket() {
    clearTimeout(browserReconnectTimer);
    browserReconnectTimer = null;
    stopBrowserFrameLoop();
    if (!browserSocket) return;
    const socket = browserSocket;
    browserSocket = null;
    socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
    if ([WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) socket.close();
}

function stopBrowserTransport() {
    closeBrowserSocket();
    stopBrowserCamera();
    if (currentTransportMode === 'browser_ws') currentTransportMode = null;
}

function startBrowserFrameLoop() {
    stopBrowserFrameLoop();
    const intervalMs = Math.max(100, Math.round(1000 / Math.max(browserTargetFps, 1)));
    browserSendTimer = setInterval(() => void sendBrowserFrame(), intervalMs);
}

function stopBrowserFrameLoop() {
    if (browserSendTimer) clearInterval(browserSendTimer);
    browserSendTimer = null;
    browserSending = false;
}

async function sendBrowserFrame() {
    if (browserSending || !browserSocket || browserSocket.readyState !== WebSocket.OPEN) return;
    const video = document.getElementById('browser-cam');
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return;
    browserSending = true;
    try {
        const blob = await captureVideoFrame(video);
        if (!blob || !browserSocket || browserSocket.readyState !== WebSocket.OPEN) return;
        latestBrowserFrameBlob = blob;
        browserSocket.send(await blob.arrayBuffer());
    } finally {
        browserSending = false;
    }
}

function captureVideoFrame(video) {
    const scale = Math.min(1, browserFrameWidth / video.videoWidth);
    const width = Math.max(1, Math.round(video.videoWidth * scale));
    const height = Math.max(1, Math.round(video.videoHeight * scale));
    browserCanvas.width = width;
    browserCanvas.height = height;
    browserCtx.drawImage(video, 0, 0, width, height);
    return new Promise(resolve => browserCanvas.toBlob(resolve, 'image/jpeg', browserJpegQuality));
}

function handleSocketMessage(event, sourceMode) {
    let message;
    try {
        message = JSON.parse(event.data);
    } catch {
        document.getElementById('autoscan-status').textContent = 'Invalid scan message';
        return;
    }
    handleRuntimeMessage(message, sourceMode);
}

function handleRuntimeMessage(message, sourceMode) {
    if (message.type === 'stream_ready') {
        if (sourceMode === 'browser_ws') {
            browserTargetFps = message.target_fps || browserTargetFps;
            browserFrameWidth = message.client_frame_width || browserFrameWidth;
            browserJpegQuality = message.client_jpeg_quality || browserJpegQuality;
            startBrowserFrameLoop();
        }
        updateStreamStatus(message, sourceMode);
        if (sourceMode === 'local_direct' && message.last_event) {
            handleRuntimeMessage(message.last_event, sourceMode);
        }
        return;
    }
    if (message.type === 'heartbeat') return;
    if (message.type === 'error') {
        document.getElementById('autoscan-status').textContent = message.message || 'Scan error';
        if (message.status === 'camera_error') showToast('Camera not available', 'error');
        return;
    }
    if (message.type === 'attendance') {
        handleAttendanceEvent(message, sourceMode);
        return;
    }
    if (message.type === 'challenge_required') {
        handleChallengeRequired(message);
        updateStreamStatus(message, sourceMode);
        return;
    }
    if (message.status === 'challenge_active') {
        handleChallengeProgress(message);
        updateStreamStatus(message, sourceMode);
        return;
    }
    if (message.status === 'challenge_failed') {
        handleChallengeFailed(message);
        return;
    }
    maybeShowScanFailure(message.result || message);
    updateStreamStatus(message, sourceMode);
}

function updateStreamStatus(message, sourceMode) {
    const prefix = sourceMode === 'local_direct' ? 'Local Direct' : 'Browser Stream';
    document.getElementById('autoscan-status').textContent =
        `${prefix}: ${message.message || message.status || 'Scanning...'}`;
}

function handleChallengeRequired(message) {
    if (!message.challenge) return;
    state.isChallengeRunning = true;
    state.activeChallenge = message.challenge;
    void showScanChallengeOverlay(message);
}

function handleChallengeProgress(message) {
    const progress = message.challenge_progress;
    if (!progress) return;
    if (progress.remaining_ms != null) {
        updateChallengeCountdown(Math.max(1, Math.ceil(progress.remaining_ms / 1000)));
    }
    const collected = progress.collected_frames ?? progress.frames_processed ?? 0;
    const required = Math.max(progress.required_frames || 1, 1);
    updateChallengeProgress(Math.min(100, Math.round((collected / required) * 100)), collected);
    document.getElementById('scan-challenge-status').textContent =
        progress.feedback || progress.label || 'Verifying...';
}

function handleChallengeFailed(message) {
    void hideScanChallengeOverlay();
    state.isChallengeRunning = false;
    state.activeChallenge = null;
    showResultCard('Challenge Failed', 'Challenge Failed', false, message.message || 'Verification failed');
}

function handleAttendanceEvent(result, sourceMode) {
    if (result.challenge_passed) {
        void hideScanChallengeOverlay();
        state.isChallengeRunning = false;
        state.activeChallenge = null;
    }
    if (sourceMode === 'browser_ws' && latestBrowserFrameBlob) {
        result.local_preview_url = URL.createObjectURL(latestBrowserFrameBlob);
    }
    const isMatch = result.status === 'present' || result.status === 'already';
    const alreadySeen = !!(result.student_id && state.presentSet.has(result.student_id));
    if (isMatch && result.student_id && !alreadySeen) {
        state.presentSet.add(result.student_id);
        state.presentStudents.set(result.student_id, {
            id: result.student_id,
            name: result.name || result.student_id,
            className: result.class_name || '',
            status: result.status
        });
        state.scanCount += 1;
        updateScanLogs();
        enqueueAttendanceSuccess(result);
        showResultCard(
            result.student_id,
            result.name,
            true,
            `Matched (Confidence: ${(result.confidence * 100).toFixed(0)}%)`
        );
    }
    document.getElementById('autoscan-count').textContent = `${state.scanCount} scanned`;
    document.getElementById('autoscan-status').textContent =
        isMatch ? `Matched: ${result.name}` : (result.message || 'Scanning...');
}

async function showScanChallengeOverlay(result) {
    const overlay = document.getElementById('scan-challenge-overlay');
    const viewfinder = document.querySelector('.viewfinder');
    if (!overlay) return;
    document.getElementById('scan-challenge-title').textContent = result.challenge.label || 'Security Challenge';
    document.getElementById('scan-challenge-instruction').textContent =
        result.challenge.instruction || 'Follow the prompt on camera';
    document.getElementById('scan-challenge-status').textContent =
        result.message || 'Additional verification required';
    updateChallengeAction(result.challenge.type);
    updateChallengeCountdown(Math.max(1, Math.ceil((result.challenge.remaining_ms || 6000) / 1000)));
    updateChallengeProgress(0, 0);
    overlay.classList.remove('is-visible');
    showElement(overlay, 'flex');
    viewfinder?.classList.add('is-challenging');
    await nextFrame();
    overlay.classList.add('is-visible');
}

async function hideScanChallengeOverlay() {
    const overlay = document.getElementById('scan-challenge-overlay');
    const viewfinder = document.querySelector('.viewfinder');
    if (!overlay) return;
    overlay.classList.remove('is-visible');
    await delay(180);
    hideElement(overlay);
    viewfinder?.classList.remove('is-challenging');
}

function updateChallengeCountdown(value) {
    const el = document.getElementById('scan-challenge-countdown');
    if (!el) return;
    el.textContent = String(value);
    el.classList.remove('is-ticking');
    void el.offsetWidth;
    el.classList.add('is-ticking');
}

function updateChallengeAction(type = '') {
    const action = document.getElementById('scan-challenge-action');
    const arrow = document.getElementById('scan-challenge-arrow');
    if (!action || !arrow) return;
    const map = {
        turn_left: { cls: 'left', icon: 'arrow_back' },
        turn_right: { cls: 'right', icon: 'arrow_forward' },
        blink: { cls: 'blink', icon: 'visibility_off' },
        center: { cls: 'front', icon: 'arrow_upward' }
    };
    const meta = map[type] || map.center;
    action.className = `scan-challenge-action ${meta.cls}`;
    arrow.textContent = meta.icon;
}

function updateChallengeProgress(percent, frameCount) {
    const bar = document.getElementById('scan-challenge-progress-bar');
    if (bar) bar.style.width = `${percent}%`;
    const status = document.getElementById('scan-challenge-status');
    if (status && frameCount > 0) status.dataset.frames = String(frameCount);
}

function maybeShowScanFailure(result) {
    const now = Date.now();
    if (now - lastFailureCardAt < FAILURE_CARD_THROTTLE_MS) return;
    const title = result.status === 'spoof'
        ? result.moire_is_screen ? 'Screen Spoof Blocked' : 'Liveness Blocked'
        : result.status === 'unknown'
            ? 'Unknown'
            : '';
    if (!title) return;
    lastFailureCardAt = now;
    showResultCard(title, title, false, buildScanFailureMessage(result));
}

function enqueueAttendanceSuccess(result) {
    successOverlayQueue.push(result);
    if (!isShowingSuccessOverlay) void showNextAttendanceSuccess();
}

async function showNextAttendanceSuccess() {
    if (isClosingSuccessOverlay) return;
    const result = successOverlayQueue.shift();
    if (!result) {
        isShowingSuccessOverlay = false;
        activeSuccessResult = null;
        return;
    }
    isShowingSuccessOverlay = true;
    activeSuccessResult = result;
    renderAttendanceSuccess(result);
    await showAttendanceSuccessOverlay();
    startSuccessCountdown();
    successOverlayTimer = setTimeout(() => void closeCurrentAttendanceSuccess(), SUCCESS_OVERLAY_DISPLAY_MS);
}

function renderAttendanceSuccess(result) {
    const overlay = document.getElementById('attendance-success-overlay');
    const img = document.getElementById('success-face-img');
    const studentId = result.student_id || '--';
    const fallbackUrl = `/api/students/${studentId}/photo`;
    const imageUrl = result.local_preview_url || result.evidence_url || fallbackUrl;
    if (overlay) {
        overlay.dataset.status = result.status === 'already' ? 'already' : 'present';
        overlay.classList.remove('is-visible', 'is-leaving');
    }
    img.classList.remove('is-loaded', 'is-hidden');
    img.onerror = () => {
        if (imageUrl !== fallbackUrl) {
            img.onerror = () => img.classList.add('is-hidden');
            img.src = fallbackUrl;
            return;
        }
        img.classList.add('is-hidden');
    };
    img.onload = () => img.classList.add('is-loaded');
    img.src = `${imageUrl}${imageUrl.includes('?') ? '&' : '?'}t=${Date.now()}`;
    document.getElementById('success-status-chip').textContent = result.status === 'already' ? 'ALREADY' : 'PRESENT';
    document.getElementById('success-title').textContent = result.status === 'already' ? 'Already Checked In' : 'Attendance Recorded';
    document.getElementById('success-subtitle').textContent = result.status === 'already' ? 'Student already marked present' : 'Saved to attendance log';
    document.getElementById('success-student-name').textContent = result.name || 'Unknown Student';
    document.getElementById('success-student-meta').textContent = `ID: ${studentId} - Class: ${result.class_name || '--'}`;
    document.getElementById('success-session-name').textContent =
        result.session_name ? `Session: ${result.session_name}` : (result.session_id ? `Session #${result.session_id}` : 'Active session');
    document.getElementById('success-confidence').textContent =
        typeof result.confidence === 'number' ? `Confidence: ${(result.confidence * 100).toFixed(0)}%` : 'Confidence: --';
    document.getElementById('success-timestamp').textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
    updateSuccessCountdown(SUCCESS_COUNTDOWN_START);
}

async function showAttendanceSuccessOverlay() {
    const overlay = document.getElementById('attendance-success-overlay');
    const viewfinder = document.querySelector('.viewfinder');
    if (!overlay) return;
    overlay.classList.remove('is-leaving', 'is-visible');
    showElement(overlay, 'flex');
    viewfinder?.classList.add('is-confirming');
    await nextFrame();
    overlay.classList.add('is-visible');
}

async function hideAttendanceSuccessOverlay() {
    const overlay = document.getElementById('attendance-success-overlay');
    const viewfinder = document.querySelector('.viewfinder');
    if (!overlay) return;
    overlay.classList.remove('is-visible');
    overlay.classList.add('is-leaving');
    await delay(SUCCESS_OVERLAY_EXIT_MS);
    hideElement(overlay);
    overlay.classList.remove('is-leaving');
    viewfinder?.classList.remove('is-confirming');
}

async function closeCurrentAttendanceSuccess() {
    if (isClosingSuccessOverlay) return;
    isClosingSuccessOverlay = true;
    clearSuccessTimers();
    await hideAttendanceSuccessOverlay();
    revokePreviewUrl(activeSuccessResult);
    activeSuccessResult = null;
    isClosingSuccessOverlay = false;
    setTimeout(() => void showNextAttendanceSuccess(), 120);
}

function startSuccessCountdown() {
    let remaining = SUCCESS_COUNTDOWN_START;
    updateSuccessCountdown(remaining);
    clearInterval(successCountdownTimer);
    successCountdownTimer = setInterval(() => {
        remaining -= 1;
        if (remaining < 1) {
            clearInterval(successCountdownTimer);
            successCountdownTimer = null;
            return;
        }
        updateSuccessCountdown(remaining);
    }, 1000);
}

function updateSuccessCountdown(value) {
    const el = document.getElementById('success-countdown-number');
    if (!el) return;
    el.textContent = String(value);
    el.classList.remove('is-ticking');
    void el.offsetWidth;
    el.classList.add('is-ticking');
}

function clearSuccessTimers() {
    if (successOverlayTimer) clearTimeout(successOverlayTimer);
    if (successCountdownTimer) clearInterval(successCountdownTimer);
    successOverlayTimer = null;
    successCountdownTimer = null;
}

function revokePreviewUrl(result) {
    if (result?.local_preview_url?.startsWith('blob:')) {
        URL.revokeObjectURL(result.local_preview_url);
    }
}

function nextFrame() {
    return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function buildScanFailureMessage(result) {
    const parts = [result.message || 'Low confidence/Spoof'];
    if (typeof result.moire_score === 'number') parts.push(`Moire score: ${(result.moire_score * 100).toFixed(0)}%`);
    if (typeof result.liveness_score === 'number') parts.push(`Liveness score: ${(result.liveness_score * 100).toFixed(0)}%`);
    if (typeof result.confidence === 'number' && result.confidence > 0) parts.push(`Confidence: ${(result.confidence * 100).toFixed(0)}%`);
    return parts.join(' - ');
}

function showResultCard(id, name, success, desc) {
    const card = document.getElementById('result-card');
    const icon = document.getElementById('result-icon');
    card.className = success ? 'scan-result-card success' : 'scan-result-card error';
    showElement(card, 'flex');
    icon.textContent = success ? 'check_circle' : 'cancel';
    setTone(icon, success ? 'primary' : 'error');
    document.getElementById('result-name').textContent = name;
    document.getElementById('result-msg').textContent = desc;
    setTimeout(() => hideElement(card), 4000);
}

export function updateScanLogs() {
    const list = document.getElementById('present-list');
    const count = document.getElementById('present-count');
    list.innerHTML = '';
    count.textContent = `${state.presentStudents.size || state.scanCount} total`;
    if (state.presentStudents.size === 0 && state.scanCount === 0) {
        list.innerHTML = '<p class="empty-state">No data yet</p>';
        document.getElementById('progress-pct').textContent = '0%';
        document.getElementById('progress-bar').style.width = '0%';
        return;
    }
    state.presentSet.forEach(id => {
        const student = state.presentStudents.get(id) || { id, name: id, className: '' };
        const item = document.createElement('div');
        item.className = 'scan-log-item';
        item.innerHTML = `
            <img src="/api/students/${id}/photo" onerror="this.classList.add('is-hidden')">
            <div>
                <p>${student.name}</p>
                <span>${student.id}${student.className ? ` - ${student.className}` : ''}</span>
            </div>
            <span class="material-symbols-outlined">check_circle</span>
        `;
        list.appendChild(item);
    });
    const total = Math.max(state.scanCount, 10);
    const pct = Math.min(Math.round((state.scanCount / total) * 100), 100);
    document.getElementById('progress-pct').textContent = `${pct}%`;
    document.getElementById('progress-bar').style.width = `${pct}%`;
}

export function skipAttendanceSuccessOverlay() {
    if (!isShowingSuccessOverlay || isClosingSuccessOverlay) return;
    void closeCurrentAttendanceSuccess();
}

export async function testAttendanceSuccessOverlay(status = 'present') {
    enqueueAttendanceSuccess({
        status,
        name: status === 'already' ? 'Already Checked Student' : 'Sample Student',
        student_id: status === 'already' ? 'HS002' : 'HS001',
        class_name: '12A1',
        session_id: 999,
        session_name: 'Popup Test Session',
        confidence: status === 'already' ? 0.91 : 0.87,
    });
}

export function updateAutoScanInterval() {
    // Backend runtime controls scan cadence.
}

export function setCameraSource(source = '') {
    // Legacy source buttons are remapped to scan mode.
    if (source === 'browser') void setScanMode('browser_ws');
    else if (source === 'webcam') void setScanMode('local_direct');
}

export function toggleDemoVideo() {
    const video = document.getElementById('demo-video');
    const feed = document.getElementById('camera-feed');
    const browserCam = document.getElementById('browser-cam');
    if (!video) return;
    if (state.demoPlaying) {
        video.pause();
        hideElement(video);
        if (state.effectiveScanMode === 'local_direct') showElement(feed, 'block');
        else showElement(browserCam, 'block');
        state.demoPlaying = false;
    } else {
        hideElement(feed);
        hideElement(browserCam);
        showElement(video, 'block');
        video.play();
        state.demoPlaying = true;
    }
}
