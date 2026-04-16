import { api } from './api.js';
import { state } from './state.js';
import { hideElement, setTone, showElement, showToast } from './ui.js';

const successOverlayQueue = [];
let isShowingSuccessOverlay = false;
const SUCCESS_OVERLAY_DISPLAY_MS = 3400;
const SUCCESS_OVERLAY_EXIT_MS = 220;
const SUCCESS_COUNTDOWN_START = 3;
const CHALLENGE_CAPTURE_MS = 3200;
const CHALLENGE_FRAME_INTERVAL_MS = 260;
const BLINK_CHALLENGE_CAPTURE_MS = 4200;
const BLINK_CHALLENGE_FRAME_INTERVAL_MS = 120;
const CHALLENGE_COUNTDOWN_START = 3;
let successOverlayTimer = null;
let successCountdownTimer = null;
let activeSuccessResult = null;
let isClosingSuccessOverlay = false;

export async function setCameraSource(source) {
    document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`src-${source}`).classList.add('active');

    state.cameraSource = source;
    state.isAutoScanning = false;
    updateAutoScanUI();

    const feed = document.getElementById('camera-feed');
    const browserCam = document.getElementById('browser-cam');

    if (state.localStream) {
        state.localStream.getTracks().forEach(t => t.stop());
        state.localStream = null;
    }

    if (source === 'browser') {
        hideElement(feed);
        showElement(browserCam, 'block');
        try {
            state.localStream = await navigator.mediaDevices.getUserMedia({ video: true });
            browserCam.srcObject = state.localStream;
            document.getElementById('source-info-text').textContent = 'Local Browser Camera';
            if (state.sessionActive) startAutoScan();
        } catch (e) {
            showToast('Camera access denied', 'error');
            setCameraSource('webcam');
        }
    } else if (source === 'phone') {
        hideElement(feed);
        hideElement(browserCam);
        document.getElementById('source-info-text').textContent = 'Android Phone Camera';
    } else {
        showElement(feed, 'block');
        hideElement(browserCam);
        document.getElementById('source-info-text').textContent = 'Server Webcam Stream';
        if (state.sessionActive) startAutoScan();
    }
}

export function updateAutoScanInterval() {
    state.scanIntervalMs = parseInt(document.getElementById('autoscan-interval').value, 10);
    if (state.isAutoScanning) {
        stopAutoScan();
        startAutoScan();
    }
}

export async function testAttendanceSuccessOverlay(status = 'present') {
    const frameBlob = await captureFrameFromSource();
    enqueueAttendanceSuccess({
        status,
        name: status === 'already' ? 'Nguyen Van An' : 'Vo Minh Sang',
        student_id: status === 'already' ? 'HS001' : 'ha001',
        class_name: status === 'already' ? '12A1' : '10A1',
        session_id: 999,
        session_name: 'Popup Test Session',
        confidence: status === 'already' ? 0.91 : 0.87,
        evidence_url: '/api/students/HS001/photo',
        local_preview_url: frameBlob ? URL.createObjectURL(frameBlob) : null
    });
}

export function skipAttendanceSuccessOverlay() {
    if (!isShowingSuccessOverlay || isClosingSuccessOverlay) return;
    closeCurrentAttendanceSuccess();
}

export async function checkAutoScanStatus() {
    if (state.sessionActive && !state.isAutoScanning && state.cameraSource !== 'phone') {
        startAutoScan();
    }
}

function updateAutoScanUI() {
    const bar = document.getElementById('autoscan-bar');
    if (state.isAutoScanning) showElement(bar, 'flex');
    else hideElement(bar);
}

export function startAutoScan() {
    if (state.autoScanInterval) clearInterval(state.autoScanInterval);
    state.isAutoScanning = true;
    updateAutoScanUI();
    state.autoScanInterval = setInterval(performScanV3, state.scanIntervalMs);
}

export function stopAutoScan() {
    state.isAutoScanning = false;
    updateAutoScanUI();
    if (state.autoScanInterval) clearInterval(state.autoScanInterval);
}

async function captureFrameFromSource() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (state.cameraSource === 'browser') {
        const video = document.getElementById('browser-cam');
        if (!video.videoWidth) return null;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
    } else {
        const img = document.getElementById('camera-feed');
        if (!img.naturalWidth) return null;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        try {
            ctx.drawImage(img, 0, 0);
        } catch (e) {
            return null;
        }
    }
    return new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
}

async function performScanV3() {
    if (!state.sessionActive || !state.isAutoScanning || state.isChallengeRunning) return;
    document.getElementById('autoscan-status').textContent = 'Scanning...';

    const blob = await captureFrameFromSource();
    if (!blob) {
        document.getElementById('autoscan-status').textContent = 'Waiting for frame...';
        return;
    }

    const formData = new FormData();
    formData.append('image', blob, 'scan.jpg');

    try {
        const data = await api('/api/scan/v3', { method: 'POST', body: formData });
        if (data.results && data.results.length > 0) {
            handleScanV3Results(data.results, blob);
        } else {
            document.getElementById('autoscan-status').textContent = 'No face detected';
        }
    } catch (e) {
        document.getElementById('autoscan-status').textContent = 'Scan error';
    }
}

function handleScanV3Results(results, frameBlob = null) {
    let validFound = 0;

    results.forEach(r => {
        const isMatch = r.status === 'present' || r.status === 'already';
        if (r.status === 'challenge_required') {
            startScanChallenge(r);
            return;
        }

        if (isMatch && r.student_id && !state.presentSet.has(r.student_id)) {
            state.presentSet.add(r.student_id);
            state.presentStudents.set(r.student_id, {
                id: r.student_id,
                name: r.name || r.student_id,
                className: r.class_name || '',
                status: r.status
            });
            state.scanCount++;
            validFound++;
            showResultCard(r.student_id, r.name, true, `Matched (Confidence: ${(r.confidence * 100).toFixed(0)}%)`);
            if (r.status === 'present' || r.status === 'already') {
                enqueueAttendanceSuccess({
                    ...r,
                    local_preview_url: frameBlob ? URL.createObjectURL(frameBlob) : null
                });
            }
        } else if (r.status === 'already') {
            // Already marked, skip silently.
        } else if (!isMatch) {
            const isScreenSpoof = r.status === 'spoof' && r.moire_is_screen === true;
            const title = isScreenSpoof
                ? 'Screen Spoof Blocked'
                : r.status === 'spoof'
                    ? 'Liveness Blocked'
                    : r.status === 'challenge_failed'
                        ? 'Challenge Failed'
                    : 'Unknown';
            showResultCard(title, title, false, buildScanFailureMessage(r));
        }
    });

    if (validFound > 0) {
        document.getElementById('autoscan-status').textContent = `Matched ${validFound} student(s)!`;
        document.getElementById('autoscan-count').textContent = `${state.scanCount} scanned`;
        updateScanLogs();
    } else {
        document.getElementById('autoscan-status').textContent = 'Waiting...';
    }
}

async function startScanChallenge(result) {
    if (state.isChallengeRunning || !result.challenge) return;

    const wasAutoScanning = state.isAutoScanning;
    state.isChallengeRunning = true;
    state.activeChallenge = result.challenge;
    state.challengeFrames = [];
    stopAutoScan();

    showScanChallengeOverlay(result);

    try {
        const frames = await captureChallengeFrames(result.challenge.type);
        state.challengeFrames = frames;

        if (frames.length === 0) {
            showResultCard('Challenge Failed', 'Challenge Failed', false, 'No camera frames captured');
            return;
        }

        document.getElementById('scan-challenge-status').textContent = 'Verifying challenge...';
        const fd = new FormData();
        fd.append('challenge_id', result.challenge.id);
        frames.forEach((blob, index) => {
            fd.append('frames', blob, `challenge_${index}.jpg`);
        });

        const data = await api('/api/scan/v3/challenge', { method: 'POST', body: fd });
        const previewFrame = frames[frames.length - 1] || null;
        if (data.results && data.results.length > 0) {
            handleScanV3Results(data.results, previewFrame);
        } else {
            showResultCard('Challenge Failed', 'Challenge Failed', false, data.message || 'Verification failed');
        }
    } catch (e) {
        showResultCard('Challenge Error', 'Challenge Error', false, e.message || 'Verification error');
    } finally {
        await hideScanChallengeOverlay();
        state.activeChallenge = null;
        state.challengeFrames = [];
        state.isChallengeRunning = false;
        if (state.sessionActive && wasAutoScanning) startAutoScan();
    }
}

async function captureChallengeFrames(type = '') {
    const frames = [];
    const started = Date.now();
    const timing = getChallengeTiming(type);
    let lastCountdown = null;
    let index = 0;

    while (Date.now() - started < timing.duration && state.isChallengeRunning) {
        const elapsed = Date.now() - started;
        const progress = Math.min(100, Math.round((elapsed / timing.duration) * 100));
        const remaining = Math.max(
            1,
            Math.ceil((timing.duration - elapsed) / 1000)
        );

        if (remaining !== lastCountdown) {
            updateChallengeCountdown(remaining);
            lastCountdown = remaining;
        }

        const blob = await captureFrameFromSource();
        if (blob) frames.push(blob);

        index += 1;
        updateChallengeProgress(progress, frames.length);
        document.getElementById('scan-challenge-status').textContent =
            type === 'blink'
                ? `Blink once - ${frames.length} frames captured`
                : `Hold this position - ${frames.length} frames captured`;

        await delay(timing.interval);
    }

    updateChallengeProgress(100, frames.length);
    return frames;
}

async function showScanChallengeOverlay(result) {
    const overlay = document.getElementById('scan-challenge-overlay');
    const viewfinder = document.querySelector('.viewfinder');
    if (!overlay) return;

    document.getElementById('scan-challenge-title').textContent =
        result.challenge.label || 'Security Challenge';
    document.getElementById('scan-challenge-instruction').textContent =
        result.challenge.instruction || 'Follow the prompt';
    document.getElementById('scan-challenge-status').textContent =
        result.message || 'Additional verification required';
    updateChallengeAction(result.challenge.type);
    updateChallengeCountdown(CHALLENGE_COUNTDOWN_START);
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
        center: { cls: 'front', icon: 'arrow_upward' },
        blink: { cls: 'blink', icon: 'visibility_off' }
    };
    const meta = map[type] || map.center;
    action.className = `scan-challenge-action ${meta.cls}`;
    arrow.textContent = meta.icon;
}

function getChallengeTiming(type = '') {
    if (type === 'blink') {
        return {
            duration: BLINK_CHALLENGE_CAPTURE_MS,
            interval: BLINK_CHALLENGE_FRAME_INTERVAL_MS
        };
    }
    return {
        duration: CHALLENGE_CAPTURE_MS,
        interval: CHALLENGE_FRAME_INTERVAL_MS
    };
}

function updateChallengeProgress(percent, frameCount) {
    const bar = document.getElementById('scan-challenge-progress-bar');
    if (bar) bar.style.width = `${percent}%`;
    const status = document.getElementById('scan-challenge-status');
    if (status && frameCount > 0) status.dataset.frames = String(frameCount);
}

function enqueueAttendanceSuccess(result) {
    successOverlayQueue.push(result);
    if (!isShowingSuccessOverlay) showNextAttendanceSuccess();
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
    await waitForSuccessPhotoReady(result);
    await showAttendanceSuccessOverlay();
    startSuccessCountdown();

    successOverlayTimer = setTimeout(() => {
        closeCurrentAttendanceSuccess();
    }, SUCCESS_OVERLAY_DISPLAY_MS);
}

function renderAttendanceSuccess(result) {
    const overlay = document.getElementById('attendance-success-overlay');
    const img = document.getElementById('success-face-img');
    const studentId = result.student_id || '--';
    const fallbackUrl = `/api/students/${studentId}/photo`;
    const evidenceUrl = result.evidence_url || fallbackUrl;
    const primaryUrl = result.local_preview_url || evidenceUrl;

    if (overlay) {
        overlay.dataset.status = result.status === 'already' ? 'already' : 'present';
        overlay.classList.remove('is-visible', 'is-leaving');
    }

    img.classList.remove('is-loaded');
    img.classList.remove('is-hidden');
    img.onerror = () => {
        img.onerror = () => img.classList.add('is-hidden');
        img.src = fallbackUrl;
    };
    img.onload = () => img.classList.add('is-loaded');
    img.src = result.local_preview_url
        ? primaryUrl
        : `${primaryUrl}${primaryUrl.includes('?') ? '&' : '?'}t=${Date.now()}`;

    document.getElementById('success-student-name').textContent = result.name || 'Unknown Student';
    document.getElementById('success-status-chip').textContent = result.status === 'already' ? 'ALREADY' : 'PRESENT';
    document.getElementById('success-title').textContent =
        result.status === 'already' ? 'Already Checked In' : 'Attendance Recorded';
    document.getElementById('success-subtitle').textContent =
        result.status === 'already' ? 'Student already marked present' : 'Saved to attendance log';
    document.getElementById('success-student-meta').textContent =
        `ID: ${studentId} - Class: ${result.class_name || '--'}`;
    document.getElementById('success-session-name').textContent =
        result.session_name
            ? `Session: ${result.session_name}`
            : (result.session_id ? `Session #${result.session_id}` : 'Active session');
    document.getElementById('success-confidence').textContent =
        typeof result.confidence === 'number'
            ? `Confidence: ${(result.confidence * 100).toFixed(0)}%`
            : 'Confidence: --';
    document.getElementById('success-timestamp').textContent =
        new Date().toLocaleTimeString('en-US', { hour12: false });
    updateSuccessCountdown(SUCCESS_COUNTDOWN_START);
}

async function waitForSuccessPhotoReady(result) {
    const img = document.getElementById('success-face-img');
    if (!img) return;
    if (img.complete && img.naturalWidth > 0) return;

    await new Promise(resolve => {
        const done = () => resolve();
        img.addEventListener('load', done, { once: true });
        img.addEventListener('error', done, { once: true });
        setTimeout(done, result.local_preview_url ? 220 : 320);
    });

    if (typeof img.decode === 'function' && img.naturalWidth > 0) {
        try {
            await img.decode();
        } catch (e) {
            // The image is already usable if load fired; decode can fail on quick src swaps.
        }
    }
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
    cleanupAttendanceSuccess(activeSuccessResult);
    activeSuccessResult = null;
    isClosingSuccessOverlay = false;
    setTimeout(showNextAttendanceSuccess, 120);
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
    if (successOverlayTimer) {
        clearTimeout(successOverlayTimer);
        successOverlayTimer = null;
    }
    if (successCountdownTimer) {
        clearInterval(successCountdownTimer);
        successCountdownTimer = null;
    }
}

function nextFrame() {
    return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function cleanupAttendanceSuccess(result) {
    if (result.local_preview_url) {
        URL.revokeObjectURL(result.local_preview_url);
        result.local_preview_url = null;
    }
}

function buildScanFailureMessage(result) {
    const parts = [result.message || 'Low confidence/Spoof'];
    if (typeof result.moire_score === 'number') {
        parts.push(`Moire score: ${(result.moire_score * 100).toFixed(0)}%`);
    }
    if (typeof result.liveness_score === 'number') {
        parts.push(`Liveness score: ${(result.liveness_score * 100).toFixed(0)}%`);
    }
    if (typeof result.confidence === 'number' && result.confidence > 0) {
        parts.push(`Confidence: ${(result.confidence * 100).toFixed(0)}%`);
    }
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
    const ct = document.getElementById('present-count');

    list.innerHTML = '';
    ct.textContent = `${state.presentStudents.size || state.scanCount} total`;

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

export function toggleDemoVideo() {
    const video = document.getElementById('demo-video');
    const label = document.getElementById('btn-demo-label');
    const feed = document.getElementById('camera-feed');
    const browserCam = document.getElementById('browser-cam');

    if (state.demoPlaying) {
        video.pause();
        hideElement(video);
        if (state.cameraSource === 'browser') {
            hideElement(feed);
            showElement(browserCam, 'block');
        } else {
            showElement(feed, 'block');
            hideElement(browserCam);
        }
        label.textContent = 'Demo';
        state.demoPlaying = false;
    } else {
        hideElement(feed);
        hideElement(browserCam);
        showElement(video, 'block');
        video.play();
        label.textContent = 'Stop Demo';
        state.demoPlaying = true;
    }
}
