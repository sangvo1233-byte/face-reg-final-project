import { api } from './api.js';
import {
    createEnrollState,
    ENROLL_CAPTURE_INTERVAL_MS,
    ENROLL_PHASE_META,
    ENROLL_PHASES,
    ENROLL_REQUIRED_GOOD_FRAMES,
    setEnrollState,
    state
} from './state.js';
import { closeModal, hideElement, hideLoading, setTone, showElement, showLoading, showModal, showToast } from './ui.js';
import { loadStudents } from './students.js';

let manualValidated = { front: false, left: false, right: false };

export function showStudentCreateModal() {
    document.getElementById('new-student-id').value = '';
    document.getElementById('new-student-name').value = '';
    document.getElementById('new-student-class').value = '';
    showModal('student-create-modal');
}

function validateNewStudentForm() {
    const id = document.getElementById('new-student-id').value.trim();
    const name = document.getElementById('new-student-name').value.trim();
    if (!id || !name) {
        showToast('ID and Name are required', 'error');
        return null;
    }
    return { id, name, class: document.getElementById('new-student-class').value.trim() };
}

export function startReEnroll(id, name, cls) {
    state.enrollState.id = id;
    state.enrollState.name = name;
    state.enrollState.className = cls;
    document.getElementById('new-student-id').value = id;
    document.getElementById('new-student-name').value = name;
    document.getElementById('new-student-class').value = cls || '';
    showModal('student-create-modal');
}

export function proceedToCameraEnroll() {
    const s = validateNewStudentForm();
    if (!s) return;

    setEnrollState(createEnrollState({ ...s }));
    closeModal('student-create-modal');
    document.getElementById('enroll-cam-title').textContent = state.enrollState.name;
    document.getElementById('enroll-cam-subtitle').textContent = `ID: ${state.enrollState.id}`;

    ENROLL_PHASES.forEach(p => {
        const c = document.getElementById(`thumb-card-${p}`);
        c.className = 'thumb-card';
        hideElement(c.querySelector('.thumb-check'));
        showElement(c.querySelector('.thumb-placeholder'), 'flex');
        c.querySelector('.thumb-placeholder').classList.remove('error');
        hideElement(c.querySelector('.thumb-img'));
    });

    document.getElementById('enroll-submit-btn').disabled = true;
    setEnrollAutoStatus(`Auto capture needs ${ENROLL_REQUIRED_GOOD_FRAMES} good frames per angle.`);
    showModal('enroll-modal');
    startEnrollCamera();
    updateEnrollUI();
}

async function startEnrollCamera() {
    try {
        const video = document.getElementById('enroll-camera');
        state.enrollState.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = state.enrollState.stream;
        video.onloadedmetadata = () => startAutoEnrollCapture();
    } catch (e) {
        document.getElementById('enroll-instruction').textContent = 'Camera access denied or failed.';
        setTone('enroll-instruction', 'error');
        setEnrollAutoStatus('Camera access failed. Please allow camera permission.', 'error');
    }
}

function stopEnrollCamera() {
    stopAutoEnrollCapture();
    if (state.enrollState.stream) {
        state.enrollState.stream.getTracks().forEach(t => t.stop());
        state.enrollState.stream = null;
    }
}

export function closeEnrollCameraModal() {
    stopEnrollCamera();
    closeModal('enroll-modal');
}

function setEnrollAutoStatus(message, type = '') {
    const el = document.getElementById('enroll-auto-status');
    if (!el) return;
    el.textContent = message;
    el.className = `auto-capture-status ${type}`.trim();
}

function updatePoseGuide(phase) {
    const meta = ENROLL_PHASE_META[phase] || ENROLL_PHASE_META.front;
    const guide = document.getElementById('enroll-pose-guide');
    const arrow = document.getElementById('enroll-pose-arrow');
    if (!guide || !arrow) return;
    guide.className = `pose-guide ${meta.guideClass}`;
    arrow.textContent = meta.arrow;
}

function startAutoEnrollCapture() {
    stopAutoEnrollCapture();
    if (!state.enrollState.stream) return;
    state.enrollState.autoTimer = setInterval(() => captureEnrollFrame('auto'), ENROLL_CAPTURE_INTERVAL_MS);
    setEnrollAutoStatus(`Auto capture is checking your face every ${ENROLL_CAPTURE_INTERVAL_MS / 1000}s...`);
}

function stopAutoEnrollCapture() {
    if (state.enrollState.autoTimer) {
        clearInterval(state.enrollState.autoTimer);
        state.enrollState.autoTimer = null;
    }
}

function getEnrollCaptures(angle) {
    const captures = state.enrollState.captures[angle];
    if (Array.isArray(captures)) return captures;
    return captures ? [captures] : [];
}

function getEnrollCaptureCount(angle) {
    return getEnrollCaptures(angle).length;
}

function updateEnrollUI() {
    const phase = ENROLL_PHASES[state.enrollState.currentPhase];
    if (!phase) {
        stopAutoEnrollCapture();
        document.getElementById('enroll-capture-btn').disabled = true;
        document.getElementById('enroll-submit-btn').disabled = false;
        updateThumbnailProgress();
        document.getElementById('enroll-instruction').textContent = 'All angles verified. Click Save Enrollment.';
        setTone('enroll-instruction');
        setEnrollAutoStatus('All angles captured. Review thumbnails, then save.', 'success');
        return;
    }

    const meta = ENROLL_PHASE_META[phase];
    const count = getEnrollCaptureCount(phase);
    updatePoseGuide(phase);
    document.getElementById('enroll-capture-btn').disabled = state.enrollState.isValidating;
    document.getElementById('enroll-submit-btn').disabled = true;
    document.getElementById('enroll-instruction').textContent =
        `${meta.instruction}. ${meta.hint}. Progress: ${count}/${ENROLL_REQUIRED_GOOD_FRAMES}`;
    setTone('enroll-instruction');
    updateThumbnailProgress();

    ENROLL_PHASES.forEach((p, idx) => {
        const card = document.getElementById(`thumb-card-${p}`);
        if (idx === state.enrollState.currentPhase) card.classList.add('active');
        else card.classList.remove('active');
    });
}

function updateThumbnailProgress() {
    ENROLL_PHASES.forEach(p => {
        const label = document.querySelector(`#thumb-card-${p} .thumb-label`);
        if (!label) return;
        const meta = ENROLL_PHASE_META[p];
        label.textContent = `${meta.label} ${getEnrollCaptureCount(p)}/${ENROLL_REQUIRED_GOOD_FRAMES}`;
    });
}

export function retakeAngle(angle) {
    if (document.getElementById('enroll-submit-btn').disabled === false) {
        document.getElementById('enroll-submit-btn').disabled = true;
    }
    const idx = ENROLL_PHASES.indexOf(angle);
    if (idx === -1) return;
    state.enrollState.currentPhase = idx;
    state.enrollState.captures[angle] = [];
    const card = document.getElementById(`thumb-card-${angle}`);
    card.className = 'thumb-card active';
    hideElement(card.querySelector('.thumb-img'));
    hideElement(card.querySelector('.thumb-check'));
    showElement(card.querySelector('.thumb-placeholder'), 'flex');
    card.querySelector('.thumb-placeholder').classList.remove('error');
    updateEnrollUI();
    startAutoEnrollCapture();
}

function flashCamera() {
    const fl = document.getElementById('enroll-flash');
    fl.classList.add('flash-on');
    setTimeout(() => fl.classList.remove('flash-on'), 100);
}

function captureEnrollBlob() {
    const video = document.getElementById('enroll-camera');
    if (!video || !video.videoWidth) return null;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.95));
}

function setEnrollThumbnail(phase, blob, thumbState = 'validating', reason = '') {
    const imgUrl = URL.createObjectURL(blob);
    const card = document.getElementById(`thumb-card-${phase}`);
    const imgEl = card.querySelector('.thumb-img');
    imgEl.src = imgUrl;
    showElement(imgEl, 'block');
    card.classList.add('has-img');
    hideElement(card.querySelector('.thumb-placeholder'));
    card.classList.toggle('success', thumbState === 'success');
    card.classList.toggle('error', thumbState === 'error');
    if (thumbState === 'success') showElement(card.querySelector('.thumb-check'), 'flex');
    else hideElement(card.querySelector('.thumb-check'));
    card.querySelector('.thumb-overlay p').textContent = thumbState === 'error' ? `FAILED: ${reason}` : 'RETAKE';
}

export async function captureEnrollFrame(source = 'manual') {
    const phase = ENROLL_PHASES[state.enrollState.currentPhase];
    if (!state.enrollState.stream || !phase || state.enrollState.isValidating) return;

    const blob = await captureEnrollBlob();
    if (!blob) return;

    state.enrollState.isValidating = true;
    document.getElementById('enroll-capture-btn').disabled = true;
    setEnrollAutoStatus(`Checking ${ENROLL_PHASE_META[phase].label} angle... hold still.`);
    if (source === 'manual') {
        flashCamera();
        setEnrollThumbnail(phase, blob, 'validating');
    }

    const fd = new FormData();
    fd.append('angle', phase);
    fd.append('image', blob, 'capture.jpg');

    try {
        const data = await api('/api/enroll/v2/validate', { method: 'POST', body: fd });
        if (data.success) {
            flashCamera();
            const captures = getEnrollCaptures(phase);
            if (captures.length < ENROLL_REQUIRED_GOOD_FRAMES) {
                captures.push(blob);
                state.enrollState.captures[phase] = captures;
            }
            setEnrollThumbnail(phase, blob, 'success');
            const count = getEnrollCaptureCount(phase);
            const label = ENROLL_PHASE_META[phase].label;
            setEnrollAutoStatus(`${label}: ${count}/${ENROLL_REQUIRED_GOOD_FRAMES} good frames captured.`, 'success');

            if (count >= ENROLL_REQUIRED_GOOD_FRAMES) {
                state.enrollState.currentPhase++;
                while (
                    state.enrollState.currentPhase < ENROLL_PHASES.length &&
                    getEnrollCaptureCount(ENROLL_PHASES[state.enrollState.currentPhase]) >= ENROLL_REQUIRED_GOOD_FRAMES
                ) {
                    state.enrollState.currentPhase++;
                }
                if (state.enrollState.currentPhase < ENROLL_PHASES.length) {
                    setEnrollAutoStatus(`${label} complete. Moving to next angle...`, 'success');
                }
            }
        } else {
            const reason = formatEnrollFailure(data.reason, phase);
            if (source === 'manual') {
                setEnrollThumbnail(phase, blob, 'error', reason);
                showToast(reason, 'error');
            }
            setEnrollAutoStatus(reason, 'error');
        }
    } catch (e) {
        const message = `Validation request failed: ${e.message}`;
        if (source === 'manual') showToast(message, 'error');
        setEnrollAutoStatus(`${message}. Please check the FastAPI server is running.`, 'error');
        stopAutoEnrollCapture();
    } finally {
        state.enrollState.isValidating = false;
        updateEnrollUI();
    }
}

function formatEnrollFailure(reason = 'Angle not ready', phase = '') {
    const lower = reason.toLowerCase();
    if (lower.includes('no face')) return 'Move closer and keep your face inside the frame.';
    if (lower.includes('too blurry')) return 'Image is blurry. Hold still and improve lighting.';
    if (lower.includes('center')) return 'Keep your face centered and look straight.';
    if (lower.includes('left enough')) return 'Turn a little further toward the LEFT arrow.';
    if (lower.includes('right enough')) return 'Turn a little further toward the RIGHT arrow.';
    if (lower.includes('embedding')) return 'Face detected but encoding failed. Hold still and try again.';
    return `${ENROLL_PHASE_META[phase]?.label || 'Angle'} not ready. ${reason}`;
}

export function proceedToManualEnroll() {
    const s = validateNewStudentForm();
    if (!s) return;
    setEnrollState(createEnrollState({ ...s }));
    manualValidated = { front: false, left: false, right: false };
    closeModal('student-create-modal');
    document.getElementById('manual-enroll-title').textContent = state.enrollState.name;
    document.getElementById('manual-enroll-subtitle').textContent = `ID: ${state.enrollState.id}`;
    document.getElementById('manual-front').value = '';
    document.getElementById('manual-left').value = '';
    document.getElementById('manual-right').value = '';
    ['front', 'left', 'right'].forEach(a => {
        const el = document.getElementById(`manual-${a}-err`);
        hideElement(el);
        el.textContent = '';
        setTone(el);
    });
    document.getElementById('manual-enroll-submit').disabled = true;
    showModal('manual-enroll-modal');
}

function checkManualReady() {
    document.getElementById('manual-enroll-submit').disabled =
        !(manualValidated.front && manualValidated.left && manualValidated.right);
}

export async function validateManualFile(angle) {
    const input = document.getElementById(`manual-${angle}`);
    const errEl = document.getElementById(`manual-${angle}-err`);
    const file = input.files[0];
    if (!file) {
        manualValidated[angle] = false;
        checkManualReady();
        return;
    }

    errEl.textContent = 'Validating...';
    showElement(errEl, 'block');
    setTone(errEl, 'muted');
    manualValidated[angle] = false;
    checkManualReady();

    const fd = new FormData();
    fd.append('angle', angle);
    fd.append('image', file);
    try {
        const data = await api('/api/enroll/v2/validate', { method: 'POST', body: fd });
        if (data.success) {
            manualValidated[angle] = true;
            errEl.textContent = `\u2713 Valid (quality: ${Math.round(data.quality || 0)})`;
            setTone(errEl, 'primary');
        } else {
            manualValidated[angle] = false;
            errEl.textContent = `\u2717 ${formatEnrollFailure(data.reason, angle)}`;
            setTone(errEl, 'error');
        }
    } catch (e) {
        manualValidated[angle] = false;
        errEl.textContent = 'Validation request failed';
        setTone(errEl, 'error');
    }
    showElement(errEl, 'block');
    checkManualReady();
}

export async function submitEnroll(isManual = false) {
    const fd = new FormData();
    fd.append('student_id', state.enrollState.id);
    fd.append('name', state.enrollState.name);
    fd.append('class_name', state.enrollState.class || '');

    if (isManual) {
        fd.append('image_front', document.getElementById('manual-front').files[0]);
        fd.append('image_left', document.getElementById('manual-left').files[0]);
        fd.append('image_right', document.getElementById('manual-right').files[0]);
    } else {
        const frontCaptures = getEnrollCaptures('front');
        const leftCaptures = getEnrollCaptures('left');
        const rightCaptures = getEnrollCaptures('right');
        if (
            frontCaptures.length < ENROLL_REQUIRED_GOOD_FRAMES ||
            leftCaptures.length < ENROLL_REQUIRED_GOOD_FRAMES ||
            rightCaptures.length < ENROLL_REQUIRED_GOOD_FRAMES
        ) {
            return showToast(`Need ${ENROLL_REQUIRED_GOOD_FRAMES} good frames per angle`, 'error');
        }
        frontCaptures.forEach((blob, i) => fd.append('image_front', blob, `front_${i + 1}.jpg`));
        leftCaptures.forEach((blob, i) => fd.append('image_left', blob, `left_${i + 1}.jpg`));
        rightCaptures.forEach((blob, i) => fd.append('image_right', blob, `right_${i + 1}.jpg`));
    }

    showLoading('Generating Encodings...');
    try {
        const data = await api('/api/enroll/v2', { method: 'POST', body: fd });
        hideLoading();
        if (data.success) {
            showToast('Enrollment Saved Successfully!', 'success');
            if (isManual) closeModal('manual-enroll-modal');
            else closeEnrollCameraModal();
            loadStudents(state.currentView);
        } else {
            showToast(data.message || 'Enrollment failed', 'error');
            if (data.phase_results) {
                const errs = data.phase_results.filter(r => !r.success).map(r => `${r.name}: ${r.reason}`).join(' | ');
                if (errs) showToast(errs, 'error');
            }
        }
    } catch (e) {
        hideLoading();
        showToast('Network error submitting enrollment', 'error');
    }
}
