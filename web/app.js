/**
 * Face Attendance - Frontend JS
 */
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function toast(msg, type = 'info') {
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.textContent = msg;
    $('#toast-container').appendChild(t);
    setTimeout(() => t.remove(), 4000);
}
function showLoading(text) { $('#loading-text').textContent = text || 'Processing...'; $('#loading').style.display = 'flex'; }
function hideLoading() { $('#loading').style.display = 'none'; }
async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

// Clock
function updateClock() {
    const now = new Date();
    $('#clock').textContent = now.toLocaleString('en-US');
}
setInterval(updateClock, 1000); updateClock();

// Sidebar Nav
$$('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        $$('.nav-item').forEach(t => t.classList.remove('active'));
        $$('.tab-content').forEach(c => c.classList.remove('active'));
        item.classList.add('active');
        $(`#tab-${item.dataset.tab}`).classList.add('active');
        if (item.dataset.tab === 'students') loadStudents();
        if (item.dataset.tab === 'history') loadSessions();
    });
});

// Status
async function checkStatus() {
    try {
        const data = await api('/api/system/status');
        const b = $('#status-badge');
        const dot = $('#header-dot');
        b.className = 'status-badge ok';
        b.textContent = `${data.students} students`;
        if (dot) { dot.classList.remove('bg-outline'); dot.classList.add('bg-secondary'); dot.style.boxShadow = '0 0 8px rgba(123,219,128,0.5)'; }
    } catch {
        $('#status-badge').className = 'status-badge';
        $('#status-badge').textContent = 'Disconnected';
        const dot = $('#header-dot');
        if (dot) { dot.classList.add('bg-outline'); dot.classList.remove('bg-secondary'); dot.style.boxShadow = 'none'; }
    }
}
checkStatus(); setInterval(checkStatus, 15000);

// ── SESSION ────────────────────────────────────────────────
let activeSession = null;

async function refreshSession() {
    try {
        const data = await api('/api/session/active');
        if (data.active) {
            activeSession = data.session;
            $('#session-info').innerHTML = `<span class="dot on"></span><span><strong>${activeSession.name}</strong> — ${activeSession.present_count}/${activeSession.total_students} present</span>`;
            $('#btn-start').style.display = 'none'; $('#session-name').style.display = 'none';
            $('#btn-stop').style.display = 'inline-block'; $('#btn-scan').disabled = false;
        } else {
            activeSession = null;
            $('#session-info').innerHTML = `<span class="dot off"></span><span>No active session</span>`;
            $('#btn-start').style.display = 'inline-block'; $('#session-name').style.display = 'inline-block';
            $('#btn-stop').style.display = 'none'; $('#btn-scan').disabled = true;
        }
    } catch { }
}

async function startSession() {
    const name = $('#session-name').value.trim() || `Session ${new Date().toLocaleDateString('en-US')}`;
    try {
        const r = await api('/api/session/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
        toast(r.message, r.success ? 'success' : 'error');
        refreshSession(); refreshAttendance();
    } catch (e) { toast(e.message, 'error'); }
}

async function endSession() {
    if (!confirm('End this session?')) return;
    try {
        const r = await api('/api/session/end', { method: 'POST' });
        toast(r.message, r.success ? 'success' : 'error');
        refreshSession(); refreshAttendance();
    } catch (e) { toast(e.message, 'error'); }
}

// ── CAMERA SOURCE ──────────────────────────────────────────
let cameraSource = 'webcam'; // 'webcam' | 'phone'
let phoneStatusInterval = null;
let demoVideoActive = false;

function setCameraSource(src) {
    // If demo video is active, stop it first
    if (demoVideoActive) toggleDemoVideo();
    cameraSource = src;
    const feed = $('#camera-feed');
    const webcamBtn = $('#src-webcam');
    const phoneBtn = $('#src-phone');
    const phoneStatus = $('#phone-status');

    webcamBtn.classList.toggle('active', src === 'webcam');
    phoneBtn.classList.toggle('active', src === 'phone');

    if (src === 'phone') {
        feed.src = '/api/live/phone-stream';
        phoneStatus.style.display = 'inline-flex';
        $('#camera-hint').textContent = 'Phone Camera — Open /phone on your mobile';
        startPhoneStatusPolling();
    } else {
        feed.src = '/api/live/stream';
        phoneStatus.style.display = 'none';
        $('#camera-hint').textContent = 'Webcam — Position face in frame';
        stopPhoneStatusPolling();
    }
}

// ── DEMO VIDEO MODE ─────────────────────────────────────────

let demoScanInterval = null;

function toggleDemoVideo() {
    const feed = $('#camera-feed');
    const vid = $('#demo-video');
    const btn = $('#btn-demo');
    const label = $('#btn-demo-label');
    const hint = $('#camera-hint');

    demoVideoActive = !demoVideoActive;

    if (demoVideoActive) {
        // Show video, hide camera img stream
        feed.style.display = 'none';
        vid.style.display = 'block';
        vid.currentTime = 0;
        vid.play();

        // Style button active (amber)
        btn.classList.remove('border-[#8b5cf6]/30', 'bg-[#8b5cf6]/10', 'text-[#c4b5fd]');
        btn.classList.add('border-yellow-500/50', 'bg-yellow-500/20', 'text-yellow-300');
        label.textContent = 'Stop Demo';
        hint.textContent = 'Demo Video — Auto scanning...';

        // AUTO-SCAN: capture a frame every 1.2s and POST to /api/scan
        let foundFaceInDemo = false;
        demoScanInterval = setInterval(async () => {
            if (!activeSession || !demoVideoActive) return;
            if (foundFaceInDemo) return; // stop after first recognition
            try {
                const c = document.createElement('canvas');
                c.width = vid.videoWidth || 1280;
                c.height = vid.videoHeight || 720;
                c.getContext('2d').drawImage(vid, 0, 0, c.width, c.height);
                const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
                const fd = new FormData();
                fd.append('image', blob, 'frame.jpg');
                const result = await api('/api/scan', { method: 'POST', body: fd });
                if (result.results && result.results.length > 0) {
                    const r = result.results[0];
                    if (r.status === 'present' || r.status === 'already') {
                        foundFaceInDemo = true;
                        showScanResult(r);
                        const msg = r.status === 'present' ? 'Marked present!' : 'Already marked';
                        toast(`✓ ${r.name} — ${msg}`, 'success');
                        hint.textContent = `Recognized: ${r.name}`;
                        refreshSession(); refreshAttendance();
                    }
                }
            } catch { /* ignore individual scan errors */ }
        }, 1200);

    } else {
        // Stop auto-scan
        if (demoScanInterval) { clearInterval(demoScanInterval); demoScanInterval = null; }

        // Restore camera stream
        vid.pause();
        vid.style.display = 'none';
        feed.style.display = 'block';
        feed.src = (cameraSource === 'phone') ? '/api/live/phone-stream' : '/api/live/stream';

        // Restore button style
        btn.classList.remove('border-yellow-500/50', 'bg-yellow-500/20', 'text-yellow-300');
        btn.classList.add('border-[#8b5cf6]/30', 'bg-[#8b5cf6]/10', 'text-[#c4b5fd]');
        label.textContent = 'Demo';
        hint.textContent = (cameraSource === 'phone')
            ? 'Phone Camera — Open /phone on your mobile'
            : 'Webcam — Position face in frame';
    }
}

function startPhoneStatusPolling() {
    stopPhoneStatusPolling();
    checkPhoneStatus();
    phoneStatusInterval = setInterval(checkPhoneStatus, 3000);
}

function stopPhoneStatusPolling() {
    if (phoneStatusInterval) { clearInterval(phoneStatusInterval); phoneStatusInterval = null; }
}

async function checkPhoneStatus() {
    try {
        const data = await api('/api/phone/status');
        const dot = $('#phone-dot');
        const text = $('#phone-status-text');
        if (data.connected) {
            dot.className = 'dot on';
            text.textContent = 'Connected';
        } else {
            dot.className = 'dot off';
            text.textContent = 'Not connected';
        }
    } catch { }
}

// ── SCAN ───────────────────────────────────────────────────
async function doScan() {
    if (!activeSession) { toast('No active session', 'error'); return; }
    showLoading('Scanning...');
    try {
        let blob;
        if (demoVideoActive) {
            // Capture current frame from the demo <video> element
            const vid = $('#demo-video');
            const c = document.createElement('canvas');
            c.width = vid.videoWidth || 1280;
            c.height = vid.videoHeight || 720;
            c.getContext('2d').drawImage(vid, 0, 0, c.width, c.height);
            blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
        } else if (cameraSource === 'phone') {
            // Fetch latest frame from phone camera buffer
            const resp = await fetch('/api/phone/latest');
            if (!resp.ok || resp.status === 204) {
                hideLoading();
                toast('No frame from phone camera', 'error');
                return;
            }
            blob = await resp.blob();
        } else {
            // Capture from MJPEG img element
            const img = $('#camera-feed');
            const c = document.createElement('canvas');
            c.width = img.naturalWidth || 640; c.height = img.naturalHeight || 480;
            c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
            blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.9));
        }

        const fd = new FormData(); fd.append('image', blob, 'frame.jpg');
        const result = await api('/api/scan', { method: 'POST', body: fd });
        hideLoading();
        if (result.results && result.results.length > 0) {
            const r = result.results[0];
            showScanResult(r);
            toast(`${r.name} — ${r.message}`, r.status === 'present' ? 'success' : 'info');
        } else {
            showScanResult({ name: '---', message: 'No face detected', status: 'none' });
            toast('No face detected', 'error');
        }
        refreshSession(); refreshAttendance();
    } catch (e) { hideLoading(); toast(e.message, 'error'); }
}

function showScanResult(r) {
    $('#result-card').style.display = 'block';
    $('#result-name').textContent = r.name || '---';
    $('#result-msg').textContent = r.message || '';
}

async function refreshAttendance() {
    try {
        const data = await api('/api/session/attendance');
        if (!data.active) {
            $('#present-count').textContent = '0 / 0';
            $('#present-list').innerHTML = '<p class="empty text-center py-8">No active session</p>';
            updateProgress(0, 0);
            return;
        }
        const result = data.result;
        $('#present-count').textContent = `${result.present_count} / ${result.total}`;
        updateProgress(result.present_count, result.total);
        if (!data.attendance.length) {
            $('#present-list').innerHTML = '<p class="empty text-center py-8">No attendance yet</p>';
            return;
        }
        $('#present-list').innerHTML = data.attendance.map(a => {
            const t = new Date(a.scanned_at).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const initials = a.student_name.split(' ').slice(-2).map(w => w[0]).join('').toUpperCase();
            const sid = a.student_id || '';
            return `<div class="present-item">
                <div class="avatar">${initials}</div>
                <div class="present-item-info">
                    <div class="name">${a.student_name}</div>
                    <div class="id">ID: ${sid}</div>
                </div>
                <div class="time-col">
                    <div class="time">${t}</div>
                    <span class="status-pill"><span class="pip"></span>Present</span>
                </div>
            </div>`;
        }).join('');
    } catch { }
}

function updateProgress(present, total) {
    const pct = total > 0 ? Math.round((present / total) * 100) : 0;
    const pctEl = $('#progress-pct');
    const barEl = $('#progress-bar');
    if (pctEl) pctEl.textContent = `${pct}%`;
    if (barEl) barEl.style.width = `${pct}%`;
}

// ── STUDENTS ───────────────────────────────────────────────
function showEnrollForm() { $('#enroll-form').style.display = 'block'; }
function hideEnrollForm() { $('#enroll-form').style.display = 'none'; $('#enroll-result').textContent = ''; }

async function submitEnroll() {
    const id = $('#enroll-id').value.trim(), name = $('#enroll-name').value.trim();
    const cls = $('#enroll-class').value.trim(), photo = $('#enroll-photo');
    if (!id || !name) { toast('Enter student ID and name', 'error'); return; }
    if (!photo.files.length) { toast('Select a photo', 'error'); return; }
    showLoading('Enrolling...');
    const fd = new FormData();
    fd.append('student_id', id); fd.append('name', name); fd.append('class_name', cls);
    fd.append('image', photo.files[0]);
    try {
        const r = await api('/api/enroll', { method: 'POST', body: fd }); hideLoading();
        const el = $('#enroll-result');
        el.className = r.success ? 'success' : 'error';
        el.textContent = r.message;
        if (r.success) { toast('Enrolled successfully', 'success'); loadStudents(); checkStatus(); }
    } catch (e) { hideLoading(); toast(e.message, 'error'); }
}

async function loadStudents() {
    try {
        const data = await api('/api/students');
        const tb = $('#student-tbody'), ss = data.students || [];
        if (!ss.length) { tb.innerHTML = '<tr><td colspan="5" class="empty">No students</td></tr>'; return; }
        tb.innerHTML = ss.map(s => {
            const d = s.enrolled_at ? new Date(s.enrolled_at).toLocaleDateString('en-US') : '--';
            return `<tr><td><strong>${s.id}</strong></td><td>${s.name}</td><td>${s.class_name || '--'}</td><td>${d}</td><td><button class="btn btn-danger" onclick="deleteStudent('${s.id}','${s.name}')">Delete</button></td></tr>`;
        }).join('');
    } catch (e) { $('#student-tbody').innerHTML = `<tr><td colspan="5" class="empty">${e.message}</td></tr>`; }
}

async function deleteStudent(id, name) {
    if (!confirm(`Delete ${name}?`)) return;
    try { await api(`/api/students/${id}`, { method: 'DELETE' }); toast(`Deleted ${name}`, 'success'); loadStudents(); checkStatus(); }
    catch (e) { toast(e.message, 'error'); }
}

// ── HISTORY ────────────────────────────────────────────────
async function loadSessions() {
    try {
        const data = await api('/api/sessions');
        const tb = $('#history-tbody'), ss = data.sessions || [];
        if (!ss.length) { tb.innerHTML = '<tr><td colspan="7" class="empty">No sessions</td></tr>'; return; }
        tb.innerHTML = ss.map(s => {
            const t = new Date(s.created_at).toLocaleString('en-US');
            const badge = s.status === 'active' ? '<span class="badge badge-active">Active</span>' : '<span class="badge badge-ended">Ended</span>';
            return `<tr><td>${s.id}</td><td><strong>${s.name}</strong></td><td>${t}</td><td>${s.present_count}</td><td>${s.absent_count}</td><td>${badge}</td><td><button class="btn btn-detail" onclick="showSessionDetail(${s.id})">View</button></td></tr>`;
        }).join('');
    } catch (e) { $('#history-tbody').innerHTML = `<tr><td colspan="7" class="empty">${e.message}</td></tr>`; }
}

async function showSessionDetail(sid) {
    try {
        const data = await api(`/api/session/${sid}/result`);
        $('#detail-panel').style.display = 'block';
        $('#detail-title').textContent = data.session.name;
        const r = data.result;
        $('#detail-present-count').textContent = r.present_count;
        $('#detail-absent-count').textContent = r.absent_count;
        $('#detail-present-list').innerHTML = r.present.map(s => `<li>${s.name} (${s.id})</li>`).join('') || '<li class="empty">None</li>';
        $('#detail-absent-list').innerHTML = r.absent.map(s => `<li>${s.name} (${s.id})</li>`).join('') || '<li class="empty">None</li>';
    } catch (e) { toast(e.message, 'error'); }
}
function hideDetail() { $('#detail-panel').style.display = 'none'; }

// Init
refreshSession(); refreshAttendance(); setInterval(refreshAttendance, 10000);

// ── DEMO SLIDESHOW ─────────────────────────────────────────
let demoSlide = 0;
const DEMO_TOTAL = 5;
let demoAutoTimer = null;

function openDemo() {
    demoSlide = 0;
    $('#demo-overlay').style.display = 'flex';
    buildDemoDots();
    updateDemoSlide();
    // Auto-advance every 8s
    startDemoAuto();
    document.addEventListener('keydown', demoKeyHandler);
}

function closeDemo() {
    $('#demo-overlay').style.display = 'none';
    stopDemoAuto();
    document.removeEventListener('keydown', demoKeyHandler);
}

function demoNav(dir) {
    demoSlide = Math.max(0, Math.min(DEMO_TOTAL - 1, demoSlide + dir));
    updateDemoSlide();
    stopDemoAuto();
}

function demoGoTo(idx) {
    demoSlide = idx;
    updateDemoSlide();
    stopDemoAuto();
}

function updateDemoSlide() {
    $$('.demo-slide').forEach((s, i) => {
        s.classList.toggle('active', i === demoSlide);
    });
    $$('#demo-dots .demo-dot').forEach((d, i) => {
        d.classList.toggle('active', i === demoSlide);
    });
    $('#demo-counter').textContent = `${demoSlide + 1} / ${DEMO_TOTAL}`;
    $('#demo-prev').disabled = demoSlide === 0;
    if (demoSlide === DEMO_TOTAL - 1) {
        $('#demo-next').textContent = 'Close';
        $('#demo-next').onclick = closeDemo;
    } else {
        $('#demo-next').textContent = 'Next';
        $('#demo-next').onclick = () => demoNav(1);
    }
}

function buildDemoDots() {
    const container = $('#demo-dots');
    container.innerHTML = '';
    for (let i = 0; i < DEMO_TOTAL; i++) {
        const dot = document.createElement('span');
        dot.className = 'demo-dot' + (i === 0 ? ' active' : '');
        dot.onclick = () => demoGoTo(i);
        container.appendChild(dot);
    }
}

function startDemoAuto() {
    stopDemoAuto();
    demoAutoTimer = setInterval(() => {
        if (demoSlide < DEMO_TOTAL - 1) {
            demoSlide++;
            updateDemoSlide();
        } else {
            stopDemoAuto();
        }
    }, 8000);
}

function stopDemoAuto() {
    if (demoAutoTimer) { clearInterval(demoAutoTimer); demoAutoTimer = null; }
}

function demoKeyHandler(e) {
    if (e.key === 'ArrowRight') demoNav(1);
    else if (e.key === 'ArrowLeft') demoNav(-1);
    else if (e.key === 'Escape') closeDemo();
}

// Close on overlay click
document.addEventListener('click', (e) => {
    if (e.target.id === 'demo-overlay') closeDemo();
});
