/* =========================================================================
   LUMIN OS — Face Attendance SPA
   Phase 5, 6, 7 Refactored
   ========================================================================= */

// --- Global State ---
let currentTheme = localStorage.getItem('theme') || 'light';
let isAutoScanning = false;
let autoScanInterval;
let scanIntervalMs = 2500;
let lastFrameTime = 0;
let scanCount = 0;
let presentSet = new Set();
let sessionActive = false;
let sessionStats = { total:0, present:0, absent:0 };
let currentView = 'active'; // Students view state

// Enrollment State
const ENROLL_PHASES = ['front', 'left', 'right'];
let enrollState = {
    id: null, name: null, className: null,
    stream: null, currentPhase: 0, 
    captures: { front: null, left: null, right: null }
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.setAttribute('data-theme', currentTheme);
    setupTabs();
    updateClock();
    setInterval(updateClock, 1000);
    checkSessionStatus();
    loadStudents('active');
    loadHistory();
});

// --- UI / Theme / Modals ---
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
}

function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { hour12: false });
}

function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    const bg = type === 'error' ? 'var(--error)' : type === 'success' ? 'var(--primary)' : 'var(--tertiary)';
    const color = type === 'error' ? 'var(--on-error)' : 'var(--on-primary)';
    toast.style = `background:${bg};color:${color};padding:12px 20px;border-radius:var(--radius-md);font-weight:600;font-size:0.875rem;box-shadow:var(--shadow-elevated);animation:toastIn 0.3s cubic-bezier(0.16,1,0.3,1)`;
    toast.innerHTML = `<style>@keyframes toastIn{from{opacity:0;transform:translateX(100%)}to{opacity:1;transform:translateX(0)}}</style>${msg}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function showLoading(text='Processing...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showModal(id) {
    const m = document.getElementById(id);
    m.style.display = 'flex';
}
function closeModal(id) {
    document.getElementById(id).style.display = 'none';
}

function confirmDialog(title, msg, type = 'warning') {
    return new Promise(resolve => {
        const d = document.getElementById('confirm-dialog');
        document.getElementById('confirm-title').textContent = title;
        document.getElementById('confirm-msg').textContent = msg;
        const iconContainer = document.getElementById('confirm-icon');
        const iconSym = document.getElementById('confirm-icon-sym');
        
        iconContainer.className = `confirm-icon ${type}`;
        iconSym.textContent = type === 'warning' ? 'warning' : type === 'error' ? 'error' : 'info';

        d.style.display = 'flex';
        
        const btnOk = document.getElementById('confirm-ok');
        const btnCancel = document.getElementById('confirm-cancel');
        
        const cleanup = () => {
            d.style.display = 'none';
            btnOk.removeEventListener('click', onOk);
            btnCancel.removeEventListener('click', onCancel);
        };
        
        const onOk = () => { cleanup(); resolve(true); };
        const onCancel = () => { cleanup(); resolve(false); };
        
        btnOk.addEventListener('click', onOk);
        btnCancel.addEventListener('click', onCancel);
    });
}

function setupTabs() {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });
}

// --- Session Management (Phase 7) ---

async function checkSessionStatus() {
    try {
        const res = await fetch('/api/session/active');
        const data = await res.json();
        const headerStatus = document.getElementById('status-badge');
        const headerDot = document.getElementById('header-dot');
        const dot = document.querySelector('#session-info .dot');
        const txt = document.querySelector('#session-info span:not(.dot)');
        const input = document.getElementById('session-name');
        
        if (data.active) {
            sessionActive = true;
            const sess = data.session;
            headerStatus.textContent = "Session Active";
            headerDot.className = "status-badge-dot on";
            dot.className = "dot on";
            txt.textContent = sess.name || "Active Session";
            input.style.display = 'none';
            document.getElementById('btn-start').style.display = 'none';
            document.getElementById('btn-stop').style.display = 'inline-flex';
            checkAutoScanStatus();
        } else {
            sessionActive = false;
            headerStatus.textContent = "Idle";
            headerDot.className = "status-badge-dot";
            dot.className = "dot off";
            txt.textContent = "No active session";
            input.style.display = 'block';
            document.getElementById('btn-start').style.display = 'inline-flex';
            document.getElementById('btn-stop').style.display = 'none';
            stopAutoScan();
        }
    } catch (e) { console.error("Status check failed", e); }
}

async function uiStartSession() {
    const input = document.getElementById('session-name');
    const name = input.value.trim() || `Class_${new Date().toLocaleDateString().replace(/\//g,'-')}`;
    
    const d = await confirmDialog("Start New Session", `Begin attendance session: ${name}?`, "info");
    if (!d) return;

    showLoading('Starting session...');
    try {
        const res = await fetch('/api/session/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name })
        });
        const data = await res.json();
        hideLoading();
        if (data.success) {
            showToast('Session Started', 'success');
            presentSet.clear();
            scanCount = 0;
            updateScanLogs();
            document.getElementById('end-session-summary').style.display = 'none';
            checkSessionStatus();
            setCameraSource('webcam'); // Start auto scan through default
        } else {
            showToast(data.message, 'error');
        }
    } catch (e) { hideLoading(); showToast('Error starting session', 'error'); }
}

async function uiEndSession() {
    const d = await confirmDialog("End Session", "Are you sure you want to end the current attendance session?", "warning");
    if (!d) return;

    showLoading('Ending session...');
    try {
        const res = await fetch('/api/session/end', { method: 'POST' });
        const data = await res.json();
        hideLoading();
        if (data.success) {
            showToast('Session Ended', 'success');
            document.getElementById('summary-total').textContent = data.total;
            document.getElementById('summary-present').textContent = data.present;
            document.getElementById('summary-absent').textContent = data.absent;
            document.getElementById('end-session-summary').style.display = 'block';
            
            checkSessionStatus();
            loadHistory();
        } else { showToast(data.message, 'error'); }
    } catch (e) { hideLoading(); showToast('Error ending session', 'error'); }
}

function hideSummary() { document.getElementById('end-session-summary').style.display = 'none'; }


// --- Students CRUD & Filter (Phase 6) ---

function setStudentView(view) {
    currentView = view;
    // update buttons state natively in HTML onclicks, this just triggers load
    loadStudents(view);
}

async function loadStudents(view) {
    try {
        const res = await fetch(`/api/students?view=${view}`);
        const data = await res.json();
        const tbody = document.getElementById('student-tbody');
        tbody.innerHTML = '';
        if (!data.students || data.students.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No students found in this view.</td></tr>`;
            return;
        }

        data.students.forEach(s => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--outline-variant)';
            
            const isActive = s.is_active !== 0;
            const statusClass = isActive ? 'active' : 'archived';
            const statusText = isActive ? 'Active' : 'Archived';
            
            tr.innerHTML = `
                <td style="padding:12px 16px;font-weight:600;color:var(--on-surface)">${s.id}</td>
                <td style="padding:12px 16px;display:flex;align-items:center;gap:10px">
                    <img src="/api/students/${s.id}/photo" onerror="this.style.display='none'" style="width:32px;height:32px;border-radius:50%;background:var(--surface-container-high);object-fit:cover">
                    <span style="font-weight:600;color:var(--on-surface)">${s.name}</span>
                </td>
                <td style="padding:12px 16px;color:var(--on-surface-variant)">${s.class_name || '--'}</td>
                <td style="padding:12px 16px;text-align:center"><span class="status-chip ${statusClass}">${statusText}</span></td>
                <td style="padding:12px 16px">
                    <div class="action-cell">
                        <button class="btn-icon view" title="View Profile" onclick="showStudentDetail('${s.id}')"><span class="material-symbols-outlined" style="font-size:18px">visibility</span></button>
                        ${isActive ? `
                            <button class="btn-icon edit" title="Edit Metadata" onclick="openEditModal('${s.id}')"><span class="material-symbols-outlined" style="font-size:18px">edit</span></button>
                            <button class="btn-icon enroll" title="Re-enroll Face" onclick="startReEnroll('${s.id}', '${s.name}', '${s.class_name}')"><span class="material-symbols-outlined" style="font-size:18px">face</span></button>
                            <button class="btn-icon archive" title="Archive Student" onclick="archiveStudent('${s.id}')"><span class="material-symbols-outlined" style="font-size:18px">archive</span></button>
                        ` : `
                            <button class="btn-icon restore" title="Restore Student" onclick="restoreStudent('${s.id}')"><span class="material-symbols-outlined" style="font-size:18px">restore_from_trash</span></button>
                        `}
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });
        
        // Setup view tabs styling dynamically
        document.querySelectorAll('.view-tab').forEach(b => {
            b.classList.remove('active');
            if(b.textContent.toLowerCase() === view.toLowerCase()) b.classList.add('active');
        });
        currentView = view;
    } catch (e) { console.error("Load students error", e); }
}

async function archiveStudent(id) {
    const d = await confirmDialog("Archive Student", `Are you sure you want to archive student ${id}? Their attendance will not be recorded while archived.`, "error");
    if (!d) return;

    showLoading('Archiving...');
    try {
        const res = await fetch(`/api/students/${id}`, { method: 'DELETE' });
        const data = await res.json();
        hideLoading();
        if (data.success) { showToast('Student archived', 'success'); loadStudents(currentView); }
        else showToast('Error archiving', 'error');
    } catch(e) { hideLoading(); showToast('Network error', 'error'); }
}

async function restoreStudent(id) {
    const d = await confirmDialog("Restore Student", `Re-activate student ${id}?`, "info");
    if (!d) return;

    showLoading('Restoring...');
    try {
        const res = await fetch(`/api/students/${id}/restore`, { method: 'POST' });
        const data = await res.json();
        hideLoading();
        if (data.success) { showToast('Student restored', 'success'); loadStudents(currentView); }
        else showToast(data.message || 'Error', 'error');
    } catch(e) { hideLoading(); showToast('Network error', 'error'); }
}

async function showStudentDetail(id) {
    showLoading();
    try {
        const res = await fetch(`/api/students/${id}`);
        const data = await res.json();
        hideLoading();
        
        const s = data.student;
        document.getElementById('detail-photo').src = `/api/students/${id}/photo?t=${Date.now()}`;
        document.getElementById('detail-name').textContent = s.name;
        document.getElementById('detail-id').textContent = `ID: ${s.id}`;
        document.getElementById('detail-class').textContent = `Class: ${s.class_name || '--'}`;
        document.getElementById('detail-embeddings').textContent = `${data.embedding_count || 0} reference angles recorded.`;
        
        const status = document.getElementById('detail-status');
        status.textContent = s.is_active === 0 ? "ARCHIVED" : "ACTIVE";
        status.className = s.is_active === 0 ? "status-chip archived" : "status-chip active";

        const hist = document.getElementById('detail-history-list');
        hist.innerHTML = '';
        if (!data.history || data.history.length === 0) {
            hist.innerHTML = '<p class="empty-state">No attendance history</p>';
        } else {
            data.history.forEach(h => {
                const el = document.createElement('div');
                el.style = "padding:8px 12px;border-bottom:1px solid var(--outline-variant);display:flex;justify-content:space-between;align-items:center";
                el.innerHTML = `
                    <div><p style="font-weight:600;color:var(--on-surface)">${h.session_name || 'Session ' + h.session_id}</p><p style="font-size:0.75rem;color:var(--on-surface-variant)">${new Date(h.timestamp).toLocaleString()}</p></div>
                    <span class="status-chip ${h.status==='Present'?'active':'archived'}">${h.status}</span>
                `;
                hist.appendChild(el);
            });
        }
        showModal('student-detail-modal');
    } catch(e) { hideLoading(); showToast("Error loading details", "error"); }
}

function openEditModal(id) {
    fetch(`/api/students/${id}`).then(r => r.json()).then(data => {
        const s = data.student;
        document.getElementById('edit-student-id').value = s.id;
        document.getElementById('edit-student-id-display').value = s.id;
        document.getElementById('edit-student-name').value = s.name;
        document.getElementById('edit-student-class').value = s.class_name || '';
        showModal('student-edit-modal');
    }).catch(e => showToast("Error loading student", "error"));
}

async function saveStudentEdit() {
    const id = document.getElementById('edit-student-id').value;
    const name = document.getElementById('edit-student-name').value;
    const cls = document.getElementById('edit-student-class').value;
    
    showLoading();
    try {
        const res = await fetch(`/api/students/${id}`, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name, class_name: cls})
        });
        hideLoading();
        if ((await res.json()).success) {
            showToast("Student metadata updated", "success");
            closeModal('student-edit-modal');
            loadStudents(currentView);
        } else {
            showToast("Update failed", "error");
        }
    } catch(e) { hideLoading(); showToast("Network error", "error"); }
}


// --- Face Enrollment System (Phase 5) ---

function showStudentCreateModal() {
    document.getElementById('new-student-id').value = '';
    document.getElementById('new-student-name').value = '';
    document.getElementById('new-student-class').value = '';
    showModal('student-create-modal');
}

function validateNewStudentForm() {
    const id = document.getElementById('new-student-id').value.trim();
    const name = document.getElementById('new-student-name').value.trim();
    if(!id || !name) { showToast("ID and Name are required", "error"); return null; }
    return {id, name, class: document.getElementById('new-student-class').value.trim()};
}

function startReEnroll(id, name, cls) {
    enrollState.id = id; enrollState.name = name; enrollState.className = cls;
    document.getElementById('new-student-id').value = id;
    document.getElementById('new-student-name').value = name;
    document.getElementById('new-student-class').value = cls;
    showModal('student-create-modal');
}

function proceedToCameraEnroll() {
    const s = validateNewStudentForm();
    if (!s) return;
    enrollState = { ...s, currentPhase: 0, stream: null, captures: {front:null, left:null, right:null} };
    closeModal('student-create-modal');
    document.getElementById('enroll-cam-title').textContent = enrollState.name;
    document.getElementById('enroll-cam-subtitle').textContent = `ID: ${enrollState.id}`;
    
    // Reset cards
    ENROLL_PHASES.forEach(p => {
        const c = document.getElementById(`thumb-card-${p}`);
        c.className = 'thumb-card';
        c.querySelector('.thumb-check').style.display = 'none';
        c.querySelector('.thumb-placeholder').style.display = 'flex';
        c.querySelector('.thumb-placeholder').classList.remove('error');
        c.querySelector('.thumb-img').style.display = 'none';
    });
    
    document.getElementById('enroll-submit-btn').disabled = true;
    showModal('enroll-modal');
    startEnrollCamera();
    updateEnrollUI();
}

async function startEnrollCamera() {
    try {
        const video = document.getElementById('enroll-camera');
        enrollState.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = enrollState.stream;
    } catch(e) {
        document.getElementById('enroll-instruction').textContent = "Camera access denied or failed.";
        document.getElementById('enroll-instruction').style.color = 'var(--error)';
    }
}

function stopEnrollCamera() {
    if (enrollState.stream) {
        enrollState.stream.getTracks().forEach(t => t.stop());
        enrollState.stream = null;
    }
}

function closeEnrollCameraModal() {
    stopEnrollCamera();
    closeModal('enroll-modal');
}

function updateEnrollUI() {
    const phase = ENROLL_PHASES[enrollState.currentPhase];
    if (!phase) { // All done
        document.getElementById('enroll-capture-btn').disabled = true;
        document.getElementById('enroll-submit-btn').disabled = false;
        document.getElementById('enroll-instruction').textContent = "All angles verified! Click Save Enrollment.";
        return;
    }
    document.getElementById('enroll-capture-btn').disabled = false;
    document.getElementById('enroll-submit-btn').disabled = true;
    document.getElementById('enroll-instruction').textContent = `Look ${phase.toUpperCase()} and capture`;
    
    ENROLL_PHASES.forEach((p, idx) => {
        const card = document.getElementById(`thumb-card-${p}`);
        if(idx === enrollState.currentPhase) card.classList.add('active');
        else card.classList.remove('active');
    });
}

function retakeAngle(angle) {
    if(document.getElementById('enroll-submit-btn').disabled === false) {
        document.getElementById('enroll-submit-btn').disabled = true; // Needs to complete
    }
    const idx = ENROLL_PHASES.indexOf(angle);
    if(idx === -1) return;
    enrollState.currentPhase = idx;
    enrollState.captures[angle] = null;
    const card = document.getElementById(`thumb-card-${angle}`);
    card.className = 'thumb-card active';
    card.querySelector('.thumb-img').style.display = 'none';
    card.querySelector('.thumb-check').style.display = 'none';
    card.querySelector('.thumb-placeholder').style.display = 'flex';
    card.querySelector('.thumb-placeholder').classList.remove('error');
    updateEnrollUI();
}

function flashCamera() {
    const fl = document.getElementById('enroll-flash');
    fl.style.opacity = '1';
    setTimeout(() => fl.style.opacity = '0', 100);
}

async function captureEnrollFrame() {
    if(!enrollState.stream) return;
    flashCamera();
    
    const video = document.getElementById('enroll-camera');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.95));
    
    const phase = ENROLL_PHASES[enrollState.currentPhase];
    
    // Set preview
    const imgUrl = URL.createObjectURL(blob);
    enrollState.captures[phase] = blob;
    
    const card = document.getElementById(`thumb-card-${phase}`);
    const imgEl = card.querySelector('.thumb-img');
    imgEl.src = imgUrl; imgEl.style.display = 'block';
    card.classList.add('has-img');
    card.querySelector('.thumb-placeholder').style.display = 'none';
    
    // Validate with backend
    card.querySelector('.thumb-overlay p').textContent = "VALIDATING...";
    document.getElementById('enroll-capture-btn').disabled = true;
    
    const fd = new FormData();
    fd.append('angle', phase);
    fd.append('image', blob, 'capture.jpg');
    
    try {
        const res = await fetch('/api/enroll/v2/validate', { method: 'POST', body: fd });
        const data = await res.json();
        if(data.success) {
            card.classList.remove('error'); card.classList.add('success');
            card.querySelector('.thumb-check').style.display = 'flex';
            card.querySelector('.thumb-overlay p').textContent = "RETAKE";
            
            // Advance phase implicitly
            enrollState.currentPhase++;
            while(enrollState.currentPhase < 3 && enrollState.captures[ENROLL_PHASES[enrollState.currentPhase]]) {
                enrollState.currentPhase++;
            }
        } else {
            card.classList.add('error');
            card.querySelector('.thumb-overlay p').textContent = "FAILED: " + data.reason;
            card.querySelector('.thumb-placeholder').classList.add('error');
            showToast(data.reason, "error");
        }
    } catch(e) {
        showToast("Validation request failed", "error");
    }
    updateEnrollUI();
}

// Manual Upload Flow
let manualValidated = { front: false, left: false, right: false };

function proceedToManualEnroll() {
    const s = validateNewStudentForm();
    if(!s) return;
    enrollState = { ...s };
    manualValidated = { front: false, left: false, right: false };
    closeModal('student-create-modal');
    document.getElementById('manual-enroll-title').textContent = enrollState.name;
    document.getElementById('manual-enroll-subtitle').textContent = `ID: ${enrollState.id}`;
    document.getElementById('manual-front').value = '';
    document.getElementById('manual-left').value = '';
    document.getElementById('manual-right').value = '';
    ['front','left','right'].forEach(a => {
        const el = document.getElementById(`manual-${a}-err`);
        el.style.display = 'none'; el.textContent = '';
    });
    document.getElementById('manual-enroll-submit').disabled = true;
    showModal('manual-enroll-modal');
}

function checkManualReady() {
    document.getElementById('manual-enroll-submit').disabled =
        !(manualValidated.front && manualValidated.left && manualValidated.right);
}

async function validateManualFile(angle) {
    const input = document.getElementById(`manual-${angle}`);
    const errEl = document.getElementById(`manual-${angle}-err`);
    const file = input.files[0];
    if(!file) { manualValidated[angle] = false; checkManualReady(); return; }

    errEl.textContent = 'Validating...'; errEl.style.display = 'block'; errEl.style.color = 'var(--on-surface-variant)';
    manualValidated[angle] = false;
    checkManualReady();

    const fd = new FormData();
    fd.append('angle', angle);
    fd.append('image', file);
    try {
        const res = await fetch('/api/enroll/v2/validate', { method: 'POST', body: fd });
        const data = await res.json();
        if(data.success) {
            manualValidated[angle] = true;
            errEl.textContent = `\u2713 Valid (quality: ${Math.round(data.quality || 0)})`;
            errEl.style.color = 'var(--primary)';
        } else {
            manualValidated[angle] = false;
            errEl.textContent = `\u2717 ${data.reason}`;
            errEl.style.color = 'var(--error)';
        }
    } catch(e) {
        manualValidated[angle] = false;
        errEl.textContent = 'Validation request failed';
        errEl.style.color = 'var(--error)';
    }
    errEl.style.display = 'block';
    checkManualReady();
}

// Final Submission API call
async function submitEnroll(isManual = false) {
    const fd = new FormData();
    fd.append('student_id', enrollState.id);
    fd.append('name', enrollState.name);
    fd.append('class_name', enrollState.class || '');
    
    if (isManual) {
        fd.append('image_front', document.getElementById('manual-front').files[0]);
        fd.append('image_left', document.getElementById('manual-left').files[0]);
        fd.append('image_right', document.getElementById('manual-right').files[0]);
    } else {
        if(!enrollState.captures.front || !enrollState.captures.left || !enrollState.captures.right) {
            return showToast("All angles required", "error");
        }
        fd.append('image_front', enrollState.captures.front, 'front.jpg');
        fd.append('image_left', enrollState.captures.left, 'left.jpg');
        fd.append('image_right', enrollState.captures.right, 'right.jpg');
    }
    
    showLoading("Generating Encodings...");
    try {
        const res = await fetch('/api/enroll/v2', { method: 'POST', body: fd });
        const data = await res.json();
        hideLoading();
        
        if (data.success) {
            showToast("Enrollment Saved Successfully!", "success");
            if (isManual) closeModal('manual-enroll-modal');
            else closeEnrollCameraModal();
            loadStudents(currentView);
        } else {
            showToast(data.message || "Enrollment failed", "error");
            if(data.phase_results) {
                // Show specific failures
                const errs = data.phase_results.filter(r => !r.success).map(r => `${r.name}: ${r.reason}`).join(" | ");
                if(errs) showToast(errs, "error");
            }
        }
    } catch(e) { hideLoading(); showToast("Network error submitting enrollment", "error"); }
}

// --- History View ---
async function loadHistory() {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '';
        if(!data.sessions || data.sessions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No history recorded</td></tr>`;
            return;
        }
        
        data.sessions.forEach((s, i) => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--outline-variant)';
            const isActive = s.status === 'active';
            const stClass = isActive ? 'active' : 'archived';
            const stText = isActive ? 'Active' : 'Closed';
            
            tr.innerHTML = `
                <td style="padding:12px 16px;color:var(--on-surface-variant)">${i+1}</td>
                <td style="padding:12px 16px;font-weight:600;color:var(--on-surface)">${s.name || 'Session ' + s.id}</td>
                <td style="padding:12px 16px;color:var(--on-surface-variant)">${new Date(s.created_at).toLocaleString()}</td>
                <td style="padding:12px 16px;color:var(--primary);font-weight:600">${s.present_count || 0}</td>
                <td style="padding:12px 16px;color:var(--error);font-weight:600">${s.absent_count || 0}</td>
                <td style="padding:12px 16px"><span class="status-chip ${stClass}">${stText}</span></td>
                <td style="padding:12px 16px;text-align:right">
                    <button class="btn-secondary" style="padding:6px 12px;font-size:0.75rem" onclick="showHistoryDetail(${s.id})">Details</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) { console.error(e); }
}

async function showHistoryDetail(id) {
    const panel = document.getElementById('detail-panel');
    panel.style.display = 'block';
    panel.scrollIntoView({behavior: 'smooth'});
    // Basic implementation since we lack a deep detail endpoint, we will load history
    document.getElementById('detail-present-count').textContent = '--';
    document.getElementById('detail-absent-count').textContent = '--';
}

function hideDetail() { document.getElementById('detail-panel').style.display = 'none'; }


// --- Scan V3 Detection Flow (Phase 7) ---

let cameraSource = 'webcam';
let localStream = null;

async function setCameraSource(source) {
    document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`src-${source}`).classList.add('active');
    
    cameraSource = source;
    isAutoScanning = false;
    updateAutoScanUI();
    
    const feed = document.getElementById('camera-feed');
    const browserCam = document.getElementById('browser-cam');
    
    if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
    }
    
    if (source === 'browser') {
        feed.style.display = 'none';
        browserCam.style.display = 'block';
        try {
            localStream = await navigator.mediaDevices.getUserMedia({video: true});
            browserCam.srcObject = localStream;
            document.getElementById('source-info-text').textContent = 'Local Browser Camera';
            if(sessionActive) startAutoScan();
        } catch(e) {
            showToast("Camera access denied", "error");
            setCameraSource('webcam');
        }
    } else if (source === 'phone') {
        feed.style.display = 'none';
        browserCam.style.display = 'none';
        document.getElementById('source-info-text').textContent = 'Android Phone Camera';
        // Connect via socket mechanism handled dynamically in backend
    } else {
        feed.style.display = 'block';
        browserCam.style.display = 'none';
        document.getElementById('source-info-text').textContent = 'Server Webcam Stream';
        if(sessionActive) startAutoScan();
    }
}

function updateAutoScanInterval() {
    scanIntervalMs = parseInt(document.getElementById('autoscan-interval').value);
    if (isAutoScanning) { stopAutoScan(); startAutoScan(); }
}

async function checkAutoScanStatus() {
    if (sessionActive && !isAutoScanning && cameraSource !== 'phone') {
        startAutoScan();
    }
}

function updateAutoScanUI() {
    const bar = document.getElementById('autoscan-bar');
    if(isAutoScanning) {
        bar.style.display = 'flex';
    } else {
        bar.style.display = 'none';
    }
}

function startAutoScan() {
    if(autoScanInterval) clearInterval(autoScanInterval);
    isAutoScanning = true;
    updateAutoScanUI();
    autoScanInterval = setInterval(performScanV3, scanIntervalMs);
}

function stopAutoScan() {
    isAutoScanning = false;
    updateAutoScanUI();
    if(autoScanInterval) clearInterval(autoScanInterval);
}

async function captureFrameFromSource() {
    /* Always returns a Blob for POST /api/scan/v3, regardless of camera source. */
    const canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');

    if (cameraSource === 'browser') {
        const video = document.getElementById('browser-cam');
        if (!video.videoWidth) return null;
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
    } else {
        // webcam (MJPEG img) or phone — draw the <img> element to canvas
        const img = document.getElementById('camera-feed');
        if (!img.naturalWidth) return null;
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        // For MJPEG cross-origin, we may need try/catch
        try { ctx.drawImage(img, 0, 0); } catch(e) { return null; }
    }
    return new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
}

async function performScanV3() {
    if(!sessionActive || !isAutoScanning) return;
    document.getElementById('autoscan-status').textContent = 'Scanning...';

    const blob = await captureFrameFromSource();
    if (!blob) {
        document.getElementById('autoscan-status').textContent = 'Waiting for frame...';
        return;
    }

    const formData = new FormData();
    formData.append('image', blob, 'scan.jpg');

    try {
        const res = await fetch('/api/scan/v3', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (data.results && data.results.length > 0) {
            handleScanV3Results(data.results);
        } else {
            document.getElementById('autoscan-status').textContent = 'No face detected';
        }
    } catch(e) {
        document.getElementById('autoscan-status').textContent = 'Scan error';
    }
}

function handleScanV3Results(results) {
    let validFound = 0;
    
    results.forEach(r => {
        const isMatch = r.status === 'present' || r.status === 'already';
        if(isMatch && r.student_id && !presentSet.has(r.student_id)) {
            presentSet.add(r.student_id);
            scanCount++;
            validFound++;
            
            showResultCard(r.student_id, r.name, true, `Matched (Confidence: ${(r.confidence*100).toFixed(0)}%)`);
        } else if (r.status === 'already') {
            // Already marked, skip silently
        } else if (!isMatch) {
            showResultCard("Unknown", r.name || "Unknown", false, r.message || "Low confidence/Spoof");
        }
    });
    
    if(validFound > 0) {
        document.getElementById('autoscan-status').textContent = `Matched ${validFound} student(s)!`;
        document.getElementById('autoscan-count').textContent = `${scanCount} scanned`;
        updateScanLogs();
    } else {
        document.getElementById('autoscan-status').textContent = 'Waiting...';
    }
}

function showResultCard(id, name, success, desc) {
    const card = document.getElementById('result-card');
    const icon = document.getElementById('result-icon');
    
    card.style.display = 'flex';
    card.className = success ? 'scan-result-card success' : 'scan-result-card error';
    icon.textContent = success ? 'check_circle' : 'cancel';
    icon.style.color = success ? 'var(--primary)' : 'var(--error)';
    
    document.getElementById('result-name').textContent = name;
    document.getElementById('result-msg').textContent = desc;
    
    setTimeout(() => { card.style.display = 'none'; }, 4000);
}

function updateScanLogs() {
    const list = document.getElementById('present-list');
    const ct = document.getElementById('present-count');
    
    list.innerHTML = '';
    ct.textContent = `${scanCount} total`;
    
    if(scanCount === 0) {
        list.innerHTML = '<p class="empty-state">No data yet</p>';
        document.getElementById('progress-pct').textContent = '0%';
        document.getElementById('progress-bar').style.width = '0%';
        return;
    }
    
    presentSet.forEach(id => {
        const item = document.createElement('div');
        item.style = 'padding:10px 12px;background:var(--surface-container-low);border-radius:var(--radius-md);display:flex;align-items:center;gap:12px;animation:modalIn 0.2s';
        item.innerHTML = `
            <img src="/api/students/${id}/photo" onerror="this.style.display='none'" style="width:36px;height:36px;border-radius:50%;object-fit:cover;background:var(--surface-container-high)">
            <div style="flex:1">
                <p style="font-weight:600;font-size:0.875rem;color:var(--on-surface)">${id}</p>
                <p style="font-size:0.75rem;color:var(--on-surface-variant)">Scanned</p>
            </div>
            <span class="material-symbols-outlined" style="color:var(--primary);font-size:20px">check_circle</span>
        `;
        list.appendChild(item);
    });
    
    const total = Math.max(scanCount, 10);
    const pct = Math.min(Math.round((scanCount / total)*100), 100);
    document.getElementById('progress-pct').textContent = `${pct}%`;
    document.getElementById('progress-bar').style.width = `${pct}%`;
}

// --- Demo Video Toggle ---
let demoPlaying = false;
function toggleDemoVideo() {
    const video = document.getElementById('demo-video');
    const label = document.getElementById('btn-demo-label');
    const feed = document.getElementById('camera-feed');
    const browserCam = document.getElementById('browser-cam');

    if (demoPlaying) {
        video.pause();
        video.style.display = 'none';
        feed.style.display = cameraSource === 'browser' ? 'none' : 'block';
        browserCam.style.display = cameraSource === 'browser' ? 'block' : 'none';
        label.textContent = 'Demo';
        demoPlaying = false;
    } else {
        feed.style.display = 'none';
        browserCam.style.display = 'none';
        video.style.display = 'block';
        video.play();
        label.textContent = 'Stop Demo';
        demoPlaying = true;
    }
}
