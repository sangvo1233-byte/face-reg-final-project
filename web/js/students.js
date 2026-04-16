import { api } from './api.js';
import { state } from './state.js';
import { closeModal, confirmDialog, formatDateTime, hideLoading, initialsAvatar, showLoading, showModal, showToast } from './ui.js';

export async function loadStudents(view = state.currentView) {
    try {
        const data = await api(`/api/students?view=${view}`);
        const tbody = document.getElementById('student-tbody');
        tbody.innerHTML = '';
        if (!data.students || data.students.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No students found in this view.</td></tr>';
            return;
        }

        data.students.forEach(s => {
            const tr = document.createElement('tr');
            tr.className = 'data-row';
            const isActive = s.is_active !== 0;
            const statusClass = isActive ? 'active' : 'archived';
            const statusText = isActive ? 'Active' : 'Archived';

            tr.innerHTML = `
                <td class="td-strong">${s.id}</td>
                <td class="student-cell">
                    <img class="student-avatar" src="/api/students/${s.id}/photo" onerror="this.classList.add('is-hidden')">
                    <span>${s.name}</span>
                </td>
                <td class="td-muted">${s.class_name || '--'}</td>
                <td class="td-center"><span class="status-chip ${statusClass}">${statusText}</span></td>
                <td>
                    <div class="action-cell">
                        <button class="btn-icon view" title="View Profile" onclick="showStudentDetail('${s.id}')"><span class="material-symbols-outlined btn-icon-symbol">visibility</span></button>
                        ${isActive ? `
                            <button class="btn-icon edit" title="Edit Metadata" onclick="openEditModal('${s.id}')"><span class="material-symbols-outlined btn-icon-symbol">edit</span></button>
                            <button class="btn-icon enroll" title="Re-enroll Face" onclick="startReEnroll('${s.id}', '${s.name}', '${s.class_name || ''}')"><span class="material-symbols-outlined btn-icon-symbol">face</span></button>
                            <button class="btn-icon archive" title="Archive Student" onclick="archiveStudent('${s.id}')"><span class="material-symbols-outlined btn-icon-symbol">archive</span></button>
                        ` : `
                            <button class="btn-icon restore" title="Restore Student" onclick="restoreStudent('${s.id}')"><span class="material-symbols-outlined btn-icon-symbol">restore_from_trash</span></button>
                        `}
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });

        document.querySelectorAll('.view-tab').forEach(b => {
            b.classList.remove('active');
            if (b.textContent.toLowerCase() === view.toLowerCase()) b.classList.add('active');
        });
        state.currentView = view;
    } catch (e) {
        console.error('Load students error', e);
    }
}

export function setStudentView(view) {
    state.currentView = view;
    loadStudents(view);
}

export async function archiveStudent(id) {
    const confirmed = await confirmDialog('Archive Student', `Are you sure you want to archive student ${id}? Their attendance will not be recorded while archived.`, 'error');
    if (!confirmed) return;

    showLoading('Archiving...');
    try {
        const data = await api(`/api/students/${id}`, { method: 'DELETE' });
        hideLoading();
        if (data.success) {
            showToast('Student archived', 'success');
            loadStudents(state.currentView);
        } else {
            showToast('Error archiving', 'error');
        }
    } catch (e) {
        hideLoading();
        showToast('Network error', 'error');
    }
}

export async function restoreStudent(id) {
    const confirmed = await confirmDialog('Restore Student', `Re-activate student ${id}?`, 'info');
    if (!confirmed) return;

    showLoading('Restoring...');
    try {
        const data = await api(`/api/students/${id}/restore`, { method: 'POST' });
        hideLoading();
        if (data.success) {
            showToast('Student restored', 'success');
            loadStudents(state.currentView);
        } else {
            showToast(data.message || 'Error', 'error');
        }
    } catch (e) {
        hideLoading();
        showToast('Network error', 'error');
    }
}

export async function showStudentDetail(id) {
    showLoading();
    try {
        const data = await api(`/api/students/${id}`);
        hideLoading();

        const s = data.student;
        const photo = document.getElementById('detail-photo');
        photo.onerror = () => { photo.onerror = null; photo.src = initialsAvatar(s.name); };
        photo.src = `/api/students/${id}/photo?t=${Date.now()}`;
        document.getElementById('detail-name').textContent = s.name;
        document.getElementById('detail-id').textContent = `ID: ${s.id}`;
        document.getElementById('detail-class').textContent = `Class: ${s.class_name || '--'}`;
        document.getElementById('detail-embeddings').textContent = `${data.embedding_count || 0} reference angles recorded.`;

        const status = document.getElementById('detail-status');
        status.textContent = s.is_active === 0 ? 'ARCHIVED' : 'ACTIVE';
        status.className = s.is_active === 0 ? 'status-chip archived' : 'status-chip active';

        const hist = document.getElementById('detail-history-list');
        hist.innerHTML = '';
        if (!data.history || data.history.length === 0) {
            hist.innerHTML = '<p class="empty-state">No attendance history</p>';
        } else {
            data.history.forEach(h => {
                const el = document.createElement('div');
                el.className = 'detail-history-item';
                const rowStatus = (h.status || 'present').toLowerCase();
                const scannedAt = h.scanned_at || h.session_date || h.created_at;
                el.innerHTML = `
                    <div><p>${h.session_name || 'Session ' + h.session_id}</p><span>${formatDateTime(scannedAt)}</span></div>
                    <span class="status-chip ${rowStatus === 'present' ? 'active' : 'archived'}">${rowStatus}</span>
                `;
                hist.appendChild(el);
            });
        }
        showModal('student-detail-modal');
    } catch (e) {
        hideLoading();
        showToast('Error loading details', 'error');
    }
}

export function openEditModal(id) {
    api(`/api/students/${id}`).then(data => {
        const s = data.student;
        document.getElementById('edit-student-id').value = s.id;
        document.getElementById('edit-student-id-display').value = s.id;
        document.getElementById('edit-student-name').value = s.name;
        document.getElementById('edit-student-class').value = s.class_name || '';
        showModal('student-edit-modal');
    }).catch(() => showToast('Error loading student', 'error'));
}

export async function saveStudentEdit() {
    const id = document.getElementById('edit-student-id').value;
    const name = document.getElementById('edit-student-name').value;
    const cls = document.getElementById('edit-student-class').value;

    showLoading();
    try {
        const data = await api(`/api/students/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, class_name: cls })
        });
        hideLoading();
        if (data.success) {
            showToast('Student metadata updated', 'success');
            closeModal('student-edit-modal');
            loadStudents(state.currentView);
        } else {
            showToast('Update failed', 'error');
        }
    } catch (e) {
        hideLoading();
        showToast('Network error', 'error');
    }
}
