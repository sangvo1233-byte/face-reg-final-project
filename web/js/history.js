import { api } from './api.js';
import { hideElement, showElement, showToast } from './ui.js';

export async function loadHistory() {
    try {
        const data = await api('/api/sessions');
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '';
        if (!data.sessions || data.sessions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No history recorded</td></tr>';
            return;
        }

        data.sessions.forEach((s, i) => {
            const tr = document.createElement('tr');
            tr.className = 'data-row';
            const isActive = s.status === 'active';
            const stClass = isActive ? 'active' : 'archived';
            const stText = isActive ? 'Active' : 'Closed';

            tr.innerHTML = `
                <td class="td-muted">${i + 1}</td>
                <td class="td-strong">${s.name || 'Session ' + s.id}</td>
                <td class="td-muted">${new Date(s.created_at).toLocaleString()}</td>
                <td class="td-present">${s.present_count || 0}</td>
                <td class="td-absent">${s.absent_count || 0}</td>
                <td><span class="status-chip ${stClass}">${stText}</span></td>
                <td class="td-actions"><button class="btn-secondary btn-compact" onclick="showHistoryDetail(${s.id})">Details</button></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error(e);
    }
}

export async function showHistoryDetail(id) {
    const panel = document.getElementById('detail-panel');
    const presentList = document.getElementById('detail-present-list');
    const absentList = document.getElementById('detail-absent-list');

    showElement(panel, 'block');
    panel.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('detail-title').textContent = `Session #${id}`;
    document.getElementById('detail-present-count').textContent = '--';
    document.getElementById('detail-absent-count').textContent = '--';
    presentList.innerHTML = '<li class="empty-state">Loading...</li>';
    absentList.innerHTML = '<li class="empty-state">Loading...</li>';

    try {
        const data = await api(`/api/session/${id}/result`);
        const result = data.result || {};
        const session = data.session || {};
        const present = result.present || [];
        const absent = result.absent || [];

        document.getElementById('detail-title').textContent = session.name || `Session #${id}`;
        document.getElementById('detail-present-count').textContent = result.present_count ?? present.length;
        document.getElementById('detail-absent-count').textContent = result.absent_count ?? absent.length;
        renderHistoryStudentList(presentList, present, 'No students present');
        renderHistoryStudentList(absentList, absent, 'No absences');
    } catch (e) {
        presentList.innerHTML = '<li class="empty-state">Could not load details</li>';
        absentList.innerHTML = '';
        showToast('Error loading session details', 'error');
    }
}

export function hideDetail() {
    hideElement('detail-panel');
}

function renderHistoryStudentList(target, students, emptyText) {
    target.innerHTML = '';
    if (!students.length) {
        target.innerHTML = `<li class="empty-state">${emptyText}</li>`;
        return;
    }
    students.forEach(s => {
        const li = document.createElement('li');
        li.className = 'present-item';
        li.innerHTML = `
            <div>
                <strong>${s.name || s.student_name || s.id || s.student_id}</strong>
                <p>${s.class_name || s.class || ''}</p>
            </div>
            <span>${s.id || s.student_id || ''}</span>
        `;
        target.appendChild(li);
    });
}
