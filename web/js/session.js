import { api } from './api.js';
import { state } from './state.js';
import { confirmDialog, hideElement, hideLoading, showElement, showLoading, showToast } from './ui.js';
import { loadHistory } from './history.js';
import { checkAutoScanStatus, setCameraSource, stopAutoScan, updateScanLogs } from './scan.js';

export async function checkSessionStatus() {
    try {
        const data = await api('/api/session/active');
        const headerStatus = document.getElementById('status-badge');
        const headerDot = document.getElementById('header-dot');
        const dot = document.querySelector('#session-info .dot');
        const txt = document.querySelector('#session-info span:not(.dot)');
        const input = document.getElementById('session-name');

        if (data.active) {
            state.sessionActive = true;
            const sess = data.session;
            headerStatus.textContent = 'Session Active';
            headerDot.className = 'status-badge-dot on';
            dot.className = 'dot on';
            txt.textContent = sess.name || 'Active Session';
            hideElement(input);
            hideElement('btn-start');
            showElement('btn-stop', 'inline-flex');
            checkAutoScanStatus();
        } else {
            state.sessionActive = false;
            headerStatus.textContent = 'Idle';
            headerDot.className = 'status-badge-dot';
            dot.className = 'dot off';
            txt.textContent = 'No active session';
            showElement(input, 'block');
            showElement('btn-start', 'inline-flex');
            hideElement('btn-stop');
            stopAutoScan();
        }
    } catch (e) {
        console.error('Status check failed', e);
    }
}

export async function uiStartSession() {
    const input = document.getElementById('session-name');
    const name = input.value.trim() || `Class_${new Date().toLocaleDateString().replace(/\//g, '-')}`;
    const confirmed = await confirmDialog('Start New Session', `Begin attendance session: ${name}?`, 'info');
    if (!confirmed) return;

    showLoading('Starting session...');
    try {
        const data = await api('/api/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        hideLoading();
        if (data.success) {
            showToast('Session Started', 'success');
            state.presentSet.clear();
            state.presentStudents.clear();
            state.scanCount = 0;
            updateScanLogs();
            hideElement('end-session-summary');
            checkSessionStatus();
            setCameraSource('webcam');
        } else {
            showToast(data.message, 'error');
        }
    } catch (e) {
        hideLoading();
        showToast('Error starting session', 'error');
    }
}

export async function uiEndSession() {
    const confirmed = await confirmDialog('End Session', 'Are you sure you want to end the current attendance session?', 'warning');
    if (!confirmed) return;

    showLoading('Ending session...');
    try {
        const data = await api('/api/session/end', { method: 'POST' });
        hideLoading();
        if (data.success) {
            showToast('Session Ended', 'success');
            document.getElementById('summary-total').textContent = data.total;
            document.getElementById('summary-present').textContent = data.present;
            document.getElementById('summary-absent').textContent = data.absent;
            showElement('end-session-summary', 'block');
            checkSessionStatus();
            loadHistory();
        } else {
            showToast(data.message, 'error');
        }
    } catch (e) {
        hideLoading();
        showToast('Error ending session', 'error');
    }
}

export function hideSummary() {
    hideElement('end-session-summary');
}
