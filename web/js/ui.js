import { state } from './state.js';

export function toggleTheme() {
    state.currentTheme = state.currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', state.currentTheme);
    localStorage.setItem('theme', state.currentTheme);
}

export function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { hour12: false });
}

export function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-leave');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function resolveElement(target) {
    return typeof target === 'string' ? document.getElementById(target) : target;
}

export function showElement(target, display = 'block') {
    const el = resolveElement(target);
    if (!el) return;
    el.classList.remove('is-hidden', 'is-block', 'is-flex', 'is-inline-flex');
    el.classList.add(`is-${display}`);
}

export function hideElement(target) {
    const el = resolveElement(target);
    if (!el) return;
    el.classList.remove('is-block', 'is-flex', 'is-inline-flex');
    el.classList.add('is-hidden');
}

export function setTone(target, tone = '') {
    const el = resolveElement(target);
    if (!el) return;
    el.classList.remove('tone-muted', 'tone-primary', 'tone-error');
    if (tone) el.classList.add(`tone-${tone}`);
}

export function showLoading(text = 'Processing...') {
    document.getElementById('loading-text').textContent = text;
    showElement('loading', 'flex');
}

export function hideLoading() {
    hideElement('loading');
}

export function showModal(id) {
    showElement(id, 'flex');
}

export function closeModal(id) {
    hideElement(id);
}

export function confirmDialog(title, msg, type = 'warning') {
    return new Promise(resolve => {
        const d = document.getElementById('confirm-dialog');
        document.getElementById('confirm-title').textContent = title;
        document.getElementById('confirm-msg').textContent = msg;
        const iconContainer = document.getElementById('confirm-icon');
        const iconSym = document.getElementById('confirm-icon-sym');

        iconContainer.className = `confirm-icon ${type}`;
        iconSym.textContent = type === 'warning' ? 'warning' : type === 'error' ? 'error' : 'info';
        showElement(d, 'flex');

        const btnOk = document.getElementById('confirm-ok');
        const btnCancel = document.getElementById('confirm-cancel');
        const cleanup = () => {
            hideElement(d);
            btnOk.removeEventListener('click', onOk);
            btnCancel.removeEventListener('click', onCancel);
        };
        const onOk = () => { cleanup(); resolve(true); };
        const onCancel = () => { cleanup(); resolve(false); };

        btnOk.addEventListener('click', onOk);
        btnCancel.addEventListener('click', onCancel);
    });
}

export function setupTabs() {
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

export function initialsAvatar(name = '') {
    const initials = name.trim().split(/\s+/).slice(0, 2).map(part => part[0] || '').join('').toUpperCase() || '?';
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" rx="16" fill="#1f2937"/><text x="48" y="56" font-size="28" font-family="Arial, sans-serif" font-weight="700" text-anchor="middle" fill="#34c759">${initials}</text></svg>`;
    return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

export function formatDateTime(value) {
    if (!value) return '--';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '--' : date.toLocaleString();
}
