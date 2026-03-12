let API_BASE_URL = window.API_BASE_URL || 'http://127.0.0.1:8000';
while (API_BASE_URL.endsWith('/')) API_BASE_URL = API_BASE_URL.slice(0, -1);

const ADMIN_USERS = { admin: 'admin' };
const ADMIN_SESSION_KEY = 'is_admin';

function isAdmin() {
    return sessionStorage.getItem(ADMIN_SESSION_KEY) === '1';
}

function showApp() {
    const login = document.getElementById('loginScreen');
    const app = document.getElementById('appContainer');
    if (login) login.classList.add('hidden');
    if (app) app.classList.remove('hidden');
}

function showLogin() {
    const login = document.getElementById('loginScreen');
    const app = document.getElementById('appContainer');
    const err = document.getElementById('loginError');
    if (app) app.classList.add('hidden');
    if (login) login.classList.remove('hidden');
    if (err) err.classList.add('hidden');
}

function initAuth() {
    if (isAdmin()) {
        showApp();
        return;
    }

    showLogin();

    const form = document.getElementById('loginForm');
    const userEl = document.getElementById('loginUsername');
    const passEl = document.getElementById('loginPassword');
    const err = document.getElementById('loginError');

    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const username = (userEl?.value || '').trim();
            const password = passEl?.value || '';

            if (ADMIN_USERS[username] && ADMIN_USERS[username] === password) {
                sessionStorage.setItem(ADMIN_SESSION_KEY, '1');
                showApp();
                loadPlots();
            } else {
                if (err) err.classList.remove('hidden');
            }
        });
    }

    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            sessionStorage.removeItem(ADMIN_SESSION_KEY);
            showLogin();
        });
    }
}

function loadPlots() {
    if (!isAdmin()) return;

    const ts = Date.now();
    const imgs = document.querySelectorAll('img[data-plot]');
    imgs.forEach((img) => {
        const filename = img.getAttribute('data-plot');
        img.src = `${API_BASE_URL}/outputs-frontend/${encodeURIComponent(filename)}?t=${ts}`;
        img.onerror = () => {
            img.closest('.plot-card')?.classList.add('plot-missing');
        };
    });
}

const churnForm = document.getElementById('churnForm');
if (churnForm) {
    churnForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        if (!isAdmin()) {
            showLogin();
            return;
        }

        const resultCard = document.getElementById('resultCard');
        const loading = document.getElementById('loading');
        const placeholder = document.getElementById('placeholderResult');
        const predictBtn = document.getElementById('predictBtn');

        resultCard.classList.add('hidden');
        placeholder.classList.add('hidden');
        loading.classList.remove('hidden');
        predictBtn.disabled = true;

        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        const payload = {
            ...data,
            tenure: parseInt(data.tenure),
            MonthlyCharges: parseFloat(data.MonthlyCharges),
            TotalCharges: parseFloat(data.TotalCharges),
            SeniorCitizen: parseInt(data.SeniorCitizen),
        };

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                throw new Error(`Errore API: ${response.statusText}`);
            }

            const result = await response.json();
            showResult(result);
        } catch (error) {
            alert('Errore nella comunicazione con il server: ' + error.message);
            placeholder.classList.remove('hidden');
            placeholder.innerText = 'Si e verificato un errore.';
        } finally {
            loading.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });
}

function showResult(data) {
    const resultCard = document.getElementById('resultCard');
    const probPercent = document.getElementById('probPercent');
    const gaugeFill = document.getElementById('gaugeFill');
    const statusBox = document.getElementById('statusBox');
    const statusText = document.getElementById('statusText');
    const statusIcon = document.getElementById('statusIcon');

    const prob = data.churn_probability;
    const percentage = Math.round(prob * 100);
    const isChurn = data.prediction === 1;

    probPercent.innerText = `${percentage}%`;
    statusText.innerText = isChurn ? 'Rischio Alto' : 'Cliente Fedele';
    statusIcon.innerText = isChurn ? 'warning' : 'verified';

    const color = isChurn ? '#ef4444' : '#10b981';
    gaugeFill.style.background = `conic-gradient(${color} ${percentage}%, #e5e7eb ${percentage}%)`;

    statusBox.className = 'status-box ' + (isChurn ? 'status-churn' : 'status-stay');
    resultCard.classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
    initAuth();

    if (isAdmin()) {
        loadPlots();
        const btn = document.getElementById('reloadPlots');
        if (btn) btn.addEventListener('click', loadPlots);
    }
});
