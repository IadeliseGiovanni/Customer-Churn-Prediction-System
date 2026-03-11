document.getElementById('churnForm').addEventListener('submit', async function(e) {
    e.preventDefault(); // Evita il reload della pagina

    // Riferimenti UI
    const resultCard = document.getElementById('resultCard');
    const loading = document.getElementById('loading');
    const placeholder = document.getElementById('placeholderResult');
    const predictBtn = document.getElementById('predictBtn');

    // Reset UI
    resultCard.classList.add('hidden');
    placeholder.classList.add('hidden');
    loading.classList.remove('hidden');
    predictBtn.disabled = true;

    // Raccolta Dati dal Form
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    // Conversione tipi (Importante: HTML restituisce sempre stringhe)
    // I select sono già stringhe corrette, ma convertiamo i numeri
    const payload = {
        ...data,
        tenure: parseInt(data.tenure),
        MonthlyCharges: parseFloat(data.MonthlyCharges),
        TotalCharges: parseFloat(data.TotalCharges),
        SeniorCitizen: parseInt(data.SeniorCitizen)
    };

    try {
        // Chiamata all'API (Assicurati che l'URL sia corretto)
        const response = await fetch('http://127.0.0.1:8000/predict', {
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
        alert("Errore nella comunicazione con il server: " + error.message);
        placeholder.classList.remove('hidden');
        placeholder.innerText = "Si è verificato un errore.";
    } finally {
        loading.classList.add('hidden');
        predictBtn.disabled = false;
    }
});

function showResult(data) {
    const resultCard = document.getElementById('resultCard');
    const probPercent = document.getElementById('probPercent');
    const gaugeFill = document.getElementById('gaugeFill');
    const statusBox = document.getElementById('statusBox');
    const statusText = document.getElementById('statusText');
    const statusIcon = document.getElementById('statusIcon');

    // Estrai dati
    const prob = data.churn_probability; // 0.0 a 1.0
    const percentage = Math.round(prob * 100);
    const isChurn = data.prediction === 1;

    // Aggiorna Testi
    probPercent.innerText = `${percentage}%`;
    statusText.innerText = isChurn ? "Rischio Alto" : "Cliente Fedele";
    statusIcon.innerText = isChurn ? "warning" : "verified";

    // Aggiorna Colori e Grafica
    const color = isChurn ? "#ef4444" : "#10b981"; // Rosso o Verde
    gaugeFill.style.background = `conic-gradient(${color} ${percentage}%, #e5e7eb ${percentage}%)`;
    
    statusBox.className = "status-box " + (isChurn ? "status-churn" : "status-stay");

    resultCard.classList.remove('hidden');
}