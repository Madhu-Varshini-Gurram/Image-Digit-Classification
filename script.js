document.addEventListener('DOMContentLoaded', () => {
    // Tab Logic
    const tabDraw = document.getElementById('tabDraw');
    const tabUpload = document.getElementById('tabUpload');
    const contentDraw = document.getElementById('contentDraw');
    const contentUpload = document.getElementById('contentUpload');

    let currentMode = 'draw'; // 'draw' or 'upload'

    function switchTab(mode) {
        currentMode = mode;
        if (mode === 'draw') {
            tabDraw.classList.add('active'); contentDraw.classList.add('active');
            tabUpload.classList.remove('active'); contentUpload.classList.remove('active');
        } else {
            tabUpload.classList.add('active'); contentUpload.classList.add('active');
            tabDraw.classList.remove('active'); contentDraw.classList.remove('active');
        }
    }

    tabDraw.addEventListener('click', () => switchTab('draw'));
    tabUpload.addEventListener('click', () => switchTab('upload'));

    // Canvas Logic
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 22;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    function getCoords(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX || e.touches[0].clientX) - rect.left,
            y: (e.clientY || e.touches[0].clientY) - rect.top
        };
    }

    canvas.addEventListener('mousedown', (e) => { isDrawing = true; ctx.beginPath(); ctx.moveTo(getCoords(e).x, getCoords(e).y); });
    canvas.addEventListener('mousemove', (e) => { if (isDrawing) { ctx.lineTo(getCoords(e).x, getCoords(e).y); ctx.stroke(); } });
    window.addEventListener('mouseup', () => isDrawing = false);

    canvas.addEventListener('touchstart', (e) => { if (e.target === canvas) e.preventDefault(); isDrawing = true; ctx.beginPath(); ctx.moveTo(getCoords(e).x, getCoords(e).y); }, { passive: false });
    canvas.addEventListener('touchmove', (e) => { if (e.target === canvas) e.preventDefault(); if (isDrawing) { ctx.lineTo(getCoords(e).x, getCoords(e).y); ctx.stroke(); } }, { passive: false });
    window.addEventListener('touchend', () => isDrawing = false);

    // Upload Logic
    const uploadArea = document.getElementById('uploadArea');
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    let uploadedImageBase64 = null;

    uploadArea.addEventListener('click', () => imageUpload.click());

    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                uploadedImageBase64 = ev.target.result;
                imagePreview.src = uploadedImageBase64;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // UI Helpers
    const predictionResult = document.getElementById('predictionResult');
    const probabilitiesSection = document.getElementById('probabilitiesSection');

    function setupProbabilities() {
        probabilitiesSection.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            probabilitiesSection.innerHTML += `
                <div class="prob-row">
                    <div class="digit-label">${i}</div>
                    <div class="bar-container"><div id="bar-${i}" class="bar-fill"></div></div>
                    <div id="pct-${i}" class="prob-pct">0%</div>
                </div>`;
        }
    }
    setupProbabilities();

    function clearApp() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        imagePreview.style.display = 'none';
        imageUpload.value = '';
        uploadedImageBase64 = null;
        predictionResult.textContent = '-';
        for (let i = 0; i < 10; i++) {
            document.getElementById(`bar-${i}`).style.width = '0%';
            document.getElementById(`pct-${i}`).textContent = '0%';
        }
    }

    document.getElementById('clearBtn').addEventListener('click', clearApp);

    // Predict
    document.getElementById('predictBtn').addEventListener('click', async () => {
        let imageData = null;

        if (currentMode === 'draw') {
            imageData = canvas.toDataURL('image/png');
        } else if (currentMode === 'upload') {
            if (!uploadedImageBase64) return alert("Please upload an image first.");
            imageData = uploadedImageBase64;
        }

        const btn = document.getElementById('predictBtn');
        btn.disabled = true; btn.textContent = 'Processing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();
            if (data.error) alert(data.error);
            else {
                predictionResult.textContent = data.prediction;
                data.probabilities.forEach((prob, i) => {
                    const pct = (prob * 100).toFixed(1);
                    document.getElementById(`bar-${i}`).style.width = `${pct}%`;
                    document.getElementById(`pct-${i}`).textContent = `${pct}%`;
                });
            }
        } catch (e) {
            alert("Error analyzing digit. Is the backend running?");
        } finally {
            btn.disabled = false; btn.textContent = 'Analyze Digit';
        }
    });
});
