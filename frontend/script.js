let scanHistory = [];

// 1. SYNC SCROLLING: Essential for XAI highlights to line up with text
function syncScroll() {
    const textarea = document.getElementById('emailText');
    const highlightLayer = document.getElementById('highlightLayer');
    highlightLayer.scrollTop = textarea.scrollTop;
}

// 2. LIVE UPDATES: Runs every time user types
function handleInput() {
    updateLiveStats();
    // Clear highlights when typing starts to maintain performance
    document.getElementById('highlightLayer').innerHTML = document.getElementById('emailText').value;
}

// 3. MAIN AI SCAN FUNCTION
async function initiateDeepScan() {
    const text = document.getElementById('emailText').value;
    if (!text.trim()) return alert("No input data found.");

    const btn = document.getElementById('scanBtn');
    const laser = document.getElementById('scanLaser');
    const tStart = performance.now();

    btn.innerText = "Analyzing Neural Patterns...";
    btn.disabled = true;
    laser.classList.remove('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        const tEnd = performance.now();
        
        renderResults(data, (tEnd - tStart).toFixed(0), text);
        
        // Apply Explainable AI highlights after results come back
        applyXAI(text, data.triggers);

    } catch (err) {
        console.error(err);
        alert("System Error: AI Backend unreachable.");
    } finally {
        btn.innerText = "Analyze the E-Mail";
        btn.disabled = false;
        laser.classList.add('hidden');
    }
}

// 4. XAI HIGHLIGHTING LOGIC
function applyXAI(text, triggers) {
    let highlighted = text;

    if (triggers && triggers.length > 0) {
        triggers.forEach(word => {
            // Use Regex to find the word safely (case-insensitive)
            const regex = new RegExp(`(${word})`, 'gi');
            highlighted = highlighted.replace(regex, `<span class="xai-danger">$1</span>`);
        });
    }
    
    document.getElementById('highlightLayer').innerHTML = highlighted + '\n';
}

// 5. RENDER UI RESULTS
function renderResults(data, latency, originalText) {
    const panel = document.getElementById('resultPanel');
    panel.classList.remove('hidden');

    const prob = (data.confidence_score * 100).toFixed(2);
    document.getElementById('percent').innerText = prob + "%";
    document.getElementById('latency').innerText = latency + "ms";
    document.getElementById('confLevel').innerText = Math.floor(prob) + "/100";

    const fill = document.getElementById('riskFill');
    const verdictLabel = document.getElementById('verdict');
    fill.style.width = prob + "%";

    verdictLabel.innerText = data.verdict;
    
    // UI Theme update based on verdict
    if (data.verdict.includes("PHISHING")) {
        verdictLabel.style.color = "#ef4444";
        fill.style.backgroundColor = "#ef4444";
        document.getElementById('reasoning').innerText = data.is_blacklisted 
            ? "CRITICAL: Malicious domain found in global blacklist." 
            : `Detected patterns: ${data.triggers.join(', ')}`;
        document.getElementById('action').innerText = "BLOCK: Do not interact. Reported to Security.";
    } else if (data.verdict.includes("SUSPICIOUS")) {
        verdictLabel.style.color = "#f59e0b";
        fill.style.backgroundColor = "#f59e0b";
        document.getElementById('reasoning').innerText = "Potential social engineering or unusual link detected.";
        document.getElementById('action').innerText = "CAUTION: Verify identity manually.";
    } else {
        verdictLabel.style.color = "#10b981";
        fill.style.backgroundColor = "#10b981";
        document.getElementById('reasoning').innerText = "Neutral tone. No dangerous signatures found.";
        document.getElementById('action').innerText = "SAFE: Standard communication.";
    }

    logToSidebar(data.verdict, prob, originalText);
}

// 6. HISTORY & UTILITIES
function logToSidebar(v, s, fullText) {
    const histContainer = document.getElementById('scanHistory');
    const entry = { id: Date.now(), verdict: v, score: s, content: fullText };
    scanHistory.push(entry);

    const div = document.createElement('div');
    div.className = "history-item";
    div.onclick = () => recallEmail(entry.id);
    div.innerHTML = `<strong>${v}</strong> <small>(${s}%)</small><br><span style="font-size:10px; opacity:0.6">${fullText.substring(0,30)}...</span>`;
    
    histContainer.prepend(div);
}

function recallEmail(id) {
    const record = scanHistory.find(item => item.id === id);
    if (record) {
        document.getElementById('emailText').value = record.content;
        handleInput(); // Re-trigger highlights and stats
    }
}

function updateLiveStats() {
    const txt = document.getElementById('emailText').value;
    document.getElementById('charCount').innerText = `${txt.length} characters`;
    document.getElementById('wordCount').innerText = `${txt.trim() ? txt.trim().split(/\s+/).length : 0} words`;
}

function resetDashboard() {
    document.getElementById('emailText').value = "";
    document.getElementById('highlightLayer').innerHTML = "";
    document.getElementById('resultPanel').classList.add('hidden');
    updateLiveStats();
}

function copyResult() {
    const report = `PHISHING SHIELD REPORT: ${document.getElementById('verdict').innerText} | Threat: ${document.getElementById('percent').innerText}`;
    navigator.clipboard.writeText(report);
    alert("Security Report Copied!");
}

function reportToIT() {
    const text = document.getElementById('emailText').value;
    console.log("Reporting threat...", text);
    alert("⚠️ ALERT: This email has been logged and sent to the IT Security Team for investigation.");
}

function clearHistory() { 
    document.getElementById('scanHistory').innerHTML = ""; 
    scanHistory = [];
}