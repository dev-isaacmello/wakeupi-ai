// WebSocket Connection
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

// DOM Elements
const leftStatus = document.getElementById('left-eye-status');
const rightStatus = document.getElementById('right-eye-status');
const leftBar = document.getElementById('left-eye-bar');
const rightBar = document.getElementById('right-eye-bar');
const earValue = document.getElementById('ear-value');
const blinkRate = document.getElementById('blink-rate');
const totalBlinks = document.getElementById('total-blinks');
const fpsValue = document.getElementById('fps-value');
const dangerOverlay = document.getElementById('danger-overlay');
const clock = document.getElementById('clock');

// Modern Palette Colors
const colors = {
    accent: '#2997ff',  // Electric Blue
    danger: '#ff453a',  // Neon Red
    success: '#30d158', // Neon Green
    warning: '#ffd60a', // Neon Yellow
    bg_accent: 'rgba(41, 151, 255, 0.2)',
    bg_danger: 'rgba(255, 69, 58, 0.2)'
};

// Chart.js Configuration (Ultra-High-End Aesthetics)
const ctx = document.getElementById('ear-graph').getContext('2d');
// Create dynamic gradient
const gradient = ctx.createLinearGradient(0, 0, 0, 160);
gradient.addColorStop(0, 'rgba(41, 151, 255, 0.5)');
gradient.addColorStop(1, 'rgba(41, 151, 255, 0.0)');

const earChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(60).fill(''),
        datasets: [{
            label: 'EAR',
            data: Array(60).fill(0.3),
            borderColor: colors.accent,
            borderWidth: 3,
            borderCapStyle: 'round',
            borderJoinStyle: 'round',
            tension: 0.5, // Organic flow
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors.accent,
            pointHoverBorderWidth: 3,
            fill: true,
            backgroundColor: gradient
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            y: {
                display: false,
                min: 0.1,
                max: 0.45
            },
            x: { display: false }
        },
        plugins: {
            legend: { display: false },
            tooltip: { 
                enabled: true,
                backgroundColor: 'rgba(20, 20, 25, 0.9)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 10,
                cornerRadius: 8,
                displayColors: false,
                callbacks: {
                    title: () => '',
                    label: (context) => `EAR: ${context.raw.toFixed(3)}`
                }
            }
        }
    }
});

function updateClock() {
    const now = new Date();
    clock.textContent = now.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
}
setInterval(updateClock, 1000);

// State Management for UI Optimization
let isAlertActive = false;

// Initialize eye status elements
function initializeEyeStatus() {
    // Initialize left eye
    leftStatus.textContent = "--";
    leftStatus.style.color = "var(--text-muted)";
    leftBar.style.width = "0%";
    leftBar.style.background = "rgba(255,255,255,0.1)";
    
    // Initialize right eye
    rightStatus.textContent = "--";
    rightStatus.style.color = "var(--text-muted)";
    rightBar.style.width = "0%";
    rightBar.style.background = "rgba(255,255,255,0.1)";
}

// Initialize on page load
initializeEyeStatus();

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // Use requestAnimationFrame for smooth UI updates synced with refresh rate
    requestAnimationFrame(() => {
        // Numeric Data
        earValue.textContent = data.ear.toFixed(2);
        blinkRate.textContent = data.blink_rate;
        totalBlinks.textContent = data.total_blinks;
        fpsValue.textContent = data.fps;

        // Status Indicators
        updateEyeStatus(leftStatus, leftBar, data.left_open);
        updateEyeStatus(rightStatus, rightBar, data.right_open);

        // Chart Update
        const currentData = earChart.data.datasets[0].data;
        currentData.shift();
        currentData.push(data.ear);
        earChart.update('none'); // 'none' mode for zero-latency updates

        // Alert Handling with state check to avoid unnecessary DOM manipulation
        if (data.alert !== isAlertActive) {
            isAlertActive = data.alert;
            if (isAlertActive) {
                dangerOverlay.classList.add('active');
            } else {
                dangerOverlay.classList.remove('active');
            }
        }
    });
};

function updateEyeStatus(textElement, barElement, isOpen) {
    if (isOpen) {
        textElement.textContent = "Aberto";
        textElement.style.color = colors.success;
        
        barElement.style.width = "100%";
        // Restore gradient
        barElement.style.background = `linear-gradient(90deg, ${colors.accent}, #5ac8fa)`;
    } else {
        textElement.textContent = "Fechado";
        textElement.style.color = colors.danger;
        
        barElement.style.width = "5%"; 
        // Override gradient with solid red
        barElement.style.background = colors.danger;
    }
}

ws.onopen = () => {
    console.log("Conectado ao sistema de telemetria.");
};

ws.onclose = () => {
    console.log("Conexão perdida. Reconectando...");
    setTimeout(() => window.location.reload(), 3000);
};
