// SilentVoice - Frontend JavaScript

// DOM Elements
let videoFeed, currentFingersEl, currentMessageEl, historyListEl, totalGesturesEl, fpsEl;
let lastGesture = null;
let frameCount = 0;
let lastFrameTime = Date.now();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    videoFeed = document.getElementById('videoFeed');
    currentFingersEl = document.getElementById('currentFingers');
    currentMessageEl = document.getElementById('currentMessage');
    historyListEl = document.getElementById('historyList');
    totalGesturesEl = document.getElementById('totalGestures');
    fpsEl = document.getElementById('fps');

    // Start updates
    startUpdates();
    startFpsCounter();
});

// Start periodic updates
function startUpdates() {
    updateGesture();
    updateHistory();
    updateStats();

    setInterval(() => {
        updateGesture();
        updateHistory();
        updateStats();
    }, 200);
}

// Update current gesture display
function updateGesture() {
    fetch('/api/gesture')
        .then(response => response.json())
        .then(data => {
            if (data.fingers !== undefined && data.fingers !== null) {
                // Update display
                const fingerIcons = ['✊', '☝️', '✌️', '🤟', '🖖', '🖐️'];
                currentFingersEl.innerHTML = `${fingerIcons[data.fingers] || '✋'} ${data.fingers}`;
                currentMessageEl.innerHTML = data.message;
                currentMessageEl.style.background = data.color;

                // Emergency animation
                if (data.fingers === 5) {
                    currentMessageEl.classList.add('emergency');
                    showEmergencyAlert();
                } else {
                    currentMessageEl.classList.remove('emergency');
                }

                lastGesture = data;
            }
        })
        .catch(error => console.error('Gesture fetch error:', error));
}

// Update message history
function updateHistory() {
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            if (!historyListEl) return;

            if (data.length === 0) {
                historyListEl.innerHTML = '<div style="text-align: center; opacity: 0.5;">No messages yet</div>';
                return;
            }

            historyListEl.innerHTML = data.map(item => `
                <div class="history-item">
                    <span class="history-time">${item.timestamp}</span>
                    <span class="history-fingers">${'✋'.repeat(item.fingers)}</span>
                    <span class="history-message">${item.message.substring(0, 40)}</span>
                </div>
            `).join('');

            if (totalGesturesEl) {
                totalGesturesEl.innerHTML = data.length;
            }
        })
        .catch(error => console.error('History fetch error:', error));
}

// Update statistics
function updateStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Additional stats can be displayed here
        })
        .catch(error => console.error('Stats fetch error:', error));
}

// Repeat last message
function repeatMessage() {
    if (lastGesture && lastGesture.message) {
        fetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: lastGesture.message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('🔊 Speaking: ' + lastGesture.message);
            }
        })
        .catch(error => console.error('Speak error:', error));
    }
}

// Clear message history
function clearHistory() {
    if (confirm('Are you sure you want to clear all message history?')) {
        fetch('/api/clear_history', { method: 'POST' })
            .then(() => {
                updateHistory();
                showToast('🗑️ History cleared');
            })
            .catch(error => console.error('Clear history error:', error));
    }
}

// Show emergency alert
function showEmergencyAlert() {
    // Check if already in emergency mode
    if (window.emergencyActive) return;
    window.emergencyActive = true;

    // Create toast notification
    showToast('🚨 EMERGENCY DETECTED! Help needed! 🚨', 5000);

    // Optional: Redirect to emergency page after 3 seconds of continuous emergency
    setTimeout(() => {
        window.emergencyActive = false;
    }, 3000);
}

// Show toast notification
function showToast(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, duration);
}

// FPS counter
function startFpsCounter() {
    setInterval(() => {
        frameCount++;
        const now = Date.now();
        if (now - lastFrameTime >= 1000) {
            if (fpsEl) {
                fpsEl.innerHTML = frameCount;
            }
            frameCount = 0;
            lastFrameTime = now;
        }
    }, 100);
}

// Refresh video feed on error
if (videoFeed) {
    videoFeed.onerror = function() {
        console.log('Video feed error, refreshing...');
        setTimeout(() => {
            videoFeed.src = videoFeed.src.split('?')[0] + '?' + Date.now();
        }, 1000);
    };
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') {
        repeatMessage();
    } else if (e.key === 'c' || e.key === 'C') {
        clearHistory();
    }
});

// Export for console debugging
window.SilentVoice = {
    repeatMessage,
    clearHistory,
    getLastGesture: () => lastGesture
};