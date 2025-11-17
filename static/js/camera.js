// Camera.js - Handle webcam access, image capture, and server communication

let stream = null;
let capturedImages = [];
const MAX_IMAGES = 5;

/**
 * Initialize camera access
 */
async function startCamera() {
    const video = document.getElementById('video');
    const startBtn = document.getElementById('startCamera');
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            } 
        });
        video.srcObject = stream;
        startBtn.textContent = 'Camera Active';
        startBtn.disabled = true;
        
        // Enable capture/authenticate button
        const captureBtn = document.getElementById('captureBtn');
        const authBtn = document.getElementById('authenticateBtn');
        if (captureBtn) captureBtn.disabled = false;
        if (authBtn) authBtn.disabled = false;
        
        showMessage('Camera started successfully', 'success');
    } catch (error) {
        showMessage('Error accessing camera: ' + error.message, 'error');
        console.error('Camera error:', error);
    }
}

/**
 * Stop camera stream
 */
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

/**
 * Capture image from video stream
 */
function captureImage() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    return imageData;
}

/**
 * Display message to user
 */
function showMessage(message, type = 'info') {
    const messageDiv = document.getElementById('message') || document.getElementById('result');
    if (!messageDiv) return;
    
    messageDiv.textContent = message;
    messageDiv.className = `message ${type}`;
    messageDiv.style.display = 'block';
}

/**
 * Display result with styling
 */
function showResult(data) {
    const resultDiv = document.getElementById('result');
    if (!resultDiv) return;
    
    let html = '';
    if (data.success) {
        html = `
            <div class="result-success">
                <div class="result-icon">✓</div>
                <h3>Authentication Successful!</h3>
                <p><strong>User ID:</strong> ${data.user_id}</p>
                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                <p>${data.message}</p>
            </div>
        `;
    } else {
        html = `
            <div class="result-failure">
                <div class="result-icon">✗</div>
                <h3>Authentication Failed</h3>
                ${data.confidence ? `<p><strong>Best Match Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>` : ''}
                <p>${data.message}</p>
            </div>
        `;
    }
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
}

/**
 * Initialize registration page
 */
function initializeRegistration() {
    const startBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureBtn');
    const registerBtn = document.getElementById('registerBtn');
    const clearBtn = document.getElementById('clearBtn');
    const userIdInput = document.getElementById('userId');
    
    // Start camera
    startBtn.addEventListener('click', startCamera);
    
    // Capture image
    captureBtn.addEventListener('click', () => {
        if (capturedImages.length >= MAX_IMAGES) {
            showMessage(`Maximum ${MAX_IMAGES} images reached`, 'warning');
            return;
        }
        
        const imageData = captureImage();
        capturedImages.push(imageData);
        
        // Update UI
        displayCapturedImages();
        updateCaptureCount();
        
        // Enable register button if enough images
        if (capturedImages.length >= 3) {
            registerBtn.disabled = false;
        }
        
        if (capturedImages.length >= MAX_IMAGES) {
            captureBtn.disabled = true;
            showMessage(`${MAX_IMAGES} images captured. Ready to register!`, 'success');
        } else {
            showMessage(`Image ${capturedImages.length} captured`, 'success');
        }
    });
    
    // Register user
    registerBtn.addEventListener('click', async () => {
        const userId = userIdInput.value.trim();
        
        if (!userId) {
            showMessage('Please enter a User ID', 'error');
            return;
        }
        
        if (capturedImages.length < 3) {
            showMessage('Please capture at least 3 images', 'error');
            return;
        }
        
        registerBtn.disabled = true;
        registerBtn.textContent = 'Registering...';
        
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    images: capturedImages
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showMessage(data.message, 'success');
                // Reset form
                setTimeout(() => {
                    capturedImages = [];
                    displayCapturedImages();
                    updateCaptureCount();
                    userIdInput.value = '';
                    registerBtn.disabled = true;
                    captureBtn.disabled = false;
                    registerBtn.textContent = 'Register User';
                }, 2000);
            } else {
                showMessage(data.message, 'error');
                registerBtn.disabled = false;
                registerBtn.textContent = 'Register User';
            }
        } catch (error) {
            showMessage('Error: ' + error.message, 'error');
            registerBtn.disabled = false;
            registerBtn.textContent = 'Register User';
        }
    });
    
    // Clear captured images
    clearBtn.addEventListener('click', () => {
        capturedImages = [];
        displayCapturedImages();
        updateCaptureCount();
        registerBtn.disabled = true;
        captureBtn.disabled = false;
        showMessage('All images cleared', 'info');
    });
    
    function displayCapturedImages() {
        const previewDiv = document.getElementById('imagePreview');
        const clearBtn = document.getElementById('clearBtn');
        
        if (capturedImages.length === 0) {
            previewDiv.innerHTML = '<p class="no-images">No images captured yet</p>';
            clearBtn.style.display = 'none';
        } else {
            previewDiv.innerHTML = capturedImages.map((img, index) => `
                <div class="preview-item">
                    <img src="${img}" alt="Capture ${index + 1}">
                    <span class="preview-label">${index + 1}</span>
                </div>
            `).join('');
            clearBtn.style.display = 'block';
        }
    }
    
    function updateCaptureCount() {
        const countSpan = document.getElementById('captureCount');
        if (countSpan) {
            countSpan.textContent = capturedImages.length;
        }
    }
    
    // Initial display
    displayCapturedImages();
}

/**
 * Initialize authentication page
 */
function initializeAuthentication() {
    const startBtn = document.getElementById('startCamera');
    const authBtn = document.getElementById('authenticateBtn');
    
    // Start camera
    startBtn.addEventListener('click', startCamera);
    
    // Authenticate
    authBtn.addEventListener('click', async () => {
        const imageData = captureImage();
        
        authBtn.disabled = true;
        authBtn.textContent = 'Authenticating...';
        
        try {
            const response = await fetch('/api/authenticate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData
                })
            });
            
            const data = await response.json();
            showResult(data);
            
            authBtn.disabled = false;
            authBtn.textContent = 'Authenticate';
        } catch (error) {
            showMessage('Error: ' + error.message, 'error');
            authBtn.disabled = false;
            authBtn.textContent = 'Authenticate';
        }
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
});
