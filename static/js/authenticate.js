let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let authenticateBtn = document.getElementById('authenticateBtn');
let startCameraBtn = document.getElementById('startCamera');
let messageDiv = document.getElementById('message');
let resultDiv = document.getElementById('result');
let resultContent = document.getElementById('resultContent');
let resultOverlay = document.getElementById('resultOverlay');

let stream = null;

// Start camera
startCameraBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        video.srcObject = stream;
        authenticateBtn.disabled = false;
        startCameraBtn.disabled = true;
        showMessage('Camera started successfully. Position your face in the frame.', 'success');
    } catch (err) {
        showMessage('Error accessing camera: ' + err.message, 'error');
    }
});

// Authenticate
authenticateBtn.addEventListener('click', async () => {
    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current frame
    ctx.drawImage(video, 0, 0);
    
    // Get image data
    let imageData = canvas.toDataURL('image/jpeg');
    
    // Disable button during authentication
    authenticateBtn.disabled = true;
    authenticateBtn.textContent = 'Authenticating...';
    showMessage('Processing...', 'info');
    
    try {
        let response = await fetch('/authenticate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        let result = await response.json();
        
        if (result.success) {
            showMessage('Authentication successful!', 'success');
            resultContent.innerHTML = `
                <p><strong>User:</strong> ${result.user}</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
            `;
            resultDiv.style.display = 'block';
            
            // Show overlay
            resultOverlay.textContent = `✓ Welcome, ${result.user}!`;
            resultOverlay.style.background = 'rgba(40, 167, 69, 0.9)';
            resultOverlay.classList.add('show');
            
            // Hide overlay after 3 seconds
            setTimeout(() => {
                resultOverlay.classList.remove('show');
            }, 3000);
        } else {
            showMessage('Authentication failed: ' + result.message, 'error');
            resultContent.innerHTML = `
                <p><strong>Status:</strong> Not authenticated</p>
                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                <p>Please try again or register if you are a new user.</p>
            `;
            resultDiv.style.display = 'block';
            
            // Show overlay
            resultOverlay.textContent = '✗ Authentication Failed';
            resultOverlay.style.background = 'rgba(220, 53, 69, 0.9)';
            resultOverlay.classList.add('show');
            
            // Hide overlay after 3 seconds
            setTimeout(() => {
                resultOverlay.classList.remove('show');
            }, 3000);
        }
    } catch (err) {
        showMessage('Error during authentication: ' + err.message, 'error');
    } finally {
        authenticateBtn.disabled = false;
        authenticateBtn.textContent = 'Authenticate';
    }
});

// Show message
function showMessage(text, type) {
    messageDiv.textContent = text;
    messageDiv.className = 'message ' + type;
    messageDiv.style.display = 'block';
}

// Stop camera when leaving page
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
