let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let captureBtn = document.getElementById('captureBtn');
let registerBtn = document.getElementById('registerBtn');
let startCameraBtn = document.getElementById('startCamera');
let imageCountSpan = document.getElementById('imageCount');
let messageDiv = document.getElementById('message');
let capturedImagesDiv = document.getElementById('capturedImages');

let capturedImages = [];
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
        captureBtn.disabled = false;
        startCameraBtn.disabled = true;
        showMessage('Camera started successfully', 'success');
    } catch (err) {
        showMessage('Error accessing camera: ' + err.message, 'error');
    }
});

// Capture image
captureBtn.addEventListener('click', () => {
    if (capturedImages.length >= 10) {
        showMessage('Maximum 10 images allowed', 'error');
        return;
    }
    
    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current frame
    ctx.drawImage(video, 0, 0);
    
    // Get image data
    let imageData = canvas.toDataURL('image/jpeg');
    capturedImages.push(imageData);
    
    // Update UI
    imageCountSpan.textContent = capturedImages.length;
    displayThumbnails();
    
    // Enable register button if we have at least 5 images
    if (capturedImages.length >= 5) {
        registerBtn.disabled = false;
    }
    
    // Disable capture button if we have 10 images
    if (capturedImages.length >= 10) {
        captureBtn.disabled = true;
    }
    
    showMessage(`Captured ${capturedImages.length} image(s). Need at least 5 to register.`, 'info');
});

// Display thumbnails
function displayThumbnails() {
    capturedImagesDiv.innerHTML = '';
    capturedImages.forEach((imgData, index) => {
        let div = document.createElement('div');
        div.className = 'thumbnail';
        
        let img = document.createElement('img');
        img.src = imgData;
        
        let removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = '×';
        removeBtn.onclick = () => removeImage(index);
        
        div.appendChild(img);
        div.appendChild(removeBtn);
        capturedImagesDiv.appendChild(div);
    });
}

// Remove image
function removeImage(index) {
    capturedImages.splice(index, 1);
    imageCountSpan.textContent = capturedImages.length;
    displayThumbnails();
    
    // Update buttons
    captureBtn.disabled = false;
    if (capturedImages.length < 5) {
        registerBtn.disabled = true;
    }
    
    showMessage(`Removed image. ${capturedImages.length} image(s) remaining.`, 'info');
}

// Register user
registerBtn.addEventListener('click', async () => {
    let username = document.getElementById('username').value.trim();
    
    if (!username) {
        showMessage('Please enter a username', 'error');
        return;
    }
    
    if (capturedImages.length < 5) {
        showMessage('Please capture at least 5 images', 'error');
        return;
    }
    
    // Disable buttons during registration
    registerBtn.disabled = true;
    captureBtn.disabled = true;
    registerBtn.textContent = 'Registering...';
    
    try {
        let response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                images: capturedImages
            })
        });
        
        let result = await response.json();
        
        if (result.success) {
            showMessage(result.message, 'success');
            // Reset form
            capturedImages = [];
            displayThumbnails();
            imageCountSpan.textContent = '0';
            document.getElementById('username').value = '';
            registerBtn.disabled = true;
            
            // Redirect to home after 2 seconds
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
        } else {
            showMessage(result.message, 'error');
            registerBtn.disabled = false;
            captureBtn.disabled = false;
        }
    } catch (err) {
        showMessage('Error registering user: ' + err.message, 'error');
        registerBtn.disabled = false;
        captureBtn.disabled = false;
    } finally {
        registerBtn.textContent = 'Register User';
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
