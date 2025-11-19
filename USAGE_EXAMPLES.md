# Usage Examples

## Quick Start Guide

### 1. Start the Application

```bash
python app.py
```

Expected output:
```
Warning: Model file best_model.keras not found
Features file user_features.json not found, starting with empty database
 * Running on http://127.0.0.1:5000
```

### 2. Access the Web Interface

Open your browser and go to: `http://localhost:5000`

You'll see the landing page with two options:
- **Register New User**
- **Authenticate**

## Registration Workflow

### Step-by-Step: Registering a User

1. **Click "Register"** on the home page

2. **Enter a Username**
   - Example: "john_doe"
   - Must be unique
   - No special validation, but keep it simple

3. **Start Camera**
   - Click "Start Camera" button
   - Browser will request camera permissions
   - Allow camera access
   - You should see yourself in the video feed

4. **Capture Images**
   - Position your face in the center
   - Click "Capture Image" button
   - Capture **5-10 images** with:
     - Different angles (front, slight left, slight right)
     - Different expressions (neutral, smile)
     - Consistent lighting
   - You'll see thumbnails of captured images below

5. **Review & Remove**
   - Review captured images
   - Click the **×** button on any thumbnail to remove unwanted images
   - Must have at least 5 images to proceed

6. **Register**
   - Click "Register User" button
   - Wait for processing (may take a few seconds)
   - Success message: "User john_doe registered successfully"
   - Automatically redirected to home page

### Registration Tips

✅ **Do's:**
- Good, even lighting
- Face clearly visible
- Multiple angles
- Natural expressions
- 8-10 images for best results

❌ **Don'ts:**
- Too dark or too bright
- Face partially hidden
- Too far from camera
- Blurry images
- Only 1-2 images

## Authentication Workflow

### Step-by-Step: Authenticating

1. **Click "Authenticate"** on the home page

2. **Start Camera**
   - Click "Start Camera" button
   - Allow camera permissions
   - Position your face in frame

3. **Authenticate**
   - Click "Authenticate" button
   - Wait for processing
   - Result will appear with:
     - User identification
     - Confidence score
     - Success/Failure message

### Authentication Results

**Successful Authentication:**
```
✓ Welcome, john_doe!
User: john_doe
Confidence: 87.5%
```

**Failed Authentication:**
```
✗ Authentication Failed
Status: Not authenticated
Confidence: 45.2%
Please try again or register if you are a new user.
```

### Authentication Tips

✅ **For Better Results:**
- Use similar lighting as registration
- Face the camera directly
- Ensure face is clearly visible
- Try again if first attempt fails
- Remove glasses if you didn't wear them during registration

## Example Scenarios

### Scenario 1: New User Setup

**Goal**: Register 3 users for a team

```
1. Start app: python app.py
2. Register User 1:
   - Name: "alice"
   - Capture: 10 images (front, left, right, smile, neutral)
   - Register successfully
   
3. Register User 2:
   - Name: "bob"
   - Capture: 8 images
   - Register successfully
   
4. Register User 3:
   - Name: "charlie"
   - Capture: 10 images
   - Register successfully

Result: 3 users stored in user_features.json
```

### Scenario 2: Daily Authentication

**Goal**: Team members authenticate each morning

```
1. Alice arrives:
   - Opens authenticate page
   - Captures face
   - Result: ✓ "Welcome, alice!" (Confidence: 85%)
   
2. Bob arrives:
   - Opens authenticate page  
   - Captures face
   - Result: ✓ "Welcome, bob!" (Confidence: 82%)
   
3. Charlie with poor lighting:
   - First attempt: ✗ Failed (Confidence: 55%)
   - Adjusts lighting
   - Second attempt: ✓ "Welcome, charlie!" (Confidence: 78%)
```

### Scenario 3: Handling Failures

**Goal**: Troubleshoot authentication issues

```
Problem: User "david" authentication always fails

Steps:
1. Check if "david" is registered:
   - Look in user_features.json
   - If not present, need to register first
   
2. If registered but still failing:
   - Check lighting conditions
   - Ensure face is clearly visible
   - Try re-registering with better quality images
   - Verify camera is working properly
   
3. Re-register "david":
   - Capture 10 high-quality images
   - Various angles and expressions
   - Good lighting
   - Test authentication again
   
Result: Authentication now works with 80%+ confidence
```

## API Usage (Advanced)

If you want to integrate with other applications:

### Registration API

**Endpoint**: `POST /register`

**Request Body**:
```json
{
  "username": "john_doe",
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    ...
  ]
}
```

**Response**:
```json
{
  "success": true,
  "message": "User john_doe registered successfully"
}
```

### Authentication API

**Endpoint**: `POST /authenticate`

**Request Body**:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response** (Success):
```json
{
  "success": true,
  "message": "Authenticated as john_doe",
  "user": "john_doe",
  "confidence": 0.875
}
```

**Response** (Failure):
```json
{
  "success": false,
  "message": "Authentication failed - no match found",
  "confidence": 0.452
}
```

## Common Issues and Solutions

### Issue 1: "Need at least 5 valid face images"

**Cause**: Some captured images couldn't be processed

**Solution**:
- Capture more images (try 8-10)
- Ensure face is clearly visible
- Check lighting
- Keep face centered in frame

### Issue 2: Low Confidence Score (Below 70%)

**Cause**: 
- Different conditions from registration
- Poor image quality
- Model not trained well

**Solution**:
- Re-register with more images
- Use consistent lighting
- Capture from multiple angles during registration
- Adjust threshold in app.py if needed

### Issue 3: Wrong User Identified

**Cause**: 
- Similar facial features
- Insufficient training images
- Low-quality registration images

**Solution**:
- Re-register both users with more images
- Use different angles and expressions
- Increase threshold for stricter matching
- Check if users look very similar

## System Files

### user_features.json

After registering users, this file stores their feature embeddings:

```json
{
  "alice": [0.123, -0.456, 0.789, ...],  // 256-dimensional vector
  "bob": [0.234, -0.567, 0.890, ...],
  "charlie": [0.345, -0.678, 0.901, ...]
}
```

**Important**: 
- Don't manually edit this file
- Backup this file if you want to preserve registered users
- Delete this file to reset all registrations

## Performance Notes

- **Registration**: 5-10 seconds (depends on number of images)
- **Authentication**: 2-5 seconds
- **Model Loading**: One-time, 3-5 seconds at startup
- **Storage**: ~1KB per registered user

## Next Steps

1. **For Development**:
   - Test with different lighting conditions
   - Try various face angles
   - Test with accessories (glasses, hats)
   - Measure accuracy with test dataset

2. **For Production**:
   - Train model with larger dataset
   - Add liveness detection
   - Implement proper database
   - Add user management features
   - Set up HTTPS
   - Add logging and monitoring
   - Deploy to server (AWS, GCP, Azure)

## Support Resources

- **SETUP.md**: Installation and configuration
- **MODEL_INFO.md**: Model training and specifications
- **README.md**: Project overview
- **app.py**: Source code with comments
