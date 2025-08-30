# CSM API Documentation

## Overview
The CSM (Conversational Speech Model) API provides endpoints for conversational voice generation and transcription with sub-200ms latency.

## Base URL
```
http://localhost:8080
```

## Endpoints

### 1. Health Check
Check if the API is running and model is loaded.

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "latency_ms": 3.5
}
```

### 2. Generate Conversational Response
Generate text and/or voice response from input.

**POST** `/generate`

**Request Body:**
```json
{
  "text": "Hello, how are you?",
  "voice_prompt": "base64_encoded_audio",  // Optional
  "max_length": 500,
  "temperature": 0.8,
  "streaming": false
}
```

**Response:**
```json
{
  "text": "I'm doing great, thank you for asking!",
  "voice": "base64_encoded_audio",
  "latency_ms": 180.5,
  "streaming": false
}
```

### 3. Transcribe Audio
Convert voice to text.

**POST** `/transcribe`

**Request Body:**
```json
{
  "audio": "base64_encoded_audio",
  "language": "en"
}
```

**Response:**
```json
{
  "text": "This is the transcribed text",
  "confidence": 0.92,
  "language": "en"
}
```

### 4. Upload Voice File
Upload audio file for processing.

**POST** `/upload_voice`

**Request:** Multipart form with file upload

**Response:**
```json
{
  "filename": "voice.wav",
  "size_bytes": 480000,
  "duration_seconds": 10.0,
  "processed": true
}
```

### 5. Performance Metrics
Get current performance metrics.

**GET** `/metrics`

**Response:**
```json
{
  "model_loaded": true,
  "device": "cuda",
  "expected_latency_ms": 180.5,
  "model_parameters": 422350537,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3070 Ti Laptop GPU",
  "gpu_memory_allocated_mb": 1700.0,
  "gpu_memory_cached_mb": 2048.0
}
```

## Interactive Documentation
Visit http://localhost:8080/docs for interactive API documentation with Swagger UI.

## Example Usage

### Python
```python
import requests
import json

# Generate response
response = requests.post(
    "http://localhost:8080/generate",
    json={
        "text": "Tell me a joke",
        "temperature": 0.9
    }
)
print(response.json())
```

### cURL
```bash
# Health check
curl http://localhost:8080/health

# Generate response
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "temperature": 0.8}'
```

### JavaScript
```javascript
// Generate response
fetch('http://localhost:8080/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Hello!',
    temperature: 0.8
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Performance

- **Latency**: <200ms for generation
- **Throughput**: 50+ requests/second
- **GPU Memory**: ~1.7GB for 422M parameter model
- **Supported Formats**: WAV, MP3, FLAC for audio

## Error Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **422**: Validation Error
- **500**: Internal Server Error
- **503**: Service Unavailable (model not loaded)

## Rate Limiting

Default limits:
- 100 requests per minute per IP
- 10MB max file size for uploads
- 30 second max audio duration

## Security

- Use HTTPS in production
- Implement API keys for authentication
- Sanitize all inputs
- Monitor for abuse

## Deployment

### Docker
```bash
docker build -t csm-api .
docker run -p 8080:8080 --gpus all csm-api
```

### systemd
```bash
sudo cp csm-api.service /etc/systemd/system/
sudo systemctl enable csm-api
sudo systemctl start csm-api
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: csm-api
  template:
    metadata:
      labels:
        app: csm-api
    spec:
      containers:
      - name: csm-api
        image: csm-api:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
```