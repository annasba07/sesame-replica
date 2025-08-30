# Hybrid Therapy AI - Setup Instructions

## Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn websockets aiohttp
```

### 2. Configure API Keys (Optional)
To use real GPT-4 instead of simulation mode:

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Mac/Linux
export OPENAI_API_KEY=your-api-key-here
```

Or edit `config.json`:
```json
{
  "openai_api_key": "your-api-key-here"
}
```

### 3. Run the System
```bash
python web_interface.py
```

### 4. Open Browser
Navigate to: http://localhost:8000

## Features

### Conversation Modes
- **Therapy**: Emotional validation, CBT/DBT techniques
- **Coaching**: Goal-oriented, action-focused
- **Support**: Companionship and active listening
- **Crisis**: Immediate safety and grounding

### Real-Time Metrics
- First response latency (target: <50ms)
- Conversation turns
- Detected emotions
- Session duration

### Response Pattern
1. **Immediate** (30-50ms): "I hear you..." 
2. **Thoughtful** (1-2s): Deep, contextual response
3. **Suggestions**: Actionable techniques

## Testing the Hybrid Approach

### Example 1: Crisis Response
Type: "I'm having a panic attack!"

Expected:
- [30ms] "I hear that you're feeling anxious..."
- [1.2s] "Right now, let's focus on getting you grounded. Look around and name 5 things you can see..."

### Example 2: Emotional Support
Type: "I feel so alone and sad"

Expected:
- [30ms] "I can hear the sadness in what you're sharing..."
- [1.2s] "Your feelings are valid. What's been weighing on you the most?"

## Architecture

```
User Input
    ↓
┌─────────────────────────────┐
│   Hybrid Therapy System     │
├─────────────┬───────────────┤
│   Sesame    │    GPT-4      │
│  (Fast ACK) │ (Deep Think)  │
└─────────────┴───────────────┘
    ↓              ↓
Immediate      Thoughtful
Response       Response
```

## Configuration Options

### config.json
```json
{
  "sesame_model": "path/to/model.pt",     // Sesame model path
  "openai_api_key": "sk-...",             // OpenAI API key
  "mode": "therapy",                      // Default mode
  "features": {
    "crisis_detection": true,             // Auto-detect crisis
    "emotion_tracking": true,             // Track emotions
    "suggestions": true                   // Provide suggestions
  }
}
```

## API Endpoints

### WebSocket
- `/ws/{session_id}` - Real-time conversation

### REST
- `GET /` - Web interface
- `GET /api/health` - System health
- `GET /api/session/{id}` - Session summary

## Deployment

### Local Testing
```bash
python web_interface.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn web_interface:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "web_interface.py"]
```

## Troubleshooting

### "Connection Error"
- Check server is running
- Verify port 8000 is available
- Check firewall settings

### "Slow Responses"
- Without GPU: Expected behavior
- With GPU: Check CUDA installation
- API mode: Check internet connection

### "No Immediate Response"
- Verify Sesame simulator is working
- Check WebSocket connection
- Look at browser console for errors

## Performance Tips

1. **Use GPU**: Dramatically improves Sesame latency
2. **Cache Responses**: For common phrases
3. **Batch Processing**: Handle multiple sessions
4. **Edge Deployment**: Minimize network latency

## Security Considerations

1. **API Keys**: Never commit to git
2. **Session Data**: Implement encryption
3. **User Privacy**: Don't log conversations
4. **Rate Limiting**: Prevent abuse

## Next Steps

1. Test different conversation scenarios
2. Monitor latency metrics
3. Gather user feedback
4. Fine-tune response templates
5. Add voice input/output

---

For issues or questions, check the logs or create an issue in the repository.