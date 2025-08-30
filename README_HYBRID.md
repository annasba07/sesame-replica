# Hybrid Therapy AI System

A production-ready implementation combining Sesame's ultra-fast conversational AI with GPT-4's deep reasoning capabilities for therapy and coaching applications.

## 🚀 Quick Start

### Option 1: Direct Run
```bash
# Install dependencies
pip install -r requirements_hybrid.txt

# Set OpenAI API key (optional - will use simulation mode without it)
export OPENAI_API_KEY=your-key-here  # Mac/Linux
set OPENAI_API_KEY=your-key-here     # Windows

# Start the system
python web_interface.py

# Open browser to http://localhost:8000
```

### Option 2: Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Open browser to http://localhost:8000
```

### Option 3: One-Click Start
- **Windows**: Double-click `start_hybrid_ai.bat`
- **Mac/Linux**: Run `./start_hybrid_ai.sh`

## 🎯 Key Features

### Dual-Stream Processing
- **Immediate Response** (<50ms): Acknowledges user instantly
- **Thoughtful Response** (1-2s): Provides deep, contextual guidance
- **Natural Flow**: No awkward waiting periods

### Conversation Modes
1. **Therapy**: CBT/DBT techniques, emotional validation
2. **Coaching**: Goal-setting, action planning
3. **Support**: Active listening, companionship
4. **Crisis**: Immediate safety, grounding techniques

### Real-Time Features
- Emotion detection from text
- Response time monitoring
- Session analytics
- Actionable suggestions

## 📊 Performance Metrics

| Metric | Traditional | Hybrid |
|--------|------------|--------|
| First Response | 1200ms | 30ms |
| User Perception | Waiting... | Immediate connection |
| Emotional Continuity | Broken | Maintained |
| Conversation Flow | Choppy | Natural |

## 🏗️ Architecture

```
User Input → WebSocket
    ↓
┌─────────────────────────────────┐
│   Hybrid Therapy System         │
├─────────────┬───────────────────┤
│   Sesame    │      GPT-4        │
│  (Fast ACK) │  (Deep Reasoning) │
└──────┬──────┴────────┬──────────┘
       ↓               ↓
  Immediate       Thoughtful
  Response        Response
```

## 🔧 Configuration

Edit `config.json`:
```json
{
  "openai_api_key": "sk-...",  // Your OpenAI key
  "mode": "therapy",            // Default mode
  "features": {
    "crisis_detection": true,   // Auto-detect crisis
    "emotion_tracking": true,   // Track emotions
    "suggestions": true         // Provide techniques
  }
}
```

## 💬 Example Conversations

### Crisis Response
```
User: "I'm having a panic attack!"
AI [30ms]: "I hear that you're feeling anxious..."
AI [1.2s]: "Let's focus on your breathing. In for 4, hold for 4, out for 4..."
```

### Emotional Support
```
User: "I feel so alone"
AI [30ms]: "I can hear the sadness..."
AI [1.3s]: "Your feelings are valid. What's been weighing on you?"
```

## 🛠️ API Reference

### WebSocket
Connect to `ws://localhost:8000/ws/{session_id}` for real-time conversation.

### REST Endpoints
- `GET /` - Web interface
- `GET /api/health` - System health check
- `GET /api/session/{id}` - Session summary

## 🚢 Production Deployment

### Environment Variables
```bash
OPENAI_API_KEY=your-key
LOG_LEVEL=INFO
PORT=8000
```

### Scaling Considerations
1. Use Redis for session management
2. Deploy Sesame on edge nodes
3. Use GPT-4 API with retry logic
4. Implement rate limiting

### Security
- Never log conversation content
- Use HTTPS in production
- Implement authentication
- Encrypt session data

## 📈 Monitoring

The web interface displays:
- Response latency (ms)
- Conversation turns
- Detected emotions
- Session duration

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Load Testing
```bash
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Sesame AI Labs for the CSM model
- OpenAI for GPT-4
- Inspired by the need for more human-like AI therapy

---

**Note**: This system is designed to supplement, not replace, professional mental health care. Always consult qualified professionals for serious mental health concerns.