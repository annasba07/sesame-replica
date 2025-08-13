#!/usr/bin/env python3
"""
Production Web Interface - REAL MODELS ONLY
Uses actual Sesame CSM + GPT-4 API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import asyncio
import os
import logging
from hybrid_production import ProductionHybridSystem, ConversationMode, get_production_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Production Hybrid Therapy AI")

# Initialize production system
try:
    config = get_production_config()
    therapy_system = ProductionHybridSystem(
        openai_api_key=config["openai_api_key"],
        use_official_sesame=config["use_official_sesame"]
    )
    system_ready = True
except Exception as e:
    logger.error(f"Failed to initialize production system: {e}")
    therapy_system = None
    system_ready = False

@app.get("/", response_class=HTMLResponse)
async def index():
    """Production interface"""
    if not system_ready:
        return """
        <html>
        <body style="font-family: sans-serif; padding: 50px;">
            <h1>System Not Ready</h1>
            <p>The production system requires:</p>
            <ul>
                <li>OPENAI_API_KEY environment variable</li>
                <li>Sesame CSM model installed</li>
                <li>CUDA-capable GPU (recommended)</li>
            </ul>
            <pre>export OPENAI_API_KEY=sk-your-key-here</pre>
        </body>
        </html>
        """
    
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Production Hybrid AI - Real Models</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, system-ui, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            overflow: hidden;
        }
        
        .container {
            height: 100vh;
            display: flex;
            flex-direction: column;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            padding: 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }
        
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
        }
        
        .main {
            flex: 1;
            display: flex;
            gap: 20px;
            padding: 20px;
            overflow: hidden;
        }
        
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #111;
            border-radius: 12px;
            border: 1px solid #333;
        }
        
        .mode-bar {
            padding: 16px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 12px;
        }
        
        .mode-button {
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid #333;
            background: transparent;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .mode-button.active {
            background: #1d4ed8;
            color: white;
            border-color: #1d4ed8;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 24px;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
        }
        
        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 14px;
            color: #888;
        }
        
        .message.user .message-header {
            justify-content: flex-end;
        }
        
        .latency-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .latency-fast {
            background: #10b981;
            color: white;
        }
        
        .latency-normal {
            background: #3b82f6;
            color: white;
        }
        
        .message-content {
            padding: 16px;
            border-radius: 12px;
            font-size: 15px;
            line-height: 1.6;
        }
        
        .message.user .message-content {
            background: #1d4ed8;
            margin-left: 20%;
        }
        
        .message.ai.immediate .message-content {
            background: #1e293b;
            border: 1px solid #10b981;
            margin-right: 20%;
        }
        
        .message.ai.thoughtful .message-content {
            background: #1e293b;
            margin-right: 20%;
        }
        
        .input-panel {
            padding: 20px;
            border-top: 1px solid #333;
        }
        
        .input-wrapper {
            display: flex;
            gap: 12px;
        }
        
        .input-field {
            flex: 1;
            background: #1e293b;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px 16px;
            color: white;
            font-size: 15px;
        }
        
        .input-field:focus {
            outline: none;
            border-color: #3b82f6;
        }
        
        .send-button {
            padding: 12px 24px;
            background: #1d4ed8;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }
        
        .send-button:hover {
            background: #1e40af;
        }
        
        .metrics-panel {
            width: 300px;
            background: #111;
            border-radius: 12px;
            border: 1px solid #333;
            padding: 20px;
        }
        
        .metrics-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .metric {
            margin-bottom: 20px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 600;
        }
        
        .metric-value.latency {
            color: #10b981;
        }
        
        .emotion-display {
            padding: 16px;
            background: #1e293b;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .emotion-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }
        
        .emotion-value {
            font-size: 18px;
            font-weight: 600;
            text-transform: capitalize;
        }
        
        .model-info {
            margin-top: auto;
            padding-top: 20px;
            border-top: 1px solid #333;
            font-size: 12px;
            color: #666;
        }
        
        .model-info div {
            margin-bottom: 4px;
        }
        
        .typing-indicator {
            padding: 8px 16px;
            color: #888;
            font-style: italic;
            display: none;
        }
        
        .crisis-alert {
            background: #dc2626;
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Production Hybrid AI</h1>
            <div class="status">
                <div class="status-item">
                    <div class="status-indicator"></div>
                    <span>Sesame CSM</span>
                </div>
                <div class="status-item">
                    <div class="status-indicator"></div>
                    <span>GPT-4 API</span>
                </div>
            </div>
        </div>
        
        <div class="main">
            <div class="chat-panel">
                <div class="mode-bar">
                    <button class="mode-button active" data-mode="therapy">Therapy</button>
                    <button class="mode-button" data-mode="coaching">Coaching</button>
                    <button class="mode-button" data-mode="support">Support</button>
                    <button class="mode-button" data-mode="crisis">Crisis</button>
                </div>
                
                <div class="messages" id="messages">
                    <div class="message ai thoughtful">
                        <div class="message-header">
                            <span>AI</span>
                        </div>
                        <div class="message-content">
                            Hello, I'm here to support you. This is a production system using real AI models. How are you feeling today?
                        </div>
                    </div>
                </div>
                
                <div class="typing-indicator" id="typing">AI is composing thoughtful response...</div>
                <div class="crisis-alert" id="crisis">Crisis support mode activated</div>
                
                <div class="input-panel">
                    <div class="input-wrapper">
                        <input 
                            type="text" 
                            class="input-field" 
                            id="input"
                            placeholder="Share what's on your mind..."
                            autofocus
                        />
                        <button class="send-button" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
            
            <div class="metrics-panel">
                <div class="metrics-title">Real-Time Metrics</div>
                
                <div class="metric">
                    <div class="metric-label">Sesame Response</div>
                    <div class="metric-value latency" id="sesameLatency">-</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">GPT-4 Response</div>
                    <div class="metric-value" id="gpt4Latency">-</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Conversation Turns</div>
                    <div class="metric-value" id="turns">0</div>
                </div>
                
                <div class="emotion-display">
                    <div class="emotion-label">Detected Emotion</div>
                    <div class="emotion-value" id="emotion">-</div>
                    <div style="margin-top: 8px; font-size: 12px; color: #888;">
                        Confidence: <span id="confidence">-</span>
                    </div>
                </div>
                
                <div class="model-info">
                    <div>Sesame: CSM-1B</div>
                    <div>GPT: GPT-4-Turbo</div>
                    <div>Device: CUDA</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let sessionId = 'prod_' + Date.now();
        let turnCount = 0;
        let currentMode = 'therapy';
        
        // Mode buttons
        document.querySelectorAll('.mode-button').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentMode = btn.dataset.mode;
            });
        });
        
        function connect() {
            ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleResponse(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleResponse(data) {
            if (data.type === 'immediate') {
                addMessage(data.content, 'ai immediate', data.latency);
                document.getElementById('sesameLatency').textContent = 
                    Math.round(data.latency * 1000) + 'ms';
                document.getElementById('typing').style.display = 'block';
            } else if (data.type === 'thoughtful') {
                // Remove immediate message
                const messages = document.getElementById('messages');
                const immediate = messages.querySelector('.message.immediate:last-child');
                if (immediate) immediate.remove();
                
                document.getElementById('typing').style.display = 'none';
                addMessage(data.content, 'ai thoughtful', data.latency);
                
                document.getElementById('gpt4Latency').textContent = 
                    Math.round(data.latency * 1000) + 'ms';
                document.getElementById('emotion').textContent = data.emotion || '-';
                document.getElementById('confidence').textContent = 
                    data.confidence ? (data.confidence * 100).toFixed(0) + '%' : '-';
                
                if (data.emotion === 'anxious' && data.confidence > 0.8) {
                    document.getElementById('crisis').style.display = 'block';
                    setTimeout(() => {
                        document.getElementById('crisis').style.display = 'none';
                    }, 5000);
                }
            }
        }
        
        function addMessage(text, type, latency) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            
            const header = document.createElement('div');
            header.className = 'message-header';
            
            if (type.includes('ai')) {
                header.innerHTML = `
                    <span>AI</span>
                    ${latency ? `<span class="latency-badge ${latency < 0.1 ? 'latency-fast' : 'latency-normal'}">${Math.round(latency * 1000)}ms</span>` : ''}
                `;
            } else {
                header.innerHTML = '<span>You</span>';
            }
            
            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = text;
            
            div.appendChild(header);
            div.appendChild(content);
            messages.appendChild(div);
            
            messages.scrollTop = messages.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('input');
            const text = input.value.trim();
            
            if (text && ws && ws.readyState === WebSocket.OPEN) {
                addMessage(text, 'user');
                
                ws.send(JSON.stringify({
                    text: text,
                    mode: currentMode
                }));
                
                input.value = '';
                turnCount++;
                document.getElementById('turns').textContent = turnCount;
            }
        }
        
        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        connect();
    </script>
</body>
</html>
"""

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time conversation"""
    if not system_ready:
        await websocket.close(code=1011, reason="System not ready")
        return
    
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Set mode
            mode = ConversationMode(data.get("mode", "therapy"))
            
            # Process with real models
            async for response_part in therapy_system.stream_response(
                data["text"], 
                session_id
            ):
                await websocket.send_json(response_part)
                
    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "ready" if system_ready else "not_ready",
        "models": {
            "sesame": "loaded" if system_ready else "not_loaded",
            "gpt4": "connected" if system_ready else "not_connected"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("PRODUCTION HYBRID AI - REAL MODELS ONLY")
    print("="*60)
    
    if not system_ready:
        print("\n❌ System not ready!")
        print("\nRequired:")
        print("1. Set OPENAI_API_KEY environment variable:")
        print("   export OPENAI_API_KEY=sk-your-key-here")
        print("\n2. Install Sesame CSM:")
        print("   git clone https://github.com/SesameAILabs/csm.git")
        print("\n3. Have CUDA GPU for best performance")
    else:
        print("\n✓ Sesame CSM loaded")
        print("✓ GPT-4 API connected")
        print("\nStarting server at http://localhost:8000")
        print("\nThis is using REAL MODELS - costs apply to API usage")
    
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)