#!/usr/bin/env python3
"""
Web Interface for Hybrid Therapy System
Simple, clean interface for testing the hybrid approach
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import uuid
from datetime import datetime
from hybrid_therapy_system import HybridTherapySystem, ConversationMode, load_config

app = FastAPI(title="Hybrid Therapy AI")

# Initialize system
config = load_config()
therapy_system = HybridTherapySystem(config)

# Store active sessions
active_sessions = {}

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid Therapy AI - Test Interface</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #7f8c8d;
        }
        
        .chat-container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        .mode-selector {
            padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .mode-selector select {
            padding: 8px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            text-align: right;
        }
        
        .message.user .bubble {
            background: #3498db;
            color: white;
            margin-left: auto;
        }
        
        .message.ai .bubble {
            background: #ecf0f1;
            color: #2c3e50;
        }
        
        .message.ai.immediate .bubble {
            background: #e8f8f5;
            border: 1px solid #1abc9c;
        }
        
        .bubble {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
        }
        
        .timestamp {
            font-size: 11px;
            color: #95a5a6;
            margin-top: 5px;
        }
        
        .latency {
            font-size: 11px;
            color: #27ae60;
            font-weight: bold;
        }
        
        .suggestions {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 13px;
        }
        
        .suggestions h4 {
            color: #7f8c8d;
            font-size: 12px;
            margin-bottom: 5px;
        }
        
        .input-container {
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        
        .input-container input {
            flex: 1;
            padding: 12px 18px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
        }
        
        .input-container input:focus {
            border-color: #3498db;
        }
        
        .input-container button {
            padding: 12px 25px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        .input-container button:hover {
            background: #2980b9;
        }
        
        .status {
            padding: 15px;
            background: #fff;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .status h3 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        
        .metric {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .metric .value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric .label {
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        .emotion-indicator {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-left: 5px;
        }
        
        .emotion-anxious { background: #ffe6e6; color: #e74c3c; }
        .emotion-sad { background: #e6f0ff; color: #3498db; }
        .emotion-neutral { background: #f0f0f0; color: #7f8c8d; }
        .emotion-happy { background: #fff3e6; color: #f39c12; }
        
        .typing-indicator {
            padding: 5px 15px;
            color: #7f8c8d;
            font-style: italic;
            display: none;
        }
        
        .info-box {
            background: #e8f8f5;
            border: 1px solid #1abc9c;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hybrid Therapy AI</h1>
            <p>Experience the future of conversational AI: Instant acknowledgment + Deep understanding</p>
        </div>
        
        <div class="info-box">
            <strong>How it works:</strong> This system combines Sesame's ultra-fast response (<50ms) 
            with GPT-4's deep reasoning. You'll see an immediate acknowledgment followed by a 
            thoughtful response, maintaining natural conversation flow.
        </div>
        
        <div class="chat-container">
            <div class="mode-selector">
                <label>Mode:</label>
                <select id="mode">
                    <option value="therapy">Therapy</option>
                    <option value="coaching">Life Coaching</option>
                    <option value="support">Emotional Support</option>
                    <option value="crisis">Crisis Support</option>
                </select>
            </div>
            
            <div class="messages" id="messages">
                <div class="message ai">
                    <div class="bubble">
                        Hello, I'm here to support you. How are you feeling today?
                    </div>
                    <div class="timestamp">System</div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typing">AI is thinking...</div>
            
            <div class="input-container">
                <input 
                    type="text" 
                    id="messageInput" 
                    placeholder="Type your message..."
                    autofocus
                />
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="status">
            <h3>Session Metrics</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="value" id="responseTime">-</div>
                    <div class="label">First Response (ms)</div>
                </div>
                <div class="metric">
                    <div class="value" id="turnCount">0</div>
                    <div class="label">Conversation Turns</div>
                </div>
                <div class="metric">
                    <div class="value" id="emotionState">-</div>
                    <div class="label">Detected Emotion</div>
                </div>
                <div class="metric">
                    <div class="value" id="sessionTime">0:00</div>
                    <div class="label">Session Duration</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let sessionId = generateSessionId();
        let turnCount = 0;
        let sessionStart = Date.now();
        
        function generateSessionId() {
            return 'session_' + Math.random().toString(36).substr(2, 9);
        }
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleResponse(data);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                addMessage('Connection error. Please refresh the page.', 'ai');
            };
        }
        
        function handleResponse(data) {
            if (data.type === 'immediate') {
                // Show immediate response
                addMessage(data.content, 'ai immediate', data.latency);
                document.getElementById('responseTime').textContent = 
                    Math.round(data.latency * 1000);
                document.getElementById('typing').style.display = 'block';
            } else if (data.type === 'thoughtful') {
                // Replace with thoughtful response
                document.getElementById('typing').style.display = 'none';
                const messages = document.getElementById('messages');
                const lastMessage = messages.lastElementChild;
                if (lastMessage && lastMessage.classList.contains('immediate')) {
                    lastMessage.remove();
                }
                
                addMessage(data.content, 'ai', data.latency, data.emotion, data.suggestions);
                
                // Update emotion indicator
                if (data.emotion) {
                    document.getElementById('emotionState').textContent = 
                        data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
                }
            }
        }
        
        function addMessage(text, sender, latency, emotion, suggestions) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            let html = `<div class="bubble">${escapeHtml(text)}`;
            
            if (emotion && sender === 'ai') {
                html += `<span class="emotion-indicator emotion-${emotion}">${emotion}</span>`;
            }
            
            html += '</div>';
            
            if (latency && sender.includes('ai')) {
                html += `<div class="timestamp">
                    <span class="latency">${Math.round(latency * 1000)}ms</span> • 
                    ${new Date().toLocaleTimeString()}
                </div>`;
            } else {
                html += `<div class="timestamp">${new Date().toLocaleTimeString()}</div>`;
            }
            
            if (suggestions && suggestions.length > 0) {
                html += `<div class="suggestions">
                    <h4>Suggestions:</h4>
                    ${suggestions.map(s => `• ${s}`).join('<br>')}
                </div>`;
            }
            
            messageDiv.innerHTML = html;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                // Add user message
                addMessage(message, 'user');
                
                // Send to server
                ws.send(JSON.stringify({
                    text: message,
                    mode: document.getElementById('mode').value
                }));
                
                // Clear input
                input.value = '';
                
                // Update turn count
                turnCount++;
                document.getElementById('turnCount').textContent = turnCount;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Update session timer
        setInterval(() => {
            const elapsed = Date.now() - sessionStart;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            document.getElementById('sessionTime').textContent = 
                `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
        
        // Enter key support
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Connect on load
        connectWebSocket();
    </script>
</body>
</html>
"""

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time conversation"""
    await websocket.accept()
    active_sessions[session_id] = websocket
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process with hybrid system
            mode = ConversationMode(data.get("mode", "therapy"))
            
            # Stream response
            async for response_part in therapy_system.stream_response(
                data["text"], 
                session_id
            ):
                await websocket.send_json(response_part)
                
    except WebSocketDisconnect:
        if session_id in active_sessions:
            del active_sessions[session_id]

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session summary"""
    summary = therapy_system.get_session_summary(session_id)
    return JSONResponse(summary)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "system": "hybrid",
        "components": {
            "sesame": "operational",
            "gpt4": "operational" if config.get("openai_api_key") else "simulation_mode"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("HYBRID THERAPY AI - WEB INTERFACE")
    print("=" * 60)
    print("\nStarting server...")
    print("Open http://localhost:8000 in your browser")
    print("\nFeatures:")
    print("- Instant acknowledgment (<50ms)")
    print("- Deep thoughtful responses")
    print("- Multiple conversation modes")
    print("- Real-time emotion detection")
    print("- Session metrics")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)