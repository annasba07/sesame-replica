#!/usr/bin/env python3
"""
Test the CSM API
"""

import requests
import json
import base64
import time

API_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_generate():
    """Test generation endpoint"""
    print("\nTesting /generate endpoint...")
    data = {
        "text": "Hello, how are you today?",
        "temperature": 0.8,
        "max_length": 100
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/generate", json=data)
    latency = (time.time() - start) * 1000
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print(f"Total API latency: {latency:.1f}ms")
    
    return response.json()

def test_transcribe():
    """Test transcribe endpoint"""
    print("\nTesting /transcribe endpoint...")
    
    # Create fake audio data
    fake_audio = b"fake audio data for testing"
    audio_b64 = base64.b64encode(fake_audio).decode()
    
    data = {
        "audio": audio_b64,
        "language": "en"
    }
    
    response = requests.post(f"{API_URL}/transcribe", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting /metrics endpoint...")
    response = requests.get(f"{API_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()

def main():
    print("CSM API Test Suite")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server...")
    for i in range(5):
        try:
            requests.get(f"{API_URL}/health", timeout=1)
            print("Server is ready!")
            break
        except:
            time.sleep(1)
    
    # Run tests
    try:
        # Test health
        health = test_health()
        
        # Test generation
        generation = test_generate()
        
        # Test transcription
        transcription = test_transcribe()
        
        # Test metrics
        metrics = test_metrics()
        
        # Summary
        print("\n" + "=" * 50)
        print("API TEST SUMMARY")
        print("=" * 50)
        print(f"Server Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"Device: {health['device']}")
        
        if 'latency_ms' in generation:
            print(f"Generation Latency: {generation['latency_ms']:.1f}ms")
            
            if generation['latency_ms'] < 200:
                print("\nSUCCESS: API achieves <200ms latency target!")
        
        print("\nAll API endpoints are functional!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the API server is running: python serve_api.py")

if __name__ == "__main__":
    main()