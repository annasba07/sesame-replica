@echo off
echo.
echo ============================================================
echo HYBRID THERAPY AI - PRODUCTION SYSTEM
echo ============================================================
echo.
echo Starting the Hybrid Therapy AI System...
echo.
echo Features:
echo - Instant acknowledgment (less than 50ms)
echo - Deep GPT-4 reasoning
echo - Multiple conversation modes
echo - Real-time emotion detection
echo.
echo Opening browser to http://localhost:8000
echo.
timeout /t 2 >nul
start http://localhost:8000
python web_interface.py