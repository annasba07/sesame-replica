@echo off
REM CSM Experiment Runner for Windows
REM One-click script to run the entire pipeline

echo ==========================================
echo      CSM Replication Experiment
echo ==========================================

REM Step 1: Setup environment
echo.
echo Step 1: Setting up environment...
python setup_environment.py
if %errorlevel% neq 0 (
    echo ERROR: Environment setup failed
    pause
    exit /b 1
)
echo Environment setup completed successfully

REM Step 2: Collect minimal data
echo.
echo Step 2: Collecting test data (1 hour)...
if not exist "data\conversations\*.wav" (
    python collect_data.py --target_hours 1 --sources librispeech
    if %errorlevel% neq 0 (
        echo ERROR: Data collection failed
        pause
        exit /b 1
    )
) else (
    echo Data already exists, skipping collection
)

REM Step 3: Run minimal training
echo.
echo Step 3: Running minimal training test...
python train_minimal.py
if %errorlevel% neq 0 (
    echo ERROR: Training failed
    pause
    exit /b 1
)
echo Training completed successfully

REM Step 4: Test the demo
echo.
echo Step 4: Testing demo with examples...
python demo.py --mode test
if %errorlevel% neq 0 (
    echo ERROR: Demo test failed
    pause
    exit /b 1
)

REM Step 5: Summary
echo.
echo ==========================================
echo      Experiment Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Run interactive demo: python demo.py
echo 2. Collect more data: python collect_data.py --target_hours 100
echo 3. Train longer: python train.py --model_size tiny
echo 4. Monitor with wandb: https://wandb.ai
echo.
echo Outputs:
echo - Model checkpoint: checkpoints\test_model.pt
echo - Audio samples: outputs\audio\
echo - Logs: logs\
echo.
pause