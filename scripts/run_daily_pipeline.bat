@echo off

SET PROJECT_DIR=E:\stock-sentiment
SET VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
SET LOG_DIR=%PROJECT_DIR%\logs


for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list ^| findstr "="') do set DATETIME_RAW=%%I
SET TODAY=%DATETIME_RAW:~0,4%-%DATETIME_RAW:~4,2%-%DATETIME_RAW:~6,2%
SET LOG_FILE=%LOG_DIR%\pipeline_run_%TODAY%.log


if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PROJECT_DIR%"

echo ============================================================ >> "%LOG_FILE%"
echo Pipeline run started: %TODAY% %time% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

"%VENV_PYTHON%" -m flows.main_flow >> "%LOG_FILE%" 2>&1

if not errorlevel 1 (
    echo. >> "%LOG_FILE%"
    echo Pipeline run finished: %TODAY% %time% with exit code 0 >> "%LIG_FILE%"
    echo Pipeline succeeded - uploading to HF Dataset repo... >> "%LOG_FILE%"
    "%VENV_PYTHON%" -m scripts.upload_to_hf_dataset >> "%LOG_FILE%" 2>&1
    if not errorlevel 1 (
        echo Upload finished successfully >> "%LOG_FILE%"
    ) else (
        echo Upload FAILED >> "%LOG_FILE%"
    )
    echo ========================================================= >> "%LOG_FILE%"
    exit /b 0
) else (
    echo. >> "%LOG_FILE%"
    echo Pipeline run finished: %TODAY% %time% with exit code 1 >> "%LOG_FILE%"
    echo Pipeline FAILED - skipping upload to avoid pushing bad data >> "%LOG_FILE%"
    echo ========================================================= >> "%LOG_FILE%"
    exit /b 1
)