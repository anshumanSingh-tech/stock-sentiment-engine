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

SET EXIT_CODE=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo Pipeline run finished: %TODAY% %time% with exit code %EXIT_CODE% >> "%LOG_FILE%"

if %EXIT_CODE% EQU 0 (
    echo. >> "%LOG_FILE%"
    echo Pipeline succeeded - uploading to HF Dataset repo... >> "%LOG_FILE%" 
    "%VENV_PYTHON%" -m scripts.upload_to_hf_dataset >> "%LOG_FILE%" 2>&1
    SET UPLOAD_EXIT_CODE=%ERRORLEVEL%
    echo Upload finished with exit code %UPLOAD_EXIT_CODE% >> "%LOG_FILE%"
) else(
    echo. "%LOG_FILE%"
    echo Pipeline FAILED - skipping upload to avoid pushing bad data >> "%LOG_FILE%"
    SET UPLOAD_EXIT_CODE=1
)

echo ============================================================ >> "%LOG_FILE%"

if %EXIT_CODE% NEQ 0 exit /b %EXIT_CODE%
exit /b %UPLOAD_EXIT_CODE%