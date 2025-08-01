@echo off
echo Activating virtual environment and launching project...

REM Change to the project directory
cd /d C:\Users\Olivier\Desktop\v1.1

REM Check if the venv folder exists
if not exist venv (
    echo Error: Virtual environment 'venv' not found. Please create it with 'python -m venv venv' first.
    pause
    exit /b 1
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Run the main script
python classify_ticket_fn.py

REM Deactivate the virtual environment (optional, runs on script exit or manually)
REM deactivate

echo.
echo Project execution completed. Press any key to exit.
pause