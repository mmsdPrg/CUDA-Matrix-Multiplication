@echo off
REM Simple one-click launcher for experiments

echo Starting HW4 CUDA Experiments...
echo.

for /f "delims=" %%i in ('wsl wslpath -a "%CD%"') do set WSL_PATH=%%i

echo Running all experiments (30-45 minutes)...
echo.
wsl bash -c "cd '%WSL_PATH%' && make run-all"

echo.
echo Analyzing results...
wsl bash -c "cd '%WSL_PATH%' && python3 analyze_results.py"

echo.
echo ========================================
echo Done! Check results/ folder
echo ========================================
pause
