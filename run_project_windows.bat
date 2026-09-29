@echo off
setlocal
cd /d "%~dp0"
echo [1/2] Running all notebooks...
python run_all_notebooks.py
if errorlevel 1 (
    echo.
    echo ERROR: Notebook execution failed. Scroll up to see the first error.
    pause
    exit /b 1
)
echo.
echo [2/2] All notebooks finished successfully.
echo To start the web demo, run: streamlit run app.py
pause
