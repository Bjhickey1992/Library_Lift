@echo off
cd /d "%~dp0"
echo Starting Streamlit app...
echo Current directory: %CD%
echo.
streamlit run streamlit_app.py
pause
