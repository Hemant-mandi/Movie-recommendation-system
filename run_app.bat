@echo off
echo ========================================================
echo   Launching OTT Recommendation Dashboard (BCS714D)
echo ========================================================
call venv\Scripts\activate.bat
streamlit run app/streamlit_app.py
pause
